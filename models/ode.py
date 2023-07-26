# coding: utf-8
# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2023-07-22

import torch
from vae import VAE
from ebm import LatentClassifier
from torchdiffeq import odeint, odeint_adjoint


class VPODE(torch.nn.Module):
    def __init__(self, ccf, y, beta_min=0.1, beta_max=20, T=1.0):
        super().__init__()
        self.y = y
        self.T = T
        self.ccf = ccf
        self.n_evals = 0
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        
    def forward(self, t_k, states):
        z = states
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            beta_t = self.beta_0 + t_k * (self.beta_1 - self.beta_0)
            cond_energy_neg = self.ccf.get_cond_energy(z, self.y)
            cond_f_prime = torch.autograd.grad(cond_energy_neg.sum(), [z])[0]            
            dz_dt = -0.5 * beta_t * cond_f_prime
        self.n_evals += 1
        return dz_dt
    
class CCF(torch.nn.Module):
    def __init__(self, classifiers):
        super(CCF, self).__init__()
        self.f = torch.nn.ModuleList()
        for cls_name in classifiers:
            print(f"Loding Model [{cls_name}]")
            self.f.append(classifiers[cls_name])

    def get_cond_energy(self, z, y_s):
        energy_outs = []

        for i, y_ in enumerate(y_s):
            cls = self.f[i]
            logits = cls(z)
            n_classes = cls.n_classes

            if n_classes > 1:
                # Classification
                sigle_energy = torch.gather(logits, 1, y_[:, None]).squeeze() - logits.logsumexp(1)
                energy_outs.append(cls.energy_weight * sigle_energy)
            else:
                # Regression
                sigma = 0.1  # this value works well
                sigle_energy = -torch.norm(logits - y_[:, None], dim=1) ** 2 * 0.5 / (sigma ** 2)
                energy_outs.append(cls.energy_weight * sigle_energy)

        energy_output = torch.stack(energy_outs).sum(dim=0)
        return energy_output # - 0.03 * torch.norm(z, dim=1) ** 2 * 0.5
    
    def forward(self, z, y):
        energy_output = self.get_cond_energy(z, y) - torch.norm(z, dim=1) ** 2 * 0.5
        return energy_output
    
if __name__ == "__main__":
    latent_labels = None
    z = None
    a_tol = 1e-3
    r_tol = 1e-3
    ode_method = "dopri5"

    ccf = CCF(
        {
            "gender": LatentClassifier(input_dim=128, output_dim=3, num_classes=3, depth=3),
            "race": LatentClassifier(input_dim=128, output_dim=3, num_classes=4, depth=3),
        }
    )
    vpode = VPODE(ccf, latent_labels)

    states = (z)
    # linspace(start, end, step)
    integration_times = torch.linspace(vpode.T, 0, 2).type(torch.float32).to("cuda")

    state_t = odeint(
        vpode,
        states,
        integration_times,
        atol = a_tol,
        rtol = r_tol,
        method = ode_method,
        options={'max_num_steps': 2 ** 9}
    )

    # Depends on the integration_times
    z_t0 = state_t[-1]
    z = z_t0
