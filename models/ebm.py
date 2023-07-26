# coding: utf-8
# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2023-07-22

import torch
import numpy as np

class LatentClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_classes, depth) -> None:
        super().__init__()

        self.n_classes = output_dim
        self.energy_weight = 1.0
        self.net = torch.nn.ModuleList()

        dims = np.linspace(input_dim, output_dim, depth).astype(int)
        for layer in range(len(dims) - 1):
            self.net.append(torch.nn.Dropout(0.1))
            self.net.append(torch.nn.Conv2d(dims[layer], dims[layer + 1], 1))
            self.net.append(torch.nn.LeakyReLU(0.1)) 
            # TODO(mingzhe): 
            # Avoid non-smooth non-linearities such as ReLU and LeakyReLU.
            # Prefer non-linearities with a theoretically unique adjoint/gradient such as Softplus.
        
        self.linear = torch.nn.Linear(output_dim, num_classes)

    def forward(self, x):
        if x.ndim == 2:
            x = x[:, :, None, None]

        for layer in self.net:
            x = layer(x)

        out = x.squeeze(-1).squeeze(-1)
        logits = self.linear(out)
        return logits