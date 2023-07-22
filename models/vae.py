# coding: utf-8
# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2023-07-22

# Definition of VAE encoder / decoder

import torch
from torch import nn
from transformers.adapters import LoRAConfig
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

class Encoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.model_name = config["model_name"]
        self.tokenizer_name = config["tokenizer_name"]

        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        self.hidden_size = config["hidden_size"]
        self.latent_size = config["latent_size"]

        self.mean_logvar_layer = nn.Linear(self.hidden_size, 2 * self.latent_size, bias=False)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask)
        mean, logvar = self.mean_logvar_layer(outputs.pooler_output).chunk(2, -1)
        return mean, logvar, outputs

class Decoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        self.model_name = config["model_name"]
        self.tokenizer_name = config["tokenizer_name"]


        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        lora_config = LoRAConfig(selfattn_lora=True, intermediate_lora=False, r=8, alpha=8)
        self.model.add_adapter("LoRA", config=lora_config)
        self.model.train_adapter('LoRA', train_embeddings=True)

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        self.hidden_size = config["hidden_size"]
        self.latent_size = config["latent_size"]

        self.latent_emb_layer = nn.Linear(self.latent_size, self.hidden_size, bias=False)

    def forward(self, input_ids, latent_z, labels, past_key_values=None, attention_mask=None):
        # Used as embeddings to add on other embeddings
        latent_emb = self.latent_emb_layer(latent_z).unsqueeze(1)
        inputs_embeds = self.model.transformer.wte(input_ids)
        inputs_embeds = latent_emb + inputs_embeds

        # TODO(mingzhe): past_key_values as memory
         
        outputs = self.model(inputs_embeds=inputs_embeds, labels=labels, past_key_values=past_key_values, attention_mask=attention_mask)
        return outputs

class VAE(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        # Encoders and Decoders
        self.encoder_config = config["encoder"]
        self.decoder_config = config["decoder"]

        # Create encoder / decoder
        self.encoder = Encoder(self.encoder_config)
        self.decoder = Decoder(self.decoder_config)

        trainable_params = 0
        frozen_params = 0

        for name, param in self.decoder.named_parameters():
            if not param.requires_grad:
                print(f"🥶 Frozen layer '{name}'")
                frozen_params += param.numel()
            else:
                print(f"🚀 Trainable layer '{name}'")
                trainable_params += param.numel()
        
        print(f"Total frozen parameters: {frozen_params}")
        print(f"Total trainable parameters: {trainable_params}")
        print(f"Trainable Precentage: {(trainable_params / (frozen_params + trainable_params)) * 100:.3}%")

    def kl_loss(self, mean, logvar):
        # Kullback–Leibler Divergence with the prior Gaussian distribution
        loss = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        return loss
        
    def reparameterize(self, mean, logvar, eps=None, n_samples=1):
        # Reparameterization
        batch_size, latent_size = mean.size()
        std = logvar.mul(0.5).exp()

        if n_samples != 0:
            mean = mean.unsqueeze(1).expand(batch_size, n_samples, latent_size)
            std = logvar.unsqueeze(1).expand(batch_size, n_samples, latent_size)

        if eps is None:
            eps = torch.randn(std.size(), device=mean.device, dtype=mean.dtype)

        return eps.mul(std) + mean
    
    def forward(self, inputs, labels):
        # Encoder
        encoder_attention_mask = (inputs!=self.encoder.tokenizer.pad_token_id).float()
        encoder_outputs = self.encoder(inputs, encoder_attention_mask)

        # Connect
        latent_z = self.reparameterize(mean=encoder_outputs[0], logvar=encoder_outputs[1], n_samples=1).squeeze(1)

        # Decoder
        decoder_outputs = self.decoder(input_ids=inputs, latent_z=latent_z, labels=labels, attention_mask=None)

        # Loss
        kl_loss = self.kl_loss(mean=encoder_outputs[0], logvar=encoder_outputs[1])
        reconstruction_loss = decoder_outputs.loss

        return kl_loss, reconstruction_loss

if __name__ == "__main__":
    vae_config = {
        "encoder": {
            "model_name": 'bert-base-uncased',
            "tokenizer_name": 'bert-base-uncased',
            "hidden_size": 768,
            "latent_size": 128,
        },
        "decoder": {
            "model_name": 'gpt2',
            "tokenizer_name": 'gpt2',
            "hidden_size": 768,
            "latent_size": 128,
        },
    }

    vae = VAE(vae_config)

    inputs = vae.encoder.tokenizer("Hello, my dog is cute", return_tensors="pt")

    vae(inputs=inputs.input_ids, labels=inputs.input_ids)