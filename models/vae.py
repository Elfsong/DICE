# coding: utf-8
# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2023-07-22

# Definition of VAE encoder / decoder
import sys 
sys.path.append("..") 

import utils
import torch
from torch import nn
from peft import LoraConfig, TaskType, get_peft_model
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

        self.peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)   
        self.model = get_peft_model(self.model, self.peft_config)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hidden_size = config["hidden_size"]
        self.latent_size = config["latent_size"]

        self.latent_emb_layer = nn.Linear(self.latent_size, self.hidden_size, bias=False)

    def forward(self, input_ids, attention_mask, latent_z, labels, past_key_values=None):
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

        # Check encoder / decoder
        utils.model_check(self.encoder)
        utils.model_check(self.decoder)

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
    
    def forward(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask, decoder_labels):
        # Encoder
        encoder_attention_mask = (encoder_input_ids!=self.encoder.tokenizer.pad_token_id).float()
        encoder_outputs = self.encoder(encoder_input_ids, encoder_attention_mask)

        # Connect
        latent_z = self.reparameterize(mean=encoder_outputs[0], logvar=encoder_outputs[1], n_samples=1).squeeze(1)

        # Decoder
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, latent_z=latent_z, labels=decoder_labels)

        # Loss
        kl_loss = self.kl_loss(mean=encoder_outputs[0], logvar=encoder_outputs[1])
        reconstruction_loss = decoder_outputs.loss

        return kl_loss, reconstruction_loss, encoder_outputs, decoder_outputs

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

    input_str = "Chinese girls are"
    for i in range(10):
        encoder_inputs = vae.encoder.tokenizer([input_str], return_tensors="pt")
        decoder_inputs = vae.decoder.tokenizer([input_str], return_tensors="pt")

        kl_loss, reconstruction_loss, encoder_outputs, decoder_outputs = vae(
                encoder_input_ids=encoder_inputs["input_ids"], 
                encoder_attention_mask=encoder_inputs["attention_mask"],
                decoder_input_ids=decoder_inputs["input_ids"], 
                decoder_attention_mask=decoder_inputs["attention_mask"],
                decoder_labels=decoder_inputs["input_ids"],
        )
        logits = decoder_outputs[1]
        next_token = vae.decoder.tokenizer.decode(torch.argmax(logits[0][-1]))
        input_str += next_token
        print(input_str)