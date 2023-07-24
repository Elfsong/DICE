# coding: utf-8
# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2023-07-22

# Environment
import os
import sys
sys.path.append("..")
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

import utils
import wandb
import torch
from tqdm import tqdm
from models.vae import VAE
from torch.optim import Adam
from datasets import load_dataset
from transformers import get_scheduler
from torch.utils.data import DataLoader

# Wandb Config
wandb.init(
    project="DICE",
    
    # Hypterparameters
    config={
        "epochs": 1,
        "learning_rate": 1e-5,
        "latent_size": 128,
        "hidden_size": 768,
        "train_sample_size": 5000,
        "eval_sample_size": 1000,
        "train_batch_size": 1,
        "eval_batch_size": 4,
        "random_seed": 42,
        "num_warmup_steps": 400,
        "show_progress": 100,
    }
)

# Load Model and Tokenizer
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

vae_config = {
    "encoder": {
        "model_name": 'bert-base-uncased',
        "tokenizer_name": 'bert-base-uncased',
        "hidden_size": 768,
        "latent_size": wandb.config["latent_size"],
    },
    "decoder": {
        "model_name": 'gpt2',
        "tokenizer_name": 'gpt2',
        "hidden_size": 768,
        "latent_size": wandb.config["latent_size"],
    },
}

# Load Model
print("Load Model")
vae = VAE(vae_config).to(device)

# Load Dataset
print("Load Dataset")
datasets = load_dataset("wikitext", "wikitext-2-v1")
datasets = datasets.map(lambda example: utils.tokenize_function(example, vae.encoder.tokenizer, "encoder_"), batched=True)
datasets = datasets.map(lambda example: utils.tokenize_function(example, vae.decoder.tokenizer, "decoder_"), batched=True)
tokenized_datasets = datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

train_dataset = tokenized_datasets["train"].shuffle(seed=wandb.config["random_seed"]).select(range(wandb.config["train_sample_size"]))
eval_dataset = tokenized_datasets["validation"].shuffle(seed=wandb.config["random_seed"]).select(range(wandb.config["eval_sample_size"]))

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=wandb.config["train_batch_size"])
eval_dataloader = DataLoader(eval_dataset, batch_size=wandb.config["eval_batch_size"])

# Optimizer & Scheduler
print("Optimizer & Scheduler")
optimizer = Adam(vae.parameters(), lr=wandb.config["learning_rate"])

num_training_steps = len(train_dataloader) * wandb.config["epochs"]
num_eval_steps = len(eval_dataloader)

lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=wandb.config["num_warmup_steps"], num_training_steps=num_training_steps)
cyclic_weights = utils.frange_cycle_zero_linear(num_training_steps+1, start=0, stop=1, n_cycle=4, ratio_increase=0.25, ratio_zero=0.25)

# Training
train_progress_bar = tqdm(range(num_training_steps))

for epoch in range(wandb.config["epochs"]):
    vae.train()
    utils.model_check(vae)
    for index, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        kl_loss, reconstruction_loss, encoder_outputs, decoder_outputs = vae(
            encoder_input_ids=batch["encoder_input_ids"], 
            encoder_attention_mask=batch["encoder_attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"], 
            decoder_attention_mask=batch["decoder_attention_mask"],
            decoder_labels=batch["decoder_input_ids"],
        )

        # Loss Compute
        cyclical_annealing = cyclic_weights[train_progress_bar.n]
        total_loss = reconstruction_loss.mean() + kl_loss * cyclical_annealing

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)

        wandb.log({
            "reconstruction_loss": reconstruction_loss.mean(),
            "kl_loss": kl_loss,
            "total_loss": total_loss,
            "cyclical_annealing": cyclical_annealing,
        })

        if train_progress_bar.n % wandb.config["show_progress"] == 0:
            print({
                "recon_loss": reconstruction_loss.mean(),
                "kl_loss": kl_loss,
                "total_loss": total_loss,
                "cyclical_annealing": cyclical_annealing,
            })

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        train_progress_bar.update(1)

print("⏱️ Saving the model...")
torch.save(vae.state_dict(), os.path.join("../checkpoints", f'model_latest.pt'))
print("🟢 Model Saved")

new_vae = VAE(vae_config)
checkpoint = torch.load(os.path.join("../checkpoints", f'model_latest.pt'))
new_vae.load_state_dict(checkpoint, strict=False)
new_vae.to(device)

# Play Play
vae.eval()

input_str = "Many people live in"
for i in range(10):
    encoder_inputs = vae.encoder.tokenizer([input_str], return_tensors="pt").to(device)
    decoder_inputs = vae.decoder.tokenizer([input_str], return_tensors="pt").to(device)

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

input_str = "Many people live in"
for i in range(10):
    encoder_inputs = vae.encoder.tokenizer([input_str], return_tensors="pt").to(device)
    decoder_inputs = vae.decoder.tokenizer([input_str], return_tensors="pt").to(device)

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

input_str = "Many people live in"
for i in range(10):
    encoder_inputs = new_vae.encoder.tokenizer([input_str], return_tensors="pt").to(device)
    decoder_inputs = new_vae.decoder.tokenizer([input_str], return_tensors="pt").to(device)

    kl_loss, reconstruction_loss, encoder_outputs, decoder_outputs = new_vae(
            encoder_input_ids=encoder_inputs["input_ids"], 
            encoder_attention_mask=encoder_inputs["attention_mask"],
            decoder_input_ids=decoder_inputs["input_ids"], 
            decoder_attention_mask=decoder_inputs["attention_mask"],
            decoder_labels=decoder_inputs["input_ids"],
    )
    logits = decoder_outputs[1]
    next_token = new_vae.decoder.tokenizer.decode(torch.argmax(logits[0][-1]))
    input_str += next_token
    print(input_str)

input_str = "She comes from"
for i in range(10):
    encoder_inputs = vae.encoder.tokenizer([input_str], return_tensors="pt").to(device)
    decoder_inputs = vae.decoder.tokenizer([input_str], return_tensors="pt").to(device)

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

input_str = "Why you likes"
for i in range(10):
    encoder_inputs = vae.encoder.tokenizer([input_str], return_tensors="pt").to(device)
    decoder_inputs = vae.decoder.tokenizer([input_str], return_tensors="pt").to(device)

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

input_str = "Chinese girls are"
for i in range(10):
    encoder_inputs = vae.encoder.tokenizer([input_str], return_tensors="pt").to(device)
    decoder_inputs = vae.decoder.tokenizer([input_str], return_tensors="pt").to(device)

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