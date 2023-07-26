# coding: utf-8
# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2023-07-22

# Device Env
import os
import sys
sys.path.append("..")
os.environ['CUDA_VISIBLE_DEVICES'] = "4"

import utils
import wandb
import torch
import pandas as pd
from torch import nn
from tqdm import tqdm
from models.vae import VAE
from datasets import Dataset
from torch.optim import Adam
from transformers import get_scheduler
from models.ebm import LatentClassifier
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Wandb Config
wandb.init(
    project="DICE-EBM",
    
    # Hypterparameters
    config={
        "epochs": 3,
        "learning_rate": 3e-4,
        "num_warmup_steps": 50,
        "random_seed": 42,
        "train_batch_size": 16,
        "eval_batch_size": 4,
        "latent_size": 128,
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
vae = VAE(vae_config)
checkpoint = torch.load(os.path.join("../checkpoints", f'model_latest.pt'))
vae.load_state_dict(checkpoint, strict=False)
vae.to(device)
# vae.eval()
# utils.model_freeze(vae)

classifier = LatentClassifier(input_dim=128, output_dim=3, num_classes=3, depth=3)
classifier.to(device)

# utils.model_check(vae)
# utils.model_check(classifier)

## Load Data
gender_df = pd.read_csv("/home/mingzhe/Projects/DebiasODE/src/data/gender_dataset.csv", index_col=0)
gender_dataset = Dataset.from_pandas(gender_df)
gender_dataset = gender_dataset.train_test_split(train_size=0.9, test_size=0.1, shuffle=True)
gender_dataset = gender_dataset.map(lambda example: utils.tokenize_function(example, vae.encoder.tokenizer, "encoder_"), batched=True)
gender_dataset = gender_dataset.map(lambda example: utils.tokenize_function(example, vae.decoder.tokenizer, "decoder_"), batched=True)
gender_dataset = gender_dataset.map(lambda example: utils.label_function(example), batched=True)
gender_dataset = gender_dataset.remove_columns(["text", "sid", "__index_level_0__"])
gender_dataset.set_format("torch")

train_dataset = gender_dataset["train"].shuffle(seed=wandb.config["random_seed"])
eval_dataset = gender_dataset["test"].shuffle(seed=wandb.config["random_seed"])

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=wandb.config["train_batch_size"])
eval_dataloader = DataLoader(eval_dataset, batch_size=wandb.config["eval_batch_size"])

# Load Optimizors and Schedulers
optimizer = Adam(list(vae.parameters()) + list(classifier.parameters()), lr=wandb.config["learning_rate"])

num_epochs = wandb.config["epochs"]
num_training_steps = len(train_dataloader) * num_epochs
num_eval_steps = len(eval_dataloader)

lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=wandb.config["num_warmup_steps"], num_training_steps=num_training_steps)

# Evaluation
def evaluate(verbose=False):
    vae.eval()
    classifier.eval()

    pred_list, gold_list = list(), list()

    for index, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}

            # Get Encoder Representation
            kl_loss, reconstruction_loss, latent_z, encoder_outputs, decoder_outputs = vae(
                encoder_input_ids=batch["encoder_input_ids"], 
                encoder_attention_mask=batch["encoder_attention_mask"],
                decoder_input_ids=batch["decoder_input_ids"], 
                decoder_attention_mask=batch["decoder_attention_mask"],
                decoder_labels=batch["decoder_input_ids"],
            )
            
            # Classifier Output
            classifier_output = classifier(latent_z)

            pred_list += classifier_output.argmax(dim=1).tolist()
            gold_list += batch["label"].tolist()
    if verbose:
        print(classification_report(gold_list, pred_list, target_names=['male', 'female', 'neutral']))
    acc = accuracy_score(gold_list, pred_list)

    wandb.log({"acc": acc})
    print({"acc": acc})
    print("==" * 50)

# Training
train_progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
    classifier.train()
    for index, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}

        # Get Encoder Representation
        kl_loss, reconstruction_loss, latent_z, encoder_outputs, decoder_outputs = vae(
            encoder_input_ids=batch["encoder_input_ids"], 
            encoder_attention_mask=batch["encoder_attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"], 
            decoder_attention_mask=batch["decoder_attention_mask"],
            decoder_labels=batch["decoder_input_ids"],
        )

        # Train Classifier
        # classifier_output = classifier(encoder_outputs[2].pooler_output)
        classifier_output = classifier(latent_z)

        # # Loss Compute
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(classifier_output.view(-1, 3), batch["label"].view(-1).to(device))
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        train_progress_bar.update(1)
        wandb.log({"train_loss": loss})

        if train_progress_bar.n % 50 == 0:
            print({"train_loss": loss})
            evaluate(verbose=True)

