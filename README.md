# 🎲 DICE

## Step 1. Define the architecture of encoders and decoders
* Encoder: BERT
* Decoder: GPT-2 / Llama
* Dataset: [C4](https://huggingface.co/datasets/c4) / [WikiText](https://huggingface.co/datasets/wikitext)

## Step 2. Train the model in the VAE manner
* Encoder: Full-sized parameters
* Decoder: LoRA

## Step 3. Train EBMs
MLP classifiers for each attribute.

## Step 4. Inference
ODE sampler (guided by the EBMs) on the VAE latent variable.

