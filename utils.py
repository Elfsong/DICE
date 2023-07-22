# coding: utf-8
# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2023-07-22

def model_check(model):
    print("=" * 50)
    trainable_params = 0
    frozen_params = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"🥶 Frozen layer '{name}'")
            frozen_params += param.numel()
        else:
            print(f"🚀 Trainable layer '{name}'")
            trainable_params += param.numel()
    
    print(f"Total frozen parameters: {frozen_params}")
    print(f"Total trainable parameters: {trainable_params}")
    print(f"Trainable Precentage: {(trainable_params / (frozen_params + trainable_params)) * 100:.3}%")