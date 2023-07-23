# coding: utf-8
# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2023-07-22

import numpy as np

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

def tokenize_function(example, tokenizer, prefix, min_length=10, max_length=500):
    # 1000 (< max_length 1024) to make sure _attn works
    # TODO(mingzhe): Change the hard code

    if len(example["text"]) >= min_length:
        tmp = tokenizer(example["text"], padding='max_length', max_length=max_length, truncation=True)
        return {prefix + str(key): val for key, val in tmp.items()}

def frange_cycle_zero_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio_increase=0.5, ratio_zero=0.25):
    L = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop-start) / (period * ratio_increase) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            if i < period * ratio_zero:
                L[int(i + c * period)] = start
            else:
                L[int(i + c * period)] = v
                v += step
            i += 1
    return L