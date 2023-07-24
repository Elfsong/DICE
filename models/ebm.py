# coding: utf-8
# Author: Du Mingzhe (mingzhe@nus.edu.sg)
# Date: 2023-07-22

import torch

class LatentClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()

        self.n_classes = output_dim
        self.energy_weight = 1.0

        # TODO(mingzhe): dynamic layer modification with parameters
        self.pipeline = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=43),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=43, out_features=22),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(in_features=22, out_features=output_dim)
        )

    def forward(self, input):
        output = self.pipeline(input)
        return output