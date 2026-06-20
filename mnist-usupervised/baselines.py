''' Implementation of MLP and CNN baselines to compare with our JEPA'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    '''
        Multilayer Perceptron implementation.
    '''

    def __init__(self, config, input_size, output_size):
        super().__init__()

        sizes = [input_size, *config['mlp']['layer_sizes'], output_size]

        layers = []
        for in_dim, out_dim in zip(sizes[:-2], sizes[1:-1]):
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
            ])

        layers.append(nn.Linear(sizes[-2], sizes[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(torch.flatten(x, start_dim=1))
    

