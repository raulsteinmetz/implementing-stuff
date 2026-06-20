import yaml

import torch
import torch.nn as nn
import torch.optim as optim

from mnist import get_loaders
from baselines import MLP


def init_mlp(config, loader):
    # get underlying datastet from loader
    dataset = loader.dataset.dataset

    # find out image shape
    sample, _ = dataset[0]
    img_shape = input_dim=sample.numel() # numel() returns total elements over all dimentions

    # init and return the model
    return MLP(
        config,
        sample.numel(),
        len(dataset.classes)
    )


def train_supervised(config, model, train_loader, eval_loader):
    # setup

    config = config['mlp']
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    loss = nn.CrossEntropyLoss()

    # loop
    model.train()
    for epoch in range(config['epochs']):
        pass




def main(config):
    # Load Dataset
    train_loader, eval_loader, test_loader = get_loaders(config)
    print('MNIST dataset loaded.', end='\n'*2)

    # Init MLP baseline
    mlp = init_mlp(config, train_loader)
    print(f'Multilayer Perceptron model initialized. Shape: \n{mlp}', end='\n'*2)

if __name__ == '__main__':
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    main(config)