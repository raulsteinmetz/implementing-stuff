''' 
    Functions for downloading MNIST and creating dataloaders
'''

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms, datasets


def get_loaders(config):
    '''
        Downloads data, trasnforms it, creates data loaders
    '''

    # Define data handling steps (not applied in the saved data, only in the program)
    transform = transforms.Compose([
        transforms.ToTensor(), # to PyTorch tensor
        transforms.Normalize((0.1307,), (0.3081,)) # normalize images so that the entire dataset has mean=0 and std=1, respectively
    ])

    # Download and load datasets into memory
    _train_dataset = datasets.MNIST( # train, val
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST( # test
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # Split train dataset into training and validation sets
    val_size = int(len(_train_dataset) * config['dataset']['val_split'])
    train_size = len(_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        _train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config['seed'])
    )

    # Create a Loader object that samples random batches from data
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False
    )
    test_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False
    )

    return train_loader, val_loader, test_loader