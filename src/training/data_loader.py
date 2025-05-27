import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
import numpy as np

cifar10_mean = (0.4914672374725342, 0.4822617471218109, 0.4467701315879822)
cifar10_std = (0.24703224003314972, 0.24348513782024384, 0.26158785820007324)

def get_transforms(augment=False, normalize=True):
    base_transform = []
    if augment:
        base_transform += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    base_transform += [transforms.ToTensor()]
    if normalize:
        base_transform += [transforms.Normalize(cifar10_mean, cifar10_std)]
    return transforms.Compose(base_transform)

def load_cifar10(data_path='./data', batch_size=128, public_split=0.8, augment=True):
    """
    Load CIFAR-10 dataset and split it into D0 (public) and Du (private).
    
    Returns:
        D0_loader: DataLoader for public training data
        Du_loader: DataLoader for private fine-tuning data
        test_loader: DataLoader for test data
    """
    # Train and test transforms
    transform_train = get_transforms(augment=augment)
    transform_test = get_transforms(augment=False)

    # Load the full CIFAR-10 training dataset
    #download the dataset and apply the transform
    full_train = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

    # Split into D0 and Du
    total_len  = len(full_train)
    
    #set the size of D0 (set by public_split in parameters) and Du
    D0_len = int(public_split * total_len)
    Du_len = total_len - D0_len

    #split the full_train dataset into D0 and Du
    #set the seed to 42
    generator = torch.Generator().manual_seed(233)
    D0, Du = random_split(full_train, [D0_len, Du_len], generator=generator)

    D0_loader = DataLoader(D0, batch_size=batch_size, shuffle=True, num_workers=2)
    Du_loader = DataLoader(Du, batch_size=batch_size, shuffle=True, num_workers=2)

    #Grab the CIFAR-10 test dataset and create a DataLoader
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return D0_loader, Du_loader, test_loader
