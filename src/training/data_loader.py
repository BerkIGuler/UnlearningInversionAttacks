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

def load_cifar10(data_path='./data', batch_size=128, public_split=0.8, augment=True, seed=233, excluded_class=None, excluded_proportion=0.0):
    """
    Load CIFAR-10 dataset and split into:
    - D0: public training data
    - Du: private fine-tuning data
    - DuX: Du minus excluded subset X
    - X: excluded subset (used in attack evaluation)
    - test_loader: CIFAR-10 test data

    Args:
        excluded_class (int): class label i to remove from Du
        excluded_fraction (float): fraction pi of class-i to exclude

    Returns:
        D0_loader, Du_loader, DuX_loader, X_loader, test_loader
    """
    # Train and test transforms
    transform_train = get_transforms(augment=augment)
    transform_test = get_transforms(augment=False)

    # Load the full CIFAR-10 training dataset
    #download the dataset and apply the transform
    full_train = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

    # Get sizes to split D0 and Du
    total_len  = len(full_train)
    D0_len = int(public_split * total_len)
    Du_len = total_len - D0_len

    #split the full_train dataset into D0 and Du
    generator = torch.Generator().manual_seed(seed)
    D0, Du = random_split(full_train, [D0_len, Du_len], generator=generator)

    #create loaders
    D0_loader = DataLoader(D0, batch_size=batch_size, shuffle=True, num_workers=2)
    Du_loader = DataLoader(Du, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    DuX_loader = None
    X_loader = None

    # Subsample Du into DuX and X
    if excluded_class is not None and excluded_proportion > 0:
        Du_targets = [Du.dataset[Du.indices[i]][1] for i in range(len(Du))]
        class_indices = [i for i, label in enumerate(Du_targets) if label == excluded_class]

        exclude_count = int(len(class_indices) * excluded_proportion)
        excluded_indices = set(class_indices[:exclude_count])  # pick top pi% of class-i

        DuX_indices = [i for i in range(len(Du)) if i not in excluded_indices]
        X_indices = [i for i in range(len(Du)) if i in excluded_indices]

        DuX = torch.utils.data.Subset(Du, DuX_indices)
        X = torch.utils.data.Subset(Du, X_indices)

        DuX_loader = DataLoader(DuX, batch_size=batch_size, shuffle=True, num_workers=2)
        X_loader = DataLoader(X, batch_size=batch_size, shuffle=False, num_workers=2)

    return D0_loader, Du_loader, DuX_loader, X_loader, test_loader
