import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

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


def load_cifar10(
        data_path='./data', batch_size=128, public_split=0.8,
        augment=False, seed=233, exclude_class=None, exclude_prop=0.0):
    """
    Load CIFAR-10 dataset and split into:
    - D0: public training data
    - Du: private fine-tuning data
    - DuX: Du minus excluded subset X
    - X: excluded subset (used in attack evaluation)
    - test_loader: CIFAR-10 test data

    Args:
        data_path (str, optional): Path to store/load CIFAR-10 data. Defaults to './data'.
        batch_size (int, optional): Batch size for all data loaders. Defaults to 128.
        public_split (float, optional): Fraction of training data used for D0 (public).
            Remaining (1 - public_split) used for Du (private). Defaults to 0.8.
        augment (bool, optional): Whether to apply data augmentation to training data.
            Defaults to False.
        seed (int, optional): Random seed for reproducible train/private split. Defaults to 233.
        exclude_class (int, optional): Class label to remove from Du for unlearning experiments.
            If None, no exclusion is performed. Defaults to None.
        exclude_prop (float, optional): Proportion of samples from exclude_class to remove
            from Du (creates subset X). Must be between 0.0 and 1.0. Defaults to 0.0.

    Returns:
        tuple: A tuple containing:
            - D0_loader (DataLoader): Public training data loader
            - Du_loader (DataLoader): Private fine-tuning data loader
            - DuX_loader (DataLoader or None): Du minus excluded samples, or None if no exclusion
            - X_loader (DataLoader or None): Excluded samples for attack evaluation, or None if no exclusion
            - test_loader (DataLoader): CIFAR-10 test data loader
    """
    transform_train = get_transforms(augment=augment)
    transform_test = get_transforms(augment=False)

    full_train = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

    total_len  = len(full_train)
    D0_len = int(public_split * total_len)
    Du_len = total_len - D0_len

    generator = torch.Generator().manual_seed(seed)
    D0, Du = random_split(full_train, [D0_len, Du_len], generator=generator)

    D0_loader = DataLoader(D0, batch_size=batch_size, shuffle=True, num_workers=2)
    Du_loader = DataLoader(Du, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    DuX_loader = None
    X_loader = None

    if exclude_class is not None and exclude_prop > 0:
        Du_targets = [Du.dataset[Du.indices[i]][1] for i in range(len(Du))]
        class_indices = [i for i, label in enumerate(Du_targets) if label == exclude_class]

        exclude_count = int(len(class_indices) * exclude_prop)
        excluded_indices = set(class_indices[:exclude_count])  # pick top pi% of class-i

        DuX_indices = [i for i in range(len(Du)) if i not in excluded_indices]
        X_indices = [i for i in range(len(Du)) if i in excluded_indices]

        DuX = torch.utils.data.Subset(Du, DuX_indices)
        X = torch.utils.data.Subset(Du, X_indices)

        DuX_loader = DataLoader(DuX, batch_size=batch_size, shuffle=True, num_workers=2)
        X_loader = DataLoader(X, batch_size=batch_size, shuffle=False, num_workers=2)

    return D0_loader, Du_loader, DuX_loader, X_loader, test_loader
