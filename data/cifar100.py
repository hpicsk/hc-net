"""
CIFAR-100 Dataset loading utilities.

Loads CIFAR-100 from the local path with proper preprocessing.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from typing import Tuple, Optional, List


class CIFAR100Dataset(Dataset):
    """
    CIFAR-100 Dataset loader.

    Loads from the pickle files directly rather than using torchvision.
    """

    def __init__(
        self,
        root: str = '/home/st/common_dataset/cifar-100-python',
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        target_transform: Optional[callable] = None
    ):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # Load data
        if train:
            data_file = os.path.join(root, 'train')
        else:
            data_file = os.path.join(root, 'test')

        with open(data_file, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')

        self.data = data_dict[b'data']
        self.fine_labels = data_dict[b'fine_labels']
        self.coarse_labels = data_dict[b'coarse_labels']

        # Reshape data: [N, 3072] -> [N, 3, 32, 32]
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose(0, 2, 3, 1)  # [N, 32, 32, 3] for transforms

        # Load label names
        meta_file = os.path.join(root, 'meta')
        with open(meta_file, 'rb') as f:
            meta = pickle.load(f, encoding='bytes')

        self.fine_label_names = [s.decode('utf-8') for s in meta[b'fine_label_names']]
        self.coarse_label_names = [s.decode('utf-8') for s in meta[b'coarse_label_names']]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.data[idx]
        target = self.fine_labels[idx]

        # Convert to PIL Image for transforms
        from PIL import Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def get_coarse_label(self, idx: int) -> int:
        """Get coarse (superclass) label for an index."""
        return self.coarse_labels[idx]

    def get_labels_for_indices(self, indices: List[int]) -> Tuple[List[int], List[int]]:
        """Get fine and coarse labels for a list of indices."""
        fine = [self.fine_labels[i] for i in indices]
        coarse = [self.coarse_labels[i] for i in indices]
        return fine, coarse


def get_transforms(train: bool = True, augment: bool = True) -> transforms.Compose:
    """
    Get standard CIFAR-100 transforms.

    Args:
        train: Whether to get training transforms
        augment: Whether to apply data augmentation

    Returns:
        Transform composition
    """
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )

    if train and augment:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def get_cifar100_loaders(
    root: str = '/home/st/common_dataset/cifar-100-python',
    batch_size: int = 128,
    num_workers: int = 4,
    augment: bool = True,
    subset_fraction: Optional[float] = None,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-100 data loaders.

    Args:
        root: Path to CIFAR-100 data
        batch_size: Batch size
        num_workers: Number of data loading workers
        augment: Whether to apply data augmentation
        subset_fraction: If set, use only this fraction of training data
        seed: Random seed for subset selection

    Returns:
        (train_loader, test_loader)
    """
    train_transform = get_transforms(train=True, augment=augment)
    test_transform = get_transforms(train=False, augment=False)

    train_dataset = CIFAR100Dataset(root, train=True, transform=train_transform)
    test_dataset = CIFAR100Dataset(root, train=False, transform=test_transform)

    # Create subset if requested
    if subset_fraction is not None and subset_fraction < 1.0:
        np.random.seed(seed)
        n_samples = len(train_dataset)
        n_subset = int(n_samples * subset_fraction)
        indices = np.random.permutation(n_samples)[:n_subset]
        train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def get_class_balanced_subset(
    dataset: CIFAR100Dataset,
    samples_per_class: int,
    seed: int = 42
) -> Subset:
    """
    Create a class-balanced subset of the dataset.

    Args:
        dataset: Full CIFAR-100 dataset
        samples_per_class: Number of samples per class
        seed: Random seed

    Returns:
        Subset with balanced classes
    """
    np.random.seed(seed)

    # Group indices by class
    class_indices = {}
    for idx in range(len(dataset)):
        label = dataset.fine_labels[idx]
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)

    # Sample from each class
    selected_indices = []
    for label in sorted(class_indices.keys()):
        indices = class_indices[label]
        np.random.shuffle(indices)
        selected_indices.extend(indices[:samples_per_class])

    return Subset(dataset, selected_indices)


def get_superclass_split(
    root: str = '/home/st/common_dataset/cifar-100-python',
    holdout_superclasses: List[int] = None,
    batch_size: int = 128,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Split dataset by superclass for zero-shot evaluation.

    Args:
        root: Path to CIFAR-100 data
        holdout_superclasses: List of superclass indices to hold out
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        (train_loader, test_seen_loader, test_unseen_loader)
    """
    if holdout_superclasses is None:
        holdout_superclasses = [0, 10]  # Hold out 2 superclasses

    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    train_dataset = CIFAR100Dataset(root, train=True, transform=train_transform)
    test_dataset = CIFAR100Dataset(root, train=False, transform=test_transform)

    # Split by superclass
    train_indices = []
    for idx in range(len(train_dataset)):
        if train_dataset.coarse_labels[idx] not in holdout_superclasses:
            train_indices.append(idx)

    test_seen_indices = []
    test_unseen_indices = []
    for idx in range(len(test_dataset)):
        if test_dataset.coarse_labels[idx] not in holdout_superclasses:
            test_seen_indices.append(idx)
        else:
            test_unseen_indices.append(idx)

    train_subset = Subset(train_dataset, train_indices)
    test_seen_subset = Subset(test_dataset, test_seen_indices)
    test_unseen_subset = Subset(test_dataset, test_unseen_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    test_seen_loader = DataLoader(test_seen_subset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)
    test_unseen_loader = DataLoader(test_unseen_subset, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)

    return train_loader, test_seen_loader, test_unseen_loader
