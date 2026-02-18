"""
Compositional Split for CIFAR-100.

Creates training/test splits that test compositional generalization:
- Train on subset of (superclass, fine_class) combinations
- Test on held-out combinations to measure zero-shot composition
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, List, Dict, Set
from .cifar100 import CIFAR100Dataset, get_transforms


def create_compositional_split(
    root: str = '/home/st/common_dataset/cifar-100-python',
    holdout_per_superclass: int = 1,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Create indices for compositional train/test split.

    CIFAR-100 has 20 superclasses, each with 5 fine classes.
    We hold out `holdout_per_superclass` fine classes from each superclass
    for testing compositional generalization.

    Example:
    - Train: "aquatic mammals: beaver, dolphin, otter, whale"
    - Test (holdout): "aquatic mammals: seal"

    The model should learn the concept of "aquatic mammal" and recognize
    a seal even if never seen during training.

    Args:
        root: Path to CIFAR-100
        holdout_per_superclass: Number of fine classes to hold out per superclass
        seed: Random seed

    Returns:
        (train_indices, test_seen_indices, test_holdout_indices, holdout_fine_classes)
    """
    np.random.seed(seed)

    # Load dataset to get labels
    dataset = CIFAR100Dataset(root, train=True)

    # Build mapping: superclass -> [fine classes]
    superclass_to_fine = {}
    for idx in range(len(dataset)):
        coarse = dataset.coarse_labels[idx]
        fine = dataset.fine_labels[idx]
        if coarse not in superclass_to_fine:
            superclass_to_fine[coarse] = set()
        superclass_to_fine[coarse].add(fine)

    # For each superclass, randomly select fine classes to hold out
    holdout_fine_classes = set()
    for coarse in sorted(superclass_to_fine.keys()):
        fine_classes = list(superclass_to_fine[coarse])
        np.random.shuffle(fine_classes)
        holdout = fine_classes[:holdout_per_superclass]
        holdout_fine_classes.update(holdout)

    # Split training data
    train_indices = []
    for idx in range(len(dataset)):
        if dataset.fine_labels[idx] not in holdout_fine_classes:
            train_indices.append(idx)

    # Split test data
    test_dataset = CIFAR100Dataset(root, train=False)
    test_seen_indices = []
    test_holdout_indices = []
    for idx in range(len(test_dataset)):
        if test_dataset.fine_labels[idx] not in holdout_fine_classes:
            test_seen_indices.append(idx)
        else:
            test_holdout_indices.append(idx)

    return train_indices, test_seen_indices, test_holdout_indices, list(holdout_fine_classes)


class CompositionallySplitCIFAR100:
    """
    CIFAR-100 with compositional train/test split.

    Provides dataloaders for:
    - Training on seen (superclass, fine_class) combinations
    - Testing on seen combinations
    - Testing on held-out combinations (compositional generalization)
    """

    def __init__(
        self,
        root: str = '/home/st/common_dataset/cifar-100-python',
        holdout_per_superclass: int = 1,
        batch_size: int = 128,
        num_workers: int = 4,
        seed: int = 42
    ):
        self.root = root
        self.holdout_per_superclass = holdout_per_superclass
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # Create splits
        (self.train_indices,
         self.test_seen_indices,
         self.test_holdout_indices,
         self.holdout_classes) = create_compositional_split(
             root, holdout_per_superclass, seed
         )

        # Load datasets
        train_transform = get_transforms(train=True)
        test_transform = get_transforms(train=False)

        self.train_dataset = CIFAR100Dataset(root, train=True, transform=train_transform)
        self.test_dataset = CIFAR100Dataset(root, train=False, transform=test_transform)

        # Create subsets
        self.train_subset = Subset(self.train_dataset, self.train_indices)
        self.test_seen_subset = Subset(self.test_dataset, self.test_seen_indices)
        self.test_holdout_subset = Subset(self.test_dataset, self.test_holdout_indices)

    def get_train_loader(self) -> DataLoader:
        """Get training data loader (seen classes only)."""
        return DataLoader(
            self.train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def get_test_seen_loader(self) -> DataLoader:
        """Get test loader for seen class combinations."""
        return DataLoader(
            self.test_seen_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_test_holdout_loader(self) -> DataLoader:
        """Get test loader for held-out class combinations."""
        return DataLoader(
            self.test_holdout_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_all_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get all three loaders."""
        return (
            self.get_train_loader(),
            self.get_test_seen_loader(),
            self.get_test_holdout_loader()
        )

    def get_holdout_class_names(self) -> List[str]:
        """Get names of held-out fine classes."""
        return [self.train_dataset.fine_label_names[c] for c in self.holdout_classes]

    def get_split_statistics(self) -> Dict:
        """Get statistics about the split."""
        return {
            'num_train_samples': len(self.train_indices),
            'num_test_seen_samples': len(self.test_seen_indices),
            'num_test_holdout_samples': len(self.test_holdout_indices),
            'num_holdout_classes': len(self.holdout_classes),
            'holdout_class_names': self.get_holdout_class_names(),
            'num_seen_classes': 100 - len(self.holdout_classes),
        }


class AttributeCompositionSplit:
    """
    Create splits based on attribute composition.

    Uses CIFAR-100's hierarchical structure where superclass = "attribute"
    and fine class = "object type".

    Train on:
    - All combinations except specific (attribute, object) pairs

    Test on:
    - Held-out (attribute, object) pairs

    This tests whether the network can compose learned attributes
    with learned objects in novel combinations.
    """

    def __init__(
        self,
        root: str = '/home/st/common_dataset/cifar-100-python',
        num_holdout_pairs: int = 20,
        seed: int = 42
    ):
        self.root = root
        self.seed = seed

        np.random.seed(seed)

        # Load dataset
        dataset = CIFAR100Dataset(root, train=True)

        # Build all (superclass, fine_class) pairs
        all_pairs = set()
        pair_to_indices = {}  # (coarse, fine) -> [indices]

        for idx in range(len(dataset)):
            coarse = dataset.coarse_labels[idx]
            fine = dataset.fine_labels[idx]
            pair = (coarse, fine)
            all_pairs.add(pair)
            if pair not in pair_to_indices:
                pair_to_indices[pair] = []
            pair_to_indices[pair].append(idx)

        # Select holdout pairs
        # Strategy: ensure each superclass has at least 4 fine classes in training
        all_pairs_list = list(all_pairs)
        np.random.shuffle(all_pairs_list)

        self.holdout_pairs = set()
        superclass_holdout_count = {}

        for pair in all_pairs_list:
            coarse, fine = pair
            if superclass_holdout_count.get(coarse, 0) < 1:  # Max 1 per superclass
                self.holdout_pairs.add(pair)
                superclass_holdout_count[coarse] = superclass_holdout_count.get(coarse, 0) + 1
                if len(self.holdout_pairs) >= num_holdout_pairs:
                    break

        # Build train/test splits
        self.train_indices = []
        for pair, indices in pair_to_indices.items():
            if pair not in self.holdout_pairs:
                self.train_indices.extend(indices)

        # Test splits
        test_dataset = CIFAR100Dataset(root, train=False)
        test_pair_to_indices = {}
        for idx in range(len(test_dataset)):
            coarse = test_dataset.coarse_labels[idx]
            fine = test_dataset.fine_labels[idx]
            pair = (coarse, fine)
            if pair not in test_pair_to_indices:
                test_pair_to_indices[pair] = []
            test_pair_to_indices[pair].append(idx)

        self.test_seen_indices = []
        self.test_holdout_indices = []
        for pair, indices in test_pair_to_indices.items():
            if pair in self.holdout_pairs:
                self.test_holdout_indices.extend(indices)
            else:
                self.test_seen_indices.extend(indices)

        self.dataset = dataset
        self.test_dataset = test_dataset

    def get_holdout_pair_names(self) -> List[Tuple[str, str]]:
        """Get (superclass_name, fine_class_name) for holdout pairs."""
        names = []
        for coarse, fine in self.holdout_pairs:
            coarse_name = self.dataset.coarse_label_names[coarse]
            fine_name = self.dataset.fine_label_names[fine]
            names.append((coarse_name, fine_name))
        return names
