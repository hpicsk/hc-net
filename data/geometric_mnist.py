"""
Geometric MNIST Dataset for compositional binding experiments.

MNIST digits with Color x Orientation attributes for testing
whether PCNN learns compositional rules more efficiently.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from PIL import Image
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass


@dataclass
class GeometricMNISTSample:
    """Metadata for a Geometric MNIST sample."""
    digit: int
    color: int  # 0=Red, 1=Blue
    orientation: int  # 0=0°, 1=90°, 2=180°, 3=270°
    label: int  # Binary label from compositional rule


class GeometricMNISTDataset(Dataset):
    """
    Geometric MNIST: MNIST digits with Color x Orientation attributes.

    Attributes:
        - Color: Red (0) or Blue (1)
        - Orientation: 0°, 90°, 180°, 270° (encoded as 0, 1, 2, 3)

    Binary classification based on compositional rule:
        Positive = (Red AND Vertical) OR (Blue AND Horizontal)
        where Vertical = {0°, 180°}, Horizontal = {90°, 270°}

    This tests whether networks can learn the *binding* between
    color and orientation, not just individual features.
    """

    def __init__(
        self,
        root: str = '/home/st/common_dataset',
        train: bool = True,
        samples_per_class: Optional[int] = None,
        rule: str = 'color_x_orientation',
        seed: int = 42,
        transform: Optional[transforms.Compose] = None,
        image_size: int = 32
    ):
        """
        Args:
            root: Path to MNIST data directory
            train: Whether to use training or test split
            samples_per_class: If set, limit samples per binary class
            rule: Compositional rule to use
            seed: Random seed for reproducibility
            transform: Additional transforms to apply
            image_size: Output image size (default 32 for CIFAR compatibility)
        """
        self.root = root
        self.train = train
        self.samples_per_class = samples_per_class
        self.rule = rule
        self.seed = seed
        self.transform = transform
        self.image_size = image_size

        # Load base MNIST
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=None
        )

        # Generate all attribute combinations
        np.random.seed(seed)
        self._generate_dataset()

        # Apply class-balanced sampling if requested
        if samples_per_class is not None:
            self._balance_classes()

    def _generate_dataset(self):
        """Generate dataset with all attribute combinations."""
        n_mnist = len(self.mnist)

        # Each MNIST sample gets random color and orientation
        # This creates 8x the data (2 colors x 4 orientations)
        # but we randomly assign to avoid excessive data

        self.indices = []  # (mnist_idx, color, orientation)
        self.labels = []

        # Assign attributes to each MNIST sample
        for mnist_idx in range(n_mnist):
            # Randomly assign color and orientation
            color = np.random.randint(0, 2)
            orientation = np.random.randint(0, 4)

            # Compute label
            label = self._compute_label(color, orientation)

            self.indices.append((mnist_idx, color, orientation))
            self.labels.append(label)

        self.indices = np.array(self.indices)
        self.labels = np.array(self.labels)

    def _compute_label(self, color: int, orientation: int) -> int:
        """
        Compute binary label based on compositional rule.

        Rule: Positive if (Red AND Vertical) OR (Blue AND Horizontal)
        - Red = color 0, Blue = color 1
        - Vertical = orientation in {0, 2} (0° or 180°)
        - Horizontal = orientation in {1, 3} (90° or 270°)
        """
        is_red = (color == 0)
        is_blue = (color == 1)
        is_vertical = (orientation in [0, 2])
        is_horizontal = (orientation in [1, 3])

        if self.rule == 'color_x_orientation':
            # Default XOR-like rule requiring binding
            positive = (is_red and is_vertical) or (is_blue and is_horizontal)
        elif self.rule == 'color_only':
            # Simple rule: just color (no binding needed)
            positive = is_red
        elif self.rule == 'orientation_only':
            # Simple rule: just orientation (no binding needed)
            positive = is_vertical
        else:
            raise ValueError(f"Unknown rule: {self.rule}")

        return 1 if positive else 0

    def _balance_classes(self):
        """Create class-balanced subset."""
        np.random.seed(self.seed + 1)

        # Find indices for each class
        pos_indices = np.where(self.labels == 1)[0]
        neg_indices = np.where(self.labels == 0)[0]

        # Sample from each class
        n_samples = self.samples_per_class

        np.random.shuffle(pos_indices)
        np.random.shuffle(neg_indices)

        selected_pos = pos_indices[:n_samples]
        selected_neg = neg_indices[:n_samples]

        selected = np.concatenate([selected_pos, selected_neg])
        np.random.shuffle(selected)

        self.indices = self.indices[selected]
        self.labels = self.labels[selected]

    def _apply_color(self, img: np.ndarray, color: int) -> np.ndarray:
        """
        Apply red or blue tint to grayscale image.

        Args:
            img: Grayscale image [H, W] with values in [0, 255]
            color: 0 for red, 1 for blue

        Returns:
            RGB image [3, H, W] with values in [0, 255]
        """
        img = img.astype(np.float32) / 255.0
        rgb = np.zeros((3, img.shape[0], img.shape[1]), dtype=np.float32)

        if color == 0:  # Red
            rgb[0] = img  # Full red
            rgb[1] = img * 0.15  # Slight green for visibility
            rgb[2] = img * 0.15  # Slight blue
        else:  # Blue
            rgb[0] = img * 0.15
            rgb[1] = img * 0.15
            rgb[2] = img  # Full blue

        return (rgb * 255).astype(np.uint8)

    def _apply_rotation(self, img: Image.Image, orientation: int) -> Image.Image:
        """
        Apply rotation to image.

        Args:
            img: PIL Image
            orientation: 0=0°, 1=90°, 2=180°, 3=270°

        Returns:
            Rotated PIL Image
        """
        angle = int(orientation) * 90  # Convert to Python int for torchvision
        return TF.rotate(img, angle)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        mnist_idx, color, orientation = self.indices[idx]
        label = self.labels[idx]

        # Get MNIST image
        img, digit = self.mnist[mnist_idx]
        img = np.array(img)  # [28, 28]

        # Apply color (creates RGB)
        img_rgb = self._apply_color(img, color)  # [3, 28, 28]

        # Convert to PIL for rotation
        img_pil = Image.fromarray(img_rgb.transpose(1, 2, 0))  # [H, W, 3]

        # Apply rotation
        img_pil = self._apply_rotation(img_pil, orientation)

        # Resize to target size
        img_pil = TF.resize(img_pil, [self.image_size, self.image_size])

        # Convert to tensor
        img_tensor = TF.to_tensor(img_pil)  # [3, H, W]

        # Normalize (similar to CIFAR)
        img_tensor = TF.normalize(
            img_tensor,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

        # Apply additional transforms if provided
        if self.transform is not None:
            img_tensor = self.transform(img_tensor)

        return img_tensor, label

    def get_metadata(self, idx: int) -> GeometricMNISTSample:
        """Get full metadata for a sample."""
        mnist_idx, color, orientation = self.indices[idx]
        label = self.labels[idx]
        _, digit = self.mnist[mnist_idx]

        return GeometricMNISTSample(
            digit=digit,
            color=color,
            orientation=orientation,
            label=label
        )

    def get_attribute_distribution(self) -> Dict:
        """Get distribution of attributes in the dataset."""
        colors = self.indices[:, 1]
        orientations = self.indices[:, 2]

        return {
            'n_samples': len(self.indices),
            'n_positive': int(np.sum(self.labels == 1)),
            'n_negative': int(np.sum(self.labels == 0)),
            'n_red': int(np.sum(colors == 0)),
            'n_blue': int(np.sum(colors == 1)),
            'n_0deg': int(np.sum(orientations == 0)),
            'n_90deg': int(np.sum(orientations == 1)),
            'n_180deg': int(np.sum(orientations == 2)),
            'n_270deg': int(np.sum(orientations == 3)),
        }


def get_geometric_mnist_transforms(train: bool = True, augment: bool = True):
    """Get transforms for Geometric MNIST."""
    # Note: Color and rotation are part of the task, so we don't augment those
    # Only apply mild spatial augmentation if requested

    if train and augment:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=2),
        ])
    else:
        return None


def get_geometric_mnist_loaders(
    root: str = '/home/st/common_dataset',
    samples_per_class: int = 1000,
    batch_size: int = 128,
    num_workers: int = 4,
    rule: str = 'color_x_orientation',
    seed: int = 42,
    augment: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Get Geometric MNIST data loaders.

    Args:
        root: Path to MNIST data
        samples_per_class: Training samples per binary class
        batch_size: Batch size
        num_workers: Data loading workers
        rule: Compositional rule to use
        seed: Random seed
        augment: Whether to apply data augmentation

    Returns:
        (train_loader, test_loader)
    """
    train_transform = get_geometric_mnist_transforms(train=True, augment=augment)
    test_transform = get_geometric_mnist_transforms(train=False, augment=False)

    train_dataset = GeometricMNISTDataset(
        root=root,
        train=True,
        samples_per_class=samples_per_class,
        rule=rule,
        seed=seed,
        transform=train_transform
    )

    # Test set: use full dataset without sampling
    test_dataset = GeometricMNISTDataset(
        root=root,
        train=False,
        samples_per_class=None,  # Use full test set
        rule=rule,
        seed=seed + 10000,
        transform=test_transform
    )

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


def visualize_samples(
    dataset: GeometricMNISTDataset,
    n_samples: int = 16,
    output_path: Optional[str] = None
):
    """
    Visualize random samples from the dataset.

    Args:
        dataset: Geometric MNIST dataset
        n_samples: Number of samples to visualize
        output_path: If provided, save figure to this path
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    indices = np.random.choice(len(dataset), n_samples, replace=False)

    color_names = ['Red', 'Blue']
    orient_names = ['0°', '90°', '180°', '270°']

    for i, idx in enumerate(indices):
        ax = axes[i // 4, i % 4]

        img, label = dataset[idx]
        meta = dataset.get_metadata(idx)

        # Denormalize for display
        img = img * 0.5 + 0.5
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        ax.imshow(img)
        ax.set_title(
            f"D:{meta.digit} {color_names[meta.color]} {orient_names[meta.orientation]}\n"
            f"Label: {label}",
            fontsize=8
        )
        ax.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    # Test the dataset
    print("Testing Geometric MNIST Dataset...")

    dataset = GeometricMNISTDataset(
        train=True,
        samples_per_class=100,
        seed=42
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Distribution: {dataset.get_attribute_distribution()}")

    # Test loading
    img, label = dataset[0]
    print(f"Sample shape: {img.shape}, label: {label}")

    # Test data loader
    train_loader, test_loader = get_geometric_mnist_loaders(
        samples_per_class=100,
        batch_size=32
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Visualize
    visualize_samples(dataset, output_path='geometric_mnist_samples.png')
    print("Saved sample visualization to geometric_mnist_samples.png")
