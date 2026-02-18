"""
Relational MNIST Dataset.

A harder compositional binding task requiring 3-way interaction:
Color × Shape × Spatial Relation.

Task: Classify images containing 2 objects based on whether they satisfy
a specific relational rule, e.g., "Red Triangle LEFT-OF Blue Circle".

This is much harder than the 2-way Geometric MNIST because:
1. Multiple objects must be identified
2. Color-shape binding for each object
3. Spatial relationship (coordinate-based) between objects

CNNs struggle with coordinate-based logic (x1 < x2) from pixels without
massive data. Clifford algebra encodes relative position naturally.

Usage:
    from pcnn.data.relational_mnist import get_relational_mnist_loaders
    train_loader, test_loader = get_relational_mnist_loaders(samples_per_class=100)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
from typing import Tuple, Optional, List, NamedTuple
from dataclasses import dataclass
from enum import IntEnum


class Shape(IntEnum):
    TRIANGLE = 0
    CIRCLE = 1
    SQUARE = 2


class Color(IntEnum):
    RED = 0
    BLUE = 1
    GREEN = 2


class Position(IntEnum):
    LEFT = 0
    CENTER = 1
    RIGHT = 2
    TOP_LEFT = 3
    TOP_RIGHT = 4
    BOTTOM_LEFT = 5
    BOTTOM_RIGHT = 6
    TOP_CENTER = 7
    BOTTOM_CENTER = 8


@dataclass
class ObjectInfo:
    shape: Shape
    color: Color
    position: Position
    x: int  # Actual x coordinate
    y: int  # Actual y coordinate
    size: int


@dataclass
class ImageMetadata:
    objects: List[ObjectInfo]
    label: int
    rule_satisfied: bool


# Color RGB values
COLOR_MAP = {
    Color.RED: (255, 80, 80),
    Color.BLUE: (80, 80, 255),
    Color.GREEN: (80, 255, 80),
}


class RelationalMNISTDataset(Dataset):
    """
    Dataset for relational reasoning over multiple objects.

    The task requires binding (Color, Shape, Position) for 2 objects
    and determining if they satisfy a spatial relationship rule.

    Default Rule: "Red Triangle LEFT-OF Blue Circle" → Positive
    """

    def __init__(
        self,
        n_samples: int = 1000,
        image_size: int = 64,
        object_size: int = 16,
        rule: str = 'red_triangle_left_of_blue_circle',
        seed: int = 42,
        train: bool = True,
        n_distractors: int = 0
    ):
        """
        Args:
            n_samples: Total number of samples (balanced between classes)
            image_size: Output image size
            object_size: Size of each object
            rule: The relational rule to evaluate
            seed: Random seed
            train: Whether this is training set
            n_distractors: Number of distractor objects to add (makes task harder)
        """
        self.n_samples = n_samples
        self.image_size = image_size
        self.object_size = object_size
        self.rule = rule
        self.seed = seed
        self.train = train
        self.n_distractors = n_distractors

        np.random.seed(seed if train else seed + 10000)

        self.images = []
        self.labels = []
        self.metadata = []

        self._generate_dataset()

    def _generate_dataset(self):
        """Generate balanced dataset with positive and negative examples."""
        n_positive = self.n_samples // 2
        n_negative = self.n_samples - n_positive

        # Generate positive examples (rule satisfied)
        for _ in range(n_positive):
            img, meta = self._generate_positive_example()
            self.images.append(img)
            self.labels.append(1)
            self.metadata.append(meta)

        # Generate negative examples (rule not satisfied)
        for _ in range(n_negative):
            img, meta = self._generate_negative_example()
            self.images.append(img)
            self.labels.append(0)
            self.metadata.append(meta)

        # Shuffle
        indices = np.random.permutation(len(self.images))
        self.images = [self.images[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.metadata = [self.metadata[i] for i in indices]

    def _generate_distractors(self, excluded_positions: List[Position]) -> List[ObjectInfo]:
        """Generate distractor objects that don't interfere with target objects."""
        distractors = []

        # Available positions for distractors (exclude target positions)
        distractor_positions = [
            Position.TOP_LEFT, Position.TOP_RIGHT, Position.BOTTOM_LEFT,
            Position.BOTTOM_RIGHT, Position.TOP_CENTER, Position.BOTTOM_CENTER,
            Position.CENTER
        ]
        available_positions = [p for p in distractor_positions if p not in excluded_positions]

        # Distractor colors (use GREEN to avoid confusion with target colors RED/BLUE)
        # Also allow muted versions of Red/Blue occasionally to make it harder
        distractor_colors = [Color.GREEN, Color.RED, Color.BLUE]
        shape_list = [Shape.TRIANGLE, Shape.CIRCLE, Shape.SQUARE]

        for i in range(min(self.n_distractors, len(available_positions))):
            pos = available_positions[i]
            # Random shape and color for distractor
            shape_idx = np.random.randint(len(shape_list))
            color_idx = np.random.randint(len(distractor_colors))
            shape = shape_list[shape_idx]
            color = distractor_colors[color_idx]
            distractor = self._create_object(shape, color, pos)
            distractors.append(distractor)

        return distractors

    def _generate_positive_example(self) -> Tuple[np.ndarray, ImageMetadata]:
        """Generate an image satisfying the rule."""
        if self.rule == 'red_triangle_left_of_blue_circle':
            # Red Triangle on left, Blue Circle on right
            obj1 = self._create_object(Shape.TRIANGLE, Color.RED, Position.LEFT)
            obj2 = self._create_object(Shape.CIRCLE, Color.BLUE, Position.RIGHT)
        else:
            raise ValueError(f'Unknown rule: {self.rule}')

        objects = [obj1, obj2]

        # Add distractors if requested
        if self.n_distractors > 0:
            distractors = self._generate_distractors([Position.LEFT, Position.RIGHT])
            objects.extend(distractors)

        img = self._render_image(objects)
        meta = ImageMetadata(objects=objects, label=1, rule_satisfied=True)

        return img, meta

    def _generate_negative_example(self) -> Tuple[np.ndarray, ImageMetadata]:
        """Generate an image NOT satisfying the rule."""
        # Multiple ways to violate the rule:
        violation_type = np.random.randint(5)

        if violation_type == 0:
            # Wrong color for triangle (Blue Triangle instead of Red)
            obj1 = self._create_object(Shape.TRIANGLE, Color.BLUE, Position.LEFT)
            obj2 = self._create_object(Shape.CIRCLE, Color.BLUE, Position.RIGHT)
        elif violation_type == 1:
            # Wrong shape (Red Square instead of Red Triangle)
            obj1 = self._create_object(Shape.SQUARE, Color.RED, Position.LEFT)
            obj2 = self._create_object(Shape.CIRCLE, Color.BLUE, Position.RIGHT)
        elif violation_type == 2:
            # Wrong position (Red Triangle on RIGHT, not left)
            obj1 = self._create_object(Shape.TRIANGLE, Color.RED, Position.RIGHT)
            obj2 = self._create_object(Shape.CIRCLE, Color.BLUE, Position.LEFT)
        elif violation_type == 3:
            # Wrong color for circle (Red Circle instead of Blue)
            obj1 = self._create_object(Shape.TRIANGLE, Color.RED, Position.LEFT)
            obj2 = self._create_object(Shape.CIRCLE, Color.RED, Position.RIGHT)
        else:
            # Completely different objects
            shape1 = np.random.choice([Shape.CIRCLE, Shape.SQUARE])
            color1 = np.random.choice(list(Color))
            shape2 = np.random.choice([Shape.TRIANGLE, Shape.SQUARE])
            color2 = np.random.choice(list(Color))
            obj1 = self._create_object(shape1, color1, Position.LEFT)
            obj2 = self._create_object(shape2, color2, Position.RIGHT)

        objects = [obj1, obj2]

        # Add distractors if requested
        if self.n_distractors > 0:
            distractors = self._generate_distractors([Position.LEFT, Position.RIGHT])
            objects.extend(distractors)

        img = self._render_image(objects)
        meta = ImageMetadata(objects=objects, label=0, rule_satisfied=False)

        return img, meta

    def _create_object(self, shape: Shape, color: Color, position: Position) -> ObjectInfo:
        """Create an object with slight position jitter."""
        # Base positions for all positions
        pos_coords = {
            Position.LEFT: (self.image_size // 4, self.image_size // 2),
            Position.CENTER: (self.image_size // 2, self.image_size // 2),
            Position.RIGHT: (3 * self.image_size // 4, self.image_size // 2),
            Position.TOP_LEFT: (self.image_size // 4, self.image_size // 4),
            Position.TOP_RIGHT: (3 * self.image_size // 4, self.image_size // 4),
            Position.BOTTOM_LEFT: (self.image_size // 4, 3 * self.image_size // 4),
            Position.BOTTOM_RIGHT: (3 * self.image_size // 4, 3 * self.image_size // 4),
            Position.TOP_CENTER: (self.image_size // 2, self.image_size // 4),
            Position.BOTTOM_CENTER: (self.image_size // 2, 3 * self.image_size // 4),
        }

        base_x, base_y = pos_coords[position]

        # Add jitter
        jitter = self.object_size // 4
        x = base_x + np.random.randint(-jitter, jitter + 1)
        y = base_y + np.random.randint(-jitter, jitter + 1)

        return ObjectInfo(
            shape=shape,
            color=color,
            position=position,
            x=x,
            y=y,
            size=self.object_size
        )

    def _render_image(self, objects: List[ObjectInfo]) -> np.ndarray:
        """Render objects to an image."""
        # Gray background
        img = Image.new('RGB', (self.image_size, self.image_size), (128, 128, 128))
        draw = ImageDraw.Draw(img)

        for obj in objects:
            color = COLOR_MAP[obj.color]
            half = obj.size // 2

            if obj.shape == Shape.TRIANGLE:
                # Upward pointing triangle
                points = [
                    (obj.x, obj.y - half),  # Top
                    (obj.x - half, obj.y + half),  # Bottom left
                    (obj.x + half, obj.y + half),  # Bottom right
                ]
                draw.polygon(points, fill=color)

            elif obj.shape == Shape.CIRCLE:
                bbox = [
                    obj.x - half, obj.y - half,
                    obj.x + half, obj.y + half
                ]
                draw.ellipse(bbox, fill=color)

            elif obj.shape == Shape.SQUARE:
                bbox = [
                    obj.x - half, obj.y - half,
                    obj.x + half, obj.y + half
                ]
                draw.rectangle(bbox, fill=color)

        return np.array(img)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img = self.images[idx]

        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        return torch.tensor(img, dtype=torch.float32), self.labels[idx]

    def get_metadata(self, idx: int) -> ImageMetadata:
        """Get metadata for a sample."""
        return self.metadata[idx]


def get_relational_mnist_loaders(
    samples_per_class: int = 500,
    batch_size: int = 64,
    image_size: int = 64,
    rule: str = 'red_triangle_left_of_blue_circle',
    seed: int = 42,
    num_workers: int = 0,
    n_distractors: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Get training and test DataLoaders for Relational MNIST.

    Args:
        samples_per_class: Samples per class for training
        batch_size: Batch size
        image_size: Image size
        rule: The relational rule
        seed: Random seed
        num_workers: DataLoader workers
        n_distractors: Number of distractor objects (makes task harder)

    Returns:
        train_loader, test_loader
    """
    n_train = samples_per_class * 2  # Binary classification
    n_test = max(1000, samples_per_class)  # Reasonable test size

    train_dataset = RelationalMNISTDataset(
        n_samples=n_train,
        image_size=image_size,
        rule=rule,
        seed=seed,
        train=True,
        n_distractors=n_distractors
    )

    test_dataset = RelationalMNISTDataset(
        n_samples=n_test,
        image_size=image_size,
        rule=rule,
        seed=seed,
        train=False,
        n_distractors=n_distractors
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, n_train),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


if __name__ == '__main__':
    # Test dataset generation
    print('Generating Relational MNIST dataset...')

    train_loader, test_loader = get_relational_mnist_loaders(
        samples_per_class=100,
        batch_size=16
    )

    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')

    # Check a batch
    for images, labels in train_loader:
        print(f'Batch shape: {images.shape}')
        print(f'Labels: {labels}')
        print(f'Positive ratio: {labels.float().mean():.2f}')
        break

    # Save a sample image
    dataset = train_loader.dataset
    for i in range(min(5, len(dataset))):
        img, label = dataset[i]
        meta = dataset.get_metadata(i)
        print(f'Sample {i}: label={label}, rule_satisfied={meta.rule_satisfied}')
        print(f'  Objects: {[(o.shape.name, o.color.name, o.position.name) for o in meta.objects]}')
