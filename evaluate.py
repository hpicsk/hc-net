"""
Evaluation utilities for HC-Net physics/molecular models.

Provides functions for:
- Model evaluation metrics (MSE, MAE, equivariance error)
- Rotation equivariance testing
- Feature analysis
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple


def evaluate_regression(
    model: nn.Module,
    test_loader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """Evaluate regression model (N-body prediction)."""
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs, targets = batch[0], batch[1]

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            mse = ((outputs - targets) ** 2).mean().item()
            mae = (outputs - targets).abs().mean().item()

            total_mse += mse
            total_mae += mae
            n_batches += 1

    return {
        'mse': total_mse / n_batches,
        'mae': total_mae / n_batches,
    }


def evaluate_classification(
    model: nn.Module,
    test_loader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """Evaluate classification model (chirality detection)."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return {
        'accuracy': correct / total,
        'n_samples': total,
    }


def test_rotation_equivariance(
    model: nn.Module,
    input_tensor: torch.Tensor,
    coord_dim: int = 3,
    device: str = 'cuda',
    n_rotations: int = 10,
) -> Dict[str, float]:
    """
    Test rotation equivariance of a model.

    Generates random rotations, applies them to input, and measures
    whether model(R@x) == R@model(x).
    """
    model.eval()
    model = model.to(device)
    x = input_tensor.to(device)

    errors = []

    with torch.no_grad():
        y = model(x)

        for _ in range(n_rotations):
            # Random rotation matrix
            if coord_dim == 2:
                theta = torch.rand(1).item() * 2 * np.pi
                R = torch.tensor([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)],
                ], dtype=x.dtype, device=device)
            else:
                # Random 3D rotation via QR decomposition
                M = torch.randn(3, 3, device=device)
                Q, _ = torch.linalg.qr(M)
                if torch.det(Q) < 0:
                    Q[:, 0] *= -1
                R = Q

            # Rotate input
            B, N, D = x.shape
            half = D // 2
            pos = x[..., :half] @ R.T
            vel = x[..., half:] @ R.T
            x_rot = torch.cat([pos, vel], dim=-1)

            # model(R@x)
            y_rot = model(x_rot)

            # R@model(x)
            y_pos = y[..., :half] @ R.T
            y_vel = y[..., half:] @ R.T
            Ry = torch.cat([y_pos, y_vel], dim=-1)

            # Equivariance error
            error = (y_rot - Ry).norm() / (Ry.norm() + 1e-8)
            errors.append(error.item())

    return {
        'mean_equivariance_error': float(np.mean(errors)),
        'std_equivariance_error': float(np.std(errors)),
        'max_equivariance_error': float(np.max(errors)),
    }
