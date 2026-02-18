"""
MD17 Molecular Dynamics Dataset.

MD17 is a benchmark for molecular dynamics force field learning.
Contains molecular dynamics trajectories for small organic molecules.

Reference:
"Machine learning of accurate energy-conserving molecular force fields"
(Chmiela et al., Science Advances 2017)

Available molecules:
- aspirin, benzene, ethanol, malonaldehyde, naphthalene,
- salicylic acid, toluene, uracil

Task: Predict atomic forces from atomic positions
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from typing import Tuple, Optional, Dict, List
import urllib.request
import hashlib


# MD17 dataset URLs and metadata
# Primary: PyTorch Geometric-compatible naming (md17_ prefix) at quantum-machine.org
# Fallback: legacy _dft naming, sgdml.org mirror
# URL pattern from: https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/md17.py
_GDML_BASE = 'http://quantum-machine.org/gdml/data/npz'

MD17_MOLECULES = {
    'aspirin': {
        'urls': [
            f'{_GDML_BASE}/md17_aspirin.npz',
            f'{_GDML_BASE}/aspirin_dft.npz',
            'https://sgdml.org/gdml/data/npz/aspirin_dft.npz',
        ],
        'n_atoms': 21,
        'formula': 'C9H8O4'
    },
    'benzene': {
        'urls': [
            f'{_GDML_BASE}/md17_benzene2017.npz',
            f'{_GDML_BASE}/benzene2017_dft.npz',
            'https://sgdml.org/gdml/data/npz/benzene2017_dft.npz',
        ],
        'n_atoms': 12,
        'formula': 'C6H6'
    },
    'ethanol': {
        'urls': [
            f'{_GDML_BASE}/md17_ethanol.npz',
            f'{_GDML_BASE}/ethanol_dft.npz',
            'https://sgdml.org/gdml/data/npz/ethanol_dft.npz',
        ],
        'n_atoms': 9,
        'formula': 'C2H6O'
    },
    'malonaldehyde': {
        'urls': [
            f'{_GDML_BASE}/md17_malonaldehyde.npz',
            f'{_GDML_BASE}/malonaldehyde_dft.npz',
            'https://sgdml.org/gdml/data/npz/malonaldehyde_dft.npz',
        ],
        'n_atoms': 9,
        'formula': 'C3H4O2'
    },
    'naphthalene': {
        'urls': [
            f'{_GDML_BASE}/md17_naphthalene.npz',
            f'{_GDML_BASE}/naphthalene_dft.npz',
            'https://sgdml.org/gdml/data/npz/naphthalene_dft.npz',
        ],
        'n_atoms': 18,
        'formula': 'C10H8'
    },
    'salicylic': {
        'urls': [
            f'{_GDML_BASE}/md17_salicylic.npz',
            f'{_GDML_BASE}/salicylic_dft.npz',
            'https://sgdml.org/gdml/data/npz/salicylic_dft.npz',
        ],
        'n_atoms': 16,
        'formula': 'C7H6O3'
    },
    'toluene': {
        'urls': [
            f'{_GDML_BASE}/md17_toluene.npz',
            f'{_GDML_BASE}/toluene_dft.npz',
            'https://sgdml.org/gdml/data/npz/toluene_dft.npz',
        ],
        'n_atoms': 15,
        'formula': 'C7H8'
    },
    'uracil': {
        'urls': [
            f'{_GDML_BASE}/md17_uracil.npz',
            f'{_GDML_BASE}/uracil_dft.npz',
            'https://sgdml.org/gdml/data/npz/uracil_dft.npz',
        ],
        'n_atoms': 12,
        'formula': 'C4H4N2O2'
    }
}


def download_md17(molecule: str, data_dir: str = './data/md17') -> str:
    """
    Download MD17 dataset for a specific molecule.

    Tries multiple URLs with fallback for robustness.

    Args:
        molecule: Name of the molecule (e.g., 'aspirin', 'benzene')
        data_dir: Directory to store downloaded data

    Returns:
        Path to the downloaded .npz file
    """
    if molecule not in MD17_MOLECULES:
        raise ValueError(f"Unknown molecule: {molecule}. "
                        f"Available: {list(MD17_MOLECULES.keys())}")

    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.join(data_dir, f'{molecule}_dft.npz')

    if os.path.exists(filename):
        return filename

    # Support both 'url' (legacy) and 'urls' (new) format
    info = MD17_MOLECULES[molecule]
    urls = info.get('urls', [info['url']] if 'url' in info else [])

    errors = []
    for url in urls:
        try:
            print(f"Downloading MD17 {molecule} from {url}...")
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded to {filename}")
            return filename
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            errors.append(f"  {url}: {e}")
            # Clean up partial download
            if os.path.exists(filename):
                os.remove(filename)
            continue

    error_msg = f"All download URLs failed for '{molecule}':\n" + "\n".join(errors)
    raise RuntimeError(error_msg)


class MD17Dataset(Dataset):
    """
    MD17 Molecular Dynamics Dataset.

    Task: Predict atomic forces from atomic positions.

    Features:
    - Positions: [N_atoms, 3] atomic coordinates in Angstrom
    - Atomic numbers: [N_atoms] element types

    Targets:
    - Forces: [N_atoms, 3] atomic forces in kcal/mol/Angstrom
    - Energy: scalar potential energy in kcal/mol
    """

    def __init__(
        self,
        molecule: str = 'ethanol',
        split: str = 'train',
        n_train: int = 1000,
        n_val: int = 500,
        n_test: int = 1000,
        data_dir: str = './data/md17',
        seed: int = 42,
        predict_forces: bool = True,
        predict_both: bool = False,
        normalize: bool = True
    ):
        """
        Args:
            molecule: Name of the molecule
            split: 'train', 'val', or 'test'
            n_train: Number of training samples
            n_val: Number of validation samples
            n_test: Number of test samples
            data_dir: Directory for data
            seed: Random seed for splitting
            predict_forces: If True, predict forces. If False, predict energy.
            predict_both: If True, return both forces and energy (4-tuple).
            normalize: Whether to normalize inputs/outputs
        """
        self.molecule = molecule
        self.split = split
        self.predict_forces = predict_forces
        self.predict_both = predict_both
        self.normalize = normalize

        # Download/load data
        npz_path = download_md17(molecule, data_dir)
        data = np.load(npz_path)

        # Extract data
        # Positions: [N_samples, N_atoms, 3]
        self.all_positions = data['R'].astype(np.float32)
        # Forces: [N_samples, N_atoms, 3]
        self.all_forces = data['F'].astype(np.float32)
        # Energy: [N_samples]
        self.all_energy = data['E'].astype(np.float32).flatten()
        # Atomic numbers: [N_atoms]
        self.atomic_numbers = data['z'].astype(np.int64)

        n_total = len(self.all_positions)
        self.n_atoms = self.all_positions.shape[1]

        # Create splits
        np.random.seed(seed)
        indices = np.random.permutation(n_total)

        train_end = min(n_train, n_total)
        val_end = min(n_train + n_val, n_total)
        test_end = min(n_train + n_val + n_test, n_total)

        if split == 'train':
            self.indices = indices[:train_end]
        elif split == 'val':
            self.indices = indices[train_end:val_end]
        elif split == 'test':
            self.indices = indices[val_end:test_end]
        else:
            raise ValueError(f"Unknown split: {split}")

        # Compute normalization statistics from training data
        train_indices = indices[:train_end]
        if normalize:
            # Position normalization: center and scale
            train_pos = self.all_positions[train_indices]
            self.pos_mean = train_pos.mean(axis=(0, 1))
            self.pos_std = train_pos.std() + 1e-8

            # Force normalization
            train_forces = self.all_forces[train_indices]
            self.force_mean = train_forces.mean()
            self.force_std = train_forces.std() + 1e-8

            # Energy normalization
            train_energy = self.all_energy[train_indices]
            self.energy_mean = train_energy.mean()
            self.energy_std = train_energy.std() + 1e-8
        else:
            self.pos_mean = 0
            self.pos_std = 1
            self.force_mean = 0
            self.force_std = 1
            self.energy_mean = 0
            self.energy_std = 1

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Returns:
            If predict_both=False (default):
                positions: [N_atoms, 3] normalized atomic positions
                atomic_numbers: [N_atoms] element types
                target: [N_atoms, 3] forces or [1] energy depending on task
            If predict_both=True:
                positions, atomic_numbers, forces [N_atoms, 3], energy [1]
        """
        real_idx = self.indices[idx]

        # Get data
        pos = self.all_positions[real_idx].copy()
        forces = self.all_forces[real_idx].copy()
        energy = self.all_energy[real_idx]

        # Normalize
        if self.normalize:
            pos = (pos - self.pos_mean) / self.pos_std
            forces = (forces - self.force_mean) / self.force_std
            energy = (energy - self.energy_mean) / self.energy_std

        # Convert to tensors
        pos_tensor = torch.tensor(pos, dtype=torch.float32)
        z_tensor = torch.tensor(self.atomic_numbers, dtype=torch.long)

        if self.predict_both:
            forces_tensor = torch.tensor(forces, dtype=torch.float32)
            energy_tensor = torch.tensor([energy], dtype=torch.float32)
            return pos_tensor, z_tensor, forces_tensor, energy_tensor

        if self.predict_forces:
            target = torch.tensor(forces, dtype=torch.float32)
        else:
            target = torch.tensor([energy], dtype=torch.float32)

        return pos_tensor, z_tensor, target

    def get_molecule_info(self) -> Dict:
        """Get information about the molecule."""
        return {
            'name': self.molecule,
            'n_atoms': self.n_atoms,
            'formula': MD17_MOLECULES[self.molecule]['formula'],
            'atomic_numbers': self.atomic_numbers.tolist(),
            'total_samples': len(self.all_positions),
            'split_samples': len(self.indices)
        }


class MD17RotatedDataset(Dataset):
    """
    MD17 dataset with random 3D rotations applied.

    Used for testing rotation equivariance/invariance.
    """

    def __init__(
        self,
        base_dataset: MD17Dataset,
        rotation_angles: Optional[List[Tuple[float, float, float]]] = None,
        seed: int = 42
    ):
        """
        Args:
            base_dataset: Base MD17 dataset
            rotation_angles: List of Euler angles (degrees) or None for random
            seed: Random seed
        """
        self.base_dataset = base_dataset
        self.rotation_angles = rotation_angles
        self.rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def _random_rotation_matrix(self) -> np.ndarray:
        """Generate a random 3D rotation matrix."""
        # Random quaternion -> rotation matrix
        q = self.rng.randn(4)
        q = q / np.linalg.norm(q)
        w, x, y, z = q

        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ], dtype=np.float32)

        return R

    def _euler_rotation_matrix(self, angles: Tuple[float, float, float]) -> np.ndarray:
        """Generate rotation matrix from Euler angles (ZYX convention, degrees)."""
        alpha, beta, gamma = np.radians(angles)

        Rz = np.array([
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(beta), 0, np.sin(beta)],
            [0, 1, 0],
            [-np.sin(beta), 0, np.cos(beta)]
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(gamma), -np.sin(gamma)],
            [0, np.sin(gamma), np.cos(gamma)]
        ])

        return (Rz @ Ry @ Rx).astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        pos, z, target = self.base_dataset[idx]

        # Generate rotation
        if self.rotation_angles is not None:
            angles = self.rotation_angles[idx % len(self.rotation_angles)]
            R = self._euler_rotation_matrix(angles)
        else:
            R = self._random_rotation_matrix()

        R = torch.tensor(R, dtype=torch.float32)

        # Apply rotation to positions
        pos_rotated = pos @ R.T

        # Apply rotation to forces (if predicting forces)
        if target.dim() == 2:  # Forces [N_atoms, 3]
            target_rotated = target @ R.T
        else:
            target_rotated = target  # Energy is scalar invariant

        return pos_rotated, z, target_rotated


def get_md17_loaders(
    molecule: str = 'ethanol',
    n_train: int = 1000,
    n_val: int = 500,
    n_test: int = 1000,
    batch_size: int = 32,
    num_workers: int = 4,
    predict_forces: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get MD17 train, validation, and test data loaders.

    Args:
        molecule: Name of the molecule
        n_train: Training samples
        n_val: Validation samples
        n_test: Test samples
        batch_size: Batch size
        num_workers: Data loading workers
        predict_forces: Predict forces (True) or energy (False)
        seed: Random seed

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset = MD17Dataset(
        molecule=molecule,
        split='train',
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        predict_forces=predict_forces,
        seed=seed
    )

    val_dataset = MD17Dataset(
        molecule=molecule,
        split='val',
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        predict_forces=predict_forces,
        seed=seed
    )

    test_dataset = MD17Dataset(
        molecule=molecule,
        split='test',
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        predict_forces=predict_forces,
        seed=seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
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

    return train_loader, val_loader, test_loader


def get_md17_loaders_both(
    molecule: str = 'ethanol',
    n_train: int = 1000,
    n_val: int = 500,
    n_test: int = 1000,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Get MD17 loaders that return both forces and energy per sample.

    Each batch yields (positions, atomic_numbers, forces, energy).

    Returns:
        (train_loader, val_loader, test_loader)
    """
    loaders = []
    for split in ['train', 'val', 'test']:
        ds = MD17Dataset(
            molecule=molecule,
            split=split,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            predict_both=True,
            seed=seed,
        )
        loaders.append(DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
        ))
    return tuple(loaders)


def get_md17_loaders_with_ood(
    molecule: str = 'ethanol',
    n_train: int = 1000,
    n_test: int = 1000,
    rotation_angles: List[Tuple[float, float, float]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    predict_forces: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict[Tuple, DataLoader]]:
    """
    Get MD17 loaders including OOD rotated test sets.

    Args:
        molecule: Name of the molecule
        n_train: Training samples
        n_test: Test samples
        rotation_angles: List of Euler angles for OOD sets
        batch_size: Batch size
        num_workers: Data loading workers
        predict_forces: Predict forces or energy
        seed: Random seed

    Returns:
        (train_loader, test_loader, ood_loaders_dict)
    """
    if rotation_angles is None:
        rotation_angles = [
            (0, 0, 0),
            (45, 0, 0),
            (0, 45, 0),
            (0, 0, 45),
            (90, 0, 0),
            (45, 45, 0),
        ]

    train_loader, _, test_loader = get_md17_loaders(
        molecule=molecule,
        n_train=n_train,
        n_val=100,  # Small val for this use case
        n_test=n_test,
        batch_size=batch_size,
        num_workers=num_workers,
        predict_forces=predict_forces,
        seed=seed
    )

    # Create OOD loaders
    ood_loaders = {}
    base_test_dataset = MD17Dataset(
        molecule=molecule,
        split='test',
        n_train=n_train,
        n_val=100,
        n_test=n_test,
        predict_forces=predict_forces,
        seed=seed
    )

    for angles in rotation_angles:
        if angles == (0, 0, 0):
            ood_loaders[angles] = test_loader
        else:
            rotated_dataset = MD17RotatedDataset(
                base_dataset=base_test_dataset,
                rotation_angles=[angles],
                seed=seed
            )
            ood_loaders[angles] = DataLoader(
                rotated_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

    return train_loader, test_loader, ood_loaders


if __name__ == '__main__':
    print("Testing MD17 Dataset...")

    # Test with ethanol (smallest molecule)
    print("\n1. Loading ethanol dataset...")
    train_dataset = MD17Dataset(
        molecule='ethanol',
        split='train',
        n_train=100,
        predict_forces=True
    )

    print(f"   Molecule info: {train_dataset.get_molecule_info()}")
    print(f"   Dataset size: {len(train_dataset)}")

    # Get a sample
    pos, z, forces = train_dataset[0]
    print(f"   Position shape: {pos.shape}")
    print(f"   Atomic numbers: {z}")
    print(f"   Forces shape: {forces.shape}")

    # Test data loaders
    print("\n2. Testing data loaders...")
    train_loader, val_loader, test_loader = get_md17_loaders(
        molecule='ethanol',
        n_train=100,
        n_val=50,
        n_test=50,
        batch_size=16
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Test a batch
    for batch_pos, batch_z, batch_forces in train_loader:
        print(f"   Batch positions: {batch_pos.shape}")
        print(f"   Batch atomic nums: {batch_z.shape}")
        print(f"   Batch forces: {batch_forces.shape}")
        break

    # Test OOD loaders
    print("\n3. Testing OOD loaders...")
    train_loader, test_loader, ood_loaders = get_md17_loaders_with_ood(
        molecule='ethanol',
        n_train=100,
        n_test=50,
        batch_size=16
    )
    print(f"   OOD rotations: {list(ood_loaders.keys())}")

    # Test rotation equivariance
    print("\n4. Testing rotation equivariance...")
    base_dataset = MD17Dataset(
        molecule='ethanol',
        split='test',
        n_train=100,
        n_test=50,
        normalize=False  # Don't normalize for equivariance test
    )
    rotated_dataset = MD17RotatedDataset(
        base_dataset=base_dataset,
        rotation_angles=[(90, 0, 0)]
    )

    pos1, z1, forces1 = base_dataset[0]
    pos2, z2, forces2 = rotated_dataset[0]

    # For equivariant model: R(f(x)) should equal f(R(x))
    # Here we just verify the rotation was applied
    print(f"   Original position norm: {pos1.norm():.4f}")
    print(f"   Rotated position norm: {pos2.norm():.4f}")
    print(f"   (Should be equal for rotation)")

    print("\nSUCCESS: MD17 dataset working!")
