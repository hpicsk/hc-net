"""
HC-Net: Hybrid Clifford Network.

Vector collapse problem -> Clifford algebra solution -> Grade hierarchy
(bivectors for rotation, trivectors for chirality) -> MD17 benchmarks.

Unified codebase merging foundational HC-Net (2D) with Chiral-Global HC-Net (3D).
"""

__version__ = '1.0.0'

from . import algebra
from . import layers
from . import models
from . import data
