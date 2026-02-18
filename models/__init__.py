"""
HC-Net Model Architectures.

Provides complete network architectures for:
- N-body physics prediction (2D and 3D)
- Molecular force prediction (MD17)
- Hybrid local MPNN + global Clifford mean-field
- Image classification (HCNetResNet, BaselineResNet)
- EGNN, CGENN, NequIP baselines
"""

__all__ = []

# 2D N-body models
try:
    from .nbody_models import (
        CliffordNBodyNet,
        CliffordNBodyNetNoBivector,
        BaselineNBodyNet,
        BaselineNBodyNetWithAttention,
    )
    __all__ += [
        'CliffordNBodyNet', 'CliffordNBodyNetNoBivector',
        'BaselineNBodyNet', 'BaselineNBodyNetWithAttention',
    ]
except ImportError:
    pass

# 3D N-body models
try:
    from .nbody_models_3d import (
        CliffordNBodyNet3D,
        BaselineNBodyNet3D,
        BaselineNBodyNetWithAttention3D,
    )
    __all__ += [
        'CliffordNBodyNet3D', 'BaselineNBodyNet3D',
        'BaselineNBodyNetWithAttention3D',
    ]
except ImportError:
    pass

# Vector mean-field strawman
try:
    from .vector_meanfield import VectorMeanFieldNet
    __all__ += ['VectorMeanFieldNet']
except ImportError:
    pass

# 3D mean-field classifiers (grade hierarchy)
try:
    from .meanfield_3d import (
        VectorMeanField3DClassifier,
        BivectorMeanField3DClassifier,
        TrivectorMeanField3DClassifier,
        FullCliffordMeanField3DClassifier,
        LearnedMeanField3DClassifier,
    )
    __all__ += [
        'VectorMeanField3DClassifier', 'BivectorMeanField3DClassifier',
        'TrivectorMeanField3DClassifier', 'FullCliffordMeanField3DClassifier',
        'LearnedMeanField3DClassifier',
    ]
except ImportError:
    pass

# Hybrid HC-Net (local MPNN + global mean-field)
try:
    from .hybrid_hcnet import HybridHCNet3D, HybridHCNet3DClassifier
    __all__ += ['HybridHCNet3D', 'HybridHCNet3DClassifier']
except ImportError:
    pass

# MD17 models
try:
    from .md17_models import MD17CliffordNet, MD17BaselineNet, MD17EGNNAdapter
    __all__ += ['MD17CliffordNet', 'MD17BaselineNet', 'MD17EGNNAdapter']
except ImportError:
    pass

try:
    from .md17_hybrid import MD17HybridHCNet
    __all__ += ['MD17HybridHCNet']
except ImportError:
    pass

# EGNN baseline
try:
    from .egnn import EGNNNBodyNet, EGNNNBodyNetSimple, EGNNNBodyNetV2
    __all__ += ['EGNNNBodyNet', 'EGNNNBodyNetSimple', 'EGNNNBodyNetV2']
except ImportError:
    pass

# CGENN baseline
try:
    from .cgenn import CGENNNBodyNet, CGENNNBodyNetSimple
    __all__ += ['CGENNNBodyNet', 'CGENNNBodyNetSimple']
except ImportError:
    pass

# NequIP baseline
try:
    from .nequip_nbody import NequIPNBodyNet, NequIPNBodyNetSimple
    __all__ += ['NequIPNBodyNet', 'NequIPNBodyNetSimple']
except ImportError:
    pass

# Image classification models
try:
    from .hcnet_resnet import HCNetResNet, HCNetResNetSmall
    from .baseline_resnet import BaselineResNet, BaselineResNetSmall
    PCNNResNet = HCNetResNet
    PCNNResNetSmall = HCNetResNetSmall
    __all__ += [
        'HCNetResNet', 'HCNetResNetSmall',
        'BaselineResNet', 'BaselineResNetSmall',
        'PCNNResNet', 'PCNNResNetSmall',
    ]
except ImportError:
    pass
