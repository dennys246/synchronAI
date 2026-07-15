"""Multi-modal models for combining video, audio, and fNIRS modalities."""

from .fusion_modules import (
    ConcatFusion,
    CrossModalAttention,
    GatedFusion,
    create_fusion_module
)
from .fusion_model import MultiModalSynchronyModel

__all__ = [
    'ConcatFusion',
    'CrossModalAttention',
    'GatedFusion',
    'create_fusion_module',
    'MultiModalSynchronyModel'
]
