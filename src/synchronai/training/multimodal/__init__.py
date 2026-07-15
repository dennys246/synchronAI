"""Multi-modal training utilities."""

from .train import (
    MultiModalTrainingConfig,
    TrainingHistory,
    train_multimodal_classifier
)

__all__ = [
    'MultiModalTrainingConfig',
    'TrainingHistory',
    'train_multimodal_classifier'
]
