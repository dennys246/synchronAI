"""Audio training modules."""

from synchronai.training.audio.train import (
    AudioTrainingConfig,
    TrainingHistory,
    train_audio_classifier,
)

__all__ = [
    "AudioTrainingConfig",
    "TrainingHistory",
    "train_audio_classifier",
]
