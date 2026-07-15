"""
Multi-modal dataset combining video and audio data.

Loads synchronized video frames and audio chunks for multi-modal training.
Delegates to existing VideoWindowDataset and AudioClassificationDataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

from synchronai.data.video.dataset import VideoWindowDataset, VideoDatasetConfig, load_video_index
from synchronai.data.audio.dataset import AudioClassificationDataset, AudioDatasetConfig

logger = logging.getLogger(__name__)


@dataclass
class MultiModalDatasetConfig:
    """Configuration for multi-modal dataset."""

    labels_file: Union[str, Path]

    # Video configuration
    sample_fps: float = 12.0
    window_seconds: float = 2.0
    frame_size: int = 640
    video_augment: bool = False
    horizontal_flip_prob: float = 0.5
    color_jitter: bool = False
    temporal_jitter_frames: int = 0
    random_erase_prob: float = 0.3
    gaussian_noise_std: float = 0.02
    mixup_alpha: float = 0.2

    # Audio configuration
    sample_rate: int = 16000
    audio_chunk_duration: float = 1.0
    audio_augment: bool = False
    volume_perturbation: float = 0.2
    additive_noise_std: float = 0.005
    time_shift_max: float = 0.1

    # Dataset options
    verify_alignment: bool = True  # Check video_path and second match
    cache_audio: bool = False


class MultiModalDataset(Dataset):
    """
    Multi-modal dataset combining video and audio.

    Loads temporally-aligned video windows and audio chunks from the same
    (video_path, second) pairs. Delegates to existing video and audio datasets
    to reuse preprocessing and augmentation logic.

    Expected CSV format:
        video_path,second,label,subject_id,session
        /path/to/video.mp4,0,0,50001,V0
        /path/to/video.mp4,1,1,50001,V0
        ...

    Args:
        labels_file: Path to labels CSV with (video_path, second, label, ...)
        video_config: Configuration for video dataset
        audio_config: Configuration for audio dataset
        verify_alignment: If True, verify video and audio data are aligned
    """

    def __init__(
        self,
        labels_file: Union[str, Path],
        video_config: Optional[VideoDatasetConfig] = None,
        audio_config: Optional[AudioDatasetConfig] = None,
        verify_alignment: bool = True
    ):
        self.labels_file = Path(labels_file)
        self.verify_alignment = verify_alignment

        # Create default configs if not provided
        if video_config is None:
            video_config = VideoDatasetConfig(labels_file=labels_file)

        if audio_config is None:
            audio_config = AudioDatasetConfig(labels_file=str(labels_file))

        # Build video specs from labels CSV (filters out timestamps exceeding video duration)
        video_specs = load_video_index(
            labels_file=video_config.labels_file,
            sample_fps=video_config.sample_fps,
            window_seconds=video_config.window_seconds,
            frame_size=video_config.frame_size,
        )

        # Filter audio labels to match valid video specs so both datasets have same rows
        valid_keys = {(s.video_path, s.second) for s in video_specs}
        original_labels = pd.read_csv(audio_config.labels_file)
        filtered_labels = original_labels[
            original_labels.apply(lambda r: (r["video_path"], int(r["second"])) in valid_keys, axis=1)
        ].reset_index(drop=True)

        if len(filtered_labels) < len(original_labels):
            n_dropped = len(original_labels) - len(filtered_labels)
            logger.info(
                f"Filtered audio labels to match video specs: {len(original_labels)} → "
                f"{len(filtered_labels)} ({n_dropped} dropped due to video duration limits)"
            )

        # Write filtered labels for audio dataset
        filtered_audio_csv = Path(audio_config.labels_file).parent / (
            Path(audio_config.labels_file).stem + "_mm_filtered.csv"
        )
        filtered_labels.to_csv(filtered_audio_csv, index=False)

        # Initialize video and audio datasets
        self.video_dataset = VideoWindowDataset(specs=video_specs, config=video_config)
        self.audio_dataset = AudioClassificationDataset(
            labels_file=str(filtered_audio_csv),
            event_classes=audio_config.event_classes,
            sample_rate=audio_config.sample_rate,
            chunk_duration=audio_config.chunk_duration,
            cache_audio=getattr(audio_config, 'cache_audio', False),
            augment=audio_config.augment,
            volume_perturbation=audio_config.volume_perturbation,
            additive_noise_std=audio_config.additive_noise_std,
            time_shift_max=audio_config.time_shift_max
        )

        # Verify datasets have same length
        assert len(self.video_dataset) == len(self.audio_dataset), (
            f"Video dataset ({len(self.video_dataset)} samples) and "
            f"audio dataset ({len(self.audio_dataset)} samples) must have same length"
        )

        # Verify temporal alignment
        if verify_alignment:
            self._verify_alignment()

        logger.info(
            f"Initialized MultiModalDataset with {len(self)} samples from {labels_file.name}"
        )

    def _verify_alignment(self):
        """Verify that video and audio datasets are aligned."""
        logger.info("Verifying temporal alignment between video and audio...")

        # Sample random indices to check alignment
        num_checks = min(100, len(self))
        import random
        check_indices = random.sample(range(len(self)), num_checks)

        misaligned = []
        for idx in check_indices:
            video_sample = self.video_dataset[idx]
            audio_sample = self.audio_dataset[idx]

            # Check if video_path and second match
            # Audio dataset uses 'audio_path' key (same file, different key name)
            video_path = video_sample['video_path']
            audio_path = audio_sample['audio_path']
            if (video_path != audio_path or
                video_sample['second'] != audio_sample['second']):
                misaligned.append({
                    'idx': idx,
                    'video': (video_path, video_sample['second']),
                    'audio': (audio_path, audio_sample['second'])
                })

        if misaligned:
            logger.warning(
                f"Found {len(misaligned)} misaligned samples out of {num_checks} checks"
            )
            logger.warning(f"First misalignment: {misaligned[0]}")
            raise ValueError(
                f"Video and audio datasets are not aligned! "
                f"Found {len(misaligned)}/{num_checks} misaligned samples."
            )

        logger.info("✓ Video and audio datasets are properly aligned")

    def __len__(self) -> int:
        return len(self.video_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a multi-modal sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - video_frames: (n_frames, 3, H, W) tensor
                - audio_chunk: (n_samples,) tensor at 16kHz
                - sync_label: Synchrony label (0 or 1)
                - event_label: Audio event label (0-6 for 7 classes)
                - video_path: Path to video file
                - second: Second timestamp
                - subject_id: Subject ID
                - session: Session ID
        """
        # Get video sample
        video_sample = self.video_dataset[idx]

        # Get audio sample
        audio_sample = self.audio_dataset[idx]

        # Verify alignment (if enabled)
        # Audio dataset uses 'audio_path' key (same file, different key name)
        if self.verify_alignment:
            assert video_sample['video_path'] == audio_sample['audio_path'], (
                f"Path mismatch at index {idx}: "
                f"{video_sample['video_path']} != {audio_sample['audio_path']}"
            )
            assert video_sample['second'] == audio_sample['second'], (
                f"Second mismatch at index {idx}: "
                f"{video_sample['second']} != {audio_sample['second']}"
            )

        # Build multi-modal sample
        return {
            'video_frames': video_sample['frames'],
            'audio_chunk': audio_sample['audio'],
            'sync_label': video_sample['label'],
            'event_label': audio_sample['label'],
            'video_path': video_sample['video_path'],
            'second': video_sample['second'],
            'subject_id': video_sample.get('subject_id'),
            'session': video_sample.get('session')
        }

    @property
    def num_sync_classes(self) -> int:
        """Number of synchrony classes (binary: 0 or 1)."""
        return 2

    @property
    def num_event_classes(self) -> int:
        """Number of audio event classes."""
        return len(self.audio_dataset.event_classes)

    @property
    def event_classes(self) -> list[str]:
        """List of audio event class names."""
        return self.audio_dataset.event_classes

    def get_class_weights_sync(self) -> float:
        """Compute pos_weight for binary synchrony labels (handle imbalance).

        Returns the weight for the positive class (label=1), computed as
        neg_count / pos_count, for use with BCEWithLogitsLoss(pos_weight=...).
        """
        weights = self.video_dataset.class_weights  # {label: weight}
        # pos_weight = weight of positive class (label 1)
        return weights.get(1, 1.0)

    def get_class_weights_events(self) -> torch.Tensor:
        """Compute class weights for audio event labels (handle imbalance)."""
        # Delegate to audio dataset
        return self.audio_dataset.get_class_weights()

    def set_augment(self, video_augment: bool, audio_augment: bool):
        """Enable or disable augmentation for video and audio."""
        self.video_dataset.augment = video_augment
        self.audio_dataset.augment = audio_augment


def create_multimodal_splits(
    labels_file: Union[str, Path],
    video_config: VideoDatasetConfig,
    audio_config: AudioDatasetConfig,
    val_split: float = 0.2,
    group_by: str = "subject_id",
    seed: int = 42
) -> tuple[MultiModalDataset, MultiModalDataset]:
    """
    Create train and validation splits for multi-modal dataset.

    Groups by subject_id or video_path to prevent data leakage.

    Args:
        labels_file: Path to labels CSV
        video_config: Video dataset configuration
        audio_config: Audio dataset configuration
        val_split: Fraction of data for validation
        group_by: Column to group by ('subject_id' or 'video_path')
        seed: Random seed for reproducibility

    Returns:
        (train_dataset, val_dataset) tuple
    """
    import random as _random

    # Load full labels
    df = pd.read_csv(labels_file)

    # Split into train/val by group (prevents data leakage)
    if group_by == "subject_id" and "subject_id" in df.columns:
        groups = df["subject_id"].unique().tolist()
    else:
        groups = df["video_path"].unique().tolist()

    rng = _random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * val_split))
    val_groups = set(groups[:n_val])

    col = "subject_id" if (group_by == "subject_id" and "subject_id" in df.columns) else "video_path"
    val_mask = df[col].isin(val_groups)
    train_df = df[~val_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)

    logger.info(
        f"Split {len(df)} samples into {len(train_df)} train, {len(val_df)} val "
        f"({len(groups) - n_val} train groups, {n_val} val groups, grouped by {col})"
    )

    # Save temporary train/val CSV files
    labels_path = Path(labels_file)
    train_csv = labels_path.parent / f"{labels_path.stem}_train.csv"
    val_csv = labels_path.parent / f"{labels_path.stem}_val.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    logger.info(f"Train split: {len(train_df)} samples → {train_csv}")
    logger.info(f"Val split: {len(val_df)} samples → {val_csv}")

    # Create independent config copies to avoid mutating the caller's objects
    from dataclasses import replace as _replace

    # Create train dataset (with augmentation)
    train_video_config = _replace(video_config, labels_file=train_csv, augment=True)
    train_audio_config = _replace(audio_config, labels_file=str(train_csv), augment=True)

    train_dataset = MultiModalDataset(
        labels_file=train_csv,
        video_config=train_video_config,
        audio_config=train_audio_config,
        verify_alignment=True
    )

    # Create val dataset (no augmentation)
    val_video_config = _replace(video_config, labels_file=val_csv, augment=False)
    val_audio_config = _replace(audio_config, labels_file=str(val_csv), augment=False)

    val_dataset = MultiModalDataset(
        labels_file=val_csv,
        video_config=val_video_config,
        audio_config=val_audio_config,
        verify_alignment=True
    )

    return train_dataset, val_dataset
