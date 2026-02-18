"""
PyTorch Dataset for video classification with per-second labels.

Supports:
- Lazy loading of video frames
- Group-based train/val splitting to prevent leakage
- Configurable augmentation
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from synchronai.data.video.processing import (
    VideoReaderPool,
    load_video_info,
    read_window_frames,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoWindowSpec:
    """Specification for a single video window."""

    video_path: str
    second: int
    label: int
    video_fps: float
    sample_fps: float
    window_seconds: float
    frame_size: int
    subject_id: Optional[str] = None
    session: Optional[str] = None

    @property
    def n_frames(self) -> int:
        """Number of frames = sample_fps * window_seconds."""
        return int(self.sample_fps * self.window_seconds)


@dataclass
class VideoDatasetConfig:
    """Configuration for video dataset."""

    labels_file: Union[str, Path]
    sample_fps: float = 12.0
    window_seconds: float = 2.0
    frame_size: int = 640
    augment: bool = False
    horizontal_flip_prob: float = 0.5
    color_jitter: bool = False
    temporal_jitter_frames: int = 0
    random_erase_prob: float = 0.3
    random_erase_scale: tuple[float, float] = (0.02, 0.2)
    gaussian_noise_std: float = 0.02
    mixup_alpha: float = 0.2
    reader_pool_size: int = 8
    video_backend: str = "auto"


def load_video_index(
    labels_file: Union[str, Path],
    sample_fps: float = 12.0,
    window_seconds: float = 1.0,
    frame_size: int = 640,
) -> list[VideoWindowSpec]:
    """Load video index from labels CSV file.

    Args:
        labels_file: Path to labels.csv
        sample_fps: Target frames per second
        window_seconds: Window duration in seconds
        frame_size: Target frame size

    Returns:
        List of VideoWindowSpec for each labeled window
    """
    labels_file = Path(labels_file)
    if not labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    df = pd.read_csv(labels_file)

    required_cols = ["video_path", "second", "label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Labels file missing required columns: {missing}")

    # Cache video info (FPS and duration) for efficiency
    video_info_cache: dict[str, tuple[float, float]] = {}  # (fps, duration)

    specs = []
    skipped_count = 0
    for _, row in df.iterrows():
        video_path = row["video_path"]

        # Get video info
        if video_path not in video_info_cache:
            try:
                info = load_video_info(video_path)
                video_info_cache[video_path] = (info.fps, info.duration)
            except Exception as e:
                logger.warning(f"Failed to get video info for {video_path}: {e}")
                video_info_cache[video_path] = (sample_fps, float("inf"))  # Fallback

        video_fps, video_duration = video_info_cache[video_path]
        second = int(row["second"])

        # Skip labels that exceed video duration (with 1-second buffer for window)
        if second + window_seconds > video_duration:
            skipped_count += 1
            continue

        spec = VideoWindowSpec(
            video_path=video_path,
            second=second,
            label=int(row["label"]),
            video_fps=video_fps,
            sample_fps=sample_fps,
            window_seconds=window_seconds,
            frame_size=frame_size,
            subject_id=row.get("subject_id"),
            session=row.get("session"),
        )
        specs.append(spec)

    if skipped_count > 0:
        logger.warning(
            f"Skipped {skipped_count} labels with timestamps exceeding video duration"
        )
    logger.info(f"Loaded {len(specs)} windows from {df['video_path'].nunique()} videos")
    return specs


def split_by_video(
    specs: list[VideoWindowSpec],
    val_split: float = 0.2,
    group_by: str = "video_path",
    seed: int = 42,
) -> tuple[list[VideoWindowSpec], list[VideoWindowSpec]]:
    """Split specs into train/val sets by group to prevent leakage.

    Args:
        specs: List of VideoWindowSpec
        val_split: Fraction for validation set
        group_by: Column to group by ("video_path" or "subject_id")
        seed: Random seed

    Returns:
        Tuple of (train_specs, val_specs)
    """
    # Get unique groups
    if group_by == "subject_id":
        # Fall back to video_path for specs with no subject_id to prevent
        # them from always landing in training set (data leakage)
        groups = list(set(
            s.subject_id if s.subject_id else s.video_path
            for s in specs
        ))
    else:
        groups = list(set(s.video_path for s in specs))

    # Shuffle groups
    rng = random.Random(seed)
    rng.shuffle(groups)

    # Split groups
    n_val = max(1, int(len(groups) * val_split))
    val_groups = set(groups[:n_val])

    # Assign specs to train/val
    train_specs = []
    val_specs = []

    for spec in specs:
        if group_by == "subject_id":
            group = spec.subject_id if spec.subject_id else spec.video_path
        else:
            group = spec.video_path
        if group in val_groups:
            val_specs.append(spec)
        else:
            train_specs.append(spec)

    logger.info(
        f"Split {len(specs)} windows into {len(train_specs)} train, {len(val_specs)} val "
        f"({len(groups) - n_val} train groups, {n_val} val groups)"
    )

    return train_specs, val_specs


def save_split_info(
    train_specs: list[VideoWindowSpec],
    val_specs: list[VideoWindowSpec],
    save_path: Union[str, Path],
) -> None:
    """Save split information for reproducibility.

    Args:
        train_specs: Training specs
        val_specs: Validation specs
        save_path: Path to save split info
    """
    import json

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    info = {
        "train_videos": sorted(set(s.video_path for s in train_specs)),
        "val_videos": sorted(set(s.video_path for s in val_specs)),
        "train_subjects": sorted(set(s.subject_id for s in train_specs if s.subject_id)),
        "val_subjects": sorted(set(s.subject_id for s in val_specs if s.subject_id)),
        "train_windows": len(train_specs),
        "val_windows": len(val_specs),
    }

    with open(save_path, "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Saved split info to {save_path}")


class VideoWindowDataset(Dataset):
    """PyTorch Dataset for video windows with per-second labels."""

    def __init__(
        self,
        specs: list[VideoWindowSpec],
        config: VideoDatasetConfig,
        augment: Optional[bool] = None,
    ):
        """Initialize dataset.

        Args:
            specs: List of VideoWindowSpec
            config: Dataset configuration
            augment: Override augmentation setting (uses config.augment if None)
        """
        self.specs = specs
        self.config = config
        self.augment = augment if augment is not None else config.augment

        # Create reader pool for efficient video access
        self.reader_pool = VideoReaderPool(
            max_readers=config.reader_pool_size,
            backend=config.video_backend,
        )

        # Compute class weights for balanced sampling
        labels = [s.label for s in specs]
        unique_labels = sorted(set(labels))
        label_counts = {l: labels.count(l) for l in unique_labels}

        self.class_weights = {
            l: len(labels) / (len(unique_labels) * c) for l, c in label_counts.items()
        }
        logger.debug(f"Class weights: {self.class_weights}")

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        spec = self.specs[idx]

        # Temporal jitter: shift start within the second
        temporal_offset = 0.0
        if self.augment and self.config.temporal_jitter_frames > 0:
            max_offset = self.config.temporal_jitter_frames / spec.sample_fps
            temporal_offset = random.uniform(-max_offset, max_offset)
            # Clamp to valid range: can't go before time 0 or past video end
            min_offset = -spec.second  # don't go before start of video
            temporal_offset = max(min_offset, temporal_offset)

        # Get reader from pool
        reader = self.reader_pool.get_reader(spec.video_path)

        # Read frames (apply temporal jitter as offset to start time)
        frames = read_window_frames(
            video_path=spec.video_path,
            second=spec.second + temporal_offset,
            sample_fps=spec.sample_fps,
            window_seconds=spec.window_seconds,
            target_size=spec.frame_size,
            reader=reader,
        )

        # Apply augmentation
        if self.augment:
            frames = self._apply_augmentation(frames)

        # Convert to tensor
        frames_tensor = torch.from_numpy(frames)  # (n_frames, C, H, W)
        label_tensor = torch.tensor(spec.label, dtype=torch.float32)

        return {
            "frames": frames_tensor,
            "label": label_tensor,
            "video_path": spec.video_path,
            "second": spec.second,
            "subject_id": spec.subject_id or "",
        }

    def _apply_augmentation(self, frames: np.ndarray) -> np.ndarray:
        """Apply augmentation consistently across all frames in window.

        Args:
            frames: Input frames (n_frames, C, H, W)

        Returns:
            Augmented frames
        """
        # Horizontal flip
        if self.config.horizontal_flip_prob > 0:
            if random.random() < self.config.horizontal_flip_prob:
                frames = frames[:, :, :, ::-1].copy()

        # Color jitter (simple brightness/contrast adjustment)
        if self.config.color_jitter:
            # Random brightness adjustment
            brightness = random.uniform(0.8, 1.2)
            frames = frames * brightness
            frames = np.clip(frames, 0, 1)

            # Random contrast adjustment
            contrast = random.uniform(0.8, 1.2)
            mean = frames.mean(axis=(2, 3), keepdims=True)
            frames = (frames - mean) * contrast + mean
            frames = np.clip(frames, 0, 1)

        # Gaussian noise (same spatial noise applied to all frames for temporal consistency)
        if self.config.gaussian_noise_std > 0:
            single_frame_noise = np.random.normal(
                0, self.config.gaussian_noise_std, frames.shape[1:]  # (C, H, W)
            )
            frames = frames + single_frame_noise[np.newaxis]
            frames = np.clip(frames, 0, 1)

        # Random erasing (applied to same region across all frames for consistency)
        if self.config.random_erase_prob > 0:
            if random.random() < self.config.random_erase_prob:
                _, C, H, W = frames.shape
                scale = random.uniform(*self.config.random_erase_scale)
                area = H * W * scale
                aspect = random.uniform(0.3, 3.3)
                h = int(min(H, (area * aspect) ** 0.5))
                w = int(min(W, (area / aspect) ** 0.5))
                top = random.randint(0, max(0, H - h))
                left = random.randint(0, max(0, W - w))
                # Fill with random values (same region in all frames)
                frames[:, :, top:top + h, left:left + w] = np.random.uniform(0, 1, (1, C, h, w))

        return frames.astype(np.float32)

    def get_pos_weight(self, clamp_range: tuple[float, float] = (0.1, 10.0)) -> float:
        """Compute pos_weight for BCEWithLogitsLoss.

        Args:
            clamp_range: Range to clamp the weight

        Returns:
            pos_weight value
        """
        labels = [s.label for s in self.specs]
        num_positive = sum(labels)
        num_negative = len(labels) - num_positive

        if num_positive == 0:
            return 1.0

        pos_weight = num_negative / num_positive
        return max(clamp_range[0], min(clamp_range[1], pos_weight))

    def close(self) -> None:
        """Release resources."""
        self.reader_pool.close_all()


def create_dataloaders(
    labels_file: Union[str, Path],
    config: VideoDatasetConfig,
    batch_size: int = 16,
    val_split: float = 0.2,
    group_by: str = "video_path",
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, float]:
    """Create train and validation dataloaders.

    Args:
        labels_file: Path to labels.csv
        config: Dataset configuration
        batch_size: Batch size
        val_split: Validation split fraction
        group_by: Column to group by for splitting
        num_workers: Number of dataloader workers
        seed: Random seed

    Returns:
        Tuple of (train_loader, val_loader, pos_weight)
    """
    from synchronai.utils.reproducibility import worker_init_fn

    # Load index
    specs = load_video_index(
        labels_file,
        sample_fps=config.sample_fps,
        window_seconds=config.window_seconds,
        frame_size=config.frame_size,
    )

    # Split
    train_specs, val_specs = split_by_video(specs, val_split, group_by, seed)

    # Create datasets
    train_config = VideoDatasetConfig(
        labels_file=config.labels_file,
        sample_fps=config.sample_fps,
        window_seconds=config.window_seconds,
        frame_size=config.frame_size,
        augment=True,
        horizontal_flip_prob=config.horizontal_flip_prob,
        color_jitter=config.color_jitter,
        temporal_jitter_frames=config.temporal_jitter_frames,
        random_erase_prob=config.random_erase_prob,
        random_erase_scale=config.random_erase_scale,
        gaussian_noise_std=config.gaussian_noise_std,
        mixup_alpha=config.mixup_alpha,
        reader_pool_size=config.reader_pool_size,
        video_backend=config.video_backend,
    )

    val_config = VideoDatasetConfig(
        labels_file=config.labels_file,
        sample_fps=config.sample_fps,
        window_seconds=config.window_seconds,
        frame_size=config.frame_size,
        augment=False,
        reader_pool_size=config.reader_pool_size,
        video_backend=config.video_backend,
    )

    train_dataset = VideoWindowDataset(train_specs, train_config, augment=True)
    val_dataset = VideoWindowDataset(val_specs, val_config, augment=False)

    # Get pos_weight from training set
    pos_weight = train_dataset.get_pos_weight()

    # Create dataloaders
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader, pos_weight
