"""
PyTorch Dataset for audio classification training.

Loads audio files with per-second labels for training the audio classifier.
"""

from __future__ import annotations

import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from synchronai.data.audio.processing import (
    extract_audio,
    get_audio_duration,
    load_audio_chunk,
)

logger = logging.getLogger(__name__)

# Default audio event classes (must match model)
AUDIO_EVENT_CLASSES = [
    "speech",
    "laughter",
    "crying",
    "babbling",
    "silence",
    "noise",
    "music",
]


@dataclass
class AudioDatasetConfig:
    """Configuration for audio dataset."""

    labels_file: str
    sample_rate: int = 16000
    chunk_duration: float = 1.0
    event_classes: list[str] = None
    augment: bool = False
    volume_perturbation: float = 0.2  # Random volume change (fraction)
    additive_noise_std: float = 0.005  # Gaussian noise std
    time_shift_max: float = 0.1  # Max random time shift in seconds

    def __post_init__(self):
        if self.event_classes is None:
            self.event_classes = AUDIO_EVENT_CLASSES.copy()


class AudioClassificationDataset(Dataset):
    """Dataset for audio classification training.

    Expected CSV format:
        video_path,second,label,subject_id,session
        /path/to/video.mp4,0,0,50001,V0
        /path/to/video.mp4,60,1,50001,V0
        ...

    Each row represents a 1-second audio chunk with its label (0 or 1).
    """

    def __init__(
        self,
        labels_file: Union[str, Path],
        event_classes: Optional[list[str]] = None,
        sample_rate: int = 16000,
        chunk_duration: float = 1.0,
        cache_audio: bool = False,
        augment: bool = False,
        volume_perturbation: float = 0.2,
        additive_noise_std: float = 0.005,
        time_shift_max: float = 0.1,
    ):
        self.labels_file = Path(labels_file)
        self.event_classes = event_classes or AUDIO_EVENT_CLASSES.copy()
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.cache_audio = cache_audio
        self.augment = augment
        self.volume_perturbation = volume_perturbation
        self.additive_noise_std = additive_noise_std
        self.time_shift_max = time_shift_max

        # Build class to index mapping
        self.class_to_idx = {cls: i for i, cls in enumerate(self.event_classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}

        # Load labels
        self.samples = self._load_labels()
        logger.info(
            f"Loaded {len(self.samples)} samples from {self.labels_file.name}"
        )

        # Audio cache (optional)
        self._audio_cache: dict[str, np.ndarray] = {}

        # Extracted audio paths (video -> wav)
        self._extracted_audio: dict[str, Path] = {}

    def _load_labels(self) -> list[dict]:
        """Load labels from CSV file.

        Expected CSV format:
            video_path,second,label,subject_id,session
        """
        samples = []

        with open(self.labels_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_path = row["video_path"]
                second = int(row["second"])
                label = int(row["label"])

                samples.append({
                    "audio_path": video_path,
                    "second": second,
                    "event_idx": label,
                    "subject_id": row.get("subject_id", ""),
                })

        return samples

    def _get_audio_path(self, original_path: str) -> Path:
        """Get audio path, extracting from video if needed."""
        original_path = Path(original_path)

        # Check if already extracted
        if str(original_path) in self._extracted_audio:
            return self._extracted_audio[str(original_path)]

        # Check if it's a video file
        if original_path.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            # Extract audio to temp location
            extracted = extract_audio(original_path)
            self._extracted_audio[str(original_path)] = extracted
            return extracted

        return original_path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        audio_path = self._get_audio_path(sample["audio_path"])

        # Apply time shift augmentation (jitter the start time)
        start_sec = float(sample["second"])
        if self.augment and self.time_shift_max > 0:
            shift = random.uniform(-self.time_shift_max, self.time_shift_max)
            start_sec = max(0.0, start_sec + shift)

        # Load audio chunk
        audio = load_audio_chunk(
            audio_path,
            start_sec=start_sec,
            duration=self.chunk_duration,
            sample_rate=self.sample_rate,
        )

        # Apply augmentation
        if self.augment:
            audio = self._apply_augmentation(audio)

        return {
            "audio": torch.from_numpy(audio).float(),
            "label": torch.tensor(sample["event_idx"], dtype=torch.long),
            "audio_path": sample["audio_path"],
            "second": sample["second"],
        }

    def _apply_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio augmentations.

        Args:
            audio: Audio samples (n_samples,)

        Returns:
            Augmented audio samples
        """
        # Volume perturbation: randomly scale amplitude
        if self.volume_perturbation > 0:
            gain = 1.0 + random.uniform(-self.volume_perturbation, self.volume_perturbation)
            audio = audio * gain

        # Additive Gaussian noise
        if self.additive_noise_std > 0:
            noise = np.random.normal(0, self.additive_noise_std, audio.shape)
            audio = audio + noise

        # Clip to valid range
        audio = np.clip(audio, -1.0, 1.0)

        return audio.astype(np.float32)

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced datasets."""
        class_counts = [0] * len(self.event_classes)
        for sample in self.samples:
            class_counts[sample["event_idx"]] += 1

        total = sum(class_counts)
        weights = [total / (len(self.event_classes) * c) if c > 0 else 0.0 for c in class_counts]
        return torch.tensor(weights, dtype=torch.float32)

    def get_class_distribution(self) -> dict[int, int]:
        """Get distribution of classes in dataset."""
        counts = {}
        for sample in self.samples:
            label = sample["event_idx"]
            counts[label] = counts.get(label, 0) + 1
        return counts


def create_audio_dataloaders(
    labels_file: Union[str, Path],
    batch_size: int = 16,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
    event_classes: Optional[list[str]] = None,
    group_by: str = "subject_id",
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, AudioClassificationDataset]:
    """Create train and validation dataloaders with group-based splitting.

    Splits by subject_id or audio_path to prevent data leakage — no samples
    from the same subject/video appear in both train and val sets.

    Args:
        labels_file: Path to labels CSV
        batch_size: Batch size
        val_split: Fraction for validation
        num_workers: Number of data loading workers
        seed: Random seed for splitting
        event_classes: List of event classes
        group_by: Column to group by ("subject_id" or "audio_path")

    Returns:
        Tuple of (train_loader, val_loader, full_dataset)
    """
    from torch.utils.data import DataLoader, Subset

    # Create full dataset (no augmentation — used for class weights/distribution)
    dataset = AudioClassificationDataset(
        labels_file=labels_file,
        event_classes=event_classes,
    )

    # Group-based split to prevent data leakage
    if group_by == "subject_id":
        groups = list(set(
            s.get("subject_id", s["audio_path"]) for s in dataset.samples
        ))
    else:
        groups = list(set(s["audio_path"] for s in dataset.samples))

    rng = random.Random(seed)
    rng.shuffle(groups)

    n_val_groups = max(1, int(len(groups) * val_split))
    val_groups = set(groups[:n_val_groups])

    train_indices = []
    val_indices = []
    for i, sample in enumerate(dataset.samples):
        group = sample.get("subject_id", sample["audio_path"]) if group_by == "subject_id" else sample["audio_path"]
        if group in val_groups:
            val_indices.append(i)
        else:
            train_indices.append(i)

    # Create separate train dataset WITH augmentation
    train_dataset_full = AudioClassificationDataset(
        labels_file=labels_file,
        event_classes=event_classes,
        augment=True,
    )
    train_dataset = Subset(train_dataset_full, train_indices)

    # Val dataset WITHOUT augmentation (use the original)
    val_dataset = Subset(dataset, val_indices)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"Created dataloaders: {len(train_indices)} train (augmented), {len(val_indices)} val samples "
        f"({len(groups) - n_val_groups} train groups, {n_val_groups} val groups, grouped by {group_by})"
    )
    return train_loader, val_loader, dataset
