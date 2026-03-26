"""
PyTorch Dataset for pre-extracted DINOv2 features.

Loads pre-computed feature tensors from disk instead of reading video frames.
This enables fast CPU-based training of the temporal (LSTM) and classification
head without needing the DINOv2 backbone loaded.

Output of scripts/extract_dinov2_features.py:
    feature_dir/
        feature_index.csv      # video_path, second, label, feature_file, ...
        features/              # .pt files, each (n_frames, feature_dim)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class FeatureWindowDataset(Dataset):
    """Dataset that loads pre-extracted DINOv2 features from disk.

    Each sample returns a dict with:
        - "features": (n_frames, feature_dim) tensor
        - "label": scalar float tensor
        - "video_path": str
        - "second": int
        - "subject_id": str
    """

    def __init__(
        self,
        feature_dir: Union[str, Path],
        entries: list[dict],
    ):
        """Initialize dataset.

        Args:
            feature_dir: Root directory containing features/ subdirectory
            entries: List of dicts from feature_index.csv rows
        """
        self.feature_dir = Path(feature_dir) / "features"
        self.entries = entries

        # Compute class weights for balanced sampling
        labels = [e["label"] for e in entries]
        unique_labels = sorted(set(labels))
        label_counts = {l: labels.count(l) for l in unique_labels}
        self.class_weights = {
            l: len(labels) / (len(unique_labels) * c) for l, c in label_counts.items()
        }
        logger.debug(f"Class weights: {self.class_weights}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[idx]

        feature_path = self.feature_dir / entry["feature_file"]
        features = torch.load(feature_path, map_location="cpu", weights_only=True)

        label = torch.tensor(entry["label"], dtype=torch.float32)

        return {
            "features": features,
            "label": label,
            "video_path": entry["video_path"],
            "second": entry["second"],
            "subject_id": entry.get("subject_id", ""),
        }

    def get_pos_weight(self, clamp_range: tuple[float, float] = (0.1, 10.0)) -> float:
        """Compute pos_weight for BCEWithLogitsLoss."""
        labels = [e["label"] for e in self.entries]
        num_positive = sum(labels)
        num_negative = len(labels) - num_positive
        if num_positive == 0:
            return 1.0
        pos_weight = num_negative / num_positive
        return max(clamp_range[0], min(clamp_range[1], pos_weight))


def load_feature_index(feature_dir: Union[str, Path]) -> pd.DataFrame:
    """Load feature index CSV from extraction output directory.

    Args:
        feature_dir: Directory containing feature_index.csv

    Returns:
        DataFrame with columns: feature_file, video_path, second, label,
        subject_id, session, feature_dim, n_frames
    """
    feature_dir = Path(feature_dir)
    index_file = feature_dir / "feature_index.csv"
    if not index_file.exists():
        raise FileNotFoundError(
            f"Feature index not found: {index_file}. "
            f"Run scripts/extract_dinov2_features.py first."
        )
    return pd.read_csv(index_file)


def split_feature_entries(
    entries: list[dict],
    val_split: float = 0.2,
    group_by: str = "subject_id",
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split entries into train/val by group to prevent data leakage.

    Groups by subject_id (preferred) or video_path to ensure all windows
    from the same subject/video are in the same split.

    Args:
        entries: List of dicts with video_path, subject_id, etc.
        val_split: Fraction for validation set
        group_by: Column to group by ("subject_id" or "video_path")
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_entries, val_entries)
    """
    if group_by == "subject_id":
        groups = list(set(
            e.get("subject_id", e["video_path"]) or e["video_path"]
            for e in entries
        ))
    else:
        groups = list(set(e["video_path"] for e in entries))

    rng = random.Random(seed)
    rng.shuffle(groups)

    n_val = max(1, int(len(groups) * val_split))
    val_groups = set(groups[:n_val])

    train_entries = []
    val_entries = []
    for entry in entries:
        if group_by == "subject_id":
            group = entry.get("subject_id", entry["video_path"]) or entry["video_path"]
        else:
            group = entry["video_path"]
        if group in val_groups:
            val_entries.append(entry)
        else:
            train_entries.append(entry)

    logger.info(
        f"Split {len(entries)} entries into {len(train_entries)} train, "
        f"{len(val_entries)} val ({len(groups) - n_val} train groups, "
        f"{n_val} val groups)"
    )
    return train_entries, val_entries


def create_feature_dataloaders(
    feature_dir: Union[str, Path],
    batch_size: int = 64,
    val_split: float = 0.2,
    group_by: str = "subject_id",
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, float, int, int]:
    """Create train and validation dataloaders from pre-extracted features.

    Args:
        feature_dir: Directory with feature_index.csv and features/
        batch_size: Batch size (can be larger than video training since
            features are small)
        val_split: Validation split fraction
        group_by: Column to group by for splitting
        num_workers: DataLoader workers
        seed: Random seed

    Returns:
        Tuple of (train_loader, val_loader, pos_weight, feature_dim, n_frames)
    """
    feature_dir = Path(feature_dir)

    # Load index
    df = load_feature_index(feature_dir)
    entries = df.to_dict("records")

    # Get metadata from first entry
    feature_dim = int(df["feature_dim"].iloc[0])
    n_frames = int(df["n_frames"].iloc[0])

    logger.info(
        f"Loaded {len(entries)} feature entries. "
        f"Feature dim: {feature_dim}, frames/window: {n_frames}"
    )

    # Split by group
    train_entries, val_entries = split_feature_entries(
        entries, val_split, group_by, seed
    )

    # Create datasets
    train_dataset = FeatureWindowDataset(feature_dir, train_entries)
    val_dataset = FeatureWindowDataset(feature_dir, val_entries)

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
        generator=generator,
        persistent_workers=num_workers > 0,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader, pos_weight, feature_dim, n_frames
