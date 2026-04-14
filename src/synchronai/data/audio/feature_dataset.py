"""
PyTorch Dataset for pre-extracted audio encoder features.

Loads pre-computed WavLM/Whisper feature tensors from disk instead of running
the encoder at training time. This enables fast CPU-based training of the
classification head without needing the audio encoder loaded.

Output of scripts/extract_audio_features.py:
    feature_dir/
        feature_index.csv      # video_path, second, label, feature_file, ...
        features/              # .pt files, each (n_frames, encoder_dim)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class AudioFeatureDataset(Dataset):
    """Dataset that loads pre-extracted audio encoder features from disk.

    Each sample returns a dict with:
        - "features": (n_frames, encoder_dim) tensor
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
        self.feature_dir = Path(feature_dir) / "features"
        self.entries = entries

        labels = [e["label"] for e in entries]
        unique_labels = sorted(set(labels))
        label_counts = {la: labels.count(la) for la in unique_labels}
        self.class_weights = {
            la: len(labels) / (len(unique_labels) * c)
            for la, c in label_counts.items()
        }

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


def _audio_feature_collate_fn(batch: list[dict]) -> dict:
    """Custom collate that pads variable-length feature sequences.

    Handles both single-layer features (T, D) and per-layer features
    (num_layers, T, D). Pads the temporal dimension to match the longest
    sequence in the batch.
    """
    feat0 = batch[0]["features"]
    is_multilayer = feat0.ndim == 3  # (num_layers, T, D)

    if is_multilayer:
        # Shape: (num_layers, T, D) — pad T dimension
        max_frames = max(b["features"].shape[1] for b in batch)
        padded = []
        for b in batch:
            feat = b["features"]  # (L, T, D)
            if feat.shape[1] < max_frames:
                pad = torch.zeros(feat.shape[0], max_frames - feat.shape[1], feat.shape[2])
                feat = torch.cat([feat, pad], dim=1)
            padded.append(feat)
    else:
        # Shape: (T, D) — pad T dimension
        max_frames = max(b["features"].shape[0] for b in batch)
        padded = []
        for b in batch:
            feat = b["features"]  # (T, D)
            if feat.shape[0] < max_frames:
                pad = torch.zeros(max_frames - feat.shape[0], feat.shape[1])
                feat = torch.cat([feat, pad], dim=0)
            padded.append(feat)

    return {
        "features": torch.stack(padded),
        "label": torch.stack([b["label"] for b in batch]),
        "video_path": [b["video_path"] for b in batch],
        "second": [b["second"] for b in batch],
        "subject_id": [b["subject_id"] for b in batch],
    }


def load_audio_feature_index(feature_dir: Union[str, Path]) -> pd.DataFrame:
    """Load feature index CSV from extraction output directory."""
    feature_dir = Path(feature_dir)
    index_file = feature_dir / "feature_index.csv"
    if not index_file.exists():
        raise FileNotFoundError(
            f"Feature index not found: {index_file}. "
            f"Run scripts/extract_audio_features.py first."
        )
    return pd.read_csv(index_file)


def split_audio_feature_entries(
    entries: list[dict],
    val_split: float = 0.2,
    group_by: str = "subject_id",
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split entries into train/val by group to prevent data leakage."""
    if group_by == "subject_id":
        empty_ids = sum(1 for e in entries if not e.get("subject_id"))
        if empty_ids:
            logger.warning(
                f"{empty_ids}/{len(entries)} entries have empty subject_id. "
                f"Falling back to video_path grouping for those entries."
            )
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


def create_audio_feature_dataloaders(
    feature_dir: Union[str, Path],
    batch_size: int = 64,
    val_split: float = 0.2,
    group_by: str = "subject_id",
    num_workers: int = 4,
    seed: int = 42,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, float, int, int]:
    """Create train and validation dataloaders from pre-extracted audio features.

    Returns:
        Tuple of (train_loader, val_loader, pos_weight, encoder_dim, n_frames,
                  n_layers). n_layers=0 means blended single-layer features.
    """
    feature_dir = Path(feature_dir)

    df = load_audio_feature_index(feature_dir)
    if len(df) == 0:
        raise ValueError(
            f"Feature index is empty: {feature_dir}/feature_index.csv. "
            f"Extraction likely failed — check extraction logs."
        )
    entries = df.to_dict("records")

    encoder_dim = int(df["feature_dim"].iloc[0])
    n_frames = int(df["n_frames"].iloc[0])
    n_layers = int(df["n_layers"].iloc[0]) if "n_layers" in df.columns else 0

    logger.info(
        f"Loaded {len(entries)} audio feature entries. "
        f"Encoder dim: {encoder_dim}, frames/chunk: {n_frames}"
        + (f", layers: {n_layers}" if n_layers > 0 else "")
    )

    train_entries, val_entries = split_audio_feature_entries(
        entries, val_split, group_by, seed
    )

    train_dataset = AudioFeatureDataset(feature_dir, train_entries)
    val_dataset = AudioFeatureDataset(feature_dir, val_entries)

    pos_weight = train_dataset.get_pos_weight()

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
        collate_fn=_audio_feature_collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=_audio_feature_collate_fn,
    )

    return train_loader, val_loader, pos_weight, encoder_dim, n_frames, n_layers
