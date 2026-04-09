"""
PyTorch Dataset for pre-extracted fNIRS encoder features.

Loads pre-computed U-Net bottleneck or multiscale features from disk,
following the same pattern as the audio and video feature datasets.

Output of scripts/extract_fnirs_features.py:
    feature_dir/
        feature_index.csv      # fnirs_path, subject_id, participant_type, ...
        features/              # .pt files
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


class FnirsFeatureDataset(Dataset):
    """Dataset that loads pre-extracted fNIRS encoder features.

    Each sample returns a dict with:
        - "features": (n_frames, feature_dim) or (feature_dim,) tensor
        - "label": scalar float tensor (0=child or 1=adult, or sync label)
        - "fnirs_path": str
        - "subject_id": str
    """

    def __init__(
        self,
        feature_dir: Union[str, Path],
        entries: list[dict],
        label_column: str = "participant_type",
        label_map: dict[str, int] | None = None,
    ):
        self.feature_dir = Path(feature_dir) / "features"
        self.entries = entries
        self.label_column = label_column
        self.label_map = label_map or {"child": 0, "adult": 1}

        labels = [self.label_map.get(e.get(label_column, ""), -1) for e in entries]
        valid_labels = [la for la in labels if la >= 0]
        if valid_labels:
            unique = sorted(set(valid_labels))
            counts = {la: valid_labels.count(la) for la in unique}
            self.class_weights = {
                la: len(valid_labels) / (len(unique) * c)
                for la, c in counts.items()
            }
            logger.info(f"Label distribution: {counts}")
        else:
            self.class_weights = {}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[idx]

        feature_path = self.feature_dir / entry["feature_file"]
        features = torch.load(feature_path, map_location="cpu", weights_only=True)

        label_str = entry.get(self.label_column, "unknown")
        label_val = self.label_map.get(label_str, -1)
        label = torch.tensor(float(label_val), dtype=torch.float32)

        return {
            "features": features,
            "label": label,
            "fnirs_path": entry.get("fnirs_path", ""),
            "subject_id": entry.get("subject_id", ""),
        }

    def get_pos_weight(self, clamp_range: tuple[float, float] = (0.1, 10.0)) -> float:
        """Compute pos_weight for BCEWithLogitsLoss."""
        labels = [
            self.label_map.get(e.get(self.label_column, ""), -1)
            for e in self.entries
        ]
        valid = [la for la in labels if la >= 0]
        num_positive = sum(valid)
        num_negative = len(valid) - num_positive
        if num_positive == 0:
            return 1.0
        pos_weight = num_negative / num_positive
        return max(clamp_range[0], min(clamp_range[1], pos_weight))


def _fnirs_feature_collate_fn(batch: list[dict]) -> dict:
    """Collate with padding for variable-length temporal features."""
    feat0 = batch[0]["features"]

    if feat0.ndim == 2:
        # (T, D) — pad temporal dimension
        max_frames = max(b["features"].shape[0] for b in batch)
        padded = []
        for b in batch:
            feat = b["features"]
            if feat.shape[0] < max_frames:
                pad = torch.zeros(max_frames - feat.shape[0], feat.shape[1])
                feat = torch.cat([feat, pad], dim=0)
            padded.append(feat)
    elif feat0.ndim == 1:
        # (D,) — already pooled, no padding needed
        padded = [b["features"] for b in batch]
    else:
        padded = [b["features"] for b in batch]

    return {
        "features": torch.stack(padded),
        "label": torch.stack([b["label"] for b in batch]),
        "fnirs_path": [b["fnirs_path"] for b in batch],
        "subject_id": [b["subject_id"] for b in batch],
    }


def load_fnirs_feature_index(feature_dir: Union[str, Path]) -> pd.DataFrame:
    """Load feature index CSV."""
    feature_dir = Path(feature_dir)
    index_file = feature_dir / "feature_index.csv"
    if not index_file.exists():
        raise FileNotFoundError(
            f"Feature index not found: {index_file}. "
            f"Run scripts/extract_fnirs_features.py first."
        )
    return pd.read_csv(index_file)


def split_fnirs_feature_entries(
    entries: list[dict],
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split entries into train/val by subject_id to prevent leakage."""
    groups = list(set(
        e.get("subject_id", e.get("fnirs_path", str(i)))
        for i, e in enumerate(entries)
    ))

    rng = random.Random(seed)
    rng.shuffle(groups)

    n_val = max(1, int(len(groups) * val_split))
    val_groups = set(groups[:n_val])

    train_entries = []
    val_entries = []
    for entry in entries:
        group = entry.get("subject_id", entry.get("fnirs_path", ""))
        if group in val_groups:
            val_entries.append(entry)
        else:
            train_entries.append(entry)

    logger.info(
        f"Split {len(entries)} entries into {len(train_entries)} train, "
        f"{len(val_entries)} val ({len(groups) - n_val} train subjects, "
        f"{n_val} val subjects)"
    )
    return train_entries, val_entries


def filter_by_quality_tier(
    entries: list[dict],
    include_tiers: list[str] | None = None,
) -> list[dict]:
    """Filter feature entries by quality tier.

    Args:
        entries: Feature index rows (dicts with optional 'quality_tier' key).
        include_tiers: List of tiers to keep (e.g. ["gold"], ["salvageable"]).
            If None, all entries are kept (no filtering).

    Returns:
        Filtered list of entries.
    """
    if include_tiers is None:
        return entries

    # Warn if quality_tier column appears to be missing
    if entries and "quality_tier" not in entries[0]:
        logger.warning(
            "quality_tier column not found in entries — was --enable-qc used "
            "during feature extraction? All entries will have tier='unknown'."
        )

    before = len(entries)
    filtered = [e for e in entries if e.get("quality_tier", "unknown") in include_tiers]
    logger.info(
        "Quality tier filter (keep=%s): %d → %d entries",
        include_tiers, before, len(filtered),
    )
    # Log tier distribution of kept entries
    tier_counts: dict[str, int] = {}
    for e in filtered:
        t = e.get("quality_tier", "unknown")
        tier_counts[t] = tier_counts.get(t, 0) + 1
    for t, c in sorted(tier_counts.items()):
        logger.info("  %s: %d entries", t, c)
    return filtered


def create_fnirs_feature_dataloaders(
    feature_dir: Union[str, Path],
    batch_size: int = 32,
    val_split: float = 0.2,
    label_column: str = "participant_type",
    label_map: dict[str, int] | None = None,
    num_workers: int = 4,
    seed: int = 42,
    include_tiers: list[str] | None = None,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, float, int]:
    """Create train/val dataloaders from pre-extracted fNIRS features.

    Args:
        include_tiers: If set, only include entries whose quality_tier is in
            this list.  E.g. ["gold"] for low-motion holdout, ["salvageable"]
            for high-motion holdout.

    Returns:
        (train_loader, val_loader, pos_weight, feature_dim)
    """
    feature_dir = Path(feature_dir)
    df = load_fnirs_feature_index(feature_dir)

    if len(df) == 0:
        raise ValueError(f"Feature index is empty: {feature_dir}/feature_index.csv")

    # Filter to entries with valid labels
    if label_column in df.columns:
        valid_mask = df[label_column].isin(
            (label_map or {"child": 0, "adult": 1}).keys()
        )
        if valid_mask.sum() < len(df):
            logger.warning(
                f"{len(df) - valid_mask.sum()}/{len(df)} entries have unknown "
                f"{label_column}, excluding from training"
            )
        df = df[valid_mask]

    if len(df) == 0:
        raise ValueError(
            f"No entries with valid labels! All {label_column} values were "
            f"outside the label map. Check detect_participant_type() output."
        )

    entries = df.to_dict("records")

    # Apply quality tier filtering before splitting
    entries = filter_by_quality_tier(entries, include_tiers)
    if not entries:
        raise ValueError(
            f"No entries remaining after quality tier filter "
            f"(include_tiers={include_tiers}). Check feature_index.csv."
        )

    feature_dim = int(entries[0]["feature_dim"])
    logger.info(f"Loaded {len(entries)} fNIRS feature entries, dim={feature_dim}")

    train_entries, val_entries = split_fnirs_feature_entries(entries, val_split, seed)

    train_dataset = FnirsFeatureDataset(
        feature_dir, train_entries, label_column, label_map
    )
    val_dataset = FnirsFeatureDataset(
        feature_dir, val_entries, label_column, label_map
    )

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
        collate_fn=_fnirs_feature_collate_fn,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        collate_fn=_fnirs_feature_collate_fn,
    )

    return train_loader, val_loader, pos_weight, feature_dim
