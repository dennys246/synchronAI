"""
PyTorch Dataset for pre-extracted fNIRS encoder features.

Loads pre-computed U-Net bottleneck or multiscale features from disk,
following the same pattern as the audio and video feature datasets.

Two on-disk formats are supported:

1. **Unpacked** (legacy): one .pt file per entry under `features/`.
   Slow on GPFS/NFS because each sample is an independent `open()` call.

2. **Packed** (preferred): a single `features_packed.bin` memmap with a
   `features_meta.json` describing the layout. `feature_index.csv` gains a
   `row_idx` column mapping each entry to its row in the packed array.
   Random access is page-faults-only — no syscall per sample.

Packed format is auto-detected by `create_fnirs_feature_dataloaders`.
Run `scripts/pack_features.py <feature_dir>` to convert an existing
unpacked directory to packed.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Union

import numpy as np
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


class FnirsPackedFeatureDataset(Dataset):
    """mmap-backed dataset for packed fNIRS features.

    Reads one big `features_packed.bin` file via `np.memmap` and indexes into it.
    No syscall per sample — the OS pages in on demand. After epoch 1, hot rows
    stay in the page cache.

    The mmap is opened lazily on first `__getitem__` access so the dataset can
    be safely pickled to DataLoader worker processes (each worker opens its own
    handle instead of inheriting a parent handle).
    """

    def __init__(
        self,
        feature_dir: Union[str, Path],
        entries: list[dict],
        label_column: str = "participant_type",
        label_map: dict[str, int] | None = None,
    ):
        self.feature_dir = Path(feature_dir)
        self.entries = entries
        self.label_column = label_column
        self.label_map = label_map or {"child": 0, "adult": 1}

        meta_path = self.feature_dir / "features_meta.json"
        with open(meta_path) as f:
            self.meta = json.load(f)
        self._mmap_shape = tuple(self.meta["shape"])
        self._mmap_dtype = self.meta["dtype"]
        self._mmap_path = self.feature_dir / "features_packed.bin"
        self._mmap = None  # lazy, per-worker

        # Validate that every entry has a row_idx
        if entries and "row_idx" not in entries[0]:
            raise ValueError(
                "Packed dataset requires 'row_idx' column in feature_index.csv. "
                "Re-run scripts/pack_features.py on this feature dir."
            )

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

    def _get_mmap(self) -> np.memmap:
        if self._mmap is None:
            self._mmap = np.memmap(
                self._mmap_path,
                dtype=self._mmap_dtype,
                mode="r",
                shape=self._mmap_shape,
            )
        return self._mmap

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        entry = self.entries[idx]
        row_idx = int(entry["row_idx"])

        mmap = self._get_mmap()
        # np.array(...) materializes the mmap row into a regular NumPy array
        # (copy). torch.from_numpy then wraps that array without copying —
        # the tensor's lifetime is tied to the NumPy copy, not the mmap file.
        features = torch.from_numpy(np.array(mmap[row_idx]))

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


def is_feature_dir_packed(feature_dir: Union[str, Path]) -> bool:
    """Return True if feature_dir has a packed binary + meta.json."""
    feature_dir = Path(feature_dir)
    return (
        (feature_dir / "features_packed.bin").exists()
        and (feature_dir / "features_meta.json").exists()
    )


def pack_features(
    feature_dir: Union[str, Path],
    delete_unpacked: bool = False,
    chunk_log: int = 10000,
) -> dict:
    """Pack individual .pt files into a single mmap-friendly binary.

    Reads feature_index.csv, loads each .pt, writes rows into
    features_packed.bin (float32), and appends a `row_idx` column to
    the CSV. Writes features_meta.json describing the layout.

    Requires all features to have the same shape. Errors if not.

    Args:
        feature_dir: Directory containing feature_index.csv + features/*.pt
        delete_unpacked: If True, remove individual .pt files after successful pack.
        chunk_log: Log progress every N entries.

    Returns:
        Dict with pack stats (n_entries, shape, size_mb, elapsed_sec).
    """
    feature_dir = Path(feature_dir)
    index_path = feature_dir / "feature_index.csv"
    files_dir = feature_dir / "features"

    if not index_path.exists():
        raise FileNotFoundError(f"No feature_index.csv at {index_path}")
    if not files_dir.exists():
        raise FileNotFoundError(f"No features/ dir at {files_dir}")

    df = pd.read_csv(index_path)
    n = len(df)
    if n == 0:
        raise ValueError(f"Empty feature index at {index_path}")

    # Peek the first file to infer shape/dtype. All subsequent files must
    # match exactly — pack_features errors loudly on any mismatch rather
    # than silently writing mixed shapes.
    first_fname = df.iloc[0]["feature_file"]
    first_feat = torch.load(
        files_dir / first_fname, map_location="cpu", weights_only=True
    )
    per_entry_shape = tuple(first_feat.shape)
    per_entry_ndim = first_feat.ndim
    dtype = "float32"
    logger.info(
        "Inferred per-entry layout from %s: shape=%s, ndim=%d",
        first_fname, per_entry_shape, per_entry_ndim,
    )

    total_shape = (n,) + per_entry_shape
    packed_path = feature_dir / "features_packed.bin"

    logger.info(
        "Packing %d entries of shape %s → %s (dtype=%s)",
        n, per_entry_shape, packed_path.name, dtype,
    )

    # Write sequentially via regular file I/O. Using np.memmap(mode="w+")
    # would require reserving the full file's virtual address space upfront,
    # which exceeds login-node ulimits for large (~tens of GB) outputs.
    # Sequential writes have no such requirement and are equally fast for
    # the one-time pack operation — mmap is only needed during training reads.
    t_start = time.time()
    with open(packed_path, "wb") as f:
        for row_idx, row in enumerate(df.itertuples(index=False)):
            fname = row.feature_file
            feat = torch.load(
                files_dir / fname, map_location="cpu", weights_only=True
            )
            if tuple(feat.shape) != per_entry_shape:
                raise ValueError(
                    f"Shape mismatch at row {row_idx} ({fname}): "
                    f"{tuple(feat.shape)} != {per_entry_shape}. All features "
                    f"must share the same shape to pack."
                )
            arr = np.ascontiguousarray(
                feat.numpy().astype(dtype, copy=False)
            )
            f.write(arr.tobytes())

            if (row_idx + 1) % chunk_log == 0:
                rate = (row_idx + 1) / (time.time() - t_start)
                logger.info("  packed %d/%d (%.0f/s)", row_idx + 1, n, rate)

    # Append row_idx to the CSV (overwrite in place)
    df["row_idx"] = range(n)
    df.to_csv(index_path, index=False)

    meta = {
        "shape": list(total_shape),
        "dtype": dtype,
        "n_entries": n,
        "per_entry_shape": list(per_entry_shape),
    }
    with open(feature_dir / "features_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = packed_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t_start
    logger.info(
        "Packed %d entries, %.1f MB, %.1fs (%.0f entries/s)",
        n, size_mb, elapsed, n / max(elapsed, 1e-6),
    )

    if delete_unpacked:
        logger.info("Deleting %d unpacked .pt files from %s", n, files_dir)
        removed = 0
        for row in df.itertuples(index=False):
            p = files_dir / row.feature_file
            try:
                p.unlink()
                removed += 1
            except FileNotFoundError:
                pass
        logger.info("Removed %d files", removed)

    return {
        "n_entries": n,
        "shape": list(total_shape),
        "size_mb": size_mb,
        "elapsed_sec": elapsed,
    }


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

    if is_feature_dir_packed(feature_dir):
        logger.info("Using packed feature format (mmap-backed)")
        dataset_cls = FnirsPackedFeatureDataset
    else:
        logger.info("Using unpacked feature format (one .pt per sample)")
        dataset_cls = FnirsFeatureDataset

    train_dataset = dataset_cls(
        feature_dir, train_entries, label_column, label_map
    )
    val_dataset = dataset_cls(
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
