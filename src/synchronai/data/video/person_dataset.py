"""
Person-aware video dataset for dyadic synchrony.

Loads pre-computed bounding boxes from JSON sidecar files and returns
per-person crops alongside full frames. The PersonAwareVideoClassifier
uses these crops for cross-person attention.

Bounding box format (JSON sidecar):
{
    "0": [{"bbox": [x1, y1, x2, y2], "confidence": 0.95, "role": "adult"}, ...],
    "1": [{"bbox": [x1, y1, x2, y2], "confidence": 0.90, "role": "child"}, ...],
    ...
}

Role assignment: larger bbox area = adult (person_a), smaller = child (person_b).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from synchronai.data.video.dataset import VideoDatasetConfig, VideoWindowSpec
from synchronai.data.video.processing import (
    VideoReaderPool,
    crop_and_preprocess_person,
    preprocess_dinov2_frame,
    sample_window_timestamps,
    create_video_reader,
)

logger = logging.getLogger(__name__)


@dataclass
class PersonAwareDatasetConfig(VideoDatasetConfig):
    """Extended config for person-aware dataset."""

    bbox_dir: Optional[str] = None  # Directory with JSON bbox sidecar files
    fallback_to_full_frame: bool = True  # Use full frame when no bbox available
    imagenet_normalize: bool = True  # Use ImageNet normalization (DINOv2)
    person_frame_size: int = 224  # Crop size for person crops


def _load_bbox_sidecar(
    video_path: str,
    bbox_dir: str,
) -> Optional[dict[int, list[dict]]]:
    """Load bounding box JSON sidecar for a video.

    Args:
        video_path: Path to video file
        bbox_dir: Directory containing bbox JSON files

    Returns:
        Dict mapping second -> list of detections, or None if not found
    """
    video_stem = Path(video_path).stem
    bbox_path = Path(bbox_dir) / f"{video_stem}_person_bboxes.json"

    if not bbox_path.exists():
        return None

    with open(bbox_path, "r") as f:
        data = json.load(f)

    # Convert string keys to int
    return {int(k): v for k, v in data.items()}


def _assign_roles(
    detections: list[dict],
) -> tuple[Optional[dict], Optional[dict]]:
    """Assign person_a (adult) and person_b (child) from detections.

    Role assignment: larger bbox area = adult (person_a).

    Args:
        detections: List of detection dicts with "bbox" key

    Returns:
        Tuple of (person_a_detection, person_b_detection), either can be None
    """
    if not detections:
        return None, None

    if len(detections) == 1:
        return detections[0], None

    # Sort by bbox area (largest first)
    def bbox_area(det):
        x1, y1, x2, y2 = det["bbox"]
        return (x2 - x1) * (y2 - y1)

    sorted_dets = sorted(detections, key=bbox_area, reverse=True)
    return sorted_dets[0], sorted_dets[1]


class PersonAwareVideoDataset(Dataset):
    """Dataset that returns per-person crops and full frames.

    Each sample includes:
    - person_a_crops: (T, C, H, W) crops of person A (adult)
    - person_b_crops: (T, C, H, W) crops of person B (child)
    - full_frames: (T, C, H, W) full-frame DINOv2-preprocessed
    - person_count: int (0, 1, or 2)
    - label: int (synchrony label)
    """

    def __init__(
        self,
        specs: list[VideoWindowSpec],
        config: PersonAwareDatasetConfig,
        augment: bool = False,
    ):
        self.specs = specs
        self.config = config
        self.augment = augment

        # Load bbox sidecars for all videos
        self._bbox_cache: dict[str, Optional[dict[int, list[dict]]]] = {}
        if config.bbox_dir:
            unique_videos = set(s.video_path for s in specs)
            for video_path in unique_videos:
                self._bbox_cache[video_path] = _load_bbox_sidecar(
                    video_path, config.bbox_dir
                )
            n_with_bboxes = sum(1 for v in self._bbox_cache.values() if v is not None)
            logger.info(
                f"PersonAwareVideoDataset: {len(specs)} specs, "
                f"{n_with_bboxes}/{len(unique_videos)} videos with bbox data"
            )
        else:
            logger.info(
                f"PersonAwareVideoDataset: {len(specs)} specs, "
                f"no bbox_dir configured (full-frame mode)"
            )

        # Video reader pool
        self._reader_pool = VideoReaderPool(
            max_readers=config.reader_pool_size,
            backend=config.video_backend,
        )

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        spec = self.specs[idx]
        target_size = self.config.person_frame_size
        timestamps = sample_window_timestamps(
            spec.second, spec.window_seconds, spec.sample_fps
        )

        # Get reader
        reader = self._reader_pool.get_reader(spec.video_path)

        # Get bbox detections for this second
        bbox_data = self._bbox_cache.get(spec.video_path)
        detections = bbox_data.get(spec.second, []) if bbox_data else []
        person_a_det, person_b_det = _assign_roles(detections)

        person_count = sum(1 for d in [person_a_det, person_b_det] if d is not None)

        # Read and preprocess frames
        person_a_crops = []
        person_b_crops = []
        full_frames = []

        for ts in timestamps:
            try:
                frame = reader.get_frame_at_timestamp(ts)
            except Exception as e:
                logger.warning(f"Failed to read frame at {ts}s: {e}")
                blank = np.zeros((3, target_size, target_size), dtype=np.float32)
                person_a_crops.append(blank)
                person_b_crops.append(blank)
                full_frames.append(blank)
                continue

            # Full frame (always computed for fallback)
            full_frames.append(preprocess_dinov2_frame(frame, target_size))

            # Person crops
            if person_a_det is not None:
                person_a_crops.append(
                    crop_and_preprocess_person(frame, tuple(person_a_det["bbox"]), target_size)
                )
            else:
                person_a_crops.append(preprocess_dinov2_frame(frame, target_size))

            if person_b_det is not None:
                person_b_crops.append(
                    crop_and_preprocess_person(frame, tuple(person_b_det["bbox"]), target_size)
                )
            else:
                person_b_crops.append(preprocess_dinov2_frame(frame, target_size))

        return {
            "person_a_crops": torch.from_numpy(np.stack(person_a_crops)),
            "person_b_crops": torch.from_numpy(np.stack(person_b_crops)),
            "full_frames": torch.from_numpy(np.stack(full_frames)),
            "person_count": torch.tensor(person_count, dtype=torch.long),
            "label": torch.tensor(spec.label, dtype=torch.float32),
            "video_path": spec.video_path,
            "second": spec.second,
        }

    def get_pos_weight(self) -> float:
        """Compute positive class weight for imbalanced data."""
        labels = [s.label for s in self.specs]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        if n_pos == 0:
            return 1.0
        return n_neg / n_pos

    def close(self) -> None:
        """Release all video readers."""
        self._reader_pool.close_all()
