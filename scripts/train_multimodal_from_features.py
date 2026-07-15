#!/usr/bin/env python3
"""Train a multi-modal classifier on pre-extracted DINOv2 + WavLM features.

Mirrors scripts/train_audio_from_features.py but joins two feature dirs by
(video_path, second). Designed for CPU — both backbones are pre-computed,
so per-batch cost is just two small LSTMs and a fusion head.

Usage:
    python scripts/train_multimodal_from_features.py \
        --video-feature-dir data/dinov2_features \
        --audio-feature-dir data/wavlm_baseplus_features \
        --save-dir runs/multimodal_features/lstm_concat \
        --epochs 50
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Honor LSF slot count on CPU. span[hosts=1] places slots on one host, but
# PyTorch's intra-op pool defaults to host-physical-cores, not LSF -n, so
# without this we either over-subscribe or under-use the reservation.
_omp = os.environ.get("OMP_NUM_THREADS")
if _omp:
    torch.set_num_threads(int(_omp))

# DataLoader workers ship batch tensors via shared memory by default.
# continuumio/anaconda3 docker container caps /dev/shm at ~64 MB, but the
# v2_patch path ships ~1.15 GB batches ((128, 12, 257, 768) float32) and
# died with `unable to allocate shared memory(shm)`. Switching to the
# file_system sharing strategy uses $TMPDIR (which the LSF tmp reservation
# sizes) instead of /dev/shm. Slight per-batch overhead, but unblocks the
# patch probe. Safe for the other archs since their batches are small.
torch.multiprocessing.set_sharing_strategy("file_system")


class MultiModalFeatureDataset(Dataset):
    """Preloads DINOv2 video features and WavLM audio features into RAM at init.

    For our scale (~59K samples × ~50KB each ≈ 11 GB), preloading is faster and
    simpler than per-sample disk reads. Avoids 100K+ small NFS reads per epoch
    and DataLoader-worker shm pressure inside Docker.
    """

    def __init__(
        self,
        video_feature_dir: Path,
        audio_feature_dir: Path,
        entries: list[dict],
        audio_shuffle_prob: float = 0.0,
        subject_id_to_int: dict[str, int] | None = None,
    ):
        video_dir = Path(video_feature_dir) / "features"
        audio_dir = Path(audio_feature_dir) / "features"

        # Probe 2: per-sample within-recording audio shuffle. When > 0, each
        # __getitem__ has this probability of replacing the audio tensor with
        # the audio from a different second in the SAME recording, and forcing
        # label=0. Constrained to within-recording so the model can't shortcut
        # via "different scene/subject" cues. shuffle_prob=0 → no change.
        self.audio_shuffle_prob = audio_shuffle_prob

        n = len(entries)
        logger.info(f"Preloading {n} samples into RAM...")
        load_start = time.time()

        # Peek at the first file to learn shapes, then pre-allocate destination
        # tensors. Pre-allocation avoids the 2× peak memory of list+torch.stack:
        # for a 9 GB tensor, stack briefly holds the source list AND the
        # destination, doubling the working set right before the list is freed.
        first_v = torch.load(
            video_dir / entries[0]["video_feature_file"],
            map_location="cpu", weights_only=True,
        ).detach()
        first_a = torch.load(
            audio_dir / entries[0]["audio_feature_file"],
            map_location="cpu", weights_only=True,
        ).detach()
        if first_v.ndim != 2 or first_a.ndim != 2:
            raise ValueError(
                f"Expected 2D feature tensors (T, D); got video {first_v.shape} "
                f"and audio {first_a.shape}. If video is 3D (T, P, D), use a "
                f"pre-pooled feature dir like dinov2_features_meanpatch instead."
            )

        self.video_tensor = torch.empty((n, *first_v.shape), dtype=torch.float32)
        self.audio_tensor = torch.empty((n, *first_a.shape), dtype=torch.float32)
        self.labels = torch.empty(n, dtype=torch.float32)
        self.video_tensor[0] = first_v
        self.audio_tensor[0] = first_a
        self.labels[0] = float(entries[0]["label"])

        for i in range(1, n):
            entry = entries[i]
            self.video_tensor[i] = torch.load(
                video_dir / entry["video_feature_file"],
                map_location="cpu", weights_only=True,
            ).detach()
            self.audio_tensor[i] = torch.load(
                audio_dir / entry["audio_feature_file"],
                map_location="cpu", weights_only=True,
            ).detach()
            self.labels[i] = float(entry["label"])
            if (i + 1) % 5000 == 0:
                logger.info(f"  Loaded {i+1}/{n} ({(i+1)/n*100:.1f}%)")

        v_mb = self.video_tensor.element_size() * self.video_tensor.nelement() / 1e9
        a_mb = self.audio_tensor.element_size() * self.audio_tensor.nelement() / 1e9
        logger.info(
            f"Preload complete in {time.time() - load_start:.1f}s. "
            f"Video tensor: {tuple(self.video_tensor.shape)} ({v_mb:.2f} GB), "
            f"Audio tensor: {tuple(self.audio_tensor.shape)} ({a_mb:.2f} GB)"
        )

        # Probe 2: build per-recording index for within-recording audio shuffle.
        # Only built when audio_shuffle_prob > 0; otherwise dead weight.
        self._by_recording: dict[str, list[int]] = {}
        if self.audio_shuffle_prob > 0:
            by_rec: dict[str, list[int]] = defaultdict(list)
            for i, e in enumerate(entries):
                by_rec[e["video_path"]].append(i)
            # Only keep recordings with at least 2 samples (otherwise nothing to swap with).
            self._by_recording = {k: v for k, v in by_rec.items() if len(v) > 1}
            self._sample_recording = [e["video_path"] for e in entries]
            n_recordings = len(self._by_recording)
            n_eligible = sum(len(v) for v in self._by_recording.values())
            logger.info(
                f"Audio-shuffle augmentation enabled: prob={self.audio_shuffle_prob:.2f}, "
                f"{n_recordings} eligible recordings, {n_eligible}/{n} eligible samples."
            )

        # Subject_id integer indices, only built when a vocab is provided.
        # Required by v2_subjinv (and mirrors v8_subjinv) for the adversarial
        # subject head. The vocab is built from TRAIN subjects only; val
        # subjects are absent and map to -1, which the training loop masks out
        # of the subject loss.
        self.subject_id_to_int = subject_id_to_int
        if subject_id_to_int is not None:
            self.subject_ids_int = torch.tensor(
                [subject_id_to_int.get(e.get("subject_id", ""), -1) for e in entries],
                dtype=torch.long,
            )
        else:
            self.subject_ids_int = None

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> dict:
        # Probe 2: within-recording audio shuffle. Replace audio with audio
        # from another second in the same recording and force label=0. Keeps
        # subject/scene/recording-conditions identical → only the cross-modal
        # temporal alignment is broken, isolating the synchrony signal.
        if self.audio_shuffle_prob > 0:
            rec = self._sample_recording[idx]
            candidates = self._by_recording.get(rec)
            if candidates is not None and random.random() < self.audio_shuffle_prob:
                # Sample a different idx from the same recording.
                other_idx = idx
                while other_idx == idx:
                    other_idx = random.choice(candidates)
                out = {
                    "video_features": self.video_tensor[idx],
                    "audio_features": self.audio_tensor[other_idx],
                    "label": torch.zeros((), dtype=torch.float32),
                }
                if self.subject_ids_int is not None:
                    out["subject_id_int"] = self.subject_ids_int[idx]
                return out
        out = {
            "video_features": self.video_tensor[idx],
            "audio_features": self.audio_tensor[idx],
            "label": self.labels[idx],
        }
        if self.subject_ids_int is not None:
            out["subject_id_int"] = self.subject_ids_int[idx]
        return out


class MultiModalPatchFeatureDataset(Dataset):
    """Lazy-load variant for 3D video features (T, P, D) — e.g. DINOv2 patch grid.

    The full DINOv2 patch features are (12, 257, 768) = 9.5 MB per file. For
    ~59K samples that's ~565 GB, so we can't preload like the meanpatch path.
    This dataset reads one video file per __getitem__ off disk. Audio features
    are 2D (49, 768) = 0.19 MB per file and we preload those as usual to keep
    audio I/O off the hot path.

    Used by --arch v2_patch (probe 1: does spatial patch information rescue
    the multimodal ceiling?).
    """

    def __init__(
        self,
        video_feature_dir: Path,
        audio_feature_dir: Path,
        entries: list[dict],
    ):
        self.video_dir = Path(video_feature_dir) / "features"
        self.audio_dir = Path(audio_feature_dir) / "features"

        n = len(entries)
        # Peek at the first video to learn shape.
        first_v = torch.load(
            self.video_dir / entries[0]["video_feature_file"],
            map_location="cpu", weights_only=True,
        ).detach()
        first_a = torch.load(
            self.audio_dir / entries[0]["audio_feature_file"],
            map_location="cpu", weights_only=True,
        ).detach()
        if first_v.ndim != 3:
            raise ValueError(
                f"PatchFeatureDataset expects 3D video features (T, P, D); got {first_v.shape}. "
                f"Use the standard MultiModalFeatureDataset for 2D (T, D) features."
            )
        if first_a.ndim != 2:
            raise ValueError(
                f"Audio features must be 2D (T, D); got {first_a.shape}."
            )
        self.video_shape = tuple(first_v.shape)

        # Preload audio (small, ~11 GB for full set) and labels; keep video lazy.
        logger.info(
            f"Preloading {n} audio samples (video is lazy-loaded per __getitem__)..."
        )
        load_start = time.time()
        self.audio_tensor = torch.empty((n, *first_a.shape), dtype=torch.float32)
        self.labels = torch.empty(n, dtype=torch.float32)
        self.video_files = [entries[i]["video_feature_file"] for i in range(n)]
        self.audio_tensor[0] = first_a
        self.labels[0] = float(entries[0]["label"])
        for i in range(1, n):
            entry = entries[i]
            self.audio_tensor[i] = torch.load(
                self.audio_dir / entry["audio_feature_file"],
                map_location="cpu", weights_only=True,
            ).detach()
            self.labels[i] = float(entry["label"])
            if (i + 1) % 5000 == 0:
                logger.info(f"  Loaded {i+1}/{n} audio samples")
        a_gb = self.audio_tensor.element_size() * self.audio_tensor.nelement() / 1e9
        logger.info(
            f"Audio preload complete in {time.time() - load_start:.1f}s. "
            f"Audio tensor: {tuple(self.audio_tensor.shape)} ({a_gb:.2f} GB). "
            f"Video lazy-loaded: per-sample shape {self.video_shape} "
            f"(~{4 * np.prod(self.video_shape) / 1e6:.1f} MB/file)."
        )

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> dict:
        v = torch.load(
            self.video_dir / self.video_files[idx],
            map_location="cpu", weights_only=True,
        ).detach()
        return {
            "video_features": v,
            "audio_features": self.audio_tensor[idx],
            "label": self.labels[idx],
        }


class MultiModalCrossPersonFeatureDataset(Dataset):
    """Preloads (per-person video, audio) pairs for the v6_crossperson probe.

    Video features are (T, 2 persons, D) — produced by
    build_perperson_video_features.py from patch features + pose bboxes.
    Audio is the standard 2D (T, D). Per-sample size is small (~76 KB
    video + ~190 KB audio), so full preload is comfortable (~15 GB total).
    """

    def __init__(
        self,
        video_feature_dir: Path,
        audio_feature_dir: Path,
        entries: list[dict],
        subject_id_to_int: dict[str, int] | None = None,
    ):
        video_dir = Path(video_feature_dir) / "features"
        audio_dir = Path(audio_feature_dir) / "features"
        # Optional: when subject_id_to_int is provided, __getitem__ returns a
        # subject_id_int tensor. Required by v8_subjinv for the adversarial
        # subject classifier. None on val sets (we don't compute the dual
        # loss on val), but the trainer also passes a "fallback" mapping
        # for val so the loader doesn't trip on missing keys.
        self.subject_id_to_int = subject_id_to_int

        n = len(entries)
        logger.info(f"Preloading {n} per-person video + audio samples into RAM...")
        load_start = time.time()

        first_v = torch.load(
            video_dir / entries[0]["video_feature_file"],
            map_location="cpu", weights_only=True,
        ).detach()
        first_a = torch.load(
            audio_dir / entries[0]["audio_feature_file"],
            map_location="cpu", weights_only=True,
        ).detach()
        if first_v.ndim != 3 or first_v.shape[1] != 2:
            raise ValueError(
                f"CrossPersonFeatureDataset expects 3D video features "
                f"(T, 2 persons, D); got {first_v.shape}."
            )
        if first_a.ndim != 2:
            raise ValueError(f"Audio features must be 2D (T, D); got {first_a.shape}.")

        self.video_tensor = torch.empty((n, *first_v.shape), dtype=torch.float32)
        self.audio_tensor = torch.empty((n, *first_a.shape), dtype=torch.float32)
        self.labels = torch.empty(n, dtype=torch.float32)
        self.video_tensor[0] = first_v
        self.audio_tensor[0] = first_a
        self.labels[0] = float(entries[0]["label"])
        for i in range(1, n):
            entry = entries[i]
            self.video_tensor[i] = torch.load(
                video_dir / entry["video_feature_file"],
                map_location="cpu", weights_only=True,
            ).detach()
            self.audio_tensor[i] = torch.load(
                audio_dir / entry["audio_feature_file"],
                map_location="cpu", weights_only=True,
            ).detach()
            self.labels[i] = float(entry["label"])
            if (i + 1) % 5000 == 0:
                logger.info(f"  Loaded {i+1}/{n}")

        # Subject_id integer indices (only built when subject_id_to_int provided).
        # -1 for any subject not in the mapping (e.g., val subjects when only
        # the train mapping was supplied). v8's train loop ignores -1.
        if self.subject_id_to_int is not None:
            self.subject_ids_int = torch.tensor([
                self.subject_id_to_int.get(e.get("subject_id", ""), -1)
                for e in entries
            ], dtype=torch.long)
        else:
            self.subject_ids_int = None

        v_gb = self.video_tensor.element_size() * self.video_tensor.nelement() / 1e9
        a_gb = self.audio_tensor.element_size() * self.audio_tensor.nelement() / 1e9
        logger.info(
            f"Preload complete in {time.time() - load_start:.1f}s. "
            f"Video {tuple(self.video_tensor.shape)} ({v_gb:.2f} GB), "
            f"Audio {tuple(self.audio_tensor.shape)} ({a_gb:.2f} GB)."
        )

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> dict:
        out = {
            "video_features": self.video_tensor[idx],
            "audio_features": self.audio_tensor[idx],
            "label": self.labels[idx],
        }
        if self.subject_ids_int is not None:
            out["subject_id_int"] = self.subject_ids_int[idx]
        return out


class MultiModalPoseFeatureDataset(Dataset):
    """Preloads (video, audio, pose) features for the v2_pose 3-modality probe.

    Pose features per file are (12, 2, 17, 3) float32 = ~4.9 KB each — even
    smaller than audio, so preload is trivial. Video uses meanpatch (2D)
    here, not the patch grid; the question this dataset's probe answers is
    "does pose ADD signal beyond the existing meanpatch+wavlm baseline?",
    not "does pose interact with patch-level spatial info?".

    Expects the merged entries to include a `pose_feature_file` column on
    top of the standard `video_feature_file` / `audio_feature_file`.
    """

    def __init__(
        self,
        video_feature_dir: Path,
        audio_feature_dir: Path,
        pose_feature_dir: Path,
        entries: list[dict],
    ):
        video_dir = Path(video_feature_dir) / "features"
        audio_dir = Path(audio_feature_dir) / "features"
        pose_dir = Path(pose_feature_dir) / "features"

        n = len(entries)
        logger.info(f"Preloading {n} video+audio+pose samples into RAM...")
        load_start = time.time()

        first_v = torch.load(
            video_dir / entries[0]["video_feature_file"],
            map_location="cpu", weights_only=True,
        ).detach()
        first_a = torch.load(
            audio_dir / entries[0]["audio_feature_file"],
            map_location="cpu", weights_only=True,
        ).detach()
        first_p = torch.load(
            pose_dir / entries[0]["pose_feature_file"],
            map_location="cpu", weights_only=True,
        ).detach()
        if first_v.ndim != 2:
            raise ValueError(
                f"PoseFeatureDataset expects 2D video features (T, D); got {first_v.shape}. "
                f"Use data/dinov2_features_meanpatch/ (not patches) for the pose probe."
            )
        if first_a.ndim != 2:
            raise ValueError(f"Audio features must be 2D (T, D); got {first_a.shape}.")
        if first_p.ndim != 4 or first_p.shape[1:] != (2, 17, 3):
            raise ValueError(
                f"Pose features must be (T, 2 people, 17 kpts, 3 channels); "
                f"got {first_p.shape}."
            )

        self.video_tensor = torch.empty((n, *first_v.shape), dtype=torch.float32)
        self.audio_tensor = torch.empty((n, *first_a.shape), dtype=torch.float32)
        self.pose_tensor = torch.empty((n, *first_p.shape), dtype=torch.float32)
        self.labels = torch.empty(n, dtype=torch.float32)
        self.video_tensor[0] = first_v
        self.audio_tensor[0] = first_a
        self.pose_tensor[0] = first_p
        self.labels[0] = float(entries[0]["label"])

        for i in range(1, n):
            entry = entries[i]
            self.video_tensor[i] = torch.load(
                video_dir / entry["video_feature_file"],
                map_location="cpu", weights_only=True,
            ).detach()
            self.audio_tensor[i] = torch.load(
                audio_dir / entry["audio_feature_file"],
                map_location="cpu", weights_only=True,
            ).detach()
            self.pose_tensor[i] = torch.load(
                pose_dir / entry["pose_feature_file"],
                map_location="cpu", weights_only=True,
            ).detach()
            self.labels[i] = float(entry["label"])
            if (i + 1) % 5000 == 0:
                logger.info(f"  Loaded {i+1}/{n} samples")

        v_gb = self.video_tensor.element_size() * self.video_tensor.nelement() / 1e9
        a_gb = self.audio_tensor.element_size() * self.audio_tensor.nelement() / 1e9
        p_gb = self.pose_tensor.element_size() * self.pose_tensor.nelement() / 1e9
        logger.info(
            f"Triple-preload complete in {time.time() - load_start:.1f}s. "
            f"Video {tuple(self.video_tensor.shape)} ({v_gb:.2f} GB), "
            f"Audio {tuple(self.audio_tensor.shape)} ({a_gb:.2f} GB), "
            f"Pose  {tuple(self.pose_tensor.shape)} ({p_gb:.2f} GB)."
        )

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "video_features": self.video_tensor[idx],
            "audio_features": self.audio_tensor[idx],
            "pose_features": self.pose_tensor[idx],
            "label": self.labels[idx],
        }


class MultiModalContextFeatureDataset(Dataset):
    """Multi-second context window dataset for v7_temporal_context.

    For each labeled (video_path, second=t), returns features for seconds
    [t-k, t-k+1, ..., t, ..., t+k] where k = context_window. Seconds outside
    the recording bounds are zero-padded.

    Synchrony unfolds over 5-15 seconds but our trainers have been predicting
    each second from a strict 1-second window with no across-second context.
    This dataset gives the model the multi-second receptive field humans use
    when coding.

    Output per sample:
      video_features: (2k+1, T_v, D_v)  — context window of video features
      audio_features: (2k+1, T_a, D_a)
      label:          single label for the center second
    """

    def __init__(
        self,
        video_feature_dir: Path,
        audio_feature_dir: Path,
        entries: list[dict],
        context_window: int = 2,
    ):
        video_dir = Path(video_feature_dir) / "features"
        audio_dir = Path(audio_feature_dir) / "features"
        self.context_window = context_window
        self.context_size = 2 * context_window + 1

        n = len(entries)
        logger.info(
            f"Preloading {n} samples (context_window={context_window} → "
            f"{self.context_size}-second windows)..."
        )
        load_start = time.time()

        # Peek at first file shapes.
        first_v = torch.load(
            video_dir / entries[0]["video_feature_file"],
            map_location="cpu", weights_only=True,
        ).detach()
        first_a = torch.load(
            audio_dir / entries[0]["audio_feature_file"],
            map_location="cpu", weights_only=True,
        ).detach()
        if first_v.ndim != 2 or first_a.ndim != 2:
            raise ValueError(
                f"ContextFeatureDataset expects 2D features (T, D); got "
                f"video {first_v.shape} and audio {first_a.shape}. Use "
                f"data/dinov2_features_meanpatch + data/wavlm_baseplus_features."
            )

        # Preload all features into RAM as flat tensors indexed by entry order.
        self.video_tensor = torch.empty((n, *first_v.shape), dtype=torch.float32)
        self.audio_tensor = torch.empty((n, *first_a.shape), dtype=torch.float32)
        self.labels = torch.empty(n, dtype=torch.float32)
        self.video_tensor[0] = first_v
        self.audio_tensor[0] = first_a
        self.labels[0] = float(entries[0]["label"])
        for i in range(1, n):
            entry = entries[i]
            self.video_tensor[i] = torch.load(
                video_dir / entry["video_feature_file"],
                map_location="cpu", weights_only=True,
            ).detach()
            self.audio_tensor[i] = torch.load(
                audio_dir / entry["audio_feature_file"],
                map_location="cpu", weights_only=True,
            ).detach()
            self.labels[i] = float(entry["label"])
            if (i + 1) % 5000 == 0:
                logger.info(f"  Loaded {i+1}/{n}")

        # Build per-recording (video_path → {second: idx}) lookup so __getitem__
        # can fetch context window features from the right entries.
        self.by_recording: dict[str, dict[int, int]] = {}
        for i, e in enumerate(entries):
            self.by_recording.setdefault(e["video_path"], {})[int(e["second"])] = i
        self._sample_recording = [e["video_path"] for e in entries]
        self._sample_second = [int(e["second"]) for e in entries]

        v_gb = self.video_tensor.element_size() * self.video_tensor.nelement() / 1e9
        a_gb = self.audio_tensor.element_size() * self.audio_tensor.nelement() / 1e9
        logger.info(
            f"Preload complete in {time.time() - load_start:.1f}s. "
            f"Video {tuple(self.video_tensor.shape)} ({v_gb:.2f} GB), "
            f"Audio {tuple(self.audio_tensor.shape)} ({a_gb:.2f} GB), "
            f"{len(self.by_recording)} recordings."
        )

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> dict:
        center_second = self._sample_second[idx]
        recording = self._sample_recording[idx]
        rec_map = self.by_recording[recording]
        zero_v = torch.zeros_like(self.video_tensor[0])
        zero_a = torch.zeros_like(self.audio_tensor[0])

        v_window = []
        a_window = []
        for offset in range(-self.context_window, self.context_window + 1):
            t = center_second + offset
            j = rec_map.get(t)
            if j is None:
                v_window.append(zero_v)
                a_window.append(zero_a)
            else:
                v_window.append(self.video_tensor[j])
                a_window.append(self.audio_tensor[j])

        return {
            "video_features": torch.stack(v_window, dim=0),  # (S, T_v, D_v)
            "audio_features": torch.stack(a_window, dim=0),  # (S, T_a, D_a)
            "label": self.labels[idx],
        }


def collate(batch: list[dict]) -> dict:
    out = {
        "video_features": torch.stack([b["video_features"] for b in batch]),
        "audio_features": torch.stack([b["audio_features"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }
    # Optional pose tensor for v2_pose runs. All-or-nothing — if the first
    # batch element has it, every element must.
    if "pose_features" in batch[0]:
        out["pose_features"] = torch.stack([b["pose_features"] for b in batch])
    # Optional subject_id_int for v8_subjinv runs (adversarial subject head).
    if "subject_id_int" in batch[0]:
        out["subject_id_int"] = torch.stack([b["subject_id_int"] for b in batch])
    return out


class MultiModalLSTMConcat(nn.Module):
    """Per-modality LSTM aggregator -> concat -> MLP head.

    Design follows the fNIRS sweep finding: LSTM dominates mean/MLP heads
    on temporal feature sequences. Concat fusion is the simplest baseline.
    """

    def __init__(
        self,
        video_feature_dim: int,
        audio_feature_dim: int,
        video_hidden: int = 64,
        audio_hidden: int = 64,
        head_hidden: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.video_lstm = nn.LSTM(
            video_feature_dim, video_hidden, batch_first=True
        )
        self.audio_lstm = nn.LSTM(
            audio_feature_dim, audio_hidden, batch_first=True
        )
        self.head = nn.Sequential(
            nn.Linear(video_hidden + audio_hidden, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"MultiModalLSTMConcat: video_dim={video_feature_dim} -> {video_hidden}, "
            f"audio_dim={audio_feature_dim} -> {audio_hidden}, "
            f"head_hidden={head_hidden}, params={n_params:,}"
        )

    def forward(self, video_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        _, (v_h, _) = self.video_lstm(video_features)
        _, (a_h, _) = self.audio_lstm(audio_features)
        fused = torch.cat([v_h.squeeze(0), a_h.squeeze(0)], dim=-1)
        return self.head(fused).squeeze(-1)


class MultiModalV2(nn.Module):
    """v2: projection bottleneck → aggregator → explicit aggregator dropout → fusion.

    Diagnoses fixed vs v1:
      D1: 1-layer LSTM ignores its `dropout` arg. v2 audio uses 2-layer LSTM
          with inter-layer dropout, AND adds explicit Dropout on the aggregated
          repr (LSTM dropout doesn't apply to top-layer output).
      D3: project 768→64 before recurrent aggregation (regularization +
          ~3× speedup on the recurrent matmul).
      D5: smaller params (~172K vs 435K) makes per-batch cost tractable on CPU.

    Video uses mean-pool over T=12 (1 second of frames; LSTM rarely beats mean).
    Audio uses 2-layer LSTM (49 timesteps; temporal modeling helps).
    """

    def __init__(
        self,
        video_feature_dim: int,
        audio_feature_dim: int,
        proj_dim: int = 64,
        head_hidden: int = 64,
        proj_dropout: float = 0.3,
        lstm_dropout: float = 0.2,
        repr_dropout: float = 0.3,
        head_dropout: float = 0.3,
        input_norm: bool = False,
    ):
        super().__init__()
        # Optional per-modality LayerNorm on the RAW frozen features (over the
        # feature dim). Frozen DINOv2/WavLM features are anisotropic (a few
        # huge-norm dims dominate); normalizing before projection is a standard
        # fix. Identity when off → structurally identical to prior v2 runs.
        self.video_in_norm = nn.LayerNorm(video_feature_dim) if input_norm else nn.Identity()
        self.audio_in_norm = nn.LayerNorm(audio_feature_dim) if input_norm else nn.Identity()
        self.video_proj = nn.Sequential(
            nn.Linear(video_feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        self.audio_lstm = nn.LSTM(
            proj_dim, proj_dim, num_layers=2,
            dropout=lstm_dropout, batch_first=True,
        )
        # Explicit dropout on aggregated reprs — LSTM `dropout` is between-layers
        # only, so h_n[-1] is otherwise undropped going into fusion.
        self.video_repr_drop = nn.Dropout(repr_dropout)
        self.audio_repr_drop = nn.Dropout(repr_dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * proj_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"MultiModalV2: video_dim={video_feature_dim} -> {proj_dim} (mean-pool), "
            f"audio_dim={audio_feature_dim} -> {proj_dim} (LSTM x2), "
            f"head_hidden={head_hidden}, "
            f"dropouts: proj={proj_dropout}/lstm={lstm_dropout}/repr={repr_dropout}/head={head_dropout}, "
            f"params={n_params:,}"
        )

    def forward(self, video_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        v = self.video_proj(self.video_in_norm(video_features))  # (B, 12, P)
        v_repr = v.mean(dim=1)                      # (B, P)
        v_repr = self.video_repr_drop(v_repr)

        a = self.audio_proj(self.audio_in_norm(audio_features))  # (B, 49, P)
        _, (h_n, _) = self.audio_lstm(a)            # h_n: (num_layers=2, B, P), NOT batch-first
        a_repr = h_n[-1, :, :]                      # top-layer hidden across batch
        a_repr = self.audio_repr_drop(a_repr)

        fused = torch.cat([v_repr, a_repr], dim=-1)  # (B, 2P)
        return self.head(fused).squeeze(-1)


class MultiModalV2SubjInv(MultiModalV2):
    """v2 + adversarial subject-invariance head — the clean 2D analogue of v8.

    Same body as MultiModalV2 (project → mean-pool video / 2-layer LSTM audio →
    concat), plus a gradient-reversed subject classifier on the fused (2P) repr.
    The GRL feeds a NEGATIVE gradient to the shared encoder, so minimizing the
    subject cross-entropy pushes the encoder to DISCARD subject identity — the
    direct lever against the identify-the-dyad confound that inflates pooled AUC
    over within-recording AUC. Unlike v8_subjinv (v6_crossperson + this head,
    which needs 3D per-person features), this runs on the standard 2D features
    and the v2 baseline we are trying to beat, isolating the invariance lever.

    n_subjects_train is the TRAIN-only subject vocab size; lambda_subj scales the
    reversed gradient. forward returns sync_logit alone at eval time, or
    (sync_logit, subject_logits) when subject_id is passed.
    """

    def __init__(
        self,
        video_feature_dim: int,
        audio_feature_dim: int,
        n_subjects_train: int,
        proj_dim: int = 64,
        head_hidden: int = 64,
        proj_dropout: float = 0.3,
        lstm_dropout: float = 0.2,
        repr_dropout: float = 0.3,
        head_dropout: float = 0.3,
        input_norm: bool = False,
        lambda_subj: float = 0.1,
    ):
        super().__init__(
            video_feature_dim=video_feature_dim,
            audio_feature_dim=audio_feature_dim,
            proj_dim=proj_dim,
            head_hidden=head_hidden,
            proj_dropout=proj_dropout,
            lstm_dropout=lstm_dropout,
            repr_dropout=repr_dropout,
            head_dropout=head_dropout,
            input_norm=input_norm,
        )
        self.lambda_subj = lambda_subj
        self.subject_head = nn.Sequential(
            nn.Linear(2 * proj_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, n_subjects_train),
        )
        logger.info(
            f"MultiModalV2SubjInv: + adversarial subject head "
            f"({2 * proj_dim}->{n_subjects_train}), lambda_subj={lambda_subj}"
        )

    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
        subject_id: torch.Tensor | None = None,
    ):
        v = self.video_proj(self.video_in_norm(video_features))
        v_repr = self.video_repr_drop(v.mean(dim=1))
        a = self.audio_proj(self.audio_in_norm(audio_features))
        _, (h_n, _) = self.audio_lstm(a)
        a_repr = self.audio_repr_drop(h_n[-1, :, :])
        fused = torch.cat([v_repr, a_repr], dim=-1)  # (B, 2P)
        sync_logit = self.head(fused).squeeze(-1)
        if subject_id is None:
            return sync_logit
        # GRL feeds a negative gradient to the shared encoder; the subject loss
        # is added normally in the training loop (no sign flip needed there).
        fused_reversed = _grad_reverse(fused, self.lambda_subj)
        subject_logits = self.subject_head(fused_reversed)
        return sync_logit, subject_logits


class MultiModalV3(nn.Module):
    """v3: V2 architecture with cross-attention fusion replacing concat.

    Concat fusion treats the two modality reprs as independent vectors at the
    head — the head has to detect synchrony from juxtaposed summaries with no
    cross-modal interaction in the representations themselves. For dyadic
    synchrony, where the label IS the relationship between modalities, that's
    structurally weak.

    v3 keeps everything from V2 up through the per-modality reprs, then
    treats them as 2 tokens in a sequence and runs multi-head self-attention
    over them. Each token gets to "query" the other; the resulting reprs are
    cross-modally informed before they reach the head.

    The cross-attention block is a single transformer block without FFN
    (~16K params on top of V2's 172K). Residual connection but no LayerNorm
    — kept minimal to isolate the fusion-architecture effect from generic
    transformer-block tuning.

    Head input is still 2P (concat of the two attended tokens) so V2 vs V3
    have an apples-to-apples head comparison; only the fusion mechanism
    differs.
    """

    def __init__(
        self,
        video_feature_dim: int,
        audio_feature_dim: int,
        proj_dim: int = 64,
        head_hidden: int = 64,
        proj_dropout: float = 0.3,
        lstm_dropout: float = 0.2,
        repr_dropout: float = 0.3,
        head_dropout: float = 0.3,
        attn_heads: int = 4,
        attn_dropout: float = 0.2,
    ):
        super().__init__()
        self.video_proj = nn.Sequential(
            nn.Linear(video_feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        self.audio_lstm = nn.LSTM(
            proj_dim, proj_dim, num_layers=2,
            dropout=lstm_dropout, batch_first=True,
        )
        self.video_repr_drop = nn.Dropout(repr_dropout)
        self.audio_repr_drop = nn.Dropout(repr_dropout)

        # Cross-modal attention. embed_dim must be divisible by num_heads.
        if proj_dim % attn_heads != 0:
            raise ValueError(
                f"proj_dim ({proj_dim}) must be divisible by attn_heads ({attn_heads})"
            )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=proj_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(2 * proj_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"MultiModalV3: video_dim={video_feature_dim} -> {proj_dim} (mean-pool), "
            f"audio_dim={audio_feature_dim} -> {proj_dim} (LSTM x2), "
            f"cross-attn(heads={attn_heads}, drop={attn_dropout}), "
            f"head_hidden={head_hidden}, "
            f"dropouts: proj={proj_dropout}/lstm={lstm_dropout}/repr={repr_dropout}/head={head_dropout}, "
            f"params={n_params:,}"
        )

    def forward(self, video_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        v = self.video_proj(video_features)         # (B, 12, P)
        v_repr = v.mean(dim=1)                      # (B, P)
        v_repr = self.video_repr_drop(v_repr)

        a = self.audio_proj(audio_features)         # (B, 49, P)
        _, (h_n, _) = self.audio_lstm(a)            # h_n: (2, B, P)
        a_repr = h_n[-1, :, :]                      # (B, P)
        a_repr = self.audio_repr_drop(a_repr)

        # Stack as 2 tokens, attend, residual back, flatten for the head.
        # Token 0 = video repr, token 1 = audio repr. Self-attention here
        # IS cross-attention because there are only 2 tokens — every token
        # attends to the other and to itself.
        tokens = torch.stack([v_repr, a_repr], dim=1)        # (B, 2, P)
        attended, _ = self.cross_attn(tokens, tokens, tokens)  # (B, 2, P)
        tokens = tokens + attended                            # residual
        fused = tokens.flatten(start_dim=1)                   # (B, 2P)
        return self.head(fused).squeeze(-1)


class MultiModalV4(nn.Module):
    """v4: token-level cross-modal transformer fusion.

    V3 (cross-attention on aggregated reprs) helped at low capacity but
    plateaued — each modality is still summarized into a single vector
    before the modalities can talk. v4 keeps the per-frame/per-step features
    and runs a transformer over the joint sequence: every video frame can
    attend to every audio frame and vice versa.

    For dyadic synchrony, this is structurally aligned with the task: the
    "is the audio at t=3.5s coordinating with the video at t=3.5s" question
    can be answered by attention between specific time-aligned tokens,
    rather than between two pre-aggregated summaries.

    Architecture:
      Video proj:  Linear(768->P) + GELU + Dropout      (B, 12, P)
      Audio proj:  Linear(768->P) + GELU + Dropout      (B, 49, P)
      + learnable modality embeddings (broadcast)
      + learnable positional embeddings (per modality)
      Concat → (B, 61, P) joint sequence
      TransformerEncoder (n_layers, n_heads, FFN, pre-LN)
      Mean-pool over 61 tokens → (B, P)
      Head: Linear(P, head_hidden) + GELU + Dropout + Linear(head_hidden, 1)

    Defaults: n_layers=1, n_heads=4, FFN=4*P, attn_dropout=0.1.
    """

    def __init__(
        self,
        video_feature_dim: int,
        audio_feature_dim: int,
        proj_dim: int = 64,
        head_hidden: int = 64,
        proj_dropout: float = 0.3,
        head_dropout: float = 0.3,
        n_layers: int = 1,
        n_heads: int = 4,
        ffn_dim: int = None,
        attn_dropout: float = 0.1,
        n_video_frames: int = 12,
        n_audio_frames: int = 49,
    ):
        super().__init__()
        if proj_dim % n_heads != 0:
            raise ValueError(
                f"proj_dim ({proj_dim}) must be divisible by n_heads ({n_heads})"
            )
        if ffn_dim is None:
            ffn_dim = 4 * proj_dim

        self.video_proj = nn.Sequential(
            nn.Linear(video_feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )

        # Modality embeddings: 1 learnable vector per modality, broadcast
        # across all positions. Distinguishes "this is a video token" from
        # "this is an audio token" so attention can route by modality.
        self.video_modality_emb = nn.Parameter(torch.zeros(1, 1, proj_dim))
        self.audio_modality_emb = nn.Parameter(torch.zeros(1, 1, proj_dim))
        nn.init.normal_(self.video_modality_emb, std=0.02)
        nn.init.normal_(self.audio_modality_emb, std=0.02)

        # Positional embeddings: per-position-within-modality. Temporal
        # order matters for synchrony detection.
        self.video_pos_emb = nn.Parameter(torch.zeros(1, n_video_frames, proj_dim))
        self.audio_pos_emb = nn.Parameter(torch.zeros(1, n_audio_frames, proj_dim))
        nn.init.normal_(self.video_pos_emb, std=0.02)
        nn.init.normal_(self.audio_pos_emb, std=0.02)

        # Pre-LN transformer block (more stable than post-LN at small scale).
        layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=attn_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(proj_dim)

        self.head = nn.Sequential(
            nn.Linear(proj_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"MultiModalV4: video_dim={video_feature_dim} -> {proj_dim}, "
            f"audio_dim={audio_feature_dim} -> {proj_dim}, "
            f"transformer(n_layers={n_layers}, n_heads={n_heads}, ffn={ffn_dim}, drop={attn_dropout}), "
            f"head_hidden={head_hidden}, "
            f"params={n_params:,}"
        )

    def forward(self, video_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        v = self.video_proj(video_features)        # (B, 12, P)
        v = v + self.video_modality_emb + self.video_pos_emb
        a = self.audio_proj(audio_features)        # (B, 49, P)
        a = a + self.audio_modality_emb + self.audio_pos_emb

        tokens = torch.cat([v, a], dim=1)          # (B, 61, P)
        tokens = self.transformer(tokens)          # (B, 61, P)
        tokens = self.final_norm(tokens)
        pooled = tokens.mean(dim=1)                # (B, P)
        return self.head(pooled).squeeze(-1)


class MultiModalV2Patch(nn.Module):
    """v2 variant: spatial attention over DINOv2 patch tokens per frame.

    Probe 1: tests whether the meanpatch collapse in v2 (which mean-pools the
    257-patch grid into a single 768-dim per frame) is what's capping train_acc
    at ~0.76. v2 with full patch features lets the model learn *which patches*
    matter per frame instead of being fed the spatial average.

    Video: (B, T_v, P, D_v) — e.g. (B, 12, 257, 768) from data/dinov2_features/
      Per-frame spatial attention: a learnable query attends over P patches.
        Q: (1, 1, D_v) learnable, broadcast to (B*T_v, 1, D_v)
        K, V: (B*T_v, P, D_v)
        → (B*T_v, D_v)
      Reshape to (B, T_v, D_v), then standard v2 path:
      Linear(D_v -> proj_dim) + GELU + Dropout
      mean over T_v
      Dropout
      → (B, proj_dim) video_repr

    Audio + Fusion: identical to MultiModalV2.

    The spatial attention is one block with no FFN — kept minimal so the
    only structural difference from v2 is the spatial aggregation strategy
    (learned attention vs. mean pool).
    """

    def __init__(
        self,
        video_feature_dim: int,
        audio_feature_dim: int,
        proj_dim: int = 64,
        head_hidden: int = 64,
        proj_dropout: float = 0.3,
        lstm_dropout: float = 0.2,
        repr_dropout: float = 0.3,
        head_dropout: float = 0.3,
        spatial_attn_heads: int = 4,
        spatial_attn_dropout: float = 0.1,
    ):
        super().__init__()
        if video_feature_dim % spatial_attn_heads != 0:
            raise ValueError(
                f"video_feature_dim ({video_feature_dim}) must be divisible by "
                f"spatial_attn_heads ({spatial_attn_heads})"
            )
        self.video_query = nn.Parameter(torch.randn(1, 1, video_feature_dim) * 0.02)
        self.video_spatial_attn = nn.MultiheadAttention(
            embed_dim=video_feature_dim,
            num_heads=spatial_attn_heads,
            dropout=spatial_attn_dropout,
            batch_first=True,
        )
        self.video_proj = nn.Sequential(
            nn.Linear(video_feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        self.audio_lstm = nn.LSTM(
            proj_dim, proj_dim, num_layers=2,
            dropout=lstm_dropout, batch_first=True,
        )
        self.video_repr_drop = nn.Dropout(repr_dropout)
        self.audio_repr_drop = nn.Dropout(repr_dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * proj_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"MultiModalV2Patch: video=(T, P, {video_feature_dim}) → spatial-attn(heads="
            f"{spatial_attn_heads}, drop={spatial_attn_dropout}) → proj({proj_dim}) → mean-T, "
            f"audio_dim={audio_feature_dim} → {proj_dim} (LSTM x2), "
            f"head_hidden={head_hidden}, "
            f"dropouts: proj={proj_dropout}/lstm={lstm_dropout}/repr={repr_dropout}/head={head_dropout}, "
            f"params={n_params:,}"
        )

    def forward(self, video_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        # video_features: (B, T, P, D)
        B, T, P, D = video_features.shape
        v = video_features.reshape(B * T, P, D)
        q = self.video_query.expand(B * T, -1, -1)            # (B*T, 1, D)
        v_attn, _ = self.video_spatial_attn(q, v, v)          # (B*T, 1, D)
        v = v_attn.squeeze(1).reshape(B, T, D)                # (B, T, D)
        v = self.video_proj(v)                                # (B, T, proj_dim)
        v_repr = v.mean(dim=1)                                # (B, proj_dim)
        v_repr = self.video_repr_drop(v_repr)

        a = self.audio_proj(audio_features)                   # (B, 49, proj_dim)
        _, (h_n, _) = self.audio_lstm(a)                      # h_n: (2, B, proj_dim)
        a_repr = h_n[-1, :, :]
        a_repr = self.audio_repr_drop(a_repr)

        fused = torch.cat([v_repr, a_repr], dim=-1)           # (B, 2*proj_dim)
        return self.head(fused).squeeze(-1)


class MultiModalV2Pose(nn.Module):
    """v2 + pose: 3-pathway fusion (video + audio + pose) for probe 3.

    Tests whether YOLO26-pose keypoints (per-frame 2-person × 17-keypoint
    × (x, y, conf) tensor) carry synchrony signal beyond what DINOv2 +
    WavLM-base-plus already encode. Same v2 video+audio path; pose enters
    as a third aggregated repr concatenated at the fusion stage.

    Video: (B, T_v, D_v) → Linear → GELU → Dropout → mean-T → Dropout
                          → (B, proj_dim)                       [same as v2]
    Audio: (B, T_a, D_a) → Linear → GELU → Dropout → LSTM x2
                          → h_n[-1] → Dropout → (B, proj_dim)   [same as v2]
    Pose:  (B, T_p, 2, 17, 3) → flatten to (B, T_p, 102)
                              → Linear(102 → proj_dim) → GELU → Dropout
                              → LSTM(proj_dim → proj_dim, num_layers=2)
                              → h_n[-1] → Dropout → (B, proj_dim)

    Fusion: concat(v_repr, a_repr, p_repr) = (B, 3*proj_dim)
                → Linear → GELU → Dropout → Linear → logit

    Per-frame 102-dim pose flatten preserves all keypoint info but discards
    the (person, keypoint) structure. A more elaborate per-person attention
    block could come later if pose contributes signal here — this minimal
    LSTM version is the apples-to-apples comparison to how v2 treats the
    other modalities.
    """

    POSE_FRAME_DIM = 2 * 17 * 3  # = 102

    def __init__(
        self,
        video_feature_dim: int,
        audio_feature_dim: int,
        proj_dim: int = 64,
        head_hidden: int = 64,
        proj_dropout: float = 0.3,
        lstm_dropout: float = 0.2,
        repr_dropout: float = 0.3,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.video_proj = nn.Sequential(
            nn.Linear(video_feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        self.audio_lstm = nn.LSTM(
            proj_dim, proj_dim, num_layers=2,
            dropout=lstm_dropout, batch_first=True,
        )
        self.pose_proj = nn.Sequential(
            nn.Linear(self.POSE_FRAME_DIM, proj_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        self.pose_lstm = nn.LSTM(
            proj_dim, proj_dim, num_layers=2,
            dropout=lstm_dropout, batch_first=True,
        )
        self.video_repr_drop = nn.Dropout(repr_dropout)
        self.audio_repr_drop = nn.Dropout(repr_dropout)
        self.pose_repr_drop = nn.Dropout(repr_dropout)
        self.head = nn.Sequential(
            nn.Linear(3 * proj_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"MultiModalV2Pose: video_dim={video_feature_dim} → {proj_dim} (mean-pool), "
            f"audio_dim={audio_feature_dim} → {proj_dim} (LSTM x2), "
            f"pose=(T, 2×17×3={self.POSE_FRAME_DIM}) → {proj_dim} (LSTM x2), "
            f"head_hidden={head_hidden}, "
            f"dropouts: proj={proj_dropout}/lstm={lstm_dropout}/repr={repr_dropout}/head={head_dropout}, "
            f"params={n_params:,}"
        )

    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
        pose_features: torch.Tensor,
    ) -> torch.Tensor:
        # Video
        v = self.video_proj(video_features)              # (B, T_v, proj_dim)
        v_repr = v.mean(dim=1)
        v_repr = self.video_repr_drop(v_repr)

        # Audio
        a = self.audio_proj(audio_features)              # (B, T_a, proj_dim)
        _, (h_n_a, _) = self.audio_lstm(a)
        a_repr = h_n_a[-1, :, :]
        a_repr = self.audio_repr_drop(a_repr)

        # Pose: flatten the (2 people, 17 kpts, 3 channels) per-frame.
        B, T_p = pose_features.shape[0], pose_features.shape[1]
        p = pose_features.reshape(B, T_p, self.POSE_FRAME_DIM)
        p = self.pose_proj(p)                            # (B, T_p, proj_dim)
        _, (h_n_p, _) = self.pose_lstm(p)
        p_repr = h_n_p[-1, :, :]
        p_repr = self.pose_repr_drop(p_repr)

        fused = torch.cat([v_repr, a_repr, p_repr], dim=-1)  # (B, 3*proj_dim)
        return self.head(fused).squeeze(-1)


class MultiModalV5Coupling(nn.Module):
    """v5: per-frame coupling + attention pool — the "synchrony lives in
    sub-second covariation" hypothesis.

    v2/v3/v4 all aggregate each modality before computing cross-modal
    structure (mean-pool for video, LSTM h_n[-1] for audio, then concat/
    cross-attention at the head). Train_acc plateaus at ~0.76 across them.
    The peer-reviewed labeling scheme has IRR substantially above this,
    which implies the signal IS in the features — we're losing it at the
    aggregation step.

    v5 flips the order: align frame timelines first, compute per-frame
    cross-modal coupling, then attention-pool with the coupling scores
    themselves as logits. The inductive bias is explicit:
    "synchrony = high covariation at matched timepoints; the model should
    focus on the frames where that happens."

    Architecture:
      Video (B, 12, D_v) → Linear → GELU → Dropout    → (B, 12, P)
      Audio (B, 49, D_a) → Linear → GELU → Dropout    → (B, 49, P)
                         → adaptive_avg_pool over T   → (B, 12, P)

      Per-frame coupling:
        c[t] = cosine(v[t], a_aligned[t])              # (B, 12)
        attn = softmax(τ · c)                           # learned τ ∈ R

      Pool joint frame-aligned representation:
        joint[t] = concat(v[t], a_aligned[t])           # (B, 12, 2P)
        pooled   = Σ_t attn[t] · joint[t]              # (B, 2P)

      Head: Linear(2P, head_hidden) → GELU → Dropout → Linear → logit

    Notes:
      - Cosine-then-softmax IS the prior. Cosine ∈ [-1, 1] gives a
        relatively flat softmax without temperature; the learned τ
        (init 5.0) sharpens it to whatever the gradient prefers.
      - No LSTM/transformer over time — the model can only weight existing
        frames by their coupling. If the ceiling is structural (the right
        info just isn't in the per-frame features), this won't help. If
        it's aggregation (the info is there but v2 averaged it out), this
        should break the wall.
      - ~17K params at proj_dim=64 — smallest of any probe yet. Capacity
        controlled tightly so any lift comes from the inductive bias,
        not extra parameters.
    """

    def __init__(
        self,
        video_feature_dim: int,
        audio_feature_dim: int,
        n_video_frames: int = 12,
        n_audio_frames: int = 49,
        proj_dim: int = 64,
        head_hidden: int = 64,
        proj_dropout: float = 0.3,
        repr_dropout: float = 0.3,
        head_dropout: float = 0.3,
        attn_temperature_init: float = 5.0,
    ):
        super().__init__()
        self.n_video_frames = n_video_frames
        self.n_audio_frames = n_audio_frames
        self.video_proj = nn.Sequential(
            nn.Linear(video_feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        # Learned softmax temperature applied to cosine coupling scores.
        # τ=5 init keeps softmax informative without saturating early.
        self.attn_temperature = nn.Parameter(
            torch.tensor(float(attn_temperature_init))
        )
        self.repr_drop = nn.Dropout(repr_dropout)
        self.head = nn.Sequential(
            nn.Linear(2 * proj_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"MultiModalV5Coupling: "
            f"video_dim={video_feature_dim}→{proj_dim}, "
            f"audio_dim={audio_feature_dim}→{proj_dim}, "
            f"audio align {n_audio_frames}→{n_video_frames} (adaptive avg pool), "
            f"per-frame cosine coupling + softmax(τ·c) attention pool, "
            f"head_hidden={head_hidden}, "
            f"dropouts: proj={proj_dropout}/repr={repr_dropout}/head={head_dropout}, "
            f"τ_init={attn_temperature_init}, params={n_params:,}"
        )

    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        v = self.video_proj(video_features)              # (B, T_v, P)
        a = self.audio_proj(audio_features)              # (B, T_a, P)

        # Align audio to video frame rate via adaptive avg pool over time.
        # F.adaptive_avg_pool1d expects (B, C, T), so transpose.
        a_aligned = F.adaptive_avg_pool1d(
            a.transpose(1, 2), self.n_video_frames
        ).transpose(1, 2)                                # (B, T_v, P)

        # Per-frame cosine coupling between matched (v, a_aligned) frames.
        v_norm = F.normalize(v, dim=-1)
        a_norm = F.normalize(a_aligned, dim=-1)
        coupling = (v_norm * a_norm).sum(dim=-1)         # (B, T_v)

        # Attention weights: softmax over coupling, sharpened by τ.
        attn = torch.softmax(self.attn_temperature * coupling, dim=-1)

        # Pool the joint frame-aligned repr by coupling-weighted attention.
        joint = torch.cat([v, a_aligned], dim=-1)        # (B, T_v, 2P)
        pooled = (attn.unsqueeze(-1) * joint).sum(dim=1) # (B, 2P)
        pooled = self.repr_drop(pooled)

        return self.head(pooled).squeeze(-1)


class MultiModalV6CrossPerson(nn.Module):
    """v6: per-person video features + symmetric cross-person attention.

    The argument: v2_pose tested "does pose add signal" by stapling pose
    onto the v2 head, but that doesn't model the relation between the two
    people — both pose streams still project then aggregate independently.
    The CrossPersonAttention module (src/synchronai/models/cv/cross_person_attention.py)
    has been sitting in the repo unwired. v6 wires it.

    Video features here are (T, 2 persons, D) — produced by
    build_perperson_video_features.py, which pools DINOv2 patch features
    inside YOLO26-pose-derived bboxes per frame per person. So each person
    has their own visual representation per frame, and the cross-person
    attention layer models how those representations couple.

    Pipeline:
      Video: (B, T_v, 2, D_v)
        → split into A and B: (B, T_v, D_v) each
        → shared projection: Linear → GELU → Dropout → (B, T_v, P) each
        → CrossPersonAttention(layers=1, heads=4): (B, T_v, P) × 2
        → mean over T → per-person reprs (B, P) × 2
        → concat → (B, 2P) video repr

      Audio: (B, T_a, D_a)
        → Linear → GELU → Dropout → LSTM(2 layers) → h_n[-1]
        → (B, P) audio repr  [same as v2]

      Fusion: concat([video_A, video_B, audio]) = (B, 3P)
        → Linear → GELU → Dropout → Linear → logit

    Decision rule (same fold-0 low-reg config as all prior probes):
      - train_acc breaks past 0.85 + val_AUC > 0.82 → cross-person
        structure WAS the missing piece. The relational signal that
        v2-style scene-pooled architectures couldn't access lives in
        explicit per-person comparison.
      - train_acc ≤ 0.78 + val_AUC ≈ 0.78 → cross-person structure
        doesn't unlock new signal either. Combined with the prior
        feature-ceiling pattern, that's strong evidence the ceiling
        is in the label space (IRR-limited) rather than the model.
      - intermediate → partial lift; warrants full CV.
    """

    def __init__(
        self,
        video_feature_dim: int,
        audio_feature_dim: int,
        proj_dim: int = 64,
        head_hidden: int = 64,
        proj_dropout: float = 0.3,
        lstm_dropout: float = 0.2,
        repr_dropout: float = 0.3,
        head_dropout: float = 0.3,
        cross_attn_layers: int = 1,
        cross_attn_heads: int = 4,
        cross_attn_dropout: float = 0.1,
    ):
        super().__init__()
        # Import locally so the trainer module isn't tightly coupled to
        # the package layout — it works whether or not synchronai is
        # pip-installed (cluster uses PYTHONPATH).
        from synchronai.models.cv.cross_person_attention import CrossPersonAttention

        if proj_dim % cross_attn_heads != 0:
            raise ValueError(
                f"proj_dim ({proj_dim}) must be divisible by "
                f"cross_attn_heads ({cross_attn_heads})"
            )

        # Shared per-person projection. One projection, applied to both
        # persons — symmetry by construction.
        self.video_proj = nn.Sequential(
            nn.Linear(video_feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        self.cross_person_attn = CrossPersonAttention(
            embed_dim=proj_dim,
            num_layers=cross_attn_layers,
            num_heads=cross_attn_heads,
            dropout=cross_attn_dropout,
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        self.audio_lstm = nn.LSTM(
            proj_dim, proj_dim, num_layers=2,
            dropout=lstm_dropout, batch_first=True,
        )
        self.video_repr_drop = nn.Dropout(repr_dropout)
        self.audio_repr_drop = nn.Dropout(repr_dropout)
        self.head = nn.Sequential(
            nn.Linear(3 * proj_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"MultiModalV6CrossPerson: video_dim={video_feature_dim}→{proj_dim} (shared per-person proj), "
            f"cross-person attn(layers={cross_attn_layers}, heads={cross_attn_heads}, drop={cross_attn_dropout}), "
            f"audio_dim={audio_feature_dim}→{proj_dim} (LSTM x2), "
            f"head_hidden={head_hidden}, "
            f"dropouts: proj={proj_dropout}/lstm={lstm_dropout}/repr={repr_dropout}/head={head_dropout}, "
            f"params={n_params:,}"
        )

    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        # video_features: (B, T_v, 2 persons, D_v)
        # Split persons. Shared projection so the model can't tell A from
        # B from the projection itself — only via the keypoint-derived
        # crop choice.
        person_a = video_features[:, :, 0, :]    # (B, T_v, D_v)
        person_b = video_features[:, :, 1, :]    # (B, T_v, D_v)
        a = self.video_proj(person_a)            # (B, T_v, P)
        b = self.video_proj(person_b)            # (B, T_v, P)

        # Symmetric cross-person attention. a_attn = A queried against B
        # (and vice versa). Output shape: (B, T_v, P) per person.
        a_attn, b_attn = self.cross_person_attn(a, b)

        # Per-person temporal aggregation: mean over T. Matches v2's
        # video-side aggregation (mean-pool) so the lift vs v2 is
        # attributable to per-person + cross-attn, not to swapping the
        # aggregator.
        a_repr = a_attn.mean(dim=1)              # (B, P)
        b_repr = b_attn.mean(dim=1)              # (B, P)
        a_repr = self.video_repr_drop(a_repr)
        b_repr = self.video_repr_drop(b_repr)

        # Audio path: identical to v2.
        a_audio = self.audio_proj(audio_features)        # (B, T_a, P)
        _, (h_n, _) = self.audio_lstm(a_audio)
        audio_repr = h_n[-1, :, :]
        audio_repr = self.audio_repr_drop(audio_repr)

        # Fuse all three reprs.
        fused = torch.cat([a_repr, b_repr, audio_repr], dim=-1)  # (B, 3P)
        return self.head(fused).squeeze(-1)


class MultiModalV7TemporalContext(nn.Module):
    """v7: per-second v2 aggregation + cross-second sequence model.

    Synchrony is a multi-second construct (5-15s of lead-lag / mirroring /
    turn-taking dynamics). v2/v3/v4/v6 all predict each second from a strict
    1-second window — no across-second context. v7 keeps the per-second
    label granularity but gives the model a (2k+1)-second receptive field
    by sequence-modeling over k-padded context windows of per-second
    representations.

    Pipeline (for context_window=k, so S=2k+1 seconds total):
      For each of S context seconds, apply v2's video mean-pool and audio
      LSTM in parallel (shared weights across seconds), producing one
      2P-dim per-second representation.
      → (B, S, 2P)
      Run a 1-layer LSTM over the S per-second reprs → (B, S, 2P).
      Take the CENTER second's output → (B, 2P).
      MLP head → logit.

    Decision rule (after fold 0 + maybe CV):
      - val_AUC > 0.78 → multi-second context unlocks signal that single-
        second models missed. Worth scaling context window (try k=3, 5).
      - val_AUC ≈ 0.72 → context doesn't help; label is more local than we
        thought, or the per-second features are too smoothed.
      - train_acc > 0.85 + val val_AUC ≈ 0.72 → similar pattern to v6:
        architecture extracts more signal but doesn't generalize at n=49.
    """

    def __init__(
        self,
        video_feature_dim: int,
        audio_feature_dim: int,
        context_window: int = 2,
        proj_dim: int = 64,
        head_hidden: int = 64,
        proj_dropout: float = 0.3,
        lstm_dropout: float = 0.2,
        repr_dropout: float = 0.3,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.context_window = context_window
        self.context_size = 2 * context_window + 1

        # Per-second v2 pipeline (shared across context seconds).
        self.video_proj = nn.Sequential(
            nn.Linear(video_feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        self.audio_lstm = nn.LSTM(
            proj_dim, proj_dim, num_layers=2,
            dropout=lstm_dropout, batch_first=True,
        )

        # Cross-second sequence model. 1-layer LSTM over per-second reprs.
        # Input dim 2P (video repr + audio repr), output dim 2P.
        self.context_lstm = nn.LSTM(
            2 * proj_dim, 2 * proj_dim, num_layers=1,
            batch_first=True,
        )

        self.video_repr_drop = nn.Dropout(repr_dropout)
        self.audio_repr_drop = nn.Dropout(repr_dropout)
        self.context_repr_drop = nn.Dropout(repr_dropout)

        self.head = nn.Sequential(
            nn.Linear(2 * proj_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"MultiModalV7TemporalContext: "
            f"video_dim={video_feature_dim}→{proj_dim} (mean-pool/sec), "
            f"audio_dim={audio_feature_dim}→{proj_dim} (LSTM x2/sec), "
            f"context_window={context_window} ({self.context_size}s receptive field), "
            f"context_lstm({2*proj_dim}→{2*proj_dim}, 1 layer), "
            f"head_hidden={head_hidden}, "
            f"params={n_params:,}"
        )

    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        # video_features: (B, S, T_v, D_v)
        # audio_features: (B, S, T_a, D_a)
        B, S = video_features.shape[:2]

        # Flatten seconds into the batch dim so we run v2's per-second
        # pipeline on (B*S, T, D) — shared weights across seconds.
        v_flat = video_features.reshape(B * S, *video_features.shape[2:])
        a_flat = audio_features.reshape(B * S, *audio_features.shape[2:])

        v = self.video_proj(v_flat)              # (B*S, T_v, P)
        v_repr = v.mean(dim=1)                   # (B*S, P) — v2-style mean-pool
        v_repr = self.video_repr_drop(v_repr)

        a = self.audio_proj(a_flat)              # (B*S, T_a, P)
        _, (h_n, _) = self.audio_lstm(a)
        a_repr = h_n[-1, :, :]                   # (B*S, P) — v2-style LSTM h_n
        a_repr = self.audio_repr_drop(a_repr)

        # Per-second concatenated repr → reshape to sequence.
        per_sec = torch.cat([v_repr, a_repr], dim=-1)  # (B*S, 2P)
        per_sec = per_sec.view(B, S, -1)               # (B, S, 2P)

        # Cross-second sequence model.
        context_out, _ = self.context_lstm(per_sec)    # (B, S, 2P)

        # Extract the CENTER second's representation. That's the second
        # the label belongs to — surrounding seconds are context.
        center_idx = self.context_window
        center_repr = context_out[:, center_idx, :]    # (B, 2P)
        center_repr = self.context_repr_drop(center_repr)

        return self.head(center_repr).squeeze(-1)


class _GradientReversal(torch.autograd.Function):
    """Identity in forward, sign-flipped scaled gradient in backward.

    Used by MultiModalV8SubjectInvariant to push the feature extractor
    AWAY from subject-discriminative representations: the subject head
    learns to predict subject identity, but its gradient is reversed
    before flowing back into the shared encoder, forcing the encoder
    to forget subject-identifying information.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def _grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, lambda_)


class MultiModalV8SubjectInvariant(nn.Module):
    """v8: v6_crossperson + subject-invariant adversarial regularization.

    v6_crossperson hit train_acc 0.84 with a 12.5-point train-val gap on
    CV — the model learns subject-specific cross-person patterns that
    don't generalize. v8 adds a domain-adversarial subject classifier
    head with gradient reversal: the subject head tries to predict
    subject identity, but its gradient is reversed before the shared
    encoder, forcing the encoder to lose subject-identifying information.

    Pipeline:
      Video (B, T, 2, D_v), Audio (B, T_a, D_a) → v6 fused repr (B, 3P)
      → synchrony_head(fused) → sync_logit
      → grad_reverse(fused, λ) → subject_head → subject_logits

    Training combines:
      loss = bce(sync_logit, label) + λ * ce(subject_logits, subject_id)

    The gradient-reversal layer makes the second term's gradient flow
    NEGATIVELY through the encoder — so the encoder is being trained to
    SIMULTANEOUSLY fit synchrony AND lose subject identifiability.

    λ (lambda_subj) controls the strength of subject invariance. λ=0
    reduces to v6_crossperson exactly. Default λ=0.1.

    n_subjects_train is the number of distinct subjects in the train
    set — required for the subject classifier head's output dimension.
    Must be passed at construction time.
    """

    def __init__(
        self,
        video_feature_dim: int,
        audio_feature_dim: int,
        n_subjects_train: int,
        proj_dim: int = 64,
        head_hidden: int = 64,
        proj_dropout: float = 0.3,
        lstm_dropout: float = 0.2,
        repr_dropout: float = 0.3,
        head_dropout: float = 0.3,
        cross_attn_layers: int = 1,
        cross_attn_heads: int = 4,
        cross_attn_dropout: float = 0.1,
        lambda_subj: float = 0.1,
    ):
        super().__init__()
        from synchronai.models.cv.cross_person_attention import CrossPersonAttention

        if proj_dim % cross_attn_heads != 0:
            raise ValueError(
                f"proj_dim ({proj_dim}) must be divisible by cross_attn_heads "
                f"({cross_attn_heads})"
            )
        self.lambda_subj = lambda_subj

        # v6's shared per-person projection, cross-person attn, audio path.
        self.video_proj = nn.Sequential(
            nn.Linear(video_feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        self.cross_person_attn = CrossPersonAttention(
            embed_dim=proj_dim,
            num_layers=cross_attn_layers,
            num_heads=cross_attn_heads,
            dropout=cross_attn_dropout,
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_feature_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
        )
        self.audio_lstm = nn.LSTM(
            proj_dim, proj_dim, num_layers=2,
            dropout=lstm_dropout, batch_first=True,
        )
        self.video_repr_drop = nn.Dropout(repr_dropout)
        self.audio_repr_drop = nn.Dropout(repr_dropout)

        # Synchrony prediction head (same shape as v6's).
        self.head = nn.Sequential(
            nn.Linear(3 * proj_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

        # Adversarial subject classifier head. Modest capacity — we want
        # it to be CAPABLE of identifying subjects (to provide a useful
        # adversarial gradient), but not so large it dominates training.
        self.subject_head = nn.Sequential(
            nn.Linear(3 * proj_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, n_subjects_train),
        )

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"MultiModalV8SubjectInvariant: "
            f"video_dim={video_feature_dim}→{proj_dim}, "
            f"audio_dim={audio_feature_dim}→{proj_dim}, "
            f"cross-person attn (layers={cross_attn_layers}, heads={cross_attn_heads}), "
            f"subject classifier({3*proj_dim}→{n_subjects_train}), "
            f"λ_subj={lambda_subj}, params={n_params:,}"
        )

    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor,
        subject_id: torch.Tensor | None = None,
    ):
        """Returns sync_logit; also subject_logits if subject_id is provided.

        At training time, pass subject_id (integer indices) to get both
        outputs and compute the dual loss. At val/test time, omit it.
        """
        person_a = video_features[:, :, 0, :]
        person_b = video_features[:, :, 1, :]
        a = self.video_proj(person_a)
        b = self.video_proj(person_b)
        a_attn, b_attn = self.cross_person_attn(a, b)
        a_repr = self.video_repr_drop(a_attn.mean(dim=1))
        b_repr = self.video_repr_drop(b_attn.mean(dim=1))

        a_audio = self.audio_proj(audio_features)
        _, (h_n, _) = self.audio_lstm(a_audio)
        audio_repr = self.audio_repr_drop(h_n[-1, :, :])

        fused = torch.cat([a_repr, b_repr, audio_repr], dim=-1)  # (B, 3P)
        sync_logit = self.head(fused).squeeze(-1)

        if subject_id is None:
            return sync_logit

        # Subject prediction with gradient reversal — encoder sees a
        # NEGATIVE gradient from this branch.
        fused_reversed = _grad_reverse(fused, self.lambda_subj)
        subject_logits = self.subject_head(fused_reversed)
        return sync_logit, subject_logits


def merge_feature_indices(
    video_feature_dir: Path,
    audio_feature_dir: Path,
    pose_feature_dir: Path | None = None,
) -> tuple[pd.DataFrame, int, int, int, int]:
    """Inner-join the feature_index.csv files on (video_path, second).

    When pose_feature_dir is provided, also joins pose features in a third
    inner merge — the result row carries video_feature_file, audio_feature_file,
    and pose_feature_file columns. Returns the merged DataFrame plus the feature
    shapes for video and audio (pose shape is fixed at (T, 2, 17, 3)).
    """
    video_idx = pd.read_csv(video_feature_dir / "feature_index.csv")
    audio_idx = pd.read_csv(audio_feature_dir / "feature_index.csv")

    # Audio CSV has 'second' as float ("900.0"); coerce both sides to int.
    video_idx["second"] = video_idx["second"].astype(int)
    audio_idx["second"] = audio_idx["second"].astype(float).astype(int)

    video_dim = int(video_idx["feature_dim"].iloc[0])
    video_frames = int(video_idx["n_frames"].iloc[0])
    audio_dim = int(audio_idx["feature_dim"].iloc[0])
    audio_frames = int(audio_idx["n_frames"].iloc[0])

    merged = video_idx.merge(
        audio_idx[["video_path", "second", "feature_file"]],
        on=["video_path", "second"],
        how="inner",
        suffixes=("_video", "_audio"),
    )
    merged = merged.rename(
        columns={
            "feature_file_video": "video_feature_file",
            "feature_file_audio": "audio_feature_file",
        }
    )

    n_dropped_v = len(video_idx) - len(merged)
    n_dropped_a = len(audio_idx) - len(merged)
    logger.info(
        f"Joined video+audio indices: {len(merged)} samples "
        f"(dropped {n_dropped_v} video-only, {n_dropped_a} audio-only)"
    )

    if pose_feature_dir is not None:
        pose_idx = pd.read_csv(pose_feature_dir / "feature_index.csv")
        pose_idx["second"] = pose_idx["second"].astype(float).astype(int)
        before = len(merged)
        merged = merged.merge(
            pose_idx[["video_path", "second", "feature_file"]].rename(
                columns={"feature_file": "pose_feature_file"}
            ),
            on=["video_path", "second"],
            how="inner",
        )
        n_dropped_after_pose = before - len(merged)
        logger.info(
            f"Joined pose: {len(merged)} samples "
            f"(dropped {n_dropped_after_pose} with no matching pose row)"
        )

    return merged, video_dim, video_frames, audio_dim, audio_frames


def filter_entries_by_fallback(
    entries: list[dict],
    fallback_csv: Path,
    max_fallback_frames: int,
    mode: str = "segment",
    max_subject_fallback_rate: float = 0.80,
) -> list[dict]:
    """Drop entries flagged as bogus by the per-person fallback diagnostic.

    Two modes:
      - "segment" (default): drop individual segments where either slot's
        fallback count exceeds max_fallback_frames. Pros: keeps every
        recording with some signal. Cons: for partially-broken recordings,
        only EARLY-session segments survive — biases content distribution.
      - "subject": compute per-(video_path, subject, session) mean fallback
        rate; drop ALL segments from recordings above
        max_subject_fallback_rate. Pros: preserves content distribution
        within remaining recordings. Cons: more data dropped per
        fully-broken dyad.

    The fallback CSV is produced by build_perperson_video_features.py.
    """
    import csv as _csv
    from collections import defaultdict as _dd
    if mode not in ("segment", "subject"):
        raise ValueError(f"--fallback-mode must be 'segment' or 'subject', got {mode!r}")

    bogus_segments = set()  # for segment mode
    rec_totals = _dd(lambda: {"frames": 0, "fb": 0})  # for subject mode
    with open(fallback_csv) as f:
        for r in _csv.DictReader(f):
            vp = r["video_path"]
            sec = int(float(r["second"]))
            n_fb = int(r["n_fallback_frames"])
            if n_fb > max_fallback_frames:
                bogus_segments.add((vp, sec))
            rec_key = (vp, r.get("subject_id", ""), r.get("session", ""))
            rec_totals[rec_key]["frames"] += int(r["n_total_frames"])
            rec_totals[rec_key]["fb"] += n_fb

    if mode == "subject":
        bogus_recordings = set()
        for rec_key, d in rec_totals.items():
            rate = d["fb"] / d["frames"] if d["frames"] > 0 else 0
            if rate > max_subject_fallback_rate:
                bogus_recordings.add(rec_key)
        n_in = len(entries)
        kept = [
            e for e in entries
            if (e["video_path"], e.get("subject_id", ""), e.get("session", ""))
            not in bogus_recordings
        ]
        n_dropped = n_in - len(kept)
        logger.info(
            f"Fallback filter (subject mode, drop recordings with "
            f">{max_subject_fallback_rate*100:.0f}% bogus rate): "
            f"{len(bogus_recordings)} recordings dropped, "
            f"{n_in} entries in, {len(kept)} kept, {n_dropped} dropped "
            f"({n_dropped/n_in*100:.1f}%). Source: {fallback_csv}"
        )
    else:
        n_in = len(entries)
        kept = [
            e for e in entries
            if (e["video_path"], int(e["second"])) not in bogus_segments
        ]
        n_dropped = n_in - len(kept)
        logger.info(
            f"Fallback filter (segment mode, >{max_fallback_frames} fb frames in any slot): "
            f"{n_in} entries in, {len(kept)} kept, {n_dropped} dropped "
            f"({n_dropped/n_in*100:.1f}%). Source: {fallback_csv}"
        )
    return kept


def _group_value(entry: dict, group_key: str) -> str:
    """Resolve the leakage-prevention grouping key for one entry.

    Prefers an explicit column on the entry; when group_key is 'family_id' and
    no such column exists (the multimodal feature_index only carries
    subject_id), it is derived from subject_id so that both members of a dyad
    (e.g. CARE parent 1102 + child 11021) land in the same partition. See
    family_id_from_subject.
    """
    val = entry.get(group_key)
    if val not in (None, ""):
        return str(val)
    if group_key == "family_id":
        return family_id_from_subject(entry["subject_id"])
    return str(entry[group_key])


def subject_grouped_split(
    entries: list[dict],
    val_split: float,
    seed: int,
    group_key: str = "subject_id",
) -> tuple[list[dict], list[dict]]:
    """80/20 split grouped by group_key to prevent leakage across train/val.

    Default group_key='subject_id'; pass 'family_id' to keep every member of a
    dyad/family on one side of the split (required for CARE, where parent and
    child have distinct subject_ids sharing a family prefix).
    """
    rng = np.random.default_rng(seed)
    by_group = defaultdict(list)
    for e in entries:
        by_group[_group_value(e, group_key)].append(e)

    groups = sorted(by_group.keys())  # deterministic order before shuffle
    rng.shuffle(groups)
    n_val = max(1, int(len(groups) * val_split))
    val_groups = set(groups[:n_val])

    train, val = [], []
    for g, group in by_group.items():
        (val if g in val_groups else train).extend(group)
    logger.info(
        f"Split by {group_key}: {len(train)} train ({len(groups) - n_val} {group_key}s), "
        f"{len(val)} val ({n_val} {group_key}s)"
    )
    return train, val


def subject_kfold_split(
    entries: list[dict],
    num_folds: int,
    fold_idx: int,
    seed: int,
    group_key: str = "subject_id",
) -> tuple[list[dict], list[dict]]:
    """Deterministic k-fold split grouped by group_key.

    For tightening confidence intervals on a single point estimate, k-fold CV
    across the same grouping units gives disjoint val sets and lets us report
    mean ± SE instead of a single-fold number.

    Groups are sorted (deterministic order) then shuffled with the same seed
    across folds. Groups in fold k are the [k*fold_size:(k+1)*fold_size] slice
    of the shuffled list. Same seed → same fold composition. Pass
    group_key='family_id' to keep dyad members together (no CARE leakage).
    """
    if num_folds < 2 or fold_idx < 0 or fold_idx >= num_folds:
        raise ValueError(
            f"Invalid fold args: num_folds={num_folds}, fold_idx={fold_idx}"
        )
    rng = np.random.default_rng(seed)
    by_group = defaultdict(list)
    for e in entries:
        by_group[_group_value(e, group_key)].append(e)

    groups = sorted(by_group.keys())  # deterministic order before shuffle
    rng.shuffle(groups)

    fold_size = len(groups) // num_folds
    val_start = fold_idx * fold_size
    val_end = val_start + fold_size if fold_idx < num_folds - 1 else len(groups)
    val_groups = set(groups[val_start:val_end])

    train, val = [], []
    for g, group in by_group.items():
        (val if g in val_groups else train).extend(group)
    logger.info(
        f"K-fold split by {group_key} (fold {fold_idx+1}/{num_folds}, seed={seed}): "
        f"{len(train)} train ({len(groups) - len(val_groups)} {group_key}s), "
        f"{len(val)} val ({len(val_groups)} {group_key}s). "
        f"Val {group_key}s: {sorted(val_groups)}"
    )
    return train, val


def compute_pos_weight(entries: list[dict]) -> float:
    """pos_weight for BCEWithLogitsLoss to handle class imbalance."""
    labels = [int(e["label"]) for e in entries]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 1.0
    return max(0.5, min(2.0, n_neg / n_pos))


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    accuracy = (preds == labels).float().mean().item()

    auc = 0.5
    if len(set(labels.tolist())) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(labels.numpy(), probs.numpy()))
        except ImportError:
            pass

    tp = ((preds == 1) & (labels == 1)).float().sum()
    fp = ((preds == 1) & (labels == 0)).float().sum()
    fn = ((preds == 0) & (labels == 1)).float().sum()
    f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
    return {"accuracy": accuracy, "auc": auc, "f1": f1}


def family_id_from_subject(subject_id: str) -> str:
    """Dyad/family key for cluster-level CIs and leak-free splits.

    Mirrors the subject_id conventions in data/metadata/studies.py so both
    members of a dyad map to one family:
      - R56 ids are "{family}-C" / "{family}-P" / "{family}-dyad" (dash suffix)
        -> strip the last "-" component.
      - R01 / P-CAT DB-DOS ids are "{family}_C" / "{family}_P" (underscore)
        -> strip the "_C" / "_P" suffix.
      - CARE ids are digit strings; the child is 5 digits and the adult 4,
        sharing the first 4 (e.g. 11021 child + 1102 adult -> family 1102).
      - anything else (EmoGrow, already family-level, unknown) -> the id itself,
        i.e. graceful fall-back to subject-level grouping (no worse than before).
    """
    s = str(subject_id)
    if "-" in s:                       # R56: {family}-{C|P|dyad}
        return s.rsplit("-", 1)[0]
    if s.endswith("_C") or s.endswith("_P"):   # R01 / P-CAT: {family}_{C|P}
        return s[:-2]
    if s.isdigit() and len(s) == 5:    # CARE child (5-digit) -> 4-digit family
        return s[:4]
    return s


def plot_history(history: dict, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history["train_losses"]) + 1)
    best = history["best_epoch"] + 1

    axes[0, 0].plot(epochs, history["train_losses"], "b-", label="Train", linewidth=2)
    axes[0, 0].plot(epochs, history["val_losses"], "r-", label="Val", linewidth=2)
    axes[0, 0].axvline(x=best, color="g", linestyle="--", alpha=0.7, label=f"Best (ep {best})")
    axes[0, 0].set(xlabel="Epoch", ylabel="Loss", title="Loss")
    axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history["train_accs"], "b-", label="Train", linewidth=2)
    axes[0, 1].plot(epochs, history["val_accs"], "r-", label="Val", linewidth=2)
    axes[0, 1].axvline(x=best, color="g", linestyle="--", alpha=0.7)
    axes[0, 1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy")
    axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history["val_aucs"], "m-", linewidth=2)
    axes[1, 0].axvline(x=best, color="g", linestyle="--", alpha=0.7)
    axes[1, 0].axhline(y=history["best_val_auc"], color="g", linestyle=":", alpha=0.5,
                       label=f"Best: {history['best_val_auc']:.4f}")
    axes[1, 0].set(xlabel="Epoch", ylabel="AUC", title="Validation AUC")
    axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history["learning_rates"], "g-", linewidth=2)
    axes[1, 1].set(xlabel="Epoch", ylabel="LR", title="Learning Rate", yscale="log")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Multi-Modal Feature Training (DINOv2 + WavLM)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def train(
    video_feature_dir: str,
    audio_feature_dir: str,
    save_dir: str,
    arch: str = "v1",
    video_hidden: int = 64,
    audio_hidden: int = 64,
    head_hidden: int = 64,
    dropout: float = 0.3,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-3,
    warmup_epochs: int = 3,
    patience: int = 15,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
    early_stop_metric: str = "val_auc",
    num_folds: int = 0,
    fold_idx: int = -1,
    group_key: str = "family_id",
    train_subsample_families: int = 0,
    modality: str = "both",
    video_dropout_prob: float = 0.0,
    audio_dropout_prob: float = 0.0,
    audio_shuffle_prob: float = 0.0,
    pose_feature_dir: str | None = None,
    fallback_csv: str | None = None,
    max_fallback_frames: int = 11,
    fallback_mode: str = "segment",
    max_subject_fallback_rate: float = 0.80,
    context_window: int = 2,
    lambda_subj: float = 0.1,
    input_norm: bool = False,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    use_pose = arch == "v2_pose"
    if use_pose and not pose_feature_dir:
        raise ValueError("--arch v2_pose requires --pose-feature-dir.")
    if pose_feature_dir and not use_pose:
        logger.warning(
            f"--pose-feature-dir provided but --arch={arch!r} doesn't consume it; ignoring."
        )

    video_dir = Path(video_feature_dir)
    audio_dir = Path(audio_feature_dir)
    pose_dir = Path(pose_feature_dir) if use_pose else None
    merged, video_dim, video_frames, audio_dim, audio_frames = merge_feature_indices(
        video_dir, audio_dir, pose_feature_dir=pose_dir,
    )
    logger.info(
        f"Video features: dim={video_dim}, frames={video_frames}. "
        f"Audio features: dim={audio_dim}, frames={audio_frames}."
    )

    entries = merged.to_dict("records")
    # Optional sample-level filter using a fallback-per-segment diagnostic.
    # Drops entries whose underlying video frame didn't exist (12/12 CLS
    # fallback in either slot means the labels.csv referenced a frame
    # past the actual video duration — a labels-side data bug discovered
    # in the v6_crossperson investigation). For other arches the filter
    # is still informative since the bogus segments are scene-level CLS
    # tokens regardless of architecture.
    if fallback_csv:
        entries = filter_entries_by_fallback(
            entries, Path(fallback_csv), max_fallback_frames,
            mode=fallback_mode,
            max_subject_fallback_rate=max_subject_fallback_rate,
        )
    if num_folds > 0 and fold_idx >= 0:
        train_entries, val_entries = subject_kfold_split(
            entries, num_folds, fold_idx, seed, group_key=group_key)
    else:
        train_entries, val_entries = subject_grouped_split(
            entries, val_split, seed, group_key=group_key)
    # Data-scaling curve: keep only N randomly-chosen TRAIN families (val
    # untouched, so every scale point is scored on the same held-out set).
    # N >= available train families is a no-op (the full-data point).
    if train_subsample_families and train_subsample_families > 0:
        by_fam = defaultdict(list)
        for e in train_entries:
            by_fam[_group_value(e, group_key)].append(e)
        fams = sorted(by_fam.keys())
        # distinct RNG stream from the split so the subset is reproducible but
        # not correlated with which families landed in val.
        rng = np.random.default_rng(seed + 90210)
        rng.shuffle(fams)
        keep = set(fams[:train_subsample_families])
        train_entries = [e for e in train_entries if _group_value(e, group_key) in keep]
        logger.info(
            f"Train subsample: kept {len(keep)}/{len(fams)} {group_key}s "
            f"({len(train_entries)} entries) for the data-scaling curve")
    pos_weight = compute_pos_weight(train_entries)
    logger.info(f"Train pos_weight: {pos_weight:.3f}")

    # Probe 1 uses --arch v2_patch with 3D video features (T, P, D) too big to
    # preload; switch to lazy-load dataset for that arch. Probe 2's audio shuffle
    # only applies to the train loader (val keeps real labels). Probe 3 uses
    # --arch v2_pose with a third (pose) feature dir. Probe 6 uses
    # --arch v6_crossperson with per-person 3D video features (T, 2, D).
    if arch == "v2_patch":
        train_dataset = MultiModalPatchFeatureDataset(video_dir, audio_dir, train_entries)
        val_dataset = MultiModalPatchFeatureDataset(video_dir, audio_dir, val_entries)
        if audio_shuffle_prob > 0:
            raise ValueError(
                "--audio-shuffle-prob is not yet wired to MultiModalPatchFeatureDataset. "
                "Run probe 1 (--arch v2_patch) and probe 2 (--audio-shuffle-prob) separately."
            )
    elif arch == "v6_crossperson":
        train_dataset = MultiModalCrossPersonFeatureDataset(
            video_dir, audio_dir, train_entries,
        )
        val_dataset = MultiModalCrossPersonFeatureDataset(
            video_dir, audio_dir, val_entries,
        )
        if audio_shuffle_prob > 0:
            raise ValueError(
                "--audio-shuffle-prob is not wired to MultiModalCrossPersonFeatureDataset."
            )
    elif arch == "v8_subjinv":
        # Build subject_id → int vocab from TRAIN subjects only. Val
        # subjects get -1 (the train loop ignores them for the subject
        # head loss, but val itself doesn't use that head anyway).
        train_subjects = sorted({e["subject_id"] for e in train_entries})
        subject_id_to_int = {s: i for i, s in enumerate(train_subjects)}
        train_dataset = MultiModalCrossPersonFeatureDataset(
            video_dir, audio_dir, train_entries,
            subject_id_to_int=subject_id_to_int,
        )
        val_dataset = MultiModalCrossPersonFeatureDataset(
            video_dir, audio_dir, val_entries,
            subject_id_to_int=subject_id_to_int,  # gives -1 for val subjects
        )
        if audio_shuffle_prob > 0:
            raise ValueError(
                "--audio-shuffle-prob is not wired to MultiModalCrossPersonFeatureDataset."
            )
        logger.info(f"v8_subjinv: built subject vocab with {len(train_subjects)} train subjects.")
    elif arch == "v2_subjinv":
        # Same train-only subject vocab as v8, but on the STANDARD 2D dataset:
        # this variant runs on the mean-pooled features + v2 baseline, isolating
        # the subject-invariance lever from v6's per-person feature quality.
        train_subjects = sorted({e["subject_id"] for e in train_entries})
        subject_id_to_int = {s: i for i, s in enumerate(train_subjects)}
        train_dataset = MultiModalFeatureDataset(
            video_dir, audio_dir, train_entries,
            audio_shuffle_prob=audio_shuffle_prob,
            subject_id_to_int=subject_id_to_int,
        )
        val_dataset = MultiModalFeatureDataset(
            video_dir, audio_dir, val_entries,
            subject_id_to_int=subject_id_to_int,  # gives -1 for val subjects
        )
        logger.info(f"v2_subjinv: built subject vocab with {len(train_subjects)} train subjects.")
    elif arch == "v7_temporal_context":
        train_dataset = MultiModalContextFeatureDataset(
            video_dir, audio_dir, train_entries, context_window=context_window,
        )
        val_dataset = MultiModalContextFeatureDataset(
            video_dir, audio_dir, val_entries, context_window=context_window,
        )
        if audio_shuffle_prob > 0:
            raise ValueError(
                "--audio-shuffle-prob is not wired to MultiModalContextFeatureDataset."
            )
    elif use_pose:
        train_dataset = MultiModalPoseFeatureDataset(
            video_dir, audio_dir, pose_dir, train_entries,
        )
        val_dataset = MultiModalPoseFeatureDataset(
            video_dir, audio_dir, pose_dir, val_entries,
        )
        if audio_shuffle_prob > 0:
            raise ValueError(
                "--audio-shuffle-prob is not wired to MultiModalPoseFeatureDataset."
            )
    else:
        train_dataset = MultiModalFeatureDataset(
            video_dir, audio_dir, train_entries,
            audio_shuffle_prob=audio_shuffle_prob,
        )
        val_dataset = MultiModalFeatureDataset(
            video_dir, audio_dir, val_entries,
            audio_shuffle_prob=0.0,  # never shuffle val
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate,
    )
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    if arch == "v1":
        model = MultiModalLSTMConcat(
            video_feature_dim=video_dim,
            audio_feature_dim=audio_dim,
            video_hidden=video_hidden,
            audio_hidden=audio_hidden,
            head_hidden=head_hidden,
            dropout=dropout,
        ).to(device)
    elif arch == "v2":
        # v2 uses proj_dim = video_hidden (BSub passes the hidden width through
        # this knob; default 64). dropout (CLI) becomes both proj and repr
        # dropout; head_dropout matches; lstm inter-layer dropout is fixed at
        # 0.2. v2 sweep variants like v2_higher_capacity bump video_hidden=128.
        model = MultiModalV2(
            video_feature_dim=video_dim,
            audio_feature_dim=audio_dim,
            proj_dim=video_hidden,
            head_hidden=head_hidden,
            proj_dropout=dropout,
            lstm_dropout=0.2,
            repr_dropout=dropout,
            head_dropout=dropout,
            input_norm=input_norm,
        ).to(device)
    elif arch == "v3":
        # v3: same per-modality pipeline as v2; replaces concat fusion with
        # cross-modal multihead self-attention over the 2 aggregated tokens.
        # attn_heads=4 fixed; proj_dim=video_hidden must be divisible by 4
        # (so video_hidden ∈ {32, 64, 128} all work for the v2 sweep matrix).
        model = MultiModalV3(
            video_feature_dim=video_dim,
            audio_feature_dim=audio_dim,
            proj_dim=video_hidden,
            head_hidden=head_hidden,
            proj_dropout=dropout,
            lstm_dropout=0.2,
            repr_dropout=dropout,
            head_dropout=dropout,
            attn_heads=4,
            attn_dropout=0.2,
        ).to(device)
    elif arch == "v4":
        # v4: token-level cross-modal transformer fusion. Each video frame
        # and audio frame becomes a token; transformer over the joint 61-token
        # sequence with modality + positional embeddings. Heaviest fusion in
        # the lineup but most aligned with what synchrony detection needs.
        model = MultiModalV4(
            video_feature_dim=video_dim,
            audio_feature_dim=audio_dim,
            proj_dim=video_hidden,
            head_hidden=head_hidden,
            proj_dropout=dropout,
            head_dropout=dropout,
            n_layers=1,
            n_heads=4,
            attn_dropout=0.1,
            n_video_frames=video_frames,
            n_audio_frames=audio_frames,
        ).to(device)
    elif arch == "v2_patch":
        # Probe 1: v2 with spatial attention over DINOv2 patch tokens per frame.
        # Requires 3D video features (T, P, D), e.g. data/dinov2_features_patches/.
        model = MultiModalV2Patch(
            video_feature_dim=video_dim,
            audio_feature_dim=audio_dim,
            proj_dim=video_hidden,
            head_hidden=head_hidden,
            proj_dropout=dropout,
            lstm_dropout=0.2,
            repr_dropout=dropout,
            head_dropout=dropout,
            spatial_attn_heads=4,
            spatial_attn_dropout=0.1,
        ).to(device)
    elif arch == "v2_pose":
        # Probe 3: v2 + pose pathway (YOLO26 keypoints) as a third modality.
        # Same v2 video+audio path; pose enters as a 102-dim/frame stream
        # (2 people × 17 kpts × 3 channels) → Linear → LSTM → repr.
        model = MultiModalV2Pose(
            video_feature_dim=video_dim,
            audio_feature_dim=audio_dim,
            proj_dim=video_hidden,
            head_hidden=head_hidden,
            proj_dropout=dropout,
            lstm_dropout=0.2,
            repr_dropout=dropout,
            head_dropout=dropout,
        ).to(device)
    elif arch == "v5_coupling":
        # Probe 5: per-frame cross-modal coupling + attention pool.
        # Tests the "synchrony = sub-second covariation" hypothesis. Audio
        # is aligned to video frame rate (49→12 via adaptive avg pool); per
        # frame, the cosine similarity between modalities becomes a softmax
        # attention weight; the joint repr is pooled by that attention. No
        # LSTM/transformer over time — coupling itself is the temporal
        # localization signal.
        model = MultiModalV5Coupling(
            video_feature_dim=video_dim,
            audio_feature_dim=audio_dim,
            n_video_frames=video_frames,
            n_audio_frames=audio_frames,
            proj_dim=video_hidden,
            head_hidden=head_hidden,
            proj_dropout=dropout,
            repr_dropout=dropout,
            head_dropout=dropout,
        ).to(device)
    elif arch == "v6_crossperson":
        # Probe 6: per-person video + symmetric cross-person attention.
        # Requires per-person video features (T, 2, D) — produced by
        # build_perperson_video_features.py from patch + pose dirs.
        model = MultiModalV6CrossPerson(
            video_feature_dim=video_dim,
            audio_feature_dim=audio_dim,
            proj_dim=video_hidden,
            head_hidden=head_hidden,
            proj_dropout=dropout,
            lstm_dropout=0.2,
            repr_dropout=dropout,
            head_dropout=dropout,
            cross_attn_layers=1,
            cross_attn_heads=4,
            cross_attn_dropout=0.1,
        ).to(device)
    elif arch == "v7_temporal_context":
        # Probe 7: per-second v2 + cross-second LSTM. (2k+1) seconds of
        # context, prediction at center second. Standard meanpatch/wavlm
        # 2D features.
        model = MultiModalV7TemporalContext(
            video_feature_dim=video_dim,
            audio_feature_dim=audio_dim,
            context_window=context_window,
            proj_dim=video_hidden,
            head_hidden=head_hidden,
            proj_dropout=dropout,
            lstm_dropout=0.2,
            repr_dropout=dropout,
            head_dropout=dropout,
        ).to(device)
    elif arch == "v8_subjinv":
        # Probe 8: v6_crossperson + adversarial subject classifier head with
        # gradient reversal. n_subjects_train comes from the dataset's vocab.
        n_subj_train = len(train_dataset.subject_id_to_int)
        model = MultiModalV8SubjectInvariant(
            video_feature_dim=video_dim,
            audio_feature_dim=audio_dim,
            n_subjects_train=n_subj_train,
            proj_dim=video_hidden,
            head_hidden=head_hidden,
            proj_dropout=dropout,
            lstm_dropout=0.2,
            repr_dropout=dropout,
            head_dropout=dropout,
            cross_attn_layers=1,
            cross_attn_heads=4,
            cross_attn_dropout=0.1,
            lambda_subj=lambda_subj,
        ).to(device)
    elif arch == "v2_subjinv":
        # v2 body + adversarial subject head on the fused (2P) repr — the clean
        # 2D-feature analogue of v8. n_subjects_train comes from the dataset vocab.
        n_subj_train = len(train_dataset.subject_id_to_int)
        model = MultiModalV2SubjInv(
            video_feature_dim=video_dim,
            audio_feature_dim=audio_dim,
            n_subjects_train=n_subj_train,
            proj_dim=video_hidden,
            head_hidden=head_hidden,
            proj_dropout=dropout,
            lstm_dropout=0.2,
            repr_dropout=dropout,
            head_dropout=dropout,
            input_norm=input_norm,
            lambda_subj=lambda_subj,
        ).to(device)
    else:
        raise ValueError(
            f"Unknown --arch {arch!r}; expected one of "
            f"'v1', 'v2', 'v3', 'v4', 'v2_patch', 'v2_pose', "
            f"'v5_coupling', 'v6_crossperson', 'v7_temporal_context', "
            f"'v8_subjinv', 'v2_subjinv'."
        )

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if warmup_epochs > 0:
        scheduler = SequentialLR(
            optimizer,
            [
                LinearLR(optimizer, start_factor=0.3, total_iters=warmup_epochs),
                CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs),
            ],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    config = {
        "arch": arch,
        "video_feature_dir": str(video_dir),
        "audio_feature_dir": str(audio_dir),
        "video_dim": video_dim, "video_frames": video_frames,
        "audio_dim": audio_dim, "audio_frames": audio_frames,
        "video_hidden": video_hidden, "audio_hidden": audio_hidden,
        "head_hidden": head_hidden, "dropout": dropout,
        "epochs": epochs, "batch_size": batch_size,
        "learning_rate": learning_rate, "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs, "patience": patience,
        "val_split": val_split, "seed": seed,
        "early_stop_metric": early_stop_metric,
        "num_folds": num_folds, "fold_idx": fold_idx,
        "group_key": group_key,
        "train_subsample_families": train_subsample_families,
        "modality": modality,
        "video_dropout_prob": video_dropout_prob,
        "audio_dropout_prob": audio_dropout_prob,
        "audio_shuffle_prob": audio_shuffle_prob,
        "pose_feature_dir": str(pose_dir) if pose_dir else None,
        "fallback_csv": fallback_csv,
        "max_fallback_frames": max_fallback_frames,
        "fallback_mode": fallback_mode,
        "max_subject_fallback_rate": max_subject_fallback_rate,
        "context_window": context_window,
        "lambda_subj": lambda_subj,
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    history = {
        "train_losses": [], "val_losses": [],
        "train_accs": [], "val_accs": [],
        "val_aucs": [], "val_f1s": [],
        "learning_rates": [],
        "best_val_auc": 0.0, "best_val_loss": None, "best_epoch": 0,
        # Per-criterion best-epoch tracking — see "v2 finding: AUC peaks at warmup
        # epoch 1 but val_loss/val_acc peak at epoch 4-5". Each criterion gets its
        # own best_<crit>.pt checkpoint so we don't have to re-run to recover the
        # right operating point. best.pt = best by `early_stop_metric` (preserves
        # the existing convention used by 30+ files in the repo).
        "best_acc_epoch": 0, "best_val_acc": 0.0,
        "best_loss_epoch": 0, "best_val_loss_min": float("inf"),
        "early_stop_metric": early_stop_metric,
    }
    epochs_without_improvement = 0
    if early_stop_metric not in ("val_auc", "val_loss", "val_acc"):
        raise ValueError(
            f"--early-stop-metric must be one of val_auc/val_loss/val_acc, got {early_stop_metric!r}"
        )

    for epoch in range(epochs):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0
        all_logits, all_labels = [], []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False):
            v = batch["video_features"].to(device)
            a = batch["audio_features"].to(device)
            y = batch["label"].to(device)
            # Single-modality ablation: zero out the suppressed modality.
            # Same architecture/hyperparameters/fold splits, just one input
            # stream removed → apples-to-apples vs the multimodal CV.
            if modality == "video":
                a = torch.zeros_like(a)
            elif modality == "audio":
                v = torch.zeros_like(v)
            elif modality == "both" and (video_dropout_prob > 0 or audio_dropout_prob > 0):
                # D3: stochastic modality dropout during training only.
                # Forces each modality's pathway to carry the prediction
                # signal on its own a fraction of the time, preventing the
                # audio pathway from collapsing into a video-redundant
                # representation (E2 from the audio-contribution
                # investigation). Per-batch Bernoulli; resample if both
                # would be dropped (degenerate input).
                while True:
                    drop_v = torch.rand((), device=device).item() < video_dropout_prob
                    drop_a = torch.rand((), device=device).item() < audio_dropout_prob
                    if not (drop_v and drop_a):
                        break
                if drop_v:
                    v = torch.zeros_like(v)
                if drop_a:
                    a = torch.zeros_like(a)
            if use_pose:
                p = batch["pose_features"].to(device)
                logits = model(v, a, p)
            elif arch in ("v8_subjinv", "v2_subjinv"):
                # Both return (sync_logits, subject_logits). The gradient-
                # reversal layer inside the model flips the gradient sign
                # before it reaches the shared encoder, so we just sum the
                # two losses normally — no sign flip needed here.
                subj_id = batch["subject_id_int"].to(device)
                logits, subj_logits = model(v, a, subj_id)
                # Mask out any -1 subject indices (val subjects accidentally
                # in train? shouldn't happen but defensive).
                valid_mask = subj_id >= 0
                if valid_mask.any():
                    subj_loss = nn.functional.cross_entropy(
                        subj_logits[valid_mask], subj_id[valid_mask]
                    )
                else:
                    subj_loss = torch.zeros((), device=device)
            else:
                logits = model(v, a)
            sync_loss = criterion(logits, y)
            if arch in ("v8_subjinv", "v2_subjinv"):
                # The GRL inside the model already flips the gradient sign.
                # So adding subj_loss here pushes the encoder AWAY from
                # subject-discriminability while pushing the subject head
                # TOWARD it. The lambda scaling is applied inside the GRL.
                loss = sync_loss + subj_loss
            else:
                loss = sync_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_labels.append(y.detach().cpu())
        train_loss /= len(train_loader)
        train_metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))

        model.eval()
        val_loss = 0.0
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]", leave=False):
                v = batch["video_features"].to(device)
                a = batch["audio_features"].to(device)
                y = batch["label"].to(device)
                if modality == "video":
                    a = torch.zeros_like(a)
                elif modality == "audio":
                    v = torch.zeros_like(v)
                if use_pose:
                    p = batch["pose_features"].to(device)
                    logits = model(v, a, p)
                elif arch in ("v8_subjinv", "v2_subjinv"):
                    # Val doesn't compute subject loss — just sync logits.
                    logits = model(v, a, subject_id=None)
                else:
                    logits = model(v, a)
                val_loss += criterion(logits, y).item()
                all_logits.append(logits.cpu())
                all_labels.append(y.cpu())
        val_loss /= len(val_loader)
        val_metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
            f"AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f} | LR: {lr:.2e}"
        )

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_accs"].append(train_metrics["accuracy"])
        history["val_accs"].append(val_metrics["accuracy"])
        history["val_aucs"].append(val_metrics["auc"])
        history["val_f1s"].append(val_metrics["f1"])
        history["learning_rates"].append(lr)

        # Per-criterion best tracking. Save a checkpoint for each criterion the
        # first time it improves; this lets us recover the right operating point
        # without re-running. best.pt always shadows the early-stop-metric's best.
        ckpt = {
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "config": config, "history": history,
        }
        new_best_auc = val_metrics["auc"] > history["best_val_auc"]
        new_best_acc = val_metrics["accuracy"] > history["best_val_acc"]
        new_best_loss = val_loss < history["best_val_loss_min"]

        if new_best_auc:
            history["best_val_auc"] = val_metrics["auc"]
            torch.save(ckpt, save_dir / "best_auc.pt")
        if new_best_acc:
            history["best_val_acc"] = val_metrics["accuracy"]
            history["best_acc_epoch"] = epoch
            torch.save(ckpt, save_dir / "best_acc.pt")
        if new_best_loss:
            history["best_val_loss_min"] = val_loss
            history["best_val_loss"] = val_loss  # legacy field, kept for plotting compat
            history["best_loss_epoch"] = epoch
            torch.save(ckpt, save_dir / "best_loss.pt")

        # Drive early stopping + the canonical best.pt off the configured metric.
        # val_loss is the smoothest signal; val_auc fluctuates with calibration
        # at low LR (artifact: AUC peaks during warmup before the model is well
        # thresholded — see v2 baseline finding).
        if early_stop_metric == "val_auc":
            is_best_for_stop = new_best_auc
            stop_metric_value = val_metrics["auc"]
        elif early_stop_metric == "val_loss":
            is_best_for_stop = new_best_loss
            stop_metric_value = val_loss
        else:  # val_acc
            is_best_for_stop = new_best_acc
            stop_metric_value = val_metrics["accuracy"]

        if is_best_for_stop:
            history["best_epoch"] = epoch
            epochs_without_improvement = 0
            torch.save(ckpt, save_dir / "best.pt")
            logger.info(
                f"  -> New best by {early_stop_metric} ({stop_metric_value:.4f}) "
                f"| AUC={val_metrics['auc']:.4f} Acc={val_metrics['accuracy']:.4f} Loss={val_loss:.4f}"
            )
        else:
            epochs_without_improvement += 1

        torch.save(ckpt, save_dir / "latest.pt")
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if epochs_without_improvement >= patience:
            logger.info(f"Early stopping at epoch {epoch+1} ({early_stop_metric} stalled for {patience} epochs)")
            break

    try:
        plot_history(history, save_dir / "training_plot.png")
    except Exception as e:
        logger.warning(f"Plot failed: {e}")

    logger.info(
        f"Done. Best by {early_stop_metric} at epoch {history['best_epoch'] + 1}. "
        f"Per-criterion bests: "
        f"AUC={history['best_val_auc']:.4f} | "
        f"Acc={history['best_val_acc']:.4f} (ep {history['best_acc_epoch'] + 1}) | "
        f"Loss={history['best_val_loss_min']:.4f} (ep {history['best_loss_epoch'] + 1})"
    )

    # Dump best-epoch per-clip val predictions so downstream stats (kappa,
    # balanced acc, sens/spec, calibration, dyad-cluster CIs, subgroups) can be
    # computed by scripts/compute_classifier_metrics.py. val_loader is
    # shuffle=False, so its emission order matches val_entries 1:1.
    try:
        import csv as _csv
        best_ckpt = torch.load(save_dir / "best.pt", map_location=device)
        model.load_state_dict(best_ckpt.get("model_state_dict", best_ckpt))
        model.eval()
        probs_all = []
        with torch.no_grad():
            for batch in val_loader:
                v = batch["video_features"].to(device)
                a = batch["audio_features"].to(device)
                if modality == "video":
                    a = torch.zeros_like(a)
                elif modality == "audio":
                    v = torch.zeros_like(v)
                # Mirror the val loop's arch dispatch exactly, else pose /
                # subjinv archs raise (wrong forward signature) and dump nothing.
                if use_pose:
                    logits = model(v, a, batch["pose_features"].to(device))
                elif arch in ("v8_subjinv", "v2_subjinv"):
                    logits = model(v, a, subject_id=None)
                else:
                    logits = model(v, a)
                probs_all.append(torch.sigmoid(logits).cpu().view(-1))
        probs_all = torch.cat(probs_all).tolist()
        if len(probs_all) != len(val_entries):
            raise RuntimeError(
                f"pred/entry count mismatch: {len(probs_all)} vs {len(val_entries)}"
            )
        pred_path = save_dir / "val_predictions.csv"
        with open(pred_path, "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["subject_id", "family_id", "video_path", "second",
                        "session", "fold", "y_true", "y_prob"])
            for e, p in zip(val_entries, probs_all):
                sid = str(e.get("subject_id", ""))
                w.writerow([sid, family_id_from_subject(sid),
                            e.get("video_path", ""), e.get("second", ""),
                            e.get("session", ""), fold_idx,
                            int(e["label"]), "%.6f" % p])
        logger.info(f"Wrote {len(probs_all)} val predictions to {pred_path}")
    except Exception as e:
        logger.warning(f"Val prediction dump failed: {e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--video-feature-dir", required=True)
    parser.add_argument("--audio-feature-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument(
        "--arch",
        choices=[
            "v1", "v2", "v3", "v4",
            "v2_patch", "v2_pose", "v5_coupling", "v6_crossperson",
            "v7_temporal_context", "v8_subjinv", "v2_subjinv",
        ],
        default="v1",
        help="Model architecture. v1=per-modality LSTM(D->H) + concat (original). "
             "v2=projection bottleneck + 2-layer LSTM (audio) + mean-pool (video) "
             "+ explicit aggregator dropout + concat. "
             "v3=v2 backbone with cross-attention fusion replacing concat. "
             "v4=token-level cross-modal transformer over the joint 61-token "
             "sequence (every video frame attends to every audio frame). "
             "v2_patch=v2 with spatial attention over DINOv2 patch tokens "
             "(probe 1: requires 3D video features (T, P, D), e.g. "
             "data/dinov2_features_patches/; uses lazy-load dataset). "
             "v2_pose=v2 + 3rd pose pathway (probe 3: requires "
             "--pose-feature-dir of YOLO26-pose features, shape (T, 2, 17, 3)). "
             "v5_coupling=per-frame cosine coupling + softmax-attention pool "
             "(probe 5: tests the sub-second covariation hypothesis). "
             "v6_crossperson=per-person video + symmetric cross-person attention "
             "(probe 6: requires --video-feature-dir of per-person features, "
             "shape (T, 2 persons, D), built by build_perperson_video_features.py).",
    )
    parser.add_argument("--video-hidden", type=int, default=64)
    parser.add_argument("--audio-hidden", type=int, default=64)
    parser.add_argument("--head-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--early-stop-metric",
        choices=["val_auc", "val_loss", "val_acc"],
        default="val_auc",
        help="Metric to drive early stopping and best.pt. Default val_auc preserves "
             "v1 behavior. v2 baseline showed val_auc peaks at warmup epoch 1 while "
             "val_loss/val_acc peak at epoch 4-5; for v2, prefer val_loss.",
    )
    parser.add_argument(
        "--num-folds", type=int, default=0,
        help="If >0, use deterministic k-fold subject-grouped CV split. "
             "Default 0 → original 80/20 random split (preserves v1 reproducibility).",
    )
    parser.add_argument(
        "--fold-idx", type=int, default=-1,
        help="Which fold to hold out as val (0..num_folds-1). Required when --num-folds > 0.",
    )
    parser.add_argument(
        "--train-subsample-families", type=int, default=0,
        help="If >0, keep only this many randomly-chosen TRAIN families (val "
             "untouched) for a data-scaling curve. 0 = use all (default).",
    )
    parser.add_argument(
        "--group-key", choices=["family_id", "subject_id"], default="family_id",
        help="Grouping unit for the leakage-safe split. Default 'family_id' keeps "
             "both members of a dyad on one side (required for CARE, where parent "
             "and child are distinct subject_ids). 'subject_id' reproduces the old "
             "(leaky-for-CARE) behavior.",
    )
    parser.add_argument(
        "--modality", choices=["both", "video", "audio"], default="both",
        help="Single-modality ablation. 'both' (default) trains the full multimodal "
             "model. 'video' zeros out audio inputs; 'audio' zeros out video. "
             "Same architecture / fold splits / hyperparams as multimodal — provides "
             "apples-to-apples ablation for the multimodal-vs-single-modality gain.",
    )
    parser.add_argument(
        "--video-dropout-prob", type=float, default=0.0,
        help="D3: stochastic video-modality dropout during training. Each training "
             "batch independently zeros video features with this probability. "
             "Only active when --modality=both. Default 0.0 = no dropout (baseline).",
    )
    parser.add_argument(
        "--audio-dropout-prob", type=float, default=0.0,
        help="D3: stochastic audio-modality dropout during training. Mirror of "
             "--video-dropout-prob. If both would be dropped on a batch, we resample.",
    )
    parser.add_argument(
        "--audio-shuffle-prob", type=float, default=0.0,
        help="Probe 2: within-recording audio-shuffle augmentation. During training, "
             "each sample has this probability of having its audio replaced with audio "
             "from a different second in the same recording, with label forced to 0. "
             "Constrained to within-recording so the model can't shortcut via "
             "subject/scene cues. Validation is never shuffled. Default 0.0 = off.",
    )
    parser.add_argument(
        "--pose-feature-dir", default=None,
        help="Probe 3: directory containing YOLO26-pose features "
             "(data/pose_features). Required when --arch=v2_pose; ignored otherwise.",
    )
    parser.add_argument(
        "--fallback-csv", default=None,
        help="Optional fallback_per_segment.csv from build_perperson_video_features.py. "
             "Filters out segments whose underlying video frame didn't exist "
             "(out-of-range second in labels.csv). Applies to all arches.",
    )
    parser.add_argument(
        "--max-fallback-frames", type=int, default=11,
        help="[segment mode] Drop segments where either slot has more than this "
             "many CLS-fallback frames (max=12). Default 11 = drop only 12/12 "
             "(strictly out-of-range frames).",
    )
    parser.add_argument(
        "--fallback-mode", choices=["segment", "subject"], default="segment",
        help="segment (default): drop individual bogus segments. subject: drop "
             "ALL segments from any (subject, session) whose bogus rate exceeds "
             "--max-subject-fallback-rate. Subject mode preserves time/content "
             "distribution within remaining subjects.",
    )
    parser.add_argument(
        "--max-subject-fallback-rate", type=float, default=0.80,
        help="[subject mode] Drop (subject, session) recordings whose mean "
             "fallback rate (across both slots) exceeds this threshold. "
             "Default 0.80 = drop the ~7 fully-broken dyads.",
    )
    parser.add_argument(
        "--context-window", type=int, default=2,
        help="[v7_temporal_context] Number of seconds of context on each side "
             "of the center labeled second. Default 2 → 5-second receptive field. "
             "Recording-edge seconds use zero padding.",
    )
    parser.add_argument(
        "--lambda-subj", type=float, default=0.1,
        help="[v8_subjinv] Strength of the adversarial subject-invariance "
             "objective. λ=0 reduces v8 to v6_crossperson exactly. Default 0.1.",
    )
    parser.add_argument(
        "--input-norm", action="store_true",
        help="[v2] Per-modality LayerNorm on the raw frozen DINOv2/WavLM features "
             "before projection. Off by default (structurally identical to prior "
             "runs). Standard fix for anisotropic frozen-backbone features.",
    )
    args = parser.parse_args()

    train(**vars(args))


if __name__ == "__main__":
    main()
