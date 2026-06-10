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
                return {
                    "video_features": self.video_tensor[idx],
                    "audio_features": self.audio_tensor[other_idx],
                    "label": torch.zeros((), dtype=torch.float32),
                }
        return {
            "video_features": self.video_tensor[idx],
            "audio_features": self.audio_tensor[idx],
            "label": self.labels[idx],
        }


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


def collate(batch: list[dict]) -> dict:
    return {
        "video_features": torch.stack([b["video_features"] for b in batch]),
        "audio_features": torch.stack([b["audio_features"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
    }


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
        v = self.video_proj(video_features)         # (B, 12, P)
        v_repr = v.mean(dim=1)                      # (B, P)
        v_repr = self.video_repr_drop(v_repr)

        a = self.audio_proj(audio_features)         # (B, 49, P)
        _, (h_n, _) = self.audio_lstm(a)            # h_n: (num_layers=2, B, P), NOT batch-first
        a_repr = h_n[-1, :, :]                      # top-layer hidden across batch
        a_repr = self.audio_repr_drop(a_repr)

        fused = torch.cat([v_repr, a_repr], dim=-1)  # (B, 2P)
        return self.head(fused).squeeze(-1)


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


def merge_feature_indices(
    video_feature_dir: Path,
    audio_feature_dir: Path,
) -> tuple[pd.DataFrame, int, int, int, int]:
    """Inner-join the two feature_index.csv files on (video_path, second).

    Returns the merged DataFrame plus the feature shapes for both modalities.
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
        f"Joined feature indices: {len(merged)} samples "
        f"(dropped {n_dropped_v} video-only, {n_dropped_a} audio-only)"
    )
    return merged, video_dim, video_frames, audio_dim, audio_frames


def subject_grouped_split(
    entries: list[dict],
    val_split: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """80/20 split grouped by subject_id to prevent leakage across train/val."""
    rng = np.random.default_rng(seed)
    by_subject = defaultdict(list)
    for e in entries:
        by_subject[e["subject_id"]].append(e)

    subjects = list(by_subject.keys())
    rng.shuffle(subjects)
    n_val = max(1, int(len(subjects) * val_split))
    val_subjects = set(subjects[:n_val])

    train, val = [], []
    for s, group in by_subject.items():
        (val if s in val_subjects else train).extend(group)
    logger.info(
        f"Split: {len(train)} train ({len(subjects) - n_val} subjects), "
        f"{len(val)} val ({n_val} subjects)"
    )
    return train, val


def subject_kfold_split(
    entries: list[dict],
    num_folds: int,
    fold_idx: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Deterministic k-fold split grouped by subject_id.

    For tightening confidence intervals on a single point estimate (like
    v2_baseline_v6's 0.7640 val_acc), 5-fold CV across the same 49 subjects
    gives 5 disjoint val sets and lets us report mean ± SE instead of a
    single-fold number.

    Subjects are sorted (deterministic order) then shuffled with the same
    seed across folds. Subjects in fold k are the [k*fold_size:(k+1)*fold_size]
    slice of the shuffled list. Same seed → same fold composition.
    """
    if num_folds < 2 or fold_idx < 0 or fold_idx >= num_folds:
        raise ValueError(
            f"Invalid fold args: num_folds={num_folds}, fold_idx={fold_idx}"
        )
    rng = np.random.default_rng(seed)
    by_subject = defaultdict(list)
    for e in entries:
        by_subject[e["subject_id"]].append(e)

    subjects = sorted(by_subject.keys())  # deterministic order before shuffle
    rng.shuffle(subjects)

    fold_size = len(subjects) // num_folds
    val_start = fold_idx * fold_size
    val_end = val_start + fold_size if fold_idx < num_folds - 1 else len(subjects)
    val_subjects = set(subjects[val_start:val_end])

    train, val = [], []
    for s, group in by_subject.items():
        (val if s in val_subjects else train).extend(group)
    logger.info(
        f"K-fold split (fold {fold_idx+1}/{num_folds}, seed={seed}): "
        f"{len(train)} train ({len(subjects) - len(val_subjects)} subjects), "
        f"{len(val)} val ({len(val_subjects)} subjects). "
        f"Val subjects: {sorted(val_subjects)}"
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
    modality: str = "both",
    video_dropout_prob: float = 0.0,
    audio_dropout_prob: float = 0.0,
    audio_shuffle_prob: float = 0.0,
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    video_dir = Path(video_feature_dir)
    audio_dir = Path(audio_feature_dir)
    merged, video_dim, video_frames, audio_dim, audio_frames = merge_feature_indices(
        video_dir, audio_dir
    )
    logger.info(
        f"Video features: dim={video_dim}, frames={video_frames}. "
        f"Audio features: dim={audio_dim}, frames={audio_frames}."
    )

    entries = merged.to_dict("records")
    if num_folds > 0 and fold_idx >= 0:
        train_entries, val_entries = subject_kfold_split(entries, num_folds, fold_idx, seed)
    else:
        train_entries, val_entries = subject_grouped_split(entries, val_split, seed)
    pos_weight = compute_pos_weight(train_entries)
    logger.info(f"Train pos_weight: {pos_weight:.3f}")

    # Probe 1 uses --arch v2_patch with 3D video features (T, P, D) too big to
    # preload; switch to lazy-load dataset for that arch. Probe 2's audio shuffle
    # only applies to the train loader (val keeps real labels).
    if arch == "v2_patch":
        train_dataset = MultiModalPatchFeatureDataset(video_dir, audio_dir, train_entries)
        val_dataset = MultiModalPatchFeatureDataset(video_dir, audio_dir, val_entries)
        if audio_shuffle_prob > 0:
            raise ValueError(
                "--audio-shuffle-prob is not yet wired to MultiModalPatchFeatureDataset. "
                "Run probe 1 (--arch v2_patch) and probe 2 (--audio-shuffle-prob) separately."
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
        # Requires 3D video features (T, P, D), e.g. data/dinov2_features/.
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
    else:
        raise ValueError(
            f"Unknown --arch {arch!r}; expected 'v1', 'v2', 'v3', 'v4', or 'v2_patch'."
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
        "modality": modality,
        "video_dropout_prob": video_dropout_prob,
        "audio_dropout_prob": audio_dropout_prob,
        "audio_shuffle_prob": audio_shuffle_prob,
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
            logits = model(v, a)
            loss = criterion(logits, y)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--video-feature-dir", required=True)
    parser.add_argument("--audio-feature-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument(
        "--arch", choices=["v1", "v2", "v3", "v4", "v2_patch"], default="v1",
        help="Model architecture. v1=per-modality LSTM(D->H) + concat (original). "
             "v2=projection bottleneck + 2-layer LSTM (audio) + mean-pool (video) "
             "+ explicit aggregator dropout + concat. "
             "v3=v2 backbone with cross-attention fusion replacing concat. "
             "v4=token-level cross-modal transformer over the joint 61-token "
             "sequence (every video frame attends to every audio frame). "
             "v2_patch=v2 with spatial attention over DINOv2 patch tokens "
             "(probe 1: requires 3D video features (T, P, D), e.g. "
             "data/dinov2_features/; uses lazy-load dataset).",
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
    args = parser.parse_args()

    train(**vars(args))


if __name__ == "__main__":
    main()
