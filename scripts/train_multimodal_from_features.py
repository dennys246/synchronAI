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
    ):
        video_dir = Path(video_feature_dir) / "features"
        audio_dir = Path(audio_feature_dir) / "features"

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

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "video_features": self.video_tensor[idx],
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
) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)

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
    train_entries, val_entries = subject_grouped_split(entries, val_split, seed)
    pos_weight = compute_pos_weight(train_entries)
    logger.info(f"Train pos_weight: {pos_weight:.3f}")

    train_loader = DataLoader(
        MultiModalFeatureDataset(video_dir, audio_dir, train_entries),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        MultiModalFeatureDataset(video_dir, audio_dir, val_entries),
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
    else:
        raise ValueError(f"Unknown --arch {arch!r}; expected 'v1' or 'v2'.")

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
        "--arch", choices=["v1", "v2"], default="v1",
        help="Model architecture. v1=per-modality LSTM(D->H) + concat (original). "
             "v2=projection bottleneck + 2-layer LSTM (audio) + mean-pool (video) "
             "+ explicit aggregator dropout + concat.",
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
    args = parser.parse_args()

    train(**vars(args))


if __name__ == "__main__":
    main()
