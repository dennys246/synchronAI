#!/usr/bin/env python3
"""Train classification head on pre-extracted WavLM/Whisper audio features.

Lightweight training script for audio — mirrors train_from_features.py
(DINOv2 video) but adapted for audio encoder features. Since features are
pre-extracted, training runs entirely on CPU in seconds per epoch.

Usage:
    python scripts/train_audio_from_features.py \
        --feature-dir data/wavlm_features \
        --save-dir runs/audio_sweep/baseline \
        --temporal-aggregation mean \
        --hidden-dim 128 \
        --epochs 50

The saved checkpoint is compatible with AudioClassifier for optional
stage 2 end-to-end fine-tuning on GPU.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synchronai.data.audio.feature_dataset import create_audio_feature_dataloaders
from synchronai.models.cv.video_classifier import (
    TemporalAttention,
    TemporalLSTM,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class AudioFeatureClassifier(nn.Module):
    """Lightweight classifier that operates on pre-extracted audio features.

    Architecture:
        [learnable layer weights] -> [optional projection] ->
        temporal aggregation -> MLP head

    Supports two input modes:
    - Single-layer: (B, T, D) — pre-blended features
    - Per-layer:    (B, L, T, D) — all hidden states, blended here via
                    learnable layer_weights

    The optional projection layer compresses the encoder dim (e.g. 768)
    down to a smaller working dim before temporal aggregation, which acts
    as a bottleneck to reduce overfitting on small datasets.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        temporal_aggregation: str = "mean",
        dropout: float = 0.5,
        output_dim: int = 1,
        n_layers: int = 0,
        project_dim: int = 0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.temporal_aggregation_type = temporal_aggregation
        self.n_layers = n_layers

        # Learnable layer weights (only used with per-layer features)
        if n_layers > 0:
            self.layer_weights = nn.Parameter(
                torch.ones(n_layers) / n_layers
            )
        else:
            self.layer_weights = None

        # Optional projection bottleneck: D -> project_dim
        working_dim = feature_dim
        if project_dim > 0 and project_dim < feature_dim:
            self.projection = nn.Sequential(
                nn.Linear(feature_dim, project_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),  # Lighter dropout in projection
            )
            working_dim = project_dim
        else:
            self.projection = None

        # Temporal aggregation
        if temporal_aggregation == "attention":
            self.temporal = TemporalAttention(working_dim, hidden_dim)
            temporal_out_dim = working_dim
        elif temporal_aggregation == "lstm":
            self.temporal = TemporalLSTM(working_dim, hidden_dim)
            temporal_out_dim = hidden_dim
        elif temporal_aggregation in ["mean", "max"]:
            self.temporal = None
            temporal_out_dim = working_dim
        else:
            raise ValueError(f"Unknown temporal aggregation: {temporal_aggregation}")

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(temporal_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"AudioFeatureClassifier: feature_dim={feature_dim}, "
            f"project_dim={project_dim or feature_dim}, "
            f"temporal={temporal_aggregation}, hidden_dim={hidden_dim}, "
            f"n_layers={n_layers}, total_params={total_params:,}"
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass on pre-extracted features.

        Args:
            features: (B, T, D) single-layer or (B, L, T, D) per-layer

        Returns:
            Logits (B, output_dim)
        """
        # Blend layers if per-layer input
        if self.layer_weights is not None and features.ndim == 4:
            # features: (B, L, T, D), layer_weights: (L,)
            weights = torch.softmax(self.layer_weights, dim=0)
            features = (features * weights[None, :, None, None]).sum(dim=1)
            # -> (B, T, D)

        # Project down if configured
        if self.projection is not None:
            features = self.projection(features)  # (B, T, project_dim)

        # Temporal aggregation
        if self.temporal_aggregation_type == "mean":
            aggregated = features.mean(dim=1)
        elif self.temporal_aggregation_type == "max":
            aggregated = features.max(dim=1).values
        else:
            aggregated = self.temporal(features)

        return self.head(aggregated)


def compute_metrics(logits, labels):
    """Compute binary classification metrics."""
    logits = logits.squeeze(-1) if logits.dim() > 1 else logits
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    accuracy = (preds == labels).float().mean().item()

    auc = 0.5
    probs_np = probs.detach().numpy()
    labels_np = labels.detach().numpy()
    if len(set(labels_np.tolist())) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(labels_np, probs_np)
        except ImportError:
            pass

    tp = ((preds == 1) & (labels == 1)).float().sum()
    fp = ((preds == 1) & (labels == 0)).float().sum()
    fn = ((preds == 0) & (labels == 1)).float().sum()

    precision = (tp / (tp + fp + 1e-8)).item()
    recall = (tp / (tp + fn + 1e-8)).item()
    f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
    }


def augment_features(
    features: torch.Tensor,
    noise_std: float = 0.05,
    frame_drop_prob: float = 0.15,
) -> torch.Tensor:
    """Apply stochastic augmentation to pre-extracted audio features.

    Uses lower noise/dropout defaults than video since audio encoder features
    tend to be more sensitive to perturbation.
    """
    if noise_std > 0:
        features = features + torch.randn_like(features) * noise_std

    if frame_drop_prob > 0:
        B, T, D = features.shape
        frame_mask = (torch.rand(B, T, 1, device=features.device) > frame_drop_prob).float()
        all_dropped = frame_mask.sum(dim=1, keepdim=True) == 0
        if all_dropped.any():
            rand_idx = torch.randint(0, T, (B, 1, 1), device=features.device)
            rescue = torch.zeros_like(frame_mask).scatter_(1, rand_idx, 1.0)
            frame_mask = torch.where(all_dropped, rescue, frame_mask)
        features = features * frame_mask

    return features


def mixup_features(
    features: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup augmentation in feature space."""
    if alpha <= 0:
        return features, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)

    batch_size = features.size(0)
    index = torch.randperm(batch_size, device=features.device)

    mixed = lam * features + (1 - lam) * features[index]
    return mixed, labels, labels[index], lam


def mixup_criterion(
    criterion: nn.Module,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute mixed loss for mixup training."""
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


def train_audio_from_features(
    feature_dir: str,
    save_dir: str,
    encoder: str = "wavlm-base-plus",
    temporal_aggregation: str = "mean",
    hidden_dim: int = 128,
    dropout: float = 0.5,
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-3,
    warmup_epochs: int = 3,
    patience: int = 15,
    label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0,
    lr_schedule: str = "cosine",
    lr_restart_period: int = 10,
    val_split: float = 0.2,
    group_by: str = "subject_id",
    num_workers: int = 4,
    seed: int = 42,
    project_dim: int = 0,
) -> None:
    """Train temporal + head classifier on pre-extracted audio features."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info(f"Loading audio features from {feature_dir}")
    train_loader, val_loader, pos_weight, encoder_dim, n_frames, n_layers = (
        create_audio_feature_dataloaders(
            feature_dir=feature_dir,
            batch_size=batch_size,
            val_split=val_split,
            group_by=group_by,
            num_workers=num_workers,
            seed=seed,
        )
    )
    logger.info(f"Encoder dim: {encoder_dim}, Frames/chunk: {n_frames}")
    if n_layers > 0:
        logger.info(f"Per-layer features: {n_layers} layers (learnable weighting)")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    logger.info(f"Pos weight: {pos_weight:.3f}")

    # Create model
    model = AudioFeatureClassifier(
        feature_dim=encoder_dim,
        hidden_dim=hidden_dim,
        temporal_aggregation=temporal_aggregation,
        dropout=dropout,
        n_layers=n_layers,
        project_dim=project_dim,
    )
    model.to(device)

    # Loss
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Scheduler
    if lr_schedule == "cosine_restarts":
        if warmup_epochs > 0:
            warmup_sched = LinearLR(
                optimizer, start_factor=0.3, total_iters=warmup_epochs
            )
            main_sched = CosineAnnealingWarmRestarts(
                optimizer, T_0=lr_restart_period, T_mult=2, eta_min=1e-7
            )
            scheduler = SequentialLR(
                optimizer, [warmup_sched, main_sched], milestones=[warmup_epochs]
            )
        else:
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=lr_restart_period, T_mult=2, eta_min=1e-7
            )
    else:
        if warmup_epochs > 0:
            warmup_sched = LinearLR(
                optimizer, start_factor=0.3, total_iters=warmup_epochs
            )
            main_sched = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
            scheduler = SequentialLR(
                optimizer, [warmup_sched, main_sched], milestones=[warmup_epochs]
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Save config
    config = {
        "feature_dir": str(feature_dir),
        "encoder": encoder,
        "encoder_dim": encoder_dim,
        "n_frames": n_frames,
        "n_layers": n_layers,
        "project_dim": project_dim,
        "temporal_aggregation": temporal_aggregation,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "patience": patience,
        "label_smoothing": label_smoothing,
        "mixup_alpha": mixup_alpha,
        "lr_schedule": lr_schedule,
        "lr_restart_period": lr_restart_period,
        "val_split": val_split,
        "group_by": group_by,
        "seed": seed,
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training history
    history = {
        "train_losses": [],
        "val_losses": [],
        "train_accs": [],
        "val_accs": [],
        "val_aucs": [],
        "val_f1s": [],
        "learning_rates": [],
        "best_val_auc": 0.0,
        "best_val_loss": None,
        "best_epoch": 0,
    }

    epochs_without_improvement = 0

    for epoch in range(epochs):
        epoch_start = time.time()

        # --- Train ---
        model.train()
        train_loss = 0.0
        all_logits = []
        all_labels = []

        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", leave=False
        ):
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            original_labels = labels.clone()

            # Feature-level augmentation
            features = augment_features(features)

            # Label smoothing
            if label_smoothing > 0:
                smooth_labels = labels * (1.0 - 2 * label_smoothing) + label_smoothing
            else:
                smooth_labels = labels

            # Feature mixup
            if mixup_alpha > 0:
                features, labels_a, labels_b, lam = mixup_features(
                    features, smooth_labels, alpha=mixup_alpha
                )
                logits = model(features)
                logits = logits.squeeze(-1) if logits.dim() > 1 else logits
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            else:
                logits = model(features)
                logits = logits.squeeze(-1) if logits.dim() > 1 else logits
                loss = criterion(logits, smooth_labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_labels.append(original_labels.detach().cpu())

        train_loss /= len(train_loader)
        train_metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]", leave=False
            ):
                features = batch["features"].to(device)
                labels = batch["label"].to(device)

                logits = model(features)
                logits = logits.squeeze(-1) if logits.dim() > 1 else logits
                loss = criterion(logits, labels)

                val_loss += loss.item()
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())

        val_loss /= len(val_loader)
        val_metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
            f"AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f} | "
            f"LR: {lr:.2e}"
        )

        # Update history
        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_accs"].append(train_metrics["accuracy"])
        history["val_accs"].append(val_metrics["accuracy"])
        history["val_aucs"].append(val_metrics["auc"])
        history["val_f1s"].append(val_metrics["f1"])
        history["learning_rates"].append(lr)

        # Check for improvement
        is_best = val_metrics["auc"] > history["best_val_auc"]
        if is_best:
            history["best_val_auc"] = val_metrics["auc"]
            history["best_val_loss"] = val_loss
            history["best_epoch"] = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "history": history,
                "audio_feature_classifier": True,
                "model_config": {
                    "encoder": encoder,
                    "temporal_aggregation": temporal_aggregation,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "project_dim": project_dim,
                    "n_layers": n_layers,
                },
            }
            torch.save(checkpoint, save_dir / "best.pt")
            logger.info(f"  -> New best model (AUC: {val_metrics['auc']:.4f})")
        else:
            epochs_without_improvement += 1

        # Save latest
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "history": history,
            "audio_feature_classifier": True,
            "model_config": {
                "encoder": encoder,
                "temporal_aggregation": temporal_aggregation,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
            },
        }
        torch.save(checkpoint, save_dir / "latest.pt")

        # Save history
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(
                f"Early stopping at epoch {epoch+1} "
                f"(no improvement for {patience} epochs)"
            )
            break

    # Generate plot
    try:
        _plot_training_history(history, save_dir / "training_plot.png", encoder)
    except Exception as e:
        logger.warning(f"Failed to generate training plot: {e}")

    logger.info("Training complete!")
    logger.info(
        f"  Best AUC: {history['best_val_auc']:.4f} "
        f"at epoch {history['best_epoch'] + 1}"
    )
    logger.info(f"  Checkpoints saved to: {save_dir}")


def _plot_training_history(history: dict, output_path: Path, encoder: str) -> None:
    """Plot training history."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history["train_losses"]) + 1)
    best_epoch = history["best_epoch"] + 1

    ax = axes[0, 0]
    ax.plot(epochs, history["train_losses"], "b-", label="Train", linewidth=2)
    ax.plot(epochs, history["val_losses"], "r-", label="Val", linewidth=2)
    ax.axvline(x=best_epoch, color="g", linestyle="--", alpha=0.7,
               label=f"Best (epoch {best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history["train_accs"], "b-", label="Train", linewidth=2)
    ax.plot(epochs, history["val_accs"], "r-", label="Val", linewidth=2)
    ax.axvline(x=best_epoch, color="g", linestyle="--", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history["val_aucs"], "m-", label="Val AUC", linewidth=2)
    ax.axvline(x=best_epoch, color="g", linestyle="--", alpha=0.7)
    ax.axhline(y=history["best_val_auc"], color="g", linestyle=":", alpha=0.5,
               label=f"Best: {history['best_val_auc']:.4f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Validation AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, history["learning_rates"], "g-", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Audio Feature Training (Head on Pre-Extracted {encoder})",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved training plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train head on pre-extracted audio encoder features"
    )
    parser.add_argument(
        "--feature-dir", required=True,
        help="Directory with pre-extracted features (from extract_audio_features.py)",
    )
    parser.add_argument(
        "--save-dir", required=True,
        help="Output directory for checkpoints and plots",
    )
    parser.add_argument("--encoder", default="wavlm-base-plus",
                        help="Encoder used for extraction (metadata only)")
    parser.add_argument("--temporal-aggregation", default="mean",
                        choices=["mean", "max", "attention", "lstm"])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="Feature mixup alpha (0=disabled, 0.2=recommended)")
    parser.add_argument("--lr-schedule", default="cosine",
                        choices=["cosine", "cosine_restarts"])
    parser.add_argument("--lr-restart-period", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--group-by", default="subject_id",
                        choices=["subject_id", "video_path"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project-dim", type=int, default=0,
                        help="Project encoder dim to this size before temporal "
                             "aggregation (0=no projection, 128/256 recommended)")

    args = parser.parse_args()

    train_audio_from_features(
        feature_dir=args.feature_dir,
        save_dir=args.save_dir,
        encoder=args.encoder,
        temporal_aggregation=args.temporal_aggregation,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        lr_schedule=args.lr_schedule,
        lr_restart_period=args.lr_restart_period,
        val_split=args.val_split,
        group_by=args.group_by,
        num_workers=args.num_workers,
        seed=args.seed,
        project_dim=args.project_dim,
    )


if __name__ == "__main__":
    main()
