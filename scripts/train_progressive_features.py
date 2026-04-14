#!/usr/bin/env python3
"""Progressive resolution training on pre-extracted DINOv2 features.

Trains LSTM + classification head across feature sets extracted at different
resolutions (e.g., 112 → 168 → 224). Each stage warm-starts from the previous,
so the model learns coarse spatial features first and refines with higher
resolution features later.

This is a CPU-friendly form of progressive growing: DINOv2 features are
pre-extracted at each resolution, and training just swaps which feature
directory to load from at each stage.

Usage:
    python scripts/train_progressive_features.py \
        --feature-dirs data/dinov2_features_small_112 \
                       data/dinov2_features_small_168 \
                       data/dinov2_features_small_meanpatch \
        --stage-epochs 10 10 15 \
        --stage-lrs 3e-4 1e-4 3e-5 \
        --save-dir runs/dinov2_progressive/baseline \
        --temporal-aggregation lstm \
        --hidden-dim 128 \
        --dropout 0.7
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

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir.parent))  # project root (for synchronai.*)
sys.path.insert(0, str(_script_dir))          # scripts/ dir (for train_from_features)

from synchronai.data.video.feature_dataset import create_feature_dataloaders

# Reuse components from train_from_features
from train_from_features import (
    FeatureClassifier,
    augment_features,
    compute_metrics,
    mixup_criterion,
    mixup_features,
)
from synchronai.utils.wandb_utils import (
    init_wandb,
    log_metrics,
    log_summary,
    log_artifact,
    finish_wandb,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def train_progressive(
    feature_dirs: list[str],
    stage_epochs: list[int],
    stage_lrs: list[float],
    save_dir: str,
    temporal_aggregation: str = "lstm",
    hidden_dim: int = 128,
    dropout: float = 0.5,
    weight_decay: float = 1e-2,
    warmup_epochs: int = 2,
    patience: int = 15,
    label_smoothing: float = 0.05,
    mixup_alpha: float = 0.2,
    lr_schedule: str = "cosine_restarts",
    lr_restart_period: int = 10,
    batch_size: int = 64,
    val_split: float = 0.2,
    group_by: str = "subject_id",
    num_workers: int = 0,
    seed: int = 42,
    backbone: str = "dinov2-small",
) -> None:
    """Train with progressive resolution feature stages.

    Each stage loads features from a different directory (extracted at a
    different resolution). The model weights carry over between stages,
    warm-starting from the previous stage's best checkpoint.

    Args:
        feature_dirs: List of feature directories, one per stage
            (e.g., [features_112, features_168, features_224])
        stage_epochs: Max epochs per stage
        stage_lrs: Peak learning rate per stage (typically decreasing)
        save_dir: Output directory for checkpoints and history
        temporal_aggregation: LSTM, attention, mean, or max
        hidden_dim: Hidden dimension for temporal + head
        dropout: Dropout rate
        weight_decay: AdamW weight decay
        warmup_epochs: Warmup epochs per stage
        patience: Early stopping patience (global across stages)
        label_smoothing: Label smoothing factor
        mixup_alpha: Feature mixup alpha (0 = disabled)
        lr_schedule: "cosine" or "cosine_restarts"
        lr_restart_period: T_0 for cosine restarts
        batch_size: Batch size
        val_split: Validation split fraction
        group_by: Group-by column for train/val split
        num_workers: DataLoader workers
        seed: Random seed
        backbone: Backbone name (metadata only)
    """
    assert len(feature_dirs) == len(stage_epochs) == len(stage_lrs), (
        f"feature_dirs ({len(feature_dirs)}), stage_epochs ({len(stage_epochs)}), "
        f"and stage_lrs ({len(stage_lrs)}) must have the same length"
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    n_stages = len(feature_dirs)

    # Verify all feature dirs exist and have compatible feature dims
    feature_dim = None
    n_frames = None
    for fd in feature_dirs:
        idx_file = Path(fd) / "feature_index.csv"
        if not idx_file.exists():
            raise FileNotFoundError(f"Feature index not found: {idx_file}")
        import pandas as pd
        df = pd.read_csv(idx_file)
        dim = int(df["feature_dim"].iloc[0])
        nf = int(df["n_frames"].iloc[0])
        if feature_dim is None:
            feature_dim = dim
            n_frames = nf
        elif dim != feature_dim:
            raise ValueError(
                f"Feature dim mismatch: {fd} has {dim}, expected {feature_dim}. "
                f"All stages must use the same backbone."
            )
        logger.info(f"Stage feature dir: {fd} (dim={dim}, frames={nf})")

    # Create model
    model = FeatureClassifier(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        temporal_aggregation=temporal_aggregation,
        dropout=dropout,
    )
    model.to(device)

    # Save config
    config = {
        "feature_dirs": [str(fd) for fd in feature_dirs],
        "stage_epochs": stage_epochs,
        "stage_lrs": stage_lrs,
        "backbone": backbone,
        "feature_dim": feature_dim,
        "n_frames": n_frames,
        "temporal_aggregation": temporal_aggregation,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "patience": patience,
        "label_smoothing": label_smoothing,
        "mixup_alpha": mixup_alpha,
        "lr_schedule": lr_schedule,
        "lr_restart_period": lr_restart_period,
        "batch_size": batch_size,
        "val_split": val_split,
        "group_by": group_by,
        "seed": seed,
        "progressive": True,
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Initialize wandb
    init_wandb(
        config=config,
        name=f"progressive-{backbone}-{temporal_aggregation}",
        tags=["progressive-training", backbone, temporal_aggregation],
        group="progressive-training",
        save_dir=save_dir,
    )

    # Global training history
    history = {
        "train_losses": [],
        "val_losses": [],
        "train_accs": [],
        "val_accs": [],
        "val_aucs": [],
        "val_f1s": [],
        "learning_rates": [],
        "stages": [],       # Stage index for each epoch
        "resolutions": [],   # Resolution label for each epoch
        "best_val_auc": 0.0,
        "best_val_loss": None,
        "best_epoch": 0,
        "best_stage": 0,
    }

    global_epoch = 0
    epochs_without_improvement = 0
    total_start = time.time()

    for stage_idx in range(n_stages):
        feature_dir = feature_dirs[stage_idx]
        stage_max_epochs = stage_epochs[stage_idx]
        stage_lr = stage_lrs[stage_idx]
        res_label = Path(feature_dir).name

        logger.info("")
        logger.info("=" * 60)
        logger.info(
            f"STAGE {stage_idx + 1}/{n_stages}: {res_label} "
            f"(lr={stage_lr}, epochs={stage_max_epochs})"
        )
        logger.info("=" * 60)

        # Load data for this stage
        train_loader, val_loader, pos_weight, fd, nf = create_feature_dataloaders(
            feature_dir=feature_dir,
            batch_size=batch_size,
            val_split=val_split,
            group_by=group_by,
            num_workers=num_workers,
            seed=seed,
        )
        logger.info(
            f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
        )

        # Loss
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=device)
        )

        # Fresh optimizer per stage (warm-started model weights carry over)
        optimizer = AdamW(
            model.parameters(), lr=stage_lr, weight_decay=weight_decay
        )

        # Scheduler for this stage
        if lr_schedule == "cosine_restarts":
            if warmup_epochs > 0:
                warmup_sched = LinearLR(
                    optimizer, start_factor=0.3, total_iters=warmup_epochs
                )
                main_sched = CosineAnnealingWarmRestarts(
                    optimizer, T_0=lr_restart_period, T_mult=2, eta_min=1e-7
                )
                scheduler = SequentialLR(
                    optimizer, [warmup_sched, main_sched],
                    milestones=[warmup_epochs]
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
                main_sched = CosineAnnealingLR(
                    optimizer, T_max=stage_max_epochs - warmup_epochs
                )
                scheduler = SequentialLR(
                    optimizer, [warmup_sched, main_sched],
                    milestones=[warmup_epochs]
                )
            else:
                scheduler = CosineAnnealingLR(optimizer, T_max=stage_max_epochs)

        # Train this stage
        for stage_epoch in range(stage_max_epochs):
            epoch_start = time.time()

            # --- Train ---
            model.train()
            train_loss = 0.0
            all_logits = []
            all_labels = []

            for batch in tqdm(
                train_loader,
                desc=f"Stage {stage_idx+1} Epoch {stage_epoch+1}/{stage_max_epochs} [train]",
                leave=False,
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
                    loss = mixup_criterion(
                        criterion, logits, labels_a, labels_b, lam
                    )
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
            train_metrics = compute_metrics(
                torch.cat(all_logits), torch.cat(all_labels)
            )

            # --- Validate ---
            model.eval()
            val_loss = 0.0
            all_logits = []
            all_labels = []

            with torch.no_grad():
                for batch in tqdm(
                    val_loader,
                    desc=f"Stage {stage_idx+1} Epoch {stage_epoch+1}/{stage_max_epochs} [val]",
                    leave=False,
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
            val_metrics = compute_metrics(
                torch.cat(all_logits), torch.cat(all_labels)
            )

            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            logger.info(
                f"[S{stage_idx+1}] Epoch {stage_epoch+1}/{stage_max_epochs} "
                f"({epoch_time:.1f}s) | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                f"AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f} | "
                f"LR: {lr:.2e}"
            )

            # Log to wandb
            log_metrics({
                "train/loss": train_loss,
                "train/accuracy": train_metrics["accuracy"],
                "val/loss": val_loss,
                "val/accuracy": val_metrics["accuracy"],
                "val/auc": val_metrics["auc"],
                "val/f1": val_metrics["f1"],
                "lr": lr,
                "stage": stage_idx,
                "resolution": res_label,
                "epoch": global_epoch,
            })

            # Update history
            history["train_losses"].append(train_loss)
            history["val_losses"].append(val_loss)
            history["train_accs"].append(train_metrics["accuracy"])
            history["val_accs"].append(val_metrics["accuracy"])
            history["val_aucs"].append(val_metrics["auc"])
            history["val_f1s"].append(val_metrics["f1"])
            history["learning_rates"].append(lr)
            history["stages"].append(stage_idx)
            history["resolutions"].append(res_label)

            # Check for improvement (global across all stages)
            is_best = val_metrics["auc"] > history["best_val_auc"]
            if is_best:
                history["best_val_auc"] = val_metrics["auc"]
                history["best_val_loss"] = val_loss
                history["best_epoch"] = global_epoch
                history["best_stage"] = stage_idx
                epochs_without_improvement = 0

                checkpoint = {
                    "epoch": global_epoch,
                    "stage": stage_idx,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "history": history,
                    "feature_classifier": True,
                    "model_config": {
                        "backbone": backbone,
                        "temporal_aggregation": temporal_aggregation,
                        "hidden_dim": hidden_dim,
                        "dropout": dropout,
                        "frame_height": 224,
                        "frame_width": 224,
                        "sample_fps": 12.0,
                        "window_seconds": 1.0,
                    },
                }
                torch.save(checkpoint, save_dir / "best.pt")
                logger.info(
                    f"  -> New best model (AUC: {val_metrics['auc']:.4f}, "
                    f"stage {stage_idx+1})"
                )
            else:
                epochs_without_improvement += 1

            # Save latest checkpoint
            checkpoint = {
                "epoch": global_epoch,
                "stage": stage_idx,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "history": history,
                "feature_classifier": True,
                "model_config": {
                    "backbone": backbone,
                    "temporal_aggregation": temporal_aggregation,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "frame_height": 224,
                    "frame_width": 224,
                    "sample_fps": 12.0,
                    "window_seconds": 1.0,
                },
            }
            torch.save(checkpoint, save_dir / "latest.pt")

            # Save history
            with open(save_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

            global_epoch += 1

            # Early stopping (global patience)
            if epochs_without_improvement >= patience:
                logger.info(
                    f"Early stopping at stage {stage_idx+1}, epoch {stage_epoch+1} "
                    f"(no improvement for {patience} epochs)"
                )
                break

        # Check if we should stop all stages
        if epochs_without_improvement >= patience:
            logger.info("Stopping all stages due to early stopping.")
            break

        logger.info(
            f"Stage {stage_idx+1} complete. "
            f"Best AUC so far: {history['best_val_auc']:.4f}"
        )

    # Generate training plot
    try:
        _plot_progressive_history(history, save_dir / "training_plot.png")
    except Exception as e:
        logger.warning(f"Failed to generate training plot: {e}")

    # Log final summary and artifact to wandb
    log_summary({
        "best_val_auc": history["best_val_auc"],
        "best_val_loss": history["best_val_loss"],
        "best_epoch": history["best_epoch"],
        "best_stage": history["best_stage"],
    })
    log_artifact(save_dir / "best.pt", artifact_type="model")
    finish_wandb()

    total_time = time.time() - total_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("Progressive training complete!")
    logger.info(
        f"  Total time: {total_time / 60:.1f} minutes, "
        f"{global_epoch} epochs across {n_stages} stages"
    )
    logger.info(
        f"  Best AUC: {history['best_val_auc']:.4f} "
        f"at epoch {history['best_epoch'] + 1} "
        f"(stage {history['best_stage'] + 1})"
    )
    logger.info(f"  Checkpoints saved to: {save_dir}")
    logger.info("=" * 60)


def _plot_progressive_history(history: dict, output_path: Path) -> None:
    """Plot progressive training history with stage boundaries."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history["train_losses"]) + 1)
    best_epoch = history["best_epoch"] + 1
    stages = history["stages"]

    # Find stage boundaries for vertical lines
    stage_boundaries = []
    for i in range(1, len(stages)):
        if stages[i] != stages[i - 1]:
            stage_boundaries.append(i + 1)  # 1-indexed

    def add_stage_lines(ax):
        for b in stage_boundaries:
            ax.axvline(x=b, color="orange", linestyle="--", alpha=0.6, linewidth=1.5)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_losses"], "b-", label="Train", linewidth=2)
    ax.plot(epochs, history["val_losses"], "r-", label="Val", linewidth=2)
    ax.axvline(
        x=best_epoch, color="g", linestyle="--", alpha=0.7,
        label=f"Best (epoch {best_epoch})"
    )
    add_stage_lines(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history["train_accs"], "b-", label="Train", linewidth=2)
    ax.plot(epochs, history["val_accs"], "r-", label="Val", linewidth=2)
    ax.axvline(x=best_epoch, color="g", linestyle="--", alpha=0.7)
    add_stage_lines(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy (orange = stage boundary)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AUC
    ax = axes[1, 0]
    ax.plot(epochs, history["val_aucs"], "m-", label="Val AUC", linewidth=2)
    ax.axvline(x=best_epoch, color="g", linestyle="--", alpha=0.7)
    ax.axhline(
        y=history["best_val_auc"], color="g", linestyle=":", alpha=0.5,
        label=f"Best: {history['best_val_auc']:.4f}",
    )
    add_stage_lines(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Validation AUC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 1]
    ax.plot(epochs, history["learning_rates"], "g-", linewidth=2)
    add_stage_lines(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        "Progressive Resolution Training (DINOv2 Features)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved training plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Progressive resolution training on pre-extracted DINOv2 features"
    )
    parser.add_argument(
        "--feature-dirs", nargs="+", required=True,
        help="Feature directories in resolution order (e.g., features_112 features_168 features_224)",
    )
    parser.add_argument(
        "--stage-epochs", nargs="+", type=int, required=True,
        help="Max epochs per stage (e.g., 10 10 15)",
    )
    parser.add_argument(
        "--stage-lrs", nargs="+", type=float, required=True,
        help="Peak LR per stage (e.g., 3e-4 1e-4 3e-5)",
    )
    parser.add_argument("--save-dir", required=True, help="Output directory")
    parser.add_argument("--backbone", default="dinov2-small")
    parser.add_argument("--temporal-aggregation", default="lstm",
                        choices=["mean", "max", "attention", "lstm"])
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.7)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--lr-schedule", default="cosine_restarts",
                        choices=["cosine", "cosine_restarts"])
    parser.add_argument("--lr-restart-period", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--group-by", default="subject_id",
                        choices=["subject_id", "video_path"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_progressive(
        feature_dirs=args.feature_dirs,
        stage_epochs=args.stage_epochs,
        stage_lrs=args.stage_lrs,
        save_dir=args.save_dir,
        backbone=args.backbone,
        temporal_aggregation=args.temporal_aggregation,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        label_smoothing=args.label_smoothing,
        mixup_alpha=args.mixup_alpha,
        lr_schedule=args.lr_schedule,
        lr_restart_period=args.lr_restart_period,
        batch_size=args.batch_size,
        val_split=args.val_split,
        group_by=args.group_by,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()