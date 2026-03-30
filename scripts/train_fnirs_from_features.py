#!/usr/bin/env python3
"""
Train a classifier on pre-extracted fNIRS encoder features.

Supports two tasks:
1. Child vs adult classification (validation of encoder quality)
2. Synchrony classification (after validation passes)

Follows the same training pattern as train_audio_from_features.py:
- AdamW optimizer, cosine LR schedule
- BCEWithLogitsLoss with pos_weight
- Early stopping on val AUC
- History JSON + checkpoint saving

Usage:
    python scripts/train_fnirs_from_features.py \
        --feature-dir data/fnirs_encoder_features \
        --save-dir runs/fnirs_classifier \
        --label-column participant_type \
        --label-map "child:0,adult:1" \
        --hidden-dim 64 \
        --epochs 50
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class FnirsFeatureClassifier(nn.Module):
    """Classifier for pre-extracted fNIRS encoder features.

    Supports:
    - Linear probe: hidden_dim=0, pool="mean", just mean pool + Linear
    - MLP: hidden_dim>0, pool="mean"/"max", pool + MLP head
    - LSTM: hidden_dim>0, pool="lstm", LSTM temporal model + MLP head
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 0,
        dropout: float = 0.3,
        output_dim: int = 1,
        pool: str = "mean",
    ):
        super().__init__()
        self.pool_type = pool

        # Temporal aggregation
        if pool == "lstm" and hidden_dim > 0:
            self.lstm = nn.LSTM(
                feature_dim, hidden_dim,
                batch_first=True, dropout=dropout if dropout > 0 else 0,
            )
            head_input_dim = hidden_dim
        else:
            self.lstm = None
            head_input_dim = feature_dim

        # Classification head
        if hidden_dim > 0 and pool != "lstm":
            self.head = nn.Sequential(
                nn.Linear(head_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )
        elif pool == "lstm":
            # LSTM already reduces dim; just add a light head
            self.head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(head_input_dim, output_dim),
            )
        else:
            # Linear probe
            self.head = nn.Linear(head_input_dim, output_dim)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"FnirsFeatureClassifier: feature_dim={feature_dim}, "
            f"hidden_dim={hidden_dim}, pool={pool}, "
            f"total_params={total_params:,}"
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) pooled or (B, T, D) temporal

        Returns:
            Logits (B, output_dim)
        """
        if features.ndim == 3:
            if self.lstm is not None:
                # LSTM over temporal sequence, take last hidden state
                lstm_out, (h_n, _) = self.lstm(features)
                features = h_n.squeeze(0)  # (B, hidden_dim)
            elif self.pool_type == "mean":
                features = features.mean(dim=1)
            elif self.pool_type == "max":
                features = features.max(dim=1).values
            else:
                raise ValueError(f"Unknown pool: {self.pool_type}")

        return self.head(features)


def compute_metrics(logits, labels):
    """Compute accuracy, AUC, F1 from logits and labels."""
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    labels_np = labels.cpu().numpy().astype(int)

    acc = (preds == labels_np).mean()

    try:
        auc = roc_auc_score(labels_np, probs)
    except ValueError:
        auc = 0.5

    try:
        f1 = f1_score(labels_np, preds, average="binary")
    except ValueError:
        f1 = 0.0

    return acc, auc, f1


def train_fnirs_from_features(
    feature_dir: str,
    save_dir: str,
    label_column: str = "participant_type",
    label_map_str: str = "child:0,adult:1",
    hidden_dim: int = 0,
    dropout: float = 0.3,
    pool: str = "mean",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    warmup_epochs: int = 3,
    patience: int = 15,
    val_split: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
) -> None:
    """Train a classifier on pre-extracted fNIRS features."""

    from synchronai.data.fnirs.feature_dataset import create_fnirs_feature_dataloaders

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Parse label map
    label_map = {}
    for pair in label_map_str.split(","):
        key, val = pair.strip().split(":")
        label_map[key.strip()] = int(val.strip())
    logger.info(f"Label map: {label_map}")

    # Load data
    train_loader, val_loader, pos_weight, feature_dim = (
        create_fnirs_feature_dataloaders(
            feature_dir=feature_dir,
            batch_size=batch_size,
            val_split=val_split,
            label_column=label_column,
            label_map=label_map,
            num_workers=num_workers,
            seed=seed,
        )
    )

    logger.info(f"Feature dim: {feature_dim}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    logger.info(f"Pos weight: {pos_weight:.3f}")

    if len(train_loader) == 0 or len(val_loader) == 0:
        logger.error(
            "No valid training or validation samples! Check that feature_index.csv "
            "has entries with valid labels matching the label map."
        )
        return

    # Create model
    model = FnirsFeatureClassifier(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        pool=pool,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight])
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # LR schedule: warmup + cosine decay
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # Save config
    config = {
        "feature_dir": str(feature_dir),
        "label_column": label_column,
        "label_map": label_map,
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "pool": pool,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "patience": patience,
        "seed": seed,
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    history = {
        "train_losses": [], "val_losses": [],
        "train_accs": [], "val_accs": [],
        "val_aucs": [], "val_f1s": [],
        "learning_rates": [],
    }

    best_auc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        all_train_logits = []
        all_train_labels = []

        for batch in train_loader:
            features = batch["features"]
            labels = batch["label"]

            # Skip invalid labels
            valid_mask = labels >= 0
            if valid_mask.sum() == 0:
                continue
            features = features[valid_mask]
            labels = labels[valid_mask]

            optimizer.zero_grad()
            logits = model(features).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * features.size(0)
            all_train_logits.append(logits.detach())
            all_train_labels.append(labels.detach())

        scheduler.step()

        if not all_train_logits:
            logger.warning(f"Epoch {epoch}: no valid training samples")
            continue

        train_loss /= sum(l.size(0) for l in all_train_labels)
        all_train_logits = torch.cat(all_train_logits)
        all_train_labels = torch.cat(all_train_labels)
        train_acc, _, _ = compute_metrics(all_train_logits, all_train_labels)

        # Validate
        model.eval()
        val_loss = 0.0
        all_val_logits = []
        all_val_labels = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"]
                labels = batch["label"]

                valid_mask = labels >= 0
                if valid_mask.sum() == 0:
                    continue
                features = features[valid_mask]
                labels = labels[valid_mask]

                logits = model(features).squeeze(-1)
                loss = criterion(logits, labels)

                val_loss += loss.item() * features.size(0)
                all_val_logits.append(logits)
                all_val_labels.append(labels)

        if not all_val_logits:
            logger.warning(f"Epoch {epoch}: no valid validation samples")
            continue

        val_loss /= sum(l.size(0) for l in all_val_labels)
        all_val_logits = torch.cat(all_val_logits)
        all_val_labels = torch.cat(all_val_labels)
        val_acc, val_auc, val_f1 = compute_metrics(all_val_logits, all_val_labels)

        lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_accs"].append(float(train_acc))
        history["val_accs"].append(float(val_acc))
        history["val_aucs"].append(float(val_auc))
        history["val_f1s"].append(float(val_f1))
        history["learning_rates"].append(lr)

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
            f"AUC: {val_auc:.4f}, F1: {val_f1:.4f} | LR: {lr:.2e}"
        )

        # Check for improvement
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": val_auc,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "config": config,
            }, save_dir / "best.pt")
        else:
            epochs_without_improvement += 1

        # Save history
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)"
            )
            break

    logger.info(f"Training complete!")
    logger.info(f"  Best AUC: {best_auc:.4f} at epoch {best_epoch}")
    logger.info(f"  Model saved to: {save_dir / 'best.pt'}")


def main():
    parser = argparse.ArgumentParser(
        description="Train classifier on pre-extracted fNIRS features"
    )
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--label-column", default="participant_type",
                        help="Column in feature_index.csv to use as label")
    parser.add_argument("--label-map", default="child:0,adult:1",
                        help="Comma-separated key:value pairs for label encoding")
    parser.add_argument("--hidden-dim", type=int, default=0,
                        help="Hidden dim (0=linear probe, >0=MLP)")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--pool", default="mean", choices=["mean", "max", "lstm"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_fnirs_from_features(
        feature_dir=args.feature_dir,
        save_dir=args.save_dir,
        label_column=args.label_column,
        label_map_str=args.label_map,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        pool=args.pool,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
