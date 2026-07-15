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


def _plot_history(
    history: dict,
    save_path: Path,
    inv_label_map: dict[int, str] | None = None,
) -> None:
    """Save a 2x3 PNG with loss/accuracy/AUC/LR plus per-(tier x class) panels.

    The bottom-middle and bottom-right panels are the motion-confound readout:
    per-class recall and mean predicted probability, stratified by holdout
    tier. If the model is exploiting motion artefacts rather than age signal,
    adult recall will drop sharply between the gold and salvageable tiers.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless — no X display required
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping history plot")
        return

    if not history.get("train_losses"):
        return

    n_epochs = len(history["train_losses"])
    epochs = list(range(1, n_epochs + 1))
    tiers = ("gold", "salvageable")

    # Tier colour map keeps lines for the same tier visually grouped across
    # the recall / mean-prob panels.
    tier_colors = {"gold": "tab:orange", "salvageable": "tab:red"}
    class_styles = {0: "-", 1: "--"}  # class 0 solid, class 1 dashed

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_losses"], label="train", lw=1.5)
    ax.plot(epochs, history["val_losses"], label="val", lw=1.5)
    for tier in tiers:
        key = f"holdout_{tier}_losses"
        if key in history and history[key]:
            ax.plot(range(1, len(history[key]) + 1), history[key],
                    label=f"holdout[{tier}]", lw=1.0, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend(fontsize=7, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), borderaxespad=0)
    ax.grid(alpha=0.3)

    # Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, history["train_accs"], label="train", lw=1.5)
    ax.plot(epochs, history["val_accs"], label="val", lw=1.5)
    for tier in tiers:
        key = f"holdout_{tier}_accs"
        if key in history and history[key]:
            ax.plot(range(1, len(history[key]) + 1), history[key],
                    label=f"holdout[{tier}]", lw=1.0, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.legend(fontsize=7, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), borderaxespad=0)
    ax.grid(alpha=0.3)

    # AUC (val + holdout tiers)
    ax = axes[0, 2]
    ax.plot(epochs, history["val_aucs"], label="val", lw=1.5)
    for tier in tiers:
        key = f"holdout_{tier}_aucs"
        if key in history and history[key]:
            ax.plot(range(1, len(history[key]) + 1), history[key],
                    label=f"holdout[{tier}]", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Validation AUC")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="gray", linestyle=":", lw=0.8, label="chance")
    ax.legend(fontsize=7, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), borderaxespad=0)
    ax.grid(alpha=0.3)

    # Learning rate
    ax = axes[1, 0]
    ax.plot(epochs, history["learning_rates"], color="purple", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("LR schedule")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")

    # Per-(tier x class) recall — the motion-confound readout
    ax = axes[1, 1]
    plotted_any = False
    for tier in tiers:
        for cls_label in (0, 1):
            key = f"holdout_{tier}_class{cls_label}_accs"
            if key not in history or not history[key]:
                continue
            cls_name = (inv_label_map or {}).get(cls_label, f"class{cls_label}")
            ax.plot(
                range(1, len(history[key]) + 1), history[key],
                label=f"{tier}/{cls_name}",
                color=tier_colors.get(tier),
                linestyle=class_styles.get(cls_label, "-"),
                lw=1.5,
            )
            plotted_any = True
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall (per-class acc)")
    ax.set_title("Holdout recall by tier × class")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color="gray", linestyle=":", lw=0.8)
    if plotted_any:
        ax.legend(fontsize=7, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), borderaxespad=0)
    ax.grid(alpha=0.3)

    # Per-(tier x class) mean predicted probability — confidence readout
    ax = axes[1, 2]
    plotted_any = False
    for tier in tiers:
        for cls_label in (0, 1):
            key = f"holdout_{tier}_class{cls_label}_mean_probs"
            if key not in history or not history[key]:
                continue
            cls_name = (inv_label_map or {}).get(cls_label, f"class{cls_label}")
            ax.plot(
                range(1, len(history[key]) + 1), history[key],
                label=f"{tier}/{cls_name}",
                color=tier_colors.get(tier),
                linestyle=class_styles.get(cls_label, "-"),
                lw=1.5,
            )
            plotted_any = True
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean P(label=1)")
    ax.set_title("Holdout mean predicted prob by tier × class")
    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color="gray", linestyle=":", lw=0.8)
    if plotted_any:
        ax.legend(fontsize=7, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), borderaxespad=0)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Training history — {save_path.parent.name} (epoch {n_epochs})")
    fig.tight_layout()
    try:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    except Exception as e:
        logger.warning("Failed to write history plot %s: %s", save_path, e)
    finally:
        plt.close(fig)


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


def compute_per_class_stats(logits, labels):
    """Break down recall and mean predicted probability by ground-truth class.

    Used to surface motion-vs-age confounds when both kids and adults appear
    in a holdout tier: a drop in recall for adults within the high-motion
    tier (relative to the clean tier) indicates the model is exploiting
    motion artefacts rather than developmental signal.

    Returns a dict keyed by integer class label: {n, acc, mean_prob}.
    """
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    labels_np = labels.cpu().numpy().astype(int)

    stats: dict[int, dict[str, float]] = {}
    for cls in sorted(np.unique(labels_np).tolist()):
        mask = labels_np == cls
        if mask.sum() == 0:
            continue
        stats[int(cls)] = {
            "n": int(mask.sum()),
            "acc": float((preds[mask] == cls).mean()),
            "mean_prob": float(probs[mask].mean()),
        }
    return stats


def _evaluate_loader(model, criterion, loader, return_per_class: bool = False):
    """Evaluate model on a dataloader.

    Returns (loss, acc, auc, f1) — or (loss, acc, auc, f1, per_class_stats)
    when return_per_class=True. Returns None if loader is empty.
    """
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"]
            labels = batch["label"]
            valid_mask = labels >= 0
            if valid_mask.sum() == 0:
                continue
            features = features[valid_mask]
            labels = labels[valid_mask]
            logits = model(features).squeeze(-1)
            loss = criterion(logits, labels)
            total_loss += loss.item() * features.size(0)
            all_logits.append(logits)
            all_labels.append(labels)

    if not all_logits:
        return None

    total_loss /= sum(l.size(0) for l in all_labels)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    acc, auc, f1 = compute_metrics(all_logits, all_labels)
    if return_per_class:
        per_class = compute_per_class_stats(all_logits, all_labels)
        return total_loss, acc, auc, f1, per_class
    return total_loss, acc, auc, f1


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
    include_tiers: list[str] | None = None,
    holdout_tiers: list[str] | None = None,
    plot_every: int = 5,
    window_idx_filter: str | None = None,
    save_final: bool = False,
) -> None:
    """Train a classifier on pre-extracted fNIRS features.

    Args:
        holdout_tiers: Additional quality tiers to evaluate each epoch as
            separate test sets (e.g. ["gold", "salvageable"]). These are
            evaluation-only — never used for training or early stopping.
            Requires feature_index.csv to have a quality_tier column.
        save_final: Also save the final-epoch checkpoint as last.pt (in
            addition to best-val best.pt). Used by the matched-optimizer-steps
            ablation, where pretrained and random encoders are compared at the
            same step count with no val-based epoch selection.
    """

    from synchronai.data.fnirs.feature_dataset import (
        create_fnirs_feature_dataloaders,
        filter_by_quality_tier,
        filter_by_window_idx,
        is_feature_dir_packed,
        load_fnirs_feature_index,
        split_fnirs_feature_entries,
        FnirsFeatureDataset,
        FnirsPackedFeatureDataset,
        _fnirs_feature_collate_fn,
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Parse label map
    label_map = {}
    for pair in label_map_str.split(","):
        key, val = pair.strip().split(":")
        label_map[key.strip()] = int(val.strip())
    inv_label_map = {v: k for k, v in label_map.items()}
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
            include_tiers=include_tiers,
            window_idx_filter=window_idx_filter,
        )
    )

    logger.info(f"Feature dim: {feature_dim}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    logger.info(f"Pos weight: {pos_weight:.3f}")

    # Build holdout-tier evaluation loaders (eval-only, never trained on)
    holdout_loaders: dict[str, torch.utils.data.DataLoader] = {}
    if holdout_tiers:
        feature_dir_path = Path(feature_dir)
        df = load_fnirs_feature_index(feature_dir_path)
        # Filter to valid labels
        if label_column in df.columns:
            df = df[df[label_column].isin(label_map.keys())]
        all_entries = df.to_dict("records")
        # Reproduce training's EXACT val subject partition by splitting the SAME
        # tier+window-filtered entries, in the same order and with the same seed.
        # Re-splitting the label-only entries diverges the seeded shuffle whenever
        # --include-tiers / --window-idx-filter drops whole subjects, leaking
        # trained subjects into the "holdout" set. Then keep holdout-tier entries
        # belonging ONLY to those val subjects -- including tiers excluded from
        # training (e.g. salvageable), which is the whole purpose of holdout eval.
        train_pool = filter_by_window_idx(
            filter_by_quality_tier(all_entries, include_tiers), window_idx_filter
        )
        _, val_train_split = split_fnirs_feature_entries(train_pool, val_split, seed)
        val_subjects = {e.get("subject_id") for e in val_train_split}
        val_entries = [e for e in all_entries if e.get("subject_id") in val_subjects]

        holdout_dataset_cls = (
            FnirsPackedFeatureDataset
            if is_feature_dir_packed(feature_dir_path)
            else FnirsFeatureDataset
        )
        for tier in holdout_tiers:
            tier_entries = filter_by_quality_tier(val_entries, [tier])
            if not tier_entries:
                logger.warning(f"Holdout tier '{tier}': no val entries, skipping")
                continue
            tier_dataset = holdout_dataset_cls(
                feature_dir_path, tier_entries, label_column, label_map
            )
            tier_loader = torch.utils.data.DataLoader(
                tier_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=_fnirs_feature_collate_fn,
            )
            holdout_loaders[tier] = tier_loader
            logger.info(
                f"Holdout tier '{tier}': {len(tier_entries)} val entries, "
                f"{len(tier_loader)} batches"
            )

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
        train_acc, train_auc, train_f1 = compute_metrics(all_train_logits, all_train_labels)

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
        history.setdefault("train_aucs", []).append(float(train_auc))
        history.setdefault("train_f1s", []).append(float(train_f1))
        history["val_aucs"].append(float(val_auc))
        history["val_f1s"].append(float(val_f1))
        history["learning_rates"].append(lr)

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
            f"AUC: {val_auc:.4f}, F1: {val_f1:.4f} | LR: {lr:.2e}"
        )

        # Evaluate on holdout tiers (eval-only, no effect on early stopping).
        # Per-class breakdown surfaces motion-vs-age confounds: a drop in
        # adult recall between gold and salvageable tiers means the model is
        # exploiting motion artefacts rather than developmental signal.
        for tier_name, tier_loader in holdout_loaders.items():
            tier_result = _evaluate_loader(
                model, criterion, tier_loader, return_per_class=True
            )
            if tier_result is None:
                continue
            t_loss, t_acc, t_auc, t_f1, t_per_class = tier_result
            history_key = f"holdout_{tier_name}"
            history.setdefault(f"{history_key}_aucs", []).append(float(t_auc))
            history.setdefault(f"{history_key}_accs", []).append(float(t_acc))
            history.setdefault(f"{history_key}_f1s", []).append(float(t_f1))
            history.setdefault(f"{history_key}_losses", []).append(float(t_loss))
            logger.info(
                f"  Holdout [{tier_name}] — "
                f"Loss: {t_loss:.4f}, Acc: {t_acc:.4f}, "
                f"AUC: {t_auc:.4f}, F1: {t_f1:.4f}"
            )
            for cls_label, cls_stats in t_per_class.items():
                cls_name = inv_label_map.get(cls_label, f"class{cls_label}")
                history.setdefault(
                    f"{history_key}_class{cls_label}_accs", []
                ).append(cls_stats["acc"])
                history.setdefault(
                    f"{history_key}_class{cls_label}_mean_probs", []
                ).append(cls_stats["mean_prob"])
                history.setdefault(
                    f"{history_key}_class{cls_label}_n", []
                ).append(cls_stats["n"])
                logger.info(
                    f"    {cls_name:<8} (n={cls_stats['n']:>4}): "
                    f"acc={cls_stats['acc']:.4f}, "
                    f"mean_prob={cls_stats['mean_prob']:.4f}"
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

        # Periodic history plot — writes a fresh history.png every
        # `plot_every` epochs so you can monitor training without parsing logs.
        if plot_every and (epoch % plot_every == 0):
            _plot_history(history, save_dir / "history.png", inv_label_map)

        # Early stopping
        if epochs_without_improvement >= patience:
            logger.info(
                f"Early stopping at epoch {epoch} "
                f"(no improvement for {patience} epochs)"
            )
            break

    # Final-epoch checkpoint (matched-optimizer-steps ablation): the comparison
    # uses this last.pt — no val-based epoch selection — so pretrained vs random
    # are compared at the identical step count.
    if save_final:
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_auc": val_auc,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "config": config,
        }, save_dir / "last.pt")
        logger.info(f"  Final-epoch checkpoint saved to: {save_dir / 'last.pt'} (epoch {epoch})")

    # Final history plot (guaranteed regardless of plot_every interval)
    _plot_history(history, save_dir / "history.png", inv_label_map)
    logger.info(f"Training complete!")
    logger.info(f"  Best AUC: {best_auc:.4f} at epoch {best_epoch}")
    logger.info(f"  Model saved to: {save_dir / 'best.pt'}")

    # Final motion-confound 2x2: per-(tier x class) recall at the best-AUC
    # epoch. A meaningful gap between gold/adult and salvageable/adult
    # indicates the model leans on motion artefacts rather than age signal.
    if holdout_loaders and best_epoch > 0:
        idx = best_epoch - 1
        logger.info("Final tier × class breakdown (best-AUC epoch):")
        header = f"  {'tier':<12} " + " ".join(
            f"{inv_label_map.get(c, f'class{c}'):>14}" for c in sorted(inv_label_map)
        )
        logger.info(header)
        for tier_name in holdout_loaders.keys():
            cells = []
            for cls_label in sorted(inv_label_map):
                acc_key = f"holdout_{tier_name}_class{cls_label}_accs"
                n_key = f"holdout_{tier_name}_class{cls_label}_n"
                if acc_key in history and len(history[acc_key]) > idx:
                    acc = history[acc_key][idx]
                    n = history[n_key][idx] if n_key in history else "?"
                    cells.append(f"acc={acc:.3f} n={n:>3}")
                else:
                    cells.append("       —      ")
            logger.info(f"  {tier_name:<12} " + " ".join(f"{c:>14}" for c in cells))


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
    parser.add_argument("--include-tiers", default=None,
                        help="Comma-separated quality tiers to include for training "
                             "(e.g. 'gold', 'gold,standard', 'salvageable'). "
                             "Filters feature_index.csv by quality_tier column.")
    parser.add_argument("--holdout-tiers", default=None,
                        help="Comma-separated quality tiers to evaluate each epoch "
                             "as separate test sets (e.g. 'gold,salvageable'). "
                             "Eval-only — never trained on or used for early stopping.")
    parser.add_argument("--plot-every", type=int, default=5,
                        help="Write history.png every N epochs (default: 5). "
                             "A final plot is always written at the end of training. "
                             "Set to 0 to disable periodic plots.")
    parser.add_argument("--window-idx-filter", default=None,
                        help="Restrict to entries matching a window-idx condition. "
                             "Format: '<op>:<int>' with op in {eq,gt,ge,lt,le}. "
                             "e.g. 'eq:0' → only the first window of each pair; "
                             "'gt:0' → all later windows.")
    parser.add_argument("--save-final", action="store_true",
                        help="Also save the final-epoch checkpoint as last.pt "
                             "(for the matched-optimizer-steps ablation). Combine "
                             "with --patience >= --epochs to disable early stopping.")

    args = parser.parse_args()

    include_tiers = [t.strip() for t in args.include_tiers.split(",")] if args.include_tiers else None
    holdout_tiers = [t.strip() for t in args.holdout_tiers.split(",")] if args.holdout_tiers else None

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
        include_tiers=include_tiers,
        holdout_tiers=holdout_tiers,
        plot_every=args.plot_every,
        window_idx_filter=args.window_idx_filter,
        save_final=args.save_final,
    )


if __name__ == "__main__":
    main()
