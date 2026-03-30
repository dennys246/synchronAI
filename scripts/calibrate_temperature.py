#!/usr/bin/env python3
"""Post-hoc temperature scaling for calibrating model predictions.

Temperature scaling (Guo et al., 2017) learns a single scalar T that divides
logits before sigmoid, improving calibration without affecting AUC or accuracy.

Usage:
    python scripts/calibrate_temperature.py \
        --checkpoint runs/dinov2_progressive/prog_full/best.pt \
        --feature-dir data/dinov2_features_small_meanpatch \
        --output runs/dinov2_progressive/prog_full/calibrated.pt

    # Also works as a standalone evaluation:
    python scripts/calibrate_temperature.py \
        --checkpoint runs/dinov2_sweep/small_heavy_reg/best.pt \
        --feature-dir data/dinov2_features_small_meanpatch
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS

_script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_script_dir.parent))  # project root (for synchronai.*)
sys.path.insert(0, str(_script_dir))          # scripts/ dir (for train_from_features)

from synchronai.data.video.feature_dataset import create_feature_dataloaders
from train_from_features import FeatureClassifier, compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class TemperatureScaler(nn.Module):
    """Wraps a model with a learned temperature parameter."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.model(features)
        return logits / self.temperature


def collect_logits_and_labels(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect all logits and labels from a dataloader."""
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            labels = batch["label"]

            logits = model(features)
            logits = logits.squeeze(-1) if logits.dim() > 1 else logits

            all_logits.append(logits.cpu())
            all_labels.append(labels)

    return torch.cat(all_logits), torch.cat(all_labels)


def find_optimal_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Find optimal temperature using L-BFGS on NLL loss.

    Args:
        logits: Pre-sigmoid logits from the model
        labels: Ground truth binary labels

    Returns:
        Optimal temperature value
    """
    temperature = nn.Parameter(torch.ones(1))
    criterion = nn.BCEWithLogitsLoss()

    optimizer = LBFGS([temperature], lr=0.01, max_iter=100)

    def closure():
        optimizer.zero_grad()
        scaled_logits = logits / temperature
        loss = criterion(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    return temperature.item()


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    ECE measures how well predicted probabilities match actual frequencies.
    A perfectly calibrated model has ECE = 0.

    Args:
        probs: Predicted probabilities
        labels: Binary ground truth labels
        n_bins: Number of bins for calibration

    Returns:
        ECE value (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += mask.sum() / len(probs) * abs(bin_acc - bin_conf)

    return ece


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc temperature scaling for model calibration"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--feature-dir", required=True, help="Feature directory for validation")
    parser.add_argument("--output", default=None,
                        help="Output path for calibrated checkpoint (default: same dir as input)")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--group-by", default="subject_id")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = ckpt.get("model_config", ckpt.get("config", {}))

    # Reconstruct model
    _, val_loader, _, feature_dim, n_frames = create_feature_dataloaders(
        feature_dir=args.feature_dir,
        batch_size=64,
        val_split=args.val_split,
        group_by=args.group_by,
        num_workers=0,
        seed=args.seed,
    )

    model = FeatureClassifier(
        feature_dim=feature_dim,
        hidden_dim=model_config.get("hidden_dim", 128),
        temporal_aggregation=model_config.get("temporal_aggregation", "lstm"),
        dropout=model_config.get("dropout", 0.5),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Collect validation logits
    logger.info("Collecting validation logits...")
    logits, labels = collect_logits_and_labels(model, val_loader, device)
    logger.info(f"Collected {len(logits)} validation samples")

    # Pre-calibration metrics
    probs_before = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy()
    ece_before = expected_calibration_error(probs_before, labels_np)
    metrics_before = compute_metrics(logits, labels)

    logger.info(f"Before calibration:")
    logger.info(f"  AUC: {metrics_before['auc']:.4f}")
    logger.info(f"  Accuracy: {metrics_before['accuracy']:.4f}")
    logger.info(f"  ECE: {ece_before:.4f}")

    # Find optimal temperature
    logger.info("Finding optimal temperature...")
    optimal_temp = find_optimal_temperature(logits.clone(), labels.clone())
    logger.info(f"Optimal temperature: {optimal_temp:.4f}")

    # Post-calibration metrics
    scaled_logits = logits / optimal_temp
    probs_after = torch.sigmoid(scaled_logits).numpy()
    ece_after = expected_calibration_error(probs_after, labels_np)
    metrics_after = compute_metrics(scaled_logits, labels)

    logger.info(f"After calibration (T={optimal_temp:.4f}):")
    logger.info(f"  AUC: {metrics_after['auc']:.4f} (should be unchanged)")
    logger.info(f"  Accuracy: {metrics_after['accuracy']:.4f}")
    logger.info(f"  ECE: {ece_after:.4f} (was {ece_before:.4f})")

    # Save calibrated checkpoint
    if args.output is None:
        output_path = Path(args.checkpoint).parent / "calibrated.pt"
    else:
        output_path = Path(args.output)

    ckpt["temperature"] = optimal_temp
    ckpt["calibration"] = {
        "temperature": optimal_temp,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "auc": metrics_after["auc"],
    }
    torch.save(ckpt, output_path)
    logger.info(f"Saved calibrated checkpoint: {output_path}")

    # Also save calibration report as JSON
    report = {
        "temperature": optimal_temp,
        "before": {
            "auc": metrics_before["auc"],
            "accuracy": metrics_before["accuracy"],
            "f1": metrics_before["f1"],
            "ece": ece_before,
        },
        "after": {
            "auc": metrics_after["auc"],
            "accuracy": metrics_after["accuracy"],
            "f1": metrics_after["f1"],
            "ece": ece_after,
        },
    }
    report_path = output_path.parent / "calibration_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved calibration report: {report_path}")


if __name__ == "__main__":
    main()