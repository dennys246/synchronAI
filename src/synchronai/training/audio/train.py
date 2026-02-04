"""
Audio classifier training module.

Provides:
- Training loop with loss tracking
- Checkpoint saving/loading
- Loss plotting over time
- Learning rate scheduling
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from synchronai.data.audio.dataset import (
    AudioClassificationDataset,
    create_audio_dataloaders,
)
from synchronai.models.audio.audio_classifier import (
    AudioClassifier,
    AudioClassifierConfig,
    build_audio_classifier,
    save_audio_classifier,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """Training history with loss tracking at both batch and epoch level."""

    # Epoch-level metrics
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_accs: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    best_val_loss: float = float("inf")
    best_val_acc: float = 0.0
    best_epoch: int = 0

    # Batch-level metrics for detailed progress tracking
    batch_losses: list[float] = field(default_factory=list)
    batch_accs: list[float] = field(default_factory=list)
    batch_indices: list[int] = field(default_factory=list)
    _global_batch: int = 0

    def add_batch_metrics(self, loss: float, accuracy: float) -> None:
        """Record loss and accuracy for a single batch."""
        self.batch_losses.append(float(loss))
        self.batch_accs.append(float(accuracy))
        self.batch_indices.append(self._global_batch)
        self._global_batch += 1

    def to_dict(self) -> dict:
        d = asdict(self)
        # Include the private field for serialization
        d["_global_batch"] = self._global_batch
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingHistory":
        # Handle older history files without batch-level fields
        data = data.copy()
        global_batch = data.pop("_global_batch", 0)
        # Remove any unknown fields for backwards compatibility
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        data = {k: v for k, v in data.items() if k in known_fields}
        instance = cls(**data)
        instance._global_batch = global_batch
        return instance

    def save(self, path: Union[str, Path]) -> None:
        """Save history to JSON."""
        data = self.to_dict()
        # Handle float("inf") which is not valid JSON
        if data.get("best_val_loss") == float("inf"):
            data["best_val_loss"] = None
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainingHistory":
        """Load history from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        # Handle None -> float("inf") conversion on load
        if data.get("best_val_loss") is None:
            data["best_val_loss"] = float("inf")
        return cls.from_dict(data)


@dataclass
class AudioTrainingConfig:
    """Configuration for audio classifier training."""

    # Data
    labels_file: str = ""
    val_split: float = 0.2

    # Training
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    use_amp: bool = True

    # Learning rate schedule
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Checkpoint
    save_every: int = 10  # Save checkpoint every N epochs
    early_stopping_patience: int = 15

    # Data loading
    num_workers: int = 4
    seed: int = 42

    # Loss weighting
    use_class_weights: bool = True


def plot_training_history(
    history: TrainingHistory,
    output_path: Union[str, Path],
    title: str = "Audio Classifier Training",
) -> Path:
    """Plot training history and save to file.

    Args:
        history: Training history
        output_path: Path to save plot
        title: Plot title

    Returns:
        Path to saved plot
    """
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs = range(1, len(history.train_losses) + 1)

    # Loss plot
    ax = axes[0, 0]
    ax.plot(epochs, history.train_losses, "b-", label="Train Loss", linewidth=2)
    ax.plot(epochs, history.val_losses, "r-", label="Val Loss", linewidth=2)
    ax.axvline(x=history.best_epoch + 1, color="g", linestyle="--", alpha=0.7, label=f"Best (epoch {history.best_epoch + 1})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy plot
    ax = axes[0, 1]
    ax.plot(epochs, history.train_accs, "b-", label="Train Acc", linewidth=2)
    ax.plot(epochs, history.val_accs, "r-", label="Val Acc", linewidth=2)
    ax.axvline(x=history.best_epoch + 1, color="g", linestyle="--", alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate plot
    ax = axes[1, 0]
    ax.plot(epochs, history.learning_rates, "g-", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Summary text
    ax = axes[1, 1]
    ax.axis("off")
    summary_text = f"""
Training Summary
================

Total Epochs: {len(history.train_losses)}
Best Epoch: {history.best_epoch + 1}

Best Validation Loss: {history.best_val_loss:.4f}
Best Validation Accuracy: {history.best_val_acc:.2%}

Final Train Loss: {history.train_losses[-1]:.4f}
Final Val Loss: {history.val_losses[-1]:.4f}

Final Train Accuracy: {history.train_accs[-1]:.2%}
Final Val Accuracy: {history.val_accs[-1]:.2%}
"""
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
            verticalalignment="top", fontfamily="monospace")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved training plot to {output_path}")
    return output_path


def plot_batch_progress(
    history: TrainingHistory,
    output_path: Union[str, Path],
    title: str = "Audio Classifier Batch Progress",
) -> Path:
    """Plot batch-level training progress and save to file.

    Args:
        history: Training history with batch-level metrics
        output_path: Path to save plot
        title: Plot title

    Returns:
        Path to saved plot
    """
    import matplotlib.pyplot as plt

    if not history.batch_losses:
        return Path(output_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    batches = np.array(history.batch_indices)
    losses = np.array(history.batch_losses)
    accs = np.array(history.batch_accs)

    # Top plot: Loss vs Batch
    ax1 = axes[0]
    ax1.plot(batches, losses, alpha=0.3, color='blue', linewidth=0.5, label='Batch Loss')

    # Plot smoothed loss (moving average)
    if len(losses) > 10:
        window_size = min(50, len(losses) // 5)
        if window_size > 1:
            smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            smoothed_x = batches[window_size-1:]
            ax1.plot(smoothed_x, smoothed, color='blue', linewidth=1.5, label=f'Smoothed (window={window_size})')

    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} - Batch Loss')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Accuracy vs Batch
    ax2 = axes[1]
    ax2.plot(batches, accs, alpha=0.3, color='green', linewidth=0.5, label='Batch Accuracy')

    # Plot smoothed accuracy
    if len(accs) > 10:
        window_size = min(50, len(accs) // 5)
        if window_size > 1:
            smoothed = np.convolve(accs, np.ones(window_size)/window_size, mode='valid')
            smoothed_x = batches[window_size-1:]
            ax2.plot(smoothed_x, smoothed, color='green', linewidth=1.5, label=f'Smoothed (window={window_size})')

    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title} - Batch Accuracy')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def train_audio_classifier(
    labels_file: Union[str, Path],
    save_dir: Union[str, Path] = "runs/audio_classifier",
    model_config: Optional[AudioClassifierConfig] = None,
    training_config: Optional[AudioTrainingConfig] = None,
    resume_from: Optional[str] = None,
) -> tuple[AudioClassifier, TrainingHistory]:
    """Train audio classifier.

    Args:
        labels_file: Path to labels CSV
        save_dir: Directory to save checkpoints and plots
        model_config: Model configuration
        training_config: Training configuration
        resume_from: Path to checkpoint to resume from

    Returns:
        Tuple of (trained model, training history)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Default configs
    if model_config is None:
        model_config = AudioClassifierConfig()
    if training_config is None:
        training_config = AudioTrainingConfig(labels_file=str(labels_file))
    else:
        training_config.labels_file = str(labels_file)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on device: {device}")

    # Create dataloaders
    train_loader, val_loader, dataset = create_audio_dataloaders(
        labels_file=labels_file,
        batch_size=training_config.batch_size,
        val_split=training_config.val_split,
        num_workers=training_config.num_workers,
        seed=training_config.seed,
        event_classes=model_config.event_classes,
    )

    # Log class distribution
    class_dist = dataset.get_class_distribution()
    logger.info(f"Class distribution: {class_dist}")

    # Build model
    model = build_audio_classifier(model_config)
    model.to(device)

    # Class weights for imbalanced data
    class_weights = None
    if training_config.use_class_weights:
        class_weights = dataset.get_class_weights().to(device)
        logger.info(f"Using class weights: {class_weights.tolist()}")

    # Loss function
    event_criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    # Learning rate scheduler (warmup + cosine annealing)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=training_config.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=training_config.epochs - training_config.warmup_epochs,
        eta_min=training_config.min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[training_config.warmup_epochs],
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if training_config.use_amp and device == "cuda" else None

    # Training history
    history = TrainingHistory()

    # Resume from checkpoint
    start_epoch = 0
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "history" in checkpoint:
            history = TrainingHistory.from_dict(checkpoint["history"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        logger.info(f"Resumed from checkpoint at epoch {start_epoch}")

    # Save configs
    with open(save_dir / "model_config.json", "w") as f:
        json.dump(asdict(model_config), f, indent=2)
    with open(save_dir / "training_config.json", "w") as f:
        json.dump(asdict(training_config), f, indent=2)

    # Training loop
    early_stop_counter = 0
    batch_plot_interval = 10  # Plot batch progress every N batches

    for epoch in range(start_epoch, training_config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            audio = batch["audio"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(audio)

                # Classification loss
                loss = event_criterion(outputs["event_logits"], label)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            # Compute batch metrics
            batch_loss = loss.item()
            _, predicted = outputs["event_logits"].max(1)
            batch_correct = predicted.eq(label).sum().item()
            batch_acc = batch_correct / audio.size(0)

            # Accumulate for epoch metrics
            train_loss += batch_loss * audio.size(0)
            train_correct += batch_correct
            train_total += audio.size(0)

            # Record batch metrics to history
            history.add_batch_metrics(batch_loss, batch_acc)

            # Plot batch progress at intervals
            if (
                batch_plot_interval > 0
                and history._global_batch % batch_plot_interval == 0
            ):
                plot_batch_progress(
                    history,
                    save_dir / "batch_progress.png",
                    title="Audio Classifier Batch Progress",
                )
                # Also save history incrementally
                history.save(save_dir / "history.json")

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device)
                label = batch["label"].to(device)

                outputs = model(audio)
                loss = event_criterion(outputs["event_logits"], label)

                val_loss += loss.item() * audio.size(0)
                _, predicted = outputs["event_logits"].max(1)
                val_correct += predicted.eq(label).sum().item()
                val_total += audio.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Update history
        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)
        history.train_accs.append(train_acc)
        history.val_accs.append(val_acc)
        history.learning_rates.append(current_lr)

        # Log progress
        logger.info(
            f"Epoch {epoch + 1}/{training_config.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | "
            f"LR: {current_lr:.2e}"
        )

        # Check for best model
        is_best = val_loss < history.best_val_loss
        if is_best:
            history.best_val_loss = val_loss
            history.best_val_acc = val_acc
            history.best_epoch = epoch
            early_stop_counter = 0

            # Save best model
            save_audio_classifier(
                model,
                save_dir / "best.pt",
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_loss": val_loss, "val_acc": val_acc},
            )
            logger.info(f"  -> New best model saved (val_loss: {val_loss:.4f})")
        else:
            early_stop_counter += 1

        # Save periodic checkpoint
        if (epoch + 1) % training_config.save_every == 0:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history.to_dict(),
                "config": asdict(model_config),
            }, checkpoint_path)

        # Save latest checkpoint (for resume)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history.to_dict(),
            "config": asdict(model_config),
        }, save_dir / "latest.pt")

        # Save history after each epoch (ensures epoch metrics are persisted)
        history.save(save_dir / "history.json")

        # Early stopping
        if early_stop_counter >= training_config.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

        # Plot training progress periodically
        if (epoch + 1) % 5 == 0 or epoch == training_config.epochs - 1:
            plot_training_history(
                history,
                save_dir / "training_plot.png",
                title=f"Audio Classifier Training (Epoch {epoch + 1})",
            )

    # Final plots
    plot_training_history(
        history,
        save_dir / "training_plot.png",
        title="Audio Classifier Training (Final)",
    )
    plot_batch_progress(
        history,
        save_dir / "batch_progress.png",
        title="Audio Classifier Batch Progress (Final)",
    )

    # Save history
    history.save(save_dir / "history.json")

    logger.info(f"Training complete! Best val_loss: {history.best_val_loss:.4f} at epoch {history.best_epoch + 1}")
    logger.info(f"Checkpoints saved to: {save_dir}")

    return model, history
