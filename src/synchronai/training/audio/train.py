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
from synchronai.utils.wandb_utils import (
    init_wandb,
    log_metrics,
    log_summary,
    log_artifact,
    finish_wandb,
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
    group_by: str = "subject_id"

    # Training
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    use_amp: bool = True
    gradient_clip_max_norm: float = 1.0
    label_smoothing: float = 0.05

    # Two-stage fine-tuning
    stage1_epochs: int = 5
    encoder_lr: float = 1e-5
    stage2_warmup_epochs: int = 3

    # Learning rate schedule
    warmup_epochs: int = 3
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


def _run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
    use_amp: bool = True,
    gradient_clip_max_norm: float = 0.0,
    label_smoothing: float = 0.0,
    history: Optional[TrainingHistory] = None,
    batch_plot_interval: int = 10,
    save_dir: Optional[Path] = None,
) -> tuple[float, float]:
    """Run a single training or validation epoch.

    Args:
        model: Model to train/evaluate
        dataloader: DataLoader
        criterion: Loss function
        device: Device
        optimizer: Optimizer (None for validation)
        scaler: GradScaler for AMP (None for validation or no AMP)
        use_amp: Whether to use automatic mixed precision
        gradient_clip_max_norm: Max norm for gradient clipping (0 = disabled)
        label_smoothing: Label smoothing factor (applied as uniform noise)
        history: TrainingHistory for batch-level tracking (training only)
        batch_plot_interval: Plot batch progress every N batches
        save_dir: Directory to save batch progress plots

    Returns:
        Tuple of (avg_loss, accuracy)
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(dataloader):
            audio = batch["audio"].to(device)
            label = batch["label"].to(device)

            if is_training:
                optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(audio)
                logits = outputs["event_logits"]
                if is_training and label_smoothing > 0:
                    num_classes = logits.shape[-1]
                    if num_classes < 2:
                        raise ValueError("Label smoothing requires at least 2 classes.")
                    # Convert hard labels to soft: mix with uniform distribution
                    smooth_label = torch.zeros(label.size(0), num_classes, device=device)
                    smooth_label.fill_(label_smoothing / (num_classes - 1))
                    smooth_label.scatter_(1, label.unsqueeze(1), 1.0 - label_smoothing)
                    # Manual cross-entropy with soft labels
                    log_probs = F.log_softmax(logits, dim=-1)
                    loss = -(smooth_label * log_probs).sum(dim=-1).mean()
                    if criterion.weight is not None:
                        # Apply class weights to soft-label loss
                        weight_per_sample = criterion.weight[label]
                        loss = (-(smooth_label * log_probs).sum(dim=-1) * weight_per_sample).mean()
                else:
                    loss = criterion(logits, label)

            if is_training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    if gradient_clip_max_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if gradient_clip_max_norm > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_max_norm)
                    optimizer.step()

            batch_loss = loss.item()
            _, predicted = outputs["event_logits"].max(1)
            batch_correct = predicted.eq(label).sum().item()
            batch_acc = batch_correct / audio.size(0)

            total_loss += batch_loss * audio.size(0)
            total_correct += batch_correct
            total_samples += audio.size(0)

            # Record batch metrics (training only)
            if is_training and history is not None:
                history.add_batch_metrics(batch_loss, batch_acc)
                if (
                    batch_plot_interval > 0
                    and save_dir is not None
                    and history._global_batch % batch_plot_interval == 0
                ):
                    plot_batch_progress(
                        history,
                        save_dir / "batch_progress.png",
                        title="Audio Classifier Batch Progress",
                    )
                    history.save(save_dir / "history.json")

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def train_audio_classifier(
    labels_file: Union[str, Path],
    save_dir: Union[str, Path] = "runs/audio_classifier",
    model_config: Optional[AudioClassifierConfig] = None,
    training_config: Optional[AudioTrainingConfig] = None,
    resume_from: Optional[str] = None,
) -> tuple[AudioClassifier, TrainingHistory]:
    """Train audio classifier with two-stage fine-tuning.

    Stage 1: Freeze Whisper encoder, train classification head only.
    Stage 2: Unfreeze encoder with lower LR, fine-tune end-to-end.

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

    # Create dataloaders (group-based split to prevent leakage)
    train_loader, val_loader, dataset = create_audio_dataloaders(
        labels_file=labels_file,
        batch_size=training_config.batch_size,
        val_split=training_config.val_split,
        num_workers=training_config.num_workers,
        seed=training_config.seed,
        event_classes=model_config.event_classes,
        group_by=training_config.group_by,
    )

    # Log class distribution
    class_dist = dataset.get_class_distribution()
    logger.info(f"Class distribution: {class_dist}")
    unique_labels = sorted(class_dist.keys())
    if unique_labels and max(unique_labels) <= 1 and len(model_config.event_classes) > 2:
        raise ValueError(
            "Audio labels appear to be binary (0/1), but event_classes has "
            f"{len(model_config.event_classes)} classes. This would train the "
            "7-class event head on synchrony labels. Provide real event labels "
            "or switch to a binary audio synchrony head instead."
        )

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

    # Training history
    history = TrainingHistory()

    # Resume from checkpoint
    start_epoch = 0
    resume_checkpoint: Optional[dict] = None
    resume_stage = 1
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "history" in checkpoint:
            history = TrainingHistory.from_dict(checkpoint["history"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        resume_stage = checkpoint.get("stage", 1)
        resume_checkpoint = checkpoint
        logger.info(f"Resumed from checkpoint at epoch {start_epoch}")

    # Save configs
    with open(save_dir / "model_config.json", "w") as f:
        json.dump(asdict(model_config), f, indent=2)
    with open(save_dir / "training_config.json", "w") as f:
        json.dump(asdict(training_config), f, indent=2)

    # Initialize wandb
    init_wandb(
        config={"model": asdict(model_config), "training": asdict(training_config)},
        name=f"audio-{model_config.model_name}",
        tags=["audio-classifier", model_config.model_name],
        group="audio-training",
        save_dir=save_dir,
    )

    # Mixed precision
    use_amp = training_config.use_amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    early_stop_counter = 0

    # ========== Stage 1: Head-only training (encoder frozen) ==========
    logger.info(f"Stage 1: Training head only for {training_config.stage1_epochs} epochs")
    model.freeze_encoder()

    # Only optimize trainable parameters (head layers)
    optimizer = AdamW(
        model.get_parameter_groups(
            encoder_lr=0.0,  # Encoder frozen, won't be in groups
            head_lr=training_config.learning_rate,
        ),
        weight_decay=training_config.weight_decay,
    )

    # LR schedule for stage 1: warmup + cosine
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=training_config.warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, training_config.stage1_epochs - training_config.warmup_epochs),
        eta_min=training_config.min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[training_config.warmup_epochs],
    )

    if resume_checkpoint is not None and resume_stage == 1 and start_epoch > 0:
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in resume_checkpoint:
            scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])

    stage1_start = min(start_epoch, training_config.stage1_epochs)
    for epoch in range(stage1_start, training_config.stage1_epochs):
        logger.info(f"Epoch {epoch + 1}/{training_config.stage1_epochs} (Stage 1)")

        train_loss, train_acc = _run_epoch(
            model, train_loader, event_criterion, device,
            optimizer=optimizer, scaler=scaler,
            use_amp=use_amp,
            gradient_clip_max_norm=training_config.gradient_clip_max_norm,
            label_smoothing=training_config.label_smoothing,
            history=history, save_dir=save_dir,
        )

        val_loss, val_acc = _run_epoch(
            model, val_loader, event_criterion, device,
            use_amp=use_amp,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Update history
        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)
        history.train_accs.append(train_acc)
        history.val_accs.append(val_acc)
        history.learning_rates.append(current_lr)

        logger.info(
            f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | LR: {current_lr:.2e}"
        )

        # Log to wandb
        log_metrics({
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "lr": current_lr,
            "stage": 1,
            "epoch": epoch,
        })

        is_best = val_loss < history.best_val_loss
        if is_best:
            history.best_val_loss = val_loss
            history.best_val_acc = val_acc
            history.best_epoch = epoch
            early_stop_counter = 0
            save_audio_classifier(
                model, save_dir / "best.pt",
                optimizer=optimizer, epoch=epoch,
                metrics={"val_loss": val_loss, "val_acc": val_acc},
            )
            logger.info(f"  -> New best model (val_loss: {val_loss:.4f})")
        else:
            early_stop_counter += 1

        # Save latest checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history.to_dict(),
            "config": asdict(model_config),
            "stage": 1,
        }
        if scaler is not None:
            checkpoint["scaler_state_dict"] = scaler.state_dict()
        torch.save(checkpoint, save_dir / "latest.pt")
        history.save(save_dir / "history.json")

    # ========== Stage 2: End-to-end fine-tuning (encoder unfrozen) ==========
    stage2_epochs = training_config.epochs - training_config.stage1_epochs
    if stage2_epochs > 0:
        early_stop_counter = 0
        logger.info(f"Stage 2: Fine-tuning encoder + head for {stage2_epochs} epochs")
        model.unfreeze_encoder()

        # Reduce head LR — head is already well-trained from stage 1
        stage2_head_lr = training_config.encoder_lr * 5
        logger.info(
            f"  Stage 2 LRs: encoder={training_config.encoder_lr:.1e}, "
            f"head={stage2_head_lr:.1e}"
        )

        # New optimizer with differential LRs (only trainable params)
        optimizer = AdamW(
            model.get_parameter_groups(
                encoder_lr=training_config.encoder_lr,
                head_lr=stage2_head_lr,
            ),
            weight_decay=training_config.weight_decay,
        )

        if use_amp:
            scaler = torch.amp.GradScaler("cuda")

        # Scheduler for stage 2: gentle warmup + cosine
        if training_config.stage2_warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.3,
                total_iters=training_config.stage2_warmup_epochs,
            )
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, stage2_epochs - training_config.stage2_warmup_epochs),
                eta_min=training_config.min_lr,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[training_config.stage2_warmup_epochs],
            )
        else:
            scheduler = CosineAnnealingLR(
                optimizer, T_max=stage2_epochs, eta_min=training_config.min_lr,
            )

        if resume_checkpoint is not None and resume_stage == 2 and start_epoch > training_config.stage1_epochs:
            if "optimizer_state_dict" in resume_checkpoint:
                optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in resume_checkpoint:
                scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            if scaler is not None and "scaler_state_dict" in resume_checkpoint:
                scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])

        stage2_start = max(0, start_epoch - training_config.stage1_epochs)
        for epoch in range(stage2_start, stage2_epochs):
            global_epoch = training_config.stage1_epochs + epoch
            logger.info(f"Epoch {global_epoch + 1}/{training_config.epochs} (Stage 2)")

            train_loss, train_acc = _run_epoch(
                model, train_loader, event_criterion, device,
                optimizer=optimizer, scaler=scaler,
                use_amp=use_amp,
                gradient_clip_max_norm=training_config.gradient_clip_max_norm,
                label_smoothing=training_config.label_smoothing,
                history=history, save_dir=save_dir,
            )

            val_loss, val_acc = _run_epoch(
                model, val_loader, event_criterion, device,
                use_amp=use_amp,
            )

            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            history.train_losses.append(train_loss)
            history.val_losses.append(val_loss)
            history.train_accs.append(train_acc)
            history.val_accs.append(val_acc)
            history.learning_rates.append(current_lr)

            logger.info(
                f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | LR: {current_lr:.2e}"
            )

            # Log to wandb
            log_metrics({
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "lr": current_lr,
                "stage": 2,
                "epoch": global_epoch,
            })

            is_best = val_loss < history.best_val_loss
            if is_best:
                history.best_val_loss = val_loss
                history.best_val_acc = val_acc
                history.best_epoch = global_epoch
                early_stop_counter = 0
                save_audio_classifier(
                    model, save_dir / "best.pt",
                    optimizer=optimizer, epoch=global_epoch,
                    metrics={"val_loss": val_loss, "val_acc": val_acc},
                )
                logger.info(f"  -> New best model (val_loss: {val_loss:.4f})")
            else:
                early_stop_counter += 1

            # Save periodic checkpoint
            if (global_epoch + 1) % training_config.save_every == 0:
                checkpoint = {
                    "epoch": global_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "history": history.to_dict(),
                    "config": asdict(model_config),
                    "stage": 2,
                }
                if scaler is not None:
                    checkpoint["scaler_state_dict"] = scaler.state_dict()
                torch.save(checkpoint, save_dir / f"checkpoint_epoch_{global_epoch + 1}.pt")

            # Save latest checkpoint
            checkpoint = {
                "epoch": global_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "history": history.to_dict(),
                "config": asdict(model_config),
                "stage": 2,
            }
            if scaler is not None:
                checkpoint["scaler_state_dict"] = scaler.state_dict()
            torch.save(checkpoint, save_dir / "latest.pt")
            history.save(save_dir / "history.json")

            # Early stopping
            if early_stop_counter >= training_config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {global_epoch + 1}")
                break

            # Plot training progress periodically
            if (global_epoch + 1) % 5 == 0:
                plot_training_history(
                    history, save_dir / "training_plot.png",
                    title=f"Audio Classifier Training (Epoch {global_epoch + 1})",
                )

    # Final plots
    plot_training_history(
        history, save_dir / "training_plot.png",
        title="Audio Classifier Training (Final)",
    )
    plot_batch_progress(
        history, save_dir / "batch_progress.png",
        title="Audio Classifier Batch Progress (Final)",
    )
    history.save(save_dir / "history.json")

    # Log final summary and artifact to wandb
    log_summary({
        "best_val_loss": history.best_val_loss,
        "best_val_acc": history.best_val_acc,
        "best_epoch": history.best_epoch,
    })
    log_artifact(save_dir / "best.pt", artifact_type="model")
    finish_wandb()

    logger.info(f"Training complete! Best val_loss: {history.best_val_loss:.4f} at epoch {history.best_epoch + 1}")
    logger.info(f"Checkpoints saved to: {save_dir}")

    return model, history
