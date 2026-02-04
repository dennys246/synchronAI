"""
Training loop for video classifier.

Supports:
- Two-stage fine-tuning (head-only, then unfreeze backbone)
- Mixed precision training
- Early stopping
- Checkpointing
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from tqdm import tqdm

from synchronai.data.video.dataset import (
    VideoDatasetConfig,
    VideoWindowDataset,
    load_video_index,
    split_by_video,
    save_split_info,
)
from synchronai.models.cv.YOLO_classifier import (
    VideoClassifier,
    VideoClassifierConfig,
    build_video_classifier,
)
from synchronai.utils.reproducibility import set_seed, log_reproducibility_info

logger = logging.getLogger(__name__)


def generate_training_heatmap(
    model: nn.Module,
    video_path: Union[str, Path],
    save_dir: Path,
    epoch: int,
    model_config: "VideoClassifierConfig",
    device: torch.device,
    threshold: float = 0.5,
    clip_duration: int = 10,
    use_gradcam: bool = False,
    gradcam_aggregate: str = "max",
) -> Optional[Path]:
    """Generate heatmap visualization during training.

    Args:
        model: Current model state
        video_path: Path to sample video for visualization
        save_dir: Directory to save heatmaps
        epoch: Current epoch number
        model_config: Model configuration for inference
        device: Device to run inference on
        threshold: Classification threshold
        clip_duration: Duration in seconds of the clip to analyze (from middle of video)
        use_gradcam: Whether to generate Grad-CAM spatial heatmap thumbnails
        gradcam_aggregate: How to aggregate Grad-CAM across frames ("max", "mean", "weighted")

    Returns:
        Path to generated heatmap directory, or None if generation failed
    """
    from synchronai.inference.video.predict import (
        PredictionResult,
        VideoPredictionResult,
    )
    from synchronai.data.video.processing import (
        VideoReaderPool,
        load_video_info,
        read_window_frames,
    )
    from synchronai.utils.heatmap import (
        HeatmapConfig,
        plot_temporal_heatmap,
        export_heatmap_data,
    )
    import numpy as np

    video_path = Path(video_path)
    if not video_path.exists():
        logger.warning(f"Heatmap video not found: {video_path}")
        return None

    heatmap_dir = save_dir / "heatmaps" / f"epoch_{epoch:04d}"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Get video info
        video_info = load_video_info(str(video_path))
        total_seconds = int(video_info.duration)

        # Select a clip from the middle of the video
        clip_duration = min(clip_duration, total_seconds)
        start_second = max(0, (total_seconds - clip_duration) // 2)
        end_second = start_second + clip_duration

        logger.info(
            f"Generating heatmap for batch {epoch} "
            f"(seconds {start_second}-{end_second} of {total_seconds}s video)..."
        )

        # Create reader pool
        reader_pool = VideoReaderPool(max_readers=1)
        reader = reader_pool.get_reader(str(video_path))

        predictions = []
        model.eval()

        try:
            with torch.no_grad():
                for second in range(start_second, end_second):
                    # Read frames for this second
                    frames = read_window_frames(
                        video_path=str(video_path),
                        second=second,
                        sample_fps=model_config.sample_fps,
                        window_seconds=model_config.window_seconds,
                        target_size=model_config.frame_height,
                        reader=reader,
                    )

                    # Convert to tensor and add batch dimension
                    frames_tensor = torch.from_numpy(frames).unsqueeze(0).to(device)

                    # Run model
                    logits = model(frames_tensor)
                    prob = torch.sigmoid(logits).item()

                    pred = 1 if prob >= threshold else 0
                    confidence = prob if pred == 1 else 1 - prob

                    predictions.append(
                        PredictionResult(
                            second=second,
                            probability=prob,
                            prediction=pred,
                            confidence=confidence,
                        )
                    )

        finally:
            reader_pool.close_all()

        # Compute overall metrics
        probs = [p.probability for p in predictions]
        preds = [p.prediction for p in predictions]

        overall_prob = float(np.mean(probs))
        synchrony_seconds = sum(preds)
        synchrony_ratio = synchrony_seconds / len(predictions) if predictions else 0.0

        result = VideoPredictionResult(
            video_path=str(video_path),
            predictions=predictions,
            overall_probability=overall_prob,
            overall_prediction=1 if overall_prob >= threshold else 0,
            total_seconds=len(predictions),
            synchrony_seconds=synchrony_seconds,
            synchrony_ratio=synchrony_ratio,
        )

        # Generate visualizations
        video_name = video_path.stem
        heatmap_config = HeatmapConfig(threshold=threshold)

        # Timeline heatmap
        plot_temporal_heatmap(
            predictions,
            config=heatmap_config,
            title=f"Batch {epoch} - {video_name} (sec {start_second}-{end_second}, {synchrony_ratio:.1%} sync)",
            save_path=str(heatmap_dir / f"{video_name}_timeline.png"),
        )

        # Export data
        export_heatmap_data(result, heatmap_dir / f"{video_name}_data.json")

        # Generate Grad-CAM thumbnails if enabled
        if use_gradcam:
            try:
                from synchronai.utils.gradcam import GradCAM, GradCAMConfig, apply_cam_to_frame
                import cv2

                logger.info("  Generating Grad-CAM thumbnail grid...")

                # Initialize Grad-CAM
                gradcam_config = GradCAMConfig(alpha=0.5)
                gradcam = GradCAM(model, gradcam_config)

                # Re-create reader pool for Grad-CAM
                reader_pool = VideoReaderPool(max_readers=1)
                reader = reader_pool.get_reader(str(video_path))

                # Open video for frame extraction
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # Generate thumbnails with Grad-CAM overlays
                thumb_size = (320, 180)
                thumbnails = []
                cols = 5
                max_thumbs = min(clip_duration, 20)

                # Sample seconds evenly from the clip
                if max_thumbs < clip_duration:
                    sample_seconds = np.linspace(start_second, end_second - 1, max_thumbs, dtype=int)
                else:
                    sample_seconds = np.arange(start_second, end_second)

                pred_by_second = {p.second: p for p in predictions}

                for second in sample_seconds:
                    # Get frame from video
                    frame_idx = int((second + 0.5) * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()

                    if not ret:
                        continue

                    # Resize to thumbnail
                    thumb = cv2.resize(frame, thumb_size, interpolation=cv2.INTER_AREA)

                    # Generate CAM
                    pred = pred_by_second.get(second)
                    if pred is not None:
                        try:
                            model_frames = read_window_frames(
                                video_path=str(video_path),
                                second=second,
                                sample_fps=model_config.sample_fps,
                                window_seconds=model_config.window_seconds,
                                target_size=model_config.frame_height,
                                reader=reader,
                            )
                            frames_tensor = torch.from_numpy(model_frames).unsqueeze(0).to(device)

                            with torch.enable_grad():
                                cam = gradcam.generate_cam_for_window(frames_tensor, aggregate=gradcam_aggregate)

                            thumb = apply_cam_to_frame(thumb, cam, gradcam_config)

                            # Add label
                            label = "S" if pred.prediction == 1 else "A"
                            color = (0, 255, 0) if pred.prediction == 1 else (0, 0, 255)
                            cv2.putText(thumb, f"{label}:{pred.probability:.0%}", (5, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            cv2.putText(thumb, f"t={second}s", (5, thumb_size[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                        except Exception as cam_err:
                            logger.debug(f"Failed to generate CAM for second {second}: {cam_err}")

                    thumbnails.append(thumb)

                cap.release()
                reader_pool.close_all()
                gradcam.remove_hooks()

                # Create grid
                if thumbnails:
                    n_thumbs = len(thumbnails)
                    rows = (n_thumbs + cols - 1) // cols
                    thumb_w, thumb_h = thumb_size

                    canvas = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)

                    for i, thumb in enumerate(thumbnails):
                        row = i // cols
                        col = i % cols
                        y = row * thumb_h
                        x = col * thumb_w
                        canvas[y : y + thumb_h, x : x + thumb_w] = thumb

                    gradcam_path = heatmap_dir / f"{video_name}_gradcam_thumbnails.png"
                    cv2.imwrite(str(gradcam_path), canvas)
                    logger.info(f"  Grad-CAM thumbnails saved: {gradcam_path}")

            except Exception as gradcam_err:
                logger.warning(f"Failed to generate Grad-CAM thumbnails: {gradcam_err}")

        logger.info(
            f"  Heatmap generated: {synchrony_seconds}/{clip_duration}s sync "
            f"({synchrony_ratio:.1%}), saved to {heatmap_dir}"
        )

        return heatmap_dir

    except Exception as e:
        logger.warning(f"Failed to generate heatmap at epoch {epoch}: {e}")
        return None


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Data
    labels_file: str = ""
    val_split: float = 0.2
    group_by: str = "subject_id"

    # Training
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_accumulation_steps: int = 1

    # Scheduler
    warmup_epochs: int = 1

    # Fine-tuning
    stage1_epochs: int = 5
    stage2_unfreeze: str = "backbone.last"
    backbone_lr: float = 1e-5
    stage2_warmup_epochs: int = 1

    # Early stopping
    patience: int = 10
    monitor: str = "val_auc"

    # Mixed precision
    use_amp: bool = True

    # Reproducibility
    seed: int = 42
    deterministic: bool = False

    # Dataloader (0 for cluster compatibility with limited /dev/shm)
    num_workers: int = 0

    # Checkpointing
    save_best: bool = True
    save_last: bool = True

    # Heatmap visualization during training
    heatmap_batch_interval: int = 10  # Generate heatmaps every N batches (0 = disabled)
    heatmap_video_path: Optional[str] = None  # Path to sample video for heatmap generation
    heatmap_use_gradcam: bool = True  # Generate Grad-CAM spatial heatmaps during training
    heatmap_gradcam_aggregate: str = "max"  # How to aggregate Grad-CAM: max, mean, weighted


@dataclass
class TrainingHistory:
    """Training history tracker with batch-level and epoch-level metrics."""

    # Epoch-level metrics
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_accs: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)
    val_aucs: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    best_val_auc: float = 0.0
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    epochs_without_improvement: int = 0

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


def plot_training_history(
    history: TrainingHistory,
    output_path: Union[str, Path],
    title: str = "Video Classifier Training",
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

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
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

    # AUC plot
    ax = axes[1, 0]
    ax.plot(epochs, history.val_aucs, "m-", label="Val AUC", linewidth=2)
    ax.axvline(x=history.best_epoch + 1, color="g", linestyle="--", alpha=0.7)
    ax.axhline(y=history.best_val_auc, color="g", linestyle=":", alpha=0.5, label=f"Best AUC: {history.best_val_auc:.4f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.set_title("Validation AUC over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate plot
    ax = axes[1, 1]
    ax.plot(epochs, history.learning_rates, "g-", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Add summary text box
    summary_text = (
        f"Best Epoch: {history.best_epoch + 1}\n"
        f"Best Val AUC: {history.best_val_auc:.4f}\n"
        f"Best Val Acc: {history.val_accs[history.best_epoch]:.4f}\n"
        f"Final Train Loss: {history.train_losses[-1]:.4f}\n"
        f"Final Val Loss: {history.val_losses[-1]:.4f}"
    )
    fig.text(0.02, 0.02, summary_text, fontsize=10, fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved training plot to {output_path}")
    return output_path


def plot_batch_progress(
    history: TrainingHistory,
    output_path: Union[str, Path],
    title: str = "Video Classifier Batch Progress",
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
    import numpy as np

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


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, float]:
    """Compute classification metrics.

    Args:
        logits: Model logits (batch,) or (batch, 1)
        labels: Ground truth labels (batch,)

    Returns:
        Dict with accuracy, precision, recall, f1, auc
    """
    logits = logits.squeeze()
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    # Accuracy
    correct = (preds == labels).float().sum()
    accuracy = correct / len(labels)

    # For AUC, we need more sophisticated handling
    try:
        from torchmetrics.functional import auroc

        auc = auroc(probs, labels.long(), task="binary").item()
    except Exception:
        auc = 0.5  # Default if computation fails

    # Precision, Recall, F1
    tp = ((preds == 1) & (labels == 1)).float().sum()
    fp = ((preds == 1) & (labels == 0)).float().sum()
    fn = ((preds == 0) & (labels == 1)).float().sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "auc": auc,
    }


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    accumulation_steps: int = 1,
    use_amp: bool = True,
    # Heatmap generation parameters
    heatmap_batch_interval: int = 0,
    heatmap_video_path: Optional[str] = None,
    save_dir: Optional[Path] = None,
    model_config: Optional["VideoClassifierConfig"] = None,
    global_batch_count: int = 0,
    heatmap_use_gradcam: bool = True,
    heatmap_gradcam_aggregate: str = "max",
    # Batch progress tracking
    history: Optional[TrainingHistory] = None,
    batch_plot_interval: int = 10,
) -> tuple[float, float, int]:
    """Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        scaler: GradScaler for AMP
        device: Device to train on
        accumulation_steps: Gradient accumulation steps
        use_amp: Whether to use automatic mixed precision
        heatmap_batch_interval: Generate heatmaps every N batches (0 = disabled)
        heatmap_video_path: Path to sample video for heatmap generation
        save_dir: Directory to save heatmaps
        model_config: Model configuration for inference
        global_batch_count: Running count of batches across epochs
        heatmap_use_gradcam: Whether to generate Grad-CAM spatial heatmaps
        heatmap_gradcam_aggregate: How to aggregate Grad-CAM across frames
        history: TrainingHistory to record batch-level metrics
        batch_plot_interval: Plot batch progress every N batches (0 = disabled)

    Returns:
        Tuple of (average loss, accuracy, updated global_batch_count)
    """
    model.train()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)

        with autocast(enabled=use_amp):
            logits = model(frames).squeeze()
            loss = criterion(logits, labels)
            loss = loss / accumulation_steps

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        batch_loss = loss.item() * accumulation_steps
        total_loss += batch_loss
        all_logits.append(logits.detach())
        all_labels.append(labels.detach())

        # Compute batch accuracy
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            batch_acc = (preds == labels).float().mean().item()

        pbar.set_postfix({"loss": f"{batch_loss:.4f}", "acc": f"{batch_acc:.4f}"})

        # Record batch metrics to history
        if history is not None:
            history.add_batch_metrics(batch_loss, batch_acc)

            # Plot batch progress at intervals
            if (
                batch_plot_interval > 0
                and save_dir is not None
                and history._global_batch % batch_plot_interval == 0
            ):
                plot_batch_progress(
                    history,
                    save_dir / "batch_progress.png",
                    title="Video Classifier Batch Progress",
                )
                # Also save history incrementally
                history.save(save_dir / "history.json")

        # Generate heatmap at specified batch intervals
        global_batch_count += 1
        if (
            heatmap_batch_interval > 0
            and save_dir is not None
            and model_config is not None
            and global_batch_count % heatmap_batch_interval == 0
        ):
            # Use the last video from this batch for efficiency (already loaded)
            batch_video_path = batch.get("video_path")
            if batch_video_path is not None and len(batch_video_path) > 0:
                # Use the last video in the batch
                video_for_heatmap = batch_video_path[-1]
            else:
                # Fall back to configured heatmap video path
                video_for_heatmap = heatmap_video_path

            if video_for_heatmap:
                generate_training_heatmap(
                    model=model,
                    video_path=video_for_heatmap,
                    save_dir=save_dir,
                    epoch=global_batch_count,  # Use batch count as identifier
                    model_config=model_config,
                    device=device,
                    threshold=0.5,
                    use_gradcam=heatmap_use_gradcam,
                    gradcam_aggregate=heatmap_gradcam_aggregate,
                )
                model.train()  # Ensure model is back in training mode

    # Compute metrics
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels)

    avg_loss = total_loss / len(dataloader)
    return avg_loss, metrics["accuracy"], global_batch_count


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[float, dict[str, float]]:
    """Validate model.

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device
        use_amp: Whether to use AMP

    Returns:
        Tuple of (average loss, metrics dict)
    """
    model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Validating", leave=False):
        frames = batch["frames"].to(device)
        labels = batch["label"].to(device)

        with autocast(enabled=use_amp):
            logits = model(frames).squeeze()
            loss = criterion(logits, labels)

        total_loss += loss.item()
        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_logits, all_labels)

    avg_loss = total_loss / len(dataloader)
    return avg_loss, metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    epoch: int,
    history: TrainingHistory,
    config: dict,
    save_path: Path,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    stage: int = 1,
) -> None:
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scaler: GradScaler state
        epoch: Current epoch
        history: Training history
        config: Training configuration
        save_path: Path to save checkpoint
        scheduler: Learning rate scheduler (optional)
        stage: Current training stage (1 or 2)
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": asdict(history),
        "config": config,
        "stage": stage,
    }

    if scaler is not None:
        checkpoint["scaler_state_dict"] = scaler.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, save_path)
    logger.debug(f"Saved checkpoint to {save_path}")


def train_video_classifier(
    labels_file: Union[str, Path],
    save_dir: Union[str, Path],
    model_config: Optional[VideoClassifierConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    data_config: Optional[VideoDatasetConfig] = None,
    resume_from: Optional[str] = None,
) -> tuple[VideoClassifier, TrainingHistory]:
    """Train a video classifier.

    Args:
        labels_file: Path to labels.csv
        save_dir: Directory to save checkpoints and logs
        model_config: Model configuration
        training_config: Training configuration
        data_config: Data configuration
        resume_from: Path to checkpoint to resume from (e.g., "latest.pt")

    Returns:
        Tuple of (trained model, training history)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Default configs
    if model_config is None:
        model_config = VideoClassifierConfig()
    if training_config is None:
        training_config = TrainingConfig()
    if data_config is None:
        data_config = VideoDatasetConfig(
            labels_file=str(labels_file),
            sample_fps=model_config.sample_fps,
            window_seconds=model_config.window_seconds,
            frame_size=model_config.frame_height,
        )

    training_config.labels_file = str(labels_file)

    # Set seed
    set_seed(training_config.seed, training_config.deterministic)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading video index...")
    specs = load_video_index(
        labels_file,
        sample_fps=model_config.sample_fps,
        window_seconds=model_config.window_seconds,
        frame_size=model_config.frame_height,
    )

    # Split data
    train_specs, val_specs = split_by_video(
        specs,
        val_split=training_config.val_split,
        group_by=training_config.group_by,
        seed=training_config.seed,
    )

    # Save split info
    save_split_info(train_specs, val_specs, save_dir / "split_info.json")

    # Create datasets
    train_dataset = VideoWindowDataset(
        train_specs,
        data_config,
        augment=True,
    )
    val_dataset = VideoWindowDataset(
        val_specs,
        data_config,
        augment=False,
    )

    # Get pos_weight
    pos_weight = train_dataset.get_pos_weight()
    logger.info(f"Computed pos_weight: {pos_weight:.3f}")

    # Create dataloaders
    from synchronai.utils.reproducibility import worker_init_fn

    generator = torch.Generator()
    generator.manual_seed(training_config.seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
        persistent_workers=training_config.num_workers > 0,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=training_config.num_workers > 0,
    )

    # Auto-select heatmap video from training data if not specified
    # Note: During training, heatmaps use the last video from each batch for efficiency.
    # This fallback is only used if batch video_path is unavailable.
    heatmap_video_path = training_config.heatmap_video_path
    if heatmap_video_path is None and training_config.heatmap_batch_interval > 0:
        # Get unique video paths from training specs
        train_video_paths = sorted(set(s.video_path for s in train_specs))
        if train_video_paths:
            # Use last video as fallback (matches behavior of using last batch video)
            heatmap_video_path = train_video_paths[-1]
            logger.info(f"Auto-selected fallback heatmap video: {heatmap_video_path}")

    # Create model
    logger.info("Building model...")
    model = build_video_classifier(model_config)
    model.to(device)

    # Loss function
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    # Training history
    history = TrainingHistory()

    # Mixed precision scaler
    scaler = GradScaler() if training_config.use_amp else None

    # Resume from checkpoint if specified
    start_epoch = 0
    resume_stage = 1
    if resume_from:
        checkpoint_path = Path(resume_from)
        if not checkpoint_path.is_absolute():
            checkpoint_path = save_dir / resume_from
        if checkpoint_path.exists():
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            if "history" in checkpoint:
                history = TrainingHistory.from_dict(checkpoint["history"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            resume_stage = checkpoint.get("stage", 1)
            logger.info(f"Resumed from epoch {start_epoch}, stage {resume_stage}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}, starting from scratch")

    # Log reproducibility info
    log_reproducibility_info(
        save_dir,
        training_config.seed,
        config={"model": asdict(model_config), "training": asdict(training_config)},
    )

    # ========================
    # Stage 1: Head-only training
    # ========================
    stage1_start = start_epoch if resume_stage == 1 else 0
    skip_stage1 = resume_stage > 1 or stage1_start >= training_config.stage1_epochs
    global_batch_count = 0  # Track batches across epochs for heatmap generation

    if not skip_stage1:
        logger.info(f"Stage 1: Training head only for {training_config.stage1_epochs} epochs")
        model.freeze_backbone()

        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

        # Warmup + cosine scheduler for stage 1
        if training_config.warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=training_config.warmup_epochs,
            )
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=training_config.stage1_epochs - training_config.warmup_epochs,
            )
            scheduler = SequentialLR(
                optimizer,
                [warmup_scheduler, main_scheduler],
                milestones=[training_config.warmup_epochs],
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=training_config.stage1_epochs)

        for epoch in range(stage1_start, training_config.stage1_epochs):
            logger.info(f"Epoch {epoch + 1}/{training_config.stage1_epochs} (Stage 1)")

            # Train
            train_loss, train_acc, global_batch_count = train_epoch(
                model, train_loader, criterion, optimizer, scaler, device,
                training_config.gradient_accumulation_steps, training_config.use_amp,
                heatmap_batch_interval=training_config.heatmap_batch_interval,
                heatmap_video_path=heatmap_video_path,
                save_dir=save_dir,
                model_config=model_config,
                global_batch_count=global_batch_count,
                heatmap_use_gradcam=training_config.heatmap_use_gradcam,
                heatmap_gradcam_aggregate=training_config.heatmap_gradcam_aggregate,
                history=history,
                batch_plot_interval=10,
            )

            # Validate
            val_loss, val_metrics = validate(model, val_loader, criterion, device, training_config.use_amp)

            # Update scheduler
            scheduler.step()

            # Log
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                f"AUC: {val_metrics['auc']:.4f} | LR: {lr:.2e}"
            )

            # Update history
            history.train_losses.append(train_loss)
            history.val_losses.append(val_loss)
            history.train_accs.append(train_acc)
            history.val_accs.append(val_metrics["accuracy"])
            history.val_aucs.append(val_metrics["auc"])
            history.learning_rates.append(lr)

            # Check for best model (track both AUC and loss)
            is_best = val_metrics["auc"] > history.best_val_auc
            if is_best:
                history.best_val_auc = val_metrics["auc"]
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                history.epochs_without_improvement = 0

                if training_config.save_best:
                    save_checkpoint(
                        model, optimizer, scaler, epoch, history,
                        {"model": asdict(model_config), "training": asdict(training_config)},
                        save_dir / "best.pt",
                        scheduler=scheduler,
                        stage=1,
                    )
                    logger.info(f"  -> New best model saved (val_auc: {val_metrics['auc']:.4f})")
            else:
                history.epochs_without_improvement += 1

            # Save latest checkpoint (for resume)
            save_checkpoint(
                model, optimizer, scaler, epoch, history,
                {"model": asdict(model_config), "training": asdict(training_config)},
                save_dir / "latest.pt",
                scheduler=scheduler,
                stage=1,
            )

            # Save history after each epoch (ensures epoch metrics are persisted)
            history.save(save_dir / "history.json")

            # Plot training progress periodically
            if (epoch + 1) % 5 == 0 or epoch == training_config.stage1_epochs - 1:
                plot_training_history(
                    history,
                    save_dir / "training_plot.png",
                    title=f"Video Classifier Training (Epoch {epoch + 1})",
                )

        # Save stage 1 checkpoint
        save_checkpoint(
            model, optimizer, scaler, training_config.stage1_epochs - 1, history,
            {"model": asdict(model_config), "training": asdict(training_config)},
            save_dir / "stage1_final.pt",
            scheduler=scheduler,
            stage=1,
        )

    # ========================
    # Stage 2: Unfreeze backbone
    # ========================
    stage2_epochs = training_config.epochs - training_config.stage1_epochs
    stage2_start = start_epoch - training_config.stage1_epochs if resume_stage == 2 else 0
    stage2_start = max(0, stage2_start)

    if stage2_epochs > 0 and training_config.stage2_unfreeze != "backbone.none":
        logger.info(f"Stage 2: Fine-tuning backbone for {stage2_epochs} epochs")

        # Unfreeze backbone
        if training_config.stage2_unfreeze == "backbone.all":
            model.unfreeze_backbone(mode="all")
        else:
            model.unfreeze_backbone(mode="last")

        # New optimizer with parameter groups
        optimizer = AdamW(
            model.get_parameter_groups(
                backbone_lr=training_config.backbone_lr,
                head_lr=training_config.learning_rate,
            ),
            weight_decay=training_config.weight_decay,
        )

        # Reset scaler
        if training_config.use_amp:
            scaler = GradScaler()

        # Scheduler for stage 2
        if training_config.stage2_warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                total_iters=training_config.stage2_warmup_epochs,
            )
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=stage2_epochs - training_config.stage2_warmup_epochs,
            )
            scheduler = SequentialLR(
                optimizer,
                [warmup_scheduler, main_scheduler],
                milestones=[training_config.stage2_warmup_epochs],
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=stage2_epochs)

        for epoch in range(stage2_start, stage2_epochs):
            global_epoch = training_config.stage1_epochs + epoch
            logger.info(f"Epoch {global_epoch + 1}/{training_config.epochs} (Stage 2)")

            # Train
            train_loss, train_acc, global_batch_count = train_epoch(
                model, train_loader, criterion, optimizer, scaler, device,
                training_config.gradient_accumulation_steps, training_config.use_amp,
                heatmap_batch_interval=training_config.heatmap_batch_interval,
                heatmap_video_path=heatmap_video_path,
                save_dir=save_dir,
                model_config=model_config,
                global_batch_count=global_batch_count,
                heatmap_use_gradcam=training_config.heatmap_use_gradcam,
                heatmap_gradcam_aggregate=training_config.heatmap_gradcam_aggregate,
                history=history,
                batch_plot_interval=10,
            )

            # Validate
            val_loss, val_metrics = validate(model, val_loader, criterion, device, training_config.use_amp)

            # Update scheduler
            scheduler.step()

            # Log
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                f"AUC: {val_metrics['auc']:.4f} | LR: {lr:.2e}"
            )

            # Update history
            history.train_losses.append(train_loss)
            history.val_losses.append(val_loss)
            history.train_accs.append(train_acc)
            history.val_accs.append(val_metrics["accuracy"])
            history.val_aucs.append(val_metrics["auc"])
            history.learning_rates.append(lr)

            # Check for best model (track both AUC and loss)
            is_best = val_metrics["auc"] > history.best_val_auc
            if is_best:
                history.best_val_auc = val_metrics["auc"]
                history.best_val_loss = val_loss
                history.best_epoch = global_epoch
                history.epochs_without_improvement = 0

                if training_config.save_best:
                    save_checkpoint(
                        model, optimizer, scaler, global_epoch, history,
                        {"model": asdict(model_config), "training": asdict(training_config)},
                        save_dir / "best.pt",
                        scheduler=scheduler,
                        stage=2,
                    )
                    logger.info(f"  -> New best model saved (val_auc: {val_metrics['auc']:.4f})")
            else:
                history.epochs_without_improvement += 1

            # Save latest checkpoint (for resume)
            save_checkpoint(
                model, optimizer, scaler, global_epoch, history,
                {"model": asdict(model_config), "training": asdict(training_config)},
                save_dir / "latest.pt",
                scheduler=scheduler,
                stage=2,
            )

            # Save history after each epoch (ensures epoch metrics are persisted)
            history.save(save_dir / "history.json")

            # Plot training progress periodically
            if (global_epoch + 1) % 5 == 0 or epoch == stage2_epochs - 1:
                plot_training_history(
                    history,
                    save_dir / "training_plot.png",
                    title=f"Video Classifier Training (Epoch {global_epoch + 1})",
                )

            # Early stopping
            if history.epochs_without_improvement >= training_config.patience:
                logger.info(f"Early stopping after {history.epochs_without_improvement} epochs without improvement")
                break

    # Save final checkpoint
    if training_config.save_last:
        save_checkpoint(
            model, optimizer, scaler, training_config.epochs - 1, history,
            {"model": asdict(model_config), "training": asdict(training_config)},
            save_dir / "last.pt",
            scheduler=scheduler if 'scheduler' in dir() else None,
            stage=2 if stage2_epochs > 0 else 1,
        )

    # Save training history
    history.save(save_dir / "history.json")

    # Final training plots
    plot_training_history(
        history,
        save_dir / "training_plot.png",
        title="Video Classifier Training (Final)",
    )
    plot_batch_progress(
        history,
        save_dir / "batch_progress.png",
        title="Video Classifier Batch Progress (Final)",
    )

    logger.info(f"Training complete. Best AUC: {history.best_val_auc:.4f} at epoch {history.best_epoch + 1}")
    logger.info(f"Checkpoints saved to: {save_dir}")

    # Generate final heatmap using the last video from training data
    # Get unique video paths from training specs and use the last one
    train_video_paths = sorted(set(s.video_path for s in train_specs))
    final_heatmap_video = train_video_paths[-1] if train_video_paths else heatmap_video_path
    if final_heatmap_video:
        logger.info(f"Generating final heatmap visualization using: {Path(final_heatmap_video).name}")
        generate_training_heatmap(
            model=model,
            video_path=final_heatmap_video,
            save_dir=save_dir,
            epoch=training_config.epochs,
            model_config=model_config,
            device=device,
            threshold=0.5,
            use_gradcam=training_config.heatmap_use_gradcam,
            gradcam_aggregate=training_config.heatmap_gradcam_aggregate,
        )

    # Clean up
    train_dataset.close()
    val_dataset.close()

    return model, history
