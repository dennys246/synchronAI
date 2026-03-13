"""
Training loop for multi-modal synchrony classifier.

Combines video and audio models with fusion for synchronized prediction.
Supports:
- Two-stage fine-tuning (freeze backbones → unfreeze with differential LRs)
- Multi-task learning (synchrony + audio events)
- Mixed precision training
- Pretrained model loading
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from tqdm import tqdm

from synchronai.models.multimodal.fusion_model import MultiModalSynchronyModel
from synchronai.data.multimodal.dataset_mm import (
    MultiModalDataset,
    MultiModalDatasetConfig,
    create_multimodal_splits
)
from synchronai.data.video.dataset import VideoDatasetConfig
from synchronai.data.audio.dataset import AudioDatasetConfig
from synchronai.utils.reproducibility import set_seed, log_reproducibility_info, worker_init_fn

logger = logging.getLogger(__name__)


@dataclass
class MultiModalTrainingConfig:
    """Configuration for multi-modal training."""

    # General training
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    use_amp: bool = True
    num_workers: int = 4
    gradient_clip_max_norm: float = 1.0
    label_smoothing: float = 0.05

    # Two-stage fine-tuning
    stage1_epochs: int = 5  # Head + fusion only
    video_backbone_lr: float = 1e-5  # Stage 2
    audio_encoder_lr: float = 1e-5  # Stage 2
    fusion_head_lr: float = 5e-5  # Stage 2 (backbone_lr × 5)
    stage2_warmup_epochs: int = 3

    # Learning rate schedule
    scheduler: str = "cosine"
    warmup_epochs: int = 2  # Stage 1 warmup

    # Multi-task loss weights
    sync_loss_weight: float = 1.0  # Primary task
    # Event aux task disabled by default: the labels CSV has binary sync
    # labels (0/1), not real audio event classes (0-6). Training the 7-class
    # event_head on sync labels produces garbage gradients.
    event_loss_weight: float = 0.0

    # Data split
    val_split: float = 0.2
    group_by: str = "subject_id"

    # Early stopping
    early_stopping_patience: int = 15
    auc_thresholds: Optional[int] = 200

    # Reproducibility
    seed: int = 42
    deterministic: bool = False

    # Pretrained models (transfer learning)
    pretrained_video: Optional[str] = None
    pretrained_audio: Optional[str] = None
    load_heads_only: bool = True  # If True, keep pretrained backbones

    # Checkpointing
    save_interval: int = 5  # Save every N epochs

    # Heatmap generation
    heatmap_epoch_interval: int = 0  # Generate heatmap every N epochs (0 = disabled)
    heatmap_video_path: Optional[str] = None  # Path to sample video for heatmaps

    # Labels file (set by train_multimodal_classifier for heatmap ground truth)
    labels_file: str = ""


@dataclass
class TrainingHistory:
    """Track training metrics over time."""

    train_losses: list[float] = None
    val_losses: list[float] = None
    train_sync_losses: list[float] = None
    train_event_losses: list[float] = None
    val_sync_losses: list[float] = None
    val_event_losses: list[float] = None
    train_accs: list[float] = None
    val_accs: list[float] = None
    val_aucs: list[float] = None
    learning_rates: list[float] = None
    best_epoch: int = 0
    best_val_loss: float = float('inf')
    best_val_auc: float = 0.0

    def __post_init__(self):
        if self.train_losses is None:
            self.train_losses = []
        if self.val_losses is None:
            self.val_losses = []
        if self.train_sync_losses is None:
            self.train_sync_losses = []
        if self.train_event_losses is None:
            self.train_event_losses = []
        if self.val_sync_losses is None:
            self.val_sync_losses = []
        if self.val_event_losses is None:
            self.val_event_losses = []
        if self.train_accs is None:
            self.train_accs = []
        if self.val_accs is None:
            self.val_accs = []
        if self.val_aucs is None:
            self.val_aucs = []
        if self.learning_rates is None:
            self.learning_rates = []

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Union[str, Path]):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainingHistory":
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class BinaryMetricTracker:
    """Track binary classification metrics without storing full logits/labels."""

    def __init__(self, device: torch.device, auc_thresholds: Optional[int] = 200) -> None:
        self.correct = 0
        self.total = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self._auc = None

        try:
            from torchmetrics.classification import BinaryAUROC

            if auc_thresholds is None:
                self._auc = BinaryAUROC().to(device)
            else:
                self._auc = BinaryAUROC(thresholds=auc_thresholds).to(device)
        except Exception:
            self._auc = None

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        logits = logits.squeeze(-1) if logits.dim() > 1 else logits
        probs = torch.sigmoid(logits.detach())
        labels_int = labels.detach().long()
        preds = (probs > 0.5).long()

        self.correct += (preds == labels_int).sum().item()
        self.total += labels_int.numel()

        self.tp += ((preds == 1) & (labels_int == 1)).sum().item()
        self.fp += ((preds == 1) & (labels_int == 0)).sum().item()
        self.fn += ((preds == 0) & (labels_int == 1)).sum().item()

        if self._auc is not None:
            self._auc.update(probs, labels_int)

    def compute(self) -> Dict[str, float]:
        if self.total == 0:
            accuracy = 0.0
        else:
            accuracy = self.correct / self.total

        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if self._auc is not None:
            try:
                auc = float(self._auc.compute().item())
            except Exception:
                auc = 0.5
        else:
            auc = 0.5

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
        }


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    device: torch.device,
    sync_criterion: nn.Module,
    event_criterion: Optional[nn.Module],
    sync_weight: float,
    event_weight: float,
    use_amp: bool = True,
    gradient_clip_max_norm: float = 1.0,
    label_smoothing: float = 0.0,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_sync_loss = 0.0
    total_event_loss = 0.0
    total_correct = 0
    total_samples = 0

    use_event = event_weight > 0 and event_criterion is not None

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for batch in pbar:
        video_frames = batch['video_frames'].to(device)
        audio_chunks = batch['audio_chunk'].to(device)
        sync_labels = batch['sync_label'].to(device).float()
        original_sync_labels = sync_labels
        event_labels = batch['event_label'].to(device).long() if use_event else None

        # Apply label smoothing to sync labels
        if label_smoothing > 0:
            sync_labels = sync_labels * (1 - label_smoothing) + label_smoothing * 0.5

        optimizer.zero_grad()

        # Forward pass with AMP
        with autocast(device.type, enabled=use_amp):
            outputs = model(video_frames, audio_chunks)

            sync_logits = outputs['sync_logits']
            sync_logits = sync_logits.squeeze(-1) if sync_logits.dim() > 1 else sync_logits
            event_logits = outputs['event_logits'] if use_event else None

            # Multi-task loss
            sync_loss = sync_criterion(sync_logits, sync_labels)
            if use_event:
                event_loss = event_criterion(event_logits, event_labels)
                loss = sync_weight * sync_loss + event_weight * event_loss
            else:
                event_loss = torch.tensor(0.0, device=device)
                loss = sync_weight * sync_loss

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            if gradient_clip_max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_max_norm)
            optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_sync_loss += sync_loss.item()
        total_event_loss += event_loss.item()

        with torch.no_grad():
            probs = torch.sigmoid(sync_logits.detach())
            preds = (probs > 0.5).float()
            batch_correct = (preds == original_sync_labels).float().sum().item()
            total_correct += batch_correct
            total_samples += original_sync_labels.numel()

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'sync': f"{sync_loss.item():.4f}",
            'event': f"{event_loss.item():.4f}"
        })

    accuracy = total_correct / max(1, total_samples)
    metrics = {
        "accuracy": float(accuracy),
        "loss": total_loss / len(dataloader),
        "sync_loss": total_sync_loss / len(dataloader),
        "event_loss": total_event_loss / len(dataloader),
    }

    return metrics


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    sync_criterion: nn.Module,
    event_criterion: Optional[nn.Module],
    sync_weight: float,
    event_weight: float,
    use_amp: bool = True,
    auc_thresholds: Optional[int] = 200,
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()

    total_loss = 0.0
    total_sync_loss = 0.0
    total_event_loss = 0.0
    metrics_tracker = BinaryMetricTracker(device=device, auc_thresholds=auc_thresholds)

    use_event = event_weight > 0 and event_criterion is not None

    pbar = tqdm(dataloader, desc="Validation", leave=False)

    for batch in pbar:
        video_frames = batch['video_frames'].to(device)
        audio_chunks = batch['audio_chunk'].to(device)
        sync_labels = batch['sync_label'].to(device).float()
        event_labels = batch['event_label'].to(device).long() if use_event else None

        with autocast(device.type, enabled=use_amp):
            outputs = model(video_frames, audio_chunks)

            sync_logits = outputs['sync_logits']
            sync_logits = sync_logits.squeeze(-1) if sync_logits.dim() > 1 else sync_logits
            event_logits = outputs['event_logits'] if use_event else None

            sync_loss = sync_criterion(sync_logits, sync_labels)
            if use_event:
                event_loss = event_criterion(event_logits, event_labels)
                loss = sync_weight * sync_loss + event_weight * event_loss
            else:
                event_loss = torch.tensor(0.0, device=device)
                loss = sync_weight * sync_loss

        total_loss += loss.item()
        total_sync_loss += sync_loss.item()
        total_event_loss += event_loss.item()

        metrics_tracker.update(sync_logits, sync_labels)

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    metrics = metrics_tracker.compute()
    metrics['loss'] = total_loss / len(dataloader)
    metrics['sync_loss'] = total_sync_loss / len(dataloader)
    metrics['event_loss'] = total_event_loss / len(dataloader)

    return metrics


def train_multimodal_classifier(
    labels_file: Union[str, Path],
    save_dir: Union[str, Path],
    video_config: Dict[str, Any],
    audio_config: Dict[str, Any],
    fusion_config: Dict[str, Any],
    training_config: Optional[MultiModalTrainingConfig] = None,
) -> tuple[MultiModalSynchronyModel, TrainingHistory]:
    """
    Train multi-modal synchrony classifier.

    Args:
        labels_file: Path to labels CSV
        save_dir: Directory to save checkpoints
        video_config: Video model configuration dict
        audio_config: Audio model configuration dict
        fusion_config: Fusion module configuration dict
        training_config: Training configuration

    Returns:
        (trained_model, training_history)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if training_config is None:
        training_config = MultiModalTrainingConfig()

    # Set seed
    set_seed(training_config.seed, training_config.deterministic)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    use_amp = training_config.use_amp and device.type == "cuda"

    # Create dataset configs
    video_data_config = VideoDatasetConfig(
        labels_file=str(labels_file),
        sample_fps=video_config.get('sample_fps', 12.0),
        window_seconds=video_config.get('window_seconds', 2.0),
        frame_size=video_config.get('frame_size', 640),
        augment=True,
        color_jitter=True,
        horizontal_flip_prob=0.5,
        temporal_jitter_frames=2,
        random_erase_prob=0.3,
        gaussian_noise_std=0.02,
        mixup_alpha=0.2
    )

    audio_data_config = AudioDatasetConfig(
        labels_file=str(labels_file),
        sample_rate=audio_config.get('sample_rate', 16000),
        chunk_duration=audio_config.get('chunk_duration', 1.0),
        augment=True,
        volume_perturbation=0.2,
        additive_noise_std=0.005,
        time_shift_max=0.1
    )

    # Create train/val splits
    logger.info("Creating train/val splits...")
    train_dataset, val_dataset = create_multimodal_splits(
        labels_file=labels_file,
        video_config=video_data_config,
        audio_config=audio_data_config,
        val_split=training_config.val_split,
        group_by=training_config.group_by,
        seed=training_config.seed
    )

    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # Create dataloaders
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
        persistent_workers=training_config.num_workers > 0
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
        num_workers=training_config.num_workers,
        pin_memory=True,
        persistent_workers=training_config.num_workers > 0
    )

    # Store labels_file on config for heatmap ground truth overlay
    training_config.labels_file = str(labels_file)

    # Auto-select heatmap video from training data if not specified
    if training_config.heatmap_epoch_interval > 0 and not training_config.heatmap_video_path:
        # Get unique video paths from training dataset
        train_video_paths = list(set(
            train_dataset.video_dataset.index[i].video_path
            for i in range(min(100, len(train_dataset)))
        ))
        if train_video_paths:
            training_config.heatmap_video_path = train_video_paths[0]
            logger.info(f"Auto-selected heatmap video: {training_config.heatmap_video_path}")

    # Create model
    logger.info("Building multi-modal model...")
    use_audio_auxiliary = training_config.event_loss_weight > 0
    if not use_audio_auxiliary:
        logger.info("Audio event auxiliary task disabled (event_loss_weight=0)")
    model = MultiModalSynchronyModel(
        video_config=video_config,
        audio_config=audio_config,
        fusion_config=fusion_config,
        num_classes=1,
        use_audio_auxiliary=use_audio_auxiliary
    )

    # Load pretrained weights if specified
    if training_config.pretrained_video or training_config.pretrained_audio:
        logger.info("\n" + "="*60)
        logger.info("Transfer Learning Configuration:")
        logger.info(f"  Strategy: {'Load heads only' if training_config.load_heads_only else 'Load complete models'}")
        if training_config.pretrained_video:
            logger.info(f"  Video checkpoint: {training_config.pretrained_video}")
        if training_config.pretrained_audio:
            logger.info(f"  Audio checkpoint: {training_config.pretrained_audio}")

        if training_config.load_heads_only:
            logger.info("\n  Backbones: Using pretrained YOLO + Whisper (from Ultralytics/OpenAI)")
            logger.info("  Heads: Loading from your trained models")
        else:
            logger.info("\n  Loading complete models (backbones + heads)")
        logger.info("="*60 + "\n")

        model.load_pretrained(
            video_ckpt=training_config.pretrained_video,
            audio_ckpt=training_config.pretrained_audio,
            load_heads_only=training_config.load_heads_only,
            strict=False
        )

    model.to(device)

    # Print parameter counts
    param_counts = model.count_parameters()
    logger.info("Parameter counts:")
    for component, counts in param_counts.items():
        logger.info(f"  {component}: {counts['trainable']:,} / {counts['total']:,} trainable")

    # Loss functions
    sync_pos_weight = train_dataset.get_class_weights_sync()
    sync_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([sync_pos_weight], device=device)
    )

    event_criterion = None
    if use_audio_auxiliary:
        event_class_weights = train_dataset.get_class_weights_events()
        event_criterion = nn.CrossEntropyLoss(weight=event_class_weights.to(device))
        logger.info(f"Event class weights: {event_class_weights.tolist()}")

    logger.info(f"Sync pos_weight: {sync_pos_weight:.3f}")

    # Training history
    history = TrainingHistory()

    # Mixed precision scaler
    scaler = GradScaler("cuda") if use_amp else None

    # Log config
    log_reproducibility_info(
        save_dir,
        training_config.seed,
        config={
            "video": video_config,
            "audio": audio_config,
            "fusion": fusion_config,
            "training": asdict(training_config)
        }
    )

    # ========================
    # Stage 1: Freeze backbones, train fusion + heads
    # ========================
    logger.info(f"\n{'='*60}")
    logger.info(f"STAGE 1: Training fusion + heads (frozen backbones)")
    logger.info(f"Epochs: {training_config.stage1_epochs}")
    logger.info(f"Learning rate: {training_config.learning_rate}")
    logger.info(f"{'='*60}\n")

    model.freeze_backbones()

    # Optimizer for stage 1
    param_groups = model.get_parameter_groups(
        video_backbone_lr=0,  # Frozen
        audio_encoder_lr=0,  # Frozen
        fusion_lr=training_config.learning_rate
    )

    optimizer = AdamW(
        [g for g in param_groups if g['params']],
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay
    )

    # Scheduler for stage 1
    if training_config.warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=training_config.warmup_epochs
        )
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, training_config.stage1_epochs - training_config.warmup_epochs)
        )
        scheduler = SequentialLR(
            optimizer,
            [warmup_scheduler, main_scheduler],
            milestones=[training_config.warmup_epochs]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=training_config.stage1_epochs)

    # Stage 1 training loop
    for epoch in range(training_config.stage1_epochs):
        logger.info(f"\nStage 1 - Epoch {epoch+1}/{training_config.stage1_epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device,
            sync_criterion, event_criterion,
            training_config.sync_loss_weight,
            training_config.event_loss_weight,
            use_amp,
            training_config.gradient_clip_max_norm,
            training_config.label_smoothing
        )

        # Validate
        val_metrics = validate_epoch(
            model, val_loader, device,
            sync_criterion, event_criterion,
            training_config.sync_loss_weight,
            training_config.event_loss_weight,
            use_amp,
            auc_thresholds=training_config.auc_thresholds,
        )

        # Update history
        history.train_losses.append(train_metrics['loss'])
        history.train_sync_losses.append(train_metrics['sync_loss'])
        history.train_event_losses.append(train_metrics['event_loss'])
        history.train_accs.append(train_metrics['accuracy'])
        history.val_losses.append(val_metrics['loss'])
        history.val_sync_losses.append(val_metrics['sync_loss'])
        history.val_event_losses.append(val_metrics['event_loss'])
        history.val_accs.append(val_metrics['accuracy'])
        history.val_aucs.append(val_metrics['auc'])
        history.learning_rates.append(optimizer.param_groups[0]['lr'])

        # Log
        logger.info(
            f"  Train - Loss: {train_metrics['loss']:.4f}, "
            f"Acc: {train_metrics['accuracy']:.4f}"
        )
        logger.info(
            f"  Val   - Loss: {val_metrics['loss']:.4f}, "
            f"Acc: {val_metrics['accuracy']:.4f}, "
            f"AUC: {val_metrics['auc']:.4f}"
        )

        # Update best model
        if val_metrics['loss'] < history.best_val_loss:
            history.best_val_loss = val_metrics['loss']
            history.best_val_auc = val_metrics['auc']
            history.best_epoch = epoch

            torch.save({
                'epoch': epoch,
                'stage': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history.to_dict()
            }, save_dir / 'best.pt')
            logger.info("  ✓ Saved new best model")

        scheduler.step()

    # ========================
    # Stage 2: Unfreeze backbones with differential LRs
    # ========================
    logger.info(f"\n{'='*60}")
    logger.info(f"STAGE 2: Fine-tuning with unfrozen backbones")
    logger.info(f"Epochs: {training_config.epochs - training_config.stage1_epochs}")
    logger.info(f"Video backbone LR: {training_config.video_backbone_lr}")
    logger.info(f"Audio encoder LR: {training_config.audio_encoder_lr}")
    logger.info(f"Fusion/head LR: {training_config.fusion_head_lr}")
    logger.info(f"{'='*60}\n")

    model.unfreeze_backbones()

    # Optimizer for stage 2 with differential LRs
    param_groups = model.get_parameter_groups(
        video_backbone_lr=training_config.video_backbone_lr,
        audio_encoder_lr=training_config.audio_encoder_lr,
        fusion_lr=training_config.fusion_head_lr
    )

    optimizer = AdamW(
        [g for g in param_groups if g['params']],
        lr=training_config.fusion_head_lr,
        weight_decay=training_config.weight_decay
    )

    # Scheduler for stage 2 (gentler warmup)
    stage2_epochs = training_config.epochs - training_config.stage1_epochs
    if training_config.stage2_warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.3,  # Gentler warmup
            total_iters=training_config.stage2_warmup_epochs
        )
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, stage2_epochs - training_config.stage2_warmup_epochs)
        )
        scheduler = SequentialLR(
            optimizer,
            [warmup_scheduler, main_scheduler],
            milestones=[training_config.stage2_warmup_epochs]
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=stage2_epochs)

    # Stage 2 training loop
    patience_counter = 0

    for epoch in range(training_config.stage1_epochs, training_config.epochs):
        logger.info(f"\nStage 2 - Epoch {epoch+1}/{training_config.epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, device,
            sync_criterion, event_criterion,
            training_config.sync_loss_weight,
            training_config.event_loss_weight,
            use_amp,
            training_config.gradient_clip_max_norm,
            training_config.label_smoothing
        )

        # Validate
        val_metrics = validate_epoch(
            model, val_loader, device,
            sync_criterion, event_criterion,
            training_config.sync_loss_weight,
            training_config.event_loss_weight,
            use_amp,
            auc_thresholds=training_config.auc_thresholds,
        )

        # Update history
        history.train_losses.append(train_metrics['loss'])
        history.train_sync_losses.append(train_metrics['sync_loss'])
        history.train_event_losses.append(train_metrics['event_loss'])
        history.train_accs.append(train_metrics['accuracy'])
        history.val_losses.append(val_metrics['loss'])
        history.val_sync_losses.append(val_metrics['sync_loss'])
        history.val_event_losses.append(val_metrics['event_loss'])
        history.val_accs.append(val_metrics['accuracy'])
        history.val_aucs.append(val_metrics['auc'])
        history.learning_rates.append(optimizer.param_groups[0]['lr'])

        # Log
        logger.info(
            f"  Train - Loss: {train_metrics['loss']:.4f}, "
            f"Acc: {train_metrics['accuracy']:.4f}"
        )
        logger.info(
            f"  Val   - Loss: {val_metrics['loss']:.4f}, "
            f"Acc: {val_metrics['accuracy']:.4f}, "
            f"AUC: {val_metrics['auc']:.4f}"
        )

        # Update best model
        if val_metrics['loss'] < history.best_val_loss:
            history.best_val_loss = val_metrics['loss']
            history.best_val_auc = val_metrics['auc']
            history.best_epoch = epoch
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'stage': 2,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history.to_dict()
            }, save_dir / 'best.pt')
            logger.info("  ✓ Saved new best model")
        else:
            patience_counter += 1

        # Save periodic checkpoint
        if (epoch + 1) % training_config.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'stage': 2,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history.to_dict()
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pt')

        # Generate heatmap if enabled
        if (training_config.heatmap_epoch_interval > 0 and
            training_config.heatmap_video_path and
            (epoch + 1) % training_config.heatmap_epoch_interval == 0):
            try:
                from synchronai.utils.multimodal_heatmap import generate_multimodal_heatmap

                logger.info(f"\n  Generating multi-modal heatmap...")
                generate_multimodal_heatmap(
                    model=model,
                    video_path=Path(training_config.heatmap_video_path),
                    save_dir=save_dir,
                    epoch=epoch + 1,
                    device=device,
                    sample_fps=video_config.get('sample_fps', 12.0),
                    window_seconds=video_config.get('window_seconds', 2.0),
                    frame_size=video_config.get('frame_size', 640),
                    sample_rate=audio_config.get('sample_rate', 16000),
                    chunk_duration=audio_config.get('chunk_duration', 1.0),
                    threshold=0.5,
                    clip_duration=10,
                    labels_file=training_config.labels_file,
                )
            except Exception as e:
                logger.warning(f"  Heatmap generation failed: {e}")

        # Early stopping
        if patience_counter >= training_config.early_stopping_patience:
            logger.info(f"\nEarly stopping triggered after {patience_counter} epochs without improvement")
            break

        scheduler.step()

    # Save final model and history
    torch.save({
        'epoch': epoch,
        'stage': 2,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history.to_dict()
    }, save_dir / 'latest.pt')

    history.save(save_dir / 'history.json')

    logger.info(f"\n{'='*60}")
    logger.info("Training complete!")
    logger.info(f"Best epoch: {history.best_epoch + 1}")
    logger.info(f"Best val AUC: {history.best_val_auc:.4f}")
    logger.info(f"Best val loss: {history.best_val_loss:.4f}")
    logger.info(f"Models saved to: {save_dir}")
    logger.info(f"{'='*60}\n")

    return model, history


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Train multi-modal synchrony classifier")

    # Data
    parser.add_argument("--labels-file", type=str, required=True,
                        help="Path to labels CSV")
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Directory to save checkpoints")

    # Model architecture
    parser.add_argument("--video-backbone", type=str, default="yolo26s",
                        help="YOLO backbone (yolo26n, yolo26s, etc.)")
    parser.add_argument("--video-temporal-agg", type=str, default="lstm",
                        help="Temporal aggregation (mean, max, attention, lstm)")
    parser.add_argument("--audio-model", type=str, default="large-v3",
                        help="Whisper model (tiny, base, small, medium, large, large-v3)")
    parser.add_argument("--fusion-type", type=str, default="cross_attention",
                        help="Fusion type (concat, cross_attention, gated)")

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--video-backbone-lr", type=float, default=1e-5)
    parser.add_argument("--audio-encoder-lr", type=float, default=1e-5)
    parser.add_argument("--fusion-head-lr", type=float, default=5e-5)

    # Loss weights
    parser.add_argument("--sync-loss-weight", type=float, default=0.6)
    parser.add_argument("--event-loss-weight", type=float, default=0.4)

    # Transfer learning
    parser.add_argument("--pretrained-video", type=str, default=None,
                        help="Path to pretrained video model")
    parser.add_argument("--pretrained-audio", type=str, default=None,
                        help="Path to pretrained audio model")
    parser.add_argument("--load-heads-only", action="store_true",
                        help="Only load heads, keep pretrained backbones")

    # Heatmap generation
    parser.add_argument("--heatmap-epoch-interval", type=int, default=0,
                        help="Generate heatmap every N epochs (0 = disabled)")
    parser.add_argument("--heatmap-video", type=str, default=None,
                        help="Path to sample video for heatmaps (auto-selected if not provided)")

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-amp", action="store_true",
                        help="Use mixed precision training")

    # Config file (optional, overrides CLI args)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Extract configs
        video_config = config_dict.get('video', {})
        audio_config = config_dict.get('audio', {})
        fusion_config = config_dict.get('fusion', {})
        training_dict = config_dict.get('training', {})
        split_dict = config_dict.get('split', {})
        pretrained_dict = config_dict.get('pretrained', {})
    else:
        # Build configs from CLI args
        video_config = {
            'backbone': args.video_backbone,
            'temporal_aggregation': args.video_temporal_agg,
            'hidden_dim': 256,
            'dropout': 0.3,
            'freeze_backbone': True,
            'sample_fps': 12.0,
            'window_seconds': 2.0,
            'frame_size': 640
        }

        audio_config = {
            'model_name': args.audio_model,
            'pooling_mode': 'mean',
            'hidden_dim': 256,
            'dropout': 0.3,
            'freeze_encoder': True,
            'num_event_classes': 7,
            'sample_rate': 16000,
            'chunk_duration': 1.0
        }

        fusion_config = {
            'type': args.fusion_type,
            'hidden_dim': 256,
            'num_heads': 4,
            'dropout': 0.3
        }

        training_dict = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'stage1_epochs': args.stage1_epochs,
            'video_backbone_lr': args.video_backbone_lr,
            'audio_encoder_lr': args.audio_encoder_lr,
            'fusion_head_lr': args.fusion_head_lr,
            'sync_loss_weight': args.sync_loss_weight,
            'event_loss_weight': args.event_loss_weight,
            'use_amp': args.use_amp,
            'num_workers': args.num_workers,
            'seed': args.seed,
            'heatmap_epoch_interval': args.heatmap_epoch_interval,
            'heatmap_video_path': args.heatmap_video
        }

        split_dict = {
            'val_split': 0.2,
            'group_by': 'subject_id'
        }

        pretrained_dict = {
            'video': args.pretrained_video,
            'audio': args.pretrained_audio,
            'load_heads_only': args.load_heads_only
        }

    # Create training config
    training_config = MultiModalTrainingConfig(
        **training_dict,
        val_split=split_dict.get('val_split', 0.2),
        group_by=split_dict.get('group_by', 'subject_id'),
        pretrained_video=pretrained_dict.get('video'),
        pretrained_audio=pretrained_dict.get('audio'),
        load_heads_only=pretrained_dict.get('load_heads_only', True)
    )

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Train
    model, history = train_multimodal_classifier(
        labels_file=args.labels_file,
        save_dir=args.save_dir,
        video_config=video_config,
        audio_config=audio_config,
        fusion_config=fusion_config,
        training_config=training_config
    )
