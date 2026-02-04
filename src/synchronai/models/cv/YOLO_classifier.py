"""
Video classifier using YOLO backbone with temporal aggregation.

Architecture:
- YOLO backbone/neck for per-frame feature extraction
- Temporal aggregation (mean, max, attention, or LSTM)
- MLP head for binary classification
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class VideoClassifierConfig:
    """Configuration for video classifier model."""

    # Input configuration
    window_seconds: float = 1.0
    sample_fps: float = 12.0
    frame_height: int = 640
    frame_width: int = 640

    # Backbone configuration
    backbone: str = "yolo11n"  # yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
    backbone_task: str = "detect"  # detect or pose
    backbone_weights: Optional[str] = None
    freeze_backbone: bool = True
    gradient_checkpointing: bool = False

    # Temporal aggregation
    temporal_aggregation: Literal["mean", "max", "attention", "lstm"] = "attention"
    hidden_dim: int = 256

    # Head configuration
    dropout: float = 0.3
    output_dim: int = 1

    # Checkpoint path
    weights_path: str = "video_classifier.pt"

    @property
    def n_frames(self) -> int:
        return int(self.sample_fps * self.window_seconds)


class TemporalAttention(nn.Module):
    """Learnable attention weights over temporal dimension."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch, n_frames, feature_dim)

        Returns:
            Aggregated features (batch, feature_dim)
        """
        # Compute attention weights
        attn_weights = self.attention(x)  # (batch, n_frames, 1)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Weighted sum
        output = (x * attn_weights).sum(dim=1)  # (batch, feature_dim)
        return output


class TemporalLSTM(nn.Module):
    """LSTM-based temporal aggregation."""

    def __init__(self, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch, n_frames, feature_dim)

        Returns:
            Aggregated features (batch, hidden_dim)
        """
        _, (h_n, _) = self.lstm(x)
        return h_n.squeeze(0)  # (batch, hidden_dim)


class YOLOFeatureExtractor(nn.Module):
    """Extract features from YOLO backbone/neck."""

    def __init__(
        self,
        backbone: str = "yolo11n",
        task: str = "detect",
        weights_path: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__()

        from ultralytics import YOLO

        # Load YOLO model
        if weights_path:
            model = YOLO(weights_path)
        else:
            # Load pretrained weights
            model_name = f"{backbone}-{task}.pt" if task == "pose" else f"{backbone}.pt"
            model = YOLO(model_name)

        # Extract backbone and neck
        # The exact layers depend on YOLO version
        self.backbone = model.model.model[:10]  # Backbone + neck layers

        # Get feature dimension by running a dummy forward
        self._feature_dim = None

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Froze YOLO backbone parameters")

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            # Compute feature dim with dummy input
            dummy = torch.zeros(1, 3, 640, 640)
            with torch.no_grad():
                features = self._forward_backbone(dummy)
            self._feature_dim = features.shape[1]
        return self._feature_dim

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through backbone layers."""
        for layer in self.backbone:
            x = layer(x)

        # Global average pooling
        if x.dim() == 4:  # (batch, C, H, W)
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input frames (batch, C, H, W)

        Returns:
            Features (batch, feature_dim)
        """
        return self._forward_backbone(x)


class VideoClassifier(nn.Module):
    """Video classifier with YOLO backbone and temporal aggregation."""

    def __init__(self, config: VideoClassifierConfig):
        super().__init__()
        self.config = config

        # Feature extractor
        self.feature_extractor = YOLOFeatureExtractor(
            backbone=config.backbone,
            task=config.backbone_task,
            weights_path=config.backbone_weights,
            freeze=config.freeze_backbone,
        )

        feature_dim = self.feature_extractor.feature_dim
        logger.info(f"YOLO feature dimension: {feature_dim}")

        # Temporal aggregation
        if config.temporal_aggregation == "attention":
            self.temporal = TemporalAttention(feature_dim, config.hidden_dim)
            temporal_out_dim = feature_dim
        elif config.temporal_aggregation == "lstm":
            self.temporal = TemporalLSTM(feature_dim, config.hidden_dim)
            temporal_out_dim = config.hidden_dim
        elif config.temporal_aggregation in ["mean", "max"]:
            self.temporal = None
            temporal_out_dim = feature_dim
        else:
            raise ValueError(f"Unknown temporal aggregation: {config.temporal_aggregation}")

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(temporal_out_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

        logger.info(
            f"Created VideoClassifier: backbone={config.backbone}, "
            f"temporal={config.temporal_aggregation}, "
            f"feature_dim={feature_dim}, hidden_dim={config.hidden_dim}"
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: Input frames (batch, n_frames, C, H, W)

        Returns:
            Logits (batch, 1)
        """
        batch_size, n_frames, C, H, W = frames.shape

        # Reshape for per-frame processing
        frames_flat = frames.view(batch_size * n_frames, C, H, W)

        # Extract features
        features = self.feature_extractor(frames_flat)  # (batch * n_frames, feature_dim)

        # Reshape back to temporal sequence
        features = features.view(batch_size, n_frames, -1)  # (batch, n_frames, feature_dim)

        # Temporal aggregation
        if self.config.temporal_aggregation == "mean":
            aggregated = features.mean(dim=1)
        elif self.config.temporal_aggregation == "max":
            aggregated = features.max(dim=1).values
        else:
            aggregated = self.temporal(features)

        # Classification
        logits = self.head(aggregated)

        return logits

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        for param in self.feature_extractor.backbone.parameters():
            param.requires_grad = False
        logger.info("Froze backbone parameters")

    def unfreeze_backbone(self, mode: str = "all") -> None:
        """Unfreeze backbone parameters.

        Args:
            mode: "all" to unfreeze all layers, "last" for last few layers
        """
        if mode == "all":
            for param in self.feature_extractor.backbone.parameters():
                param.requires_grad = True
            logger.info("Unfroze all backbone parameters")
        elif mode == "last":
            # Unfreeze last 3 layers
            layers = list(self.feature_extractor.backbone.children())
            for layer in layers[-3:]:
                for param in layer.parameters():
                    param.requires_grad = True
            logger.info("Unfroze last 3 backbone layers")

    def get_parameter_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        """Get parameter groups with different learning rates.

        Args:
            backbone_lr: Learning rate for backbone
            head_lr: Learning rate for temporal and head

        Returns:
            List of parameter group dicts
        """
        groups = [
            {"params": self.feature_extractor.parameters(), "lr": backbone_lr},
            {"params": self.head.parameters(), "lr": head_lr},
        ]

        if self.temporal is not None:
            groups.append({"params": self.temporal.parameters(), "lr": head_lr})

        return groups


def build_video_classifier(config: VideoClassifierConfig) -> VideoClassifier:
    """Build a video classifier from config.

    Args:
        config: Model configuration

    Returns:
        VideoClassifier model
    """
    return VideoClassifier(config)


def load_video_classifier(
    weights_path: str,
    config: Optional[VideoClassifierConfig] = None,
    device: str = "cpu",
) -> tuple[VideoClassifier, VideoClassifierConfig]:
    """Load a trained video classifier.

    Args:
        weights_path: Path to checkpoint
        config: Optional config (loaded from checkpoint if None)
        device: Device to load model on

    Returns:
        Tuple of (model, config)
    """
    checkpoint = torch.load(weights_path, map_location=device)

    if config is None:
        config = VideoClassifierConfig(**checkpoint["config"])

    model = build_video_classifier(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded video classifier from {weights_path}")
    return model, config
