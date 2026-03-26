"""
Video classifier with pluggable backbone and temporal aggregation.

Architecture:
- Feature backbone (DINOv2 or YOLO) for per-frame feature extraction
- Temporal aggregation (mean, max, attention, or LSTM)
- MLP head for binary classification

Supported backbones:
- "dinov2-small", "dinov2-base", "dinov2-large" (Apache 2.0, HuggingFace)
- "yolo26s", "yolo11s", etc. (AGPL-3.0, Ultralytics — legacy only)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class VideoClassifierConfig:
    """Configuration for video classifier model."""

    # Input configuration
    window_seconds: float = 2.0
    sample_fps: float = 12.0
    frame_height: int = 224
    frame_width: int = 224

    # Backbone configuration
    backbone: str = "dinov2-small"
    backbone_task: str = "detect"  # Only used for YOLO backbones
    backbone_weights: Optional[str] = None
    freeze_backbone: bool = True
    gradient_checkpointing: bool = False

    # Temporal aggregation
    temporal_aggregation: Literal["mean", "max", "attention", "lstm"] = "lstm"
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
        attn_weights = self.attention(x)  # (batch, n_frames, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
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
    """Extract features from YOLO backbone/neck (legacy, AGPL-3.0)."""

    def __init__(
        self,
        backbone: str = "yolo26s",
        task: str = "detect",
        weights_path: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__()

        from ultralytics import YOLO

        if weights_path:
            model = YOLO(weights_path)
        else:
            model_name = f"{backbone}-{task}.pt" if task == "pose" else f"{backbone}.pt"
            model = YOLO(model_name)

        self.backbone = model.model.model[:10]
        self._feature_dim = None

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Froze YOLO backbone parameters")

    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            dummy = torch.zeros(1, 3, 640, 640)
            with torch.no_grad():
                features = self._forward_backbone(dummy)
            self._feature_dim = features.shape[1]
        return self._feature_dim

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.backbone:
            x = layer(x)
        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_backbone(x)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_parameter_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        groups = []
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": backbone_lr})
        return groups


def _create_feature_extractor(config: VideoClassifierConfig) -> nn.Module:
    """Factory function to create the appropriate feature extractor.

    Args:
        config: Video classifier configuration

    Returns:
        Feature extractor module with .feature_dim property
    """
    backbone = config.backbone.lower()

    if backbone.startswith("dinov2"):
        from synchronai.models.cv.dinov2_encoder import DINOv2FeatureExtractor
        return DINOv2FeatureExtractor(
            model_name=config.backbone,
            freeze=config.freeze_backbone,
        )
    elif backbone.startswith("yolo"):
        try:
            return YOLOFeatureExtractor(
                backbone=config.backbone,
                task=config.backbone_task,
                weights_path=config.backbone_weights,
                freeze=config.freeze_backbone,
            )
        except ImportError:
            raise ImportError(
                f"ultralytics is required for YOLO backbone '{config.backbone}'. "
                f"Install with: pip install ultralytics\n"
                f"Consider using a DINOv2 backbone instead (Apache 2.0 license)."
            )
    else:
        raise ValueError(
            f"Unknown backbone: '{config.backbone}'. "
            f"Supported: 'dinov2-small', 'dinov2-base', 'dinov2-large', "
            f"'yolo26s', 'yolo11s', etc."
        )


class VideoClassifier(nn.Module):
    """Video classifier with pluggable backbone and temporal aggregation."""

    def __init__(self, config: VideoClassifierConfig):
        super().__init__()
        self.config = config

        # Feature extractor (DINOv2 or YOLO)
        self.feature_extractor = _create_feature_extractor(config)

        feature_dim = self.feature_extractor.feature_dim
        logger.info(f"Feature backbone: {config.backbone}, dimension: {feature_dim}")

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
        head_mid_dim = max(temporal_out_dim, 2 * config.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(temporal_out_dim, head_mid_dim),
            nn.BatchNorm1d(head_mid_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(head_mid_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

        logger.info(
            f"Created VideoClassifier: backbone={config.backbone}, "
            f"temporal={config.temporal_aggregation}, "
            f"feature_dim={feature_dim}, hidden_dim={config.hidden_dim}, "
            f"head_mid_dim={head_mid_dim}"
        )

    def forward(self, frames: torch.Tensor, return_features: bool = False):
        """
        Args:
            frames: Input frames (batch, n_frames, C, H, W)
            return_features: If True, return dict with features and logits

        Returns:
            If return_features=False: Logits (batch, output_dim)
            If return_features=True: Dict with logits, temporal_features, frame_features
        """
        batch_size, n_frames, C, H, W = frames.shape

        # Reshape for per-frame processing
        frames_flat = frames.view(batch_size * n_frames, C, H, W)

        # Extract features
        features = self.feature_extractor(frames_flat)  # (B*T, feature_dim)

        # Reshape back to temporal sequence
        features = features.view(batch_size, n_frames, -1)  # (B, T, feature_dim)

        # Temporal aggregation
        if self.config.temporal_aggregation == "mean":
            aggregated = features.mean(dim=1)
        elif self.config.temporal_aggregation == "max":
            aggregated = features.max(dim=1).values
        else:
            aggregated = self.temporal(features)

        # Classification
        logits = self.head(aggregated)

        if return_features:
            return {
                'logits': logits,
                'temporal_features': aggregated,
                'frame_features': features,
            }
        else:
            return logits

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        self.feature_extractor.freeze_backbone()

    def unfreeze_backbone(self, mode: str = "all") -> None:
        """Unfreeze backbone parameters.

        Args:
            mode: "all" to unfreeze all layers, "last" for last few layers
                  (last mode only supported for YOLO backbones)
        """
        if mode == "all":
            self.feature_extractor.unfreeze_backbone()
        elif mode == "last":
            if isinstance(self.feature_extractor, YOLOFeatureExtractor):
                layers = list(self.feature_extractor.backbone.children())
                for layer in layers[-3:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                logger.info("Unfroze last 3 YOLO backbone layers")
            else:
                # For DINOv2, "last" unfreezes all — partial unfreezing
                # can be done via manual parameter manipulation
                self.feature_extractor.unfreeze_backbone()
                logger.info("Unfroze all backbone parameters (partial unfreeze not supported for this backbone)")

    def get_parameter_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        """Get parameter groups with different learning rates.

        Only includes parameters that require gradients.

        Args:
            backbone_lr: Learning rate for backbone (unfrozen layers only)
            head_lr: Learning rate for temporal and head

        Returns:
            List of parameter group dicts
        """
        groups = self.feature_extractor.get_parameter_groups(backbone_lr, head_lr)

        groups.append({"params": list(self.head.parameters()), "lr": head_lr})

        if self.temporal is not None:
            groups.append({"params": list(self.temporal.parameters()), "lr": head_lr})

        return groups


def build_video_classifier(config: VideoClassifierConfig) -> VideoClassifier:
    """Build a video classifier from config."""
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
