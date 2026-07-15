"""
Person-aware video classifier for dyadic synchrony.

Processes each person in the dyad separately through a shared DINOv2 backbone,
then uses cross-person attention to model their coupling before temporal
aggregation and classification.

Graceful fallback:
- 2 persons detected: full cross-attention pipeline
- 1 person detected: self-attention (features attend to themselves)
- 0 persons detected: full-frame mode (same as standard VideoClassifier)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from synchronai.models.cv.cross_person_attention import CrossPersonAttention
from synchronai.models.cv.video_classifier import (
    TemporalAttention,
    TemporalLSTM,
    _create_feature_extractor,
)

logger = logging.getLogger(__name__)


@dataclass
class PersonAwareConfig:
    """Configuration for person-aware video classifier."""

    # Input
    window_seconds: float = 2.0
    sample_fps: float = 12.0
    frame_height: int = 224
    frame_width: int = 224

    # Backbone
    backbone: str = "dinov2-base"
    freeze_backbone: bool = True

    # Cross-person attention
    num_cross_attn_layers: int = 1
    cross_attn_heads: int = 4
    cross_attn_dropout: float = 0.1

    # Temporal aggregation
    temporal_aggregation: Literal["mean", "max", "attention", "lstm"] = "lstm"
    hidden_dim: int = 256

    # Head
    dropout: float = 0.3
    output_dim: int = 1

    # Checkpoint
    weights_path: str = "person_aware_classifier.pt"

    @property
    def n_frames(self) -> int:
        return int(self.sample_fps * self.window_seconds)


class PersonAwareVideoClassifier(nn.Module):
    """Person-aware video classifier with cross-person attention.

    Architecture:
    1. Shared DINOv2 backbone extracts features for each person
    2. CrossPersonAttention models inter-person coupling
    3. Concatenated features [A; B] pass through temporal LSTM
    4. Classification head produces synchrony prediction

    For single-person or no-person fallback, the model degrades gracefully:
    - 1 person: self-attention, single-person temporal path
    - 0 persons: full-frame features, single-person temporal path
    """

    def __init__(self, config: PersonAwareConfig):
        super().__init__()
        self.config = config

        # Shared feature extractor (DINOv2)
        # We create a dummy VideoClassifierConfig to reuse the factory
        from synchronai.models.cv.video_classifier import VideoClassifierConfig
        _dummy_cfg = VideoClassifierConfig(
            backbone=config.backbone,
            freeze_backbone=config.freeze_backbone,
            frame_height=config.frame_height,
            frame_width=config.frame_width,
        )
        self.feature_extractor = _create_feature_extractor(_dummy_cfg)
        feature_dim = self.feature_extractor.feature_dim

        # Cross-person attention
        self.cross_attention = CrossPersonAttention(
            embed_dim=feature_dim,
            num_layers=config.num_cross_attn_layers,
            num_heads=config.cross_attn_heads,
            dropout=config.cross_attn_dropout,
        )

        # Temporal aggregation — dual-person path uses 2x feature dim
        dual_feature_dim = feature_dim * 2
        if config.temporal_aggregation == "attention":
            self.temporal_dual = TemporalAttention(dual_feature_dim, config.hidden_dim)
            self.temporal_single = TemporalAttention(feature_dim, config.hidden_dim)
            temporal_out_dim_dual = dual_feature_dim
            temporal_out_dim_single = feature_dim
        elif config.temporal_aggregation == "lstm":
            self.temporal_dual = TemporalLSTM(dual_feature_dim, config.hidden_dim)
            self.temporal_single = TemporalLSTM(feature_dim, config.hidden_dim)
            temporal_out_dim_dual = config.hidden_dim
            temporal_out_dim_single = config.hidden_dim
        elif config.temporal_aggregation in ["mean", "max"]:
            self.temporal_dual = None
            self.temporal_single = None
            temporal_out_dim_dual = dual_feature_dim
            temporal_out_dim_single = feature_dim
        else:
            raise ValueError(f"Unknown temporal aggregation: {config.temporal_aggregation}")

        # Use the larger of the two for the head (dual-person case)
        # A projection layer maps single-person features to the same dim
        self._temporal_out_dual = temporal_out_dim_dual
        self._temporal_out_single = temporal_out_dim_single

        if temporal_out_dim_single != temporal_out_dim_dual:
            self.single_to_dual_proj = nn.Linear(temporal_out_dim_single, temporal_out_dim_dual)
        else:
            self.single_to_dual_proj = nn.Identity()

        # Classification head (shared for both paths)
        head_input_dim = temporal_out_dim_dual
        head_mid_dim = max(head_input_dim, 2 * config.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, head_mid_dim),
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
            f"PersonAwareVideoClassifier: backbone={config.backbone}, "
            f"feature_dim={feature_dim}, cross_attn_layers={config.num_cross_attn_layers}, "
            f"temporal={config.temporal_aggregation}, hidden_dim={config.hidden_dim}"
        )

    def _extract_temporal_features(
        self,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """Extract per-frame features and reshape to temporal sequence.

        Args:
            frames: (B, T, C, H, W) input frames

        Returns:
            (B, T, D) feature sequence
        """
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        features = self.feature_extractor(flat)  # (B*T, D)
        return features.view(B, T, -1)  # (B, T, D)

    def _aggregate_temporal(
        self, features: torch.Tensor, dual: bool = True,
    ) -> torch.Tensor:
        """Apply temporal aggregation.

        Args:
            features: (B, T, D) feature sequence
            dual: Whether to use the dual-person temporal module

        Returns:
            (B, temporal_out_dim) aggregated features
        """
        temporal = self.temporal_dual if dual else self.temporal_single

        if self.config.temporal_aggregation == "mean":
            return features.mean(dim=1)
        elif self.config.temporal_aggregation == "max":
            return features.max(dim=1).values
        else:
            return temporal(features)

    def forward(
        self,
        person_a_crops: Optional[torch.Tensor] = None,
        person_b_crops: Optional[torch.Tensor] = None,
        full_frames: Optional[torch.Tensor] = None,
        person_count: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ):
        """Forward pass with graceful person-count fallback.

        Args:
            person_a_crops: (B, T, C, H, W) person A crops (larger/adult)
            person_b_crops: (B, T, C, H, W) person B crops (smaller/child)
            full_frames: (B, T, C, H, W) full-frame fallback
            person_count: (B,) number of persons detected per sample (0, 1, or 2)
                         If None, inferred from which inputs are provided
            return_features: If True, return dict with intermediate features

        Returns:
            Logits (B, output_dim) or dict with logits + features
        """
        B = None
        for tensor in [person_a_crops, person_b_crops, full_frames]:
            if tensor is not None:
                B = tensor.shape[0]
                break
        if B is None:
            raise ValueError("At least one of person_a_crops, person_b_crops, full_frames must be provided")

        device = next(self.parameters()).device

        # Infer person_count if not provided
        if person_count is None:
            if person_a_crops is not None and person_b_crops is not None:
                person_count = torch.full((B,), 2, dtype=torch.long, device=device)
            elif person_a_crops is not None or person_b_crops is not None:
                person_count = torch.full((B,), 1, dtype=torch.long, device=device)
            else:
                person_count = torch.full((B,), 0, dtype=torch.long, device=device)

        # Process per sample based on person count
        # For simplicity and GPU efficiency, we handle the common case
        # where all samples in a batch have the same person count
        all_logits = []
        all_features = [] if return_features else None

        # Group by person count for batched processing
        for count_val in [2, 1, 0]:
            mask = person_count == count_val
            if not mask.any():
                continue

            indices = mask.nonzero(as_tuple=True)[0]

            if count_val == 2:
                feats_a = self._extract_temporal_features(person_a_crops[indices])
                feats_b = self._extract_temporal_features(person_b_crops[indices])

                # Cross-person attention (per-frame features attend across persons)
                feats_a, feats_b = self.cross_attention(feats_a, feats_b)

                # Concatenate [A; B] along feature dim
                combined = torch.cat([feats_a, feats_b], dim=-1)  # (N, T, 2D)
                aggregated = self._aggregate_temporal(combined, dual=True)

            elif count_val == 1:
                single_crops = person_a_crops[indices] if person_a_crops is not None else person_b_crops[indices]
                feats = self._extract_temporal_features(single_crops)

                # Self-attention
                feats = self.cross_attention.forward_self_attention(feats)
                aggregated = self._aggregate_temporal(feats, dual=False)
                aggregated = self.single_to_dual_proj(aggregated)

            else:  # count_val == 0
                if full_frames is None:
                    raise ValueError("full_frames required when person_count=0")
                feats = self._extract_temporal_features(full_frames[indices])
                aggregated = self._aggregate_temporal(feats, dual=False)
                aggregated = self.single_to_dual_proj(aggregated)

            logits = self.head(aggregated)

            # Place results back in order
            for i, idx in enumerate(indices):
                all_logits.append((idx.item(), logits[i]))
                if return_features:
                    all_features.append((idx.item(), aggregated[i]))

        # Reassemble in original order
        all_logits.sort(key=lambda x: x[0])
        logits_out = torch.stack([l for _, l in all_logits])

        if return_features:
            all_features.sort(key=lambda x: x[0])
            features_out = torch.stack([f for _, f in all_features])
            return {
                'logits': logits_out,
                'temporal_features': features_out,
            }

        return logits_out

    def freeze_backbone(self) -> None:
        """Freeze backbone parameters."""
        self.feature_extractor.freeze_backbone()

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        self.feature_extractor.unfreeze_backbone()

    def get_parameter_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        """Get parameter groups with different learning rates.

        Args:
            backbone_lr: Learning rate for backbone (unfrozen only)
            head_lr: Learning rate for attention, temporal, and head

        Returns:
            List of parameter group dicts
        """
        groups = self.feature_extractor.get_parameter_groups(backbone_lr, head_lr)

        # Cross-attention params
        cross_attn_params = [p for p in self.cross_attention.parameters() if p.requires_grad]
        if cross_attn_params:
            groups.append({"params": cross_attn_params, "lr": head_lr})

        # Temporal params
        for temporal in [self.temporal_dual, self.temporal_single]:
            if temporal is not None:
                params = [p for p in temporal.parameters() if p.requires_grad]
                if params:
                    groups.append({"params": params, "lr": head_lr})

        # Projection layer
        if not isinstance(self.single_to_dual_proj, nn.Identity):
            proj_params = [p for p in self.single_to_dual_proj.parameters() if p.requires_grad]
            if proj_params:
                groups.append({"params": proj_params, "lr": head_lr})

        # Head
        head_params = [p for p in self.head.parameters() if p.requires_grad]
        if head_params:
            groups.append({"params": head_params, "lr": head_lr})

        return groups


def build_person_aware_classifier(config: PersonAwareConfig) -> PersonAwareVideoClassifier:
    """Build a person-aware video classifier from config."""
    return PersonAwareVideoClassifier(config)


def load_person_aware_classifier(
    weights_path: str,
    config: Optional[PersonAwareConfig] = None,
    device: str = "cpu",
) -> tuple[PersonAwareVideoClassifier, PersonAwareConfig]:
    """Load a trained person-aware classifier.

    Args:
        weights_path: Path to checkpoint
        config: Optional config (loaded from checkpoint if None)
        device: Device to load model on

    Returns:
        Tuple of (model, config)
    """
    checkpoint = torch.load(weights_path, map_location=device)

    if config is None:
        config = PersonAwareConfig(**checkpoint["config"])

    model = build_person_aware_classifier(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded person-aware classifier from {weights_path}")
    return model, config
