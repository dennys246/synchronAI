"""
DINOv2 encoder feature extraction for video frames.

Uses Meta's DINOv2 (Apache 2.0 license) as a pretrained vision feature extractor.
DINOv2 was trained with self-supervised learning on 142M curated images, producing
rich visual features without task-specific labels.

Input: 224x224 images with ImageNet normalization.
Output: CLS token (B, D) or mean of patch tokens (B, D).

Requires: pip install transformers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# DINOv2 model configs: hidden_size and num_patches for 224x224 input
# patch_size=14 → 16x16 = 256 patches for all variants
DINOV2_CONFIGS = {
    "facebook/dinov2-small": {"hidden_size": 384, "num_patches": 256},
    "facebook/dinov2-base": {"hidden_size": 768, "num_patches": 256},
    "facebook/dinov2-large": {"hidden_size": 1024, "num_patches": 256},
    "facebook/dinov2-giant": {"hidden_size": 1536, "num_patches": 256},
    # Short aliases
    "dinov2-small": {"hidden_size": 384, "num_patches": 256},
    "dinov2-base": {"hidden_size": 768, "num_patches": 256},
    "dinov2-large": {"hidden_size": 1024, "num_patches": 256},
    "dinov2-giant": {"hidden_size": 1536, "num_patches": 256},
}

# Map short names to full HuggingFace model IDs
DINOV2_MODEL_MAP = {
    "dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
    "dinov2-large": "facebook/dinov2-large",
    "dinov2-giant": "facebook/dinov2-giant",
}


def _resolve_model_name(model_name: str) -> str:
    """Resolve short model names to full HuggingFace IDs."""
    return DINOV2_MODEL_MAP.get(model_name, model_name)


@dataclass
class DINOv2EncoderConfig:
    """Configuration for DINOv2 encoder feature extraction."""

    model_name: str = "facebook/dinov2-base"
    device: Optional[str] = None
    freeze: bool = True
    pool_mode: str = "cls"  # "cls" for CLS token, "mean_patch" for mean of patch tokens

    @property
    def feature_dim(self) -> int:
        """Get encoder output dimension for this model."""
        resolved = _resolve_model_name(self.model_name)
        config = DINOV2_CONFIGS.get(resolved, DINOV2_CONFIGS.get(self.model_name))
        if config:
            return config["hidden_size"]
        return 768  # Default to dinov2-base


def _lazy_import_transformers():
    """Lazy import transformers to avoid import errors if not installed."""
    try:
        from transformers import Dinov2Model
        return Dinov2Model
    except ImportError:
        raise ImportError(
            "transformers is required for DINOv2 feature extraction. "
            "Install with: pip install transformers"
        )


class DINOv2FeatureExtractor(nn.Module):
    """Extract features from DINOv2 vision transformer.

    DINOv2 was trained with self-supervised learning (iBOT + DINO objectives)
    on 142M curated images (LVD-142M). It produces excellent visual features
    for downstream tasks without any task-specific training.

    Key properties:
    - Input: 224x224 with ImageNet normalization (NOT /255 like YOLO)
    - Output: CLS token or mean of patch tokens, shape (B, hidden_size)
    - Patch size: 14x14 → 16x16 = 256 patches for 224x224 input
    - Apache 2.0 license (no AGPL restrictions)
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: Optional[str] = None,
        freeze: bool = True,
        pool_mode: str = "cls",
    ):
        super().__init__()

        self.model_name = _resolve_model_name(model_name)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._freeze = freeze
        self._pool_mode = pool_mode
        self._is_loaded = False

        # Get config for this model
        config = DINOV2_CONFIGS.get(self.model_name)
        if config is None:
            logger.warning(
                f"Unknown DINOv2 model: {self.model_name}, "
                f"defaulting to dinov2-base dimensions"
            )
            config = DINOV2_CONFIGS["facebook/dinov2-base"]

        self._feature_dim = config["hidden_size"]
        self._num_patches = config["num_patches"]

        # DINOv2 model (lazy loaded)
        self.dinov2 = None

        logger.info(
            f"DINOv2FeatureExtractor: model={self.model_name}, "
            f"feature_dim={self._feature_dim}, pool_mode={pool_mode}, "
            f"freeze={freeze}"
        )

    def _load_model(self) -> None:
        """Lazy load DINOv2 model on first use."""
        if self._is_loaded:
            return

        Dinov2Model = _lazy_import_transformers()
        logger.info(f"Loading DINOv2 model: {self.model_name}")

        self.dinov2 = Dinov2Model.from_pretrained(self.model_name)
        self.dinov2 = self.dinov2.to(self._device)

        if self._freeze:
            for param in self.dinov2.parameters():
                param.requires_grad = False
            logger.info("Froze DINOv2 backbone parameters")

        self._is_loaded = True

    @property
    def feature_dim(self) -> int:
        """Output feature dimension (read from config, no dummy forward needed)."""
        return self._feature_dim

    @property
    def num_patches(self) -> int:
        """Number of patch tokens for 224x224 input."""
        return self._num_patches

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Override to ensure DINOv2 is loaded before restoring weights."""
        has_dinov2_keys = any(k.startswith("dinov2.") for k in state_dict)
        if has_dinov2_keys and not self._is_loaded:
            self._load_model()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def freeze_backbone(self) -> None:
        """Freeze DINOv2 backbone parameters."""
        if self.dinov2 is not None:
            for param in self.dinov2.parameters():
                param.requires_grad = False
        self._freeze = True
        logger.info("Froze DINOv2 backbone parameters")

    def unfreeze_backbone(self) -> None:
        """Unfreeze DINOv2 backbone parameters for fine-tuning."""
        if self.dinov2 is not None:
            for param in self.dinov2.parameters():
                param.requires_grad = True
        self._freeze = False
        logger.info("Unfroze DINOv2 backbone parameters")

    def get_parameter_groups(
        self,
        backbone_lr: float,
        head_lr: float,
    ) -> list[dict]:
        """Get parameter groups with different learning rates.

        Only includes parameters that require gradients.

        Args:
            backbone_lr: Learning rate for DINOv2 backbone (unfrozen only)
            head_lr: Not used here (no head params in extractor), but kept
                     for API compatibility with YOLOFeatureExtractor

        Returns:
            List of parameter group dicts for optimizer
        """
        groups = []

        if self.dinov2 is not None:
            backbone_params = [
                p for p in self.dinov2.parameters() if p.requires_grad
            ]
            if backbone_params:
                groups.append({"params": backbone_params, "lr": backbone_lr})

        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images.

        Args:
            x: Input images (B, 3, 224, 224), ImageNet-normalized

        Returns:
            Features (B, feature_dim)
        """
        self._load_model()

        x = x.to(self._device)

        # Determine gradient context
        any_requires_grad = (
            self.dinov2 is not None
            and any(p.requires_grad for p in self.dinov2.parameters())
        )
        grad_ctx = torch.enable_grad() if any_requires_grad else torch.no_grad()

        with grad_ctx:
            outputs = self.dinov2(pixel_values=x)

            if self._pool_mode == "cls":
                # CLS token: shape (B, hidden_size)
                features = outputs.last_hidden_state[:, 0, :]
            elif self._pool_mode == "mean_patch":
                # Mean of patch tokens (exclude CLS at index 0)
                features = outputs.last_hidden_state[:, 1:, :].mean(dim=1)
            else:
                raise ValueError(f"Unknown pool_mode: {self._pool_mode}")

        return features

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial feature map for Grad-CAM visualization.

        Returns patch tokens reshaped to a spatial grid.

        Args:
            x: Input images (B, 3, 224, 224), ImageNet-normalized

        Returns:
            Spatial features (B, feature_dim, H_patches, W_patches)
            For 224x224 input with patch_size=14: (B, D, 16, 16)
        """
        self._load_model()

        x = x.to(self._device)

        any_requires_grad = (
            self.dinov2 is not None
            and any(p.requires_grad for p in self.dinov2.parameters())
        )
        grad_ctx = torch.enable_grad() if any_requires_grad else torch.no_grad()

        with grad_ctx:
            outputs = self.dinov2(pixel_values=x)
            # Patch tokens (exclude CLS): (B, num_patches, D)
            patch_tokens = outputs.last_hidden_state[:, 1:, :]

            # Reshape to spatial grid: (B, 256, D) → (B, 16, 16, D) → (B, D, 16, 16)
            B, N, D = patch_tokens.shape
            h = w = int(N ** 0.5)
            spatial = patch_tokens.reshape(B, h, w, D).permute(0, 3, 1, 2)

        return spatial

    def to(self, *args, **kwargs) -> "DINOv2FeatureExtractor":
        """Move model to device, updating internal device tracking."""
        result = super().to(*args, **kwargs)
        if self.dinov2 is not None:
            try:
                self._device = str(next(self.dinov2.parameters()).device)
            except StopIteration:
                pass
        return result


# Cached global encoder to avoid reloading
_cached_encoder: Optional[DINOv2FeatureExtractor] = None
_cached_encoder_config: Optional[str] = None


def get_dinov2_encoder(
    model_name: str = "facebook/dinov2-base",
    device: Optional[str] = None,
    freeze: bool = True,
    pool_mode: str = "cls",
    use_cache: bool = True,
) -> DINOv2FeatureExtractor:
    """Get a DINOv2 encoder, optionally using a cached instance.

    Args:
        model_name: DINOv2 model name (short or full HuggingFace ID)
        device: Device to use
        freeze: Whether to freeze backbone parameters
        pool_mode: "cls" for CLS token, "mean_patch" for mean of patch tokens
        use_cache: Whether to use/update global cache

    Returns:
        DINOv2FeatureExtractor instance
    """
    global _cached_encoder, _cached_encoder_config

    resolved = _resolve_model_name(model_name)
    cache_key = f"{resolved}_{device}_{freeze}_{pool_mode}"

    if use_cache and _cached_encoder is not None and _cached_encoder_config == cache_key:
        return _cached_encoder

    encoder = DINOv2FeatureExtractor(
        model_name=model_name,
        device=device,
        freeze=freeze,
        pool_mode=pool_mode,
    )

    if use_cache:
        _cached_encoder = encoder
        _cached_encoder_config = cache_key

    return encoder


def clear_dinov2_cache() -> None:
    """Clear the cached DINOv2 encoder to free memory."""
    global _cached_encoder, _cached_encoder_config
    _cached_encoder = None
    _cached_encoder_config = None
