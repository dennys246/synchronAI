"""
WavLM encoder feature extraction.

Uses Microsoft's WavLM model as a pretrained audio feature extractor.
WavLM was trained with utterance mixing (overlapping speakers), making it
ideal for dyadic conversation analysis. Unlike Whisper, WavLM:
- Processes 1s audio natively (~50 frames) without 30s padding
- Provides learnable layer-wise weighted sum across transformer layers
- Captures paralinguistic features (prosody, emotion, turn-taking)

Requires: pip install transformers
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# WavLM model dimensions and layer counts
WAVLM_CONFIGS = {
    "microsoft/wavlm-large": {"hidden_size": 1024, "num_layers": 24},
    "microsoft/wavlm-base-plus": {"hidden_size": 768, "num_layers": 12},
    "microsoft/wavlm-base": {"hidden_size": 768, "num_layers": 12},
    # Short aliases (resolved to full HuggingFace names)
    "wavlm-large": {"hidden_size": 1024, "num_layers": 24},
    "wavlm-base-plus": {"hidden_size": 768, "num_layers": 12},
    "wavlm-base": {"hidden_size": 768, "num_layers": 12},
}

# Map short names to full HuggingFace model IDs
WAVLM_MODEL_MAP = {
    "wavlm-large": "microsoft/wavlm-large",
    "wavlm-base-plus": "microsoft/wavlm-base-plus",
    "wavlm-base": "microsoft/wavlm-base",
}


def _resolve_model_name(model_name: str) -> str:
    """Resolve short model names to full HuggingFace IDs."""
    return WAVLM_MODEL_MAP.get(model_name, model_name)


@dataclass
class WavLMEncoderConfig:
    """Configuration for WavLM encoder feature extraction."""

    model_name: str = "microsoft/wavlm-large"
    device: Optional[str] = None
    freeze: bool = True
    chunk_duration: float = 1.0
    sample_rate: int = 16000

    @property
    def encoder_dim(self) -> int:
        """Get encoder output dimension for this model."""
        resolved = _resolve_model_name(self.model_name)
        config = WAVLM_CONFIGS.get(resolved, WAVLM_CONFIGS.get(self.model_name))
        if config:
            return config["hidden_size"]
        return 1024  # Default to wavlm-large

    @property
    def num_layers(self) -> int:
        """Get number of transformer layers (hidden states = num_layers + 1)."""
        resolved = _resolve_model_name(self.model_name)
        config = WAVLM_CONFIGS.get(resolved, WAVLM_CONFIGS.get(self.model_name))
        if config:
            return config["num_layers"]
        return 24  # Default to wavlm-large


def _lazy_import_transformers():
    """Lazy import transformers to avoid import errors if not installed."""
    try:
        from transformers import WavLMModel
        return WavLMModel
    except ImportError:
        raise ImportError(
            "transformers is required for WavLM audio encoding. "
            "Install with: pip install transformers"
        )


class WavLMEncoderFeatures(nn.Module):
    """Extract features from WavLM encoder with learnable layer-wise weighting.

    WavLM was pretrained on 94k hours of audio with utterance mixing,
    making it excellent for dyadic conversation and paralinguistic analysis.

    Features:
    - Native 1s chunk processing (~50 frames, no 30s padding waste)
    - Learnable layer-wise weighted sum across all transformer layers
    - Different layers encode different information:
        - Lower layers: acoustic/phonetic features
        - Middle layers: prosodic features (pitch, rhythm, stress)
        - Upper layers: semantic/speaker identity features
    """

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        device: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__()

        self.model_name = _resolve_model_name(model_name)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._freeze = freeze
        self._is_loaded = False

        # Get config for this model
        config = WAVLM_CONFIGS.get(self.model_name)
        if config is None:
            logger.warning(
                f"Unknown WavLM model: {self.model_name}, "
                f"defaulting to wavlm-large dimensions"
            )
            config = WAVLM_CONFIGS["microsoft/wavlm-large"]

        self._encoder_dim = config["hidden_size"]
        self._num_layers = config["num_layers"]

        # WavLM model (lazy loaded)
        self.wavlm = None

        # Learnable layer-wise weights for weighted sum across transformer layers.
        # num_hidden_states = num_layers + 1 (includes embedding layer output).
        # These are ALWAYS trainable, even when the encoder is frozen,
        # because they are task-specific and very cheap (just n_layers+1 floats).
        n_hidden_states = self._num_layers + 1
        self.layer_weights = nn.Parameter(
            torch.ones(n_hidden_states) / n_hidden_states
        )

        logger.info(
            f"WavLMEncoderFeatures: model={self.model_name}, "
            f"encoder_dim={self._encoder_dim}, num_layers={self._num_layers}, "
            f"freeze={freeze}"
        )

    def _load_model(self) -> None:
        """Lazy load WavLM model on first use."""
        if self._is_loaded:
            return

        WavLMModel = _lazy_import_transformers()
        logger.info(f"Loading WavLM model: {self.model_name}")

        self.wavlm = WavLMModel.from_pretrained(self.model_name)
        self.wavlm = self.wavlm.to(self._device)

        if self._freeze:
            for param in self.wavlm.parameters():
                param.requires_grad = False
            logger.info("Froze WavLM encoder parameters (layer_weights remain trainable)")

        self._is_loaded = True

    @property
    def encoder_dim(self) -> int:
        """Output feature dimension."""
        return self._encoder_dim

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Override to ensure WavLM is loaded before restoring weights."""
        has_wavlm_keys = any(k.startswith("wavlm.") for k in state_dict)
        if has_wavlm_keys and not self._is_loaded:
            self._load_model()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def freeze_encoder(self) -> None:
        """Freeze WavLM encoder parameters. layer_weights remain trainable."""
        if self.wavlm is not None:
            for param in self.wavlm.parameters():
                param.requires_grad = False
        # layer_weights intentionally NOT frozen — they're task-specific
        logger.info("Froze WavLM encoder (layer_weights remain trainable)")

    def unfreeze_encoder(self) -> None:
        """Unfreeze WavLM encoder parameters for fine-tuning."""
        if self.wavlm is not None:
            for param in self.wavlm.parameters():
                param.requires_grad = True
        logger.info("Unfroze WavLM encoder parameters")

    def get_parameter_groups(
        self,
        encoder_lr: float,
        head_lr: float,
    ) -> list[dict]:
        """Get parameter groups with different learning rates.

        layer_weights always go in the head group (always trainable).
        WavLM encoder params go in the encoder group only if unfrozen.

        Args:
            encoder_lr: Learning rate for WavLM encoder
            head_lr: Learning rate for layer_weights and downstream heads

        Returns:
            List of parameter group dicts for optimizer
        """
        groups = []

        # WavLM encoder params (only if unfrozen)
        if self.wavlm is not None:
            encoder_params = [
                p for p in self.wavlm.parameters() if p.requires_grad
            ]
            if encoder_params:
                groups.append({"params": encoder_params, "lr": encoder_lr})

        # layer_weights always trainable
        groups.append({"params": [self.layer_weights], "lr": head_lr})

        return groups

    def extract_features(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        pool: str = "mean",
        chunk_duration: float = 1.0,
    ) -> torch.Tensor:
        """Extract encoder features from audio with learnable layer weighting.

        Args:
            audio: Audio waveform, shape (batch, n_samples) or (n_samples,)
                   Expected: 16kHz mono
            pool: Pooling strategy for temporal dimension
                  "mean" - average over time
                  "max" - max over time
                  "first" - first frame only
                  "last" - last frame
                  "none" - return full sequence
            chunk_duration: Duration of audio content in seconds.
                  WavLM processes natively without padding, so this is used
                  to compute content frames for consistency with Whisper API.

        Returns:
            Features tensor:
            - If pool != "none": shape (batch, encoder_dim)
            - If pool == "none": shape (batch, n_frames, encoder_dim)
        """
        self._load_model()

        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        # Ensure batched input
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        # Move to device
        audio = audio.to(self._device)

        # Determine gradient context based on actual parameter state
        any_requires_grad = any(p.requires_grad for p in self.wavlm.parameters())
        # layer_weights always need gradients
        grad_ctx = torch.enable_grad() if (any_requires_grad or self.layer_weights.requires_grad) else torch.no_grad()

        with grad_ctx:
            # WavLM takes raw waveform directly (no mel spectrogram needed)
            outputs = self.wavlm(audio, output_hidden_states=True)

            # hidden_states: tuple of (num_layers+1) tensors, each (batch, T, hidden_size)
            # Index 0 = embedding layer output, 1..N = transformer layer outputs
            hidden_states = torch.stack(outputs.hidden_states)  # (n_states, batch, T, D)

            # Learnable weighted sum across layers
            weights = F.softmax(self.layer_weights, dim=0)  # (n_states,)
            # Weighted sum: (n_states, B, T, D) * (n_states, 1, 1, 1) -> sum -> (B, T, D)
            features = (hidden_states * weights[:, None, None, None]).sum(dim=0)

        # WavLM produces ~50 frames per second of audio at 16kHz
        # (every 20ms = 50 frames/s). For chunk_duration=1.0, all frames are content.
        total_frames = features.shape[1]
        # For consistency with Whisper API, compute content frames
        # WavLM at 16kHz: ~50 frames per second
        content_frames = max(1, min(total_frames, int(50 * chunk_duration)))

        # Pool temporal dimension
        if pool == "mean":
            features = features[:, :content_frames, :].mean(dim=1)
        elif pool == "max":
            features = features[:, :content_frames, :].max(dim=1).values
        elif pool == "first":
            features = features[:, 0, :]
        elif pool == "last":
            features = features[:, content_frames - 1, :]
        elif pool == "none":
            pass  # Keep full sequence
        else:
            raise ValueError(f"Unknown pooling strategy: {pool}")

        return features

    def forward(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        pool: str = "mean",
        chunk_duration: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass (alias for extract_features)."""
        return self.extract_features(audio, pool=pool, chunk_duration=chunk_duration)

    def to(self, *args, **kwargs) -> "WavLMEncoderFeatures":
        """Move model to device, updating internal device tracking."""
        result = super().to(*args, **kwargs)
        if self.wavlm is not None:
            try:
                self._device = str(next(self.wavlm.parameters()).device)
            except StopIteration:
                pass
        return result


# Cached global encoder to avoid reloading
_cached_encoder: Optional[WavLMEncoderFeatures] = None
_cached_encoder_config: Optional[str] = None


def get_wavlm_encoder(
    model_name: str = "microsoft/wavlm-large",
    device: Optional[str] = None,
    freeze: bool = True,
    use_cache: bool = True,
) -> WavLMEncoderFeatures:
    """Get a WavLM encoder, optionally using a cached instance.

    Args:
        model_name: WavLM model name (short or full HuggingFace ID)
        device: Device to use
        freeze: Whether to freeze encoder parameters
        use_cache: Whether to use/update global cache

    Returns:
        WavLMEncoderFeatures instance
    """
    global _cached_encoder, _cached_encoder_config

    resolved = _resolve_model_name(model_name)
    cache_key = f"{resolved}_{device}_{freeze}"

    if use_cache and _cached_encoder is not None and _cached_encoder_config == cache_key:
        return _cached_encoder

    encoder = WavLMEncoderFeatures(
        model_name=model_name,
        device=device,
        freeze=freeze,
    )

    if use_cache:
        _cached_encoder = encoder
        _cached_encoder_config = cache_key

    return encoder


def clear_wavlm_cache() -> None:
    """Clear the cached WavLM encoder to free memory."""
    global _cached_encoder, _cached_encoder_config
    _cached_encoder = None
    _cached_encoder_config = None
