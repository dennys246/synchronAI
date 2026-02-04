"""
Whisper encoder feature extraction.

Uses the encoder portion of OpenAI's Whisper model as a pretrained
audio feature extractor. The encoder transforms mel spectrograms
into rich audio representations without requiring transcription.

This enables audio classification tasks (event detection, vocalization
type, etc.) using Whisper's powerful audio understanding capabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from synchronai.data.audio.processing import get_whisper_cache_dir

logger = logging.getLogger(__name__)

# Whisper model dimensions by size
WHISPER_DIMS = {
    "tiny": 384,
    "base": 512,
    "small": 768,
    "medium": 1024,
    "large": 1280,
    "large-v2": 1280,
    "large-v3": 1280,
}


@dataclass
class WhisperEncoderConfig:
    """Configuration for Whisper encoder feature extraction."""

    model_size: str = "large-v3"
    device: Optional[str] = None
    freeze: bool = True
    # Whisper expects 30s of audio, but we process 1s chunks
    # We pad to 30s and extract features
    chunk_duration: float = 1.0
    sample_rate: int = 16000

    @property
    def encoder_dim(self) -> int:
        """Get encoder output dimension for this model size."""
        base_size = self.model_size.split("-")[0]  # Handle "large-v3" -> "large"
        return WHISPER_DIMS.get(base_size, WHISPER_DIMS.get(self.model_size, 1280))


def _lazy_import_whisper():
    """Lazy import whisper to avoid import errors if not installed."""
    try:
        import whisper

        return whisper
    except ImportError:
        raise ImportError(
            "openai-whisper is required for audio classification. "
            "Install with: pip install openai-whisper"
        )


class WhisperEncoderFeatures(nn.Module):
    """Extract features from Whisper encoder (frozen by default).

    The Whisper encoder is pretrained on 680k hours of multilingual audio,
    making it an excellent feature extractor for audio classification tasks.

    Features:
    - Works with any audio, not just speech
    - Language-agnostic representations
    - Captures acoustic patterns, vocalization types, and audio events
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__()

        self.model_size = model_size
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._freeze = freeze
        self._whisper = None
        self._encoder_dim = WhisperEncoderConfig(model_size=model_size).encoder_dim

        logger.info(
            f"WhisperEncoderFeatures: model={model_size}, "
            f"encoder_dim={self._encoder_dim}, freeze={freeze}"
        )

    def _load_model(self) -> None:
        """Lazy load Whisper model on first use."""
        if self._whisper is not None:
            return

        whisper = _lazy_import_whisper()
        cache_dir = get_whisper_cache_dir()
        logger.info(f"Loading Whisper model: {self.model_size} (cache: {cache_dir})")
        self._whisper = whisper.load_model(
            self.model_size, device=self._device, download_root=str(cache_dir)
        )

        if self._freeze:
            for param in self._whisper.encoder.parameters():
                param.requires_grad = False
            logger.info("Froze Whisper encoder parameters")

    @property
    def encoder_dim(self) -> int:
        """Output feature dimension."""
        return self._encoder_dim

    @property
    def whisper(self):
        """Get Whisper model (lazy loaded)."""
        self._load_model()
        return self._whisper

    def _audio_to_mel(self, audio: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert audio to log-mel spectrogram.

        Args:
            audio: Audio samples, shape (batch, n_samples) or (n_samples,)

        Returns:
            Mel spectrogram, shape (batch, n_mels, n_frames)
        """
        whisper_module = _lazy_import_whisper()

        # Ensure numpy array
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Handle batched input
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]

        # Determine n_mels based on model (large-v3 uses 128, others use 80)
        n_mels = 128 if self.model_size == "large-v3" else 80

        batch_size = audio.shape[0]
        mels = []

        for i in range(batch_size):
            # Pad or trim to 30 seconds (Whisper requirement)
            audio_30s = whisper_module.pad_or_trim(audio[i])
            # Compute mel spectrogram with correct n_mels for model
            mel = whisper_module.log_mel_spectrogram(audio_30s, n_mels=n_mels)
            mels.append(mel)

        return torch.stack(mels).to(self._device)

    def extract_features(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        pool: str = "mean",
    ) -> torch.Tensor:
        """Extract encoder features from audio.

        Args:
            audio: Audio samples, shape (batch, n_samples) or (n_samples,)
                   Expected sample rate: 16kHz
            pool: Pooling strategy for temporal dimension
                  "mean" - average over time
                  "max" - max over time
                  "first" - first frame only
                  "last" - last frame only
                  "none" - return full sequence

        Returns:
            Features tensor:
            - If pool != "none": shape (batch, encoder_dim)
            - If pool == "none": shape (batch, n_frames, encoder_dim)
        """
        self._load_model()

        # Convert to mel spectrogram
        mel = self._audio_to_mel(audio)

        # Run through encoder
        with torch.no_grad() if self._freeze else torch.enable_grad():
            # Whisper encoder expects (batch, n_mels, n_frames)
            features = self._whisper.encoder(mel)
            # Output shape: (batch, n_frames, encoder_dim)

        # Pool temporal dimension
        if pool == "mean":
            features = features.mean(dim=1)
        elif pool == "max":
            features = features.max(dim=1).values
        elif pool == "first":
            features = features[:, 0, :]
        elif pool == "last":
            features = features[:, -1, :]
        elif pool == "none":
            pass  # Keep full sequence
        else:
            raise ValueError(f"Unknown pooling strategy: {pool}")

        return features

    def forward(
        self,
        audio: Union[np.ndarray, torch.Tensor],
        pool: str = "mean",
    ) -> torch.Tensor:
        """Forward pass (alias for extract_features)."""
        return self.extract_features(audio, pool=pool)

    def to(self, device: str) -> "WhisperEncoderFeatures":
        """Move model to device."""
        self._device = device
        if self._whisper is not None:
            self._whisper = self._whisper.to(device)
        return self


# Cached global encoder to avoid reloading for batch processing
_cached_encoder: Optional[WhisperEncoderFeatures] = None
_cached_encoder_config: Optional[str] = None


def get_whisper_encoder(
    model_size: str = "large-v3",
    device: Optional[str] = None,
    freeze: bool = True,
    use_cache: bool = True,
) -> WhisperEncoderFeatures:
    """Get a Whisper encoder, optionally using a cached instance.

    Args:
        model_size: Whisper model size
        device: Device to use
        freeze: Whether to freeze encoder parameters
        use_cache: Whether to use/update global cache

    Returns:
        WhisperEncoderFeatures instance
    """
    global _cached_encoder, _cached_encoder_config

    cache_key = f"{model_size}_{device}_{freeze}"

    if use_cache and _cached_encoder is not None and _cached_encoder_config == cache_key:
        return _cached_encoder

    encoder = WhisperEncoderFeatures(
        model_size=model_size,
        device=device,
        freeze=freeze,
    )

    if use_cache:
        _cached_encoder = encoder
        _cached_encoder_config = cache_key

    return encoder


def clear_whisper_cache() -> None:
    """Clear the cached Whisper encoder to free memory."""
    global _cached_encoder, _cached_encoder_config
    _cached_encoder = None
    _cached_encoder_config = None
