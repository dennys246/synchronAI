"""
Audio event classifier using Whisper or WavLM encoder features.

Architecture:
- Audio encoder (Whisper or WavLM, frozen by default) for feature extraction
- Classification head for audio event detection
- Designed for per-second classification aligned with video synchrony

Supported encoders:
- Whisper (OpenAI): ASR-pretrained, good general audio features
- WavLM (Microsoft): Utterance-mixing pretrained, ideal for dyadic conversation
  and paralinguistic features (prosody, emotion, turn-taking)

Audio Event Classes:
- speech: Human speech
- laughter: Laughter
- crying: Crying/distress
- babbling: Baby/child babbling
- silence: No significant audio
- noise: Background noise
- music: Music/singing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from synchronai.models.audio.whisper_encoder import (
    WhisperEncoderFeatures,
    get_whisper_encoder,
)

logger = logging.getLogger(__name__)

# Default audio event classes
AUDIO_EVENT_CLASSES = [
    "speech",
    "laughter",
    "crying",
    "babbling",
    "silence",
    "noise",
    "music",
]


@dataclass
class AudioClassifierConfig:
    """Configuration for audio classifier model."""

    # Encoder backend: "whisper" or "wavlm"
    encoder_type: str = "whisper"

    # Whisper encoder settings (used when encoder_type="whisper")
    whisper_model_size: str = "large-v3"

    # WavLM encoder settings (used when encoder_type="wavlm")
    wavlm_model_name: str = "microsoft/wavlm-large"

    freeze_encoder: bool = True

    # Classification head settings
    hidden_dim: int = 256
    dropout: float = 0.3

    # Audio event classes
    event_classes: list[str] = field(default_factory=lambda: AUDIO_EVENT_CLASSES.copy())

    # Additional outputs
    predict_vocalization: bool = True  # Binary: any human vocalization
    predict_energy: bool = True  # Audio energy level

    # Audio settings
    sample_rate: int = 16000
    chunk_duration: float = 1.0  # Process 1-second chunks

    @property
    def num_event_classes(self) -> int:
        return len(self.event_classes)

    @property
    def encoder_dim(self) -> int:
        """Encoder output dimension (Whisper or WavLM)."""
        if self.encoder_type == "wavlm":
            from synchronai.models.audio.wavlm_encoder import WavLMEncoderConfig
            return WavLMEncoderConfig(model_name=self.wavlm_model_name).encoder_dim

        from synchronai.models.audio.whisper_encoder import WHISPER_DIMS
        base_size = self.whisper_model_size.split("-")[0]
        return WHISPER_DIMS.get(base_size, WHISPER_DIMS.get(self.whisper_model_size, 1280))


class AudioClassifier(nn.Module):
    """Audio event classifier using Whisper or WavLM encoder features.

    Input: 1-second audio waveform (16kHz, mono)
    Output: Audio event classification + optional vocalization/energy predictions
    """

    def __init__(self, config: AudioClassifierConfig):
        super().__init__()
        self.config = config

        # Build encoder based on encoder_type
        if config.encoder_type == "wavlm":
            from synchronai.models.audio.wavlm_encoder import WavLMEncoderFeatures
            self.encoder = WavLMEncoderFeatures(
                model_name=config.wavlm_model_name,
                freeze=config.freeze_encoder,
            )
        else:
            # Default: Whisper encoder
            self.encoder = WhisperEncoderFeatures(
                model_size=config.whisper_model_size,
                freeze=config.freeze_encoder,
            )

        encoder_dim = self.encoder.encoder_dim

        # Shared feature projection (deeper with BatchNorm for better gradient flow)
        proj_mid_dim = max(encoder_dim // 2, 2 * config.hidden_dim)
        self.feature_proj = nn.Sequential(
            nn.Linear(encoder_dim, proj_mid_dim),
            nn.BatchNorm1d(proj_mid_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(proj_mid_dim, config.hidden_dim),
            nn.BatchNorm1d(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Audio event classification head
        self.event_head = nn.Linear(config.hidden_dim, config.num_event_classes)

        # Vocalization detection head (binary: any human sound)
        if config.predict_vocalization:
            self.vocalization_head = nn.Linear(config.hidden_dim, 1)
        else:
            self.vocalization_head = None

        # Energy prediction head (regression)
        if config.predict_energy:
            self.energy_head = nn.Linear(config.hidden_dim, 1)
        else:
            self.energy_head = None

        encoder_name = (
            config.wavlm_model_name if config.encoder_type == "wavlm"
            else config.whisper_model_size
        )
        logger.info(
            f"AudioClassifier: encoder={config.encoder_type} ({encoder_name}), "
            f"encoder_dim={encoder_dim}, hidden_dim={config.hidden_dim}, "
            f"num_classes={config.num_event_classes}"
        )

    def forward(
        self,
        audio: Union[np.ndarray, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            audio: Audio waveform, shape (batch, n_samples) or (n_samples,)
                   Expected: 16kHz, 1 second = 16000 samples

        Returns:
            Dictionary with:
            - event_logits: (batch, num_classes) - raw logits for audio events
            - event_probs: (batch, num_classes) - softmax probabilities
            - vocalization_logit: (batch, 1) - vocalization logit (if enabled)
            - vocalization_prob: (batch, 1) - vocalization probability (if enabled)
            - energy: (batch, 1) - predicted energy level (if enabled)
        """
        # Extract Whisper encoder features (pool only over real content frames)
        features = self.encoder.extract_features(
            audio, pool="mean", chunk_duration=self.config.chunk_duration
        )

        # Project features
        hidden = self.feature_proj(features)

        # Event classification
        event_logits = self.event_head(hidden)
        event_probs = F.softmax(event_logits, dim=-1)

        outputs = {
            "event_logits": event_logits,
            "event_probs": event_probs,
            "features": features,  # Raw encoder features for analysis
            "hidden_features": hidden,  # Projected features (hidden_dim)
        }

        # Vocalization detection
        if self.vocalization_head is not None:
            voc_logit = self.vocalization_head(hidden)
            outputs["vocalization_logit"] = voc_logit
            outputs["vocalization_prob"] = torch.sigmoid(voc_logit)

        # Energy prediction
        if self.energy_head is not None:
            energy = self.energy_head(hidden)
            outputs["energy"] = energy

        return outputs

    def predict_event(
        self,
        audio: Union[np.ndarray, torch.Tensor],
    ) -> tuple[str, float]:
        """Predict single audio event class.

        Args:
            audio: Audio waveform

        Returns:
            Tuple of (event_class, confidence)
        """
        with torch.no_grad():
            outputs = self.forward(audio)
            probs = outputs["event_probs"]

            if probs.dim() == 2:
                probs = probs[0]  # Remove batch dim

            confidence, idx = probs.max(dim=-1)
            event_class = self.config.event_classes[idx.item()]

            return event_class, confidence.item()

    def freeze_encoder(self) -> None:
        """Freeze encoder parameters.

        For WavLM, delegates to encoder.freeze_encoder() which preserves
        layer_weights trainability. For Whisper, freezes all parameters.
        """
        if hasattr(self.encoder, 'freeze_encoder'):
            self.encoder.freeze_encoder()
        else:
            for param in self.encoder.parameters():
                param.requires_grad = False
        logger.info("Froze encoder parameters")

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters for fine-tuning."""
        if hasattr(self.encoder, 'unfreeze_encoder'):
            self.encoder.unfreeze_encoder()
        else:
            for param in self.encoder.parameters():
                param.requires_grad = True
        logger.info("Unfroze encoder parameters")

    def get_parameter_groups(
        self,
        encoder_lr: float,
        head_lr: float,
    ) -> list[dict]:
        """Get parameter groups with different learning rates.

        Only includes parameters that require gradients, so frozen encoder
        layers don't waste optimizer state or risk accidental updates.

        For WavLM, delegates to encoder.get_parameter_groups() which properly
        separates WavLM backbone params from always-trainable layer_weights.

        Args:
            encoder_lr: Learning rate for encoder (typically lower)
            head_lr: Learning rate for classification heads

        Returns:
            List of parameter group dicts for optimizer
        """
        groups = []

        # Encoder parameter groups
        if hasattr(self.encoder, 'get_parameter_groups'):
            # WavLM encoder: separates backbone from layer_weights
            groups.extend(self.encoder.get_parameter_groups(encoder_lr, head_lr))
        else:
            # Whisper encoder: simple requires_grad filter
            encoder_params = [
                p for p in self.encoder.parameters() if p.requires_grad
            ]
            if encoder_params:
                groups.append({"params": encoder_params, "lr": encoder_lr})

        # Head parameter groups (always present)
        groups.append({"params": self.feature_proj.parameters(), "lr": head_lr})
        groups.append({"params": self.event_head.parameters(), "lr": head_lr})

        if self.vocalization_head is not None:
            groups.append({"params": self.vocalization_head.parameters(), "lr": head_lr})

        if self.energy_head is not None:
            groups.append({"params": self.energy_head.parameters(), "lr": head_lr})

        return groups


def build_audio_classifier(config: AudioClassifierConfig) -> AudioClassifier:
    """Build an audio classifier from config.

    Args:
        config: Model configuration

    Returns:
        AudioClassifier model
    """
    return AudioClassifier(config)


def load_audio_classifier(
    weights_path: str,
    config: Optional[AudioClassifierConfig] = None,
    device: str = "cpu",
) -> tuple[AudioClassifier, AudioClassifierConfig]:
    """Load a trained audio classifier.

    Args:
        weights_path: Path to checkpoint
        config: Optional config (loaded from checkpoint if None)
        device: Device to load model on

    Returns:
        Tuple of (model, config)
    """
    checkpoint = torch.load(weights_path, map_location=device)

    if config is None:
        config_dict = checkpoint.get("config", {})
        config = AudioClassifierConfig(**config_dict)

    model = build_audio_classifier(config)

    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    logger.info(f"Loaded audio classifier from {weights_path}")
    return model, config


def save_audio_classifier(
    model: AudioClassifier,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[dict] = None,
) -> None:
    """Save audio classifier checkpoint.

    Args:
        model: Model to save
        path: Output path
        optimizer: Optional optimizer state
        epoch: Optional epoch number
        metrics: Optional metrics dict
    """
    from dataclasses import asdict

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(model.config),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if epoch is not None:
        checkpoint["epoch"] = epoch

    if metrics is not None:
        checkpoint["metrics"] = metrics

    torch.save(checkpoint, path)
    logger.info(f"Saved audio classifier to {path}")
