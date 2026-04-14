"""
Multi-modal fusion model for synchrony prediction.

Combines video (DINOv2/YOLO + temporal) and audio (Whisper/WavLM) models
for joint prediction. Preserves existing model architectures and adds
fusion layer on top.
"""

import logging
from dataclasses import fields as dataclass_fields

import torch
import torch.nn as nn
from typing import Dict, Optional, List
from pathlib import Path

from ..cv.video_classifier import VideoClassifier, VideoClassifierConfig
from ..audio.audio_classifier import AudioClassifier, AudioClassifierConfig
from .fusion_modules import create_fusion_module

logger = logging.getLogger(__name__)


class MultiModalSynchronyModel(nn.Module):
    """
    Multi-modal model combining video and audio for synchrony prediction.

    Architecture:
        Video: DINOv2/YOLO backbone → temporal aggregation → features
        Audio: Whisper/WavLM encoder → projection → features
        Fusion: Temporal cross-attention (on frame sequences) or
                concat/gated (on pooled vectors) → synchrony prediction

    Args:
        video_config: Configuration dict for VideoClassifier
        audio_config: Configuration dict for AudioClassifier
        fusion_config: Configuration dict for fusion module
            - type: 'concat', 'cross_attention', or 'gated'
            - hidden_dim: Output dimension (default: 256)
            - num_heads: Attention heads for cross_attention (default: 4)
            - dropout: Dropout probability (default: 0.3)
        num_classes: Number of output classes (1 for binary synchrony)
        use_audio_auxiliary: If True, also output audio event predictions
    """
    def __init__(
        self,
        video_config: Dict,
        audio_config: Dict,
        fusion_config: Dict,
        num_classes: int = 1,
        use_audio_auxiliary: bool = True
    ):
        super().__init__()

        self.use_audio_auxiliary = use_audio_auxiliary
        self.num_classes = num_classes
        self.fusion_type = fusion_config.get('type', 'concat')

        # Build config dataclasses from dicts, mapping YAML keys to dataclass fields
        video_cfg = self._build_video_config(video_config)
        audio_cfg = self._build_audio_config(audio_config)

        # Initialize video and audio models (reuse existing architectures)
        self.video_model = VideoClassifier(video_cfg)
        self.audio_model = AudioClassifier(audio_cfg)

        # Get feature dimensions from models
        # Video pooled: head input = temporal aggregation output dimension
        video_pooled_dim = self.video_model.head[0].in_features
        # Video frame-level: raw backbone feature dimension (before temporal agg)
        video_frame_dim = self.video_model.feature_extractor.feature_dim
        # Audio pooled: projected feature dimension
        audio_pooled_dim = self.audio_model.config.hidden_dim
        # Audio frame-level: raw encoder output dimension (before projection)
        audio_frame_dim = self.audio_model.config.encoder_dim

        # Fusion module
        fusion_hidden_dim = fusion_config.get('hidden_dim', 256)
        num_heads = fusion_config.get('num_heads', 4)
        dropout = fusion_config.get('dropout', 0.3)

        # cross_attention operates on temporal sequences (B, T, D) using
        # raw frame-level features; concat/gated use pooled vectors.
        if self.fusion_type == 'cross_attention':
            video_dim = video_frame_dim
            audio_dim = audio_frame_dim
        else:
            video_dim = video_pooled_dim
            audio_dim = audio_pooled_dim

        self.fusion_module = create_fusion_module(
            fusion_type=self.fusion_type,
            video_dim=video_dim,
            audio_dim=audio_dim,
            hidden_dim=fusion_hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Synchrony prediction head (multi-task with audio)
        self.sync_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.BatchNorm1d(fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, num_classes)
        )

    @staticmethod
    def _build_video_config(cfg: Dict) -> VideoClassifierConfig:
        """Convert a config dict (from YAML) to VideoClassifierConfig."""
        mapped = dict(cfg)
        # Map frame_size → frame_height/frame_width
        if 'frame_size' in mapped:
            size = mapped.pop('frame_size')
            mapped.setdefault('frame_height', size)
            mapped.setdefault('frame_width', size)
        else:
            # Default frame size based on backbone
            backbone = mapped.get('backbone', 'dinov2-small')
            if backbone.startswith('dinov2'):
                mapped.setdefault('frame_height', 224)
                mapped.setdefault('frame_width', 224)
        # Filter to valid dataclass fields only
        valid = {f.name for f in dataclass_fields(VideoClassifierConfig)}
        return VideoClassifierConfig(**{k: v for k, v in mapped.items() if k in valid})

    @staticmethod
    def _build_audio_config(cfg: Dict) -> AudioClassifierConfig:
        """Convert a config dict (from YAML) to AudioClassifierConfig.

        Supports both Whisper and WavLM model names:
        - Whisper: model_name = "large-v3", "tiny", "base", etc.
        - WavLM: model_name = "wavlm-large", "wavlm-base-plus",
                 or "microsoft/wavlm-large", etc.
        """
        mapped = dict(cfg)
        if 'model_name' in mapped:
            model_name = mapped.pop('model_name')
            # Detect WavLM model names
            if model_name.startswith('wavlm') or model_name.startswith('microsoft/wavlm'):
                mapped.setdefault('encoder_type', 'wavlm')
                mapped.setdefault('wavlm_model_name', model_name)
            else:
                mapped.setdefault('encoder_type', 'whisper')
                mapped.setdefault('whisper_model_size', model_name)
        # Filter to valid dataclass fields only
        valid = {f.name for f in dataclass_fields(AudioClassifierConfig)}
        return AudioClassifierConfig(**{k: v for k, v in mapped.items() if k in valid})

    def forward(
        self,
        video_frames: torch.Tensor,
        audio_chunks: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-modal model.

        Args:
            video_frames: (batch, n_frames, 3, H, W)
            audio_chunks: (batch, n_samples) at 16kHz
            return_features: If True, return intermediate features

        Returns:
            Dictionary containing:
                - sync_logits: (batch, num_classes) synchrony logits
                - sync_probs: (batch, num_classes) synchrony probabilities
                - event_logits: (batch, num_event_classes) [if use_audio_auxiliary]
                - event_probs: (batch, num_event_classes) [if use_audio_auxiliary]
                - video_features: (batch, video_dim) [if return_features]
                - audio_features: (batch, audio_dim) [if return_features]
                - fused_features: (batch, fusion_dim) [if return_features]
        """
        # Get video features (don't use video classifier head)
        video_output = self.video_model(video_frames, return_features=True)
        video_features = video_output['temporal_features']  # (batch, video_dim)

        # For cross_attention, also get frame-level sequences
        need_sequences = self.fusion_type == 'cross_attention'

        # Get audio features
        audio_output = self.audio_model(
            audio_chunks, return_sequence=need_sequences
        )
        audio_features = audio_output['hidden_features']  # (batch, audio_dim)

        # Fuse modalities
        if need_sequences:
            # Cross-attention uses temporal sequences (B, T, D)
            video_sequence = video_output['frame_features']       # (B, T_v, feat_dim)
            audio_sequence = audio_output['sequence_features']    # (B, T_a, enc_dim)
            fused_features = self.fusion_module(video_sequence, audio_sequence)
        else:
            # Concat/gated use pooled vectors (B, D)
            fused_features = self.fusion_module(video_features, audio_features)

        # Synchrony prediction
        sync_logits = self.sync_head(fused_features)

        # Build output dictionary
        output = {
            'sync_logits': sync_logits,
            'sync_probs': torch.sigmoid(sync_logits) if self.num_classes == 1 else torch.softmax(sync_logits, dim=1)
        }

        # Add audio auxiliary outputs (for multi-task learning)
        if self.use_audio_auxiliary:
            output['event_logits'] = audio_output['event_logits']
            output['event_probs'] = audio_output['event_probs']

        # Add features if requested
        if return_features:
            output['video_features'] = video_features
            output['audio_features'] = audio_features
            output['fused_features'] = fused_features

        return output

    def get_parameter_groups(
        self,
        video_backbone_lr: float,
        audio_encoder_lr: float,
        fusion_lr: float
    ) -> List[Dict]:
        """
        Get parameter groups for optimizer with differential learning rates.

        Only includes parameters with requires_grad=True.

        Args:
            video_backbone_lr: Learning rate for video backbone
            audio_encoder_lr: Learning rate for audio encoder
            fusion_lr: Learning rate for fusion module and heads

        Returns:
            List of parameter group dictionaries
        """
        param_groups = []

        # Video model parameters
        video_groups = self.video_model.get_parameter_groups(
            backbone_lr=video_backbone_lr,
            head_lr=fusion_lr  # Video head gets same LR as fusion
        )
        param_groups.extend(video_groups)

        # Audio model parameters
        audio_groups = self.audio_model.get_parameter_groups(
            encoder_lr=audio_encoder_lr,
            head_lr=fusion_lr  # Audio head gets same LR as fusion
        )
        param_groups.extend(audio_groups)

        # Fusion module parameters
        fusion_params = [p for p in self.fusion_module.parameters() if p.requires_grad]
        if fusion_params:
            param_groups.append({
                'params': fusion_params,
                'lr': fusion_lr,
                'name': 'fusion_module'
            })

        # Synchrony head parameters
        sync_params = [p for p in self.sync_head.parameters() if p.requires_grad]
        if sync_params:
            param_groups.append({
                'params': sync_params,
                'lr': fusion_lr,
                'name': 'sync_head'
            })

        return param_groups

    def freeze_backbones(self):
        """Freeze both video and audio backbones (for stage 1 training)."""
        self.video_model.freeze_backbone()
        self.audio_model.freeze_encoder()

    def unfreeze_backbones(self):
        """Unfreeze both video and audio backbones (for stage 2 training)."""
        self.video_model.unfreeze_backbone()
        self.audio_model.unfreeze_encoder()

    def load_pretrained(
        self,
        video_ckpt: Optional[str] = None,
        audio_ckpt: Optional[str] = None,
        load_heads_only: bool = False,
        strict: bool = False
    ):
        """
        Load pretrained weights for video and/or audio models.

        Transfer Learning Strategies:

        1. **Default (from scratch)**:
           - Video: Pretrained YOLO backbone (Ultralytics) + random head
           - Audio: Pretrained Whisper encoder (OpenAI) + random head
           - This happens automatically when you don't provide checkpoints

        2. **Load complete models** (load_heads_only=False):
           - Loads full model state (backbone + head + temporal/projection layers)
           - Use when you want to start from your fully trained video/audio models
           - Example: video_ckpt="runs/video_classifier/best.pt"

        3. **Load heads only** (load_heads_only=True):
           - Keeps pretrained YOLO/Whisper backbones
           - Only loads trained classification heads from your models
           - Best of both worlds: strong backbones + task-adapted heads

        Args:
            video_ckpt: Path to your trained video model checkpoint (optional)
            audio_ckpt: Path to your trained audio model checkpoint (optional)
            load_heads_only: If True, only load heads/temporal layers (not backbones)
            strict: If True, strictly enforce state dict keys match
        """
        if video_ckpt:
            video_path = Path(video_ckpt)
            if not video_path.exists():
                raise FileNotFoundError(f"Video checkpoint not found: {video_ckpt}")

            checkpoint = torch.load(video_ckpt, map_location='cpu')

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            if load_heads_only:
                # Only load non-backbone weights (head + temporal layers)
                # Filter out backbone keys for both YOLO and DINOv2:
                # - YOLO: feature_extractor.backbone.*
                # - DINOv2: feature_extractor.dinov2.*
                filtered_state_dict = {
                    k: v for k, v in state_dict.items()
                    if not k.startswith('feature_extractor.backbone.')
                    and not k.startswith('feature_extractor.dinov2.')
                }
                missing, unexpected = self.video_model.load_state_dict(
                    filtered_state_dict, strict=False
                )
                num_head = len([k for k in filtered_state_dict.keys() if 'head' in k])
                num_temporal = len([k for k in filtered_state_dict.keys() if 'temporal' in k])
                print(f"Loaded video HEAD + TEMPORAL from {video_ckpt}")
                print(f"  Kept pretrained backbone")
                print(f"  Loaded: {num_head} head params, {num_temporal} temporal params")
            else:
                # Load complete model
                self.video_model.load_state_dict(state_dict, strict=strict)
                print(f"✓ Loaded complete video model from {video_ckpt}")

        if audio_ckpt:
            audio_path = Path(audio_ckpt)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio checkpoint not found: {audio_ckpt}")

            checkpoint = torch.load(audio_ckpt, map_location='cpu')

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            if load_heads_only:
                # Only load non-encoder weights (projection + classification heads)
                # Filter out backbone keys for both Whisper and WavLM:
                # - Whisper: encoder.encoder.* (WhisperEncoderFeatures.encoder)
                # - WavLM: encoder.wavlm.* (WavLMEncoderFeatures.wavlm)
                filtered_state_dict = {
                    k: v for k, v in state_dict.items()
                    if not k.startswith('encoder.encoder.')
                    and not k.startswith('encoder.wavlm.')
                }
                missing, unexpected = self.audio_model.load_state_dict(
                    filtered_state_dict, strict=False
                )
                num_proj = len([k for k in filtered_state_dict.keys() if 'feature_proj' in k])
                num_heads = len([k for k in filtered_state_dict.keys() if '_head' in k])
                encoder_type = getattr(self.audio_model.config, 'encoder_type', 'whisper')
                backbone_name = "WavLM" if encoder_type == "wavlm" else "Whisper"
                print(f"✓ Loaded audio PROJECTION + HEADS from {audio_ckpt}")
                print(f"  Kept pretrained {backbone_name} encoder")
                print(f"  Loaded: {num_proj} projection params, {num_heads} head params")
            else:
                # Load complete model
                self.audio_model.load_state_dict(state_dict, strict=strict)
                print(f"✓ Loaded complete audio model from {audio_ckpt}")

    def count_parameters(self) -> Dict[str, int]:
        """
        Count trainable and total parameters in each component.

        Returns:
            Dictionary with parameter counts
        """
        def count_params(module):
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return {'total': total, 'trainable': trainable}

        return {
            'video_model': count_params(self.video_model),
            'audio_model': count_params(self.audio_model),
            'fusion_module': count_params(self.fusion_module),
            'sync_head': count_params(self.sync_head),
            'total': count_params(self)
        }
