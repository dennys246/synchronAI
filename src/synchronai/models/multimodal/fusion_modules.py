"""
Fusion modules for combining video and audio features in multi-modal models.

Provides multiple fusion strategies:
- Concatenation fusion: Simple feature concatenation
- Cross-modal attention: Attention-based feature interaction
- Gated fusion: Learnable gating mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    """
    Simple concatenation-based fusion.

    Concatenates video and audio features and projects to output dimension.

    Args:
        video_dim: Video feature dimension
        audio_dim: Audio feature dimension
        hidden_dim: Output hidden dimension
        dropout: Dropout probability
    """
    def __init__(self, video_dim: int, audio_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()

        concat_dim = video_dim + audio_dim

        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, video_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_features: (batch, video_dim)
            audio_features: (batch, audio_dim)

        Returns:
            fused_features: (batch, hidden_dim)
        """
        # Concatenate features
        concat_features = torch.cat([video_features, audio_features], dim=1)

        # Project through fusion network
        fused = self.fusion(concat_features)

        return fused


class CrossModalAttention(nn.Module):
    """
    Temporal cross-modal attention fusion using multi-head attention.

    Operates on temporal token sequences (B, T, D) from each modality so
    that attention can meaningfully weight across time steps. Video and
    audio may have different sequence lengths (T_v != T_a).

    After cross-attention, each modality's attended sequence is mean-pooled
    to produce a single vector per sample, then both are concatenated and
    projected to the output dimension.

    Args:
        video_dim: Video per-frame feature dimension
        audio_dim: Audio per-frame feature dimension
        hidden_dim: Output hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()

        # Project features to common dimension for attention
        self.video_proj = nn.Linear(video_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Cross-modal attention: video queries, audio keys/values
        self.video_to_audio_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-modal attention: audio queries, video keys/values
        self.audio_to_video_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer norms for residual connections
        self.video_norm = nn.LayerNorm(hidden_dim)
        self.audio_norm = nn.LayerNorm(hidden_dim)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        video_features: torch.Tensor,
        audio_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            video_features: (batch, T_video, video_dim) temporal sequence
            audio_features: (batch, T_audio, audio_dim) temporal sequence

        Returns:
            fused_features: (batch, hidden_dim)
        """
        # Project to common dimension: (B, T, hidden_dim)
        video_proj = self.video_proj(video_features)
        audio_proj = self.audio_proj(audio_features)

        # Video attends to audio (video queries look at audio keys)
        video_attended, _ = self.video_to_audio_attn(
            query=video_proj,
            key=audio_proj,
            value=audio_proj
        )
        # Residual + LayerNorm
        video_attended = self.video_norm(video_proj + video_attended)

        # Audio attends to video (audio queries look at video keys)
        audio_attended, _ = self.audio_to_video_attn(
            query=audio_proj,
            key=video_proj,
            value=video_proj
        )
        audio_attended = self.audio_norm(audio_proj + audio_attended)

        # Mean-pool over time after cross-attention
        video_pooled = video_attended.mean(dim=1)  # (batch, hidden_dim)
        audio_pooled = audio_attended.mean(dim=1)   # (batch, hidden_dim)

        # Combine and project
        combined = torch.cat([video_pooled, audio_pooled], dim=1)
        fused = self.fusion(combined)

        return fused


class GatedFusion(nn.Module):
    """
    Gated fusion with learnable modality weighting.

    Uses a gating mechanism to learn how to weight video vs audio features.

    Args:
        video_dim: Video feature dimension
        audio_dim: Audio feature dimension
        hidden_dim: Output hidden dimension
        dropout: Dropout probability
    """
    def __init__(self, video_dim: int, audio_dim: int, hidden_dim: int, dropout: float = 0.3):
        super().__init__()

        # Project both modalities to common dimension
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Gating network: learns to weight video vs audio
        self.gate = nn.Sequential(
            nn.Linear(video_dim + audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 2 weights: [video_weight, audio_weight]
            nn.Softmax(dim=1)
        )

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, video_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_features: (batch, video_dim)
            audio_features: (batch, audio_dim)

        Returns:
            fused_features: (batch, hidden_dim)
        """
        # Project features
        video_proj = self.video_proj(video_features)  # (batch, hidden_dim)
        audio_proj = self.audio_proj(audio_features)  # (batch, hidden_dim)

        # Compute gating weights
        gate_input = torch.cat([video_features, audio_features], dim=1)
        gate_weights = self.gate(gate_input)  # (batch, 2)

        # Apply gates
        video_weight = gate_weights[:, 0:1]  # (batch, 1)
        audio_weight = gate_weights[:, 1:2]  # (batch, 1)

        gated_features = video_weight * video_proj + audio_weight * audio_proj

        # Final fusion
        fused = self.fusion(gated_features)

        return fused


def create_fusion_module(
    fusion_type: str,
    video_dim: int,
    audio_dim: int,
    hidden_dim: int,
    num_heads: int = 4,
    dropout: float = 0.3
) -> nn.Module:
    """
    Factory function to create fusion modules.

    Args:
        fusion_type: Type of fusion ('concat', 'cross_attention', 'gated')
        video_dim: Video feature dimension
        audio_dim: Audio feature dimension
        hidden_dim: Output hidden dimension
        num_heads: Number of attention heads (for cross_attention)
        dropout: Dropout probability

    Returns:
        Fusion module
    """
    fusion_type = fusion_type.lower()

    if fusion_type == 'concat':
        return ConcatFusion(video_dim, audio_dim, hidden_dim, dropout)
    elif fusion_type == 'cross_attention':
        return CrossModalAttention(video_dim, audio_dim, hidden_dim, num_heads, dropout)
    elif fusion_type == 'gated':
        return GatedFusion(video_dim, audio_dim, hidden_dim, dropout)
    else:
        raise ValueError(
            f"Unknown fusion type: {fusion_type}. "
            f"Choose from: 'concat', 'cross_attention', 'gated'"
        )