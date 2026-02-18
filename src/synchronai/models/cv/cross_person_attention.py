"""
Cross-person attention for dyadic synchrony modeling.

Implements symmetric cross-attention where each person's features attend
to the other person's features. Both directions share the same weights,
encoding the assumption that synchrony is a symmetric phenomenon.

Architecture:
- 1-2 layers of nn.MultiheadAttention (weight-tied for A→B and B→A)
- LayerNorm + residual connections
- Dropout for regularization
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CrossPersonAttentionLayer(nn.Module):
    """Single layer of symmetric cross-person attention.

    A queries B and B queries A using the same attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_a = nn.LayerNorm(embed_dim)
        self.norm_b = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Symmetric cross-attention between two persons.

        Args:
            features_a: Person A features (B, T, D) or (B, D)
            features_b: Person B features (B, T, D) or (B, D)

        Returns:
            Tuple of (attended_a, attended_b) with same shapes as input
        """
        # Ensure 3D for attention: (B, T, D)
        squeeze = features_a.dim() == 2
        if squeeze:
            features_a = features_a.unsqueeze(1)
            features_b = features_b.unsqueeze(1)

        # A attends to B (A is query, B is key/value)
        attended_a, _ = self.cross_attn(
            query=features_a, key=features_b, value=features_b,
        )
        attended_a = self.norm_a(features_a + self.dropout(attended_a))

        # B attends to A (same weights, symmetric)
        attended_b, _ = self.cross_attn(
            query=features_b, key=features_a, value=features_a,
        )
        attended_b = self.norm_b(features_b + self.dropout(attended_b))

        if squeeze:
            attended_a = attended_a.squeeze(1)
            attended_b = attended_b.squeeze(1)

        return attended_a, attended_b


class CrossPersonAttention(nn.Module):
    """Multi-layer cross-person attention module.

    Stacks 1-2 layers of symmetric cross-attention with shared weights
    per layer (A→B and B→A use the same nn.MultiheadAttention).

    Args:
        embed_dim: Feature dimension
        num_layers: Number of cross-attention layers (1 or 2)
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            CrossPersonAttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        logger.info(
            f"CrossPersonAttention: embed_dim={embed_dim}, "
            f"num_layers={num_layers}, num_heads={num_heads}, dropout={dropout}"
        )

    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-person attention layers.

        Args:
            features_a: Person A features (B, T, D) or (B, D)
            features_b: Person B features (B, T, D) or (B, D)

        Returns:
            Tuple of (attended_a, attended_b) with same shapes as input
        """
        for layer in self.layers:
            features_a, features_b = layer(features_a, features_b)

        return features_a, features_b

    def forward_self_attention(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Self-attention fallback for single-person mode.

        When only one person is detected, features attend to themselves.

        Args:
            features: Single person features (B, T, D) or (B, D)

        Returns:
            Self-attended features with same shape
        """
        for layer in self.layers:
            features, _ = layer(features, features)
        return features
