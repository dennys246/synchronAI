"""
PyTorch replica of the TensorFlow 1D U-Net encoder for fNIRS feature extraction.

Replicates only the encoder (down-path) and bottleneck of build_unet_1d()
from src/synchronai/models/fnirs/diffusion.py. The decoder (up-path) is not
needed since we only use the encoder as a frozen feature extractor.

Weight conversion from TF is handled by scripts/convert_fnirs_tf_to_pt.py.

Architecture (for base_width=64, depth=3, 60s windows):
    Input: (B, 472, 20) — channels-last, transposed to channels-first internally
    Timestep embed: sinusoidal(128) -> Dense(512, swish) -> Dense(128, swish)
    Level 0: ResBlock(20->64) + Conv1D(64, k=4, s=2) -> (B, 236, 64)
    Level 1: ResBlock(64->128) + Conv1D(128, k=4, s=2) -> (B, 118, 128)
    Level 2: ResBlock(128->256) + Conv1D(256, k=4, s=2) -> (B, 59, 256)
    Bottleneck: 2x ResBlock(256->512) -> (B, 59, 512)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal position embedding for diffusion timesteps.

    Matches the TF implementation in diffusion.py exactly.
    """
    timesteps = timesteps.float().unsqueeze(-1)  # (B, 1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / (half - 1)
    ).unsqueeze(0)  # (1, half)
    args = timesteps * freqs  # (B, half)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock1D(nn.Module):
    """1D residual block with timestep conditioning.

    Mirrors _res_block() in diffusion.py:
        Conv1D(3, same) -> LayerNorm -> Swish
        Dense(temb) -> Reshape+Add
        Conv1D(3, same) -> LayerNorm -> Swish -> [Dropout]
        [Conv1D(1) if channel mismatch] -> Add residual
    """

    def __init__(self, in_channels: int, out_channels: int, temb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(out_channels, eps=1e-5)

        self.temb_proj = nn.Linear(temb_dim, out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(out_channels, eps=1e-5)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 1x1 conv for channel mismatch (residual projection)
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T) — channels-first
            temb: (B, temb_dim) — timestep embedding
        Returns:
            (B, C_out, T)
        """
        residual = x

        # Conv1 -> LayerNorm -> Swish
        h = self.conv1(x)  # (B, C_out, T)
        h = h.transpose(1, 2)  # (B, T, C_out) for LayerNorm
        h = self.norm1(h)
        h = h.transpose(1, 2)  # (B, C_out, T)
        h = F.silu(h)

        # Timestep projection -> broadcast add
        t_proj = self.temb_proj(temb)  # (B, C_out)
        h = h + t_proj.unsqueeze(-1)  # (B, C_out, T) + (B, C_out, 1)

        # Conv2 -> LayerNorm -> Swish -> Dropout
        h = self.conv2(h)
        h = h.transpose(1, 2)
        h = self.norm2(h)
        h = h.transpose(1, 2)
        h = F.silu(h)
        h = self.dropout(h)

        # Residual connection
        return self.residual_proj(residual) + h


class FnirsUNetEncoderPT(nn.Module):
    """PyTorch replica of the TF 1D U-Net encoder path.

    Extracts bottleneck features from fNIRS time series using weights
    converted from the trained TensorFlow DDPM model.

    For feature extraction, pass t=0 to get representations of clean
    (unnoised) hemodynamic data. The timestep embedding weights are
    preserved because ResBlock weights were trained with them.

    Supports multi-scale feature extraction: returns bottleneck features
    by default, or intermediate encoder level outputs for richer
    representations.
    """

    def __init__(
        self,
        input_length: int = 472,
        feature_dim: int = 20,
        base_width: int = 64,
        depth: int = 3,
        time_embed_dim: int = 128,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.input_length = input_length
        self.feature_dim = feature_dim
        self.base_width = base_width
        self.depth = depth
        self.time_embed_dim = time_embed_dim

        # Timestep embedding MLP
        self.temb_dense1 = nn.Linear(time_embed_dim, time_embed_dim * 4)
        self.temb_dense2 = nn.Linear(time_embed_dim * 4, time_embed_dim)

        # Encoder down-path
        widths = [base_width * (2 ** i) for i in range(depth)]
        self.encoder_blocks = nn.ModuleList()
        self.downsample_convs = nn.ModuleList()

        in_ch = feature_dim
        for filters in widths:
            self.encoder_blocks.append(
                ResBlock1D(in_ch, filters, time_embed_dim, dropout)
            )
            self.downsample_convs.append(
                nn.Conv1d(filters, filters, kernel_size=4, stride=2, padding=1)
            )
            in_ch = filters

        # Bottleneck: 2 ResBlocks at widths[-1] * 2
        bottleneck_ch = widths[-1] * 2
        self.bottleneck1 = ResBlock1D(in_ch, bottleneck_ch, time_embed_dim, dropout)
        self.bottleneck2 = ResBlock1D(bottleneck_ch, bottleneck_ch, time_embed_dim, dropout)

        self.bottleneck_dim = bottleneck_ch
        self.encoder_dims = widths  # [64, 128, 256] for depth=3

    @property
    def output_dim(self) -> int:
        """Bottleneck feature dimension."""
        return self.bottleneck_dim

    def _compute_bottleneck_length(self, input_length: int) -> int:
        """Compute temporal length at bottleneck given input length."""
        length = input_length
        for _ in range(self.depth):
            length = (length + 1) // 2
        return length

    @classmethod
    def from_config_json(cls, config_path: str) -> "FnirsUNetEncoderPT":
        """Create encoder from a saved FnirsDiffusionConfig JSON."""
        with open(config_path) as f:
            cfg = json.load(f)
        return cls(
            input_length=int(cfg["model_len"]),
            feature_dim=int(cfg["feature_dim"]),
            base_width=int(cfg.get("unet_base_width", 64)),
            depth=int(cfg.get("unet_depth", 3)),
            time_embed_dim=int(cfg.get("unet_time_embed_dim", 128)),
            dropout=float(cfg.get("unet_dropout", 0.15)),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        return_all_levels: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Extract encoder features from fNIRS time series.

        Args:
            x: (B, T, C) fNIRS signal — channels-last (matching TF convention)
            t: (B,) diffusion timestep. Default: 0 (clean signal).
            return_all_levels: If True, return dict with all encoder level
                outputs + bottleneck for multi-scale features.

        Returns:
            If return_all_levels=False:
                (B, T_bottleneck, bottleneck_dim) — channels-last
            If return_all_levels=True:
                dict with keys 'bottleneck', 'level_0', 'level_1', ...
                Each value is (B, T_level, dim_level) — channels-last
        """
        B = x.shape[0]

        # Default t=0 for clean signal
        if t is None:
            t = torch.zeros(B, dtype=torch.int32, device=x.device)

        # Timestep embedding
        temb = sinusoidal_timestep_embedding(t, self.time_embed_dim)
        temb = F.silu(self.temb_dense1(temb))
        temb = F.silu(self.temb_dense2(temb))

        # Transpose to channels-first for Conv1d: (B, T, C) -> (B, C, T)
        h = x.transpose(1, 2)

        # Encoder down-path
        level_features = []
        for i, (block, downsample) in enumerate(
            zip(self.encoder_blocks, self.downsample_convs)
        ):
            h = block(h, temb)
            # Save level features before downsampling (channels-last)
            level_features.append(h.transpose(1, 2))
            h = downsample(h)

        # Bottleneck
        h = self.bottleneck1(h, temb)
        h = self.bottleneck2(h, temb)

        # Back to channels-last: (B, C, T) -> (B, T, C)
        bottleneck = h.transpose(1, 2)

        if return_all_levels:
            result = {"bottleneck": bottleneck}
            for i, feat in enumerate(level_features):
                result[f"level_{i}"] = feat
            return result

        return bottleneck

    def extract_features(
        self,
        x: torch.Tensor,
        pool: str = "mean",
        return_all_levels: bool = False,
    ) -> torch.Tensor:
        """Extract pooled features for classification.

        Args:
            x: (B, T, C) normalized fNIRS signal
            pool: "mean", "max", or "none"
            return_all_levels: If True, concatenate all encoder levels

        Returns:
            Pooled features:
            - pool != "none": (B, D)
            - pool == "none": (B, T_bottleneck, D)
        """
        with torch.no_grad():
            if return_all_levels:
                outputs = self.forward(x, return_all_levels=True)
                # Mean-pool each level, then concatenate
                pooled = []
                for key in sorted(outputs.keys()):
                    feat = outputs[key]  # (B, T, D)
                    pooled.append(feat.mean(dim=1))  # (B, D)
                return torch.cat(pooled, dim=-1)
            else:
                features = self.forward(x)  # (B, T, D)
                if pool == "mean":
                    return features.mean(dim=1)
                elif pool == "max":
                    return features.max(dim=1).values
                elif pool == "none":
                    return features
                else:
                    raise ValueError(f"Unknown pool: {pool}")

    @property
    def multiscale_dim(self) -> int:
        """Total feature dim when using return_all_levels=True.

        Concatenation of mean-pooled bottleneck + all encoder levels.
        For depth=3, base_width=64: 512 + 256 + 128 + 64 = 960
        """
        return self.bottleneck_dim + sum(self.encoder_dims)
