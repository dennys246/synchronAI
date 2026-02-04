"""
Diffusion model components for fNIRS.

Implements:
- A small 1D U-Net that predicts diffusion noise (epsilon) for hemoglobin time series
- A DDPM-style diffusion schedule
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf


def sinusoidal_timestep_embedding(timesteps: tf.Tensor, dim: int) -> tf.Tensor:
    """
    Standard sinusoidal position embedding for diffusion timesteps.
    """
    timesteps = tf.cast(timesteps, tf.float32)[:, None]  # (batch, 1)
    half = dim // 2
    freqs = tf.exp(
        -np.log(10_000.0) * tf.range(half, dtype=tf.float32)[None, :] / tf.cast(half - 1, tf.float32)
    )
    args = timesteps * freqs
    emb = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
    if dim % 2 == 1:
        emb = tf.pad(emb, [[0, 0], [0, 1]])
    return emb


@dataclass(frozen=True)
class DiffusionSchedule:
    timesteps: int
    betas: tf.Tensor  # (T,)
    alphas: tf.Tensor  # (T,)
    alpha_bars: tf.Tensor  # (T,)
    sqrt_alpha_bars: tf.Tensor  # (T,)
    sqrt_one_minus_alpha_bars: tf.Tensor  # (T,)


def make_linear_beta_schedule(
    timesteps: int,
    *,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> DiffusionSchedule:
    """
    A simple linear beta schedule (good enough for a baseline).
    """
    if timesteps < 2:
        raise ValueError("timesteps must be >= 2")

    betas_np = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
    alphas_np = 1.0 - betas_np
    alpha_bars_np = np.cumprod(alphas_np, axis=0)

    betas = tf.constant(betas_np, dtype=tf.float32)
    alphas = tf.constant(alphas_np, dtype=tf.float32)
    alpha_bars = tf.constant(alpha_bars_np, dtype=tf.float32)
    sqrt_alpha_bars = tf.constant(np.sqrt(alpha_bars_np), dtype=tf.float32)
    sqrt_one_minus_alpha_bars = tf.constant(np.sqrt(1.0 - alpha_bars_np), dtype=tf.float32)

    return DiffusionSchedule(
        timesteps=timesteps,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=sqrt_alpha_bars,
        sqrt_one_minus_alpha_bars=sqrt_one_minus_alpha_bars,
    )


def make_cosine_beta_schedule(
    timesteps: int,
    *,
    s: float = 0.008,
) -> DiffusionSchedule:
    """
    Cosine beta schedule as proposed in https://arxiv.org/abs/2102.09672

    Provides more uniform noise addition across timesteps compared to linear schedule,
    often resulting in better sample quality.

    Args:
        timesteps: Number of diffusion timesteps
        s: Small offset to prevent beta from being too small near t=0

    Returns:
        DiffusionSchedule with cosine-based beta values
    """
    if timesteps < 2:
        raise ValueError("timesteps must be >= 2")

    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps, dtype=np.float64)

    # Compute alpha_bar using cosine schedule
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1.0 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    # Derive betas from alpha_bar
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, 0.0001, 0.9999).astype(np.float32)

    # Compute alphas and alpha_bars
    alphas_np = 1.0 - betas
    alpha_bars_np = np.cumprod(alphas_np, axis=0)

    betas = tf.constant(betas, dtype=tf.float32)
    alphas = tf.constant(alphas_np, dtype=tf.float32)
    alpha_bars = tf.constant(alpha_bars_np, dtype=tf.float32)
    sqrt_alpha_bars = tf.constant(np.sqrt(alpha_bars_np), dtype=tf.float32)
    sqrt_one_minus_alpha_bars = tf.constant(np.sqrt(1.0 - alpha_bars_np), dtype=tf.float32)

    return DiffusionSchedule(
        timesteps=timesteps,
        betas=betas,
        alphas=alphas,
        alpha_bars=alpha_bars,
        sqrt_alpha_bars=sqrt_alpha_bars,
        sqrt_one_minus_alpha_bars=sqrt_one_minus_alpha_bars,
    )


def build_unet_1d(
    *,
    input_length: int,
    feature_dim: int,
    base_width: int = 64,
    depth: int = 3,
    time_embed_dim: int = 128,
    dropout: float = 0.0,
    name: str = "fnirs_unet_1d",
) -> tf.keras.Model:
    """
    A small 1D U-Net that predicts noise epsilon(x_t, t).

    Inputs:
      x: (batch, time, features)
      t: (batch,) int32 diffusion timestep
    """

    x_in = tf.keras.layers.Input(shape=(input_length, feature_dim), dtype=tf.float32, name="x")
    t_in = tf.keras.layers.Input(shape=(), dtype=tf.int32, name="t")

    # Timestep embedding MLP
    temb = tf.keras.layers.Lambda(lambda tt: sinusoidal_timestep_embedding(tt, time_embed_dim), name="t_embed")(t_in)
    temb = tf.keras.layers.Dense(time_embed_dim * 4, activation="swish")(temb)
    temb = tf.keras.layers.Dense(time_embed_dim, activation="swish")(temb)

    def _res_block(x, filters: int, temb: tf.Tensor):
        h = tf.keras.layers.Conv1D(filters, 3, padding="same")(x)
        h = tf.keras.layers.LayerNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.Activation("swish")(h)

        tproj = tf.keras.layers.Dense(filters)(temb)
        h = tf.keras.layers.Add()([h, tf.keras.layers.Reshape((1, filters))(tproj)])

        h = tf.keras.layers.Conv1D(filters, 3, padding="same")(h)
        h = tf.keras.layers.LayerNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.Activation("swish")(h)
        if dropout:
            h = tf.keras.layers.Dropout(dropout)(h)

        if x.shape[-1] != filters:
            x = tf.keras.layers.Conv1D(filters, 1, padding="same")(x)
        return tf.keras.layers.Add()([x, h])

    skips = []
    x = x_in

    # Down path
    widths = [base_width * (2**i) for i in range(depth)]
    for filters in widths:
        x = _res_block(x, filters, temb)
        skips.append(x)
        x = tf.keras.layers.Conv1D(filters, 4, strides=2, padding="same")(x)

    # Bottleneck
    x = _res_block(x, widths[-1] * 2, temb)
    x = _res_block(x, widths[-1] * 2, temb)

    # Up path
    for filters, skip in reversed(list(zip(widths, skips))):
        x = tf.keras.layers.UpSampling1D(size=2)(x)
        x = tf.keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = tf.keras.layers.Concatenate(axis=-1)([x, skip])
        x = _res_block(x, filters, temb)

    out = tf.keras.layers.Conv1D(feature_dim, 1, padding="same", name="eps")(x)
    return tf.keras.Model(inputs=[x_in, t_in], outputs=out, name=name)


def pad_length_to_multiple(length: int, multiple: int) -> int:
    if multiple <= 1:
        return int(length)
    return int(((length + multiple - 1) // multiple) * multiple)


@dataclass
class FnirsDiffusionConfig:
    """
    Serializable config to reproduce training + generation exactly.
    """

    # Data / shape
    duration_seconds: float
    sfreq_hz: float
    target_len: int
    model_len: int
    pair_names: list[str]
    hb_types: list[str]
    feature_dim: int

    # Normalization (feature-wise)
    feature_mean: list[float]
    feature_std: list[float]

    # Model
    unet_base_width: int = 64
    unet_depth: int = 3
    unet_time_embed_dim: int = 128
    unet_dropout: float = 0.0

    # Diffusion schedule
    diffusion_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

    # Checkpointing
    weights_path: str = "fnirs_unet.weights.h5"

    def to_dict(self) -> dict:
        return {
            "duration_seconds": self.duration_seconds,
            "sfreq_hz": self.sfreq_hz,
            "target_len": self.target_len,
            "model_len": self.model_len,
            "pair_names": list(self.pair_names),
            "hb_types": list(self.hb_types),
            "feature_dim": self.feature_dim,
            "feature_mean": list(self.feature_mean),
            "feature_std": list(self.feature_std),
            "unet_base_width": self.unet_base_width,
            "unet_depth": self.unet_depth,
            "unet_time_embed_dim": self.unet_time_embed_dim,
            "unet_dropout": self.unet_dropout,
            "diffusion_timesteps": self.diffusion_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "weights_path": self.weights_path,
        }

    @staticmethod
    def from_dict(d: dict) -> "FnirsDiffusionConfig":
        return FnirsDiffusionConfig(
            duration_seconds=float(d["duration_seconds"]),
            sfreq_hz=float(d["sfreq_hz"]),
            target_len=int(d["target_len"]),
            model_len=int(d["model_len"]),
            pair_names=list(d["pair_names"]),
            hb_types=list(d["hb_types"]),
            feature_dim=int(d["feature_dim"]),
            feature_mean=list(d["feature_mean"]),
            feature_std=list(d["feature_std"]),
            unet_base_width=int(d.get("unet_base_width", 64)),
            unet_depth=int(d.get("unet_depth", 3)),
            unet_time_embed_dim=int(d.get("unet_time_embed_dim", 128)),
            unet_dropout=float(d.get("unet_dropout", 0.0)),
            diffusion_timesteps=int(d.get("diffusion_timesteps", 1000)),
            beta_start=float(d.get("beta_start", 1e-4)),
            beta_end=float(d.get("beta_end", 2e-2)),
            weights_path=str(d.get("weights_path", "fnirs_unet.weights.h5")),
        )
