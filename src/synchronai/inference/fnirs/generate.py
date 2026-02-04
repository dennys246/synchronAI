"""
Generate synthetic fNIRS hemoglobin data from a trained diffusion model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from synchronai.models.fnirs.diffusion import FnirsDiffusionConfig, build_unet_1d, make_linear_beta_schedule
from synchronai.utils.logging import get_logger
from synchronai.utils.trace import trace
from synchronai.utils.visualization import plot_hemoglobin_signal, plot_multiple_samples


def _load_config(path: str) -> FnirsDiffusionConfig:
    with open(path, "r") as f:
        payload = json.load(f)
    return FnirsDiffusionConfig.from_dict(payload)


def generate_fnirs_diffusion(
    *,
    save_dir: str,
    n_samples: int = 1,
    config_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    seed: int = 0,
    log_every: int = 25,
    out_path: Optional[str] = None,
) -> str:
    """
    Generate `n_samples` synthetic hemoglobin windows and save as `.npz`.

    Returns the output path written.
    """
    logger = get_logger(__name__)
    trace("generate_fnirs_diffusion: start")
    save_root = Path(save_dir)
    config_path = config_path or str(save_root / "fnirs_diffusion_config.json")
    trace(f"generate_fnirs_diffusion: loading config {config_path}")
    config = _load_config(config_path)
    weights_path = weights_path or str(save_root / config.weights_path)

    trace("generate_fnirs_diffusion: building diffusion schedule + U-Net")
    schedule = make_linear_beta_schedule(
        config.diffusion_timesteps, beta_start=config.beta_start, beta_end=config.beta_end
    )
    model = build_unet_1d(
        input_length=config.model_len,
        feature_dim=config.feature_dim,
        base_width=config.unet_base_width,
        depth=config.unet_depth,
        time_embed_dim=config.unet_time_embed_dim,
        dropout=config.unet_dropout,
    )

    _ = model([tf.zeros((1, config.model_len, config.feature_dim)), tf.zeros((1,), dtype=tf.int32)])
    model.load_weights(weights_path)
    logger.info("Loaded weights: %s", weights_path)
    trace(f"generate_fnirs_diffusion: loaded weights {weights_path}")

    mean = tf.constant(np.asarray(config.feature_mean, dtype=np.float32)[None, None, :])
    std = tf.constant(np.asarray(config.feature_std, dtype=np.float32)[None, None, :])

    rng = tf.random.Generator.from_seed(int(seed))
    x = rng.normal((n_samples, config.model_len, config.feature_dim), dtype=tf.float32)

    logger.info(
        "Sampling diffusion trajectory: timesteps=%d, samples=%d",
        config.diffusion_timesteps,
        n_samples,
    )
    trace("generate_fnirs_diffusion: starting sampling loop")

    for t in reversed(range(config.diffusion_timesteps)):
        t_batch = tf.fill([n_samples], tf.cast(t, tf.int32))
        eps = model([x, t_batch], training=False)

        beta_t = schedule.betas[t]
        alpha_t = schedule.alphas[t]
        alpha_bar_t = schedule.alpha_bars[t]

        # DDPM mean prediction (x_{t-1} mean)
        x = (1.0 / tf.sqrt(alpha_t)) * (x - (beta_t / tf.sqrt(1.0 - alpha_bar_t)) * eps)

        if t > 0:
            alpha_bar_prev = schedule.alpha_bars[t - 1]
            posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            x = x + tf.sqrt(tf.maximum(posterior_var, 1e-8)) * rng.normal(tf.shape(x), dtype=tf.float32)

        if log_every and (t % log_every == 0 or t == config.diffusion_timesteps - 1):
            logger.info("t=%d", t)

    # De-standardize back to hemoglobin-ish scale and crop to requested duration.
    x = x * std + mean
    x = x[:, : config.target_len, :]

    hb_count = len(config.hb_types)
    n_pairs = len(config.pair_names)
    x_np = x.numpy().reshape(n_samples, config.target_len, n_pairs, hb_count).astype(np.float32)

    # Save to generated/ subdirectory
    generated_dir = save_root / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    out_path = out_path or str(generated_dir / f"fnirs_synthetic_{timestamp}.npz")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        data=x_np,
        sfreq_hz=np.float32(config.sfreq_hz),
        duration_seconds=np.float32(config.duration_seconds),
        pair_names=np.asarray(config.pair_names),
        hb_types=np.asarray(config.hb_types),
    )
    logger.info("Wrote synthetic data: %s", out_path)
    trace(f"generate_fnirs_diffusion: wrote output {out_path}")

    # Generate and save visualization plots
    if n_samples == 1:
        plot_path = str(generated_dir / f"fnirs_synthetic_{timestamp}_single.png")
        plot_hemoglobin_signal(
            x_np[0],
            sfreq_hz=config.sfreq_hz,
            pair_names=config.pair_names,
            hb_types=config.hb_types,
            title="Generated fNIRS Hemoglobin Signal",
            save_path=plot_path,
        )
        logger.info("Saved visualization plot: %s", plot_path)
    else:
        plot_path = str(generated_dir / f"fnirs_synthetic_{timestamp}_multiple.png")
        plot_multiple_samples(
            x_np,
            sfreq_hz=config.sfreq_hz,
            pair_names=config.pair_names,
            hb_types=config.hb_types,
            title=f"Generated fNIRS Samples (n={n_samples})",
            save_path=plot_path,
            max_samples=min(n_samples, 3),
        )
        logger.info("Saved visualization plot: %s", plot_path)

    return out_path
