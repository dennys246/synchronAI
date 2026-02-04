"""
Training loop for the fNIRS diffusion U-Net.

Baseline DDPM objective: predict the noise added at timestep t.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
import tensorflow as tf

from synchronai.data.fnirs.dataset import load_training_windows, standardize_with_stats
from synchronai.models.fnirs.diffusion import (
    FnirsDiffusionConfig,
    build_unet_1d,
    make_cosine_beta_schedule,
    make_linear_beta_schedule,
    pad_length_to_multiple,
)
from synchronai.utils.logging import get_logger
from synchronai.utils.trace import trace
from synchronai.utils.visualization import plot_hemoglobin_signal

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt


class TrainingHistory:
    """
    Track and save training metrics (loss, etc.) with plotting capabilities.
    """

    def __init__(self):
        self.batch_losses: List[float] = []
        self.epoch_losses: List[float] = []
        self.batch_indices: List[int] = []  # Global batch index
        self.epoch_indices: List[int] = []
        self.recordings_processed: List[int] = []  # Cumulative recordings processed
        self._global_batch = 0
        self._total_recordings = 0
        self.current_epoch = 0  # Track current epoch for resume support
        # Best model tracking
        self.best_loss: float = float("inf")
        self.best_epoch: int = 0
        self.last_loss: float = float("inf")

    def add_batch_loss(self, loss: float, recordings_in_batch: int = 0) -> None:
        """Record loss for a single batch."""
        self.batch_losses.append(float(loss))
        self.batch_indices.append(self._global_batch)
        self._global_batch += 1
        self._total_recordings += recordings_in_batch
        self.recordings_processed.append(self._total_recordings)

    def add_epoch_loss(self, loss: float, epoch: int) -> None:
        """Record average loss for an epoch and update best tracking."""
        loss = float(loss)
        self.epoch_losses.append(loss)
        self.epoch_indices.append(epoch)
        self.last_loss = loss
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch

    def save(self, path: str) -> None:
        """Save training history to a JSON file."""
        data = {
            "batch_losses": self.batch_losses,
            "epoch_losses": self.epoch_losses,
            "batch_indices": self.batch_indices,
            "epoch_indices": self.epoch_indices,
            "recordings_processed": self.recordings_processed,
            "global_batch": self._global_batch,
            "total_recordings": self._total_recordings,
            "current_epoch": self.current_epoch,
            "best_loss": self.best_loss if self.best_loss != float("inf") else None,
            "best_epoch": self.best_epoch,
            "last_loss": self.last_loss if self.last_loss != float("inf") else None,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingHistory":
        """Load training history from a JSON file."""
        history = cls()
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            history.batch_losses = data.get("batch_losses", [])
            history.epoch_losses = data.get("epoch_losses", [])
            history.batch_indices = data.get("batch_indices", [])
            history.epoch_indices = data.get("epoch_indices", [])
            history.recordings_processed = data.get("recordings_processed", [])
            history._global_batch = data.get("global_batch", len(history.batch_losses))
            history._total_recordings = data.get("total_recordings", 0)
            history.current_epoch = data.get("current_epoch", len(history.epoch_losses))
            # Load best/last loss tracking (with backwards compatibility)
            best_loss = data.get("best_loss")
            history.best_loss = best_loss if best_loss is not None else float("inf")
            history.best_epoch = data.get("best_epoch", 0)
            last_loss = data.get("last_loss")
            history.last_loss = last_loss if last_loss is not None else float("inf")
            # Backwards compatibility: compute from epoch_losses if not saved
            if history.best_loss == float("inf") and history.epoch_losses:
                history.best_loss = min(history.epoch_losses)
                history.best_epoch = history.epoch_indices[history.epoch_losses.index(history.best_loss)]
                history.last_loss = history.epoch_losses[-1]
        return history

    def get_summary(self) -> str:
        """Get a human-readable summary of training state."""
        if not self.epoch_losses:
            return "No training history"
        lines = [
            f"Epoch: {self.current_epoch}",
            f"Last loss: {self.last_loss:.6f}",
            f"Best loss: {self.best_loss:.6f} (epoch {self.best_epoch})",
            f"Total batches: {self._global_batch}",
            f"Total recordings: {self._total_recordings}",
        ]
        return " | ".join(lines)

    def plot(self, save_path: str, title: str = "Training Loss") -> None:
        """
        Plot training loss history and save to PNG.

        Creates a figure with:
        - Batch-level loss (light line)
        - Smoothed loss (moving average, darker line)
        - Epoch markers if available
        """
        if not self.batch_losses:
            return

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

        # Top plot: Loss vs Batch
        ax1 = axes[0]
        batches = np.array(self.batch_indices)
        losses = np.array(self.batch_losses)

        # Plot raw batch losses (semi-transparent)
        ax1.plot(batches, losses, alpha=0.3, color='blue', linewidth=0.5, label='Batch Loss')

        # Plot smoothed loss (moving average)
        if len(losses) > 10:
            window_size = min(50, len(losses) // 5)
            if window_size > 1:
                smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                smoothed_x = batches[window_size-1:]
                ax1.plot(smoothed_x, smoothed, color='blue', linewidth=1.5, label=f'Smoothed (window={window_size})')

        # Mark epoch boundaries if we have epoch data
        if self.epoch_indices and len(self.epoch_indices) > 1:
            for i, epoch in enumerate(self.epoch_indices[:-1]):
                # Find approximate batch index for this epoch
                if i < len(self.batch_indices):
                    ax1.axvline(x=self.batch_indices[min(i * (len(self.batch_indices) // len(self.epoch_indices)), len(self.batch_indices)-1)],
                               color='red', alpha=0.3, linestyle='--', linewidth=0.5)

        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title(f'{title} - Batch Loss')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale often better for loss

        # Bottom plot: Loss vs Recordings Processed
        ax2 = axes[1]
        if self.recordings_processed and len(self.recordings_processed) == len(losses):
            recordings = np.array(self.recordings_processed)
            ax2.plot(recordings, losses, alpha=0.3, color='green', linewidth=0.5, label='Batch Loss')

            # Smoothed version
            if len(losses) > 10:
                window_size = min(50, len(losses) // 5)
                if window_size > 1:
                    smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
                    smoothed_recordings = recordings[window_size-1:]
                    ax2.plot(smoothed_recordings, smoothed, color='green', linewidth=1.5, label=f'Smoothed (window={window_size})')

            ax2.set_xlabel('Recordings Processed')
            ax2.set_ylabel('Loss (MSE)')
            ax2.set_title(f'{title} - Loss vs Data Seen')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        else:
            # Fallback: just show epoch losses if available
            if self.epoch_losses:
                ax2.plot(self.epoch_indices, self.epoch_losses, 'o-', color='purple', linewidth=2, markersize=6)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss (MSE)')
                ax2.set_title(f'{title} - Epoch Loss')
                ax2.grid(True, alpha=0.3)
                ax2.set_yscale('log')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def __repr__(self) -> str:
        return f"TrainingHistory(batches={len(self.batch_losses)}, epochs={len(self.epoch_losses)})"


class RunningStats:
    """
    Welford's online algorithm for computing running mean and variance.

    This allows accurate computation of mean/std across all batches without
    needing to load all data into memory at once. Supports save/load for
    checkpointing across training runs.
    """

    def __init__(self, feature_dim: int):
        self.feature_dim = feature_dim
        self.n = 0  # Total number of samples seen
        self.mean = np.zeros(feature_dim, dtype=np.float64)
        self.M2 = np.zeros(feature_dim, dtype=np.float64)  # Sum of squared differences

    def update(self, batch: np.ndarray) -> None:
        """
        Update running statistics with a new batch of data.

        Args:
            batch: Array of shape (n_windows, time, features) or (n_samples, features)
        """
        # Flatten to (n_samples, features) if needed
        if batch.ndim == 3:
            batch = batch.reshape(-1, batch.shape[-1])

        for x in batch:
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2

    def update_batch(self, batch: np.ndarray) -> None:
        """
        Batch update for efficiency - processes entire batch at once.

        Args:
            batch: Array of shape (n_windows, time, features) or (n_samples, features)
        """
        # Flatten to (n_samples, features) if needed
        if batch.ndim == 3:
            batch = batch.reshape(-1, batch.shape[-1])

        batch_n = batch.shape[0]
        if batch_n == 0:
            return

        batch_mean = np.mean(batch, axis=0, dtype=np.float64)
        batch_var = np.var(batch, axis=0, dtype=np.float64)
        batch_M2 = batch_var * batch_n

        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.M2 = batch_M2
        else:
            # Parallel algorithm for combining statistics
            total_n = self.n + batch_n
            delta = batch_mean - self.mean
            self.mean = (self.n * self.mean + batch_n * batch_mean) / total_n
            self.M2 = self.M2 + batch_M2 + delta**2 * self.n * batch_n / total_n
            self.n = total_n

    @property
    def variance(self) -> np.ndarray:
        """Return the current variance estimate."""
        if self.n < 2:
            return np.ones(self.feature_dim, dtype=np.float32)
        return (self.M2 / self.n).astype(np.float32)

    @property
    def std(self) -> np.ndarray:
        """Return the current standard deviation estimate."""
        return np.sqrt(np.maximum(self.variance, 1e-6)).astype(np.float32)

    def get_mean(self) -> np.ndarray:
        """Return the current mean estimate."""
        return self.mean.astype(np.float32)

    def save(self, path: str) -> None:
        """Save running statistics to a .npz file."""
        np.savez(
            path,
            feature_dim=self.feature_dim,
            n=self.n,
            mean=self.mean,
            M2=self.M2,
        )

    @classmethod
    def load(cls, path: str) -> "RunningStats":
        """Load running statistics from a .npz file."""
        data = np.load(path)
        stats = cls(int(data["feature_dim"]))
        stats.n = int(data["n"])
        stats.mean = data["mean"].astype(np.float64)
        stats.M2 = data["M2"].astype(np.float64)
        return stats

    def __repr__(self) -> str:
        return f"RunningStats(n={self.n}, feature_dim={self.feature_dim})"


class LazyFnirsDiscovery:
    """
    Lazily discover fNIRS recordings incrementally to speed up training startup.
    Cache discovered paths across epochs.

    Supports filtering by signal type:
    - 'hemodynamic': Raw hemoglobin signals (HbO/HbR) - default
    - 'neural': Deconvolved neural activity estimates from HRfunc
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 4,
        signal_type: str = "hemodynamic"
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.signal_type = signal_type
        self.cached_paths: List[str] = []
        self.discovery_complete = False
        self.logger = get_logger(__name__)
        self._generator = None
        self._data_dirs = []

        if signal_type not in ("hemodynamic", "neural"):
            raise ValueError(f"signal_type must be 'hemodynamic' or 'neural', got: {signal_type}")

        # Parse colon-separated directories
        data_dirs = data_dir.split(":")
        for dir_path in data_dirs:
            dir_path = dir_path.strip()
            if dir_path:
                self._data_dirs.append(Path(dir_path))

        # Validate directories exist
        for dir_path in self._data_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(f"data_dir not found: {dir_path}")

        self.logger.info(f"Discovering {signal_type} fNIRS recordings...")

        # Start by discovering first batch eagerly for immediate training
        self._discover_next_batch()

    def _iter_fnirs_paths(self) -> Iterator[str]:
        """Generator that yields fNIRS paths one at a time."""
        for root in self._data_dirs:
            if root.is_file():
                yield str(root)
                continue

            # Check if root itself is a recording
            if (root / "probeInfo.mat").exists():
                yield str(root)
                continue

            # Walk directory tree incrementally
            for dirpath, dirnames, filenames in os.walk(root):
                dirpath_obj = Path(dirpath)

                # Check for probeInfo.mat (NIRx format)
                if "probeInfo.mat" in filenames:
                    yield str(dirpath_obj)

                # Check for .snirf and .fif files (only if they actually exist)
                for filename in filenames:
                    if filename.endswith(".snirf") or filename.endswith(".fif"):
                        file_path = dirpath_obj / filename

                        # Filter by signal type
                        is_deconvolved = "_Deconvolved" in filename or "deconvolved" in filename.lower()

                        if self.signal_type == "hemodynamic" and is_deconvolved:
                            # Skip neural activity files when training hemodynamic model
                            continue
                        elif self.signal_type == "neural" and not is_deconvolved:
                            # Skip raw hemodynamic files when training neural model
                            continue

                        # Verify file actually exists (could be broken symlink)
                        if file_path.exists() and file_path.is_file():
                            yield str(file_path)

    def _discover_next_batch(self) -> None:
        """Discover the next batch of recordings."""
        if self.discovery_complete:
            return

        # Initialize generator if not already done
        if self._generator is None:
            self._generator = self._iter_fnirs_paths()

        # Discover batch_size * 10 recordings at a time
        batch_to_add = []
        target_batch = self.batch_size * 10

        try:
            for _ in range(target_batch):
                path = next(self._generator)
                batch_to_add.append(path)
        except StopIteration:
            self.discovery_complete = True

        if batch_to_add:
            # Remove duplicates and sort
            unique_batch = sorted(set(batch_to_add) - set(self.cached_paths))
            self.cached_paths.extend(unique_batch)

            self.logger.info(
                f"Discovered {len(unique_batch)} recordings (batch discovery), "
                f"total cached: {len(self.cached_paths)}"
            )

        if self.discovery_complete:
            self.logger.info(f"Discovery complete: {len(self.cached_paths)} total recordings")

    def get_paths(self, max_recordings: Optional[int] = None) -> List[str]:
        """Get all discovered paths, discovering more if needed."""
        # Keep discovering until we have enough or discovery is complete
        while not self.discovery_complete:
            if max_recordings is not None and len(self.cached_paths) >= max_recordings:
                break
            self._discover_next_batch()

        if max_recordings is None:
            return self.cached_paths
        return self.cached_paths[:max_recordings]

    def __len__(self) -> int:
        """Return number of currently discovered recordings."""
        return len(self.cached_paths)


def _pad_time(windows: np.ndarray, target_len: int) -> np.ndarray:
    if windows.shape[1] == target_len:
        return windows
    pad = target_len - windows.shape[1]
    if pad < 0:
        return windows[:, :target_len, :]
    return np.pad(windows, ((0, 0), (0, pad), (0, 0)), mode="constant", constant_values=0.0)


def _save_json(path: str, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _generate_sample(
    model: tf.keras.Model,
    schedule,
    config: FnirsDiffusionConfig,
    n_samples: int = 1,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate synthetic hemoglobin samples using the trained diffusion model.

    Returns: Array of shape (n_samples, target_len, n_pairs, n_hb)
    """
    mean = tf.constant(np.asarray(config.feature_mean, dtype=np.float32)[None, None, :])
    std = tf.constant(np.asarray(config.feature_std, dtype=np.float32)[None, None, :])

    rng = tf.random.Generator.from_seed(int(seed))
    x = rng.normal((n_samples, config.model_len, config.feature_dim), dtype=tf.float32)

    # DDPM sampling loop - use schedule.timesteps to ensure consistency
    for t in reversed(range(schedule.timesteps)):
        t_batch = tf.fill([n_samples], tf.cast(t, tf.int32))
        eps = model([x, t_batch], training=False)

        beta_t = schedule.betas[t]
        alpha_t = schedule.alphas[t]
        alpha_bar_t = schedule.alpha_bars[t]

        x = (1.0 / tf.sqrt(alpha_t)) * (x - (beta_t / tf.sqrt(1.0 - alpha_bar_t)) * eps)

        if t > 0:
            alpha_bar_prev = schedule.alpha_bars[t - 1]
            posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            x = x + tf.sqrt(tf.maximum(posterior_var, 1e-8)) * rng.normal(tf.shape(x), dtype=tf.float32)

    # De-standardize and reshape
    x = x * std + mean
    x = x[:, : config.target_len, :]

    hb_count = len(config.hb_types)
    n_pairs = len(config.pair_names)
    x_np = x.numpy().reshape(n_samples, config.target_len, n_pairs, hb_count).astype(np.float32)

    return x_np


def train_fnirs_diffusion(
    *,
    data_dir: str,
    save_dir: str,
    duration_seconds: float = 120.0,
    target_sfreq_hz: Optional[float] = None,
    segments_per_recording: int = 8,
    max_recordings: Optional[int] = None,
    recordings_per_batch: int = 4,
    diffusion_timesteps: int = 1000,
    beta_schedule: str = "cosine",
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    unet_base_width: int = 64,
    unet_depth: int = 3,
    unet_time_embed_dim: int = 128,
    unet_dropout: float = 0.0,
    batch_size: int = 8,
    epochs: int = 10,
    learning_rate: float = 1e-4,
    seed: int = 0,
    deconvolution: bool = False,
    signal_type: str = "hemodynamic",
    config_filename: str = "fnirs_diffusion_config.json",
) -> FnirsDiffusionConfig:
    """
    Train a diffusion model on fNIRS windows and save weights + config.

    Args:
        data_dir: Directory or colon-separated directories to discover recordings from
        max_recordings: Total number of recordings to process (None for unlimited)
        recordings_per_batch: Number of recordings to load into memory at once
        signal_type: Type of signal to train on - 'hemodynamic' (HbO/HbR) or 'neural' (deconvolved)
    """
    logger = get_logger(__name__)
    trace("train_fnirs_diffusion: start")
    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    # Initialize lazy discovery
    discovery = LazyFnirsDiscovery(
        data_dir,
        batch_size=recordings_per_batch,
        signal_type=signal_type
    )

    # Get initial batch for immediate training start
    initial_paths = discovery.get_paths(max_recordings=recordings_per_batch)

    # Determine total_recordings:
    # - If max_recordings specified, that's our target
    # - If unlimited (None), we'll discover as we go and update dynamically
    if max_recordings is not None:
        total_recordings = max_recordings
        logger.info(
            "Training with batch loading: target_recordings=%d, recordings_per_batch=%d (load at once), epochs=%d",
            total_recordings,
            recordings_per_batch,
            epochs,
        )
    else:
        # Unlimited mode: keep discovering during training
        # We'll update total_recordings as we discover more
        total_recordings = len(initial_paths)  # Start with what we have
        logger.info(
            "Training with dynamic discovery: starting with %d recordings, recordings_per_batch=%d (load at once), epochs=%d",
            total_recordings,
            recordings_per_batch,
            epochs,
        )
        logger.info("Will continue discovering and training on new recordings as found...")

    # Load first batch to initialize config and model
    trace("train_fnirs_diffusion: loading initial training windows")
    initial_batch_size = min(recordings_per_batch, total_recordings)
    logger.info("Loading initial batch of %d recordings for config setup...", initial_batch_size)

    # Use the already-discovered initial_paths (don't trigger more discovery yet)
    # Load without normalization first so we can use running stats
    training = load_training_windows(
        initial_paths[:initial_batch_size],  # Use the already-discovered initial paths
        duration_seconds=duration_seconds,
        target_sfreq_hz=target_sfreq_hz,
        segments_per_recording=segments_per_recording,
        seed=seed,
        deconvolution=deconvolution,
        max_recordings=None,  # Load all paths in the slice (which is just the first batch)
        normalize=False,  # Don't normalize yet - we'll use running stats
    )
    trace("train_fnirs_diffusion: loaded initial training windows (raw)")

    target_len = int(training.windows.shape[1])
    feature_dim = int(training.windows.shape[2])
    model_len = pad_length_to_multiple(target_len, 2**unet_depth)

    if model_len != target_len:
        logger.info(
            "Padded time axis for U-Net compatibility: target_len=%d -> model_len=%d",
            target_len,
            model_len,
        )

    weights_path = str(save_root / "fnirs_unet.weights.h5")
    config_path = str(save_root / config_filename)
    running_stats_path = str(save_root / "running_stats.npz")
    history_path = str(save_root / "training_history.json")
    loss_plot_path = str(save_root / "training_loss.png")

    # Initialize or load training history
    history = TrainingHistory.load(history_path)
    start_epoch = history.current_epoch
    if history.batch_losses:
        logger.info("Resuming training from checkpoint:")
        logger.info("  %s", history.get_summary())
    else:
        logger.info("Starting fresh training history")

    # Initialize or load running statistics for normalization
    if os.path.exists(running_stats_path):
        running_stats = RunningStats.load(running_stats_path)
        logger.info("Loaded existing running stats: %s (n=%d samples)", running_stats_path, running_stats.n)
    else:
        running_stats = RunningStats(feature_dim)
        logger.info("Initialized new running stats for %d features", feature_dim)

    # Update running stats with initial batch (using raw windows)
    running_stats.update_batch(training.windows)
    running_stats.save(running_stats_path)
    logger.info("Updated running stats with initial batch: n=%d samples, saving to %s", running_stats.n, running_stats_path)

    # Now normalize the initial windows using running stats
    windows_normalized = standardize_with_stats(
        training.windows,
        running_stats.get_mean(),
        running_stats.std
    )
    windows = _pad_time(windows_normalized, model_len)

    config = FnirsDiffusionConfig(
        duration_seconds=float(training.duration_seconds),
        sfreq_hz=float(training.sfreq_hz),
        target_len=target_len,
        model_len=model_len,
        pair_names=training.pair_names,
        hb_types=training.hb_types,
        feature_dim=feature_dim,
        feature_mean=running_stats.get_mean().tolist(),
        feature_std=running_stats.std.tolist(),
        unet_base_width=unet_base_width,
        unet_depth=unet_depth,
        unet_time_embed_dim=unet_time_embed_dim,
        unet_dropout=unet_dropout,
        diffusion_timesteps=diffusion_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        weights_path=os.path.basename(weights_path),
    )

    _save_json(config_path, config.to_dict())
    logger.info("Wrote config: %s", config_path)

    trace("train_fnirs_diffusion: building diffusion schedule + U-Net")
    if beta_schedule == "cosine":
        schedule = make_cosine_beta_schedule(diffusion_timesteps)
        logger.info(f"Using cosine beta schedule with {diffusion_timesteps} timesteps")
    elif beta_schedule == "linear":
        schedule = make_linear_beta_schedule(
            diffusion_timesteps, beta_start=beta_start, beta_end=beta_end
        )
        logger.info(f"Using linear beta schedule with {diffusion_timesteps} timesteps (beta: {beta_start} → {beta_end})")
    else:
        raise ValueError(f"Unknown beta_schedule: {beta_schedule}")
    model = build_unet_1d(
        input_length=model_len,
        feature_dim=feature_dim,
        base_width=unet_base_width,
        depth=unet_depth,
        time_embed_dim=unet_time_embed_dim,
        dropout=unet_dropout,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Build once so model variables exist before loading/saving.
    _ = model([tf.zeros((1, model_len, feature_dim)), tf.zeros((1,), dtype=tf.int32)])

    # Load existing weights if resuming training
    if start_epoch > 0 and os.path.exists(weights_path):
        model.load_weights(weights_path)
        logger.info("Loaded existing model weights from %s (resuming from epoch %d)", weights_path, start_epoch)

    trace("train_fnirs_diffusion: starting training loop")
    @tf.function
    def train_step(x0: tf.Tensor) -> tf.Tensor:
        batch = tf.shape(x0)[0]
        t = tf.random.uniform((batch,), 0, schedule.timesteps, dtype=tf.int32)
        noise = tf.random.normal(tf.shape(x0))

        sqrt_ab = tf.gather(schedule.sqrt_alpha_bars, t)[:, None, None]
        sqrt_1mab = tf.gather(schedule.sqrt_one_minus_alpha_bars, t)[:, None, None]
        x_t = sqrt_ab * x0 + sqrt_1mab * noise

        with tf.GradientTape() as tape:
            pred = model([x_t, t], training=True)
            loss = tf.reduce_mean(tf.square(noise - pred))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    # Create generated samples directory
    generated_dir = save_root / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)

    # Handle unlimited epochs (epochs=0 means run forever)
    if epochs == 0:
        logger.info(
            "Training diffusion U-Net: batch_size=%d, epochs=unlimited (∞), starting from epoch %d",
            batch_size,
            start_epoch + 1,
        )
        save_sample_every = 1  # Save every epoch when unlimited
        epochs_display = "∞"
        remaining_epochs = None  # Unlimited
    else:
        remaining_epochs = epochs - start_epoch
        if remaining_epochs <= 0:
            logger.info("Training already complete (epoch %d/%d), nothing to do", start_epoch, epochs)
            return config
        logger.info(
            "Training diffusion U-Net: batch_size=%d, epochs=%d (resuming from %d, %d remaining)",
            batch_size,
            epochs,
            start_epoch,
            remaining_epochs,
        )
        save_sample_every = max(1, epochs // 10)  # Save samples ~10 times during training (every epoch for 10 epochs)
        epochs_display = str(epochs)

    # Training loop with batch loading
    epoch = start_epoch
    epoch_count = 0
    while remaining_epochs is None or epoch_count < remaining_epochs:
        epoch += 1
        epoch_count += 1
        loss_metric = tf.keras.metrics.Mean()

        # Start with already discovered paths (don't trigger full discovery)
        all_paths = discovery.get_paths(max_recordings=len(discovery.cached_paths) if max_recordings is None else max_recordings)

        # Cycle through all recordings in batches
        recording_idx = 0
        batch_count = 0

        while recording_idx < len(all_paths):
            # Determine batch of recordings to load
            batch_end = min(recording_idx + recordings_per_batch, len(all_paths))

            # Trigger discovery of more paths ahead of time if we're getting close to the end
            if not discovery.discovery_complete and recording_idx + recordings_per_batch * 2 >= len(all_paths):
                # We're getting close to the end, try to discover more
                # Only discover a bit more (current + 10 batches)
                target_count = len(all_paths) + (recordings_per_batch * 10)
                all_paths = discovery.get_paths(max_recordings=target_count if max_recordings is None else max_recordings)
                if max_recordings is None:
                    total_recordings = len(all_paths)

            batch_paths = all_paths[recording_idx:batch_end]

            logger.info(
                "Epoch %d/%s - Loading recordings %d-%d/%d...",
                epoch,
                epochs_display,
                recording_idx + 1,
                batch_end,
                total_recordings,
            )

            # Load this batch of recordings
            try:
                # Load raw (unnormalized) data first
                batch_training = load_training_windows(
                    batch_paths,
                    duration_seconds=duration_seconds,
                    target_sfreq_hz=target_sfreq_hz,
                    segments_per_recording=segments_per_recording,
                    seed=seed + epoch + recording_idx,
                    deconvolution=deconvolution,
                    max_recordings=None,  # Load all in this batch
                    normalize=False,  # Don't normalize - we'll use running stats
                )

                # Update running statistics with this batch
                running_stats.update_batch(batch_training.windows)

                # Normalize using current running stats
                batch_windows_normalized = standardize_with_stats(
                    batch_training.windows,
                    running_stats.get_mean(),
                    running_stats.std
                )
                batch_windows = _pad_time(batch_windows_normalized, model_len)

                # Create dataset for this batch
                dataset = tf.data.Dataset.from_tensor_slices(batch_windows)
                dataset = dataset.shuffle(
                    buffer_size=min(len(batch_windows), 4096),
                    seed=seed + epoch + recording_idx,
                    reshuffle_each_iteration=True
                )
                dataset = dataset.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

                # Train on this batch
                for batch_data in dataset:
                    loss = train_step(batch_data)
                    loss_metric.update_state(loss)
                    batch_count += 1

                logger.info(
                    "Epoch %d/%s - Completed recordings %d-%d/%d - running loss=%.6f",
                    epoch,
                    epochs_display,
                    recording_idx + 1,
                    batch_end,
                    total_recordings,
                    loss_metric.result(),
                )

                # Save model weights after each batch
                model.save_weights(weights_path)
                logger.info("Saved weights after batch: %s", weights_path)

                # Save running stats and update config with latest normalization parameters
                running_stats.save(running_stats_path)
                config = FnirsDiffusionConfig(
                    duration_seconds=config.duration_seconds,
                    sfreq_hz=config.sfreq_hz,
                    target_len=config.target_len,
                    model_len=config.model_len,
                    pair_names=config.pair_names,
                    hb_types=config.hb_types,
                    feature_dim=config.feature_dim,
                    feature_mean=running_stats.get_mean().tolist(),
                    feature_std=running_stats.std.tolist(),
                    unet_base_width=config.unet_base_width,
                    unet_depth=config.unet_depth,
                    unet_time_embed_dim=config.unet_time_embed_dim,
                    unet_dropout=config.unet_dropout,
                    diffusion_timesteps=config.diffusion_timesteps,
                    beta_start=config.beta_start,
                    beta_end=config.beta_end,
                    weights_path=config.weights_path,
                )
                _save_json(config_path, config.to_dict())
                logger.info("Saved running stats (n=%d) and updated config", running_stats.n)

                # Record loss in training history
                history.add_batch_loss(float(loss_metric.result()), recordings_in_batch=batch_end - recording_idx)
                history.save(history_path)
                history.plot(loss_plot_path, title="fNIRS Diffusion Training")
                logger.info("Updated training history and loss plot")

                # Generate and save a checkpoint sample after each batch for progress monitoring
                try:
                    sample = _generate_sample(model, schedule, config, n_samples=1, seed=seed + epoch * 10000 + recording_idx)
                    plot_path = str(generated_dir / f"epoch_{epoch:04d}_batch_{recording_idx:05d}_sample.png")
                    plot_hemoglobin_signal(
                        sample[0],
                        sfreq_hz=config.sfreq_hz,
                        pair_names=config.pair_names,
                        hb_types=config.hb_types,
                        title=f"Generated fNIRS (Epoch {epoch}/{epochs_display}, Rec {recording_idx+1}-{batch_end}/{total_recordings})",
                        save_path=plot_path,
                    )
                    logger.info("Saved checkpoint sample: %s", plot_path)
                except Exception as sample_err:
                    logger.warning(f"Failed to generate checkpoint sample: {sample_err}")

            except Exception as e:
                logger.warning(f"Error loading recordings {recording_idx}-{batch_end}: {e}")

            recording_idx = batch_end

        model.save_weights(weights_path)
        epoch_loss = float(loss_metric.result())
        is_best = epoch_loss < history.best_loss
        logger.info("Epoch %d/%s - loss=%.6f%s - saved weights: %s",
                    epoch, epochs_display, epoch_loss,
                    " (NEW BEST)" if is_best else f" (best: {history.best_loss:.6f} @ epoch {history.best_epoch})",
                    weights_path)

        # Record epoch loss in training history and update current epoch for resume support
        history.add_epoch_loss(epoch_loss, epoch)
        history.current_epoch = epoch
        history.save(history_path)
        history.plot(loss_plot_path, title="fNIRS Diffusion Training")

        # Generate and save sample every N epochs
        if epoch % save_sample_every == 0 or (epochs > 0 and epoch == epochs):
            try:
                logger.info("Generating sample at epoch %d...", epoch)
                sample = _generate_sample(model, schedule, config, n_samples=1, seed=seed + epoch)

                # Save plot
                plot_path = str(generated_dir / f"epoch_{epoch:04d}_sample.png")
                plot_hemoglobin_signal(
                    sample[0],  # First sample
                    sfreq_hz=config.sfreq_hz,
                    pair_names=config.pair_names,
                    hb_types=config.hb_types,
                    title=f"Generated fNIRS Signal (Epoch {epoch}/{epochs_display})",
                    save_path=plot_path,
                )
                logger.info("Saved generated sample plot: %s", plot_path)

                # Save raw data
                npz_path = str(generated_dir / f"epoch_{epoch:04d}_sample.npz")
                np.savez_compressed(
                    npz_path,
                    data=sample,
                    sfreq_hz=np.float32(config.sfreq_hz),
                    duration_seconds=np.float32(config.duration_seconds),
                    pair_names=np.asarray(config.pair_names),
                    hb_types=np.asarray(config.hb_types),
                )
                logger.info("Saved generated sample data: %s", npz_path)
            except Exception as e:
                logger.error(f"Failed to generate sample at epoch {epoch}: {e}", exc_info=True)

    return config
