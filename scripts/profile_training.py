#!/usr/bin/env python3
"""
Training pipeline profiler for the synchronAI project.

Profiles data loading, forward pass, backward pass, optimizer step, and GPU memory
to identify bottlenecks in the video classifier training pipeline. Also tests
different num_workers settings to find the optimal data loading configuration.

Usage:
    python scripts/profile_training.py \
        --config configs/train/video_classifier.yaml \
        --labels-file scripts/data/labels.csv \
        --n-batches 20 \
        --test-workers 0,2,4,8

Output files (in --output-dir):
    timing_breakdown.json        Per-component timing stats
    num_workers_comparison.json  Data loading times per num_workers setting
    recommendations.txt          Prioritized optimization suggestions
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Attempt to import project-level dependencies. If the project isn't
# installed we still want the script to report a clear error.
# ---------------------------------------------------------------------------

_MISSING_DEPS: list[str] = []

try:
    import numpy as np
except ImportError:
    _MISSING_DEPS.append("numpy")

try:
    import torch
    import torch.nn as nn
    from torch.amp import GradScaler, autocast
    from torch.optim import AdamW
except ImportError:
    _MISSING_DEPS.append("torch")

try:
    import yaml
except ImportError:
    _MISSING_DEPS.append("pyyaml")

try:
    from synchronai.data.video.dataset import (
        VideoDatasetConfig,
        VideoWindowDataset,
        load_video_index,
        split_by_video,
    )
    from synchronai.models.cv.YOLO_classifier import (
        VideoClassifierConfig,
        build_video_classifier,
    )
    from synchronai.utils.reproducibility import set_seed, worker_init_fn
except ImportError as exc:
    _MISSING_DEPS.append(f"synchronai ({exc})")


logger = logging.getLogger("profile_training")


# ============================================================================
# Dataclasses for collected metrics
# ============================================================================


@dataclass
class TimingSample:
    """Raw timing values for a single profiled batch."""

    data_load_s: float = 0.0
    forward_s: float = 0.0
    backward_s: float = 0.0
    optimizer_step_s: float = 0.0
    end_to_end_s: float = 0.0


@dataclass
class TimingStats:
    """Aggregated statistics for a series of timing samples."""

    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    median: float = 0.0
    n_samples: int = 0

    @classmethod
    def from_values(cls, values: list[float]) -> "TimingStats":
        if not values:
            return cls()
        return cls(
            mean=statistics.mean(values),
            std=statistics.stdev(values) if len(values) > 1 else 0.0,
            min=min(values),
            max=max(values),
            median=statistics.median(values),
            n_samples=len(values),
        )


@dataclass
class ProfilingResult:
    """Complete profiling result container."""

    # Per-component timing stats (seconds)
    data_loading: Optional[TimingStats] = None
    forward_pass: Optional[TimingStats] = None
    backward_pass: Optional[TimingStats] = None
    optimizer_step: Optional[TimingStats] = None
    end_to_end: Optional[TimingStats] = None

    # GPU memory (bytes)
    gpu_peak_memory_bytes: int = 0
    gpu_allocated_memory_bytes: int = 0
    gpu_reserved_memory_bytes: int = 0

    # num_workers comparison: {num_workers -> TimingStats}
    workers_comparison: dict[int, TimingStats] = field(default_factory=dict)

    # Meta
    device: str = "cpu"
    batch_size: int = 0
    n_batches_profiled: int = 0
    n_dataset_samples: int = 0
    model_param_count: int = 0
    model_trainable_param_count: int = 0
    config_path: str = ""
    skipped_batches: int = 0
    errors: list[str] = field(default_factory=list)


# ============================================================================
# Helpers
# ============================================================================


def _format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1.0:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def _format_bytes(n_bytes: int) -> str:
    """Format bytes into a human-readable string."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    if n_bytes < 1024**2:
        return f"{n_bytes / 1024:.1f} KB"
    if n_bytes < 1024**3:
        return f"{n_bytes / 1024**2:.1f} MB"
    return f"{n_bytes / 1024**3:.2f} GB"


def _stats_to_dict(stats: Optional[TimingStats]) -> Optional[dict[str, Any]]:
    """Convert TimingStats to a JSON-serialisable dict with readable time strings."""
    if stats is None:
        return None
    return {
        "mean_s": round(stats.mean, 6),
        "std_s": round(stats.std, 6),
        "min_s": round(stats.min, 6),
        "max_s": round(stats.max, 6),
        "median_s": round(stats.median, 6),
        "n_samples": stats.n_samples,
        "mean_readable": _format_time(stats.mean),
    }


def load_yaml_config(config_path: str) -> dict[str, Any]:
    """Load a YAML training config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_configs_from_yaml(
    config_dict: dict[str, Any],
    labels_file: str,
) -> tuple[VideoClassifierConfig, VideoDatasetConfig, dict[str, Any]]:
    """Parse the YAML config dict into dataclass configs.

    Returns:
        (model_config, data_config, training_dict)
    """
    model_section = config_dict.get("model", {})
    data_section = config_dict.get("data", {})
    training_section = config_dict.get("training", {})

    # Build VideoClassifierConfig from model + data sections
    model_config = VideoClassifierConfig(
        backbone=model_section.get("backbone", "yolo26s"),
        backbone_task=model_section.get("backbone_task", "detect"),
        backbone_weights=model_section.get("backbone_weights"),
        temporal_aggregation=model_section.get("temporal_aggregation", "lstm"),
        hidden_dim=model_section.get("hidden_dim", 256),
        dropout=model_section.get("dropout", 0.3),
        freeze_backbone=model_section.get("freeze_backbone", True),
        window_seconds=data_section.get("window_seconds", 2.0),
        sample_fps=data_section.get("sample_fps", 12.0),
        frame_height=data_section.get("frame_size", 640),
        frame_width=data_section.get("frame_size", 640),
    )

    data_config = VideoDatasetConfig(
        labels_file=labels_file,
        sample_fps=data_section.get("sample_fps", 12.0),
        window_seconds=data_section.get("window_seconds", 2.0),
        frame_size=data_section.get("frame_size", 640),
        augment=False,  # No augmentation during profiling
    )

    return model_config, data_config, training_section


# ============================================================================
# Core profiling functions
# ============================================================================


def profile_data_loading(
    specs: list,
    data_config: VideoDatasetConfig,
    batch_size: int,
    num_workers: int,
    n_batches: int,
    seed: int = 42,
) -> tuple[TimingStats, int]:
    """Profile data loading time for a given num_workers setting.

    Returns:
        (timing_stats, n_skipped_batches)
    """
    dataset = VideoWindowDataset(specs, data_config, augment=False)

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": worker_init_fn,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs["generator"] = generator

    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

    times: list[float] = []
    skipped = 0

    try:
        batch_iter = iter(loader)
        for i in range(n_batches):
            try:
                t0 = time.perf_counter()
                _batch = next(batch_iter)
                t1 = time.perf_counter()
                times.append(t1 - t0)
            except StopIteration:
                logger.info(
                    f"  Dataset exhausted after {i} batches "
                    f"(num_workers={num_workers}); stopping early."
                )
                break
            except Exception as exc:
                logger.warning(f"  Batch {i} failed (num_workers={num_workers}): {exc}")
                skipped += 1
                continue
    finally:
        dataset.close()

    return TimingStats.from_values(times), skipped


def profile_training_step(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    gradient_clip_max_norm: float,
    n_batches: int,
) -> tuple[list[TimingSample], int, int]:
    """Profile a full training step (data load + forward + backward + optimizer).

    Returns:
        (list_of_timing_samples, gpu_peak_memory_bytes, n_skipped)
    """
    model.train()

    # Criterion
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer — only track params that require grad
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(
            "No trainable parameters found. Set freeze_backbone=False in the "
            "config or ensure the model has unfrozen layers."
        )
    optimizer = AdamW(trainable_params, lr=1e-4, weight_decay=1e-5)

    scaler: Optional[GradScaler] = None
    if use_amp and device.type == "cuda":
        scaler = GradScaler("cuda")

    # Reset peak memory tracking
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    samples: list[TimingSample] = []
    skipped = 0

    batch_iter = iter(loader)
    for i in range(n_batches):
        sample = TimingSample()

        # ---- Data loading ----
        try:
            t0 = time.perf_counter()
            batch = next(batch_iter)
            t1 = time.perf_counter()
            sample.data_load_s = t1 - t0
        except StopIteration:
            logger.info(f"  Dataset exhausted after {i} batches; stopping early.")
            break
        except Exception as exc:
            logger.warning(f"  Batch {i} data load failed: {exc}")
            skipped += 1
            continue

        frames = batch["frames"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # ---- Forward pass ----
        try:
            optimizer.zero_grad(set_to_none=True)

            t_fwd_start = time.perf_counter()
            with autocast("cuda", enabled=(use_amp and device.type == "cuda")):
                logits = model(frames)
                logits = logits.squeeze(-1) if logits.dim() > 1 else logits
                loss = criterion(logits, labels)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_fwd_end = time.perf_counter()
            sample.forward_s = t_fwd_end - t_fwd_start
        except Exception as exc:
            logger.warning(f"  Batch {i} forward pass failed: {exc}")
            skipped += 1
            continue

        # ---- Backward pass ----
        try:
            t_bwd_start = time.perf_counter()
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_bwd_end = time.perf_counter()
            sample.backward_s = t_bwd_end - t_bwd_start
        except Exception as exc:
            logger.warning(f"  Batch {i} backward pass failed: {exc}")
            skipped += 1
            continue

        # ---- Optimizer step ----
        try:
            t_opt_start = time.perf_counter()
            if scaler is not None:
                scaler.unscale_(optimizer)
                if gradient_clip_max_norm > 0:
                    nn.utils.clip_grad_norm_(trainable_params, gradient_clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if gradient_clip_max_norm > 0:
                    nn.utils.clip_grad_norm_(trainable_params, gradient_clip_max_norm)
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_opt_end = time.perf_counter()
            sample.optimizer_step_s = t_opt_end - t_opt_start
        except Exception as exc:
            logger.warning(f"  Batch {i} optimizer step failed: {exc}")
            skipped += 1
            continue

        sample.end_to_end_s = (
            sample.data_load_s
            + sample.forward_s
            + sample.backward_s
            + sample.optimizer_step_s
        )
        samples.append(sample)

    # Peak GPU memory
    gpu_peak = 0
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        gpu_peak = torch.cuda.max_memory_allocated(device)

    return samples, gpu_peak, skipped


# ============================================================================
# Report generation
# ============================================================================


def generate_recommendations(result: ProfilingResult) -> list[str]:
    """Generate prioritised optimisation recommendations from profiling results."""

    recs: list[str] = []

    # ---- Identify the dominant bottleneck ----
    components = {
        "data_loading": result.data_loading,
        "forward_pass": result.forward_pass,
        "backward_pass": result.backward_pass,
        "optimizer_step": result.optimizer_step,
    }

    total_mean = result.end_to_end.mean if result.end_to_end else 0.0

    if total_mean > 0:
        fracs: dict[str, float] = {}
        for name, stats in components.items():
            if stats is not None and stats.mean > 0:
                fracs[name] = stats.mean / total_mean
            else:
                fracs[name] = 0.0

        ranked = sorted(fracs.items(), key=lambda x: x[1], reverse=True)
        top_name, top_frac = ranked[0]

        recs.append(
            f"1. BOTTLENECK: {top_name.replace('_', ' ').title()} accounts for "
            f"{top_frac:.0%} of each training step ({_format_time(components[top_name].mean)} / "
            f"{_format_time(total_mean)})."
        )

    # ---- num_workers recommendations ----
    if result.workers_comparison:
        baseline = result.workers_comparison.get(0)
        if baseline and baseline.mean > 0:
            best_nw = 0
            best_time = baseline.mean
            for nw, stats in result.workers_comparison.items():
                if stats.mean < best_time:
                    best_time = stats.mean
                    best_nw = nw

            if best_nw != 0:
                speedup = baseline.mean / best_time
                recs.append(
                    f"2. DATA LOADING: Setting num_workers={best_nw} gives a "
                    f"{speedup:.1f}x speedup over num_workers=0 "
                    f"({_format_time(baseline.mean)} -> {_format_time(best_time)} "
                    f"per batch). Update your training config or YAML accordingly."
                )
            else:
                recs.append(
                    "2. DATA LOADING: num_workers=0 is already the fastest. "
                    "This may indicate that the dataset is very small, I/O is "
                    "not the bottleneck, or multiprocessing overhead dominates. "
                    "If your dataset grows, re-run this profiler to check again."
                )
        else:
            recs.append(
                "2. DATA LOADING: Could not measure a baseline with num_workers=0 "
                "(no successful batches). Check for missing video files or data errors."
            )

    # ---- GPU memory ----
    if result.gpu_peak_memory_bytes > 0:
        peak_gb = result.gpu_peak_memory_bytes / 1024**3
        if peak_gb > 10.0:
            recs.append(
                f"3. GPU MEMORY: Peak usage is {peak_gb:.1f} GB. Consider reducing "
                f"batch_size, frame_size, or using gradient accumulation to lower "
                f"memory pressure."
            )
        else:
            recs.append(
                f"3. GPU MEMORY: Peak usage is {peak_gb:.1f} GB, which is within "
                f"a comfortable range."
            )
    elif result.device == "cpu":
        recs.append(
            "3. GPU: No GPU detected. Training on CPU will be significantly slower. "
            "If a GPU is available on the cluster, ensure CUDA is properly configured "
            "and torch.cuda.is_available() returns True."
        )

    # ---- Forward vs backward balance ----
    if result.forward_pass and result.backward_pass:
        fwd = result.forward_pass.mean
        bwd = result.backward_pass.mean
        if fwd > 0 and bwd > 0:
            ratio = bwd / fwd
            if ratio > 3.0:
                recs.append(
                    f"4. BACKWARD PASS: Backward takes {ratio:.1f}x longer than forward. "
                    f"This is higher than typical (~2x). Consider gradient checkpointing "
                    f"(gradient_checkpointing: true in model config) to trade compute "
                    f"for memory and potentially improve throughput."
                )
            elif ratio < 1.0:
                recs.append(
                    f"4. FORWARD/BACKWARD: Backward ({_format_time(bwd)}) is faster "
                    f"than forward ({_format_time(fwd)}). This is unusual; verify that "
                    f"the backbone is frozen (expected for stage 1 profiling)."
                )

    # ---- Data loading variability ----
    if result.data_loading and result.data_loading.std > 0:
        cv = result.data_loading.std / result.data_loading.mean if result.data_loading.mean > 0 else 0
        if cv > 0.5:
            recs.append(
                f"5. DATA LOADING VARIANCE: Coefficient of variation is {cv:.2f} "
                f"(std={_format_time(result.data_loading.std)}, mean={_format_time(result.data_loading.mean)}). "
                f"High variability suggests some video files are slower to decode "
                f"than others. Consider pre-extracting frames to disk or ensuring "
                f"uniform video codecs."
            )

    # ---- AMP ----
    if result.device == "cuda" and result.forward_pass:
        recs.append(
            "6. MIXED PRECISION: Ensure use_amp=true in your training config. "
            "AMP typically provides 1.5-2x speedup on modern NVIDIA GPUs with "
            "minimal accuracy impact."
        )

    # ---- Throughput estimate ----
    if result.end_to_end and result.end_to_end.mean > 0 and result.batch_size > 0:
        samples_per_sec = result.batch_size / result.end_to_end.mean
        recs.append(
            f"7. THROUGHPUT: ~{samples_per_sec:.1f} samples/sec at batch_size={result.batch_size}. "
            f"A full epoch over {result.n_dataset_samples} samples would take "
            f"~{result.n_dataset_samples / samples_per_sec:.0f} seconds "
            f"({result.n_dataset_samples / samples_per_sec / 60:.1f} minutes)."
        )

    if not recs:
        recs.append("No specific recommendations. Profiling did not collect enough data.")

    return recs


def write_timing_breakdown(result: ProfilingResult, output_dir: Path) -> Path:
    """Write timing_breakdown.json."""
    data: dict[str, Any] = {
        "device": result.device,
        "batch_size": result.batch_size,
        "n_batches_profiled": result.n_batches_profiled,
        "n_dataset_samples": result.n_dataset_samples,
        "model_param_count": result.model_param_count,
        "model_trainable_param_count": result.model_trainable_param_count,
        "skipped_batches": result.skipped_batches,
        "components": {
            "data_loading": _stats_to_dict(result.data_loading),
            "forward_pass": _stats_to_dict(result.forward_pass),
            "backward_pass": _stats_to_dict(result.backward_pass),
            "optimizer_step": _stats_to_dict(result.optimizer_step),
            "end_to_end": _stats_to_dict(result.end_to_end),
        },
        "gpu_memory": {
            "peak_allocated_bytes": result.gpu_peak_memory_bytes,
            "peak_allocated_readable": _format_bytes(result.gpu_peak_memory_bytes),
            "allocated_bytes": result.gpu_allocated_memory_bytes,
            "reserved_bytes": result.gpu_reserved_memory_bytes,
        },
    }

    path = output_dir / "timing_breakdown.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Wrote timing breakdown to {path}")
    return path


def write_workers_comparison(result: ProfilingResult, output_dir: Path) -> Path:
    """Write num_workers_comparison.json."""
    baseline = result.workers_comparison.get(0)
    data: dict[str, Any] = {}

    for nw in sorted(result.workers_comparison.keys()):
        stats = result.workers_comparison[nw]
        entry = _stats_to_dict(stats)
        if entry is not None and baseline is not None and baseline.mean > 0:
            entry["speedup_vs_0"] = round(baseline.mean / stats.mean, 2) if stats.mean > 0 else None
        data[str(nw)] = entry

    path = output_dir / "num_workers_comparison.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Wrote num_workers comparison to {path}")
    return path


def write_recommendations(recs: list[str], output_dir: Path) -> Path:
    """Write recommendations.txt."""
    path = output_dir / "recommendations.txt"
    header = (
        "synchronAI Training Pipeline Profiling Recommendations\n"
        "=" * 55 + "\n"
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for rec in recs:
            f.write(rec + "\n\n")
    logger.info(f"Wrote recommendations to {path}")
    return path


def print_summary(result: ProfilingResult, recs: list[str]) -> None:
    """Print a human-readable summary to stdout."""
    sep = "=" * 70

    print(f"\n{sep}")
    print("  synchronAI Training Pipeline Profiling Summary")
    print(sep)

    print(f"\n  Device              : {result.device}")
    print(f"  Batch size          : {result.batch_size}")
    print(f"  Dataset samples     : {result.n_dataset_samples}")
    print(f"  Batches profiled    : {result.n_batches_profiled}")
    print(f"  Skipped batches     : {result.skipped_batches}")
    print(f"  Model parameters    : {result.model_param_count:,} total, "
          f"{result.model_trainable_param_count:,} trainable")

    print(f"\n{'  Component':<28} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("  " + "-" * 66)

    for name, stats in [
        ("Data Loading", result.data_loading),
        ("Forward Pass", result.forward_pass),
        ("Backward Pass", result.backward_pass),
        ("Optimizer Step", result.optimizer_step),
        ("End-to-End Step", result.end_to_end),
    ]:
        if stats and stats.n_samples > 0:
            print(
                f"  {name:<26} {_format_time(stats.mean):>12} "
                f"{_format_time(stats.std):>12} "
                f"{_format_time(stats.min):>12} "
                f"{_format_time(stats.max):>12}"
            )
        else:
            print(f"  {name:<26} {'(no data)':>12}")

    # Percentage breakdown
    if result.end_to_end and result.end_to_end.mean > 0:
        total = result.end_to_end.mean
        print(f"\n  {'Component':<28} {'% of step':>12}")
        print("  " + "-" * 40)
        for name, stats in [
            ("Data Loading", result.data_loading),
            ("Forward Pass", result.forward_pass),
            ("Backward Pass", result.backward_pass),
            ("Optimizer Step", result.optimizer_step),
        ]:
            if stats and stats.mean > 0:
                pct = stats.mean / total * 100
                bar = "#" * int(pct / 2)
                print(f"  {name:<26} {pct:>10.1f}%  {bar}")

    # GPU memory
    if result.gpu_peak_memory_bytes > 0:
        print(f"\n  GPU Peak Memory     : {_format_bytes(result.gpu_peak_memory_bytes)}")
        if result.gpu_allocated_memory_bytes > 0:
            print(f"  GPU Allocated       : {_format_bytes(result.gpu_allocated_memory_bytes)}")
        if result.gpu_reserved_memory_bytes > 0:
            print(f"  GPU Reserved        : {_format_bytes(result.gpu_reserved_memory_bytes)}")

    # num_workers comparison
    if result.workers_comparison:
        baseline = result.workers_comparison.get(0)
        print(f"\n  num_workers Comparison (data loading only):")
        print(f"  {'Workers':>10} {'Mean':>12} {'Speedup':>10}")
        print("  " + "-" * 34)
        for nw in sorted(result.workers_comparison.keys()):
            stats = result.workers_comparison[nw]
            if stats.n_samples > 0:
                speedup = ""
                if baseline and baseline.mean > 0 and stats.mean > 0:
                    s = baseline.mean / stats.mean
                    speedup = f"{s:.2f}x"
                print(f"  {nw:>10} {_format_time(stats.mean):>12} {speedup:>10}")
            else:
                print(f"  {nw:>10} {'(no data)':>12}")

    # Recommendations
    print(f"\n{sep}")
    print("  Recommendations")
    print(sep)
    for rec in recs:
        print(f"\n  {rec}")

    print(f"\n{sep}\n")


# ============================================================================
# Main profiling orchestration
# ============================================================================


def run_profiling(
    config_path: str,
    labels_file: str,
    n_batches: int,
    output_dir: str,
    test_workers: list[int],
) -> ProfilingResult:
    """Run the full profiling pipeline.

    Args:
        config_path: Path to the YAML training config.
        labels_file: Path to labels.csv.
        n_batches: Number of batches to profile per component.
        output_dir: Directory to write output files.
        test_workers: List of num_workers values to benchmark.

    Returns:
        ProfilingResult with all collected metrics.
    """
    result = ProfilingResult(config_path=config_path)

    # ---- Load config ----
    logger.info(f"Loading config from {config_path}")
    config_dict = load_yaml_config(config_path)
    model_config, data_config, training_dict = build_configs_from_yaml(
        config_dict, labels_file
    )

    batch_size = training_dict.get("batch_size", 16)
    use_amp = training_dict.get("use_amp", True)
    gradient_clip_max_norm = training_dict.get("gradient_clip_max_norm", 1.0)
    seed = config_dict.get("seed", 42)

    result.batch_size = batch_size

    logger.info(
        f"Config: backbone={model_config.backbone}, "
        f"temporal={model_config.temporal_aggregation}, "
        f"batch_size={batch_size}, use_amp={use_amp}, "
        f"frame_size={model_config.frame_height}x{model_config.frame_width}, "
        f"n_frames={model_config.n_frames}"
    )

    # ---- Set seed ----
    set_seed(seed)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result.device = str(device)
    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_mem = torch.cuda.get_device_properties(device).total_mem
        logger.info(f"GPU: {gpu_name}, {_format_bytes(gpu_mem)} total memory")
    else:
        logger.warning(
            "No CUDA GPU detected. GPU profiling will be skipped. "
            "AMP will be disabled for CPU profiling."
        )
        use_amp = False

    # ---- Load data ----
    logger.info(f"Loading video index from {labels_file}...")
    try:
        specs = load_video_index(
            labels_file,
            sample_fps=model_config.sample_fps,
            window_seconds=model_config.window_seconds,
            frame_size=model_config.frame_height,
        )
    except FileNotFoundError:
        logger.error(f"Labels file not found: {labels_file}")
        result.errors.append(f"Labels file not found: {labels_file}")
        return result

    if not specs:
        logger.error("No valid video window specs loaded. Check labels file and video paths.")
        result.errors.append("No valid video window specs loaded.")
        return result

    # Use training split only (same split logic as the actual training pipeline)
    train_specs, _val_specs = split_by_video(
        specs,
        val_split=config_dict.get("split", {}).get("val_split", 0.2),
        group_by=config_dict.get("split", {}).get("group_by", "subject_id"),
        seed=seed,
    )
    result.n_dataset_samples = len(train_specs)
    logger.info(f"Using {len(train_specs)} training samples for profiling")

    # ---- Build model ----
    logger.info("Building model...")
    try:
        model = build_video_classifier(model_config)
        model.to(device)
    except Exception as exc:
        logger.error(f"Failed to build model: {exc}")
        result.errors.append(f"Model build failed: {exc}")
        return result

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result.model_param_count = total_params
    result.model_trainable_param_count = trainable_params
    logger.info(
        f"Model: {total_params:,} total params, {trainable_params:,} trainable"
    )

    # ===================================================================
    # Phase 1: Profile num_workers (data loading only)
    # ===================================================================
    logger.info("")
    logger.info("=" * 50)
    logger.info("Phase 1: Profiling num_workers settings")
    logger.info("=" * 50)

    for nw in test_workers:
        logger.info(f"  Testing num_workers={nw} ({n_batches} batches)...")
        try:
            stats, skipped = profile_data_loading(
                train_specs,
                data_config,
                batch_size,
                num_workers=nw,
                n_batches=n_batches,
                seed=seed,
            )
            result.workers_comparison[nw] = stats
            if stats.n_samples > 0:
                logger.info(
                    f"    num_workers={nw}: mean={_format_time(stats.mean)}, "
                    f"std={_format_time(stats.std)}, "
                    f"n={stats.n_samples} batches "
                    f"({skipped} skipped)"
                )
            else:
                logger.warning(
                    f"    num_workers={nw}: No successful batches "
                    f"({skipped} skipped)"
                )
        except Exception as exc:
            logger.warning(f"    num_workers={nw} failed: {exc}")
            result.workers_comparison[nw] = TimingStats()
            result.errors.append(f"num_workers={nw} failed: {exc}")

    # ===================================================================
    # Phase 2: Profile full training step (data + fwd + bwd + optim)
    # ===================================================================
    logger.info("")
    logger.info("=" * 50)
    logger.info("Phase 2: Profiling full training step")
    logger.info("=" * 50)

    # Use num_workers=0 for consistent step profiling (isolates compute cost)
    profiling_num_workers = 0
    dataset = VideoWindowDataset(train_specs, data_config, augment=False)
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": profiling_num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": worker_init_fn,
    }
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs["generator"] = generator

    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

    try:
        samples, gpu_peak, skipped = profile_training_step(
            model=model,
            loader=loader,
            device=device,
            use_amp=use_amp,
            gradient_clip_max_norm=gradient_clip_max_norm,
            n_batches=n_batches,
        )
    except Exception as exc:
        logger.error(f"Training step profiling failed: {exc}")
        result.errors.append(f"Training step profiling failed: {exc}")
        samples = []
        gpu_peak = 0
        skipped = 0
    finally:
        dataset.close()

    result.n_batches_profiled = len(samples)
    result.skipped_batches = skipped
    result.gpu_peak_memory_bytes = gpu_peak

    if device.type == "cuda":
        result.gpu_allocated_memory_bytes = torch.cuda.memory_allocated(device)
        result.gpu_reserved_memory_bytes = torch.cuda.memory_reserved(device)

    if samples:
        result.data_loading = TimingStats.from_values([s.data_load_s for s in samples])
        result.forward_pass = TimingStats.from_values([s.forward_s for s in samples])
        result.backward_pass = TimingStats.from_values([s.backward_s for s in samples])
        result.optimizer_step = TimingStats.from_values(
            [s.optimizer_step_s for s in samples]
        )
        result.end_to_end = TimingStats.from_values([s.end_to_end_s for s in samples])

        logger.info(f"  Profiled {len(samples)} batches ({skipped} skipped)")
        logger.info(f"  End-to-end mean: {_format_time(result.end_to_end.mean)}")
    else:
        logger.warning("  No successful training step samples collected.")

    return result


# ============================================================================
# CLI
# ============================================================================


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile the synchronAI training pipeline to identify bottlenecks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Profile with default settings\n"
            "  python scripts/profile_training.py \\\n"
            "      --config configs/train/video_classifier.yaml \\\n"
            "      --labels-file scripts/data/labels.csv\n\n"
            "  # Profile with more batches and custom worker counts\n"
            "  python scripts/profile_training.py \\\n"
            "      --config configs/train/video_classifier.yaml \\\n"
            "      --labels-file scripts/data/labels.csv \\\n"
            "      --n-batches 50 \\\n"
            "      --test-workers 0,2,4,8,16\n"
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training YAML config file (e.g. configs/train/video_classifier.yaml)",
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        default="scripts/data/labels.csv",
        help="Path to labels CSV file (default: scripts/data/labels.csv)",
    )
    parser.add_argument(
        "--n-batches",
        type=int,
        default=20,
        help="Number of batches to profile per component (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/profiling/",
        help="Directory to write profiling results (default: runs/profiling/)",
    )
    parser.add_argument(
        "--test-workers",
        type=str,
        default="0,2,4,8",
        help="Comma-separated list of num_workers values to test (default: 0,2,4,8)",
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point."""

    # ---- Check dependencies ----
    if _MISSING_DEPS:
        print(
            f"ERROR: Missing required dependencies: {', '.join(_MISSING_DEPS)}",
            file=sys.stderr,
        )
        print(
            "Install them with: pip install -e . (from the synchronAI root)",
            file=sys.stderr,
        )
        return 1

    # ---- Parse args ----
    args = parse_args(argv)

    # ---- Validate inputs ----
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        return 1

    labels_file = Path(args.labels_file)
    if not labels_file.exists():
        print(f"ERROR: Labels file not found: {labels_file}", file=sys.stderr)
        return 1

    # Parse test_workers
    try:
        test_workers = [int(w.strip()) for w in args.test_workers.split(",")]
    except ValueError:
        print(
            f"ERROR: Invalid --test-workers value: {args.test_workers}. "
            "Expected comma-separated integers.",
            file=sys.stderr,
        )
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Configure logging ----
    log_format = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    log_datefmt = "%Y-%m-%d %H:%M:%S"

    # Log to both console and file
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=log_datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "profiling.log", mode="w"),
        ],
    )

    # Suppress noisy loggers from dependencies
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logger.info("synchronAI Training Pipeline Profiler")
    logger.info(f"  Config        : {config_path}")
    logger.info(f"  Labels file   : {labels_file}")
    logger.info(f"  Batches       : {args.n_batches}")
    logger.info(f"  Output dir    : {output_dir}")
    logger.info(f"  Test workers  : {test_workers}")
    logger.info(f"  CPU cores     : {os.cpu_count()}")

    # ---- Run profiling ----
    t_start = time.perf_counter()

    result = run_profiling(
        config_path=str(config_path),
        labels_file=str(labels_file),
        n_batches=args.n_batches,
        output_dir=str(output_dir),
        test_workers=test_workers,
    )

    elapsed = time.perf_counter() - t_start
    logger.info(f"Profiling completed in {_format_time(elapsed)}")

    # ---- Generate outputs ----
    write_timing_breakdown(result, output_dir)
    write_workers_comparison(result, output_dir)

    recs = generate_recommendations(result)
    write_recommendations(recs, output_dir)

    # ---- Print summary ----
    print_summary(result, recs)

    logger.info(f"All results written to {output_dir}/")

    if result.errors:
        logger.warning(f"Profiling completed with {len(result.errors)} error(s):")
        for err in result.errors:
            logger.warning(f"  - {err}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
