"""Weights & Biases integration utilities.

Provides a thin wrapper around wandb that gracefully degrades when wandb
is not installed or not configured. Training runs work identically with
or without wandb -- all logging calls become no-ops when unavailable.

Setup:
    1. pip install wandb
    2. wandb login          # paste your API key from https://wandb.ai/authorize
    3. (Optional) Set WANDB_API_KEY env var for headless/HPC environments

Environment variables:
    WANDB_API_KEY   - API key (alternative to `wandb login`)
    WANDB_PROJECT   - Override default project name
    WANDB_ENTITY    - Team/org name (optional)
    WANDB_MODE      - "online" (default), "offline", or "disabled"
    WANDB_DIR       - Directory for local wandb files (default: ./wandb)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    WANDB_AVAILABLE = False


_DEFAULT_PROJECT = "synchronAI"


def init_wandb(
    config: Dict[str, Any],
    *,
    project: Optional[str] = None,
    name: Optional[str] = None,
    tags: Optional[list[str]] = None,
    group: Optional[str] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> bool:
    """Initialize a wandb run. Returns True if wandb is active.

    Args:
        config: Hyperparameter dict to log (e.g., training config).
        project: wandb project name. Defaults to WANDB_PROJECT env or "synchronAI".
        name: Run name (e.g., "dinov2-small-lstm"). Auto-generated if None.
        tags: Tags for filtering runs (e.g., ["feature-training", "dinov2-small"]).
        group: Group name for related runs (e.g., a sweep or experiment).
        save_dir: Directory for local wandb files.

    Returns:
        True if wandb was initialized successfully, False otherwise.
    """
    if not WANDB_AVAILABLE:
        logger.info("wandb not installed -- skipping tracking. Install with: pip install wandb")
        return False

    try:
        wandb.init(
            project=project or _DEFAULT_PROJECT,
            name=name,
            config=config,
            tags=tags,
            group=group,
            dir=str(save_dir) if save_dir else None,
            reinit=True,
        )
        logger.info(f"wandb run initialized: {wandb.run.url}")
        return True
    except Exception as e:
        logger.warning(f"wandb init failed: {e} -- continuing without tracking")
        return False


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log metrics to wandb. No-op if wandb is not active."""
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    try:
        wandb.log(metrics, step=step)
    except Exception as e:
        logger.debug(f"wandb.log failed: {e}")


def log_summary(metrics: Dict[str, Any]) -> None:
    """Log summary metrics (best values, etc.) to wandb."""
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    try:
        for key, value in metrics.items():
            wandb.run.summary[key] = value
    except Exception as e:
        logger.debug(f"wandb summary update failed: {e}")


def log_artifact(
    file_path: Union[str, Path],
    name: Optional[str] = None,
    artifact_type: str = "model",
) -> None:
    """Log a file as a wandb artifact (e.g., best checkpoint)."""
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    try:
        artifact_name = name or f"model-{wandb.run.id}"
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(str(file_path))
        wandb.log_artifact(artifact)
        logger.info(f"Logged wandb artifact: {artifact_name}")
    except Exception as e:
        logger.debug(f"wandb artifact logging failed: {e}")


def finish_wandb() -> None:
    """Finish the current wandb run. No-op if not active."""
    if not WANDB_AVAILABLE or wandb.run is None:
        return
    try:
        wandb.finish()
    except Exception as e:
        logger.debug(f"wandb.finish failed: {e}")
