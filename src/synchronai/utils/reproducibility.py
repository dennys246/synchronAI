"""
Reproducibility utilities for training.

Provides:
- Seed management across PyTorch, NumPy, and Python
- DataLoader worker initialization
- Environment logging for debugging
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value.
        deterministic: If True, enable fully deterministic operations
                      (may reduce performance).
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

        logger.debug(f"Set PyTorch seed to {seed}, deterministic={deterministic}")
    except ImportError:
        logger.debug("PyTorch not available, skipping torch seed setting")

    # Set environment variable for other libraries
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Set random seed to {seed}")


def worker_init_fn(worker_id: int) -> None:
    """Initialize random state for DataLoader workers.

    Each worker needs its own seed to ensure reproducibility
    when num_workers > 0.
    """
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def get_environment_info() -> dict[str, Any]:
    """Capture environment information for reproducibility logging.

    Returns:
        Dictionary with environment details
    """
    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "hostname": platform.node(),
    }

    # PyTorch info
    try:
        import torch

        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass

    # NumPy info
    try:
        info["numpy_version"] = np.__version__
    except Exception:
        pass

    return info


def log_reproducibility_info(
    save_dir: Union[str, Path],
    seed: int,
    config: Optional[dict] = None,
    split_info: Optional[dict] = None,
) -> Path:
    """Log all reproducibility-relevant information.

    Args:
        save_dir: Directory to save the log
        seed: Random seed used
        config: Training configuration
        split_info: Train/val split information

    Returns:
        Path to saved log file
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    log_data = {
        "seed": seed,
        "environment": get_environment_info(),
    }

    if config:
        log_data["config"] = config

    if split_info:
        log_data["split_info"] = split_info

    log_path = save_dir / "reproducibility_info.json"
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)

    logger.info(f"Saved reproducibility info to {log_path}")
    return log_path


def create_dataloader_with_seed(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    **kwargs,
):
    """Create a DataLoader with reproducible worker initialization.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers
        seed: Random seed
        **kwargs: Additional DataLoader arguments

    Returns:
        Configured DataLoader
    """
    import torch

    generator = torch.Generator()
    generator.manual_seed(seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=generator if shuffle else None,
        **kwargs,
    )
