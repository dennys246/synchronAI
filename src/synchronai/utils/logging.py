"""
Project logging helpers.

Centralizing setup keeps CLI output consistent across training/inference scripts.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


def setup_logging(level: str = "INFO", *, tf_cpp_min_log_level: Optional[str] = "2") -> None:
    """
    Configure Python logging to stdout.

    Args:
      level: Python logging level string (e.g. "INFO", "DEBUG").
      tf_cpp_min_log_level: Sets TF_CPP_MIN_LOG_LEVEL if provided.
        "2" hides INFO/WARNING from TensorFlow C++ backend.
    """
    if tf_cpp_min_log_level is not None:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", str(tf_cpp_min_log_level))

    root = logging.getLogger()
    if root.handlers:
        # Avoid double-logging if caller already configured.
        root.setLevel(level.upper())
        for handler in root.handlers:
            handler.setLevel(level.upper())
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
    )

    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
