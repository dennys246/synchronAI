"""
Lightweight tracing helpers for pinpointing segfaults.
"""

from __future__ import annotations

import os
import sys


def _trace_enabled() -> bool:
    value = os.environ.get("SYNCHRONAI_TRACE", "")
    return value.lower() in {"1", "true", "yes", "on"}


def trace(message: str) -> None:
    if not _trace_enabled():
        return
    print(f"[TRACE] {message}", file=sys.stderr, flush=True)
