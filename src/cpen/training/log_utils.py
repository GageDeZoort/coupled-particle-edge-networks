"""Rank-zero logging helpers for multi-process training."""

from __future__ import annotations

import os


def is_rank_zero() -> bool:
    """True on the primary process (SLURM task 0 / LOCAL_RANK 0)."""
    for key in ("LOCAL_RANK", "SLURM_PROCID", "RANK"):
        if key in os.environ:
            return os.environ[key] == "0"
    return True


def log_info(message: str) -> None:
    """Print a line on rank 0 only."""
    if is_rank_zero():
        print(message, flush=True)
