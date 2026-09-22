"""Compatibility shim — prefer ``cpen.training.init_from``."""

from __future__ import annotations

from cpen.training.init_from import (  # noqa: F401
    freeze_except_last_blocks,
    is_readout_param,
    load_init_checkpoint,
    read_ckpt_eta0,
    read_ckpt_global_step,
)

__all__ = [
    "freeze_except_last_blocks",
    "is_readout_param",
    "load_init_checkpoint",
    "read_ckpt_eta0",
    "read_ckpt_global_step",
]
