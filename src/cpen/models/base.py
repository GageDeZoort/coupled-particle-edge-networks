"""Shared hyperparameter-transfer hooks for jet models."""

from __future__ import annotations

import math
from typing import Protocol


class TransferModel(Protocol):
    """Models that expose width-aware Adam LR mapping."""

    width: int
    out_dim: int
    optimizer: str

    def get_lr(self, eta_0: float, corr: float = 1.0) -> float: ...


def adam_lr(eta_0: float, width: int, *, corr: float = 1.0) -> float:
    """Adam global LR: eta_0 / sqrt(D)."""
    return corr * eta_0 / math.sqrt(width)


def lambda_0_from_t_epoch(
    *,
    eta_0: float,
    batch_size: int,
    n_gpus: int,
    n_train: int,
    t_epoch: float,
) -> float:
    """Base weight decay lambda_0 in lambda = lambda_0 sqrt(D)."""
    batch_eff = batch_size * n_gpus
    return batch_eff / (t_epoch * eta_0 * n_train)


def t_epoch_weight_decay(
    *,
    eta_0: float,
    width: int,
    batch_size: int,
    n_gpus: int,
    n_train: int,
    t_epoch: float,
) -> float:
    """
    AdamW weight decay from the EMA timescale t_epoch.

    lambda = lambda_0 sqrt(D) with lambda_0 = B / (t_epoch eta_0 n_train), so
    the product lambda * eta = 1 / (t_iter) is width-invariant and t_epoch
    transfers across model sizes at fixed batch and dataset size.
    """
    return math.sqrt(width) * lambda_0_from_t_epoch(
        eta_0=eta_0,
        batch_size=batch_size,
        n_gpus=n_gpus,
        n_train=n_train,
        t_epoch=t_epoch,
    )
