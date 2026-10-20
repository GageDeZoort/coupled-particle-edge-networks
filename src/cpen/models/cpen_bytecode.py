"""Minimal CPEN stub so ``cpen.models`` can import after ``cpen.pyc`` loss.

Full CPEN operator-net is not needed for CAPEN-Llama-att μP probes. This
satisfies ``from cpen.models.cpen import CPEN`` with a thin CompleteP-ish
placeholder; raise on forward if someone trains vanilla CPEN without restoring
the real backend.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from cpen.models.base import adam_lr


class CPEN(nn.Module):
    """Placeholder CPEN — restore real ``cpen.pyc`` for operator-mode training."""

    def __init__(
        self,
        n_features: int,
        n_edge_features: int,
        out_dim: int,
        depth: int,
        width: int,
        *,
        operators: Any = "incidence",
        normalization: str = "uniform",
        energy_index: int = 0,
        optimizer: str = "adam",
        residual_structure: str = "standard",
        operator_normalization: str = "degree",
        operator_backend: str = "dense",
        layer_norm: bool = False,
        gamma_rs: Any = None,
        readout_mode: str = "graph",
        dropout: float = 0.0,
        **_kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.n_edge_features = int(n_edge_features)
        self.out_dim = int(out_dim)
        self.depth = int(depth)
        self.width = int(width)
        self.D = int(width)
        self.operators = operators
        self.normalization = normalization
        self.energy_index = int(energy_index)
        self.optimizer = str(optimizer).lower()
        self.residual_structure = residual_structure
        self.operator_normalization = operator_normalization
        self.operator_backend = operator_backend
        self.layer_norm = bool(layer_norm)
        self.gamma_rs = gamma_rs
        self.readout_mode = readout_mode
        self.dropout = float(dropout)
        sigma = 1.0 / math.sqrt(self.D)
        self.encoder_x = nn.Linear(self.n_features, self.D, bias=False)
        self.encoder_e = nn.Linear(self.n_edge_features, self.D, bias=False)
        self.decoder_x = nn.Linear(self.D, self.out_dim, bias=False)
        self.decoder_e = nn.Linear(self.D, self.out_dim, bias=False)
        self._encoder_x_scale = math.sqrt(self.D / max(self.n_features, 1))
        self._encoder_e_scale = math.sqrt(self.D / max(self.n_edge_features, 1))
        self._readout_scale = 1.0 / (self.D * sigma)
        self._mlp_residual_scale = 1.0 / float(max(self.depth, 1))
        self._feature_probe = None

    def get_lr(self, eta_0: float, corr: float = 1.0) -> float:
        return adam_lr(eta_0, self.width, corr=corr)

    def get_weight_decay(self, lambda_0: float) -> float:
        return float(lambda_0) * math.sqrt(self.width)

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: ARG002
        raise NotImplementedError(
            "CPEN bytecode was lost; this stub cannot train. "
            "Restore cpen.pyc or a full cpen_bytecode.py reconstruction."
        )
