"""
Particle-centric T1->T1 baseline (particle-only network).

Implements the encoder, normalized residual f_{1->1}, and alpha-weighted decoder
with the same muP / CAPEN-Llama scaling conventions (σ = D^{-1/2}).
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn

from cpen.models.base import adam_lr, t_epoch_weight_decay
from cpen.models.operators import apply_t1_operator, parse_operators, particle_weights


class ParticleOnlyNetwork(nn.Module):
    """
    Particle-only residual network over T1 tensors X in R^{N x n_0}.

    With width D and σ = D^{-1/2} (CAPEN-Llama convention):

    - Adam LR: η = η₀ / √D
    - Encoder: W^{(0)} ~ N(0, σ²), forward × 1/(σ √n₀) = √(D/n₀)
    - Residual: W ~ N(0, 1), forward × (1/L)(1/√(n_ops D))
    - Decoder: W^{(L+1)} ~ N(0, σ²), forward × 1/(D σ) = 1/√D
    """

    def __init__(
        self,
        n_features: int,
        out_dim: int,
        depth: int,
        width: int,
        *,
        operators: str | list[str] = "identity",
        normalization: str = "uniform",
        energy_index: int = 0,
        activation: str = "gelu",
        optimizer: str = "adam",
        operator_normalization: str = "degree",
        gamma_11: float = 1.0,
        readout_mode: str = "graph",
    ) -> None:
        super().__init__()
        if isinstance(operators, str):
            operators = parse_operators(operators)
        if normalization not in {"uniform", "energy-weights"}:
            raise ValueError(
                f"normalization must be 'uniform' or 'energy-weights', got {normalization!r}"
            )
        if operator_normalization not in {"degree", "gamma"}:
            raise ValueError(
                f"operator_normalization must be 'degree' or 'gamma', got {operator_normalization!r}"
            )
        if operator_normalization == "gamma" and "adjacency" not in operators:
            raise ValueError("operator_normalization='gamma' requires the adjacency operator")
        if readout_mode not in {"graph", "node"}:
            raise ValueError(f"readout_mode must be 'graph' or 'node', got {readout_mode!r}")

        self.n_features = n_features
        self.out_dim = out_dim
        self.depth = depth
        self.width = width
        self.operator_names = operators
        self.n_operators = len(operators)
        self.normalization = normalization
        self.energy_index = energy_index
        self.optimizer = optimizer.lower()
        self.operator_normalization = operator_normalization
        self.readout_mode = readout_mode
        if operator_normalization == "gamma":
            self.register_buffer(
                "inv_gamma_11",
                torch.tensor(1.0 / gamma_11, dtype=torch.float32),
            )

        act_fn = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}.get(activation.lower())
        if act_fn is None:
            raise ValueError(f"Unknown activation {activation!r}")
        self.activation: Callable[[torch.Tensor], torch.Tensor] = act_fn()

        sigma = 1.0 / math.sqrt(width)

        # Encoder: W^(0) ~ N(0, σ²), forward × 1/(σ √n₀) = √(D/n₀).
        self.encoder = nn.Linear(n_features, width, bias=False)
        nn.init.normal_(self.encoder.weight, mean=0.0, std=sigma)
        self._encoder_scale = 1.0 / (sigma * math.sqrt(n_features))

        # Residual blocks: one weight matrix per operator at each layer.
        # W ~ N(0, 1), forward × (1/L)(1/√(n_ops D)).
        self.residual_weights = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Linear(width, width, bias=False) for _ in range(self.n_operators)]
                )
                for _ in range(depth)
            ]
        )
        for layer_weights in self.residual_weights:
            for linear in layer_weights:
                nn.init.normal_(linear.weight, mean=0.0, std=1.0)
        self._residual_scale = (1.0 / depth) * (
            1.0 / math.sqrt(self.n_operators * width)
        )

        # Decoder: W^(L+1) ~ N(0, σ²), forward × 1/(D σ) = 1/√D (CAPEN-Llama).
        self.decoder = nn.Linear(width, out_dim, bias=False)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=sigma)
        self._readout_scale = 1.0 / (width * sigma)

    @property
    def D(self) -> int:
        return self.width

    def get_lr(self, eta_0: float, corr: float = 1.0) -> float:
        if self.optimizer in {"adam", "adamw"}:
            return adam_lr(eta_0, self.width, corr=corr)
        if self.optimizer == "sgd":
            raise NotImplementedError("SGD LR mapping not yet implemented for ParticleOnlyNetwork.")
        raise ValueError(f"Unknown optimizer {self.optimizer!r}")

    def get_weight_decay(self, lambda_0: float) -> float:
        """AdamW weight decay lambda = lambda_0 sqrt(D), keeping lambda*eta width-invariant."""
        return lambda_0 * math.sqrt(self.width)

    @classmethod
    def t_epoch_weight_decay(
        cls,
        *,
        eta_0: float,
        width: int,
        batch_size: int,
        n_gpus: int,
        n_train: int,
        t_epoch: float,
    ) -> float:
        return t_epoch_weight_decay(
            eta_0=eta_0,
            width=width,
            batch_size=batch_size,
            n_gpus=n_gpus,
            n_train=n_train,
            t_epoch=t_epoch,
        )

    def _pooling_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Decoder weights alpha_u (sum to 1 over particles)."""
        return particle_weights(
            x,
            normalization=self.normalization,
            energy_index=self.energy_index,
        )

    def _f11(
        self,
        x: torch.Tensor,
        layer_idx: int,
        adjacency: torch.Tensor | None,
        pairwise: torch.Tensor | None,
    ) -> torch.Tensor:
        update = x.new_zeros(x.shape)
        inv_gamma_11 = getattr(self, "inv_gamma_11", 1.0)
        for op_idx, op_name in enumerate(self.operator_names):
            tx = apply_t1_operator(
                x,
                op_name,
                normalization=self.normalization,
                energy_index=self.energy_index,
                adjacency=adjacency,
                pairwise=pairwise,
                operator_normalization=self.operator_normalization,
                inv_gamma_11=inv_gamma_11,
            )
            update = update + self.residual_weights[layer_idx][op_idx](self.activation(tx))
        return update * self._residual_scale

    def forward(
        self,
        x: torch.Tensor,
        *,
        adjacency: torch.Tensor | None = None,
        pairwise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Particle features with shape (batch, n_particles, n_features).
        adjacency:
            Optional adjacency matrix with shape (batch, n_particles, n_particles)
            or (n_particles, n_particles). Required for degree-normalized adjacency.
        pairwise:
            Optional unnormalized pairwise matrix S^T S with the same shape as
            *adjacency*. Required for gamma-normalized adjacency.
        """
        if x.dim() != 3:
            raise ValueError(
                f"ParticleOnlyNetwork expects input shape (batch, n_particles, n_features); got {tuple(x.shape)}"
            )

        h = self.encoder(x) * self._encoder_scale
        for layer_idx in range(self.depth):
            h = h + self._f11(h, layer_idx, adjacency, pairwise)

        if self.readout_mode == "node":
            return self.decoder(h) * self._readout_scale

        alpha = self._pooling_weights(h).unsqueeze(-1)
        pooled = (alpha * h).sum(dim=1)
        return self.decoder(pooled) * self._readout_scale
