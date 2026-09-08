"""Placeholder linear encoder; extend for deeper MLP or LapPE-style features."""

from __future__ import annotations

import torch.nn as nn


class LinearEncoder(nn.Module):
    def __init__(self, in_dim: int, width: int, *, sigma_0: float = 1.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, width)
        self.sigma_0 = sigma_0

    def forward(self, x):
        return self.linear(x) * self.sigma_0
