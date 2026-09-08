"""
Wave-Induced Rotary Encodings (WIRE): RoPE over graph-spectral coordinates.

WIRE assigns each graph token a spectral coordinate ``r_i`` (typically the first
``m`` nontrivial Laplacian eigenvectors) and rotates query/key channel pairs by
angles ``theta[i,h,a] = omega[h,a]^T r_i``. Values are never rotated.

Because the rotations are orthogonal,

    q̃_i^T k̃_j = q_i^T R(r_j - r_i) k_j,

so attention scores depend on *relative* spectral coordinates while preserving
Q/K norms (and therefore this framework's ``alpha_A = 1`` logit scale
``(q·k)/d``).

Laplacian eigenvectors have sign and degenerate-eigenspace ambiguities; optional
sign augmentation and per-column sign canonicalization mitigate some of that
without resolving arbitrary within-degenerate rotations.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def apply_paired_rotation(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Rotate adjacent channel pairs of ``x`` by ``(cos, sin)``.

    Parameters
    ----------
    x:
        ``(..., d)`` with even ``d``.
    cos, sin:
        ``(..., d/2)`` broadcastable against the paired channels.
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rot_even
    out[..., 1::2] = rot_odd
    return out


class WIRERotaryEmbedding(nn.Module):
    """
    Learned (or fixed) WIRE frequencies that rotate Q and K from spectral coords.

    Frequency tensor shape is ``[H, d/2, m]`` or ``[1, d/2, m]`` when sharing
    across heads. Angles / sin / cos are computed in float32 for stability under
    mixed precision; rotated Q/K are cast back to the input dtype.
    """

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        coordinate_dim: int,
        *,
        share_across_heads: bool = False,
        learned_frequencies: bool = True,
        frequency_init_std: float | None = None,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(
                f"WIRE requires even head_dim (pairs of channels); got head_dim={head_dim}"
            )
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1; got {num_heads}")
        if coordinate_dim < 1:
            raise ValueError(f"coordinate_dim must be >= 1; got {coordinate_dim}")

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.coordinate_dim = coordinate_dim
        self.share_across_heads = share_across_heads
        self.learned_frequencies = learned_frequencies
        self.n_pairs = head_dim // 2

        head_rep = 1 if share_across_heads else num_heads
        std = (
            1.0 / math.sqrt(coordinate_dim)
            if frequency_init_std is None
            else float(frequency_init_std)
        )
        self.frequency_init_std = std
        omega = torch.empty(head_rep, self.n_pairs, coordinate_dim)
        nn.init.normal_(omega, mean=0.0, std=std)
        if learned_frequencies:
            self.omega = nn.Parameter(omega)
        else:
            self.register_buffer("omega", omega, persistent=True)

    def _angles(
        self,
        coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` with shape ``(..., H, d/2)`` from coords ``(..., m)``."""
        coords = coordinates.float()
        omega = self.omega.float()
        # coords: (..., m); omega: (H_or_1, A, m) -> theta: (..., H_or_1, A)
        theta = torch.einsum("...m,ham->...ha", coords, omega)
        if self.share_across_heads and theta.shape[-2] == 1:
            theta = theta.expand(*theta.shape[:-2], self.num_heads, self.n_pairs)
        return theta.cos(), theta.sin()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        target_coordinates: torch.Tensor,
        source_coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Rotate queries/keys with target/source spectral coordinates.

        Accepts either unbatched ``(N, H, d)`` or batched ``(B, N, H, d)`` layouts
        used by CAPEN's dense masked attention. Coordinates are ``(N, m)`` or
        ``(B, N, m)`` respectively.
        """
        if q.shape[-1] != self.head_dim or k.shape[-1] != self.head_dim:
            raise ValueError(
                f"WIRE head_dim mismatch: expected {self.head_dim}, "
                f"got q={q.shape[-1]}, k={k.shape[-1]}"
            )
        if q.shape[-2] != self.num_heads or k.shape[-2] != self.num_heads:
            raise ValueError(
                f"WIRE num_heads mismatch: expected {self.num_heads}, "
                f"got q_heads={q.shape[-2]}, k_heads={k.shape[-2]}"
            )

        cos_q, sin_q = self._angles(target_coordinates)
        cos_k, sin_k = self._angles(source_coordinates)
        # Align (..., H, A) with q/k (..., H, d) channel pairs.
        q_rot = apply_paired_rotation(q.float(), cos_q, sin_q)
        k_rot = apply_paired_rotation(k.float(), cos_k, sin_k)
        return q_rot.to(dtype=q.dtype), k_rot.to(dtype=k.dtype)
