"""Axial :math:`(\\eta, \\phi)`-RoPE for particle–particle (relation 11) attention.

Particles live on the rapidity–azimuth cylinder. The usual HEP distance is

    :math:`\\Delta R = \\sqrt{(\\Delta\\eta)^2 + (\\Delta\\phi)^2}`

with :math:`\\phi` periodic. Rotary embeddings do not encode that polar radius
as a single angle; they are *axial*: half the head-channel pairs rotate in
:math:`\\eta` and half in :math:`\\phi`. Relative scores then depend on
:math:`(\\Delta\\eta, \\Delta\\phi)` separately — the Euclidean chart on the
cylinder, which is the right analogue of 2D ViT RoPE.

Coordinates are jet-centered :math:`(\\Delta\\eta, \\Delta\\phi)` from
:func:`cpen.apps.jets.part_kin.jet_centered_deta_dphi` (already :math:`\\phi`-wrapped).
Rotations are orthogonal, so Q/K norms and CompleteP :math:`\\alpha_A=1`
are preserved. Values are never rotated.

Frequencies are fixed (true RoPE, not learned WIRE :math:`\\omega`). Default
``theta=100`` because jet-centered angles are :math:`O(0.4)`, not token
indices :math:`O(10^3)`; LLaMA's 10000 still works but spends most pairs at
vanishing frequency.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cpen.models.wire import apply_paired_rotation


class EtaPhiRoPE(nn.Module):
    """Fixed axial 2D RoPE; same ``(q, k, r_t, r_s)`` contract as WIRE."""

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        *,
        theta: float = 100.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(
                f"(η,φ)-RoPE requires even head_dim (pairs of channels); got head_dim={head_dim}"
            )
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1; got {num_heads}")
        if float(theta) <= 0.0:
            raise ValueError(f"rope theta must be > 0; got {theta}")

        self.head_dim = int(head_dim)
        self.num_heads = int(num_heads)
        self.theta = float(theta)
        self.n_pairs = head_dim // 2
        # Extra pair (odd n_pairs) goes to η.
        n_eta = (self.n_pairs + 1) // 2
        n_phi = self.n_pairs - n_eta
        self.n_eta_pairs = n_eta
        self.n_phi_pairs = n_phi

        # 1D RoPE schedule on each axis, treating that axis as dim = head_dim/2.
        dim_axis = max(head_dim // 2, 2)
        inv_eta = _rope_inv_freq(n_eta, dim_axis, self.theta)
        inv_phi = _rope_inv_freq(n_phi, dim_axis, self.theta)
        self.register_buffer("inv_freq_eta", inv_eta, persistent=True)
        self.register_buffer("inv_freq_phi", inv_phi, persistent=True)

    def _angles(self, coordinates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` with shape ``(..., H, d/2)`` from coords ``(..., 2)``."""
        if coordinates.size(-1) != 2:
            raise ValueError(
                f"(η,φ)-RoPE expects coordinates (..., 2); got {tuple(coordinates.shape)}"
            )
        coords = coordinates.float()
        deta = coords[..., 0]
        dphi = coords[..., 1]
        parts: list[torch.Tensor] = []
        if self.n_eta_pairs > 0:
            parts.append(deta.unsqueeze(-1) * self.inv_freq_eta.float())
        if self.n_phi_pairs > 0:
            parts.append(dphi.unsqueeze(-1) * self.inv_freq_phi.float())
        theta = torch.cat(parts, dim=-1)
        # Share frequencies across heads: (..., 1, A) broadcasts to (..., H, A).
        theta = theta.unsqueeze(-2)
        if theta.size(-2) == 1:
            theta = theta.expand(*theta.shape[:-2], self.num_heads, self.n_pairs)
        return theta.cos(), theta.sin()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        target_coordinates: torch.Tensor,
        source_coordinates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.shape[-1] != self.head_dim or k.shape[-1] != self.head_dim:
            raise ValueError(
                f"(η,φ)-RoPE head_dim mismatch: expected {self.head_dim}, "
                f"got q={q.shape[-1]}, k={k.shape[-1]}"
            )
        if q.shape[-2] != self.num_heads or k.shape[-2] != self.num_heads:
            raise ValueError(
                f"(η,φ)-RoPE num_heads mismatch: expected {self.num_heads}, "
                f"got q_heads={q.shape[-2]}, k_heads={k.shape[-2]}"
            )
        cos_q, sin_q = self._angles(target_coordinates)
        cos_k, sin_k = self._angles(source_coordinates)
        q_rot = apply_paired_rotation(q.float(), cos_q, sin_q)
        k_rot = apply_paired_rotation(k.float(), cos_k, sin_k)
        return q_rot.to(dtype=q.dtype), k_rot.to(dtype=k.dtype)


def _rope_inv_freq(n_pairs: int, dim_axis: int, theta: float) -> torch.Tensor:
    """``theta**(-2i / dim_axis)`` for ``i = 0 .. n_pairs-1``."""
    if n_pairs <= 0:
        return torch.empty(0, dtype=torch.float32)
    idx = torch.arange(n_pairs, dtype=torch.float32)
    return theta ** (-2.0 * idx / float(dim_axis))
