"""T1 particle operators T_a used in particle-centric residual blocks."""

from __future__ import annotations

import torch


def particle_weights(
    x: torch.Tensor,
    *,
    normalization: str,
    energy_index: int,
) -> torch.Tensor:
    """
    Return per-particle weights z with shape (batch, n_particles).

    *normalization* is ``uniform`` (1/N) or ``energy-weights`` (E_i / sum E).
    """
    batch, n_particles, _ = x.shape
    if normalization == "uniform":
        return x.new_full((batch, n_particles), 1.0 / n_particles)
    if normalization == "energy-weights":
        energy = x[..., energy_index]
        energy = torch.clamp(energy, min=0.0)
        denom = energy.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return energy / denom
    raise ValueError(f"Unknown normalization {normalization!r}")


def apply_t1_operator(
    x: torch.Tensor,
    operator: str,
    *,
    normalization: str,
    energy_index: int,
    adjacency: torch.Tensor | None = None,
    pairwise: torch.Tensor | None = None,
    operator_normalization: str = "degree",
    inv_gamma_11: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """
    Apply a T1 operator to particle features *x* of shape (batch, n_particles, width).

    Supported operators: ``identity``, ``ones``, ``adjacency``.
    """
    if operator == "identity":
        return x

    if operator == "ones":
        z = particle_weights(x, normalization=normalization, energy_index=energy_index)
        pooled = torch.einsum("bn,bnd->bd", z, x)
        return pooled.unsqueeze(1).expand_as(x)

    if operator == "adjacency":
        if operator_normalization == "gamma":
            if pairwise is None:
                raise ValueError("gamma-normalized adjacency requires a pairwise matrix")
            mat = pairwise
            if mat.dim() == 2:
                mat = mat.unsqueeze(0).expand(x.size(0), -1, -1)
            return torch.bmm(mat, x) * inv_gamma_11

        if adjacency is None:
            raise ValueError("adjacency operator requires an adjacency matrix")
        mat = adjacency
        if mat.dim() == 2:
            mat = mat.unsqueeze(0).expand(x.size(0), -1, -1)
        return torch.bmm(normalized_adjacency(mat), x)

    raise ValueError(f"Unknown operator {operator!r}")


def normalized_adjacency(adjacency: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Symmetric normalization: D^{-1/2} (A + I) D^{-1/2}."""
    adj = adjacency
    if adj.dim() == 2:
        eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype)
        adj = adj + eye
    else:
        eye = torch.eye(adj.size(-1), device=adj.device, dtype=adj.dtype).unsqueeze(0)
        adj = adj + eye

    degree = adj.sum(dim=-1).clamp_min(eps)
    inv_sqrt = degree.pow(-0.5)
    if adj.dim() == 2:
        return inv_sqrt.unsqueeze(1) * adj * inv_sqrt.unsqueeze(0)
    return inv_sqrt.unsqueeze(-1) * adj * inv_sqrt.unsqueeze(-2)


def parse_operators(operators: str) -> list[str]:
    """Parse comma-separated operator names."""
    names = [part.strip() for part in operators.split(",") if part.strip()]
    if not names:
        raise ValueError("At least one operator is required")
    allowed = {"identity", "ones", "adjacency"}
    unknown = set(names) - allowed
    if unknown:
        raise ValueError(f"Unknown operators {sorted(unknown)}; allowed: {sorted(allowed)}")
    return names
