"""Hypergraph constructions built on the same particle geometry as kNN graphs."""

from __future__ import annotations

import torch

from cpen.utils.graphs import (
    build_knn_graph,
    lorentz_edge_features,
    particle_mask_from_four_vectors,
    spatial_coords_from_four_vectors,
)


def _energy_weighted_centroid(
    x_raw: torch.Tensor,
    member_idx: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Energy-weighted mean four-vector for hyperedge members (N_members, 4) -> (4,)."""
    members = x_raw[member_idx]
    weights = members[..., 0].clamp_min(0.0)
    denom = weights.sum().clamp_min(eps)
    return (members * (weights / denom).unsqueeze(-1)).sum(dim=0)


def build_star_hypergraph(
    x_raw: torch.Tensor,
    *,
    k: int,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Star hypergraph: one hyperedge per real particle.

    Hyperedge ``i`` connects particle ``i`` to its ``k`` nearest neighbors in
    (px, py, pz) space (same geometry as :func:`build_knn_graph`). The incidence
    row has ones on all members. Edge features compare the center four-vector to
    the energy-weighted centroid of the hyperedge.

    Returns ``(edge_index, edge_x, incidence)`` with shapes matching the kNN
    convention: ``n_edges = n_real_particles`` (not ``n_particles * k``).
    ``edge_index[e] = (center, center)`` stores the hyperedge center index only.
    """
    if x_raw.dim() == 2:
        x_raw = x_raw.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    if mask is None:
        mask = particle_mask_from_four_vectors(x_raw)

    batch, n_particles, _ = x_raw.shape
    if batch != 1:
        raise ValueError("build_star_hypergraph currently supports one jet at a time")
    x0 = x_raw[0]
    mask0 = mask[0]

    coords = spatial_coords_from_four_vectors(x0.unsqueeze(0))[0]
    k_eff = min(k, n_particles)
    dist2 = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).pow(2).squeeze(0)
    invalid = ~mask0
    dist2 = dist2.masked_fill(invalid.unsqueeze(0), float("inf"))
    dist2 = dist2.masked_fill(invalid.unsqueeze(1), float("inf"))

    nn_idx = dist2.topk(k_eff, largest=False).indices  # (N, k)
    active = mask0.nonzero(as_tuple=False).squeeze(-1)
    n_edges = int(active.numel())
    n_particles_int = int(n_particles)

    incidence = x0.new_zeros(n_edges, n_particles_int)
    edge_x = x0.new_zeros(n_edges, 4)
    edge_index = x0.new_zeros(n_edges, 2, dtype=torch.long)

    for row, center in enumerate(active.tolist()):
        neighbors = nn_idx[center]
        members = torch.unique(torch.cat([torch.tensor([center], device=x0.device), neighbors]))
        members = members[mask0[members]]
        incidence[row, members] = 1.0
        centroid = _energy_weighted_centroid(x0, members)
        edge_x[row] = lorentz_edge_features(x0[center].unsqueeze(0), centroid.unsqueeze(0)).squeeze(0)
        edge_index[row, 0] = center
        edge_index[row, 1] = center

    if squeeze:
        return edge_index, edge_x, incidence
    return edge_index.unsqueeze(0), edge_x.unsqueeze(0), incidence.unsqueeze(0)


def build_knn_graph_single(
    x_raw: torch.Tensor,
    *,
    k: int,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wrapper around :func:`build_knn_graph` for a single jet ``(N, 4)``."""
    if x_raw.dim() == 2:
        x_raw = x_raw.unsqueeze(0)
    if mask is not None and mask.dim() == 1:
        mask = mask.unsqueeze(0)
    edge_index, edge_x, incidence = build_knn_graph(x_raw, k=k, mask=mask)
    return edge_index[0], edge_x[0], incidence[0]
