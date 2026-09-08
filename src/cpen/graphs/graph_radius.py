"""Star hypergraphs and radius diagnostics in (eta, phi)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class HypergraphBuild:
    """Star hypergraph stored as incidence S (E x N)."""

    incidence: np.ndarray  # (n_hyperedges, n_particles), float {0,1}
    particle_support_sizes: np.ndarray  # |support| per hyperedge row
    eta_phi: np.ndarray  # (n_active, 2) for active particles only
    active_indices: np.ndarray  # particle indices in full jet array
    center_particles: np.ndarray  # (n_hyperedges,) star center per row
    construction: str = "star-radius"


def eta_phi_from_four_vectors(x: np.ndarray, *, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """
    Pseudorapidity and azimuth from four-vectors ``[E, px, py, pz]``.

    Uses the massless relation eta = -ln(tan(theta/2)) with theta the polar angle.
    """
    px, py, pz = x[..., 1], x[..., 2], x[..., 3]
    phi = np.arctan2(py, px)
    pt = np.hypot(px, py)
    eta = np.arcsinh(pz / np.clip(pt, eps, None))
    return eta, phi


def delta_r(eta1: np.ndarray, phi1: np.ndarray, eta2: np.ndarray, phi2: np.ndarray) -> np.ndarray:
    """Delta R in (eta, phi) with phi wrapping."""
    deta = eta1 - eta2
    dphi = np.abs(phi1 - phi2)
    dphi = np.minimum(dphi, 2.0 * np.pi - dphi)
    return np.hypot(deta, dphi)


def particle_mask_from_four_vectors(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    px, py = x[..., 1], x[..., 2]
    pt = np.hypot(px, py)
    return pt > eps


def build_particle_radius_base_edges(
    x_raw: np.ndarray,
    *,
    radius: float,
    mask: np.ndarray | None = None,
    include_self_loops: bool = True,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
    """
    Undirected pairwise edges from a radius graph in (eta, phi).

    Diagnostic helper: counts ΔR-neighbor pairs without forming star hyperedges.
    """
    if mask is None:
        mask = particle_mask_from_four_vectors(x_raw)
    active = np.flatnonzero(mask)
    eta, phi = eta_phi_from_four_vectors(x_raw[active])
    n = len(active)
    edges: list[tuple[int, int]] = []

    for a in range(n):
        ia = int(active[a])
        if include_self_loops:
            edges.append((ia, ia))
        for b in range(a + 1, n):
            ib = int(active[b])
            if delta_r(eta[a], phi[a], eta[b], phi[b]) < radius:
                edges.append((ia, ib))
    return edges, eta, phi, active


def particle_support_histogram(build: HypergraphBuild) -> dict[int, int]:
    """Count hyperedges by number of participating particles (|support|)."""
    counts: dict[int, int] = {}
    for size in build.particle_support_sizes.tolist():
        counts[int(size)] = counts.get(int(size), 0) + 1
    return counts


def build_star_radius_hypergraph(
    x_raw: np.ndarray,
    *,
    radius: float,
    mask: np.ndarray | None = None,
) -> HypergraphBuild:
    """
    Star hypergraph in (eta, phi): one hyperedge per active particle.

    Hyperedge centered at particle ``c`` includes ``{c} ∪ {j : ΔR(c,j) < R}``.
    """
    if mask is None:
        mask = particle_mask_from_four_vectors(x_raw)
    active = np.flatnonzero(mask)
    eta, phi = eta_phi_from_four_vectors(x_raw[active])
    n_particles = x_raw.shape[0]
    n_active = len(active)

    if n_active == 0:
        return HypergraphBuild(
            incidence=np.zeros((0, n_particles), dtype=np.float32),
            particle_support_sizes=np.zeros(0, dtype=np.int64),
            eta_phi=np.zeros((0, 2), dtype=np.float64),
            active_indices=active,
            center_particles=np.zeros(0, dtype=np.int64),
        )

    n_hyper = n_active
    incidence = np.zeros((n_hyper, n_particles), dtype=np.float32)
    center_particles = np.empty(n_hyper, dtype=np.int64)
    particle_support_sizes = np.zeros(n_hyper, dtype=np.int64)

    for row, a in enumerate(range(n_active)):
        center_global = int(active[a])
        center_particles[row] = center_global
        incidence[row, center_global] = 1.0
        support = 1
        for b in range(n_active):
            if b == a:
                continue
            neighbor_global = int(active[b])
            if delta_r(eta[a], phi[a], eta[b], phi[b]) < radius:
                incidence[row, neighbor_global] = 1.0
                support += 1
        particle_support_sizes[row] = support

    return HypergraphBuild(
        incidence=incidence,
        particle_support_sizes=particle_support_sizes,
        eta_phi=np.stack([eta, phi], axis=1),
        active_indices=active,
        center_particles=center_particles,
    )


def hyperedges_incident_on_particle(build: HypergraphBuild, particle_idx: int) -> np.ndarray:
    """Row indices of hyperedges whose support includes ``particle_idx``."""
    return np.flatnonzero(build.incidence[:, particle_idx] > 0)


def hyperedge_centered_on_particle(build: HypergraphBuild, particle_idx: int) -> int | None:
    """Row index of the star hyperedge centered on ``particle_idx``, if active."""
    match = np.flatnonzero(build.center_particles == particle_idx)
    return int(match[0]) if match.size else None


def node_degree_histogram(build: HypergraphBuild) -> dict[int, int]:
    """Count active particles by incidence column sum (hyperedge membership degree)."""
    active = build.active_indices
    degrees = build.incidence[:, active].sum(axis=0).astype(int)
    counts: dict[int, int] = {}
    for deg in degrees.tolist():
        counts[int(deg)] = counts.get(int(deg), 0) + 1
    return counts


def torch_incidence(build: HypergraphBuild) -> torch.Tensor:
    """Incidence matrix S as float tensor (n_hyperedges, n_particles)."""
    return torch.from_numpy(build.incidence)
