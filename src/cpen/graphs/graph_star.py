"""Star-$R$ hypergraph construction in $(\\eta, \\phi)$ for CPEN / PyG caches."""

from __future__ import annotations

import torch

from cpen.utils.graphs import lorentz_edge_features, particle_mask_from_four_vectors


def eta_phi_from_four_vectors_torch(
    x: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pseudorapidity and azimuth from four-vectors ``[E, px, py, pz]``."""
    px, py, pz = x[..., 1], x[..., 2], x[..., 3]
    phi = torch.atan2(py, px)
    pt = torch.hypot(px, py)
    eta = torch.asinh(pz / pt.clamp_min(eps))
    return eta, phi


def rapidity_phi_from_four_vectors_torch(
    x: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    True rapidity and azimuth from four-vectors ``[E, px, py, pz]``.

    Rapidity shifts by a constant under a boost along the beam, so rapidity
    differences are boost invariant for *massive* four-vectors too. Pseudorapidity
    coincides with it only in the massless limit, which is why
    :func:`eta_phi_from_four_vectors_torch` is not sufficient for composite
    objects such as a hyperedge centroid.
    """
    phi = torch.atan2(x[..., 2], x[..., 1])
    ratio = x[..., 3] / x[..., 0].clamp_min(eps)
    return torch.atanh(ratio.clamp(-1.0 + 1e-7, 1.0 - 1e-7)), phi


def delta_r_torch(
    eta1: torch.Tensor,
    phi1: torch.Tensor,
    eta2: torch.Tensor,
    phi2: torch.Tensor,
) -> torch.Tensor:
    """Delta R with phi wrapping; inputs broadcast."""
    deta = eta1 - eta2
    dphi = (phi1 - phi2).abs()
    dphi = torch.minimum(dphi, 2.0 * torch.pi - dphi)
    return torch.hypot(deta, dphi)


#: Floors on the log arguments of :func:`part_interaction_features`, one per
#: feature. Chosen below detector angular and mass resolution so that
#: degenerate hyperedges land just outside the bulk rather than at
#: ``ln(1e-8) = -18.4``. Degeneracy is common: at $R=0.15$ about 6.6% of
#: star centers have no neighbour inside the radius, so the support centroid
#: coincides with the center and $\\Delta$, $k_T$, $m^2$ all vanish.
PART_INTERACTION_FLOORS: tuple[float, float, float, float] = (
    1e-3,  # Delta   [rad]
    1e-3,  # k_T     [GeV]
    1e-3,  # z       (dimensionless; does not fire when the center is in its support)
    1e-3,  # m^2     [GeV^2]
)


def part_interaction_features(
    p: torch.Tensor,
    q: torch.Tensor,
    *,
    floors: tuple[float, float, float, float] = PART_INTERACTION_FLOORS,
) -> torch.Tensor:
    """
    Particle Transformer pairwise interaction features.

    Returns ``(..., 4)`` holding $(\\ln\\Delta, \\ln k_T, \\ln z, \\ln m^2)$ as
    defined in Qu et al. (arXiv:2202.03772), where $k_T$ and $z$ are the
    splitting variables of the parton shower and $m^2$ is the invariant mass of
    the pair. Unlike ``lorentz_edge_features``, all four are invariant under
    rotations about the beam axis and boosts along it.

    $\\Delta$ uses rapidity rather than pseudorapidity so that the invariance is
    exact when ``q`` is a composite (massive) four-vector such as a hyperedge
    centroid.

    Outputs are raw logs and are *not* $\\Theta(1)$; apply
    :func:`standardize_part_interaction` before feeding a model.
    """
    f_delta, f_kt, f_z, f_m2 = floors

    pt_p = torch.hypot(p[..., 1], p[..., 2])
    pt_q = torch.hypot(q[..., 1], q[..., 2])
    y_p, phi_p = rapidity_phi_from_four_vectors_torch(p)
    y_q, phi_q = rapidity_phi_from_four_vectors_torch(q)

    delta = delta_r_torch(y_p, phi_p, y_q, phi_q)
    pt_min = torch.minimum(pt_p, pt_q)
    k_t = pt_min * delta
    z = pt_min / (pt_p + pt_q).clamp_min(f_z)

    energy = p[..., 0] + q[..., 0]
    px = p[..., 1] + q[..., 1]
    py = p[..., 2] + q[..., 2]
    pz = p[..., 3] + q[..., 3]
    m_sq = energy**2 - (px**2 + py**2 + pz**2)

    return torch.stack(
        [
            delta.clamp_min(f_delta).log(),
            k_t.clamp_min(f_kt).log(),
            z.clamp_min(f_z).log(),
            m_sq.clamp_min(f_m2).log(),
        ],
        dim=-1,
    )


#: Per-feature ``(subtract, multiply)`` for :func:`part_interaction_features`,
#: keyed by star radius, in the same spirit as the ParT/Weaver node-feature
#: constants. Robust statistics (median, ``1.35 / IQR``) measured on 24k
#: JetClass train jets across six classes; robust rather than mean/std because
#: the floored degenerate hyperedges form a spike well outside the bulk.
PART_INTERACTION_PREPROCESS: dict[float, tuple[tuple[float, float], ...]] = {
    0.15: (
        (-3.13, 0.962),  # ln Delta
        (-1.38, 0.906),  # ln k_T
        (-1.36, 1.020),  # ln z
        (0.62, 0.731),  # ln m^2
    ),
    0.20: (
        (-2.90, 0.954),  # ln Delta
        (-1.18, 0.947),  # ln k_T
        (-1.49, 0.957),  # ln z
        (1.07, 0.709),  # ln m^2
    ),
}

EDGE_FEATURE_MODES = ("logdot-dp", "part-interaction")


def part_interaction_preprocess(radius: float) -> tuple[tuple[float, float], ...]:
    """Standardization constants for the star radius nearest to ``radius``."""
    return PART_INTERACTION_PREPROCESS[
        min(PART_INTERACTION_PREPROCESS, key=lambda r: abs(r - radius))
    ]


def standardize_part_interaction(
    edge_x: torch.Tensor,
    *,
    radius: float,
) -> torch.Tensor:
    """Center and scale raw interaction features to $\\Theta(1)$."""
    preprocess = part_interaction_preprocess(radius)
    shift = edge_x.new_tensor([c for c, _ in preprocess])
    scale = edge_x.new_tensor([s for _, s in preprocess])
    return (edge_x - shift) * scale


CENTROID_WEIGHTS = ("energy", "pt")


def _weighted_centroid(
    x_raw: torch.Tensor,
    support: torch.Tensor,
    *,
    weight: str = "energy",
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Weighted mean four-vector per hyperedge center.

    ``x_raw`` is ``(B, N, 4)``, ``support`` is ``(B, N, N)`` with support[b, c, j]
    indicating membership of particle ``j`` in hyperedge centered at ``c``.
    Returns ``(B, N, 4)`` centroids (one per center row).

    ``weight`` selects the scalar weight. ``"pt"`` is invariant under rotations
    about the beam and boosts along it, which makes the resulting centroid
    transform covariantly and the interaction features exactly invariant.
    ``"energy"`` is not boost invariant and breaks that guarantee.
    """
    if weight == "energy":
        scalar = x_raw[..., 0]
    elif weight == "pt":
        scalar = torch.hypot(x_raw[..., 1], x_raw[..., 2])
    else:
        raise ValueError(f"weight must be one of {CENTROID_WEIGHTS}; got {weight!r}")
    weights = scalar.clamp_min(0.0).unsqueeze(-2) * support
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(eps)
    return (x_raw.unsqueeze(-3) * weights.unsqueeze(-1)).sum(dim=-2) / denom


def _energy_weighted_centroid(
    x_raw: torch.Tensor,
    support: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Backward-compatible alias for :func:`_weighted_centroid` with energy weights."""
    return _weighted_centroid(x_raw, support, weight="energy", eps=eps)


def build_star_radius_graph(
    x_raw: torch.Tensor,
    *,
    radius: float,
    mask: torch.Tensor | None = None,
    edge_features: str = "logdot-dp",
    centroid_weight: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched star hypergraph in $(\\eta, \\phi)$.

    One hyperedge per particle slot ``c`` (row ``c`` of ``incidence``). Inactive
    centers produce zero rows.

    ``edge_features`` selects how each hyperedge is featurized against the
    weighted centroid of its support: ``"logdot-dp"`` (default, the Minkowski
    dot plus the lab-frame momentum difference) or ``"part-interaction"`` (the
    Particle Transformer set, standardized).

    ``centroid_weight`` is ``"energy"`` or ``"pt"``. It defaults to ``"energy"``
    for ``"logdot-dp"`` (preserving existing caches) and to ``"pt"`` for
    ``"part-interaction"``, where it is required for exact invariance.

    Returns
    -------
    edge_x:
        ``(B, N, 4)`` features comparing center four-vector to the
        energy-weighted centroid of its star support.
    incidence:
        ``(B, N, N)`` binary incidence matrix $S$ (hyperedge $\\times$ particle).
    hyperedge_index:
        ``(2, nnz)`` PyG-style COO with ``[particle, hyperedge]`` (batch 0 only).
        For ``batch > 1``, use ``incidence``; this field is mainly for inspection.
    """
    if mask is None:
        mask = particle_mask_from_four_vectors(x_raw)

    batch, n_particles, _ = x_raw.shape
    device = x_raw.device
    dtype = x_raw.dtype

    eta, phi = eta_phi_from_four_vectors_torch(x_raw)
    eta1 = eta.unsqueeze(-1)
    phi1 = phi.unsqueeze(-1)
    eta2 = eta.unsqueeze(-2)
    phi2 = phi.unsqueeze(-2)
    dr = delta_r_torch(eta1, phi1, eta2, phi2)

    pair_valid = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    within = (dr < radius) & pair_valid
    eye = torch.eye(n_particles, device=device, dtype=torch.bool).view(1, n_particles, n_particles)
    support = within | (eye & mask.unsqueeze(-1))

    center_active = mask.unsqueeze(-1).to(dtype)
    incidence = support.to(dtype) * center_active

    if edge_features not in EDGE_FEATURE_MODES:
        raise ValueError(
            f"edge_features must be one of {EDGE_FEATURE_MODES}; got {edge_features!r}"
        )
    if centroid_weight is None:
        centroid_weight = "pt" if edge_features == "part-interaction" else "energy"

    centroid = _weighted_centroid(
        x_raw, support.to(dtype), weight=centroid_weight
    )
    if edge_features == "logdot-dp":
        edge_x = lorentz_edge_features(x_raw, centroid)
    else:
        edge_x = standardize_part_interaction(
            part_interaction_features(x_raw, centroid),
            radius=radius,
        )
    edge_x = edge_x * center_active

    if batch == 1:
        node_idx, edge_idx = incidence[0].nonzero(as_tuple=True)
        hyperedge_index = torch.stack([node_idx, edge_idx], dim=0)
    else:
        hyperedge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

    return edge_x, incidence, hyperedge_index


def build_star_radius_graph_single(
    x_raw: torch.Tensor,
    *,
    radius: float,
    mask: torch.Tensor | None = None,
    edge_features: str = "logdot-dp",
    centroid_weight: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-jet wrapper around :func:`build_star_radius_graph`."""
    if x_raw.dim() == 2:
        x_raw = x_raw.unsqueeze(0)
    if mask is not None and mask.dim() == 1:
        mask = mask.unsqueeze(0)
    edge_x, incidence, hyperedge_index = build_star_radius_graph(
        x_raw,
        radius=radius,
        mask=mask,
        edge_features=edge_features,
        centroid_weight=centroid_weight,
    )
    return edge_x.squeeze(0), incidence.squeeze(0), hyperedge_index


def incidence_to_hyperedge_index(incidence: torch.Tensor) -> torch.Tensor:
    """Convert ``(N, N)`` or ``(B, N, N)`` incidence to PyG ``hyperedge_index``."""
    if incidence.dim() == 3:
        if incidence.size(0) != 1:
            raise ValueError("incidence_to_hyperedge_index expects one jet; use incidence tensor")
        incidence = incidence.squeeze(0)
    node_idx, edge_idx = incidence.nonzero(as_tuple=True)
    return torch.stack([node_idx, edge_idx], dim=0)


def to_pyg_data(
    *,
    x: torch.Tensor,
    edge_x: torch.Tensor,
    incidence: torch.Tensor,
    y: torch.Tensor | int,
    mask: torch.Tensor | None = None,
):
    """
    Build a PyG ``Data`` object for one jet.

    Requires ``torch_geometric`` at call time.
    """
    try:
        from torch_geometric.data import Data
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torch_geometric is required for to_pyg_data. Install with: pip install torch-geometric"
        ) from exc

    hyperedge_index = incidence_to_hyperedge_index(incidence)
    data = Data(
        x=x,
        edge_attr=edge_x,
        hyperedge_index=hyperedge_index,
        y=torch.as_tensor(y, dtype=torch.long),
        num_nodes=x.size(0),
    )
    if mask is not None:
        data.mask = mask
    data.incidence = incidence
    return data
