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
#: keyed by star radius. Robust statistics (median, ``1.35 / IQR``) on JetClass
#: train jets; robust rather than mean/std because floored degenerate hyperedges
#: form a spike well outside the bulk.
#:
#: ``0.15`` / ``0.20`` were measured on star-only hyperedges (~24k jets).
#: ``0.10`` / ``0.125`` are measured on the **production** star+\(k\)NN edge
#: stream at ``k=12`` (2.4k jets, 6 classes) — silent nearest-key fallback is
#: forbidden because wrong Θ(1) scaling breaks edge-encoder μP.
PART_INTERACTION_PREPROCESS: dict[float, tuple[tuple[float, float], ...]] = {
    # star + ΔR-kNN (k=12) live stream — panel (a)/(b)/(c) operating points
    0.10: (
        (-2.84, 0.945),  # ln Delta
        (-1.43, 0.878),  # ln k_T
        (-1.25, 1.346),  # ln z
        (-0.98, 0.515),  # ln m^2
    ),
    0.125: (
        (-2.82, 0.958),  # ln Delta
        (-1.41, 0.885),  # ln k_T
        (-1.26, 1.338),  # ln z
        (-0.95, 0.512),  # ln m^2
    ),
    # star-only hyperedges (legacy caches / older R★=0.15–0.20 runs)
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

_PREPROCESS_RADIUS_ATOL = 1e-6

#: ``((subtract, multiply), …)`` × 4 features — see :func:`standardize_part_interaction`.
PartInteractionPreprocess = tuple[tuple[float, float], ...]


def part_interaction_preprocess(radius: float) -> PartInteractionPreprocess:
    """Exact-key table lookup for ``radius`` (no nearest-key fallback).

    Prefer :func:`robust_part_interaction_preprocess` / live-batch estimation for
    production star+kNN graphs — table entries are geometry-specific.
    """
    r = float(radius)
    for key, consts in PART_INTERACTION_PREPROCESS.items():
        if abs(float(key) - r) <= _PREPROCESS_RADIUS_ATOL:
            return consts
    known = ", ".join(str(k) for k in sorted(PART_INTERACTION_PREPROCESS))
    raise ValueError(
        f"no part-interaction preprocess constants for star radius R={r:g}; "
        f"known keys: {known}. Estimate from the live edge stream "
        f"(default in training) or add an entry to PART_INTERACTION_PREPROCESS."
    )


def robust_part_interaction_preprocess(
    raw_features: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> PartInteractionPreprocess:
    """Median / ``1.35·IQR`` constants from raw ``(..., 4)`` interaction features."""
    if raw_features.numel() == 0:
        raise ValueError("robust_part_interaction_preprocess needs at least one live edge")
    flat = raw_features.detach().float().reshape(-1, 4)
    if flat.size(0) < 1:
        raise ValueError("robust_part_interaction_preprocess needs at least one live edge")
    q25 = torch.quantile(flat, 0.25, dim=0)
    q50 = torch.quantile(flat, 0.50, dim=0)
    q75 = torch.quantile(flat, 0.75, dim=0)
    iqr = (q75 - q25).clamp_min(eps)
    mul = 1.35 / iqr
    return tuple(
        (float(q50[i].item()), float(mul[i].item())) for i in range(4)
    )


def standardize_part_interaction(
    edge_x: torch.Tensor,
    *,
    radius: float | None = None,
    preprocess: PartInteractionPreprocess | None = None,
) -> torch.Tensor:
    """Center and scale raw interaction features to $\\Theta(1)$.

    Pass ``preprocess`` from a live-batch estimate whenever possible; ``radius``
    table lookup is only a fallback for unit tests / legacy caches.
    """
    if preprocess is None:
        if radius is None:
            raise ValueError(
                "standardize_part_interaction requires preprocess=... or radius=..."
            )
        preprocess = part_interaction_preprocess(radius)
    if len(preprocess) != 4:
        raise ValueError(f"preprocess must have 4 (sub, mul) pairs; got {len(preprocess)}")
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
    part_interaction_preprocess: PartInteractionPreprocess | None = None,
    standardize: bool = True,
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
        raw = part_interaction_features(x_raw, centroid)
        if standardize:
            pp = part_interaction_preprocess
            if pp is None:
                r = float(radius)
                pp = next(
                    (
                        consts
                        for key, consts in PART_INTERACTION_PREPROCESS.items()
                        if abs(float(key) - r) <= _PREPROCESS_RADIUS_ATOL
                    ),
                    None,
                )
                if pp is None:
                    active = center_active.squeeze(-1) > 0.5
                    live = raw[active]
                    if live.numel() == 0:
                        raise ValueError(
                            f"no live star edges to estimate part-interaction "
                            f"preprocess at R={r:g}"
                        )
                    pp = robust_part_interaction_preprocess(live)
            edge_x = standardize_part_interaction(raw, preprocess=pp)
        else:
            edge_x = raw
    edge_x = edge_x * center_active

    if batch == 1:
        node_idx, edge_idx = incidence[0].nonzero(as_tuple=True)
        hyperedge_index = torch.stack([node_idx, edge_idx], dim=0)
    else:
        hyperedge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

    return edge_x, incidence, hyperedge_index


def build_delta_r_knn_edges(
    x_raw: torch.Tensor,
    *,
    k: int,
    mask: torch.Tensor | None = None,
    edge_features: str = "part-interaction",
    radius: float = 0.2,
    part_interaction_preprocess: PartInteractionPreprocess | None = None,
    standardize: bool = True,
    include_self: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Directed ΔR-kNN 2-edges among particles.

    By default ``include_self=False``: self is masked to \(+\infty\) so the \(k\)
    slots are true neighbors. Particle self-attention already lives in M11;
    forcing self into kNN created floored part-interaction spikes and made
    CLI \(k{=}1\) a self-loop-only control rather than one neighbor.

    Returns
    -------
    edge_x:
        ``(B, N·k, 4)`` — empty for padded / inactive slots.
    incidence:
        ``(B, N·k, N)`` — each live slot is a deg-2 edge (or deg-1 only if
        ``include_self=True`` and the self slot is selected).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if edge_features not in EDGE_FEATURE_MODES:
        raise ValueError(
            f"edge_features must be one of {EDGE_FEATURE_MODES}; got {edge_features!r}"
        )
    if mask is None:
        mask = particle_mask_from_four_vectors(x_raw)

    batch, n_particles, _ = x_raw.shape
    device = x_raw.device
    dtype = x_raw.dtype
    # When excluding self, at most N-1 neighbors exist.
    knn_k = min(int(k), max(n_particles - (0 if include_self else 1), 1))
    n_slots = n_particles * int(k)

    eta, phi = eta_phi_from_four_vectors_torch(x_raw)
    dr = delta_r_torch(eta.unsqueeze(-1), phi.unsqueeze(-1), eta.unsqueeze(-2), phi.unsqueeze(-2))
    pair_ok = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    dr = dr.masked_fill(~pair_ok, float("inf"))
    eye = torch.eye(n_particles, device=device, dtype=torch.bool).unsqueeze(0)
    if include_self:
        # Legacy: self at distance 0 so every active particle has ≥1 finite slot.
        dr = torch.where(eye & mask.unsqueeze(-1), torch.zeros_like(dr), dr)
    else:
        dr = dr.masked_fill(eye, float("inf"))

    knn_dist, knn_idx = dr.topk(knn_k, dim=-1, largest=False)
    finite = torch.isfinite(knn_dist) & mask.unsqueeze(-1)

    src = (
        torch.arange(n_particles, device=device)
        .view(1, n_particles, 1)
        .expand(batch, -1, knn_k)
    )
    dst = knn_idx
    e_ids = (
        torch.arange(n_particles, device=device).view(1, n_particles, 1) * int(k)
        + torch.arange(knn_k, device=device).view(1, 1, knn_k)
    ).expand(batch, -1, -1)
    b_ids = (
        torch.arange(batch, device=device).view(batch, 1, 1).expand(batch, n_particles, knn_k)
    )

    incidence = torch.zeros(batch, n_slots, n_particles, dtype=dtype, device=device)
    edge_x = torch.zeros(batch, n_slots, 4, dtype=dtype, device=device)
    flat_b = b_ids[finite]
    flat_e = e_ids[finite]
    flat_s = src[finite]
    flat_d = dst[finite]
    if flat_e.numel():
        incidence[flat_b, flat_e, flat_s] = 1.0
        incidence[flat_b, flat_e, flat_d] = 1.0
        p_s = x_raw[flat_b, flat_s]
        p_d = x_raw[flat_b, flat_d]
        if edge_features == "logdot-dp":
            feats = lorentz_edge_features(p_s.unsqueeze(1), p_d.unsqueeze(1)).squeeze(1)
        else:
            raw = part_interaction_features(p_s, p_d)
            if standardize:
                pp = part_interaction_preprocess
                if pp is None:
                    r = float(radius)
                    pp = next(
                        (
                            consts
                            for key, consts in PART_INTERACTION_PREPROCESS.items()
                            if abs(float(key) - r) <= _PREPROCESS_RADIUS_ATOL
                        ),
                        None,
                    )
                    if pp is None:
                        pp = robust_part_interaction_preprocess(raw)
                feats = standardize_part_interaction(raw, preprocess=pp)
            else:
                feats = raw
        edge_x[flat_b, flat_e] = feats.to(dtype)
    return edge_x, incidence


def build_star_plus_knn_graph(
    x_raw: torch.Tensor,
    *,
    radius: float,
    k: int,
    mask: torch.Tensor | None = None,
    edge_features: str = "part-interaction",
    centroid_weight: str | None = None,
    part_interaction_preprocess: PartInteractionPreprocess | None = None,
    standardize: bool = True,
    include_self: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Concatenate star-$R$ hyperedges with directed ΔR-kNN 2-edges.

    Layout along the edge axis: ``[star (N) | knn (N·k)]``. CAPEN-Llama-att
    M22 still keys off incidence degree (>2 ⇒ hyperedge attention; 2-edges
    keep identity ``f_22``) so the kNN slots participate in M12/M21 only.
    """
    star_x, star_inc, _ = build_star_radius_graph(
        x_raw,
        radius=radius,
        mask=mask,
        edge_features=edge_features,
        centroid_weight=centroid_weight,
        part_interaction_preprocess=part_interaction_preprocess,
        standardize=standardize,
    )
    knn_x, knn_inc = build_delta_r_knn_edges(
        x_raw,
        k=k,
        mask=mask,
        edge_features=edge_features,
        radius=radius,
        part_interaction_preprocess=part_interaction_preprocess,
        standardize=standardize,
        include_self=include_self,
    )
    return torch.cat([star_x, knn_x], dim=1), torch.cat([star_inc, knn_inc], dim=1)


def build_star_kin7_plus_knn_graph(
    x_raw: torch.Tensor,
    *,
    radius: float,
    k: int,
    mask: torch.Tensor | None = None,
    seed_min_pt: float = 0.0,
    part_interaction_preprocess: PartInteractionPreprocess | None = None,
    standardize_knn: bool = True,
    include_self: bool = False,
    typed_features: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Canonical star-$R$ hyperedges with **kin7 aggregate** features + ΔR-kNN.

    * **Seeds:** all live particles by default (``seed_min_pt=0``). Set
      ``seed_min_pt>0`` to optionally drop soft centers.
    * **Support:** ΔR ball of radius ``radius`` (plus the seed).
    * **Star feature:** ParT kin7 of the support 4-vector **sum** vs the jet.
    * **kNN:** part-interaction (4).

    Layout: ``[star (N) | knn (N·k)]``.

    When ``typed_features=False`` (default, legacy): kNN is zero-padded to 7 and
    the return is ``(edge_x, incidence)`` with feature dim 7.

    When ``typed_features=True``: no padding; return
    ``(edge_x_star, edge_x_knn, incidence)`` with dims 7 and 4.
    """
    from cpen.apps.jets.part_kin import (
        N_PART_KIN_FEATURES,
        build_part_kin_features_against_jet,
    )

    if mask is None:
        mask = particle_mask_from_four_vectors(x_raw)
    mask = mask.bool()

    batch, n_particles, _ = x_raw.shape
    device = x_raw.device
    dtype = x_raw.dtype
    n_ef = int(N_PART_KIN_FEATURES)

    eta, phi = eta_phi_from_four_vectors_torch(x_raw)
    pt = torch.hypot(x_raw[..., 1], x_raw[..., 2])
    seed_ok = mask & (pt > float(seed_min_pt))

    eta1, phi1 = eta.unsqueeze(-1), phi.unsqueeze(-1)
    eta2, phi2 = eta.unsqueeze(-2), phi.unsqueeze(-2)
    dr = delta_r_torch(eta1, phi1, eta2, phi2)
    pair_valid = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    within = (dr < float(radius)) & pair_valid
    eye = torch.eye(n_particles, device=device, dtype=torch.bool).view(
        1, n_particles, n_particles
    )
    support = within | (eye & mask.unsqueeze(-1))
    # Only pT-hard centers seed a star; soft particles may still be members.
    support = support & seed_ok.unsqueeze(-1)

    star_inc = support.to(dtype)
    jet = (x_raw * mask.unsqueeze(-1).to(dtype)).sum(dim=1)  # (B, 4)
    # Aggregate 4-sum per center; zero for non-seeds.
    agg = (x_raw.unsqueeze(-3) * support.unsqueeze(-1).to(dtype)).sum(dim=-2)
    star_x = build_part_kin_features_against_jet(agg, jet, mask=seed_ok)
    star_x = star_x.to(dtype) * seed_ok.unsqueeze(-1).to(dtype)

    knn_x, knn_inc = build_delta_r_knn_edges(
        x_raw,
        k=int(k),
        mask=mask,
        edge_features="part-interaction",
        radius=float(radius),
        part_interaction_preprocess=part_interaction_preprocess,
        standardize=standardize_knn,
        include_self=include_self,
    )
    incidence = torch.cat([star_inc, knn_inc.to(dtype)], dim=1)
    if typed_features:
        return star_x, knn_x.to(dtype), incidence

    knn_pad = torch.zeros(batch, knn_x.size(1), n_ef, device=device, dtype=dtype)
    knn_pad[..., : knn_x.size(-1)] = knn_x.to(dtype)
    return torch.cat([star_x, knn_pad], dim=1), incidence


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
