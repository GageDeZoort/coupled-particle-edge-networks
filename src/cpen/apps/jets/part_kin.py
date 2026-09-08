"""ParT TopLandscape ``kin`` particle features from raw four-vectors.

Matches ``utils/convert_top_datasets.py`` + ``data/TopLandscape/top_kin.yaml``
in https://github.com/jet-universe/particle_transformer:

- Jet axis = sum of constituent four-vectors
- ``part_deta = (η − η_J) * sign(η_J)`` (sign 0 → +1)
- ``part_dphi`` = wrapped Δφ to the jet
- Seven pf_features with the same manual standardization as ``top_kin.yaml``

Constituent layout is this repo's ``[E, px, py, pz]``. Pads (pT == 0) are
zeroed in the returned features. After ParT standardization, each particle
row is L2-scaled so ``||x||^2 = 7``. ``z`` (pT fraction) is kept separately
for energy-weight pooling and is not a model input feature.
"""

from __future__ import annotations

import torch

from cpen.utils.graphs import scale_features_l2_sqrt_dim

# ParT TopLandscape pf_features order (top_kin.yaml).
PART_KIN_FEATURE_NAMES = (
    "part_pt_log",
    "part_e_log",
    "part_logptrel",
    "part_logerel",
    "part_deltaR",
    "part_deta",
    "part_dphi",
)
N_PART_KIN_FEATURES = len(PART_KIN_FEATURE_NAMES)
PARTICLE_NORMALIZATION = "part-kin-l2-sqrt-dim"

# ParT pairwise interaction features (Qu, Li, Qian 2022, eq. 3). Used as
# *edge inputs*, not as an attention-logit bias. Order matches the paper:
# (ln Δ, ln k_T, ln z, ln m²). After the logs, each edge row is L2-scaled so
# ``||e||^2 = 4``, matching particle inputs (``||x||^2 = 7``).
PART_INT_FEATURE_NAMES = (
    "ln_delta",
    "ln_kt",
    "ln_z",
    "ln_m2",
)
N_PART_INT_FEATURES = len(PART_INT_FEATURE_NAMES)
EDGE_FEATURE_SET = "part-int-v1-l2"
EDGE_NORMALIZATION = "part-int-l2-sqrt-dim"

# Manual standardization: (subtract_by, multiply_by). Matches top_kin.yaml.
_PART_KIN_PREPROCESS: dict[str, tuple[float, float]] = {
    "part_pt_log": (1.7, 0.7),
    "part_e_log": (2.0, 0.7),
    "part_logptrel": (-4.7, 0.7),
    "part_logerel": (-4.7, 0.7),
    "part_deltaR": (0.2, 4.0),
}


def pt_fraction_weights(
    constituents: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    pT fractions z_i = pT_i / sum_j pT_j over valid constituents.

    Used for energy-weight pooling; not part of the ParT kin feature vector.
    """
    px, py = constituents[..., 1], constituents[..., 2]
    pt = torch.hypot(px, py)
    if mask is not None:
        pt = pt * mask.to(pt.dtype)
    return pt / pt.sum(dim=-1, keepdim=True).clamp_min(eps)


def _pseudorapidity(px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor, *, eps: float) -> torch.Tensor:
    """η = asinh(pz / pT), matching Awkward/vector ``.eta`` for massive-less usage."""
    pt = torch.hypot(px, py)
    return torch.asinh(pz / pt.clamp_min(eps))


def _standardize(name: str, values: torch.Tensor) -> torch.Tensor:
    spec = _PART_KIN_PREPROCESS.get(name)
    if spec is None:
        return values
    subtract, multiply = spec
    return (values - subtract) * multiply


def build_part_kin_features(
    constituents: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    ParT-kin features ``(N, 7)`` or ``(B, N, 7)`` from ``[E, px, py, pz]``.

    Padded slots (mask False / pT≈0) are all-zero rows. Features follow
    ``PART_KIN_FEATURE_NAMES`` with ``top_kin.yaml`` standardization, then
    per-particle L2 scaling so ``||x||^2 = 7``.
    """
    single = constituents.dim() == 2
    if single:
        constituents = constituents.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)
    if constituents.dim() != 3 or constituents.size(-1) != 4:
        raise ValueError(
            f"constituents must be (batch, n, 4) [E, px, py, pz]; got {tuple(constituents.shape)}"
        )

    input_dtype = constituents.dtype
    x = constituents.to(torch.float64)
    energy, px, py, pz = x.unbind(-1)
    pt = torch.hypot(px, py)

    if mask is None:
        mask = pt > eps
    mask = mask.to(torch.bool)
    valid = mask.to(torch.float64)

    n_valid = mask.sum(dim=-1)
    if (n_valid == 0).any():
        bad = torch.nonzero(n_valid == 0).flatten().tolist()
        raise ValueError(f"Jets with no valid constituents at batch indices {bad}")

    summed = (x * valid.unsqueeze(-1)).sum(dim=-2)
    e_j, px_j, py_j, pz_j = summed.unbind(-1)
    pt_j = torch.hypot(px_j, py_j)
    energy_j = e_j
    eta_j = _pseudorapidity(px_j, py_j, pz_j, eps=eps)
    phi_j = torch.atan2(py_j, px_j)

    eta = _pseudorapidity(px, py, pz, eps=eps)
    phi = torch.atan2(py, px)

    eta_sign = torch.sign(eta_j)
    eta_sign = torch.where(eta_sign == 0, torch.ones_like(eta_sign), eta_sign)
    part_deta = (eta - eta_j.unsqueeze(-1)) * eta_sign.unsqueeze(-1)

    raw_dphi = phi - phi_j.unsqueeze(-1)
    part_dphi = torch.atan2(torch.sin(raw_dphi), torch.cos(raw_dphi))

    # Avoid log(0) on pads; values are zeroed by the mask below.
    part_pt_log = torch.log(pt.clamp_min(eps))
    part_e_log = torch.log(energy.clamp_min(eps))
    part_logptrel = torch.log((pt / pt_j.unsqueeze(-1).clamp_min(eps)).clamp_min(eps))
    part_logerel = torch.log((energy / energy_j.unsqueeze(-1).clamp_min(eps)).clamp_min(eps))
    part_deltaR = torch.hypot(part_deta, part_dphi)

    raw_by_name = {
        "part_pt_log": part_pt_log,
        "part_e_log": part_e_log,
        "part_logptrel": part_logptrel,
        "part_logerel": part_logerel,
        "part_deltaR": part_deltaR,
        "part_deta": part_deta,
        "part_dphi": part_dphi,
    }
    feats = [_standardize(name, raw_by_name[name]) for name in PART_KIN_FEATURE_NAMES]
    features = torch.stack(feats, dim=-1) * valid.unsqueeze(-1)
    features = scale_features_l2_sqrt_dim(features, eps=eps, mask=mask)
    features = features.to(input_dtype)
    if single:
        features = features.squeeze(0)
    return features


def _rapidity(energy: torch.Tensor, pz: torch.Tensor, *, eps: float) -> torch.Tensor:
    r"""Minkowski rapidity \(y = \tfrac{1}{2}\ln\frac{E+p_z}{E-p_z}\) (ParT uses \(y\), not \(\eta\))."""
    return 0.5 * torch.log(
        (energy + pz).clamp_min(eps) / (energy - pz).clamp_min(eps)
    )


def build_part_interaction_features(
    p: torch.Tensor,
    q: torch.Tensor,
    *,
    eps: float = 1e-12,
    l2_normalize: bool = True,
) -> torch.Tensor:
    r"""
    ParT pairwise kinematics ``(..., 4)`` from two four-vectors ``[E, px, py, pz]``.

    .. math::

        \Delta=\sqrt{(\Delta y)^2+(\Delta\phi)^2},\quad
        k_T=\min(p_{T,a},p_{T,b})\,\Delta,\quad
        z=\min(p_{T,a},p_{T,b})/(p_{T,a}+p_{T,b}),\quad
        m^2=(E_a+E_b)^2-\|\mathbf{p}_a+\mathbf{p}_b\|^2.

    Returns ``(ln Δ, ln k_T, ln z, ln m²)``, then (by default) L2-scales each
    row so ``||e||^2 = 4``. Pass ``l2_normalize=False`` to inspect the raw logs
    (e.g. cluster mass checks). Inputs broadcast on leading dims.

    Numerical clamps: Δ, k_T, z, m² are floored at ``eps`` so self-loops and
    collinear pairs stay finite (ParT's N×N matrix includes the diagonal too).
    """
    p, q = torch.broadcast_tensors(p, q)
    if p.size(-1) != 4:
        raise ValueError(f"expected last dim 4 [E, px, py, pz]; got {tuple(p.shape)}")
    input_dtype = p.dtype
    p64 = p.to(torch.float64)
    q64 = q.to(torch.float64)

    e_a, px_a, py_a, pz_a = p64.unbind(-1)
    e_b, px_b, py_b, pz_b = q64.unbind(-1)
    pt_a = torch.hypot(px_a, py_a)
    pt_b = torch.hypot(px_b, py_b)
    y_a = _rapidity(e_a, pz_a, eps=eps)
    y_b = _rapidity(e_b, pz_b, eps=eps)
    phi_a = torch.atan2(py_a, px_a)
    phi_b = torch.atan2(py_b, px_b)
    dphi = torch.atan2(torch.sin(phi_a - phi_b), torch.cos(phi_a - phi_b))
    delta = torch.hypot(y_a - y_b, dphi).clamp_min(eps)
    pt_min = torch.minimum(pt_a, pt_b)
    kt = (pt_min * delta).clamp_min(eps)
    z = (pt_min / (pt_a + pt_b).clamp_min(eps)).clamp_min(eps)
    e_sum = e_a + e_b
    p_sum_sq = (px_a + px_b).square() + (py_a + py_b).square() + (pz_a + pz_b).square()
    m2 = (e_sum.square() - p_sum_sq).clamp_min(eps)

    feats = torch.stack(
        [torch.log(delta), torch.log(kt), torch.log(z), torch.log(m2)],
        dim=-1,
    )
    if l2_normalize:
        feats = scale_features_l2_sqrt_dim(feats, eps=eps)
    return feats.to(input_dtype)


def leading_split_four_vectors(
    constituents: torch.Tensor,
    member_index: torch.Tensor,
    *,
    pt: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hardest member and the remainder 4-sum among ``member_index``.

    ``m²(hardest, remainder)`` is the invariant mass of the whole subset — the
    1→2 kinematics of that substructure component. ``constituents`` is
    ``(N, 4)``; ``member_index`` is 1-d long.
    """
    if constituents.dim() != 2 or constituents.size(-1) != 4:
        raise ValueError(
            f"constituents must be (N, 4); got {tuple(constituents.shape)}"
        )
    members = constituents[member_index]
    if members.size(0) == 0:
        raise ValueError("leading_split_four_vectors needs at least one member")
    if pt is None:
        pt_all = torch.hypot(constituents[:, 1], constituents[:, 2])
    else:
        pt_all = pt
    hard_local = int(pt_all[member_index].argmax().item())
    hardest = members[hard_local]
    remainder = members.sum(dim=0) - hardest
    return hardest, remainder
