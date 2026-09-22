"""Tests for ParT TopLandscape kin particle features."""

from __future__ import annotations

import math

import pytest
import torch

from cpen.utils.part_kin import (
    N_PART_INT_FEATURES,
    N_PART_KIN_FEATURES,
    PART_INT_FEATURE_NAMES,
    PART_KIN_FEATURE_NAMES,
    build_part_interaction_features,
    build_part_kin_features,
    leading_split_four_vectors,
    pt_fraction_weights,
)
from cpen.utils.toptagging import preprocess_particle_features

TOL = 1e-5


def _toy_jet(n: int = 8, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    px = torch.randn(n, generator=g)
    py = torch.randn(n, generator=g)
    pz = torch.randn(n, generator=g)
    e = torch.linalg.norm(torch.stack([px, py, pz], dim=-1), dim=-1) + 0.5
    return torch.stack([e, px, py, pz], dim=-1)


def test_feature_shape_and_names() -> None:
    jet = _toy_jet()
    feats = build_part_kin_features(jet)
    assert feats.shape == (8, N_PART_KIN_FEATURES)
    assert len(PART_KIN_FEATURE_NAMES) == N_PART_KIN_FEATURES


def test_pads_are_zero() -> None:
    jet = _toy_jet(6)
    padded = torch.zeros(10, 4)
    padded[:6] = jet
    feats = build_part_kin_features(padded)
    assert torch.count_nonzero(feats[6:]) == 0
    torch.testing.assert_close(feats[:6], build_part_kin_features(jet), atol=TOL, rtol=0)


def test_jet_centered_deta_dphi_matches_kin_columns() -> None:
    from cpen.utils.part_kin import jet_centered_deta_dphi

    jet = _toy_jet(8, seed=6)
    padded = torch.zeros(12, 4)
    padded[:8] = jet
    mask = torch.zeros(12, dtype=torch.bool)
    mask[:8] = True
    coords = jet_centered_deta_dphi(padded, mask)
    feats = build_part_kin_features(padded, mask)
    assert PART_KIN_FEATURE_NAMES[-2:] == ("part_deta", "part_dphi")
    torch.testing.assert_close(coords, feats[..., -2:], atol=TOL, rtol=0)
    assert torch.count_nonzero(coords[8:]) == 0


def test_permutation_invariance_of_jet_axis_centering() -> None:
    """Feature rows permute with particles; jet-relative coords stay consistent."""
    jet = _toy_jet(7, seed=3)
    perm = torch.randperm(7)
    feats = build_part_kin_features(jet)
    feats_perm = build_part_kin_features(jet[perm])
    torch.testing.assert_close(feats[perm], feats_perm, atol=TOL, rtol=0)


def test_eta_sign_flip_canonicalization() -> None:
    """part_deta uses sign(η_J); flipping all pz flips jet eta and cancels."""
    jet = _toy_jet(9, seed=4)
    # Force a clearly positive jet eta, then the global sign flip.
    jet = jet.clone()
    jet[..., 3] = jet[..., 3].abs() + 1.0
    jet[..., 0] = torch.linalg.norm(jet[..., 1:], dim=-1) + 0.1

    flipped = jet.clone()
    flipped[..., 3] *= -1.0

    f1 = build_part_kin_features(jet)
    f2 = build_part_kin_features(flipped)
    # deta, deltaR, and standardized kinematics that depend on angles should match;
    # absolute log E / log pT are unchanged under pz → -pz.
    torch.testing.assert_close(f1, f2, atol=TOL, rtol=0)


def test_standardization_matches_parT_top_kin_yaml() -> None:
    """Affine-only kin features — same as ParT top_kin.yaml / JetClass_kin.yaml."""
    jet = _toy_jet(5, seed=5)
    feats = build_part_kin_features(jet)
    energy, px, py, pz = jet.unbind(-1)
    pt = torch.hypot(px, py)
    pt_j = torch.hypot(px.sum(), py.sum())
    e_j = energy.sum()
    eta = torch.asinh(pz / pt.clamp_min(1e-12))
    phi = torch.atan2(py, px)
    eta_j = torch.asinh(pz.sum() / pt_j.clamp_min(1e-12))
    phi_j = torch.atan2(py.sum(), px.sum())
    eta_sign = 1.0 if eta_j >= 0 else -1.0
    deta = (eta - eta_j) * eta_sign
    dphi = torch.atan2(torch.sin(phi - phi_j), torch.cos(phi - phi_j))
    expected = torch.stack(
        [
            (torch.log(pt) - 1.7) * 0.7,
            (torch.log(energy) - 2.0) * 0.7,
            (torch.log(pt / pt_j) + 4.7) * 0.7,
            (torch.log(energy / e_j) + 4.7) * 0.7,
            (torch.hypot(deta, dphi) - 0.2) * 4.0,
            deta,
            dphi,
        ],
        dim=-1,
    )
    torch.testing.assert_close(feats, expected.to(feats.dtype), atol=TOL, rtol=0)


def test_optional_l2_sqrt_dim_ablation() -> None:
    jet = _toy_jet(5, seed=5)
    raw = build_part_kin_features(jet, l2_normalize=False)
    feats = build_part_kin_features(jet, l2_normalize=True)
    from cpen.utils.graphs import scale_features_l2_sqrt_dim

    expected = scale_features_l2_sqrt_dim(raw)
    torch.testing.assert_close(feats, expected.to(feats.dtype), atol=TOL, rtol=0)
    norms_sq = feats.square().sum(dim=-1)
    torch.testing.assert_close(
        norms_sq, torch.full_like(norms_sq, float(N_PART_KIN_FEATURES)), atol=1e-4, rtol=0
    )


def test_batch_matches_single() -> None:
    jets = torch.stack([_toy_jet(6, seed=i) for i in range(4)], dim=0)
    batched = build_part_kin_features(jets)
    for i in range(4):
        single = build_part_kin_features(jets[i])
        torch.testing.assert_close(batched[i], single, atol=TOL, rtol=0)


def test_empty_jet_raises() -> None:
    with pytest.raises(ValueError, match="no valid constituents"):
        build_part_kin_features(torch.zeros(10, 4))


def test_preprocess_pipeline() -> None:
    jet = _toy_jet(12, seed=9)
    padded = torch.zeros(40, 4)
    padded[:12] = jet
    x_raw, x_model, mask = preprocess_particle_features(padded)
    assert x_model.shape == (40, N_PART_KIN_FEATURES)
    assert mask[:12].all() and not mask[12:].any()
    z = pt_fraction_weights(x_raw, mask)
    assert math.isclose(float(z.sum()), 1.0, rel_tol=0, abs_tol=1e-5)
    assert torch.count_nonzero(x_model[12:]) == 0
    torch.testing.assert_close(
        x_model[:12], build_part_kin_features(jet), atol=TOL, rtol=0
    )


def test_edge_features_have_squared_norm_m0() -> None:
    from cpen.utils.graphs import lorentz_edge_features

    g = torch.Generator().manual_seed(0)
    p = torch.randn(5, 4, generator=g)
    p[:, 0] = torch.linalg.norm(p[:, 1:], dim=-1) + 0.2
    q = torch.randn(5, 4, generator=g)
    q[:, 0] = torch.linalg.norm(q[:, 1:], dim=-1) + 0.2
    edge = lorentz_edge_features(p, q)
    norms_sq = edge.square().sum(dim=-1)
    torch.testing.assert_close(norms_sq, torch.full((5,), 4.0), atol=1e-5, rtol=0)


def _back_to_back(energy: float = 40.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Massless pair with m = 2E (CM, p along ±x)."""
    p = torch.tensor([[energy, energy, 0.0, 0.0]])
    q = torch.tensor([[energy, -energy, 0.0, 0.0]])
    return p, q


def test_part_interaction_w_mass() -> None:
    p, q = _back_to_back(40.0)
    feat = build_part_interaction_features(p, q, l2_normalize=False).squeeze(0)
    assert list(PART_INT_FEATURE_NAMES) == ["ln_delta", "ln_kt", "ln_z", "ln_m2"]
    assert feat.shape == (N_PART_INT_FEATURES,)
    # Δφ = π, y = 0 → Δ = π; z = 1/2; m² = (80)² = 6400.
    torch.testing.assert_close(feat[0], torch.tensor(math.log(math.pi)), atol=1e-5, rtol=0)
    torch.testing.assert_close(feat[2], torch.tensor(math.log(0.5)), atol=1e-5, rtol=0)
    torch.testing.assert_close(feat[3], torch.tensor(math.log(6400.0)), atol=1e-5, rtol=0)
    kt = 40.0 * math.pi
    torch.testing.assert_close(feat[1], torch.tensor(math.log(kt)), atol=1e-5, rtol=0)


def test_part_interaction_mass_scale_survives() -> None:
    """Raw ln m² tracks physical mass; L2 is applied after this coordinate is built."""
    p, q = _back_to_back(40.0)
    f1 = build_part_interaction_features(p, q, l2_normalize=False)
    f2 = build_part_interaction_features(2.0 * p, 2.0 * q, l2_normalize=False)
    torch.testing.assert_close(f2[0, 3] - f1[0, 3], torch.tensor(math.log(4.0)), atol=1e-5, rtol=0)


def test_part_interaction_l2_sqrt_dim() -> None:
    p, q = _back_to_back(40.0)
    feat = build_part_interaction_features(p, q)
    from cpen.utils.graphs import scale_features_l2_sqrt_dim

    raw = build_part_interaction_features(p, q, l2_normalize=False)
    expected = scale_features_l2_sqrt_dim(raw)
    torch.testing.assert_close(feat, expected.to(feat.dtype), atol=1e-5, rtol=0)
    norms_sq = feat.square().sum(dim=-1)
    torch.testing.assert_close(norms_sq, torch.full_like(norms_sq, 4.0), atol=1e-5, rtol=0)


def test_part_interaction_symmetric_and_self_finite() -> None:
    p, q = _back_to_back(25.0)
    fwd = build_part_interaction_features(p, q)
    rev = build_part_interaction_features(q, p)
    torch.testing.assert_close(fwd, rev, atol=1e-6, rtol=0)
    self = build_part_interaction_features(p, p)
    assert torch.isfinite(self).all()


def test_leading_split_m2_is_subset_mass() -> None:
    jet = _toy_jet(6, seed=2)
    idx = torch.tensor([0, 2, 4])
    hard, rest = leading_split_four_vectors(jet, idx)
    feat = build_part_interaction_features(hard.unsqueeze(0), rest.unsqueeze(0), l2_normalize=False).squeeze(0)
    total = jet[idx].sum(0)
    m2 = (total[0].square() - total[1:].square().sum()).clamp_min(1e-12)
    torch.testing.assert_close(feat[3], torch.log(m2), atol=1e-5, rtol=0)
