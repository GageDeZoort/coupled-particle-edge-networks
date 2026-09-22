"""Star-$R$ hyperedge featurization: ParT interaction features vs logdot-dp."""

from __future__ import annotations

import math

import pytest
import torch

from cpen.graphs.graph_star import (
    EDGE_FEATURE_MODES,
    PART_INTERACTION_FLOORS,
    build_star_radius_graph,
    part_interaction_features,
    part_interaction_preprocess,
    standardize_part_interaction,
)

RADIUS = 0.15


def _jet(n_particles: int = 24, *, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """A collimated jet of massless particles plus zero padding."""
    g = torch.Generator().manual_seed(seed)
    n_slots = 32
    pt = torch.rand(n_particles, generator=g) * 40.0 + 0.5
    eta = torch.randn(n_particles, generator=g) * 0.1
    phi = torch.randn(n_particles, generator=g) * 0.1
    px, py, pz = pt * torch.cos(phi), pt * torch.sin(phi), pt * torch.sinh(eta)
    e = torch.sqrt(px**2 + py**2 + pz**2)
    x = torch.zeros(1, n_slots, 4)
    x[0, :n_particles] = torch.stack([e, px, py, pz], dim=-1)
    mask = torch.zeros(1, n_slots, dtype=torch.bool)
    mask[0, :n_particles] = True
    return x, mask


def _rotate_phi(x: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate all momenta about the beam axis; energy is unchanged."""
    c, s = math.cos(angle), math.sin(angle)
    out = x.clone()
    out[..., 1] = c * x[..., 1] - s * x[..., 2]
    out[..., 2] = s * x[..., 1] + c * x[..., 2]
    return out


def _boost_z(x: torch.Tensor, rapidity: float) -> torch.Tensor:
    """Boost along the beam axis."""
    ch, sh = math.cosh(rapidity), math.sinh(rapidity)
    out = x.clone()
    out[..., 0] = ch * x[..., 0] + sh * x[..., 3]
    out[..., 3] = sh * x[..., 0] + ch * x[..., 3]
    return out


def test_part_interaction_features_are_rotation_and_boost_invariant():
    """The physics claim: all four features respect the beam-axis symmetries."""
    x, mask = _jet()
    base, _, _ = build_star_radius_graph(
        x, radius=RADIUS, mask=mask, edge_features="part-interaction"
    )
    for transform in (
        lambda t: _rotate_phi(t, 0.7),
        lambda t: _boost_z(t, 0.4),
        lambda t: _boost_z(_rotate_phi(t, -1.9), 0.25),
    ):
        moved, _, _ = build_star_radius_graph(
            transform(x), radius=RADIUS, mask=mask, edge_features="part-interaction"
        )
        torch.testing.assert_close(moved, base, rtol=1e-4, atol=1e-4)


def test_logdot_dp_features_are_not_rotation_invariant():
    """Contrast: 3 of the 4 current features are lab-frame momentum differences."""
    x, mask = _jet()
    base, _, _ = build_star_radius_graph(
        x, radius=RADIUS, mask=mask, edge_features="logdot-dp"
    )
    moved, _, _ = build_star_radius_graph(
        _rotate_phi(x, 0.7), radius=RADIUS, mask=mask, edge_features="logdot-dp"
    )
    assert not torch.allclose(moved, base, rtol=1e-3, atol=1e-3)


def test_edge_feature_modes_share_shape_and_zero_padded_centers():
    x, mask = _jet()
    for mode in EDGE_FEATURE_MODES:
        edge_x, incidence, _ = build_star_radius_graph(
            x, radius=RADIUS, mask=mask, edge_features=mode
        )
        assert edge_x.shape == (1, 32, 4)
        assert torch.all(edge_x[0, ~mask[0]] == 0.0)
        assert torch.all(incidence[0, ~mask[0]] == 0.0)


def test_isolated_center_stays_in_range_instead_of_diverging():
    """
    An isolated center has a support centroid equal to itself, so Delta, k_T and
    m^2 vanish. The physical floors must keep those inside a usable input range.
    """
    x = torch.zeros(1, 4, 4)
    # Two particles far apart in eta so neither is inside the other's radius.
    for slot, eta in ((0, -1.0), (1, 1.0)):
        pt = 10.0
        pz = pt * math.sinh(eta)
        x[0, slot] = torch.tensor([math.sqrt(pt**2 + pz**2), pt, 0.0, pz])
    mask = torch.tensor([[True, True, False, False]])

    edge_x, incidence, _ = build_star_radius_graph(
        x, radius=RADIUS, mask=mask, edge_features="part-interaction"
    )
    assert incidence[0, 0].sum() == 1.0  # isolated: supports only itself
    assert torch.isfinite(edge_x).all()
    assert edge_x.abs().max() < 8.0, "floored features must stay near the bulk"

    # ln z is exactly ln(1/2) for a self-pair, so it should not be floored.
    raw = part_interaction_features(x, x)
    torch.testing.assert_close(raw[0, 0, 2], torch.tensor(math.log(0.5)), atol=1e-5, rtol=1e-5)
    for idx in (0, 1, 3):
        expected = math.log(PART_INTERACTION_FLOORS[idx])
        torch.testing.assert_close(
            raw[0, 0, idx], torch.tensor(expected), atol=1e-4, rtol=1e-4
        )


def test_standardized_features_are_order_one_on_a_jet_sample():
    xs, masks = zip(*(_jet(seed=s) for s in range(64)))
    x, mask = torch.cat(xs), torch.cat(masks)
    edge_x, _, _ = build_star_radius_graph(
        x, radius=RADIUS, mask=mask, edge_features="part-interaction"
    )
    active = edge_x[mask]
    assert active.abs().mean() < 3.0
    for i in range(4):
        assert active[:, i].std() < 3.0, f"feature {i} is not Theta(1)"


def test_preprocess_constants_selected_by_nearest_radius():
    assert part_interaction_preprocess(0.15) == part_interaction_preprocess(0.16)
    assert part_interaction_preprocess(0.15) != part_interaction_preprocess(0.20)


def test_standardization_is_invertible():
    raw = torch.randn(5, 7, 4) * 2.0 - 1.0
    std = standardize_part_interaction(raw, radius=RADIUS)
    preprocess = part_interaction_preprocess(RADIUS)
    shift = torch.tensor([c for c, _ in preprocess])
    scale = torch.tensor([s for _, s in preprocess])
    torch.testing.assert_close(std / scale + shift, raw)


def test_unknown_edge_feature_mode_raises():
    x, mask = _jet()
    with pytest.raises(ValueError, match="edge_features must be one of"):
        build_star_radius_graph(x, radius=RADIUS, mask=mask, edge_features="nope")
