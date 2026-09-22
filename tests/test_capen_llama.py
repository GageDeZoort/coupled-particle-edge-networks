"""Tests for CAPEN-Llama (all-to-all particle attention + SwiGLU)."""

from __future__ import annotations

import math

import pytest
import torch

from cpen.models.capen import CAPEN
from cpen.models.capen_llama import CAPENLlama


def small_graph_batch(
    *,
    n_pad_particles: int = 0,
    n_pad_edges: int = 0,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    n, m = 4, 2
    x = torch.randn(1, n, 3)
    edge_x = torch.randn(1, m, 4)
    incidence = torch.zeros(1, m, n, dtype=torch.bool)
    incidence[0, 0, 0] = incidence[0, 0, 1] = True
    incidence[0, 1, 1] = incidence[0, 1, 2] = True
    mask = torch.ones(1, n, dtype=torch.bool)
    z = torch.rand(1, n)
    z = z / z.sum(dim=-1, keepdim=True)

    if n_pad_particles or n_pad_edges:
        x = torch.cat([x, torch.zeros(1, n_pad_particles, 3)], dim=1)
        edge_x = torch.cat([edge_x, torch.zeros(1, n_pad_edges, 4)], dim=1)
        padded = torch.zeros(
            1, m + n_pad_edges, n + n_pad_particles, dtype=torch.bool
        )
        padded[:, :m, :n] = incidence
        incidence = padded
        mask = torch.cat([mask, torch.zeros(1, n_pad_particles, dtype=torch.bool)], dim=1)
        z = torch.cat([z, torch.zeros(1, n_pad_particles)], dim=1)

    return {"x": x, "edge_x": edge_x, "incidence": incidence, "mask": mask, "z": z}


def make_model(**overrides) -> CAPENLlama:
    kwargs = dict(
        n_features=3,
        n_edge_features=4,
        out_dim=2,
        depth=2,
        width=16,
        heads=4,
        readout_mode="graph",
    )
    kwargs.update(overrides)
    torch.manual_seed(1)
    return CAPENLlama(**kwargs)


def make_capen(**overrides) -> CAPEN:
    kwargs = dict(
        n_features=3,
        n_edge_features=4,
        out_dim=2,
        depth=2,
        width=16,
        heads=4,
        readout_mode="graph",
    )
    kwargs.update(overrides)
    torch.manual_seed(1)
    return CAPEN(**kwargs)


def test_is_capen_subclass() -> None:
    assert issubclass(CAPENLlama, CAPEN)
    assert isinstance(make_model(), CAPEN)


def test_graph_mode_output_shape() -> None:
    model = make_model()
    batch = small_graph_batch()
    out = model(batch["x"], batch["edge_x"], batch["incidence"], mask=batch["mask"])
    assert out.shape == (1, 2)
    assert torch.isfinite(out).all()


def test_node_mode_output_shape() -> None:
    model = make_model(readout_mode="node", out_dim=5)
    batch = small_graph_batch()
    out = model(batch["x"], batch["edge_x"], batch["incidence"], mask=batch["mask"])
    assert out.shape == (1, 4, 5)
    assert torch.isfinite(out).all()


def test_m11_is_all_to_all_over_valid_particles() -> None:
    batch = small_graph_batch(n_pad_particles=2, n_pad_edges=1)
    model = make_model(all_to_all_particle_attention=True)
    m_11, m_21, m_12, m_22 = model._build_supports(
        batch["incidence"], mask=batch["mask"]
    )
    # 4 valid particles → 4x4 all-ones block; pads excluded.
    assert m_11.shape == (1, 6, 6)
    assert bool(m_11[0, :4, :4].all())
    assert not bool(m_11[0, 4:, :].any())
    assert not bool(m_11[0, :, 4:].any())
    # Cross / edge supports still from incidence (not all-to-all).
    assert m_12.shape == batch["incidence"].shape
    assert torch.equal(m_12, batch["incidence"])
    assert torch.equal(m_21, batch["incidence"].transpose(1, 2))
    # Edge-edge still has self-loops on every slot.
    assert bool(m_22[0].diag().all())


def test_m11_differs_from_capen_adjacency() -> None:
    batch = small_graph_batch()
    llama = make_model(all_to_all_particle_attention=True)
    m11_llama, _, _, _ = llama._build_supports(
        batch["incidence"], mask=batch["mask"]
    )
    m11_capen, _, _, _ = CAPEN._build_supports(batch["incidence"], mask=batch["mask"])
    # Star/hypergraph adjacency is not full; Llama m_11 is dense over valids.
    assert bool(m11_llama[0].all())
    assert not bool(m11_capen[0].all())
    assert m11_llama[0].sum() > m11_capen[0].sum()


def test_m11_graph_local_matches_capen_when_disabled() -> None:
    batch = small_graph_batch()
    llama = make_model(all_to_all_particle_attention=False)
    m11_llama, _, _, _ = llama._build_supports(
        batch["incidence"], mask=batch["mask"]
    )
    m11_capen, _, _, _ = CAPEN._build_supports(batch["incidence"], mask=batch["mask"])
    assert torch.equal(m11_llama, m11_capen)

def test_padding_stays_zero() -> None:
    model = make_model()
    batch = small_graph_batch(n_pad_particles=3, n_pad_edges=2)
    model._feature_probe = []
    _ = model(
        batch["x"],
        batch["edge_x"],
        batch["incidence"],
        mask=batch["mask"],
    )
    for stage, h_x, h_e, _mask in model._feature_probe:
        assert torch.equal(h_x[0, 4:], torch.zeros_like(h_x[0, 4:])), stage
        assert torch.equal(h_e[0, 2:], torch.zeros_like(h_e[0, 2:])), stage


def test_swiglu_mlp_modules_and_scales() -> None:
    model = make_model(width=16, depth=2)
    assert not hasattr(model, "mlp_x_w1")
    assert len(model.mlp_x_gate) == 2
    assert model.mlp_x_gate[0].weight.shape == (64, 16)
    assert model.mlp_x_up[0].weight.shape == (64, 16)
    assert model.mlp_x_down[0].weight.shape == (16, 64)
    assert model._mlp_w1_scale == pytest.approx(1.0 / math.sqrt(16))
    assert model._mlp_w2_scale == pytest.approx(1.0 / math.sqrt(64))
    assert model._attn_residual_scale == pytest.approx(1.0 / (math.sqrt(2.0) * 2))
    assert model._mlp_residual_scale == pytest.approx(1.0 / 2)


def test_mup_lr_matches_capen() -> None:
    llama = make_model(width=32, heads=8)
    capen = make_capen(width=32, heads=8)
    assert llama.get_lr(0.25) == pytest.approx(capen.get_lr(0.25))
    assert llama.get_weight_decay(1e-3) == pytest.approx(capen.get_weight_decay(1e-3))


def test_more_mlp_params_than_capen() -> None:
    """SwiGLU has three D↔4D matrices vs CAPEN's two → 1.5× MLP params."""
    llama = make_model(width=32, depth=2, heads=4)
    capen = make_capen(width=32, depth=2, heads=4)
    n_llama = sum(p.numel() for p in llama.parameters())
    n_capen = sum(p.numel() for p in capen.parameters())
    assert n_llama > n_capen


def test_registry_builds_capen_llama() -> None:
    import argparse

    from cpen.models.registry import build_model

    args = argparse.Namespace(
        model="capen-llama",
        graph_construction="star-R=0.15",
        normalization="uniform",
        energy_index=0,
        optimizer="adamw",
        readout_mode="graph",
        dropout=0.1,
    )
    model = build_model(
        args,
        n_features=3,
        n_edge_features=4,
        out_dim=2,
        depth=2,
        width=32,
        heads=8,
    )
    assert isinstance(model, CAPENLlama)
    assert model.dropout == pytest.approx(0.1)
    assert model.get_lr(0.3) == pytest.approx(0.3 / math.sqrt(32))


def test_attention_probe_sees_large_m11_support() -> None:
    model = make_model()
    batch = small_graph_batch()
    model._attention_probe = []
    _ = model(batch["x"], batch["edge_x"], batch["incidence"], mask=batch["mask"])
    layer0_11 = [
        e for e in model._attention_probe
        if e.get("relation") == "11" and e.get("stage") == "layer0"
    ]
    assert layer0_11 and "support_size" in layer0_11[0]
    assert layer0_11[0]["support_size"] == pytest.approx(4.0)
