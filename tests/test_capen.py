"""Tests for the Coupled Attentional Particle-Edge Network (CAPEN)."""

from __future__ import annotations

import math

import pytest
import torch

from cpen.models.capen import (
    CAPEN,
    SupportAttention,
    attention_diversity_stats,
    incidence_ss_t,
    supports_without_m22,
)


def small_graph_batch(
    *,
    n_pad_particles: int = 0,
    n_pad_edges: int = 0,
) -> dict[str, torch.Tensor]:
    """
    One graph: 4 particles, 2 hyperedges (edge0 = {0, 1}, edge1 = {1, 2}).

    Particle 3 is valid but belongs to no edge (empty cross-type neighborhood).
    Optional zero-padding rows are appended to particles and edges.
    """
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


def make_model(**overrides) -> CAPEN:
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


def test_heads_must_divide_width() -> None:
    with pytest.raises(ValueError, match="heads must divide width"):
        make_model(width=16, heads=3)


def test_dropout_rate_validation() -> None:
    for dropout in (-0.1, 1.0, 1.1):
        with pytest.raises(ValueError, match="attention dropout"):
            make_model(dropout=dropout)


def test_pre_softmax_dropout_masks_and_renormalizes() -> None:
    """Dropped entries are zero and retained nonempty rows sum to one."""
    torch.manual_seed(11)
    module = SupportAttention(width=8, heads=2, dropout=0.75)
    module.train()
    support = torch.ones(2, 5, 16, dtype=torch.bool)
    support[0, 4] = False  # originally empty row must remain empty
    logits = torch.randn(2, 2, 5, 16)

    weights = module._attention_weights(logits, support)
    nonempty = support.any(dim=-1).unsqueeze(1).expand(-1, 2, -1)
    row_sums = weights.sum(dim=-1)
    torch.testing.assert_close(
        row_sums[nonempty], torch.ones_like(row_sums[nonempty])
    )
    assert weights[0, :, 4].abs().max() == 0.0
    # With this seed and matrix size, dropout removes supported entries.
    assert (weights == 0).sum() > (~support.unsqueeze(1)).sum()


def test_dropout_is_train_only_and_stochastic() -> None:
    torch.manual_seed(12)
    module = SupportAttention(width=8, heads=2, dropout=0.5)
    target = torch.randn(2, 6, 8)
    source = torch.randn(2, 6, 8)
    support = torch.ones(2, 6, 6, dtype=torch.bool)

    module.train()
    train_a = module(target, source, support)
    train_b = module(target, source, support)
    assert not torch.allclose(train_a, train_b)
    assert torch.isfinite(train_a).all() and torch.isfinite(train_b).all()

    module.eval()
    eval_a = module(target, source, support)
    eval_b = module(target, source, support)
    torch.testing.assert_close(eval_a, eval_b)


def test_coo_incidence_matches_dense() -> None:
    model = make_model()
    batch = small_graph_batch()
    out_dense = model(batch["x"], batch["edge_x"], batch["incidence"], mask=batch["mask"])

    node_idx, edge_idx = batch["incidence"][0].nonzero(as_tuple=True)
    out_coo = model(
        batch["x"],
        batch["edge_x"],
        incidence_node=edge_idx.unsqueeze(0).to(torch.int16),
        incidence_edge=node_idx.unsqueeze(0).to(torch.int16),
        incidence_nnz=torch.tensor([node_idx.numel()], dtype=torch.int32),
        mask=batch["mask"],
    )
    torch.testing.assert_close(out_coo, out_dense, atol=1e-6, rtol=0)


def test_padded_tokens_stay_zero_and_valid_logits_unchanged() -> None:
    model = make_model(readout_mode="node")
    base = small_graph_batch()
    padded = small_graph_batch(n_pad_particles=3, n_pad_edges=2)

    out_base = model(base["x"], base["edge_x"], base["incidence"], mask=base["mask"])
    out_padded = model(padded["x"], padded["edge_x"], padded["incidence"], mask=padded["mask"])

    torch.testing.assert_close(out_padded[:, :4], out_base, atol=1e-6, rtol=0)
    assert out_padded[:, 4:].abs().max() == 0.0


def test_graph_mode_padding_invariant_with_energy_weights() -> None:
    model = make_model(normalization="energy-weights")
    base = small_graph_batch()
    padded = small_graph_batch(n_pad_particles=3, n_pad_edges=2)
    out_base = model(
        base["x"], base["edge_x"], base["incidence"], mask=base["mask"], z=base["z"]
    )
    out_padded = model(
        padded["x"], padded["edge_x"], padded["incidence"], mask=padded["mask"], z=padded["z"]
    )
    torch.testing.assert_close(out_padded, out_base, atol=1e-6, rtol=0)


def test_all_parameters_receive_gradients() -> None:
    """Every parameter is on the forward path (DDP-safe in both readout modes)."""
    for readout_mode in ("graph", "node"):
        model = make_model(readout_mode=readout_mode)
        batch = small_graph_batch()
        out = model(
            batch["x"], batch["edge_x"], batch["incidence"], mask=batch["mask"]
        )
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no grad ({readout_mode})"
            assert param.grad.abs().sum() > 0, f"{name} grad is zero ({readout_mode})"


def test_batched_matches_single() -> None:
    model = make_model(readout_mode="node")
    torch.manual_seed(2)
    x = torch.randn(3, 4, 3)
    edge_x = torch.randn(3, 2, 4)
    incidence = torch.zeros(3, 2, 4, dtype=torch.bool)
    for b in range(3):
        incidence[b, 0, 0] = incidence[b, 0, 1] = True
        incidence[b, 1, b + 1] = incidence[b, 1, 3] = True

    batched = model(x, edge_x, incidence)
    for b in range(3):
        single = model(x[b : b + 1], edge_x[b : b + 1], incidence[b : b + 1])
        torch.testing.assert_close(batched[b : b + 1], single, atol=1e-6, rtol=0)


def test_support_attention_masking() -> None:
    """Targets only see sources inside their support; empty rows give zero."""
    torch.manual_seed(3)
    attn = SupportAttention(width=8, heads=2)
    target = torch.randn(1, 3, 8)
    source = torch.randn(1, 4, 8)
    support = torch.zeros(1, 3, 4, dtype=torch.bool)
    support[0, 0, 0] = support[0, 0, 1] = True  # target 0 sees sources {0, 1}
    support[0, 1, 2] = True  # target 1 sees source {2}
    # target 2 sees nothing -> zero update

    out = attn(target, source, support)
    assert out.shape == (1, 3, 8)
    assert out[0, 2].abs().max() == 0.0

    # Perturbing a source outside target 0's neighborhood leaves it unchanged.
    source_perturbed = source.clone()
    source_perturbed[0, 3] += 10.0
    out_perturbed = attn(target, source_perturbed, support)
    torch.testing.assert_close(out_perturbed[0, 0], out[0, 0], atol=1e-6, rtol=0)
    # Perturbing a source inside the neighborhood changes the output.
    source_inside = source.clone()
    source_inside[0, 1] += 10.0
    out_inside = attn(target, source_inside, support)
    assert not torch.allclose(out_inside[0, 0], out[0, 0])


def test_supports_include_self_connections() -> None:
    s = torch.zeros(1, 2, 4, dtype=torch.bool)
    s[0, 0, 0] = s[0, 0, 1] = True
    s[0, 1, 1] = s[0, 1, 2] = True
    m_11, m_21, m_12, m_22 = CAPEN._build_supports(s)
    assert m_11.shape == (1, 4, 4)
    assert m_22.shape == (1, 2, 2)
    assert m_11[0].diagonal().all()  # every particle slot has a self-connection
    assert m_22[0].diagonal().all()  # every edge slot has a self-connection
    # Particles 0 and 2 never share an edge.
    assert not m_11[0, 0, 2] and not m_11[0, 2, 0]
    # Edges 0 and 1 share particle 1.
    assert m_22[0, 0, 1] and m_22[0, 1, 0]
    assert torch.equal(m_12, s)
    assert torch.equal(m_21, s.transpose(1, 2))


def test_encoder_scale_matches_writeup() -> None:
    model = make_model()
    assert model._encoder_x_scale == pytest.approx(math.sqrt(16 / 3))
    assert model._encoder_e_scale == pytest.approx(math.sqrt(16 / 4))
    assert model._attn_residual_scale == pytest.approx(1.0 / (math.sqrt(2.0) * 2))
    assert model._mlp_residual_scale == pytest.approx(1.0 / 2)
    layer_attn = model.attn_11[0]
    assert layer_attn._logit_scale == pytest.approx(1.0 / 4)  # 1/d, d = D/H = 4
    assert layer_attn._proj_scale == pytest.approx(1.0 / 4)  # 1/sqrt(D), D = 16


def test_registry_builds_capen() -> None:
    import argparse

    from cpen.models.registry import build_model

    args = argparse.Namespace(
        model="capen",
        graph_construction="star-R=0.15",
        normalization="uniform",
        energy_index=0,
        optimizer="adamw",
        readout_mode="graph",
        dropout=0.2,
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
    assert isinstance(model, CAPEN)
    assert model.heads == 8
    assert model.head_dim == 4
    assert model.dropout == pytest.approx(0.2)
    assert model.attn_11[0].dropout == pytest.approx(0.2)
    assert model.get_lr(0.3) == pytest.approx(0.3 / math.sqrt(32))


def test_dropout_is_tagged_in_run_name() -> None:
    from cpen.utils.run_tags import RunOptions, run_basename

    name = run_basename(
        "capen",
        4,
        256,
        0.25,
        options=RunOptions(heads=8, dropout=0.15),
    )
    assert name == "capen_4_256_0p25_h8_drop0p15"


def test_nan_logits_do_not_poison_attention_or_leave_support() -> None:
    """Fallback must stay in-support even when every logit is NaN."""
    torch.manual_seed(0)
    module = SupportAttention(width=8, heads=2, dropout=0.99)
    module.train()
    support = torch.zeros(1, 3, 5, dtype=torch.bool)
    support[0, 0, 1] = support[0, 0, 3] = True
    support[0, 1, 2] = True
    # row 2 empty
    logits = torch.full((1, 2, 3, 5), float("nan"))

    mask = module._drop_support(support, logits.float())
    assert mask.shape == (1, 2, 3, 5)
    # Never attend outside the original support.
    assert not (mask & ~support.unsqueeze(1)).any()
    # Nonempty rows keep ≥1 source; empty rows stay empty.
    assert mask[0, :, 0].any()
    assert mask[0, :, 1].any()
    assert not mask[0, :, 2].any()

    attn = module._attention_weights(logits, support)
    assert torch.isfinite(attn).all()
    assert (attn >= 0).all()
    nonempty = support.any(dim=-1).unsqueeze(1).expand_as(attn[..., 0])
    torch.testing.assert_close(
        attn.sum(dim=-1)[nonempty], torch.ones_like(attn.sum(dim=-1)[nonempty])
    )
    assert attn[0, :, 2].abs().max() == 0.0


def test_high_dropout_forward_stays_finite_with_large_features() -> None:
    """Stress the path that blew up job 11566240: high dropout + large activations."""
    torch.manual_seed(7)
    model = make_model(dropout=0.9, depth=2, width=16, heads=4)
    model.train()
    batch = small_graph_batch()
    x = batch["x"] * 50.0
    edge_x = batch["edge_x"] * 50.0
    out = model(x, edge_x, batch["incidence"], mask=batch["mask"])
    assert torch.isfinite(out).all(), "NaN/Inf under high dropout"
    out.sum().backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_attention_diversity_stats_detect_collapse_and_uniform() -> None:
    support = torch.ones(2, 4, 6, dtype=torch.bool)
    # Uniform attention → high entropy, width ≈ 6.
    uniform = support.float().unsqueeze(1).expand(-1, 3, -1, -1)
    uniform = uniform / uniform.sum(dim=-1, keepdim=True)
    stats_u = attention_diversity_stats(uniform, support)
    assert stats_u is not None
    assert float(stats_u["entropy"].mean()) > 0.99
    assert float(stats_u["eff_width"].mean()) == pytest.approx(6.0, abs=1e-4)
    assert float(stats_u["max_weight"].mean()) == pytest.approx(1.0 / 6.0, abs=1e-4)

    # Fully collapsed onto source 0 → entropy 0, width 1, max 1.
    collapsed = torch.zeros(2, 3, 4, 6)
    collapsed[..., 0] = 1.0
    stats_c = attention_diversity_stats(collapsed, support)
    assert stats_c is not None
    assert float(stats_c["entropy"].max()) < 1e-5
    assert float(stats_c["eff_width"].mean()) == pytest.approx(1.0, abs=1e-4)
    assert float(stats_c["max_weight"].mean()) == pytest.approx(1.0, abs=1e-4)


def test_attention_probe_records_per_layer_relation_heads() -> None:
    model = make_model(depth=2, heads=4)
    batch = small_graph_batch()
    probe: list = []
    model._attention_probe = probe
    try:
        model.eval()
        model(batch["x"], batch["edge_x"], batch["incidence"], mask=batch["mask"])
    finally:
        model._attention_probe = None

    # 2 layers × 4 relations
    assert len(probe) == 8
    relations = {e["relation"] for e in probe}
    assert relations == {"11", "12", "21", "22"}
    for entry in probe:
        assert "entropy" in entry
        assert entry["entropy"].shape == (4,)
        assert entry["eff_width"].shape == (4,)
        assert entry["max_weight"].shape == (4,)
        assert torch.isfinite(entry["entropy"]).all()
        assert torch.isfinite(entry["eff_width"]).all()
        # Width is in (1, n_sources]; entropy in [0, 1].
        assert (entry["eff_width"] >= 1.0 - 1e-4).all()
        assert (entry["entropy"] >= -1e-5).all() and (entry["entropy"] <= 1.0 + 1e-5).all()


def test_identity_m22_skips_edge_edge_attention() -> None:
    """identity_m22: f_22 = e_ln; relation 22 never hits SupportAttention."""
    batch = small_graph_batch()
    model = make_model(identity_m22=True)
    assert model.identity_m22 is True

    calls: list[str] = []
    orig = model.attn_22[0].forward

    def _spy(*args, **kwargs):
        calls.append("22")
        return orig(*args, **kwargs)

    for layer in range(model.depth):
        model.attn_22[layer].forward = _spy  # type: ignore[method-assign]

    model.eval()
    out = model(batch["x"], batch["edge_x"], batch["incidence"], mask=batch["mask"])
    assert out.shape == (1, 2)
    assert torch.isfinite(out).all()
    assert calls == [], f"M22 attention should be skipped; got calls={calls}"

    # Default path still runs M22 attention.
    model2 = make_model(identity_m22=False)
    calls2: list[str] = []
    orig2 = model2.attn_22[0].forward

    def _spy2(*args, **kwargs):
        calls2.append("22")
        return orig2(*args, **kwargs)

    for layer in range(model2.depth):
        model2.attn_22[layer].forward = _spy2  # type: ignore[method-assign]
    model2.eval()
    model2(batch["x"], batch["edge_x"], batch["incidence"], mask=batch["mask"])
    assert len(calls2) == model2.depth


def test_identity_m22_does_not_materialize_line_graph() -> None:
    """Pascal OOM is the (B, M, M) support; identity path must not build it."""
    s = torch.zeros(2, 16, 8, dtype=torch.bool)
    s[:, 0, 0] = s[:, 0, 1] = True
    s[:, 1, 1] = s[:, 1, 2] = True
    m_11, m_21, m_12, m_22 = supports_without_m22(s)
    m11_full, _, _, m22_full = CAPEN._build_supports(s)
    assert torch.equal(m_11, m11_full)
    assert m_12.shape == s.shape
    assert m_21.shape == (2, 8, 16)
    assert m_22.numel() == 0
    assert m22_full.shape == (2, 16, 16)


def test_identity_m22_matches_manual_identity_residual() -> None:
    """With zero init on other edge attns, identity_m22 equals f_22=e_ln path."""
    batch = small_graph_batch()
    model = make_model(identity_m22=True, depth=1, heads=1, width=8)
    # Zero f_12 contribution so edge residual is only scale * f_22.
    with torch.no_grad():
        for p in model.attn_12.parameters():
            p.zero_()
        for p in model.attn_11.parameters():
            p.zero_()
        for p in model.attn_21.parameters():
            p.zero_()
    model.eval()
    out = model(batch["x"], batch["edge_x"], batch["incidence"], mask=batch["mask"])
    assert torch.isfinite(out).all()


def test_incidence_m22_is_ss_t_not_attention() -> None:
    batch = small_graph_batch()
    model = make_model(incidence_m22=True)
    assert model.incidence_m22 is True
    calls: list[str] = []
    orig = model.attn_22[0].forward

    def _spy(*args, **kwargs):
        calls.append("22")
        return orig(*args, **kwargs)

    for layer in range(model.depth):
        model.attn_22[layer].forward = _spy  # type: ignore[method-assign]
    model.eval()
    out = model(batch["x"], batch["edge_x"], batch["incidence"], mask=batch["mask"])
    assert out.shape == (1, 2)
    assert torch.isfinite(out).all()
    assert calls == [], f"M22 attention should be skipped; got calls={calls}"


def test_incidence_ss_t_matches_explicit_line_graph() -> None:
    s = torch.zeros(1, 2, 4)
    s[0, 0, 0] = s[0, 0, 1] = 1.0
    s[0, 1, 1] = s[0, 1, 2] = 1.0
    h = torch.randn(1, 2, 5)
    got = incidence_ss_t(s, h)
    want = torch.bmm(s, torch.bmm(s.transpose(1, 2), h))
    torch.testing.assert_close(got, want)
    model = make_model(incidence_m22=True, depth=1, width=8, heads=1)
    model.eval()
    model._pending_s = s
    dummy_h = torch.randn(1, 2, 8)
    applied = model._run_attn(
        model.attn_22[0], dummy_h, dummy_h, s.bool(), relation="22"
    )
    torch.testing.assert_close(applied, incidence_ss_t(s, dummy_h))
    model._pending_s = None


def test_node_readout_uses_last_layer_edge_params() -> None:
    """Pascal node-only + DDP: last-layer f_12/f_22/mlp_e must sit in the loss."""
    model = make_model(
        readout_mode="node",
        out_dim=5,
        incidence_m22=True,
        depth=2,
        width=8,
        heads=1,
    )
    model.x_only_node_readout = True
    batch = small_graph_batch()
    model.train()
    out = model(batch["x"], batch["edge_x"], batch["incidence"], mask=batch["mask"])
    out.sum().backward()
    unused = [n for n, p in model.named_parameters() if p.grad is None]
    assert unused == [], unused
