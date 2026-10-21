"""Typed dual-encoder edge streams — §1 routing (TYPED_EDGE_ATTENTION_TARGET.md)."""

from __future__ import annotations

import math

import torch

from cpen.graphs.graph_star import build_star_kin7_plus_knn_graph
from cpen.models.capen_llama_att import CAPENLlamaAtt


def _toy_jets(batch: int = 2, n: int = 8):
    torch.manual_seed(0)
    px = torch.randn(batch, n)
    py = torch.randn(batch, n)
    pz = torch.randn(batch, n)
    e = (px.pow(2) + py.pow(2) + pz.pow(2)).sqrt() + 0.5
    x_raw = torch.stack([e, px, py, pz], dim=-1)
    mask = torch.ones(batch, n, dtype=torch.bool)
    mask[:, -2:] = False
    x_raw = x_raw * mask.unsqueeze(-1)
    return x_raw, mask


def _clustered_jets(batch: int = 2, n: int = 12):
    """Tight spatial clusters so star-R supports have deg>2 hyperedges."""
    torch.manual_seed(1)
    x_raw = torch.zeros(batch, n, 4)
    mask = torch.ones(batch, n, dtype=torch.bool)
    for b in range(batch):
        for c, (cx, cy) in enumerate(((-0.5, 0.0), (0.5, 0.0), (0.0, 0.8))):
            sl = slice(c * 4, (c + 1) * 4)
            px = cx + 0.02 * torch.randn(4)
            py = cy + 0.02 * torch.randn(4)
            pz = 0.05 * torch.randn(4)
            e = (px.pow(2) + py.pow(2) + pz.pow(2)).sqrt() + 0.2
            x_raw[b, sl] = torch.stack([e, px, py, pz], dim=-1)
    return x_raw, mask


def _typed_model(**kwargs):
    defaults = dict(
        n_features=7,
        n_edge_features=7,
        out_dim=2,
        depth=2,
        width=32,
        heads=4,
        typed_edge_streams=True,
        n_star_edge_features=7,
        n_knn_edge_features=4,
        hyperedge_m22_only=False,
    )
    defaults.update(kwargs)
    return CAPENLlamaAtt(**defaults)


def test_typed_graph_no_pad_and_dims():
    x_raw, mask = _toy_jets()
    star_x, knn_x, incidence = build_star_kin7_plus_knn_graph(
        x_raw,
        radius=0.5,
        k=3,
        mask=mask,
        typed_features=True,
        include_self=False,
    )
    assert star_x.shape[-1] == 7
    assert knn_x.shape[-1] == 4
    assert incidence.shape[1] == star_x.size(1) + knn_x.size(1)


def test_typed_encoder_scales_and_forward():
    x_raw, mask = _toy_jets()
    star_x, knn_x, incidence = build_star_kin7_plus_knn_graph(
        x_raw,
        radius=0.5,
        k=3,
        mask=mask,
        typed_features=True,
        include_self=False,
    )
    model = _typed_model()
    assert abs(model._encoder_e_star_scale - math.sqrt(32 / 7)) < 1e-6
    assert abs(model._encoder_e_knn_scale - math.sqrt(32 / 4)) < 1e-6
    assert abs(model._attn_residual_scale_single - 0.5) < 1e-6
    assert not hasattr(model, "encoder_e")

    x = torch.randn(x_raw.size(0), x_raw.size(1), 7)
    edge_mask = incidence.bool().any(dim=-1)
    model.eval()
    with torch.no_grad():
        out = model(
            x,
            edge_x=None,
            incidence=incidence.bool(),
            mask=mask,
            edge_mask=edge_mask,
            edge_x_star=star_x,
            edge_x_knn=knn_x,
        )
    assert out.shape == (x_raw.size(0), 2)
    assert torch.isfinite(out).all()


def test_typed_encoder_init_std_is_d_inv_sqrt():
    """CompleteP: typed encoders must be N(0, σ²) with σ=D^{-1/2}, not N(0,1)."""
    for width in (64, 256):
        torch.manual_seed(0)
        model = _typed_model(width=width, heads=4 if width >= 64 else 2)
        sigma = 1.0 / math.sqrt(width)
        for name, w in (
            ("star", model.encoder_e_star.weight),
            ("knn", model.encoder_e_knn.weight),
            ("x", model.encoder_x.weight),
        ):
            msq = float(w.detach().float().pow(2).mean())
            # Finite-sample mean-square vs σ² (N(0,1) would be ~1).
            assert abs(msq - sigma**2) / sigma**2 < 0.35, (
                f"D={width} {name}: E[w²]={msq:.4g} vs σ²={sigma**2:.4g}"
            )
            assert msq < 0.25, f"D={width} {name}: looks like N(0,1) (msq={msq:.3g})"


def test_typed_per_row_residual_scales():
    """Hyper edges: 1/(√2 L); non-hyper: 1/L — applied in the residual update."""
    depth = 4
    model = _typed_model(depth=depth, width=32)
    attn_s = 1.0 / (math.sqrt(2.0) * depth)
    attn_s1 = 1.0 / float(depth)
    assert abs(model._attn_residual_scale - attn_s) < 1e-6
    assert abs(model._attn_residual_scale_single - attn_s1) < 1e-6

    x_raw, mask = _clustered_jets()
    star_x, knn_x, incidence = build_star_kin7_plus_knn_graph(
        x_raw,
        radius=0.4,
        k=3,
        mask=mask,
        typed_features=True,
        include_self=False,
    )
    s = incidence.bool()
    deg = s.float().sum(dim=-1)
    is_hyper = deg > 2
    edge_mask = s.any(dim=-1)
    assert is_hyper.any() and ((~is_hyper) & edge_mask).any()

    x = torch.randn(x_raw.size(0), x_raw.size(1), 7)
    # Zero MLP residual so layer0 Δh_e is purely the attention residual.
    model._mlp_residual_scale = 0.0

    def _const_attn(module, target, source, support, **kwargs):  # noqa: ARG001
        del module, source, support, kwargs
        return torch.ones_like(target)

    def _const_attn_22(
        module, target, source, *, member_mask, full_support, zeros_elsewhere, probe
    ):  # noqa: ARG001
        del module, source, full_support, zeros_elsewhere, probe
        out = torch.zeros_like(target)
        out[member_mask] = 1.0
        return out

    model._run_attn = _const_attn  # type: ignore[method-assign]
    model._run_attn_22_member_subset = _const_attn_22  # type: ignore[method-assign]
    from cpen.training.stability import disable_probe, enable_probe, parse_probe

    probe = enable_probe(model)
    try:
        model.eval()
        with torch.no_grad():
            model(
                x,
                edge_x=None,
                incidence=s,
                mask=mask,
                edge_mask=edge_mask,
                edge_x_star=star_x,
                edge_x_knn=knn_x,
            )
        snaps = parse_probe(probe, node_mask=mask, edge_mask=edge_mask)
    finally:
        disable_probe(model)

    h0 = snaps["encode"].h_e
    h1 = snaps["layer0"].h_e
    d = h1 - h0
    # f_12=f_22=1 ⇒ hyper Δ = attn_s * 2; non-hyper Δ = attn_s1 * 1 (per feature).
    assert torch.allclose(
        d[is_hyper], torch.full_like(d[is_hyper], 2.0 * attn_s), atol=1e-5
    )
    assert torch.allclose(
        d[(~is_hyper) & edge_mask],
        torch.full_like(d[(~is_hyper) & edge_mask], attn_s1),
        atol=1e-5,
    )


def test_typed_skips_gamma_22():
    model = _typed_model()
    model.attention_normalization = "gamma"
    # Buffer would be used if skip were absent; force a non-1 value.
    model.attn_gamma_22.fill_(7.0)
    assert model._gamma_for("22") == 1.0
    model.attn_gamma_12.fill_(3.0)
    assert model._gamma_for("12") == 3.0


def test_init_from_rejects_untyped_into_typed(tmp_path):
    from cpen.training.init_from import load_init_checkpoint

    class _Lit(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

    untyped = CAPENLlamaAtt(
        n_features=7,
        n_edge_features=7,
        out_dim=2,
        depth=1,
        width=32,
        heads=4,
    )
    typed = _typed_model(depth=1)
    ckpt = tmp_path / "untyped.ckpt"
    torch.save({"state_dict": _Lit(untyped).state_dict()}, ckpt)
    try:
        load_init_checkpoint(_Lit(typed), ckpt, skip_readout=True)
        raise AssertionError("expected typed/untyped mismatch RuntimeError")
    except RuntimeError as exc:
        assert "typed/untyped" in str(exc)


def test_section1_routing_knn_m12_hyper_m22_particles_all_edges():
    """§1: knn get M12 & f22=0; hypers get M12+M22; M21 sees all edges."""
    x_raw, mask = _clustered_jets()
    star_x, knn_x, incidence = build_star_kin7_plus_knn_graph(
        x_raw,
        radius=0.4,
        k=3,
        mask=mask,
        typed_features=True,
        include_self=False,
    )
    s = incidence.bool()
    deg = s.float().sum(dim=-1)
    n_star = star_x.size(1)
    is_hyper = deg > 2
    is_knn = torch.zeros_like(is_hyper)
    is_knn[:, n_star:] = s[:, n_star:].any(dim=-1)
    assert is_hyper.any(), "fixture must produce deg>2 hyperedges"
    assert is_knn.any(), "fixture must produce knn edges"

    model = _typed_model(depth=1)
    x = torch.randn(x_raw.size(0), x_raw.size(1), 7)
    edge_mask = s.any(dim=-1)
    model.eval()
    with torch.no_grad():
        h_star = model.encoder_e_star(star_x) * model._encoder_e_star_scale
        h_knn = model.encoder_e_knn(knn_x) * model._encoder_e_knn_scale
        h_e = torch.cat([h_star, h_knn], dim=1)
        h_x = model.encoder_x(x) * model._encoder_x_scale
        e_ln = model.ln(h_e)
        x_ln = model.ln(h_x)
        model._pending_edge_mask = edge_mask
        m_11, m_21, m_12, _m_22 = model._build_supports(s, mask=mask)
        assert model._pending_is_hyper is not None
        assert torch.equal(model._pending_is_hyper, is_hyper & edge_mask)

        f_12 = model._run_attn(model.attn_12[0], e_ln, x_ln, m_12, relation="12")
        f_22 = model._run_attn_22_member_subset(
            model.attn_22[0],
            e_ln,
            e_ln,
            member_mask=model._pending_is_hyper,
            full_support=None,
            zeros_elsewhere=True,
            probe=None,
        )
        f_21 = model._run_attn(model.attn_21[0], x_ln, e_ln, m_21, relation="21")

        # knn: f_22 identically 0; f_12 path is live (not force-zeroed).
        assert (f_22[is_knn].abs() < 1e-6).all()
        # At least some knn rows of raw f_12 are nonzero (particles→edges).
        assert f_12[is_knn].abs().sum() > 0

        # Hyper: may receive nonzero f_22; off-hyper is exactly 0.
        assert (f_22[~is_hyper].abs() < 1e-6).all()

        # M21 support includes both hyper and knn columns.
        hyper_cols = model._pending_is_hyper
        knn_cols = is_knn
        assert (m_21 & hyper_cols.unsqueeze(1)).any()
        assert (m_21 & knn_cols.unsqueeze(1)).any()
        assert torch.isfinite(f_21).all()

        # Full forward stays finite.
        out = model(
            x,
            edge_x=None,
            incidence=s,
            mask=mask,
            edge_mask=edge_mask,
            edge_x_star=star_x,
            edge_x_knn=knn_x,
        )
        assert torch.isfinite(out).all()
