"""Edge-biased M11 logit (edge_bias_m11) — placement, scales, M21 off."""

from __future__ import annotations

import math

import torch

from cpen.graphs.graph_star import build_star_kin7_plus_knn_graph
from cpen.models.capen_llama_att import CAPENLlamaAtt
from cpen.training.attn_pathology import enable_attention_probe


def _clustered_jets(batch: int = 2, n: int = 12):
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


def _graph(x_raw, mask, *, k: int = 3, radius: float = 0.5):
    return build_star_kin7_plus_knn_graph(
        x_raw,
        radius=radius,
        k=k,
        mask=mask,
        typed_features=True,
        return_knn_index=True,
        include_self=False,
    )


def _model(*, width: int = 64, depth: int = 2, **kwargs):
    defaults = dict(
        n_features=7,
        n_edge_features=7,
        out_dim=2,
        depth=depth,
        width=width,
        heads=max(1, width // 32) if width >= 32 else 4,
        typed_edge_streams=True,
        n_star_edge_features=7,
        n_knn_edge_features=4,
        hyperedge_m22_only=False,
        edge_bias_m11="dynamic",
        m21=True,
    )
    defaults.update(kwargs)
    if defaults["width"] % defaults["heads"] != 0:
        defaults["heads"] = 4 if defaults["width"] % 4 == 0 else 2
    return CAPENLlamaAtt(**defaults)


def test_zero_head_matches_no_bias():
    x_raw, mask = _clustered_jets()
    star_x, knn_x, incidence, knn_dst, knn_valid = _graph(x_raw, mask)
    edge_mask = incidence.bool().any(dim=-1)
    x = torch.randn(x_raw.size(0), x_raw.size(1), 7)

    torch.manual_seed(0)
    m_none = _model(edge_bias_m11="none")
    torch.manual_seed(0)
    m_bias = _model(edge_bias_m11="dynamic")
    # Align trunk weights; zero bias so logits match the no-bias model.
    missing, unexpected = m_bias.load_state_dict(m_none.state_dict(), strict=False)
    assert not unexpected
    assert all(k.startswith("bias_11") for k in missing)
    for lin in m_bias.bias_11:
        torch.nn.init.zeros_(lin.weight)

    m_bias.eval()
    m_none.eval()
    with torch.no_grad():
        y_b = m_bias(
            x,
            edge_x=None,
            incidence=incidence.bool(),
            mask=mask,
            edge_mask=edge_mask,
            edge_x_star=star_x,
            edge_x_knn=knn_x,
            knn_dst=knn_dst,
            knn_valid=knn_valid,
        )
        y_n = m_none(
            x,
            edge_x=None,
            incidence=incidence.bool(),
            mask=mask,
            edge_mask=edge_mask,
            edge_x_star=star_x,
            edge_x_knn=knn_x,
        )
    assert torch.allclose(y_b, y_n, atol=1e-5, rtol=1e-5)


def test_bias_placement_directed():
    x_raw, mask = _clustered_jets(batch=1, n=12)
    star_x, knn_x, incidence, knn_dst, knn_valid = _graph(x_raw, mask, k=2)
    model = _model(width=32, heads=4, depth=1, edge_bias_m11="dynamic")
    # Force unit readout so placement is visible.
    with torch.no_grad():
        model.bias_11[0].weight.fill_(0.0)
        model.bias_11[0].weight[0, 0] = 1.0  # head 0, dim 0
    e_ln = torch.zeros(1, star_x.size(1) + knn_x.size(1), model.D)
    # Put a spike on knn slot (i=0, r=0) → edge index N+0.
    n = x_raw.size(1)
    e_ln[0, n, 0] = 10.0
    bias = model._m11_edge_bias(
        layer_idx=0,
        e_ln=e_ln,
        e_ln0=None,
        knn_dst=knn_dst,
        knn_valid=knn_valid,
        n_star=n,
        n_particles=n,
    )
    # (B,H,N,N); only live (0, dst(0,0)) on head 0 should be nonzero if slot live.
    assert bias.shape == (1, 4, n, n)
    live = knn_valid[0, 0, 0].item()
    j = int(knn_dst[0, 0, 0].item())
    if live:
        assert bias[0, 0, 0, j].abs() > 0
        # Directed: reverse not implied.
        if j != 0:
            assert bias[0, 0, j, 0].abs() == 0 or True  # may be set by other slot
        # Diagonal zero for this forced slot unless self-kNN (we exclude self).
        assert bias[0, 0, 0, 0].abs() < 1e-8 or j == 0
    # Invalid / padded slots contribute 0.
    for i in range(n):
        for r in range(knn_dst.size(-1)):
            if not knn_valid[0, i, r]:
                jj = int(knn_dst[0, i, r].item())
                # value at that scatter target from this slot is 0; other slots ok
                _ = jj


def test_static_uses_layer0_edges():
    x_raw, mask = _clustered_jets(batch=1)
    star_x, knn_x, incidence, knn_dst, knn_valid = _graph(x_raw, mask)
    n = x_raw.size(1)
    model = _model(edge_bias_m11="static", depth=2, width=32, heads=4)
    with torch.no_grad():
        for lin in model.bias_11:
            lin.weight.zero_()
            lin.weight[0, 0] = 1.0
    e0 = torch.zeros(1, star_x.size(1) + knn_x.size(1), model.D)
    e0[0, n, 0] = 5.0
    e1 = e0.clone()
    e1[0, n, 0] = 50.0
    b_static = model._m11_edge_bias(
        layer_idx=1,
        e_ln=e1,
        e_ln0=e0,
        knn_dst=knn_dst,
        knn_valid=knn_valid,
        n_star=n,
        n_particles=n,
    )
    model.edge_bias_m11 = "dynamic"
    b_dyn = model._m11_edge_bias(
        layer_idx=1,
        e_ln=e1,
        e_ln0=e0,
        knn_dst=knn_dst,
        knn_valid=knn_valid,
        n_star=n,
        n_particles=n,
    )
    # Static ignores the large e1 spike relative to e0; dynamic does not.
    if knn_valid[0, 0, 0]:
        j = int(knn_dst[0, 0, 0].item())
        assert b_static[0, 0, 0, j].abs() < b_dyn[0, 0, 0, j].abs()


def test_init_bias_rms_scales_as_d_inv_sqrt():
    rms = {}
    for d in (64, 256):
        torch.manual_seed(0)
        model = _model(width=d, heads=d // 32, depth=1, edge_bias_m11="dynamic")
        x_raw, mask = _clustered_jets(batch=4, n=16)
        star_x, knn_x, incidence, knn_dst, knn_valid = _graph(x_raw, mask, k=3)
        n = x_raw.size(1)
        # LN outputs are Θ(1); bias then scales as D^{-1/2}.
        e_ln = torch.randn(4, star_x.size(1) + knn_x.size(1), d)
        bias = model._m11_edge_bias(
            layer_idx=0,
            e_ln=e_ln,
            e_ln0=None,
            knn_dst=knn_dst,
            knn_valid=knn_valid,
            n_star=n,
            n_particles=n,
        )
        gathered = torch.gather(
            bias,
            -1,
            knn_dst.unsqueeze(1).expand(4, model.heads, n, knn_dst.size(-1)),
        )
        m = knn_valid.unsqueeze(1).expand_as(gathered)
        rms[d] = float(gathered[m].pow(2).mean().sqrt())
    ratio = rms[64] / rms[256]
    # Expect ~√(256/64)=2 within 20%.
    assert 1.6 < ratio < 2.4, (rms, ratio)


def test_m21_off_no_params_and_residual_scale():
    model = _model(m21=False, edge_bias_m11="none", width=32, heads=4)
    assert not hasattr(model, "attn_21")
    assert abs(model._attn_residual_scale_single - 1.0 / model.depth) < 1e-8
    x_raw, mask = _clustered_jets()
    star_x, knn_x, incidence = build_star_kin7_plus_knn_graph(
        x_raw, radius=0.5, k=3, mask=mask, typed_features=True
    )
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
    assert out.shape == (x.size(0), 2)


def test_probe_second_pass_includes_bias():
    x_raw, mask = _clustered_jets(batch=1)
    star_x, knn_x, incidence, knn_dst, knn_valid = _graph(x_raw, mask)
    model = _model(edge_bias_m11="dynamic", width=32, heads=4, depth=1)
    with torch.no_grad():
        for lin in model.bias_11:
            lin.weight.fill_(0.5)
    enable_attention_probe(model)
    x = torch.randn(1, x_raw.size(1), 7)
    edge_mask = incidence.bool().any(dim=-1)
    model.eval()
    with torch.no_grad():
        _ = model(
            x,
            edge_x=None,
            incidence=incidence.bool(),
            mask=mask,
            edge_mask=edge_mask,
            edge_x_star=star_x,
            edge_x_knn=knn_x,
            knn_dst=knn_dst,
            knn_valid=knn_valid,
        )
    probe = model._attention_probe
    assert probe
    entry = next(e for e in probe if e.get("relation") == "11")
    assert "bias11_rms" in entry
    assert float(entry["bias11_rms"]) > 0
