"""Tests for axial (η, φ)-RoPE on CAPEN-Llama-att relation 11."""

from __future__ import annotations

import argparse

import pytest
import torch

from cpen.models.capen_llama import CAPENLlama
from cpen.models.capen_llama_att import CAPENLlamaAtt
from cpen.models.etaphi_rope import EtaPhiRoPE
from cpen.utils.part_kin import jet_centered_deta_dphi


def _toy_jet_batch(batch: int = 2, n: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.zeros(batch, n, 4)
    mask = torch.zeros(batch, n, dtype=torch.bool)
    for b in range(batch):
        n_active = 6 if b == 0 else 5
        mask[b, :n_active] = True
        for i in range(n_active):
            phi = 0.12 * i + 0.05 * b
            pt = 15.0 + i
            x[b, i, 1] = pt * torch.cos(torch.tensor(phi))
            x[b, i, 2] = pt * torch.sin(torch.tensor(phi))
            x[b, i, 3] = 4.0 + 0.2 * i
            x[b, i, 0] = torch.sqrt(
                x[b, i, 1] ** 2 + x[b, i, 2] ** 2 + x[b, i, 3] ** 2 + 0.25
            )
    return x, mask


def test_etaphi_rope_shapes_and_norms() -> None:
    rope = EtaPhiRoPE(head_dim=8, num_heads=2, theta=100.0)
    q = torch.randn(3, 5, 2, 8)
    k = torch.randn(3, 6, 2, 8)
    rt = torch.randn(3, 5, 2)
    rs = torch.randn(3, 6, 2)
    q_out, k_out = rope(q, k, rt, rs)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert torch.allclose(q_out.norm(dim=-1), q.norm(dim=-1), atol=1e-5)
    assert torch.allclose(k_out.norm(dim=-1), k.norm(dim=-1), atol=1e-5)


def test_etaphi_rope_zero_coordinates_noop() -> None:
    rope = EtaPhiRoPE(head_dim=8, num_heads=2)
    q = torch.randn(4, 2, 8)
    k = torch.randn(5, 2, 8)
    q_out, k_out = rope(q, k, torch.zeros(4, 2), torch.zeros(5, 2))
    assert torch.allclose(q_out, q, atol=1e-5)
    assert torch.allclose(k_out, k, atol=1e-5)


def test_etaphi_rope_common_translation_invariance() -> None:
    rope = EtaPhiRoPE(head_dim=8, num_heads=2, theta=50.0)
    q = torch.randn(4, 2, 8)
    k = torch.randn(5, 2, 8)
    rt = torch.randn(4, 2)
    rs = torch.randn(5, 2)
    q1, k1 = rope(q, k, rt, rs)
    shift = torch.tensor([0.3, -0.2])
    q2, k2 = rope(q, k, rt + shift, rs + shift)
    dots1 = torch.einsum("ihd,jhd->hij", q1, k1)
    dots2 = torch.einsum("ihd,jhd->hij", q2, k2)
    assert torch.allclose(dots1, dots2, atol=1e-5)


def test_etaphi_rope_rejects_odd_head_dim() -> None:
    with pytest.raises(ValueError, match="even head_dim"):
        EtaPhiRoPE(head_dim=5, num_heads=1)


def test_base_llama_rejects_use_rope() -> None:
    with pytest.raises(ValueError, match="CAPEN-Llama-att"):
        CAPENLlama(
            n_features=7,
            n_edge_features=4,
            out_dim=2,
            depth=1,
            width=16,
            heads=2,
            use_rope=True,
        )


def test_wire_and_rope_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        CAPENLlamaAtt(
            n_features=7,
            n_edge_features=4,
            out_dim=2,
            depth=1,
            width=16,
            heads=2,
            use_wire=True,
            use_rope=True,
        )


def test_capen_llama_att_rope_forward() -> None:
    torch.manual_seed(0)
    model = CAPENLlamaAtt(
        n_features=7,
        n_edge_features=4,
        out_dim=2,
        depth=2,
        width=16,
        heads=4,
        use_rope=True,
        readout_mode="graph",
        n_class_tokens=2,
    )
    assert model.use_rope is True
    assert model.rope_11 is not None
    assert len(model.rope_11) == 2
    x_raw, mask = _toy_jet_batch()
    n = x_raw.size(1)
    x = torch.randn(x_raw.size(0), n, 7)
    m = 3
    edge_x = torch.randn(x_raw.size(0), m, 4)
    incidence = torch.zeros(x_raw.size(0), m, n, dtype=torch.bool)
    incidence[:, 0, :4] = True
    incidence[:, 1, :2] = True
    incidence[:, 2, :5] = True
    out = model(x, edge_x, incidence, mask=mask, x_raw=x_raw)
    assert out.shape == (x_raw.size(0), 2)
    assert torch.isfinite(out).all()


def test_rope_params_are_used_in_loss() -> None:
    torch.manual_seed(0)
    model = CAPENLlamaAtt(
        n_features=7,
        n_edge_features=4,
        out_dim=2,
        depth=1,
        width=16,
        heads=4,
        use_rope=True,
        n_class_tokens=2,
    )
    x_raw, mask = _toy_jet_batch(batch=1, n=6)
    x = torch.randn(1, 6, 7)
    edge_x = torch.randn(1, 2, 4)
    incidence = torch.zeros(1, 2, 6, dtype=torch.bool)
    incidence[0, 0, :4] = True
    incidence[0, 1, :3] = True
    model.train()
    logits = model(x, edge_x, incidence, mask=mask, x_raw=x_raw)
    logits.sum().backward()
    unused = [n for n, p in model.named_parameters() if p.grad is None]
    assert unused == [], f"unused parameters: {unused}"


def test_live_star_batch_attaches_rope_coordinates() -> None:
    from cpen.lit_models.lit_cpen_jetclass import LitCPENJetClass

    model = CAPENLlamaAtt(
        n_features=7,
        n_edge_features=4,
        out_dim=2,
        depth=1,
        width=16,
        heads=4,
        use_rope=True,
        n_class_tokens=2,
    )
    lit = LitCPENJetClass(
        model=model,
        model_name="capen-llama-att",
        eta_0=0.1,
        live_star_radius=0.2,
        live_edge_features="logdot-dp",
    )
    x_raw, mask = _toy_jet_batch()
    batch = {
        "x": torch.randn(x_raw.size(0), x_raw.size(1), 7),
        "x_raw": x_raw,
        "mask": mask,
        "y": torch.zeros(x_raw.size(0), dtype=torch.long),
    }
    out = lit.on_after_batch_transfer(batch, 0)
    assert "x_raw" not in out
    assert "rope_coordinates" in out
    expected = jet_centered_deta_dphi(x_raw, mask)
    torch.testing.assert_close(out["rope_coordinates"], expected, atol=1e-5, rtol=0)
    assert out["rope_coordinates"].shape[-1] == 2


def test_cli_use_rope_tags_and_builds() -> None:
    from scans.common.sweep_common import (
        add_common_args,
        build_model,
        configure_graph_args,
        run_options_from_args,
    )

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(
        [
            "--model",
            "capen-llama-att",
            "--mode",
            "sweep_lr",
            "--operators",
            "incidence",
            "--star-radius",
            "0.2",
            "--etas",
            "0.25",
            "--data-root",
            "/tmp",
            "--root",
            "/tmp",
            "--normalization",
            "uniform",
            "--optimizer",
            "adam",
            "--use-rope",
        ]
    )
    configure_graph_args(args)
    tag = run_options_from_args(args).extra_tag or ""
    assert "rope" in tag.split("_")
    model = build_model(
        args, n_features=7, n_edge_features=4, out_dim=2, depth=1, width=16, heads=2
    )
    assert isinstance(model, CAPENLlamaAtt)
    assert model.use_rope is True
    assert model.rope_11 is not None


def test_cli_rejects_rope_plus_wire() -> None:
    from scans.common.sweep_common import add_common_args, configure_graph_args

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(
        [
            "--model",
            "capen-llama-att",
            "--mode",
            "sweep_lr",
            "--operators",
            "incidence",
            "--star-radius",
            "0.2",
            "--etas",
            "0.1",
            "--data-root",
            "/tmp",
            "--root",
            "/tmp",
            "--use-rope",
            "--use-wire",
        ]
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        configure_graph_args(args)


def test_cli_rejects_rope_on_cpen() -> None:
    from scans.common.sweep_common import add_common_args, configure_graph_args

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(
        [
            "--model",
            "cpen",
            "--mode",
            "sweep_lr",
            "--operators",
            "incidence",
            "--star-radius",
            "0.2",
            "--etas",
            "0.1",
            "--data-root",
            "/tmp",
            "--root",
            "/tmp",
            "--use-rope",
        ]
    )
    with pytest.raises(ValueError, match="capen-llama-att"):
        configure_graph_args(args)
