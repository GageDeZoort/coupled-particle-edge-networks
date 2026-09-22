"""Tests for CAPEN-Llama-att (class-token MHSA readout + hyperedge-only M22)."""

from __future__ import annotations

import argparse

import pytest
import torch

from cpen.models.capen import CAPEN
from cpen.models.capen_llama import CAPENLlama
from cpen.models.capen_llama_att import BaselineTransformer, CAPENLlamaAtt


def small_hier_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    n, m = 5, 4
    x = torch.randn(1, n, 7)
    edge_x = torch.randn(1, m, 4)
    incidence = torch.zeros(1, m, n, dtype=torch.bool)
    incidence[0, 0, :3] = True  # hyper
    incidence[0, 1, :2] = True  # 2-edge
    incidence[0, 2, 0] = True  # singleton / pad-like
    incidence[0, 3, :4] = True  # hyper
    mask = torch.ones(1, n, dtype=torch.bool)
    edge_mask = torch.tensor([[True, True, True, True]])
    return {
        "x": x,
        "edge_x": edge_x,
        "incidence": incidence,
        "mask": mask,
        "edge_mask": edge_mask,
        "node_mask": mask,
    }


def make_model(**overrides) -> CAPENLlamaAtt:
    kwargs = dict(
        n_features=7,
        n_edge_features=4,
        out_dim=2,
        depth=2,
        width=16,
        heads=4,
        readout_mode="graph",
        hyperedge_m22_only=True,
        n_class_tokens=2,
    )
    kwargs.update(overrides)
    torch.manual_seed(1)
    return CAPENLlamaAtt(**kwargs)


def test_is_capen_llama_subclass() -> None:
    assert issubclass(CAPENLlamaAtt, CAPENLlama)
    assert issubclass(CAPENLlamaAtt, CAPEN)
    assert isinstance(make_model(), CAPEN)


def test_writeup_pooling_geometry() -> None:
    model = make_model()
    pool = model.output_pool
    d = int(model.D)
    assert pool.attn_x.heads == 1
    assert pool.attn_e.heads == 1
    assert pool.attn_x._logit_scale == pytest.approx(1.0 / d)
    assert pool.attn_x._proj_scale == pytest.approx(d**-0.5)
    assert pool.class_unembed.shape == (2, d)
    assert pool._fuse_scale == pytest.approx(0.5**0.5)
    assert pool._readout_scale == pytest.approx(d**-0.5)


def test_independent_pools_are_not_joint_softmax() -> None:
    """Node and edge softmaxes are separate, so extra edges cannot steal node mass."""
    torch.manual_seed(0)
    pool = make_model().output_pool
    h_x = torch.randn(1, 3, 16)
    h_e = torch.randn(1, 2, 16)
    mask = torch.ones(1, 3, dtype=torch.bool)
    edge_mask = torch.ones(1, 2, dtype=torch.bool)
    base = pool(h_x, h_e, mask=mask, edge_mask=edge_mask)
    wider = pool(
        h_x,
        torch.cat([h_e, torch.randn(1, 6, 16)], dim=1),
        mask=mask,
        edge_mask=torch.cat([edge_mask, torch.ones(1, 6, dtype=torch.bool)], dim=1),
    )
    assert not torch.allclose(base, wider, atol=1e-6)
    # Padding-only extra edges must not change logits.
    padded = pool(
        h_x,
        torch.cat([h_e, torch.randn(1, 6, 16)], dim=1),
        mask=mask,
        edge_mask=torch.cat([edge_mask, torch.zeros(1, 6, dtype=torch.bool)], dim=1),
    )
    torch.testing.assert_close(base, padded, atol=1e-5, rtol=1e-5)


def test_graph_mode_output_shape() -> None:
    model = make_model()
    batch = small_hier_batch()
    out = model(
        batch["x"],
        batch["edge_x"],
        batch["incidence"],
        mask=batch["mask"],
        edge_mask=batch["edge_mask"],
        node_mask=batch["node_mask"],
    )
    assert out.shape == (1, 2)
    assert torch.isfinite(out).all()


def test_m22_only_among_hyperedges() -> None:
    batch = small_hier_batch()
    model = make_model()
    model._pending_edge_mask = batch["edge_mask"]
    _, _, _, m_22 = model._build_supports(batch["incidence"], mask=batch["mask"])
    # Placeholder support is identity; hyper mask is stashed for gather-attend.
    assert bool(m_22[0].diag().all())
    assert model._pending_is_hyper is not None
    assert bool(model._pending_is_hyper[0, 0])  # hyper
    assert bool(model._pending_is_hyper[0, 3])  # hyper
    assert not bool(model._pending_is_hyper[0, 1])  # 2-edge
    assert not bool(model._pending_is_hyper[0, 2])  # singleton

    # Gather path: hypers attend; pairwise slots stay identity (equal to LN target
    # when residual is applied outside — here we check _run_attn output).
    e_ln = torch.randn(1, 4, 16)
    out = model._run_attn_22_hyper_subset(model.attn_22[0], e_ln, e_ln)
    assert out.shape == e_ln.shape
    # Non-hypers unchanged (identity).
    assert torch.allclose(out[0, 1], e_ln[0, 1])
    assert torch.allclose(out[0, 2], e_ln[0, 2])
    # Hypers generally move (unless attention happens to be identity).
    assert torch.isfinite(out).all()


def test_default_capen_llama_unchanged_readout() -> None:
    """CAPEN-Llama keeps energy/incidence pooling (not class tokens)."""
    assert not hasattr(CAPENLlama, "class_tokens") or not issubclass(
        CAPENLlama, CAPENLlamaAtt
    )
    llama = CAPENLlama(
        n_features=7, n_edge_features=4, out_dim=2, depth=1, width=16, heads=2
    )
    assert not hasattr(llama, "readout_attn")


def test_all_parameters_used_in_loss() -> None:
    """DDP find_unused_parameters=False requires every param in the loss graph."""
    model = make_model()
    batch = small_hier_batch()
    model.train()
    logits = model(
        batch["x"],
        batch["edge_x"],
        batch["incidence"],
        mask=batch["mask"],
        edge_mask=batch["edge_mask"],
        node_mask=batch["node_mask"],
    )
    loss = logits.sum()
    loss.backward()
    unused = [n for n, p in model.named_parameters() if p.grad is None]
    assert unused == [], f"unused parameters (DDP will abort): {unused}"


def test_uniform_output_pool_uses_all_parameters() -> None:
    model = make_model(uniform_output_pool=True, out_dim=2)
    assert model.uniform_output_pool is True
    assert model.output_pool is None
    batch = small_hier_batch()
    model.train()
    logits = model(
        batch["x"],
        batch["edge_x"],
        batch["incidence"],
        mask=batch["mask"],
        edge_mask=batch["edge_mask"],
        node_mask=batch["node_mask"],
    )
    assert logits.shape == (1, 2)
    loss = logits.sum()
    loss.backward()
    unused = [n for n, p in model.named_parameters() if p.grad is None]
    assert unused == [], f"unused parameters (DDP will abort): {unused}"


def test_all_edge_m22_keeps_parent_line_graph() -> None:
    model = make_model(hyperedge_m22_only=False)
    batch = small_hier_batch()
    s = batch["incidence"]
    model._pending_edge_mask = batch["edge_mask"]
    _m11, _m21, _m12, m_22 = model._build_supports(s, mask=batch["mask"])
    assert model._pending_is_hyper is None
    assert m_22.shape[-1] == s.size(1)
    assert bool((m_22[0] != torch.eye(s.size(1), dtype=torch.bool)).any())


def test_cli_tags_fullm22_and_unipool() -> None:
    from scans.common.sweep_common import add_common_args, run_options_from_args

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
            "--all-edge-m22",
            "--uniform-output-pool",
            "--etas",
            "0.25",
            "--root",
            "/tmp/x",
            "--data-root",
            "/tmp",
        ]
    )
    tag = run_options_from_args(args).extra_tag
    assert "fullm22" in tag
    assert "unipool" in tag


def test_registry_builds_capen_llama_att() -> None:
    from scans.common.sweep_common import add_common_args, build_model, configure_graph_args

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
            "--hier-k",
            "8",
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
        ]
    )
    configure_graph_args(args)
    assert args.graph_construction.startswith("hier_k8_")
    model = build_model(
        args, n_features=7, n_edge_features=4, out_dim=2, depth=2, width=16, heads=2
    )
    assert isinstance(model, CAPENLlamaAtt)
    assert model.ignore_knn_edges is False


def test_hyperedge_only_sets_ignore_knn_edges() -> None:
    from scans.common.sweep_common import add_common_args, build_model, configure_graph_args

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
            "--hier-k",
            "8",
            "--hyperedge-only",
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
        ]
    )
    configure_graph_args(args)
    assert args.hyperedge_only is True
    model = build_model(
        args, n_features=7, n_edge_features=4, out_dim=2, depth=2, width=16, heads=2
    )
    assert isinstance(model, CAPENLlamaAtt)
    assert model.ignore_knn_edges is True


def test_build_model_gamma_explicit_values() -> None:
    from scans.common.sweep_common import add_common_args, build_model, configure_graph_args

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
            "--hier-k",
            "8",
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
            "--attention-normalization",
            "gamma",
            "--no-estimate-gamma-rs",
            "--gamma-11",
            "10",
            "--gamma-12",
            "2",
            "--gamma-21",
            "3",
            "--gamma-22",
            "4",
        ]
    )
    configure_graph_args(args)
    model = build_model(
        args, n_features=7, n_edge_features=4, out_dim=2, depth=2, width=16, heads=2
    )
    assert isinstance(model, CAPENLlamaAtt)


def test_build_model_gamma_missing_is_not_nonetype() -> None:
    from scans.common.sweep_common import add_common_args, build_model, configure_graph_args

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
            "--hier-k",
            "8",
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
            "--attention-normalization",
            "gamma",
            "--no-estimate-gamma-rs",
        ]
    )
    configure_graph_args(args)
    with pytest.raises((ValueError, FileNotFoundError)):
        build_model(
            args, n_features=7, n_edge_features=4, out_dim=2, depth=2, width=16, heads=2
        )


def make_baseline(**overrides) -> BaselineTransformer:
    kwargs = dict(
        n_features=7,
        n_edge_features=4,
        out_dim=2,
        depth=2,
        width=16,
        heads=4,
        readout_mode="graph",
        n_class_tokens=2,
    )
    kwargs.update(overrides)
    torch.manual_seed(1)
    return BaselineTransformer(**kwargs)


def test_baseline_is_capen_llama_att_subclass() -> None:
    from cpen.models.capen_llama_att import BaselineTransformer

    assert issubclass(BaselineTransformer, CAPENLlamaAtt)
    model = make_baseline()
    assert model.m11_only is True
    assert model.output_pool.node_only is True
    assert isinstance(model, CAPEN)


def test_default_att_is_not_m11_only() -> None:
    model = make_model()
    assert model.m11_only is False
    assert model.output_pool.node_only is False
    assert model.output_pool._fuse_scale == pytest.approx(0.5**0.5)


def test_baseline_pool_skips_fuse_scale() -> None:
    pool = make_baseline().output_pool
    d = 16
    assert pool.node_only is True
    assert pool.attn_x.heads == 1
    assert pool.attn_x._logit_scale == pytest.approx(1.0 / d)
    assert pool._readout_scale == pytest.approx(d**-0.5)
    torch.manual_seed(0)
    h_x = torch.randn(2, 5, d)
    mask = torch.ones(2, 5, dtype=torch.bool)
    logits = pool(h_x, mask=mask)
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()
    # Extra dummy edges must not change node-only logits.
    h_e = torch.randn(2, 7, d)
    logits_e = pool(h_x, h_e, mask=mask, edge_mask=torch.ones(2, 7, dtype=torch.bool))
    torch.testing.assert_close(logits, logits_e, atol=1e-5, rtol=1e-5)


def test_baseline_forward_without_incidence() -> None:
    model = make_baseline()
    x = torch.randn(2, 6, 7)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[1, 4:] = False
    out = model(x, mask=mask, node_mask=mask)
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()


def test_baseline_all_parameters_used_in_loss() -> None:
    model = make_baseline()
    model.train()
    x = torch.randn(1, 5, 7)
    mask = torch.ones(1, 5, dtype=torch.bool)
    loss = model(x, mask=mask).sum()
    loss.backward()
    unused = [n for n, p in model.named_parameters() if p.grad is None]
    assert unused == [], f"unused parameters (DDP will abort): {unused}"


def test_registry_builds_baseline_transformer() -> None:
    from cpen.models.capen_llama_att import BaselineTransformer
    from scans.common.sweep_common import add_common_args, build_model, configure_graph_args

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(
        [
            "--model",
            "baseline-transformer",
            "--mode",
            "sweep_lr",
            "--operators",
            "incidence",
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
            "--attention-normalization",
            "none",
        ]
    )
    configure_graph_args(args)
    model = build_model(
        args, n_features=7, n_edge_features=4, out_dim=2, depth=2, width=16, heads=2
    )
    assert isinstance(model, BaselineTransformer)
    assert model.m11_only is True
    assert model.output_pool.node_only is True


def test_baseline_stream_cli_does_not_require_star_and_tags_m11() -> None:
    from scans.common.sweep_common import (
        add_common_args,
        configure_graph_args,
        jetclass_live_graph_kwargs,
        run_options_from_args,
    )

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(
        [
            "--mode",
            "sweep_lr",
            "--etas",
            "0.1",
            "--epochs",
            "1",
            "--model",
            "baseline-transformer",
            "--operators",
            "incidence",
            "--jetclass-stream",
            "--jetclass-features",
            "kin7",
            "--num-particles",
            "80",
            "--n-train",
            "1000000",
            "--root",
            "/tmp/x",
            "--data-root",
            "/tmp",
        ]
    )
    args.dataset = "jetclass"
    configure_graph_args(args)
    assert args.star_radius is None
    assert jetclass_live_graph_kwargs(args) == {}
    tag = run_options_from_args(args).extra_tag
    assert "m11" in tag
    assert "stream" in tag
    assert "D1M" in tag
    assert "gstar" not in tag


def test_baseline_lit_attaches_dummy_edges_from_x_raw() -> None:
    from cpen.lit_models.lit_cpen_jetclass import LitCPENJetClass

    model = make_baseline()
    lit = LitCPENJetClass(model=model, model_name="baseline-transformer", eta_0=0.1)
    batch = {
        "x": torch.randn(2, 8, 7),
        "x_raw": torch.randn(2, 8, 4),
        "mask": torch.ones(2, 8, dtype=torch.bool),
        "y": torch.zeros(2, dtype=torch.long),
    }
    out = lit.on_after_batch_transfer(batch, 0)
    assert "x_raw" not in out
    assert out["edge_x"].shape[:2] == (2, 1)
    assert out["incidence"].shape == (2, 1, 8)
    assert not bool(out["incidence"].any())
    logits = lit.forward(out["x"], **lit._model_kwargs(out))
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all()

