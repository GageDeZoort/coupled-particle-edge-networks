from types import SimpleNamespace

import torch

from cpen.apps.mnist.lit_cpen_mnist import LitCPENMNIST
from cpen.apps.mnist.mnist_graph_cache import graph_to_cache_row
from cpen.models.cpen import CPEN


def test_mnist_row_is_graph_label_and_two_node_incidence():
    data = SimpleNamespace(
        x=torch.tensor([[0.2], [0.8], [0.1]]),
        pos=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        y=torch.tensor([7]),
    )
    row = graph_to_cache_row(data)
    assert int(row["y"]) == 7
    assert row["y"].ndim == 0
    assert int(row["n_edges"]) == 2
    assert int(row["incidence_nnz"]) == 4
    assert row["x"].shape == (3, 3)
    assert row["edge_x"].shape == (2, 2)
    assert torch.allclose(row["x"].square().sum(dim=1), torch.full((3,), 3.0))
    assert torch.allclose(row["edge_x"].square().sum(dim=1), torch.full((2,), 2.0))


def _mnist_batch() -> dict[str, torch.Tensor]:
    return {
        "x": torch.randn(2, 3, 3),
        "edge_x": torch.randn(2, 2, 2),
        "y": torch.tensor([1, 4]),
        "mask": torch.ones(2, 3, dtype=torch.bool),
        "edge_mask": torch.ones(2, 2, dtype=torch.bool),
        "incidence_node": torch.tensor([[0, 1, 1, 2], [0, 1, 1, 2]], dtype=torch.int16),
        "incidence_edge": torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.int16),
        "incidence_nnz": torch.tensor([4, 4], dtype=torch.int32),
        "node_degree_inv": torch.tensor([[1.0, 0.5, 1.0], [1.0, 0.5, 1.0]]),
        "edge_degree_inv": torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
    }


def _make_cpen(*, x_only: bool) -> CPEN:
    model = CPEN(
        n_features=3,
        n_edge_features=2,
        out_dim=10,
        depth=1,
        width=16,
        operators="incidence",
        operator_backend="sparse",
        readout_mode="graph",
    )
    model.x_only_graph_readout = x_only
    return model


def test_mnist_node_readout_pools_to_graph_logits():
    model = _make_cpen(x_only=True)
    batch = _mnist_batch()
    logits = model(
        batch["x"],
        batch["edge_x"],
        incidence_node=batch["incidence_node"],
        incidence_edge=batch["incidence_edge"],
        incidence_nnz=batch["incidence_nnz"],
        node_degree_inv=batch["node_degree_inv"],
        edge_degree_inv=batch["edge_degree_inv"],
        mask=batch["mask"],
        edge_mask=batch["edge_mask"],
    )
    assert logits.shape == (2, 10)
    logits.sum().backward()
    assert model.decoder_e.weight.grad is not None


def test_mnist_node_plus_edge_differs_from_x_only():
    torch.manual_seed(0)
    x_only = _make_cpen(x_only=True)
    mixed = _make_cpen(x_only=False)
    mixed.load_state_dict(x_only.state_dict())
    batch = _mnist_batch()
    kwargs = dict(
        incidence_node=batch["incidence_node"],
        incidence_edge=batch["incidence_edge"],
        incidence_nnz=batch["incidence_nnz"],
        node_degree_inv=batch["node_degree_inv"],
        edge_degree_inv=batch["edge_degree_inv"],
        mask=batch["mask"],
        edge_mask=batch["edge_mask"],
    )
    a = x_only(batch["x"], batch["edge_x"], **kwargs)
    b = mixed(batch["x"], batch["edge_x"], **kwargs)
    assert a.shape == b.shape == (2, 10)
    assert not torch.allclose(a, b)


def test_lit_mnist_sets_graph_readout_flag():
    model = _make_cpen(x_only=False)
    lit = LitCPENMNIST(model, eta_0=0.25, readout_mode="node")
    assert lit.model.x_only_graph_readout is True
    lit_mix = LitCPENMNIST(_make_cpen(x_only=True), eta_0=0.25, readout_mode="node+edge")
    assert lit_mix.model.x_only_graph_readout is False


def test_mnist_attn_readout_shape_and_padding():
    torch.manual_seed(0)
    model = CPEN(
        n_features=3,
        n_edge_features=2,
        out_dim=10,
        depth=1,
        width=16,
        operators="incidence",
        operator_backend="sparse",
        readout_mode="attn",
        output_attention_heads=4,
    )
    assert model.output_attention_pool is not None
    batch = _mnist_batch()
    kwargs = dict(
        incidence_node=batch["incidence_node"],
        incidence_edge=batch["incidence_edge"],
        incidence_nnz=batch["incidence_nnz"],
        node_degree_inv=batch["node_degree_inv"],
        edge_degree_inv=batch["edge_degree_inv"],
        mask=batch["mask"],
        edge_mask=batch["edge_mask"],
    )
    logits = model(batch["x"], batch["edge_x"], **kwargs)
    assert logits.shape == (2, 10)
    logits.sum().backward()
    assert model.output_attention_pool.class_unembed.grad is not None
    assert model.decoder_x.weight.grad is not None

    model.zero_grad()
    with torch.no_grad():
        base = model(batch["x"], batch["edge_x"], **kwargs)
        x_pad = torch.cat([batch["x"], torch.randn(2, 1, 3)], dim=1)
        mask_pad = torch.cat(
            [batch["mask"], torch.zeros(2, 1, dtype=torch.bool)], dim=1
        )
        deg_pad = torch.cat([batch["node_degree_inv"], torch.ones(2, 1)], dim=1)
    padded = model(
        x_pad,
        batch["edge_x"],
        incidence_node=batch["incidence_node"],
        incidence_edge=batch["incidence_edge"],
        incidence_nnz=batch["incidence_nnz"],
        node_degree_inv=deg_pad,
        edge_degree_inv=batch["edge_degree_inv"],
        mask=mask_pad,
        edge_mask=batch["edge_mask"],
    )
    assert torch.allclose(base, padded, atol=1e-5)


def test_lit_mnist_attn_keeps_pool():
    model = CPEN(
        n_features=3,
        n_edge_features=2,
        out_dim=10,
        depth=1,
        width=16,
        operators="incidence",
        operator_backend="sparse",
        readout_mode="attn",
        output_attention_heads=4,
    )
    lit = LitCPENMNIST(model, eta_0=0.25, readout_mode="attn")
    assert lit.model.output_attention_pool is not None
    assert lit.model.x_only_graph_readout is None


def test_cli_accepts_mnist_attn():
    import argparse

    from scans.common.sweep_common import add_common_args

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(
        [
            "--dataset",
            "mnist-sp",
            "--model",
            "cpen",
            "--readout-mode",
            "attn",
            "--mode",
            "sweep_lr",
        ]
    )
    assert args.readout_mode == "attn"


def test_cli_accepts_mnist_sp():
    import argparse

    from scans.common.sweep_common import add_common_args

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(
        [
            "--dataset",
            "mnist-sp",
            "--model",
            "cpen",
            "--readout-mode",
            "node+edge",
            "--mode",
            "sweep_lr",
        ]
    )
    assert args.dataset == "mnist-sp"
    assert args.readout_mode == "node+edge"
