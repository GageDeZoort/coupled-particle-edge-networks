from types import SimpleNamespace

import torch

from cpen.models.cpen import CPEN
from cpen.models.particle_only import ParticleOnlyNetwork
from cpen.utils.graphs import adjacency_from_batch
from cpen.utils.pascal_graph_cache import canonical_undirected_edges, graph_to_cache_row


def test_pascal_edges_are_deduplicated_with_aligned_features():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = torch.tensor([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]])
    pairs, features = canonical_undirected_edges(edge_index, edge_attr, num_nodes=3)
    assert torch.equal(pairs, torch.tensor([[0, 1], [1, 2]]))
    assert torch.allclose(features, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_pascal_row_is_two_node_incidence():
    data = SimpleNamespace(
        x=torch.randn(3, 14),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.randn(4, 2).repeat_interleave(1, dim=0),
        y=torch.tensor([0, 1, 2]),
    )
    # Reverse directions carry the same features in PascalVOC-SP.
    data.edge_attr[1] = data.edge_attr[0]
    data.edge_attr[3] = data.edge_attr[2]
    row = graph_to_cache_row(data, wire_coordinate_dim=None)
    assert int(row["n_edges"]) == 2
    assert int(row["incidence_nnz"]) == 4
    assert torch.equal(row["incidence_edge"], torch.tensor([0, 0, 1, 1], dtype=torch.int16))
    assert torch.allclose(row["x"].square().sum(dim=1), torch.full((3,), 14.0))
    assert torch.allclose(row["edge_x"].square().sum(dim=1), torch.full((2,), 2.0))
    assert "wire_coordinates" not in row


def test_pascal_row_default_includes_wire():
    data = SimpleNamespace(
        x=torch.randn(3, 14),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.randn(4, 2),
        y=torch.tensor([0, 1, 2]),
    )
    data.edge_attr[1] = data.edge_attr[0]
    data.edge_attr[3] = data.edge_attr[2]
    row = graph_to_cache_row(data)  # default m=8
    assert row["wire_coordinates"].shape == (3, 8)


def test_particle_only_uses_cached_incidence_for_node_logits():
    batch = {
        "x": torch.randn(1, 3, 14),
        "mask": torch.ones(1, 3, dtype=torch.bool),
        "edge_mask": torch.ones(1, 2, dtype=torch.bool),
        "incidence_node": torch.tensor([[0, 1, 1, 2]], dtype=torch.int16),
        "incidence_edge": torch.tensor([[0, 0, 1, 1]], dtype=torch.int16),
        "incidence_nnz": torch.tensor([4], dtype=torch.int32),
    }
    adjacency = adjacency_from_batch(batch)
    model = ParticleOnlyNetwork(
        n_features=14,
        out_dim=21,
        depth=2,
        width=16,
        operators="adjacency",
        readout_mode="node",
    )
    assert model(batch["x"], adjacency=adjacency).shape == (1, 3, 21)


def test_cpen_node_readout_uses_pascal_edge_stream():
    model = CPEN(
        n_features=14,
        n_edge_features=2,
        out_dim=21,
        depth=1,
        width=16,
        operators="incidence",
        operator_backend="sparse",
        readout_mode="node",
    )
    x = torch.randn(1, 3, 14)
    edge_x = torch.randn(1, 2, 2)
    incidence_kwargs = dict(
        incidence_node=torch.tensor([[0, 1, 1, 2]], dtype=torch.int16),
        incidence_edge=torch.tensor([[0, 0, 1, 1]], dtype=torch.int16),
        incidence_nnz=torch.tensor([4], dtype=torch.int32),
        node_degree_inv=torch.tensor([[1.0, 0.5, 1.0]]),
        edge_degree_inv=torch.tensor([[0.5, 0.5]]),
        mask=torch.ones(1, 3, dtype=torch.bool),
    )
    logits = model(x, edge_x, **incidence_kwargs)
    assert logits.shape == (1, 3, 21)

    logits_zero_edges = model(x, torch.zeros_like(edge_x), **incidence_kwargs)
    assert not torch.allclose(logits, logits_zero_edges)

    loss = logits.sum()
    loss.backward()
    assert model.decoder_e.weight.grad is not None
    assert model.decoder_e.weight.grad.abs().sum() > 0


def _pascal_sparse_batch() -> dict:
    """3 nodes, 2 edges: (0,1) interior, (1,2) boundary."""
    return {
        "x": torch.randn(1, 3, 14),
        "edge_x": torch.randn(1, 2, 2),
        "y": torch.tensor([[0, 0, 1]]),
        "mask": torch.ones(1, 3, dtype=torch.bool),
        "n_edges": torch.tensor([2], dtype=torch.int32),
        "incidence_node": torch.tensor([[0, 1, 1, 2]], dtype=torch.int16),
        "incidence_edge": torch.tensor([[0, 0, 1, 1]], dtype=torch.int16),
        "incidence_nnz": torch.tensor([4], dtype=torch.int32),
        "node_degree_inv": torch.tensor([[1.0, 0.5, 1.0]]),
        "edge_degree_inv": torch.tensor([[0.5, 0.5]]),
    }


def test_boundary_edge_targets_from_incidence():
    from cpen.apps.pascal.lit_cpen_pascal import boundary_edge_targets

    edge_y, valid = boundary_edge_targets(_pascal_sparse_batch())
    assert torch.equal(edge_y, torch.tensor([[0, 1]]))
    assert torch.equal(valid, torch.tensor([[True, True]]))


def test_lit_pascal_node_plus_edge_trains_boundary_decoder():
    from cpen.apps.pascal.lit_cpen_pascal import LitCPENPascal

    model = CPEN(
        n_features=14,
        n_edge_features=2,
        out_dim=21,
        depth=1,
        width=16,
        operators="incidence",
        operator_backend="sparse",
        readout_mode="node+edge",
    )
    batch = _pascal_sparse_batch()
    logits = model(
        batch["x"],
        batch["edge_x"],
        incidence_node=batch["incidence_node"],
        incidence_edge=batch["incidence_edge"],
        incidence_nnz=batch["incidence_nnz"],
        node_degree_inv=batch["node_degree_inv"],
        edge_degree_inv=batch["edge_degree_inv"],
        mask=batch["mask"],
    )
    assert isinstance(logits, tuple) and len(logits) == 2
    assert logits[0].shape == (1, 3, 21)
    assert logits[1].shape == (1, 2, 21)

    lit = LitCPENPascal(
        model,
        eta_0=0.25,
        edge_aux="hard-ce",
        edge_loss_weight=1.0,
    )
    assert lit._use_edge_readout
    assert lit.model.decoder_e.out_features == 2
    assert lit.model.x_only_node_readout is True
    node_logits, edge_logits = lit._unpack_node_logits(
        lit.forward(batch["x"], **lit._model_kwargs(batch))
    )
    assert node_logits.shape == (1, 3, 21)
    assert edge_logits is not None and edge_logits.shape == (1, 2, 2)
    loss = lit._shared_step(batch, "train")
    assert torch.isfinite(loss)
    loss.backward()
    assert lit.model.decoder_e.weight.grad is not None
    assert lit.model.decoder_e.weight.grad.abs().sum() > 0
    assert lit.model.decoder_x.weight.grad is not None


def test_lit_pascal_node_only_skips_edge_aux():
    from cpen.apps.pascal.lit_cpen_pascal import LitCPENPascal

    model = CPEN(
        n_features=14,
        n_edge_features=2,
        out_dim=21,
        depth=1,
        width=16,
        operators="incidence",
        operator_backend="sparse",
        readout_mode="node",
    )
    lit = LitCPENPascal(model, eta_0=0.25, edge_aux="hard-ce")
    assert not lit._use_edge_readout
    assert lit.model.decoder_e.out_features == 21
    assert lit.model.x_only_node_readout is True
    loss = lit._shared_step(_pascal_sparse_batch(), "train")
    assert torch.isfinite(loss)
    loss.backward()
    assert lit.model.decoder_x.weight.grad is not None
    assert lit.model.decoder_x.weight.grad.abs().sum() > 0


def test_pascal_cli_readout_defaults_and_checkpoint():
    import argparse

    from scans.common.sweep_common import (
        _resolve_checkpoint_settings,
        add_common_args,
        configure_graph_args,
    )

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(
        [
            "--dataset",
            "pascalvoc-sp",
            "--model",
            "cpen",
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
            "adamw",
        ]
    )
    configure_graph_args(args)
    assert args.readout_mode == "node"
    assert hasattr(args, "seed")
    monitor, mode = _resolve_checkpoint_settings(args)
    assert (monitor, mode) == ("val_f1", "max")

    args_edge = parser.parse_args(
        [
            "--dataset",
            "pascalvoc-sp",
            "--model",
            "cpen",
            "--mode",
            "sweep_lr",
            "--operators",
            "incidence",
            "--readout-mode",
            "node+edge",
            "--etas",
            "0.25",
            "--data-root",
            "/tmp",
            "--root",
            "/tmp",
            "--normalization",
            "uniform",
            "--optimizer",
            "adamw",
        ]
    )
    configure_graph_args(args_edge)
    assert args_edge.readout_mode == "node+edge"

    from scans.common.sweep_common import run_options_from_args

    opts = run_options_from_args(args)
    assert f"seed{int(args.seed)}" in str(opts.extra_tag)
    args.seed = 7
    opts7 = run_options_from_args(args)
    assert "seed7" in str(opts7.extra_tag)
    assert "edgeboundary" not in str(opts7.extra_tag)

    opts_e = run_options_from_args(args_edge)
    tag = str(opts_e.extra_tag)
    assert "edgeboundary" in tag
    assert f"seed{int(args_edge.seed)}" in tag


def test_lit_pascal_capen_llama_node_plus_edge():
    from cpen.apps.pascal.lit_cpen_pascal import LitCPENPascal
    from cpen.models.capen_llama import CAPENLlama

    model = CAPENLlama(
        n_features=14,
        n_edge_features=2,
        out_dim=21,
        depth=1,
        width=16,
        heads=4,
        readout_mode="node+edge",
        use_wire=False,
        all_to_all_particle_attention=False,
        incidence_m22=True,
    )
    batch = _pascal_sparse_batch()
    batch["incidence"] = torch.zeros(1, 2, 3, dtype=torch.bool)
    batch["incidence"][0, 0, 0] = batch["incidence"][0, 0, 1] = True
    batch["incidence"][0, 1, 1] = batch["incidence"][0, 1, 2] = True

    lit = LitCPENPascal(
        model, eta_0=0.25, edge_aux="hard-ce", edge_loss_weight=1.0
    )
    assert lit._use_edge_readout
    assert lit.model.x_only_node_readout is True
    assert lit.model.decoder_e.out_features == 2
    node_logits, edge_logits = lit._unpack_node_logits(
        lit.forward(batch["x"], **lit._model_kwargs(batch))
    )
    assert node_logits.shape == (1, 3, 21)
    assert edge_logits is not None and edge_logits.shape == (1, 2, 2)
    loss = lit._shared_step(batch, "train")
    assert torch.isfinite(loss)
    loss.backward()
    assert lit.model.decoder_e.weight.grad is not None
    assert lit.model.decoder_x.weight.grad is not None

