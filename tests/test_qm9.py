from types import SimpleNamespace

import pytest
import torch

from cpen.apps.qm9.lit_cpen_qm9 import LitCPENQM9
from cpen.apps.qm9.qm9_graph_cache import (
    N_TRAIN,
    N_VAL,
    QM9_TARGETS,
    feature_stats_from_rows,
    graph_to_cache_row,
    resolve_target,
    split_indices,
)
from cpen.models.cpen import CPEN


def _molecule() -> SimpleNamespace:
    """Three atoms, two bonds, targets filled with distinguishable values."""
    return SimpleNamespace(
        x=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0, 2.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        ),
        pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 3.0, 0.0]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        edge_attr=torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ),
        y=torch.arange(19, dtype=torch.float32).unsqueeze(0),
        name="test_mol",
    )


def test_qm9_target_table_matches_pyg_units():
    """Guard the upstream mislabeling: "dipole" there is index 4, the gap."""
    assert QM9_TARGETS["mu"] == (0, "D")
    assert QM9_TARGETS["gap"] == (4, "eV")
    assert QM9_TARGETS["u0"] == (7, "eV")
    assert resolve_target(None) == ("mu", 0, "D")
    assert resolve_target("GAP") == ("gap", 4, "eV")
    with pytest.raises(ValueError):
        resolve_target("dipole")


def test_qm9_row_keeps_all_targets_and_two_node_incidence():
    row = graph_to_cache_row(_molecule())
    assert row["y"].shape == (19,)
    assert float(row["y"][0]) == 0.0
    assert int(row["n_nodes"]) == 3
    assert int(row["n_edges"]) == 2
    assert int(row["incidence_nnz"]) == 4
    assert row["x"].shape == (3, 11)
    # 4-d bond-type one-hot + interatomic distance.
    assert row["edge_x"].shape == (2, 5)
    assert torch.equal(row["incidence_node"].to(torch.long), torch.tensor([0, 1, 1, 2]))
    assert torch.equal(row["incidence_edge"].to(torch.long), torch.tensor([0, 0, 1, 1]))
    assert torch.allclose(row["node_degree_inv"], torch.tensor([1.0, 0.5, 1.0]))
    assert torch.allclose(row["edge_degree_inv"], torch.full((2,), 0.5))


def test_qm9_edge_feature_carries_interatomic_distance():
    row = graph_to_cache_row(_molecule())
    # Bonds 0-1 and 1-2 have lengths 1 and 3 by construction.
    assert torch.allclose(row["edge_x"][:, 4], torch.tensor([1.0, 3.0]))
    assert torch.allclose(row["edge_x"][0, :4], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(row["edge_x"][1, :4], torch.tensor([0.0, 1.0, 0.0, 0.0]))


def test_qm9_feature_stats_leave_constant_columns_alone():
    rows = [graph_to_cache_row(_molecule())]
    stats = feature_stats_from_rows(rows)
    node_std = torch.tensor(stats["node_std"])
    node_mean = torch.tensor(stats["node_mean"])
    # Columns 6-8 never fire in the fixture: identity, not a blown-up scale.
    assert torch.allclose(node_std[6:9], torch.ones(3))
    assert torch.allclose(node_mean[6:9], torch.zeros(3))
    assert node_std[5] > 0.0
    # Standardizing with these stats gives unit-scale columns where defined.
    x = (rows[0]["x"] - node_mean) / node_std
    assert torch.isfinite(x).all()
    assert float(x.abs().max()) < 10.0


def test_qm9_benchmark_split_is_disjoint_and_sized():
    n_total = 130_831
    train = split_indices(n_total, "train")
    val = split_indices(n_total, "val")
    test = split_indices(n_total, "test")
    assert len(train) == N_TRAIN
    assert len(val) == N_VAL
    assert len(test) == n_total - N_TRAIN - N_VAL
    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    assert len(set(train) | set(val) | set(test)) == n_total
    # Fixed seed: same partition on a rebuild.
    assert split_indices(n_total, "val") == val


def _qm9_batch() -> dict[str, torch.Tensor]:
    return {
        "x": torch.randn(2, 3, 11),
        "edge_x": torch.randn(2, 2, 5),
        "y": torch.tensor([2.5, 3.5]),
        "mask": torch.ones(2, 3, dtype=torch.bool),
        "edge_mask": torch.ones(2, 2, dtype=torch.bool),
        "incidence_node": torch.tensor([[0, 1, 1, 2], [0, 1, 1, 2]], dtype=torch.int16),
        "incidence_edge": torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.int16),
        "incidence_nnz": torch.tensor([4, 4], dtype=torch.int32),
        "node_degree_inv": torch.tensor([[1.0, 0.5, 1.0], [1.0, 0.5, 1.0]]),
        "edge_degree_inv": torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
    }


def _make_cpen(out_dim: int = 1, *, x_only: bool | None = None) -> CPEN:
    model = CPEN(
        n_features=11,
        n_edge_features=5,
        out_dim=out_dim,
        depth=1,
        width=16,
        operators="incidence",
        operator_backend="sparse",
        readout_mode="graph",
    )
    if x_only is not None:
        model.x_only_graph_readout = x_only
    return model


def test_qm9_graph_readout_gives_one_scalar_per_molecule():
    torch.manual_seed(0)
    model = _make_cpen(x_only=False)
    batch = _qm9_batch()
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
    assert logits.shape == (2, 1)
    logits.sum().backward()
    # Both streams feed the pooled readout.
    assert model.decoder_x.weight.grad is not None
    assert model.decoder_e.weight.grad is not None


def test_qm9_pooled_readout_ignores_padded_edges():
    """Molecules have variable bond counts, so padding must not enter the mean."""
    torch.manual_seed(0)
    lit = LitCPENQM9(_make_cpen(), eta_0=0.25)
    model = lit.model
    batch = _qm9_batch()
    kwargs = dict(
        incidence_node=batch["incidence_node"],
        incidence_edge=batch["incidence_edge"],
        incidence_nnz=batch["incidence_nnz"],
        node_degree_inv=batch["node_degree_inv"],
        edge_degree_inv=batch["edge_degree_inv"],
        mask=batch["mask"],
        edge_mask=batch["edge_mask"],
    )
    with torch.no_grad():
        base = model(batch["x"], batch["edge_x"], **kwargs)
        padded = model(
            batch["x"],
            torch.cat([batch["edge_x"], torch.randn(2, 1, 5)], dim=1),
            **{
                **kwargs,
                "edge_mask": torch.cat(
                    [batch["edge_mask"], torch.zeros(2, 1, dtype=torch.bool)], dim=1
                ),
                "edge_degree_inv": torch.cat(
                    [batch["edge_degree_inv"], torch.zeros(2, 1)], dim=1
                ),
            },
        )
    assert base.shape == (2, 1)
    assert torch.allclose(base, padded, atol=1e-5)


def test_lit_qm9_sets_pooled_readout_flag():
    mixed = LitCPENQM9(_make_cpen(), eta_0=0.25, readout_mode="graph")
    assert mixed.model.x_only_graph_readout is False
    x_only = LitCPENQM9(_make_cpen(), eta_0=0.25, readout_mode="node")
    assert x_only.model.x_only_graph_readout is True


def test_lit_qm9_standardization_round_trip():
    lit = LitCPENQM9(
        _make_cpen(),
        eta_0=0.25,
        target_mean=2.673,
        target_std=1.504,
        target_name="mu",
        target_unit="D",
    )
    y = torch.tensor([0.0, 2.673, 5.0])
    assert torch.allclose(lit.unstandardize(lit.standardize(y)), y, atol=1e-5)
    assert torch.allclose(lit.standardize(torch.tensor([2.673])), torch.zeros(1), atol=1e-6)


def test_lit_qm9_step_reports_mae_in_physical_units():
    torch.manual_seed(0)
    std = 1.504
    lit = LitCPENQM9(
        _make_cpen(),
        eta_0=0.25,
        target_mean=2.673,
        target_std=std,
        loss="l1",
    )
    batch = _qm9_batch()
    pred = lit.forward(batch["x"], **lit._model_kwargs(batch)).reshape(-1)
    target = lit.standardize(batch["y"])
    expected_mae = float((pred - target).abs().mean().detach()) * std

    loss = lit._shared_step(batch, "val")
    assert loss.requires_grad
    assert loss.ndim == 0
    # L1 on the standardized target; MAE is that residual times the scale.
    assert float(loss) == pytest.approx(expected_mae / std, rel=1e-5)
    assert float(lit.val_mae.compute()) == pytest.approx(expected_mae, rel=1e-5)


def test_lit_qm9_rejects_classification_shaped_heads():
    with pytest.raises(ValueError, match="out_dim=1"):
        LitCPENQM9(_make_cpen(out_dim=10), eta_0=0.25)
    with pytest.raises(ValueError, match="graph readout"):
        LitCPENQM9(_make_cpen(), eta_0=0.25, readout_mode="node+edge")
    with pytest.raises(ValueError, match="loss"):
        LitCPENQM9(_make_cpen(), eta_0=0.25, loss="bce")


def test_lit_qm9_has_no_classification_metrics():
    lit = LitCPENQM9(_make_cpen(), eta_0=0.25)
    assert not hasattr(lit, "train_acc")
    # The base running-metric readout must survive a missing accuracy metric.
    lit.train_loss_running.update(torch.tensor(0.5), weight=2)
    metrics = lit.running_train_metrics(sync=False)
    assert metrics["train_loss"] == pytest.approx(0.5)
    assert "train_acc" not in metrics


def test_qm9_parquet_logger_tracks_mae():
    from cpen.training.callbacks import ParquetLoggerCallback

    for key in ("train_mae", "val_mae", "test_mae"):
        assert key in ParquetLoggerCallback.METRIC_KEYS


def test_cli_accepts_qm9_dataset_and_target():
    import argparse

    from scans.common.sweep_common import add_common_args, configure_graph_args

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args(
        [
            "--dataset",
            "qm9",
            "--qm9-target",
            "mu",
            "--model",
            "cpen",
            "--mode",
            "sweep_lr",
        ]
    )
    assert args.dataset == "qm9"
    assert args.qm9_target == "mu"
    assert args.qm9_loss == "l1"
    configure_graph_args(args)
    # Molecular property is graph-level, not per-atom.
    assert args.readout_mode == "graph"
    assert args.dataset == "qm9"


def test_qm9_checkpoint_monitors_val_mae():
    from scans.common.sweep_common import _resolve_checkpoint_settings

    args = SimpleNamespace(dataset="qm9", checkpoint_monitor=None, checkpoint_mode=None)
    assert _resolve_checkpoint_settings(args) == ("val_mae", "min")
