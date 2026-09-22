"""Streaming JetClass reader: featurization, exact dataset sizes, single pass."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from cpen.apps.jets.jetclass_stream import (
    FEATURE_CONFIGS,
    JETS_PER_FILE,
    JetClassStreamDataset,
    JetClassSubsetDataset,
    RAW_BRANCHES,
    featurize_chunk,
    n_features,
    chunk_size_for_workers,
    plan_shards,
    split_plan_over_workers,
    steps_per_epoch,
)
from cpen.utils.jetclass import N_CLASSES

RAW_ROOT = Path("/projects/j/jdezoort/jetclass")

pytestmark = pytest.mark.skipif(
    not (RAW_ROOT / "train_100M").is_dir(),
    reason=f"raw JetClass ROOT files not available under {RAW_ROOT}",
)


def _checksums(x_raw: torch.Tensor) -> np.ndarray:
    """Per-jet fingerprint, for identity comparisons across readers."""
    return x_raw.reshape(x_raw.shape[0], -1).double().sum(-1).numpy()


# --------------------------------------------------------------------------
# Featurization
# --------------------------------------------------------------------------


def test_featurization_matches_cache_pipeline_up_to_the_l2_step():
    """
    The streaming path must reproduce the archive pipeline exactly except for
    the row-$L^2$ renormalization it deliberately drops.
    """
    import uproot

    from cpen.utils.jetclass import build_part_features, read_jetclass_root_file

    path = RAW_ROOT / "train_100M" / "TTBar_000.root"
    n = 500
    raw = read_jetclass_root_file(path, max_num_particles=128, entry_start=0, entry_stop=n)
    x_ref, mask_ref, z_ref = build_part_features(raw)

    with uproot.open(path) as handle:
        arrays = handle["tree"].arrays(list(RAW_BRANCHES), entry_start=0, entry_stop=n)
    out = featurize_chunk(arrays, num_particles=128, feature_config="full", label=8)

    assert np.array_equal(out["mask"], mask_ref)
    np.testing.assert_allclose(out["z"], z_ref, atol=1e-6)

    norm = np.linalg.norm(out["x"], axis=-1, keepdims=True)
    x_l2 = np.where(norm > 1e-12, out["x"] / np.clip(norm, 1e-12, None) * np.sqrt(17), 0.0)
    x_l2 = x_l2 * out["mask"][..., None]
    np.testing.assert_allclose(x_l2, x_ref, atol=5e-5)


def test_row_l2_step_would_rescale_particles_by_a_kinematics_dependent_factor():
    """
    Why the L2 step is dropped: the divisor varies several-fold across
    particles, so it multiplies the particle-ID one-hots by a factor that
    depends on the particle's kinematics.
    """
    import uproot

    path = RAW_ROOT / "train_100M" / "TTBar_000.root"
    with uproot.open(path) as handle:
        arrays = handle["tree"].arrays(list(RAW_BRANCHES), entry_start=0, entry_stop=500)
    out = featurize_chunk(arrays, num_particles=128, feature_config="full", label=8)

    norm = np.linalg.norm(out["x"], axis=-1)[out["mask"]] / np.sqrt(17)
    assert np.percentile(norm, 99) / np.percentile(norm, 1) > 2.0


def test_padded_slots_are_zero_and_z_sums_to_one():
    import uproot

    path = RAW_ROOT / "train_100M" / "HToBB_000.root"
    with uproot.open(path) as handle:
        arrays = handle["tree"].arrays(list(RAW_BRANCHES), entry_start=0, entry_stop=200)
    out = featurize_chunk(arrays, num_particles=128, feature_config="full", label=1)

    inactive = ~out["mask"]
    assert np.all(out["x"][inactive] == 0.0)
    assert np.all(out["x_raw"][inactive] == 0.0)
    assert np.all(out["z"][inactive] == 0.0)
    np.testing.assert_allclose(out["z"].sum(-1), 1.0, atol=1e-5)
    assert np.isfinite(out["x"]).all(), "log(0) on padded slots must be scrubbed"


@pytest.mark.parametrize("config", sorted(FEATURE_CONFIGS))
def test_feature_configs_give_the_declared_width(config):
    import uproot

    path = RAW_ROOT / "train_100M" / "WToQQ_000.root"
    with uproot.open(path) as handle:
        arrays = handle["tree"].arrays(list(RAW_BRANCHES), entry_start=0, entry_stop=64)
    out = featurize_chunk(arrays, num_particles=40, feature_config=config, label=7)
    assert out["x"].shape == (64, 40, n_features(config))


def test_kin_config_is_a_subset_of_the_full_columns():
    """The ablation arms must share one standardization convention."""
    import uproot

    path = RAW_ROOT / "train_100M" / "WToQQ_000.root"
    with uproot.open(path) as handle:
        arrays = handle["tree"].arrays(list(RAW_BRANCHES), entry_start=0, entry_stop=64)
    full = featurize_chunk(arrays, num_particles=40, feature_config="full", label=7)
    kin = featurize_chunk(arrays, num_particles=40, feature_config="kin", label=7)

    names = list(FEATURE_CONFIGS["full"])
    for i, name in enumerate(FEATURE_CONFIGS["kin"]):
        np.testing.assert_allclose(kin["x"][..., i], full["x"][..., names.index(name)])


def test_labels_match_the_one_hot_branches():
    """Labels come from the filename; confirm that against the ROOT branches."""
    import uproot

    from cpen.utils.jetclass import LABEL_NAMES

    path = RAW_ROOT / "train_100M" / "TTBarLep_000.root"
    with uproot.open(path) as handle:
        labels = handle["tree"].arrays(list(LABEL_NAMES), entry_stop=128)
    onehot = np.stack([np.asarray(labels[n]) for n in LABEL_NAMES], axis=-1)
    assert np.all(onehot.argmax(-1) == LABEL_NAMES.index("label_Tbl"))


# --------------------------------------------------------------------------
# Read plan: exact D, class balance, nesting
# --------------------------------------------------------------------------


def test_plan_is_class_balanced_and_hits_the_requested_size():
    shards = plan_shards(RAW_ROOT, "train", n_jets=50_000)
    assert sum(s.n_jets for s in shards) == 50_000
    per_class: dict[int, int] = {}
    for shard in shards:
        per_class[shard.label] = per_class.get(shard.label, 0) + shard.n_jets
    assert set(per_class) == set(range(N_CLASSES))
    assert set(per_class.values()) == {5_000}


def test_plan_prefix_is_class_balanced():
    """Interleaving means any prefix of the plan already mixes all ten classes."""
    shards = plan_shards(RAW_ROOT, "train", n_jets=200_000, chunk_size=1_000)
    labels = {s.label for s in shards[:N_CLASSES]}
    assert labels == set(range(N_CLASSES))


def test_smaller_budgets_nest_inside_larger_ones():
    """Scaling-law points must be nested so D is the only thing that varies."""
    small = plan_shards(RAW_ROOT, "train", n_jets=10_000, chunk_size=1_000)
    large = plan_shards(RAW_ROOT, "train", n_jets=100_000, chunk_size=1_000)

    def covered(shards):
        spans: dict[Path, set[int]] = {}
        for s in shards:
            spans.setdefault(s.path, set()).update(range(s.entry_start, s.entry_stop))
        return spans

    small_span, large_span = covered(small), covered(large)
    for path, entries in small_span.items():
        assert entries <= large_span[path], f"{path.name} not nested"


def test_plan_without_a_budget_covers_the_whole_split():
    shards = plan_shards(RAW_ROOT, "val", n_jets=None)
    assert sum(s.n_jets for s in shards) == N_CLASSES * 5 * JETS_PER_FILE


def test_plan_rejects_unknown_split():
    with pytest.raises(ValueError, match="split must be one of"):
        plan_shards(RAW_ROOT, "nope", n_jets=10)


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------


def test_subset_dataset_is_exact_and_balanced():
    dataset = JetClassSubsetDataset(
        RAW_ROOT, "val", n_jets=2_000, num_particles=40, feature_config="full"
    )
    assert len(dataset) == 2_000
    counts = torch.bincount(dataset._data["y"], minlength=N_CLASSES)
    assert torch.all(counts == 200)
    sample = dataset[0]
    assert sample["x"].shape == (40, 17)
    assert sample["x_raw"].shape == (40, 4)


def test_stream_dataset_yields_each_jet_exactly_once():
    """The compute-optimal regime requires a genuine single pass."""
    dataset = JetClassStreamDataset(
        RAW_ROOT, "val", n_jets=2_000, num_particles=40, chunk_size=250
    )
    seen = list(dataset)
    assert len(seen) == 2_000 == dataset.n_jets

    keys = _checksums(torch.stack([s["x_raw"] for s in seen]))
    assert len(np.unique(np.round(keys, 4))) > 0.99 * len(keys), "jets repeated"

    counts = torch.bincount(
        torch.stack([s["y"] for s in seen]), minlength=N_CLASSES
    )
    assert torch.all(counts == 200)


def test_stream_and_subset_select_the_same_jets():
    """Both flavours must read the same D jets, only the order differs."""
    n = 1_000
    subset = JetClassSubsetDataset(RAW_ROOT, "val", n_jets=n, num_particles=40)
    stream = JetClassStreamDataset(
        RAW_ROOT, "val", n_jets=n, num_particles=40, chunk_size=100
    )
    a = np.sort(np.round(_checksums(subset._data["x_raw"]), 3))
    b = np.sort(
        np.round(_checksums(torch.stack([s["x_raw"] for s in stream])), 3)
    )
    np.testing.assert_allclose(a, b, atol=1e-2)


def test_stream_shuffle_seed_changes_order_but_not_content():
    kwargs = dict(n_jets=500, num_particles=40, chunk_size=100)
    first = list(JetClassStreamDataset(RAW_ROOT, "val", shuffle_seed=0, **kwargs))
    second = list(JetClassStreamDataset(RAW_ROOT, "val", shuffle_seed=7, **kwargs))

    ka = _checksums(torch.stack([s["x_raw"] for s in first]))
    kb = _checksums(torch.stack([s["x_raw"] for s in second]))
    assert not np.allclose(ka, kb), "seed did not change the order"
    np.testing.assert_allclose(np.sort(ka), np.sort(kb), atol=1e-3)


@pytest.mark.parametrize("num_workers", [2, 4, 5, 6, 8])
def test_every_worker_sees_every_class(num_workers):
    """
    Striding the interleaved plan directly aliases with the class cycle: with an
    even worker count ``shards[id::num_workers]`` hands each worker only the
    five even labels, and the model never learns the other five per worker.
    """
    n_jets = 1_000_000
    chunk = chunk_size_for_workers(n_jets, num_workers)
    shards = plan_shards(RAW_ROOT, "val", n_jets=n_jets, chunk_size=chunk)
    plans = [
        split_plan_over_workers(shards, i, num_workers) for i in range(num_workers)
    ]
    for i, plan in enumerate(plans):
        labels = {s.label for s in plan}
        assert labels == set(range(N_CLASSES)), f"worker {i} only saw {sorted(labels)}"

    # No shard dropped or handed to two workers.
    dealt = [s for plan in plans for s in plan]
    assert len(dealt) == len(shards)
    assert {id(s) for s in dealt} == {id(s) for s in shards}


def test_stream_windows_are_class_balanced():
    """
    Every batch must mix all ten labels. Shards hold one class each, so relying
    on a shuffle buffer to undo that leaves batches class correlated.
    """
    dataset = JetClassStreamDataset(
        RAW_ROOT, "val", n_jets=40_000, num_particles=8, feature_config="kin"
    )
    labels = np.array([int(jet["y"]) for jet in dataset])
    assert labels.size == dataset.n_jets

    # Arrivals are exact round-robin; the shuffle buffer then resamples within
    # blocks, so a window of 2000 is a balanced draw with binomial spread
    # (mean 200 per class, sigma ~13). Six sigma either side is a wide margin
    # that still fails hard on the block-correlated behaviour.
    for start in range(0, labels.size - 2_000, 2_000):
        counts = np.bincount(labels[start : start + 2_000], minlength=N_CLASSES)
        assert counts.min() > 120, f"window at {start} starved a class: {counts}"
        assert counts.max() < 280, f"window at {start} over-weighted a class: {counts}"


def test_stream_shuffles_within_class_rather_than_emitting_file_order():
    dataset = JetClassStreamDataset(
        RAW_ROOT, "val", n_jets=40_000, num_particles=8, feature_config="kin"
    )
    labels = np.array([int(jet["y"]) for jet in dataset])
    # Perfect round-robin with no buffer would make the label sequence exactly
    # periodic; the shuffle buffer must break that.
    cycle = labels[: N_CLASSES * 50].reshape(-1, N_CLASSES)
    assert not all(np.array_equal(cycle[0], row) for row in cycle[1:])


def test_steps_per_epoch_accounts_for_per_worker_partial_batches():
    assert steps_per_epoch(1_000, 100, 1) == 10
    # Four workers each hold 250 jets -> 3 batches each (100+100+50).
    assert steps_per_epoch(1_000, 100, 4) == 12
    assert steps_per_epoch(1_000, 100, 0) == 10
    # Two DDP ranks split the stream; each rank's epoch is half as long.
    assert steps_per_epoch(1_000, 100, 1, world_size=2) == 5
    assert steps_per_epoch(100_000_000, 512, 4, world_size=2) == 97_660


# --------------------------------------------------------------------------
# Datamodule and live graphs
# --------------------------------------------------------------------------


def _datamodule(**overrides):
    from cpen.apps.jets.jetclass_stream_datamodule import JetClassStreamDatamodule

    kwargs = dict(
        data_root=str(RAW_ROOT),
        batch_size=16,
        num_workers=0,
        num_particles=40,
        n_train=500,
        n_val=200,
        n_test=200,
    )
    kwargs.update(overrides)
    return JetClassStreamDatamodule(**kwargs)


def test_datamodule_emits_x_raw_and_no_edges():
    dm = _datamodule()
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert set(batch) == {"x", "x_raw", "mask", "z", "y"}
    assert "edge_x" not in batch, "edges must be built live, not in workers"
    assert dm.get_dims() == (17, 10)
    assert dm.n_train == 500


def test_datamodule_switches_to_streaming_above_the_ram_limit():
    from cpen.apps.jets.jetclass_stream import JetClassStreamDataset as Stream

    assert not _datamodule(n_train=500).streaming
    big = _datamodule(n_train=None)
    assert big.streaming
    assert isinstance(big._build_train_dataset(), Stream)


def test_probe_split_sizes_answers_before_setup():
    """
    The sweep driver reads these before ``setup``, so they must come from the
    read plan rather than from a built dataset.
    """
    dm = _datamodule(n_train=5_000, n_val=2_000, n_test=2_000)
    assert dm._train is None
    assert dm.probe_split_sizes() == {
        "n_train": 5_000,
        "n_val": 2_000,
        "n_test": 2_000,
    }


def test_probe_split_sizes_reports_the_full_split_when_unbounded():
    dm = _datamodule(n_train=None)
    assert dm.probe_split_sizes()["n_train"] == 100_000_000


def test_datamodule_reports_the_hooks_the_sweep_driver_reads():
    """
    The driver calls ``run_metadata``/``metadata``, so a datamodule that only
    defines its own name for them silently reports n_train=0 and an unknown
    graph mode in the banner and the parquet record.
    """
    dm = _datamodule(n_train=500)
    sizes = dm.split_sizes()
    run_meta = dm.run_metadata(sizes)
    assert run_meta["n_train"] == 500
    assert run_meta["graph_loading"] == "live"
    assert run_meta["edge_mode"] == "star-radius"
    assert run_meta["jetclass_particle_normalization"] == "part-full-affine"

    # ``metadata`` feeds the muP LR scaling and must answer before setup.
    assert dm._train is None
    meta = dm.metadata()
    assert meta["n_train"] == 500
    assert meta["dataset"] == dm.dataset_name()
    assert meta["graph_loading"] == "live"


def test_datamodule_metadata_reports_n_train_while_streaming():
    dm = _datamodule(n_train=None, n_val=200, n_test=200)
    assert dm.streaming
    assert dm.metadata()["n_train"] == 100_000_000


def test_datamodule_rejects_unknown_feature_config():
    with pytest.raises(ValueError, match="feature_config must be one of"):
        _datamodule(feature_config="nope")


def test_kin_config_narrows_the_model_input_dim():
    assert _datamodule(feature_config="kin").get_dims() == (3, 10)
    assert _datamodule(feature_config="kin7").get_dims() == (7, 10)


@pytest.mark.parametrize("edge_features", ["logdot-dp", "part-interaction"])
def test_live_star_graphs_feed_a_cpen_forward(edge_features):
    from cpen.lit_models.lit_cpen_jetclass import LitCPENJetClass
    from cpen.models.cpen import CPEN

    dm = _datamodule()
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))

    model = CPEN(
        n_features=17, n_edge_features=4, out_dim=10, depth=2, width=32,
        operators="incidence", operator_backend="sparse", readout_mode="graph",
    )
    lit = LitCPENJetClass(
        model=model, model_name="cpen", eta_0=0.1,
        live_star_radius=0.15, live_edge_features=edge_features,
    )
    out = lit.on_after_batch_transfer(dict(batch), 0)

    assert "x_raw" not in out, "x_raw should be consumed by graph construction"
    assert out["edge_x"].shape == (16, 40, 4)
    logits = model(
        out["x"], out["edge_x"],
        incidence_node=out["incidence_node"], incidence_edge=out["incidence_edge"],
        incidence_nnz=out["incidence_nnz"], node_degree_inv=out["node_degree_inv"],
        edge_degree_inv=out["edge_degree_inv"], mask=out["mask"],
    )
    assert logits.shape == (16, 10)
    assert torch.isfinite(logits).all()


def test_live_star_and_knn_are_mutually_exclusive():
    from cpen.lit_models.lit_cpen_jetclass import LitCPENJetClass
    from cpen.models.cpen import CPEN

    model = CPEN(
        n_features=17, n_edge_features=4, out_dim=10, depth=1, width=16,
        operators="incidence", operator_backend="sparse", readout_mode="graph",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        LitCPENJetClass(
            model=model, model_name="cpen", eta_0=0.1,
            live_star_radius=0.15, live_graph_k=8,
        )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_tags_dataset_size_into_the_run_name():
    """Without D in the run name, a D=1e5 point would resume a D=1e6 one."""
    import argparse

    from scans.common.sweep_common import (
        add_common_args,
        configure_graph_args,
        jetclass_live_graph_kwargs,
        run_options_from_args,
    )

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args([
        "--mode", "sweep_lr", "--etas", "0.1", "--epochs", "1",
        "--model", "cpen", "--operators", "incidence", "--star-radius", "0.15",
        "--jetclass-stream", "--jetclass-edge-features", "part-interaction",
        "--num-particles", "40", "--n-train", "1000000",
        "--root", "/tmp/x", "--data-root", str(RAW_ROOT),
    ])
    args.dataset = "jetclass"
    configure_graph_args(args)

    tag = run_options_from_args(args).extra_tag
    assert "stream" in tag
    assert "D1M" in tag
    assert "partint" in tag
    assert "p40" in tag
    assert "s50k" not in tag

    args.max_steps = 50_000
    tag_steps = run_options_from_args(args).extra_tag
    assert "s50k" in tag_steps

    assert jetclass_live_graph_kwargs(args) == {
        "live_star_radius": 0.15,
        "live_edge_features": "part-interaction",
        "live_centroid_weight": None,
    }


def test_cli_stream_requires_a_star_radius():
    import argparse

    from scans.common.sweep_common import add_common_args, jetclass_live_graph_kwargs

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args([
        "--mode", "sweep_lr", "--etas", "0.1", "--epochs", "1",
        "--model", "cpen", "--operators", "incidence",
        "--jetclass-stream", "--root", "/tmp/x",
    ])
    args.dataset = "jetclass"
    with pytest.raises(ValueError, match="requires --star-radius"):
        jetclass_live_graph_kwargs(args)


def test_cli_stream_baseline_transformer_skips_star_radius():
    import argparse

    from scans.common.sweep_common import add_common_args, jetclass_live_graph_kwargs

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args([
        "--mode", "sweep_lr", "--etas", "0.1", "--epochs", "1",
        "--model", "baseline-transformer", "--operators", "incidence",
        "--jetclass-stream", "--root", "/tmp/x",
    ])
    args.dataset = "jetclass"
    assert jetclass_live_graph_kwargs(args) == {}


def test_cli_without_stream_flag_leaves_the_cache_path_alone():
    import argparse

    from scans.common.sweep_common import add_common_args, jetclass_live_graph_kwargs

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args([
        "--mode", "sweep_lr", "--etas", "0.1", "--epochs", "1",
        "--model", "cpen", "--operators", "incidence", "--star-radius", "0.15",
        "--root", "/tmp/x",
    ])
    args.dataset = "jetclass"
    assert jetclass_live_graph_kwargs(args) == {}
