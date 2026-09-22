"""Resume skip-ahead for the JetClass stream (no ROOT required)."""

from pathlib import Path

from cpen.apps.jets.jetclass_stream import (
    Shard,
    skip_jets_for_worker,
    skip_round_robin_shards,
    split_plan_over_replicas,
    split_plan_over_workers,
    take_round_robin_shards,
)
from cpen.apps.jets.jetclass_stream_datamodule import JetClassStreamDatamodule
from cpen.utils.jetclass import N_CLASSES


def test_skip_jets_for_worker_splits_remainder():
    assert [skip_jets_for_worker(10, worker_id=i, num_workers=4) for i in range(4)] == [
        3,
        3,
        2,
        2,
    ]
    assert skip_jets_for_worker(7, worker_id=0, num_workers=1) == 7
    assert skip_jets_for_worker(0, worker_id=2, num_workers=4) == 0


def _one_file_per_class(n_jets: int = 1000) -> list[Shard]:
    return [
        Shard(Path(f"{label}.root"), label, 0, n_jets) for label in range(N_CLASSES)
    ]


def test_skip_round_robin_trims_an_equal_class_prefix():
    out = skip_round_robin_shards(_one_file_per_class(), 100)
    by_label = {s.label: s for s in out}
    assert set(by_label) == set(range(N_CLASSES))
    for label in range(N_CLASSES):
        assert by_label[label].entry_start == 10
        assert by_label[label].entry_stop == 1000


def test_skip_round_robin_leftover_goes_to_low_labels():
    out = skip_round_robin_shards(_one_file_per_class(), 101)
    by_label = {s.label: s for s in out}
    assert by_label[0].entry_start == 11
    for label in range(1, N_CLASSES):
        assert by_label[label].entry_start == 10


def test_skip_prefix_and_suffix_partition_the_plan():
    """Resume skip must continue through the files, not reshuffle the same jets."""
    shards = _one_file_per_class(n_jets=1000)
    prefix = take_round_robin_shards(shards, 250)
    suffix = skip_round_robin_shards(shards, 250)
    pre, suf = _jet_ids(prefix), _jet_ids(suffix)
    assert pre.isdisjoint(suf)
    assert pre | suf == _jet_ids(shards)
    assert len(pre) == 250
    assert len(suf) == 9750


def test_skip_round_robin_zero_is_a_noop():
    shards = _one_file_per_class()
    out = skip_round_robin_shards(shards, 0)
    assert [(s.label, s.entry_start, s.entry_stop) for s in out] == [
        (s.label, s.entry_start, s.entry_stop) for s in shards
    ]


def test_datamodule_resume_skip_uses_global_step():
    """A preempted stream must skip by optimizer steps, not rewind the epoch."""

    class _Trainer:
        global_step = 20

    dummy = type("D", (), {})()
    dummy.batch_size = 512
    dummy._trainer = _Trainer()
    assert JetClassStreamDatamodule._resume_skip_jets(dummy) == 20 * 512


def _balanced_plan(n_shards_per_class: int = 4, jets_per_shard: int = 100) -> list[Shard]:
    shards: list[Shard] = []
    for depth in range(n_shards_per_class):
        for label in range(N_CLASSES):
            start = depth * jets_per_shard
            shards.append(
                Shard(Path(f"{label}.root"), label, start, start + jets_per_shard)
            )
    return shards


def _jet_ids(shards: list[Shard]) -> set[tuple[str, int, int]]:
    ids: set[tuple[str, int, int]] = set()
    for shard in shards:
        for entry in range(shard.entry_start, shard.entry_stop):
            ids.add((str(shard.path), shard.label, entry))
    return ids


def test_take_round_robin_keeps_a_balanced_prefix():
    out = take_round_robin_shards(_one_file_per_class(), 100)
    by_label = {s.label: s for s in out}
    assert set(by_label) == set(range(N_CLASSES))
    for label in range(N_CLASSES):
        assert by_label[label].entry_start == 0
        assert by_label[label].entry_stop == 10


def test_single_process_replica_plan_matches_worker_split():
    shards = _balanced_plan()
    for worker_id in range(4):
        assert split_plan_over_replicas(
            shards, rank=0, world_size=1, worker_id=worker_id, num_workers=4
        ) == split_plan_over_workers(shards, worker_id, 4)


def test_ddp_replicas_partition_the_stream_without_overlap():
    """Two ranks must not re-walk the same jets; that doubled the epoch length."""
    shards = _balanced_plan(n_shards_per_class=8, jets_per_shard=50)
    world, workers = 2, 4
    plans = [
        split_plan_over_replicas(
            shards, rank=rank, world_size=world, worker_id=wid, num_workers=workers
        )
        for rank in range(world)
        for wid in range(workers)
    ]
    counts = [sum(s.n_jets for s in plan) for plan in plans]
    assert len(set(counts)) == 1
    assert counts[0] > 0
    for plan in plans:
        assert {s.label for s in plan} == set(range(N_CLASSES))

    all_ids = [_jet_ids(plan) for plan in plans]
    for i, left in enumerate(all_ids):
        for right in all_ids[i + 1 :]:
            assert left.isdisjoint(right)

    # Equalizing to the shortest replica may drop a few remainder jets, but
    # most of the plan must still be covered exactly once.
    covered = set().union(*all_ids)
    full = _jet_ids(shards)
    assert covered <= full
    assert len(covered) >= int(0.95 * len(full))
    assert len(covered) == counts[0] * world * workers


def test_datamodule_ddp_identity_reads_trainer_world_size():
    class _Trainer:
        global_rank = 1
        world_size = 2

    dummy = type("D", (), {})()
    dummy._trainer = _Trainer()
    assert JetClassStreamDatamodule._ddp_identity(dummy) == (1, 2)
    assert JetClassStreamDatamodule._ddp_identity(type("D", (), {})()) == (0, 1)
