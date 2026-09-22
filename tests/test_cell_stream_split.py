"""Cell-stream split helpers for real-blob fine-tuning."""

from __future__ import annotations

from cpen.apps.streams.cell_stream_split import (
    CellStreamOccupancy,
    assign_cells_by_train_streams,
    parse_stream_name_list,
    partition_manifest_by_cells,
    resolve_train_holdout_streams,
)
import pandas as pd


def _toy_occupancy() -> CellStreamOccupancy:
    # Cells 1–3: train streams; 4–5: holdout only; 6: empty.
    cell_streams = {
        1: {"Jhelum": 50, "Orphan": 10},
        2: {"Jhelum": 20},
        3: {"Willka": 15, "AAU": 5},  # train + holdout
        4: {"Phoenix": 40},
        5: {"Chenab": 12, "Elqui": 8},
        6: {},
    }
    stream_counts: dict[str, int] = {}
    for counts in cell_streams.values():
        for k, v in counts.items():
            stream_counts[k] = stream_counts.get(k, 0) + v
    return CellStreamOccupancy(
        galaxy_id="0000",
        cell_streams=cell_streams,
        stream_counts=stream_counts,
    )


def test_parse_stream_name_list() -> None:
    assert parse_stream_name_list(None) is None
    assert parse_stream_name_list("auto") is None
    assert parse_stream_name_list("Jhelum, Orphan") == {"Jhelum", "Orphan"}


def test_default_holdout_assignment() -> None:
    occ = _toy_occupancy()
    train, holdout = resolve_train_holdout_streams(occ)
    assert holdout == {"AAU", "Chenab", "Elqui", "Phoenix"}
    assert "Jhelum" in train and "Orphan" in train and "Willka" in train
    assert "Phoenix" not in train


def test_assign_cells_preponderance_test() -> None:
    occ = _toy_occupancy()
    split = assign_cells_by_train_streams(
        occ, val_frac=0.0, seed=0, min_train_stream_stars=1
    )
    # Train cells: 1,2,3 (contain Jhelum/Orphan/Willka). Test: 4,5,6.
    assert split.train_cells == frozenset({1, 2, 3})
    assert split.test_cells == frozenset({4, 5, 6})
    assert split.train_cells.isdisjoint(split.test_cells)
    assert "Phoenix" in split.holdout_streams
    # Real-0000 has more empty/holdout cells than train; toy case is balanced.


def test_val_frac_carves_train_cells() -> None:
    occ = _toy_occupancy()
    split = assign_cells_by_train_streams(occ, val_frac=0.34, seed=0)
    assert split.train_cells.isdisjoint(split.val_cells)
    assert split.train_cells | split.val_cells == frozenset({1, 2, 3})
    assert len(split.val_cells) >= 1


def test_partition_manifest_by_cells() -> None:
    occ = _toy_occupancy()
    split = assign_cells_by_train_streams(occ, val_frac=0.0, seed=0)
    man = pd.DataFrame(
        {
            "cell_id": [1, 1, 4, 6, 2],
            "n_stars": [100, 80, 90, 70, 60],
            "n_mock": [10, 0, 5, 0, 8],
            "file": ["a", "b", "c", "d", "e"],
        }
    )
    train, val, test = partition_manifest_by_cells(man, split)
    assert set(train["cell_id"]) == {1, 2}
    assert len(val) == 0
    assert set(test["cell_id"]) == {4, 6}
