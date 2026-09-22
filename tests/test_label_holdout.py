"""Label-holdout masking for real-stream fine-tune."""

from __future__ import annotations

import numpy as np
import torch

from cpen.apps.streams.blob_graph_cache import (
    edge_supervise_mask_from_nodes,
    node_supervise_mask,
)
from cpen.apps.streams.cell_stream_split import CellStreamOccupancy
from cpen.apps.streams.real_discovery import (
    DiscoveryConfig,
    cell_has_other_streams,
    holdout_recovery_breakdown,
    rebuild_cell_split,
)


def test_node_supervise_mask_excludes_holdout() -> None:
    labels = np.array(["Jhelum", "AAU", "Background", "Phoenix"], dtype=object)
    m = node_supervise_mask(labels, {"AAU", "Phoenix"})
    assert m.tolist() == [True, False, True, False]


def test_edge_supervise_mask_requires_all_members() -> None:
    # 2 knn edges: (0,1) and (2,3); node 1 holdout → edge0 unsupervised
    node_ok = torch.tensor([True, False, True, True])
    # edge0: nodes 0,1; edge1: nodes 2,3
    incidence_node = torch.tensor([0, 1, 2, 3])
    incidence_edge = torch.tensor([0, 0, 1, 1])
    eok = edge_supervise_mask_from_nodes(
        incidence_node, incidence_edge, n_edges=2, node_ok=node_ok
    )
    assert eok.tolist() == [False, True]


def test_mask_holdout_rebuild_uses_all_non_val_cells() -> None:
    # Toy occupancy via monkeypatch is heavy; use real galaxy if present, else skip.
    from pathlib import Path

    from cpen.apps.streams.preprocess.config import DEFAULT_REAL_PIPELINE_ROOT

    gdir = Path(DEFAULT_REAL_PIPELINE_ROOT) / "train" / "0000" / "cells"
    if not gdir.is_dir():
        return
    cfg = DiscoveryConfig(mask_holdout=True, cell_val_frac=0.15, split_seed=0)
    split = rebuild_cell_split(cfg)
    all_cells = set(split.occupancy.cell_streams)
    assert split.train_cells | split.val_cells == all_cells
    assert split.test_cells == frozenset(all_cells)
    assert split.train_cells.isdisjoint(split.val_cells)
    assert "AAU" in split.holdout_streams


def test_holdout_recovery_breakdown_solo_flag() -> None:
    occ = CellStreamOccupancy(
        galaxy_id="0000",
        cell_streams={
            1: {"AAU": 10},
            2: {"AAU": 5, "Jhelum": 20},
        },
        stream_counts={"AAU": 15, "Jhelum": 20},
    )
    assert not cell_has_other_streams(occ, 1, "AAU")
    assert cell_has_other_streams(occ, 2, "AAU")

    import pandas as pd

    stars = pd.DataFrame(
        {
            "cell_id": [1, 1, 2, 2],
            "stream_label": ["AAU", "AAU", "AAU", "AAU"],
            "p": [0.9, 0.8, 0.2, 0.1],
            "tp": [True, True, False, False],
            "fn": [False, False, True, True],
        }
    )
    tab = holdout_recovery_breakdown(stars, holdout_streams={"AAU"}, occupancy=occ)
    solo = tab[(tab.stream_label == "AAU") & (tab.subset == "solo")].iloc[0]
    coloc = tab[(tab.stream_label == "AAU") & (tab.subset == "colocated")].iloc[0]
    assert solo.n_members == 2 and solo.recall == 1.0
    assert coloc.n_members == 2 and coloc.recall == 0.0
