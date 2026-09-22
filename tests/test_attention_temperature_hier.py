"""Hierarchical attention-γ estimation (mean nonempty support row degrees)."""

from __future__ import annotations

import numpy as np
import torch

from cpen.utils.attention_temperature import (
    estimate_attention_gammas_from_hier_payload,
)
from cpen.utils.graph_hierarchical import (
    EDGE_TYPE_KNN,
    EDGE_TYPE_VN_LINK,
    build_hierarchical_graph,
    drop_pairwise_edges_from_batch,
)


def _toy_jet_batch(batch: int = 4, n: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.zeros(batch, n, 4)
    mask = torch.zeros(batch, n, dtype=torch.bool)
    for b in range(batch):
        n_active = min(n, 12 if b % 2 == 0 else 10)
        mask[b, :n_active] = True
        for i in range(0, n_active // 2):
            x[b, i] = torch.tensor([5.0, 3.0 + 0.05 * i, 0.1 * i, 1.0])
        for i in range(n_active // 2, n_active):
            j = i - n_active // 2
            x[b, i] = torch.tensor([4.0, -2.5 - 0.05 * j, 2.0 + 0.05 * j, -0.5])
        x[b, :n_active, 0] = torch.linalg.vector_norm(x[b, :n_active, 1:], dim=-1) + 1.0
    return x, mask


def test_hier_gamma_hyperedge_m22_and_drop_pairwise() -> None:
    x, mask = _toy_jet_batch(batch=4, n=12)
    g = build_hierarchical_graph(
        x,
        k=3,
        eps=0.15,
        min_samples=2,
        n_virtual_nodes=1,
        n_virtual_edges=1,
        max_dbscan_edges=8,
        mask=mask,
    )
    payload = g.to_cache_payload()
    payload["labels"] = torch.zeros(x.size(0), dtype=torch.long)
    idx = np.arange(x.size(0))

    with_knn, _ = estimate_attention_gammas_from_hier_payload(
        payload,
        idx,
        batch_size=2,
        all_to_all_particle_attention=True,
        hyperedge_m22_only=True,
        drop_pairwise_edges=False,
    )
    no_pair, summaries = estimate_attention_gammas_from_hier_payload(
        payload,
        idx,
        batch_size=2,
        all_to_all_particle_attention=True,
        hyperedge_m22_only=True,
        drop_pairwise_edges=True,
    )

    # M11 is all-to-all among valid nodes — independent of pairwise 2-edges.
    assert abs(with_knn.gamma_11 - no_pair.gamma_11) < 1e-6
    # Dropping kNN + vn_link changes incidence degrees (M12 / M21).
    assert no_pair.gamma_12 > with_knn.gamma_12
    assert no_pair.gamma_21 < with_knn.gamma_21
    # Hyperedge-only M22 ignores pairwise slots, so the drop is a no-op there.
    assert abs(with_knn.gamma_22 - no_pair.gamma_22) < 1e-6
    assert summaries["22"]["n_rows"] > 0
    # Compacted batch should have no live pairwise rows.
    fields = {
        "x": g.x,
        "edge_x": g.edge_x,
        "edge_type": g.edge_type,
        "edge_mask": g.edge_mask,
        "incidence": g.incidence,
        "node_mask": g.node_mask,
    }
    dropped = drop_pairwise_edges_from_batch(dict(fields))
    live = dropped["edge_mask"]
    assert not ((dropped["edge_type"] == EDGE_TYPE_KNN) & live).any()
    assert not ((dropped["edge_type"] == EDGE_TYPE_VN_LINK) & live).any()
