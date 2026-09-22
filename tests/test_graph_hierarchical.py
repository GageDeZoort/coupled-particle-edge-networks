"""Tests for hierarchical jet graph construction."""

from __future__ import annotations

import torch

from cpen.utils.graph_hierarchical import (
    EDGE_TYPE_DBSCAN,
    EDGE_TYPE_KNN,
    EDGE_TYPE_VIRTUAL_HYPER,
    EDGE_TYPE_VN_LINK,
    NODE_TYPE_PARTICLE,
    NODE_TYPE_VIRTUAL,
    build_hierarchical_graph,
    hierarchical_construction_tag,
)


def _toy_jet_batch(batch: int = 2, n: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """Synthetic jets with a few spatial clumps so DBSCAN finds structure."""
    torch.manual_seed(0)
    x = torch.zeros(batch, n, 4)
    mask = torch.zeros(batch, n, dtype=torch.bool)
    # Two clumps in (eta, phi) ≈ from (px,py,pz).
    for b in range(batch):
        n_active = min(n, 12 if b == 0 else 10)
        mask[b, :n_active] = True
        # Clump A
        for i in range(0, n_active // 2):
            x[b, i] = torch.tensor([5.0, 3.0 + 0.05 * i, 0.1 * i, 1.0])
        # Clump B (separated in phi via py sign)
        for i in range(n_active // 2, n_active):
            j = i - n_active // 2
            x[b, i] = torch.tensor([4.0, -2.5 - 0.05 * j, 2.0 + 0.05 * j, -0.5])
        # Energies positive
        x[b, :n_active, 0] = torch.linalg.vector_norm(x[b, :n_active, 1:], dim=-1) + 1.0
    return x, mask


def test_tag_format() -> None:
    tag = hierarchical_construction_tag(
        k=8, eps=0.08, min_samples=2, n_virtual_nodes=1, n_virtual_edges=2, max_dbscan_edges=16
    )
    assert tag == "hier_k8_eps0p08_ms2_vn1_ve2_mdb16_pint"


def test_tag_parse_roundtrip() -> None:
    from cpen.utils.graph_hierarchical import parse_hierarchical_construction_tag

    tag = hierarchical_construction_tag(
        k=8, eps=0.08, min_samples=2, n_virtual_nodes=1, n_virtual_edges=1, max_dbscan_edges=32
    )
    parsed = parse_hierarchical_construction_tag(tag)
    assert parsed["k"] == 8
    assert abs(float(parsed["eps"]) - 0.08) < 1e-12
    assert parsed["max_dbscan_edges"] == 32


def test_shapes_and_types() -> None:
    x, mask = _toy_jet_batch()
    k, n_vn, n_ve, max_db = 4, 2, 1, 8
    g = build_hierarchical_graph(
        x,
        k=k,
        eps=0.08,
        min_samples=2,
        n_virtual_nodes=n_vn,
        n_virtual_edges=n_ve,
        max_dbscan_edges=max_db,
        mask=mask,
    )
    b, n = x.shape[:2]
    n_tot = n + n_vn
    m = n * k + max_db + n_vn * n + n_ve
    assert g.x.shape == (b, n_tot, 4)
    assert g.incidence.shape == (b, m, n_tot)
    assert g.edge_x.shape == (b, m, 4)
    assert g.edge_type.shape == (b, m)
    assert g.node_type.shape == (b, n_tot)
    assert g.node_mask.shape == (b, n_tot)
    assert g.particle_mask.shape == (b, n)
    assert torch.equal(g.particle_mask, mask)
    assert (g.node_type[:, :n] == NODE_TYPE_PARTICLE).all()
    assert (g.node_type[:, n:] == NODE_TYPE_VIRTUAL).all()
    assert g.node_mask[:, n:].all()
    live = g.edge_mask
    norms_sq = g.edge_x[live].square().sum(dim=-1)
    torch.testing.assert_close(
        norms_sq, torch.full_like(norms_sq, 4.0), atol=1e-4, rtol=0
    )


def test_edge_type_slices() -> None:
    x, mask = _toy_jet_batch()
    k, n_vn, n_ve, max_db = 3, 1, 2, 8
    g = build_hierarchical_graph(
        x,
        k=k,
        eps=0.15,
        min_samples=2,
        n_virtual_nodes=n_vn,
        n_virtual_edges=n_ve,
        max_dbscan_edges=max_db,
        mask=mask,
    )
    n = x.size(1)
    n_knn = n * k
    # Active knn edges are typed KNN
    knn_active = g.edge_mask[:, :n_knn]
    assert (g.edge_type[:, :n_knn][knn_active] == EDGE_TYPE_KNN).all()
    # DBSCAN slots
    db = g.edge_mask[:, n_knn : n_knn + max_db]
    assert (g.edge_type[:, n_knn : n_knn + max_db][db] == EDGE_TYPE_DBSCAN).all()
    # At least one DBSCAN cluster on this toy
    assert int(db.sum()) >= 1
    # Virtual-node links
    vn0 = n_knn + max_db
    vn1 = vn0 + n_vn * n
    vn_active = g.edge_mask[:, vn0:vn1]
    assert (g.edge_type[:, vn0:vn1][vn_active] == EDGE_TYPE_VN_LINK).all()
    # Each active particle × each virtual node → one *link edge*
    assert int(vn_active[0].sum()) == int(mask[0].sum()) * n_vn
    # Virtual nodes are nodes only (``node_type``); they are not edge slots.
    assert (g.node_type[:, n:] == NODE_TYPE_VIRTUAL).all()
    assert set(g.edge_type[g.edge_mask].unique().tolist()).issubset(
        {EDGE_TYPE_KNN, EDGE_TYPE_DBSCAN, EDGE_TYPE_VN_LINK, EDGE_TYPE_VIRTUAL_HYPER}
    )
    # Virtual hyperedges cover all active particles, no virtual nodes
    ve = g.edge_mask[:, vn1:]
    assert ve.shape[1] == n_ve
    assert ve.all()
    assert (g.edge_type[:, vn1:] == EDGE_TYPE_VIRTUAL_HYPER).all()
    for e in range(n_ve):
        row = g.incidence[0, vn1 + e]
        assert torch.allclose(row[:n], mask[0].float())
        assert (row[n:] == 0).all()


def test_knn_self_loop_and_degree() -> None:
    x, mask = _toy_jet_batch(batch=1, n=8)
    g = build_hierarchical_graph(
        x, k=2, eps=0.08, min_samples=2, n_virtual_nodes=0, n_virtual_edges=0, max_dbscan_edges=4, mask=mask
    )
    n = 8
    # Each active particle should have k knn edges (self + 1 neighbor)
    for p in range(n):
        if not mask[0, p]:
            continue
        e0 = p * 2
        e1 = p * 2 + 1
        assert g.edge_mask[0, e0] and g.edge_mask[0, e1]
        assert g.incidence[0, e0, p] == 1.0  # self endpoint on first slot


def test_cache_payload_keys() -> None:
    x, mask = _toy_jet_batch(batch=1)
    g = build_hierarchical_graph(
        x, k=2, n_virtual_nodes=1, n_virtual_edges=1, max_dbscan_edges=4, mask=mask
    )
    payload = g.to_cache_payload()
    for key in (
        "x",
        "edge_x",
        "edge_type",
        "edge_mask",
        "node_type",
        "node_mask",
        "particle_mask",
        "mask",
        "z",
        "incidence",
        "incidence_node",
        "incidence_edge",
        "incidence_nnz",
        "node_degree_inv",
        "edge_degree_inv",
    ):
        assert key in payload, key


def test_zero_virtuals() -> None:
    x, mask = _toy_jet_batch(batch=1)
    g = build_hierarchical_graph(
        x, k=2, n_virtual_nodes=0, n_virtual_edges=0, max_dbscan_edges=4, mask=mask
    )
    assert g.x.shape[1] == x.shape[1]
    assert g.n_virtual_nodes == 0
    assert g.n_virtual_hyperedges == 0
    assert not (g.edge_type == EDGE_TYPE_VN_LINK).any()
    assert not (g.edge_type == EDGE_TYPE_VIRTUAL_HYPER).any()


def test_drop_knn_edges_from_batch_compacts() -> None:
    from cpen.utils.graph_hierarchical import drop_knn_edges_from_batch
    from cpen.utils.sparse_incidence import finalize_incidence_storage

    x, mask = _toy_jet_batch(batch=2)
    g = build_hierarchical_graph(
        x, k=4, n_virtual_nodes=1, n_virtual_edges=1, max_dbscan_edges=8, mask=mask
    )
    n_knn_live = int(((g.edge_type == EDGE_TYPE_KNN) & g.edge_mask).sum())
    n_other_live = int(((g.edge_type != EDGE_TYPE_KNN) & g.edge_mask).sum())
    assert n_knn_live > 0
    assert n_other_live > 0

    fields = finalize_incidence_storage(g.incidence)
    batch = {
        "x": torch.zeros(g.x.size(0), g.x.size(1), 7),
        "edge_x": g.edge_x,
        "edge_type": g.edge_type,
        "edge_mask": g.edge_mask,
        "incidence_node": fields["incidence_node"],
        "incidence_edge": fields["incidence_edge"],
        "incidence_nnz": fields["incidence_nnz"],
        "node_mask": g.node_mask,
        "mask": g.node_mask,
    }
    out = drop_knn_edges_from_batch(batch)
    assert out["edge_x"].size(1) < g.edge_x.size(1)
    assert not ((out["edge_type"] == EDGE_TYPE_KNN) & out["edge_mask"]).any()
    live_types = set(out["edge_type"][out["edge_mask"]].unique().tolist())
    assert EDGE_TYPE_KNN not in live_types
    assert live_types <= {EDGE_TYPE_DBSCAN, EDGE_TYPE_VN_LINK, EDGE_TYPE_VIRTUAL_HYPER}
    assert int(out["edge_mask"].sum()) == n_other_live
    # Incidence rows match remaining live edges; no kNN endpoints left in COO.
    assert out["incidence"].shape[1] == out["edge_x"].size(1)
    assert int(out["incidence"].any(dim=-1).sum()) == n_other_live


def test_drop_pairwise_edges_from_batch_drops_vn_links() -> None:
    from cpen.utils.graph_hierarchical import drop_pairwise_edges_from_batch
    from cpen.utils.sparse_incidence import finalize_incidence_storage

    x, mask = _toy_jet_batch(batch=2)
    g = build_hierarchical_graph(
        x, k=4, n_virtual_nodes=1, n_virtual_edges=1, max_dbscan_edges=8, mask=mask
    )
    deg = g.incidence.to(torch.int64).sum(dim=-1)
    n_hyper_live = int(((deg > 2) & g.edge_mask).sum())
    assert int(((g.edge_type == EDGE_TYPE_VN_LINK) & g.edge_mask).sum()) > 0
    assert n_hyper_live > 0

    fields = finalize_incidence_storage(g.incidence)
    batch = {
        "x": torch.zeros(g.x.size(0), g.x.size(1), 7),
        "edge_x": g.edge_x,
        "edge_type": g.edge_type,
        "edge_mask": g.edge_mask,
        "incidence_node": fields["incidence_node"],
        "incidence_edge": fields["incidence_edge"],
        "incidence_nnz": fields["incidence_nnz"],
        "node_mask": g.node_mask,
        "mask": g.node_mask,
    }
    out = drop_pairwise_edges_from_batch(batch)
    assert out["edge_x"].size(1) < g.edge_x.size(1)
    live_types = set(out["edge_type"][out["edge_mask"]].unique().tolist())
    assert live_types <= {EDGE_TYPE_DBSCAN, EDGE_TYPE_VIRTUAL_HYPER}
    assert EDGE_TYPE_KNN not in live_types
    assert EDGE_TYPE_VN_LINK not in live_types
    assert int(out["edge_mask"].sum()) == n_hyper_live
    assert int((out["incidence"].sum(dim=-1) > 2).sum()) == n_hyper_live
    # Virtual node remains in X; only its 2-edges are gone.
    assert out["node_mask"].shape == g.node_mask.shape
    assert bool(out["node_mask"][:, g.n_particles :].all())


def test_dbscan_edge_features_match_cluster_mass_then_l2() -> None:
    from cpen.utils.graphs import scale_features_l2_sqrt_dim
    from cpen.utils.part_kin import build_part_interaction_features, leading_split_four_vectors

    x, mask = _toy_jet_batch(batch=1, n=16)
    g = build_hierarchical_graph(
        x, k=2, eps=0.15, min_samples=2, n_virtual_nodes=0, n_virtual_edges=0, max_dbscan_edges=8, mask=mask
    )
    n = x.size(1)
    db = (g.edge_type[0] == EDGE_TYPE_DBSCAN) & g.edge_mask[0]
    assert int(db.sum()) >= 1
    for e in db.nonzero(as_tuple=False).flatten().tolist():
        members = g.incidence[0, e, :n].bool()
        members_idx = members.nonzero(as_tuple=False).flatten()
        hard, rest = leading_split_four_vectors(x[0], members_idx)
        expected = scale_features_l2_sqrt_dim(
            build_part_interaction_features(
                hard.unsqueeze(0), rest.unsqueeze(0), l2_normalize=False
            )
        ).squeeze(0)
        torch.testing.assert_close(g.edge_x[0, e], expected.to(g.edge_x.dtype), atol=1e-4, rtol=0)
        norms_sq = g.edge_x[0, e].square().sum()
        torch.testing.assert_close(norms_sq, torch.tensor(4.0, dtype=norms_sq.dtype), atol=1e-4, rtol=0)

