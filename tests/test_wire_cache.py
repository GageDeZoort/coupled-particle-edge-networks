"""Tests for WIRE coordinates baked into primary star / Pascal caches."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from cpen.models.capen_llama import CAPENLlama
from cpen.utils.pascal_graph_cache import (
    _stack_rows,
    graph_to_cache_row,
)
from cpen.utils.sparse_incidence import finalize_incidence_storage
from cpen.utils.star_graph_cache import (
    CachedStarGraphJetDataset,
    star_cache_meta,
    star_radius_tag,
)
from cpen.utils.wire_coordinates import (
    compute_wire_coordinates_batched,
    particle_adjacency_from_incidence,
)


def _write_mini_star_cache_with_wire(
    tmp: Path, *, radius: float = 0.15, n: int = 6, m: int = 3
) -> Path:
    cache_dir = tmp / "processed" / star_radius_tag(radius) / f"n{n}"
    cache_dir.mkdir(parents=True)
    incidence = torch.zeros(2, n, n, dtype=torch.bool)
    for b, alive in enumerate((4, 3)):
        for i in range(alive):
            incidence[b, i, i] = True
            if i + 1 < alive:
                incidence[b, i, i + 1] = True
                incidence[b, i + 1, i] = True
    mask = torch.zeros(2, n, dtype=torch.bool)
    mask[0, :4] = True
    mask[1, :3] = True
    sparse = finalize_incidence_storage(incidence)
    adj = particle_adjacency_from_incidence(incidence, mask=mask)
    wire = compute_wire_coordinates_batched(adj, mask, m, warn_on_pad=False)
    payload = {
        "x": torch.randn(2, n, 3),
        "edge_x": torch.randn(2, n, 4),
        **sparse,
        "mask": mask,
        "z": mask.float() / mask.float().sum(dim=-1, keepdim=True).clamp_min(1),
        "labels": torch.tensor([0, 1]),
        "wire_coordinates": wire,
    }
    torch.save(payload, cache_dir / "train.pt")
    meta = star_cache_meta(
        radius=radius,
        num_particles=n,
        max_jets=None,
        seed=42,
        wire_coordinate_dim=m,
    )
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (cache_dir / "split_counts.json").write_text(json.dumps({"train": 2}))
    return cache_dir


def test_star_dataset_collates_primary_wire(tmp_path: Path) -> None:
    radius, n, m = 0.15, 6, 3
    _write_mini_star_cache_with_wire(tmp_path, radius=radius, n=n, m=m)
    ds = CachedStarGraphJetDataset(
        data_root=str(tmp_path),
        split="train",
        radius=radius,
        num_particles=n,
        wire_coordinate_dim=m,
        require_wire_cache=True,
    )
    batch = ds.collate_samples([0, 1])
    assert "wire_coordinates" in batch
    assert batch["wire_coordinates"].shape == (2, n, m)


def test_star_dataset_requires_wire_when_asked(tmp_path: Path) -> None:
    radius, n = 0.15, 6
    cache_dir = tmp_path / "processed" / star_radius_tag(radius) / f"n{n}"
    cache_dir.mkdir(parents=True)
    incidence = torch.eye(n, dtype=torch.bool).unsqueeze(0).repeat(1, 1, 1)
    mask = torch.ones(1, n, dtype=torch.bool)
    sparse = finalize_incidence_storage(incidence)
    torch.save(
        {
            "x": torch.randn(1, n, 3),
            "edge_x": torch.randn(1, n, 4),
            **sparse,
            "mask": mask,
            "z": mask.float(),
            "labels": torch.tensor([0]),
        },
        cache_dir / "train.pt",
    )
    meta = star_cache_meta(radius=radius, num_particles=n, max_jets=None, seed=42)
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (cache_dir / "split_counts.json").write_text(json.dumps({"train": 1}))

    ds = CachedStarGraphJetDataset(
        data_root=str(tmp_path),
        split="train",
        radius=radius,
        num_particles=n,
        wire_coordinate_dim=4,
        require_wire_cache=True,
    )
    try:
        ds.collate_samples([0])
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised


def test_pascal_row_includes_wire_coordinates() -> None:
    from types import SimpleNamespace

    data = SimpleNamespace(
        x=torch.randn(4, 14),
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        edge_attr=torch.randn(6, 2),
        y=torch.tensor([0, 1, 2, 3]),
    )
    for i in range(0, 6, 2):
        data.edge_attr[i + 1] = data.edge_attr[i]
    row = graph_to_cache_row(data, wire_coordinate_dim=3)
    assert row["wire_coordinates"].shape == (4, 3)
    stacked = _stack_rows([row], wire_coordinate_dim=3)
    assert stacked["wire_coordinates"].shape[0] == 1
    assert stacked["wire_coordinates"].shape[-1] == 3


def test_capen_llama_uses_batch_wire_on_pascal_shapes() -> None:
    torch.manual_seed(0)
    model = CAPENLlama(
        n_features=14,
        n_edge_features=2,
        out_dim=21,
        depth=1,
        width=8,
        heads=2,
        use_wire=True,
        wire_coordinate_dim=2,
        readout_mode="node",
    )
    n, e = 5, 3
    x = torch.randn(1, n, 14)
    edge_x = torch.randn(1, e, 2)
    incidence = torch.zeros(1, e, n, dtype=torch.bool)
    incidence[0, 0, 0] = incidence[0, 0, 1] = True
    incidence[0, 1, 1] = incidence[0, 1, 2] = True
    incidence[0, 2, 2] = incidence[0, 2, 3] = True
    mask = torch.ones(1, n, dtype=torch.bool)
    coords = torch.randn(1, n, 2)
    out = model(x, edge_x, incidence, mask=mask, wire_coordinates=coords)
    assert out.shape == (1, n, 21)
    assert torch.isfinite(out).all()
