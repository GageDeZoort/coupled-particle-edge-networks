"""Cached PascalVOC-SP graphs in CPEN's padded COO incidence format.

``incidence-v2`` stores optional precomputed WIRE spectral coordinates in the
primary ``{split}.pt`` (same Lightning datamodule path as TopTagging star
caches). Adjacency for WIRE is the native undirected graph, not star
co-occurrence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from cpen.graphs.undirected import canonical_undirected_edges
from cpen.graphs.wire_coordinates import compute_wire_coordinates
from cpen.training.log_utils import log_info


PASCAL_CACHE_VERSION = 2
PASCAL_NAME = "pascalvoc-sp"
PASCAL_SPLITS = ("train", "val", "test")
PASCAL_SPLIT_SIZES = {"train": 8498, "val": 1428, "test": 1429}
DEFAULT_WIRE_COORDINATE_DIM = 8


def pascal_cache_dir(data_root: str | Path) -> Path:
    return Path(data_root) / "processed" / PASCAL_NAME / f"incidence-v{PASCAL_CACHE_VERSION}"


def pascal_cache_path(data_root: str | Path, split: str) -> Path:
    return pascal_cache_dir(data_root) / f"{split}.pt"


def pascal_cache_available(data_root: str | Path, split: str) -> bool:
    return (
        pascal_cache_path(data_root, split).is_file()
        and (pascal_cache_dir(data_root) / "meta.json").is_file()
    )


def missing_pascal_cache_message(data_root: str | Path) -> str:
    return (
        f"PascalVOC-SP incidence cache is missing under {pascal_cache_dir(data_root)}. "
        "Build it once with scans/testing/build_pascal_graphs.py "
        f"(cache version incidence-v{PASCAL_CACHE_VERSION})."
    )


def graph_to_cache_row(
    data: Any,
    *,
    wire_coordinate_dim: int | None = DEFAULT_WIRE_COORDINATE_DIM,
    wire_normalized_laplacian: bool = True,
    wire_standardize: bool = True,
    wire_canonicalize_sign: bool = True,
) -> dict[str, torch.Tensor]:
    """Convert one Pascal PyG graph to a two-node-hyperedge representation."""
    x = F.normalize(data.x.to(torch.float32), p=2, dim=-1) * math.sqrt(data.x.size(-1))
    pairs, edge_x = canonical_undirected_edges(
        data.edge_index,
        data.edge_attr,
        num_nodes=x.size(0),
    )
    edge_x = F.normalize(edge_x, p=2, dim=-1) * math.sqrt(edge_x.size(-1))
    n_edges = pairs.size(1)

    incidence_node = pairs.t().reshape(-1).to(torch.int16)
    incidence_edge = (
        torch.arange(n_edges, dtype=torch.int16).unsqueeze(1).expand(n_edges, 2).reshape(-1)
    )
    node_degree = torch.bincount(
        incidence_node.to(torch.long), minlength=x.size(0)
    ).to(torch.float32)
    row = {
        "x": x,
        "edge_x": edge_x,
        "y": data.y.reshape(-1).to(torch.long),
        "incidence_node": incidence_node,
        "incidence_edge": incidence_edge,
        "incidence_nnz": torch.tensor(2 * n_edges, dtype=torch.int32),
        "node_degree_inv": node_degree.clamp_min(1).reciprocal(),
        "edge_degree_inv": torch.full((n_edges,), 0.5, dtype=torch.float32),
        "n_nodes": torch.tensor(x.size(0), dtype=torch.int32),
        "n_edges": torch.tensor(n_edges, dtype=torch.int32),
    }
    if wire_coordinate_dim is not None:
        # Undirected pairs as edge_index; pad modes if the graph is tiny.
        row["wire_coordinates"] = compute_wire_coordinates(
            pairs,
            int(x.size(0)),
            int(wire_coordinate_dim),
            normalized_laplacian=wire_normalized_laplacian,
            standardize=wire_standardize,
            canonicalize_sign=wire_canonicalize_sign,
            warn_on_pad=False,
        )
    return row


def _stack_rows(
    rows: list[dict[str, torch.Tensor]],
    *,
    wire_coordinate_dim: int | None = None,
) -> dict[str, torch.Tensor]:
    n_graphs = len(rows)
    max_nodes = max(int(row["n_nodes"]) for row in rows)
    max_edges = max(int(row["n_edges"]) for row in rows)
    max_nnz = 2 * max_edges
    n_features = rows[0]["x"].size(-1)
    n_edge_features = rows[0]["edge_x"].size(-1)

    payload = {
        "x": torch.zeros(n_graphs, max_nodes, n_features, dtype=torch.float32),
        "edge_x": torch.zeros(n_graphs, max_edges, n_edge_features, dtype=torch.float32),
        "y": torch.full((n_graphs, max_nodes), -100, dtype=torch.long),
        "mask": torch.zeros(n_graphs, max_nodes, dtype=torch.bool),
        "edge_mask": torch.zeros(n_graphs, max_edges, dtype=torch.bool),
        "incidence_node": torch.zeros(n_graphs, max_nnz, dtype=torch.int16),
        "incidence_edge": torch.zeros(n_graphs, max_nnz, dtype=torch.int16),
        "incidence_nnz": torch.zeros(n_graphs, dtype=torch.int32),
        "node_degree_inv": torch.zeros(n_graphs, max_nodes, dtype=torch.float32),
        "edge_degree_inv": torch.zeros(n_graphs, max_edges, dtype=torch.float32),
        "n_nodes": torch.zeros(n_graphs, dtype=torch.int32),
        "n_edges": torch.zeros(n_graphs, dtype=torch.int32),
    }
    if wire_coordinate_dim is not None:
        payload["wire_coordinates"] = torch.zeros(
            n_graphs, max_nodes, wire_coordinate_dim, dtype=torch.float32
        )
    for index, row in enumerate(rows):
        n_nodes, n_edges = int(row["n_nodes"]), int(row["n_edges"])
        nnz = int(row["incidence_nnz"])
        payload["x"][index, :n_nodes] = row["x"]
        payload["edge_x"][index, :n_edges] = row["edge_x"]
        payload["y"][index, :n_nodes] = row["y"]
        payload["mask"][index, :n_nodes] = True
        payload["edge_mask"][index, :n_edges] = True
        payload["incidence_node"][index, :nnz] = row["incidence_node"]
        payload["incidence_edge"][index, :nnz] = row["incidence_edge"]
        payload["incidence_nnz"][index] = nnz
        payload["node_degree_inv"][index, :n_nodes] = row["node_degree_inv"]
        payload["edge_degree_inv"][index, :n_edges] = row["edge_degree_inv"]
        payload["n_nodes"][index] = n_nodes
        payload["n_edges"][index] = n_edges
        if wire_coordinate_dim is not None:
            payload["wire_coordinates"][index, :n_nodes] = row["wire_coordinates"]
    return payload


def build_pascal_graph_cache(
    *,
    data_root: str | Path,
    split: str,
    rebuild: bool = False,
    wire_coordinate_dim: int | None = DEFAULT_WIRE_COORDINATE_DIM,
    wire_normalized_laplacian: bool = True,
    wire_standardize: bool = True,
    wire_canonicalize_sign: bool = True,
) -> Path:
    """Download/load an official LRGB split and write the CPEN mmap cache."""
    if split not in PASCAL_SPLITS:
        raise ValueError(f"split must be one of {PASCAL_SPLITS}, got {split!r}")
    output = pascal_cache_path(data_root, split)
    if output.is_file() and not rebuild:
        log_info(f"[pascal-cache] exists: {output}")
        return output

    try:
        from torch_geometric.datasets import LRGBDataset
    except ImportError as exc:
        raise ImportError("Building PascalVOC-SP caches requires torch-geometric") from exc

    dataset = LRGBDataset(str(data_root), PASCAL_NAME, split=split)
    log_info(
        f"[pascal-cache] converting {split}: {len(dataset)} graphs "
        f"(wire_m={wire_coordinate_dim})"
    )
    rows = [
        graph_to_cache_row(
            dataset[index],
            wire_coordinate_dim=wire_coordinate_dim,
            wire_normalized_laplacian=wire_normalized_laplacian,
            wire_standardize=wire_standardize,
            wire_canonicalize_sign=wire_canonicalize_sign,
        )
        for index in range(len(dataset))
    ]
    payload = _stack_rows(rows, wire_coordinate_dim=wire_coordinate_dim)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    meta_path = output.parent / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    meta.update(
        {
            "version": PASCAL_CACHE_VERSION,
            "dataset": PASCAL_NAME,
            "node_features": int(payload["x"].size(-1)),
            "edge_features": int(payload["edge_x"].size(-1)),
            "edge_policy": "canonical-undirected-two-node-incidence",
            "feature_normalization": "row-l2-sqrt-dimension",
            "wire_source": "native-undirected-graph",
        }
    )
    if wire_coordinate_dim is not None:
        meta.update(
            {
                "wire_coordinate_dim": int(wire_coordinate_dim),
                "wire_normalized_laplacian": bool(wire_normalized_laplacian),
                "wire_standardize": bool(wire_standardize),
                "wire_canonicalize_sign": bool(wire_canonicalize_sign),
            }
        )
    else:
        for key in (
            "wire_coordinate_dim",
            "wire_normalized_laplacian",
            "wire_standardize",
            "wire_canonicalize_sign",
        ):
            meta.pop(key, None)
    split_shapes = meta.setdefault("splits", {})
    split_shapes[split] = {
        "graphs": len(dataset),
        "max_nodes": int(payload["x"].size(1)),
        "max_edges": int(payload["edge_x"].size(1)),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    log_info(f"[pascal-cache] wrote {output}")
    return output


def open_pascal_graph_cache(
    data_root: str | Path,
    split: str,
) -> dict[str, torch.Tensor]:
    path = pascal_cache_path(data_root, split)
    if not path.is_file():
        raise FileNotFoundError(missing_pascal_cache_message(data_root))
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


class CachedPascalGraphDataset(Dataset):
    """Logical indices backed by sorted vectorized reads from a mmap split."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        split: str,
        max_graphs: int | None = None,
        seed: int = 42,
        include_incidence: bool = True,
        include_edge_features: bool = True,
        wire_coordinate_dim: int | None = None,
        require_wire_cache: bool = False,
    ) -> None:
        self.data_root = str(data_root)
        self.split = split
        self.include_incidence = include_incidence
        self.include_edge_features = include_edge_features
        self.wire_coordinate_dim = wire_coordinate_dim
        self.require_wire_cache = require_wire_cache
        self._data: dict[str, torch.Tensor] | None = None
        self._include_wire = False
        n_total = PASCAL_SPLIT_SIZES[split]
        if max_graphs is not None and max_graphs < n_total:
            rng = np.random.default_rng(seed + PASCAL_SPLITS.index(split))
            self._indices = np.sort(rng.choice(n_total, size=max_graphs, replace=False))
        else:
            self._indices = None
        self._length = len(self._indices) if self._indices is not None else n_total

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_data"] = None
        return state

    def _ensure_open(self) -> None:
        if self._data is not None:
            return
        self._data = open_pascal_graph_cache(self.data_root, self.split)
        if self.wire_coordinate_dim is None:
            self._include_wire = False
            return
        has = "wire_coordinates" in self._data
        if has:
            if self._data["wire_coordinates"].size(-1) != self.wire_coordinate_dim:
                raise ValueError(
                    f"Pascal wire_coordinates dim "
                    f"{self._data['wire_coordinates'].size(-1)} != "
                    f"requested {self.wire_coordinate_dim}"
                )
            self._include_wire = True
            return
        if self.require_wire_cache:
            raise FileNotFoundError(
                f"Pascal cache split={self.split!r} has no wire_coordinates. "
                f"Rebuild with build_pascal_graphs.py --wire-coordinate-dim "
                f"{self.wire_coordinate_dim} (incidence-v{PASCAL_CACHE_VERSION})."
            )
        log_info(
            f"[wire] Pascal split={self.split} has no wire_coordinates; "
            "model will eigh on the fly"
        )
        self._include_wire = False

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> int:
        return index

    def collate_samples(self, logical_indices: list[int]) -> dict[str, torch.Tensor]:
        self._ensure_open()
        assert self._data is not None
        physical = (
            [int(self._indices[index]) for index in logical_indices]
            if self._indices is not None
            else [int(index) for index in logical_indices]
        )
        indices = torch.as_tensor(physical, dtype=torch.long)
        order = torch.argsort(indices)
        sorted_indices = indices[order]
        restore = torch.argsort(order)

        keys = ["x", "y", "mask", "n_nodes"]
        if self.include_incidence:
            keys += [
                "edge_mask",
                "incidence_node",
                "incidence_edge",
                "incidence_nnz",
                "node_degree_inv",
                "edge_degree_inv",
                "n_edges",
            ]
        if self.include_edge_features:
            keys.append("edge_x")
        if self._include_wire:
            keys.append("wire_coordinates")
        return {
            key: self._data[key][sorted_indices][restore]
            for key in keys
        }
