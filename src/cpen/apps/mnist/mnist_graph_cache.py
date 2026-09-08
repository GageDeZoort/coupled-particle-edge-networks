"""MNISTSuperpixels → CPEN padded COO incidence cache (graph-level labels).

Layout matches Pascal incidence-v2 except ``y`` is ``(n_graphs,)`` (digit),
not per-superpixel. Node features are ``[intensity, pos]`` with the same
row-\(L^2\) \(\times\sqrt{n_0}\) scale as Pascal / hp-transfer CustomMNIST.
Self-loops are dropped; edges are canonical undirected 2-edges.
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
from cpen.training.log_utils import log_info

MNIST_CACHE_VERSION = 1
MNIST_NAME = "mnist-sp"
MNIST_SPLITS = ("train", "val", "test")
VAL_FRACTION = 0.1
SPLIT_SEED = 42
# Official 60k train → 90/10; official 10k test (hp-transfer-gts).
MNIST_SPLIT_SIZES = {"train": 54000, "val": 6000, "test": 10000}


def mnist_raw_root(data_root: str | Path) -> Path:
    return Path(data_root) / "MNISTSuperpixels"


def mnist_cache_dir(data_root: str | Path) -> Path:
    return Path(data_root) / "processed" / MNIST_NAME / f"incidence-v{MNIST_CACHE_VERSION}"


def mnist_cache_path(data_root: str | Path, split: str) -> Path:
    return mnist_cache_dir(data_root) / f"{split}.pt"


def mnist_cache_available(data_root: str | Path, split: str) -> bool:
    return mnist_cache_path(data_root, split).is_file()


def missing_mnist_cache_message(data_root: str | Path) -> str:
    return (
        f"MNIST Superpixels incidence cache missing under {mnist_cache_dir(data_root)}. "
        "Build with: python scans/mnist/build_mnist_graphs.py --data-root <root> "
        "or sbatch scans/mnist/build_mnist_graphs.slurm"
    )


def load_mnist_meta(data_root: str | Path) -> dict[str, Any]:
    path = mnist_cache_dir(data_root) / "meta.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def mnist_split_size(data_root: str | Path, split: str) -> int:
    meta = load_mnist_meta(data_root)
    splits = meta.get("splits") or {}
    entry = splits.get(split) or {}
    if "graphs" in entry:
        return int(entry["graphs"])
    return MNIST_SPLIT_SIZES[split]


def _row_l2_sqrt_dim(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.to(torch.float32), p=2, dim=-1) * math.sqrt(x.size(-1))


def graph_to_cache_row(data: Any) -> dict[str, torch.Tensor]:
    intensity = data.x.to(torch.float32)
    if intensity.dim() == 1:
        intensity = intensity.unsqueeze(-1)
    pos = data.pos.to(torch.float32)
    x = _row_l2_sqrt_dim(torch.cat([intensity, pos], dim=-1))
    n_nodes = int(x.size(0))
    edge_index = data.edge_index.to(torch.long)
    if getattr(data, "edge_attr", None) is not None:
        edge_attr = data.edge_attr.to(torch.float32)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
    else:
        src, dst = edge_index
        edge_attr = (pos[src] - pos[dst]).abs()
    pairs, edge_x = canonical_undirected_edges(edge_index, edge_attr, num_nodes=n_nodes)
    edge_x = _row_l2_sqrt_dim(edge_x)
    n_edges = int(pairs.size(1))
    incidence_node = pairs.t().reshape(-1).to(torch.int16)
    incidence_edge = (
        torch.arange(n_edges, dtype=torch.int16).unsqueeze(1).expand(n_edges, 2).reshape(-1)
    )
    node_degree = torch.bincount(incidence_node.to(torch.long), minlength=n_nodes).to(
        torch.float32
    )
    y = data.y.reshape(-1)[0].to(torch.long)
    return {
        "x": x,
        "edge_x": edge_x,
        "y": y,
        "incidence_node": incidence_node,
        "incidence_edge": incidence_edge,
        "incidence_nnz": torch.tensor(2 * n_edges, dtype=torch.int32),
        "node_degree_inv": node_degree.clamp_min(1).reciprocal(),
        "edge_degree_inv": torch.full((n_edges,), 0.5, dtype=torch.float32),
        "n_nodes": torch.tensor(n_nodes, dtype=torch.int32),
        "n_edges": torch.tensor(n_edges, dtype=torch.int32),
    }


def _stack_rows(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    n_graphs = len(rows)
    max_nodes = max(int(row["n_nodes"]) for row in rows)
    max_edges = max(int(row["n_edges"]) for row in rows)
    max_nnz = 2 * max_edges
    n_features = rows[0]["x"].size(-1)
    n_edge_features = rows[0]["edge_x"].size(-1)
    payload = {
        "x": torch.zeros(n_graphs, max_nodes, n_features, dtype=torch.float32),
        "edge_x": torch.zeros(n_graphs, max_edges, n_edge_features, dtype=torch.float32),
        "y": torch.zeros(n_graphs, dtype=torch.long),
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
    for index, row in enumerate(rows):
        n_nodes, n_edges = int(row["n_nodes"]), int(row["n_edges"])
        nnz = int(row["incidence_nnz"])
        payload["x"][index, :n_nodes] = row["x"]
        payload["edge_x"][index, :n_edges] = row["edge_x"]
        payload["y"][index] = row["y"]
        payload["mask"][index, :n_nodes] = True
        payload["edge_mask"][index, :n_edges] = True
        payload["incidence_node"][index, :nnz] = row["incidence_node"]
        payload["incidence_edge"][index, :nnz] = row["incidence_edge"]
        payload["incidence_nnz"][index] = nnz
        payload["node_degree_inv"][index, :n_nodes] = row["node_degree_inv"]
        payload["edge_degree_inv"][index, :n_edges] = row["edge_degree_inv"]
        payload["n_nodes"][index] = n_nodes
        payload["n_edges"][index] = n_edges
    return payload


def _official_indices(n_total: int, split: str) -> list[int]:
    """hp-transfer-gts: random_split on official train; official test unchanged."""
    if split == "test":
        return list(range(n_total))
    n_val = int(VAL_FRACTION * n_total)
    n_train = n_total - n_val
    gen = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(n_total, generator=gen)
    if split == "train":
        return perm[:n_train].tolist()
    if split == "val":
        return perm[n_train:].tolist()
    raise ValueError(f"split must be one of {MNIST_SPLITS}, got {split!r}")


def build_mnist_graph_cache(
    *,
    data_root: str | Path,
    split: str,
    rebuild: bool = False,
) -> Path:
    if split not in MNIST_SPLITS:
        raise ValueError(f"split must be one of {MNIST_SPLITS}, got {split!r}")
    output = mnist_cache_path(data_root, split)
    if output.is_file() and not rebuild:
        log_info(f"[mnist-cache] exists: {output}")
        return output

    try:
        from torch_geometric.datasets import MNISTSuperpixels
    except ImportError as exc:
        raise ImportError("Building MNIST caches requires torch-geometric") from exc

    official_train = split != "test"
    raw = MNISTSuperpixels(root=str(mnist_raw_root(data_root)), train=official_train)
    indices = _official_indices(len(raw), split)
    log_info(f"[mnist-cache] converting {split}: {len(indices)} graphs")
    rows = [graph_to_cache_row(raw[i]) for i in indices]
    payload = _stack_rows(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    meta_path = output.parent / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    meta.update(
        {
            "version": MNIST_CACHE_VERSION,
            "dataset": MNIST_NAME,
            "source": "torch_geometric.datasets.MNISTSuperpixels",
            "node_features": int(payload["x"].size(-1)),
            "edge_features": int(payload["edge_x"].size(-1)),
            "out_dim": 10,
            "task": "graph-cls",
            "val_fraction": VAL_FRACTION,
            "split_seed": SPLIT_SEED,
            "edge_policy": "canonical-undirected-two-node-incidence",
            "feature_normalization": "row-l2-sqrt-dimension",
            "node_feature_layout": "intensity+pos",
            "self_loops": False,
        }
    )
    split_shapes = meta.setdefault("splits", {})
    split_shapes[split] = {
        "graphs": len(indices),
        "max_nodes": int(payload["x"].size(1)),
        "max_edges": int(payload["edge_x"].size(1)),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    log_info(f"[mnist-cache] wrote {output}")
    return output


def open_mnist_graph_cache(
    data_root: str | Path,
    split: str,
) -> dict[str, torch.Tensor]:
    path = mnist_cache_path(data_root, split)
    if not path.is_file():
        raise FileNotFoundError(missing_mnist_cache_message(data_root))
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


class CachedMNISTGraphDataset(Dataset):
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
    ) -> None:
        self.data_root = str(data_root)
        self.split = split
        self.include_incidence = include_incidence
        self.include_edge_features = include_edge_features
        self._data: dict[str, torch.Tensor] | None = None
        n_total = mnist_split_size(data_root, split)
        if max_graphs is not None and max_graphs < n_total:
            rng = np.random.default_rng(seed + MNIST_SPLITS.index(split))
            self._indices = np.sort(rng.choice(n_total, size=max_graphs, replace=False))
        else:
            self._indices = None
        self._length = len(self._indices) if self._indices is not None else n_total

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_data"] = None
        return state

    def _ensure_open(self) -> None:
        if self._data is None:
            self._data = open_mnist_graph_cache(self.data_root, self.split)

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
        return {key: self._data[key][sorted_indices][restore] for key in keys}
