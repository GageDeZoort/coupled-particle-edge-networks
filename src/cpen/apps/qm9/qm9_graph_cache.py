"""QM9 → CPEN padded COO incidence cache (graph-level regression targets).

Layout matches the MNIST incidence cache except ``y`` is ``(n_graphs, 19)``
float32: every PyG target column is stored so ``--qm9-target`` can be changed
without rebuilding. Graphs are the RDKit **bond** graph with hydrogens, edges
deduplicated into canonical unordered 2-edges.

Edge features are the 4-d bond-type one-hot plus the **interatomic distance**
from ``pos``. Dipole moment is a geometric quantity, so distance is what gives
the edge stream physical content; upstream ``hp-transfer-gts`` caches ``pos``
and never reads it.

Node and edge features are standardized per feature using **train-split**
statistics, which are written to ``meta.json``. QM9 columns are heterogeneous
(one-hot flags next to atomic number and H counts), so the row-\\(L^2\\) scaling
used by the Pascal / MNIST caches would conflate them. Build ``train`` first:
``val`` / ``test`` reuse the persisted train statistics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from cpen.graphs.undirected import canonical_undirected_edges
from cpen.training.log_utils import log_info

QM9_CACHE_VERSION = 1
QM9_NAME = "qm9"
QM9_SPLITS = ("train", "val", "test")

# Standard QM9 benchmark protocol (Gilmer et al.; PyG examples): 110k train,
# 10k val, all remaining molecules (10,831 of 130,831) as test.
N_TRAIN = 110_000
N_VAL = 10_000
SPLIT_SEED = 42
QM9_SPLIT_SIZES = {"train": N_TRAIN, "val": N_VAL, "test": 10_831}

# Verified against torch_geometric 2.5.3 ``QM9`` docstring. The upstream repo
# maps "dipole" to index 4, which is the HOMO-LUMO gap, so these names are
# spelled out here with units to keep that mistake from recurring.
QM9_TARGETS: dict[str, tuple[int, str]] = {
    "mu": (0, "D"),
    "alpha": (1, "a0^3"),
    "homo": (2, "eV"),
    "lumo": (3, "eV"),
    "gap": (4, "eV"),
    "r2": (5, "a0^2"),
    "zpve": (6, "eV"),
    "u0": (7, "eV"),
    "u": (8, "eV"),
    "h": (9, "eV"),
    "g": (10, "eV"),
    "cv": (11, "cal/(mol K)"),
}
DEFAULT_TARGET = "mu"
N_TARGET_COLUMNS = 19
# Largest molecule in QM9 has 29 atoms (hydrogens included).
MAX_NODES = 29
# Numerically dead columns (a one-hot that never fires on a split) are left
# untouched rather than amplified by a ~0 standard deviation.
_MIN_STD = 1e-6


def resolve_target(name: str | None) -> tuple[str, int, str]:
    """Return ``(name, column index, unit)`` for a QM9 target."""
    key = str(name or DEFAULT_TARGET).strip().lower()
    if key not in QM9_TARGETS:
        raise ValueError(
            f"Unknown QM9 target {name!r}. Choose from {sorted(QM9_TARGETS)}."
        )
    index, unit = QM9_TARGETS[key]
    return key, index, unit


def qm9_raw_root(data_root: str | Path) -> Path:
    return Path(data_root) / "QM9"


def qm9_cache_dir(data_root: str | Path) -> Path:
    return Path(data_root) / "processed" / QM9_NAME / f"incidence-v{QM9_CACHE_VERSION}"


def qm9_cache_path(data_root: str | Path, split: str) -> Path:
    return qm9_cache_dir(data_root) / f"{split}.pt"


def qm9_cache_available(data_root: str | Path, split: str) -> bool:
    return qm9_cache_path(data_root, split).is_file()


def missing_qm9_cache_message(data_root: str | Path) -> str:
    return (
        f"QM9 incidence cache missing under {qm9_cache_dir(data_root)}. "
        "Build with: python scans/qm9/build_qm9_graphs.py --data-root <root> "
        "or sbatch scans/qm9/build_qm9_graphs.slurm"
    )


def load_qm9_meta(data_root: str | Path) -> dict[str, Any]:
    path = qm9_cache_dir(data_root) / "meta.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def qm9_split_size(data_root: str | Path, split: str) -> int:
    meta = load_qm9_meta(data_root)
    entry = (meta.get("splits") or {}).get(split) or {}
    if "graphs" in entry:
        return int(entry["graphs"])
    return QM9_SPLIT_SIZES[split]


def qm9_target_stats(data_root: str | Path, target: str | None = None) -> tuple[float, float]:
    """Train-split ``(mean, std)`` of one target, in its physical unit."""
    name, _, _ = resolve_target(target)
    meta = load_qm9_meta(data_root)
    if not meta:
        raise FileNotFoundError(missing_qm9_cache_message(data_root))
    stats = (meta.get("target_stats") or {}).get(name)
    if not stats:
        raise KeyError(
            f"No cached train statistics for QM9 target {name!r} under "
            f"{qm9_cache_dir(data_root)}. Rebuild the train split."
        )
    return float(stats["mean"]), float(stats["std"])


def split_indices(n_total: int, split: str) -> list[int]:
    """Fixed-permutation 110k / 10k / rest benchmark split."""
    if split not in QM9_SPLITS:
        raise ValueError(f"split must be one of {QM9_SPLITS}, got {split!r}")
    if n_total < N_TRAIN + N_VAL:
        raise ValueError(
            f"QM9 benchmark split needs >= {N_TRAIN + N_VAL} molecules, got {n_total}"
        )
    gen = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(n_total, generator=gen)
    if split == "train":
        return perm[:N_TRAIN].tolist()
    if split == "val":
        return perm[N_TRAIN : N_TRAIN + N_VAL].tolist()
    return perm[N_TRAIN + N_VAL :].tolist()


def graph_to_cache_row(data: Any) -> dict[str, torch.Tensor]:
    """One molecule → raw (unstandardized) cache row."""
    x = data.x.to(torch.float32)
    pos = data.pos.to(torch.float32)
    n_nodes = int(x.size(0))
    edge_index = data.edge_index.to(torch.long)
    bond = data.edge_attr.to(torch.float32)
    if bond.dim() == 1:
        bond = bond.unsqueeze(-1)
    src, dst = edge_index
    dist = (pos[src] - pos[dst]).norm(dim=-1, keepdim=True)
    pairs, edge_x = canonical_undirected_edges(
        edge_index, torch.cat([bond, dist], dim=-1), num_nodes=n_nodes
    )
    n_edges = int(pairs.size(1))
    if n_edges == 0:
        # Single heavy atom with no bonds cannot happen in QM9 (every molecule
        # has >= 1 bond), but a zero-edge graph would break the COO layout.
        raise ValueError(f"molecule {getattr(data, 'name', '?')} has no bonds")
    incidence_node = pairs.t().reshape(-1).to(torch.int16)
    incidence_edge = (
        torch.arange(n_edges, dtype=torch.int16).unsqueeze(1).expand(n_edges, 2).reshape(-1)
    )
    node_degree = torch.bincount(incidence_node.to(torch.long), minlength=n_nodes).to(
        torch.float32
    )
    return {
        "x": x,
        "edge_x": edge_x,
        "y": data.y.reshape(-1).to(torch.float32),
        "incidence_node": incidence_node,
        "incidence_edge": incidence_edge,
        "incidence_nnz": torch.tensor(2 * n_edges, dtype=torch.int32),
        "node_degree_inv": node_degree.clamp_min(1).reciprocal(),
        "edge_degree_inv": torch.full((n_edges,), 0.5, dtype=torch.float32),
        "n_nodes": torch.tensor(n_nodes, dtype=torch.int32),
        "n_edges": torch.tensor(n_edges, dtype=torch.int32),
    }


def _column_stats(values: torch.Tensor) -> tuple[list[float], list[float]]:
    mean = values.mean(dim=0)
    std = values.std(dim=0)
    dead = std < _MIN_STD
    mean = torch.where(dead, torch.zeros_like(mean), mean)
    std = torch.where(dead, torch.ones_like(std), std)
    return mean.tolist(), std.tolist()


def feature_stats_from_rows(
    rows: list[dict[str, torch.Tensor]],
) -> dict[str, list[float]]:
    """Per-feature node / edge statistics over every token in *rows*."""
    node_mean, node_std = _column_stats(torch.cat([row["x"] for row in rows], dim=0))
    edge_mean, edge_std = _column_stats(
        torch.cat([row["edge_x"] for row in rows], dim=0)
    )
    return {
        "node_mean": node_mean,
        "node_std": node_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
    }


def _standardize_rows(
    rows: list[dict[str, torch.Tensor]],
    stats: dict[str, list[float]],
) -> None:
    node_mean = torch.tensor(stats["node_mean"], dtype=torch.float32)
    node_std = torch.tensor(stats["node_std"], dtype=torch.float32)
    edge_mean = torch.tensor(stats["edge_mean"], dtype=torch.float32)
    edge_std = torch.tensor(stats["edge_std"], dtype=torch.float32)
    for row in rows:
        row["x"] = (row["x"] - node_mean) / node_std
        row["edge_x"] = (row["edge_x"] - edge_mean) / edge_std


def _stack_rows(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    n_graphs = len(rows)
    max_nodes = max(int(row["n_nodes"]) for row in rows)
    max_edges = max(int(row["n_edges"]) for row in rows)
    max_nnz = 2 * max_edges
    n_features = rows[0]["x"].size(-1)
    n_edge_features = rows[0]["edge_x"].size(-1)
    n_targets = rows[0]["y"].numel()
    payload = {
        "x": torch.zeros(n_graphs, max_nodes, n_features, dtype=torch.float32),
        "edge_x": torch.zeros(n_graphs, max_edges, n_edge_features, dtype=torch.float32),
        "y": torch.zeros(n_graphs, n_targets, dtype=torch.float32),
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


def _target_stats(y: torch.Tensor) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for name, (index, unit) in QM9_TARGETS.items():
        column = y[:, index].to(torch.float64)
        stats[name] = {
            "index": index,
            "unit": unit,
            "mean": float(column.mean()),
            "std": float(column.std().clamp_min(_MIN_STD)),
        }
    return stats


def build_qm9_graph_cache(
    *,
    data_root: str | Path,
    split: str,
    rebuild: bool = False,
    pyg_root: str | Path | None = None,
) -> Path:
    """Convert one benchmark split into the mmap incidence cache.

    ``train`` must be built before ``val`` / ``test``: standardization uses
    train statistics, which are persisted to ``meta.json`` by the train build.
    """
    if split not in QM9_SPLITS:
        raise ValueError(f"split must be one of {QM9_SPLITS}, got {split!r}")
    output = qm9_cache_path(data_root, split)
    if output.is_file() and not rebuild:
        log_info(f"[qm9-cache] exists: {output}")
        return output

    try:
        from torch_geometric.datasets import QM9
    except ImportError as exc:
        raise ImportError("Building QM9 caches requires torch-geometric") from exc

    root = Path(pyg_root) if pyg_root is not None else qm9_raw_root(data_root)
    raw = QM9(root=str(root))
    indices = split_indices(len(raw), split)
    log_info(f"[qm9-cache] converting {split}: {len(indices)} molecules from {root}")
    rows = [graph_to_cache_row(raw[i]) for i in indices]

    meta_path = qm9_cache_dir(data_root) / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    if split == "train":
        stats = feature_stats_from_rows(rows)
        meta["feature_stats"] = stats
        meta["target_stats"] = _target_stats(
            torch.stack([row["y"] for row in rows], dim=0)
        )
    else:
        stats = meta.get("feature_stats")
        if not stats:
            raise RuntimeError(
                "QM9 val/test standardization needs train statistics. Build the "
                "train split first: python scans/qm9/build_qm9_graphs.py "
                "--data-root <root> --splits train"
            )
    _standardize_rows(rows, stats)

    payload = _stack_rows(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    meta.update(
        {
            "version": QM9_CACHE_VERSION,
            "dataset": QM9_NAME,
            "source": "torch_geometric.datasets.QM9",
            "node_features": int(payload["x"].size(-1)),
            "edge_features": int(payload["edge_x"].size(-1)),
            "out_dim": 1,
            "task": "graph-reg",
            "targets": {name: {"index": i, "unit": u} for name, (i, u) in QM9_TARGETS.items()},
            "default_target": DEFAULT_TARGET,
            "n_train": N_TRAIN,
            "n_val": N_VAL,
            "split_seed": SPLIT_SEED,
            "split_protocol": "benchmark-110k-10k-rest",
            "edge_policy": "canonical-undirected-two-node-incidence",
            "edge_construction": "rdkit-bonds",
            "edge_feature_layout": "bond-type-onehot+interatomic-distance",
            "node_feature_layout": "pyg-qm9-atom-features",
            "feature_normalization": "per-feature-standardize-train-stats",
            "hydrogens": True,
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
    log_info(f"[qm9-cache] wrote {output}")
    return output


def open_qm9_graph_cache(
    data_root: str | Path,
    split: str,
) -> dict[str, torch.Tensor]:
    path = qm9_cache_path(data_root, split)
    if not path.is_file():
        raise FileNotFoundError(missing_qm9_cache_message(data_root))
    return torch.load(path, map_location="cpu", mmap=True, weights_only=False)


class CachedQM9GraphDataset(Dataset):
    """Logical indices backed by sorted vectorized reads from a mmap split."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        split: str,
        target: str | None = None,
        max_graphs: int | None = None,
        seed: int = 42,
        include_incidence: bool = True,
        include_edge_features: bool = True,
    ) -> None:
        self.data_root = str(data_root)
        self.split = split
        self.target_name, self.target_index, self.target_unit = resolve_target(target)
        self.include_incidence = include_incidence
        self.include_edge_features = include_edge_features
        self._data: dict[str, torch.Tensor] | None = None
        n_total = qm9_split_size(data_root, split)
        if max_graphs is not None and max_graphs < n_total:
            rng = np.random.default_rng(seed + QM9_SPLITS.index(split))
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
            self._data = open_qm9_graph_cache(self.data_root, self.split)

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

        keys = ["x", "mask", "n_nodes"]
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
        batch = {key: self._data[key][sorted_indices][restore] for key in keys}
        # Raw target in its physical unit; the LightningModule standardizes.
        batch["y"] = self._data["y"][sorted_indices][restore][:, self.target_index]
        return batch
