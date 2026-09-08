"""TopTagging hierarchical graph caches (kNN + DBSCAN + virtual nodes/edges).

Caches land under::

  {data-root}/processed/{hier_tag}/n{N}/{train,val,test}.pt

where ``hier_tag`` comes from :func:`hierarchical_construction_tag`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from cpen.utils.graph_hierarchical import (
    build_hierarchical_graph,
    hierarchical_construction_tag,
)
from cpen.utils.log_utils import log_info
from cpen.utils.part_kin import pt_fraction_weights
from cpen.utils.toptagging import (
    RawTopTaggingSplit,
    normalize_data_root,
    preprocess_particle_features,
    raw_hdf5_path,
)

PAYLOAD_KEYS: tuple[str, ...] = (
    "x",
    "x_model",
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
    "labels",
)


def processed_hier_split_dir(
    data_root: str | Path,
    *,
    k: int,
    eps: float = 0.08,
    min_samples: int = 2,
    n_virtual_nodes: int = 1,
    n_virtual_edges: int = 1,
    max_dbscan_edges: int = 32,
    num_particles: int,
) -> Path:
    tag = hierarchical_construction_tag(
        k=k,
        eps=eps,
        min_samples=min_samples,
        n_virtual_nodes=n_virtual_nodes,
        n_virtual_edges=n_virtual_edges,
        max_dbscan_edges=max_dbscan_edges,
    )
    return normalize_data_root(data_root) / "processed" / tag / f"n{int(num_particles)}"


def hier_cache_meta(
    *,
    k: int,
    eps: float,
    min_samples: int,
    n_virtual_nodes: int,
    n_virtual_edges: int,
    max_dbscan_edges: int,
    num_particles: int,
    max_jets: int | None,
    seed: int,
) -> dict[str, Any]:
    tag = hierarchical_construction_tag(
        k=k,
        eps=eps,
        min_samples=min_samples,
        n_virtual_nodes=n_virtual_nodes,
        n_virtual_edges=n_virtual_edges,
        max_dbscan_edges=max_dbscan_edges,
    )
    return {
        "construction": "hierarchical",
        "hierarchical_construction_tag": tag,
        "k": int(k),
        "eps": float(eps),
        "min_samples": int(min_samples),
        "n_virtual_nodes": int(n_virtual_nodes),
        "n_virtual_edges": int(n_virtual_edges),
        "max_dbscan_edges": int(max_dbscan_edges),
        "num_particles": int(num_particles),
        "max_jets": max_jets,
        "seed": int(seed),
        "particle_z": "pt-fraction",
        "edge_features": 4,
        "edge_feature_set": "part-int-v1-l2",
        "edge_normalization": "part-int-l2-sqrt-dim",
        "edge_feature_names": [
            "ln_delta",
            "ln_kt",
            "ln_z",
            "ln_m2",
        ],
        "storage": "sparse-bool-incidence-v1",
        "payload_keys": list(PAYLOAD_KEYS),
    }


def write_hier_cache_split_count(
    *,
    cache_dir: Path,
    split: str,
    n_jets: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "split_counts.json"
    counts: dict[str, int] = {}
    if path.is_file():
        counts = {str(k): int(v) for k, v in json.loads(path.read_text()).items()}
    counts[split] = int(n_jets)
    path.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")


def materialize_hier_batch(
    samples: list[dict[str, torch.Tensor]],
    *,
    k: int,
    eps: float = 0.08,
    min_samples: int = 2,
    n_virtual_nodes: int = 1,
    n_virtual_edges: int = 1,
    max_dbscan_edges: int = 32,
) -> dict[str, torch.Tensor]:
    """Build one batched hierarchical payload (including ``labels`` / ``x_model``)."""
    if not samples:
        raise ValueError("samples must be non-empty")
    x_raw = torch.stack([s["x_raw"] for s in samples], dim=0)
    mask = torch.stack([s["mask"] for s in samples], dim=0)
    x_model = torch.stack([s["x"] for s in samples], dim=0)
    labels = torch.stack([s["y"] for s in samples]).view(-1).to(torch.int64)
    g = build_hierarchical_graph(
        x_raw,
        k=k,
        eps=eps,
        min_samples=min_samples,
        n_virtual_nodes=n_virtual_nodes,
        n_virtual_edges=n_virtual_edges,
        max_dbscan_edges=max_dbscan_edges,
        mask=mask,
    )
    payload = g.to_cache_payload()
    payload["x_model"] = x_model
    payload["labels"] = labels
    return payload


class LiveHierJetDataset(Dataset):
    """Raw TopTagging jets as dicts for hierarchical materialization."""

    def __init__(
        self,
        *,
        data_root: str,
        split: str,
        num_particles: int = 128,
        max_jets: int | None = None,
        seed: int = 42,
    ) -> None:
        hdf5_path = raw_hdf5_path(data_root, split)
        if not hdf5_path.is_file():
            raise FileNotFoundError(
                f"Missing TopTagging split file {hdf5_path}. "
                "Download on a login node with --download."
            )
        log_info(f"[hier] live dataset split={split} from {hdf5_path.name}")
        self._backend = RawTopTaggingSplit(
            data_root=data_root,
            split=split,
            num_particles=num_particles,
        )
        n_total = len(self._backend)
        if max_jets is not None and max_jets < n_total:
            split_offset = {"train": 0, "val": 1, "test": 2}[split]
            rng = np.random.default_rng(seed + split_offset)
            self._indices = rng.choice(n_total, size=max_jets, replace=False)
        else:
            self._indices = np.arange(n_total)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        particles, jet = self._backend[int(self._indices[idx])]
        x_raw, x_model, mask = preprocess_particle_features(particles.float())
        return {
            "x_raw": x_raw,
            "x": x_model,
            "mask": mask,
            "z": pt_fraction_weights(x_raw, mask),
            "y": torch.tensor(int(jet[0].item()), dtype=torch.long),
        }


def read_hier_graph_cache_split_counts(
    data_root: str | Path,
    *,
    k: int,
    eps: float = 0.08,
    min_samples: int = 2,
    n_virtual_nodes: int = 1,
    n_virtual_edges: int = 1,
    max_dbscan_edges: int = 32,
    num_particles: int,
) -> dict[str, int] | None:
    cache_dir = processed_hier_split_dir(
        data_root,
        k=k,
        eps=eps,
        min_samples=min_samples,
        n_virtual_nodes=n_virtual_nodes,
        n_virtual_edges=n_virtual_edges,
        max_dbscan_edges=max_dbscan_edges,
        num_particles=num_particles,
    )
    path = cache_dir / "split_counts.json"
    if not path.is_file():
        return None
    return {str(k_): int(v) for k_, v in json.loads(path.read_text()).items()}


def hier_graph_cache_available(
    *,
    data_root: str | Path,
    split: str,
    k: int,
    eps: float = 0.08,
    min_samples: int = 2,
    n_virtual_nodes: int = 1,
    n_virtual_edges: int = 1,
    max_dbscan_edges: int = 32,
    num_particles: int,
) -> bool:
    cache_dir = processed_hier_split_dir(
        data_root,
        k=k,
        eps=eps,
        min_samples=min_samples,
        n_virtual_nodes=n_virtual_nodes,
        n_virtual_edges=n_virtual_edges,
        max_dbscan_edges=max_dbscan_edges,
        num_particles=num_particles,
    )
    return (cache_dir / f"{split}.pt").is_file()


def missing_hier_graph_cache_message(
    *,
    data_root: str | Path,
    split: str,
    k: int,
    eps: float = 0.08,
    min_samples: int = 2,
    n_virtual_nodes: int = 1,
    n_virtual_edges: int = 1,
    max_dbscan_edges: int = 32,
    num_particles: int,
) -> str:
    cache_dir = processed_hier_split_dir(
        data_root,
        k=k,
        eps=eps,
        min_samples=min_samples,
        n_virtual_nodes=n_virtual_nodes,
        n_virtual_edges=n_virtual_edges,
        max_dbscan_edges=max_dbscan_edges,
        num_particles=num_particles,
    )
    tag = hierarchical_construction_tag(
        k=k,
        eps=eps,
        min_samples=min_samples,
        n_virtual_nodes=n_virtual_nodes,
        n_virtual_edges=n_virtual_edges,
        max_dbscan_edges=max_dbscan_edges,
    )
    return (
        f"Missing pre-built hierarchical graph cache for split={split!r} at {cache_dir}/.\n"
        "Build once (sharded cputest jobs):\n"
        f"  bash scans/jets/submit_hier_graphs_cputest.sh\n"
        f"(writes processed/{tag}/n{int(num_particles)}/...)"
    )


def _subsample_indices(
    split: str,
    n_total: int,
    max_jets: int | None,
    seed: int,
) -> np.ndarray | None:
    if max_jets is None or max_jets >= n_total:
        return None
    split_offset = {"train": 0, "val": 1, "test": 2}.get(split, 0)
    rng = np.random.default_rng(seed + split_offset)
    return rng.choice(n_total, size=int(max_jets), replace=False)


class CachedHierJetDataset(Dataset):
    """Memory-mapped hierarchical TopTagging cache (``collate_samples`` batches).

    Returns model inputs with:

    * ``x`` — particle ParT-kin (``x_model``) padded with zeros for virtual nodes
    * ``mask`` / ``node_mask`` — all valid nodes (particles + virtuals)
    * ``edge_type`` / ``edge_mask`` — typed hypergraph rows for CAPEN-Llama-att
    """

    _TENSOR_KEYS: tuple[str, ...] = (
        "x_model",
        "x",
        "edge_x",
        "edge_type",
        "edge_mask",
        "node_type",
        "node_mask",
        "particle_mask",
        "z",
        "incidence_node",
        "incidence_edge",
        "incidence_nnz",
        "node_degree_inv",
        "edge_degree_inv",
    )

    def __init__(
        self,
        *,
        data_root: str,
        split: str,
        k: int = 8,
        eps: float = 0.08,
        min_samples: int = 2,
        n_virtual_nodes: int = 1,
        n_virtual_edges: int = 1,
        max_dbscan_edges: int = 32,
        num_particles: int = 128,
        max_jets: int | None = None,
        seed: int = 42,
    ) -> None:
        self.data_root = str(data_root)
        self.split = split
        self.k = int(k)
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.n_virtual_nodes = int(n_virtual_nodes)
        self.n_virtual_edges = int(n_virtual_edges)
        self.max_dbscan_edges = int(max_dbscan_edges)
        self.num_particles = int(num_particles)
        self.max_jets = max_jets
        self.seed = int(seed)
        self._indices: np.ndarray | None = None
        self._data: dict[str, torch.Tensor] | None = None

        if not hier_graph_cache_available(
            data_root=self.data_root,
            split=split,
            k=self.k,
            eps=self.eps,
            min_samples=self.min_samples,
            n_virtual_nodes=self.n_virtual_nodes,
            n_virtual_edges=self.n_virtual_edges,
            max_dbscan_edges=self.max_dbscan_edges,
            num_particles=self.num_particles,
        ):
            raise FileNotFoundError(
                missing_hier_graph_cache_message(
                    data_root=self.data_root,
                    split=split,
                    k=self.k,
                    eps=self.eps,
                    min_samples=self.min_samples,
                    n_virtual_nodes=self.n_virtual_nodes,
                    n_virtual_edges=self.n_virtual_edges,
                    max_dbscan_edges=self.max_dbscan_edges,
                    num_particles=self.num_particles,
                )
            )

        counts = read_hier_graph_cache_split_counts(
            self.data_root,
            k=self.k,
            eps=self.eps,
            min_samples=self.min_samples,
            n_virtual_nodes=self.n_virtual_nodes,
            n_virtual_edges=self.n_virtual_edges,
            max_dbscan_edges=self.max_dbscan_edges,
            num_particles=self.num_particles,
        )
        if counts is not None and split in counts:
            n_total = int(counts[split])
        else:
            self._ensure_open()
            assert self._data is not None
            n_total = int(self._data["labels"].size(0))

        self._indices = _subsample_indices(split, n_total, max_jets, seed)
        self._length = len(self._indices) if self._indices is not None else n_total

    def _cache_path(self) -> Path:
        return (
            processed_hier_split_dir(
                self.data_root,
                k=self.k,
                eps=self.eps,
                min_samples=self.min_samples,
                n_virtual_nodes=self.n_virtual_nodes,
                n_virtual_edges=self.n_virtual_edges,
                max_dbscan_edges=self.max_dbscan_edges,
                num_particles=self.num_particles,
            )
            / f"{self.split}.pt"
        )

    def _ensure_open(self) -> None:
        if self._data is not None:
            return
        path = self._cache_path()
        log_info(f"[graphs] split={self.split} mode=hier-cache mmap {path}")
        self._data = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        n = int(self._data["labels"].size(0))
        n_use = int(getattr(self, "_length", n))
        log_info(f"[graphs] split={self.split} hier-cache mmap indexed {n_use}/{n} jets")

    def __len__(self) -> int:
        return int(self._length)

    def __getitem__(self, idx: int) -> int:
        # Indices are resolved in ``collate_samples`` (mmap-friendly).
        return int(idx)

    def collate_samples(self, logical_indices: list[int]) -> dict[str, torch.Tensor]:
        self._ensure_open()
        assert self._data is not None
        if self._indices is not None:
            jet_idx = [int(self._indices[i]) for i in logical_indices]
        else:
            jet_idx = [int(i) for i in logical_indices]
        ji = torch.as_tensor(jet_idx, dtype=torch.long)
        order = torch.argsort(ji)
        ji_sorted = ji[order]
        stacked = {key: self._data[key][ji_sorted] for key in self._TENSOR_KEYS}
        labels = self._data["labels"][ji_sorted]
        restore = torch.argsort(order)
        batch = {key: stacked[key][restore] for key in self._TENSOR_KEYS}
        batch["y"] = labels[restore]

        # Model node features: ParT-kin particles + zero-filled virtual slots.
        x_model = batch.pop("x_model")  # (B, N, 7)
        x_raw = batch.pop("x")  # (B, N+V, 4) — drop raw four-vectors from batch
        n_part = x_model.size(1)
        n_tot = x_raw.size(1)
        if n_tot < n_part:
            raise RuntimeError(
                f"hier cache inconsistency: N_tot={n_tot} < N_part={n_part}"
            )
        x = x_raw.new_zeros(x_raw.size(0), n_tot, x_model.size(-1))
        x[:, :n_part] = x_model
        batch["x"] = x
        batch["mask"] = batch["node_mask"]
        return batch
