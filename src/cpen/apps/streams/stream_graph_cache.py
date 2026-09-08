"""Stellar-stream PyG graphs → CPEN padded COO incidence batches.

Each ``stream_*.pt`` is one region graph with semisupervised masks:

- ``y``: full target-stream membership (1 = member of this stream)
- ``train_mask`` (on disk): obvious stream ∪ obvious background (supervised pool)
- ``val_mask`` / ``test_mask`` (on disk): held-out discovery targets (typically all-positive)
- ``edge_y``: both endpoints are targets (recomputed after undirected dedup)

By default, ``pyg_to_incidence_payload`` **re-partitions obvious stream stars**
(``train_mask ∧ y=1``) into train/val/test with roughly equal counts, using a
deterministic per-graph RNG seed. Obvious background stays in train; former
hard val/test members become unlabeled (discovery pool via ``~train_mask``).

Message passing uses the **full** edge set. Node/edge supervision uses
train masks only; val/test masks are evaluation sets.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from cpen.graphs.graphs import scale_features_l2_sqrt_dim
from cpen.graphs.undirected import canonical_undirected_edges


DEFAULT_STREAM_DATA_DIR = Path(
    "/projects/BHANIN/adri_gage_gnn_streams/gnn_streams/mocks/"
    "streams_per_region_iter3_pm_stepping2_k4_edges_pmcoh/"
    "pyg_data_pmcoh_dir0p97_dlog0p25"
)

N_FEATURES = 7
N_EDGE_FEATURES = 4
OUT_DIM = 2
FEATURE_NAMES = (
    "phi",
    "lam",
    "pm_phi",
    "pm_lam",
    "parallax_gaia",
    "psf_mag_aper_8_g_corrected_des",
    "psf_mag_aper_8_r_corrected_des",
)
EDGE_ATTR_NAMES = ("round", "knn_dist", "dir_cos", "dlog_speed")


def resolve_stream_graph_path(
    data_root: str | Path,
    *,
    stream_name: str,
) -> Path:
    """Map ``MOCK_1863`` / ``stream_MOCK_1863`` → ``stream_MOCK_1863.pt``."""
    root = Path(data_root)
    name = stream_name.strip()
    if name.endswith(".pt"):
        name = name[: -len(".pt")]
    if not name.startswith("stream_"):
        name = f"stream_{name}"
    path = root / f"{name}.pt"
    if path.is_file():
        return path
    raise FileNotFoundError(
        f"Missing stellar-stream graph {path}. "
        f"Expected files like {root / 'stream_MOCK_1863.pt'}."
    )


def list_mock_stream_names(data_root: str | Path) -> list[str]:
    """Return sorted MOCK_* stems present under ``data_root``."""
    root = Path(data_root)
    names = sorted(
        p.stem[len("stream_") :]
        for p in root.glob("stream_MOCK_*.pt")
        if p.is_file()
    )
    if not names:
        raise FileNotFoundError(f"No stream_MOCK_*.pt graphs under {root}")
    return names


def list_real_stream_names(data_root: str | Path) -> list[str]:
    """Return sorted non-MOCK ``stream_*.pt`` stems under ``data_root``."""
    root = Path(data_root)
    names = sorted(
        p.stem[len("stream_") :]
        for p in root.glob("stream_*.pt")
        if p.is_file() and not p.name.startswith("stream_MOCK_")
    )
    if not names:
        raise FileNotFoundError(f"No non-MOCK stream_*.pt graphs under {root}")
    return names


def list_all_stream_names(data_root: str | Path) -> list[str]:
    """Return sorted MOCK + real stream stems under ``data_root``."""
    return sorted(set(list_mock_stream_names(data_root)) | set(list_real_stream_names(data_root)))


def resolve_stream_names(
    data_root: str | Path,
    stream_name: str | Sequence[str] | None,
    *,
    include_real: bool = False,
) -> list[str]:
    """
    Resolve which graphs to load.

    ``None`` / ``\"all\"`` / ``\"mock\"`` → every ``stream_MOCK_*.pt``
    (plus real streams when ``include_real`` is True).

    ``\"real\"`` → every non-MOCK ``stream_*.pt``.
    ``\"mock+real\"`` / ``\"both\"`` → MOCK and real graphs.
    A single name or comma-separated list selects a subset (``include_real``
    is ignored for explicit lists).
    """
    _ALL_TOKENS = {"", "all", "mock", "mocks"}
    _REAL_TOKENS = {"real", "reals"}
    _BOTH_TOKENS = {"mock+real", "real+mock", "both", "all+real", "all_streams"}

    if stream_name is None:
        text = "all"
    elif isinstance(stream_name, str):
        text = stream_name.strip()
    else:
        return [str(s).strip() for s in stream_name if str(s).strip()]

    key = text.lower()
    if key in _ALL_TOKENS:
        if include_real:
            return list_all_stream_names(data_root)
        return list_mock_stream_names(data_root)
    if key in _REAL_TOKENS:
        return list_real_stream_names(data_root)
    if key in _BOTH_TOKENS:
        return list_all_stream_names(data_root)
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return parts


def load_stream_pyg(path: str | Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _stable_mask_seed(base_seed: int, stream_key: str) -> int:
    """Deterministic 31-bit seed from ``base_seed`` and a stream id string."""
    digest = hashlib.sha256(f"{int(base_seed)}::{stream_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


def repartition_obvious_stream_masks(
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int,
    stream_key: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    """
    Split on-disk **obvious stream** stars evenly across train/val/test.

    - Obvious stream: ``train_mask ∧ (y=1)`` (file convention).
    - Obvious background: ``train_mask ∧ (y=0)`` → stays in train.
    - Former hard val/test members (typically non-obvious ``y=1``) → unlabeled
      (not in any of the returned masks) so discovery AUROC on ``~train_mask``
      still sees them.

    The shuffle is fixed by ``(seed, stream_key)`` for the whole run.
    """
    train_mask = train_mask.to(torch.bool).reshape(-1)
    val_mask = val_mask.to(torch.bool).reshape(-1)
    test_mask = test_mask.to(torch.bool).reshape(-1)
    y = y.to(torch.long).reshape(-1)
    if train_mask.shape != y.shape:
        raise ValueError("train_mask and y must have the same shape")

    obvious_stream = train_mask & (y == 1)
    obvious_bg = train_mask & (y == 0)
    hard_holdout = (val_mask | test_mask) & (y == 1) & ~obvious_stream

    idx = torch.nonzero(obvious_stream, as_tuple=False).flatten()
    n = int(idx.numel())
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_stable_mask_seed(seed, stream_key))
    if n > 0:
        perm = idx[torch.randperm(n, generator=gen)]
    else:
        perm = idx

    n_train = n // 3
    n_val = n // 3
    # Remainder (including n % 3) goes to test so counts stay as equal as possible.
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]

    new_train = torch.zeros_like(train_mask)
    new_val = torch.zeros_like(val_mask)
    new_test = torch.zeros_like(test_mask)
    if train_idx.numel():
        new_train[train_idx] = True
    if val_idx.numel():
        new_val[val_idx] = True
    if test_idx.numel():
        new_test[test_idx] = True
    new_train = new_train | obvious_bg

    stats = {
        "obvious_stream": n,
        "obvious_stream_train": int(train_idx.numel()),
        "obvious_stream_val": int(val_idx.numel()),
        "obvious_stream_test": int(test_idx.numel()),
        "obvious_bg_train": int(obvious_bg.sum()),
        "hard_unlabeled": int(hard_holdout.sum()),
    }
    return new_train, new_val, new_test, stats


def pyg_to_incidence_payload(
    data: Any,
    *,
    repartition_obvious: bool = True,
    mask_seed: int = 0,
    stream_key: str | None = None,
) -> dict[str, torch.Tensor]:
    """Convert one stream PyG graph to a CPEN COO incidence batch payload (unbatched)."""
    x = data.x.to(torch.float32)
    if x.size(-1) != N_FEATURES:
        raise ValueError(f"expected {N_FEATURES} node features; got {x.size(-1)}")
    x = scale_features_l2_sqrt_dim(x)

    edge_attr = data.edge_attr.to(torch.float32)
    if edge_attr.size(-1) != N_EDGE_FEATURES:
        raise ValueError(f"expected {N_EDGE_FEATURES} edge features; got {edge_attr.size(-1)}")

    pairs, edge_x = canonical_undirected_edges(
        data.edge_index,
        edge_attr,
        num_nodes=x.size(0),
    )
    edge_x = scale_features_l2_sqrt_dim(edge_x)
    n_nodes = x.size(0)
    n_edges = pairs.size(1)

    # int64: MOCK graphs can exceed int16 node indexing used by Pascal caches.
    incidence_node = pairs.t().reshape(-1).to(torch.int64)
    incidence_edge = (
        torch.arange(n_edges, dtype=torch.int64).unsqueeze(1).expand(n_edges, 2).reshape(-1)
    )
    node_degree = torch.bincount(incidence_node, minlength=n_nodes).to(torch.float32)

    y = data.y.reshape(-1).to(torch.long)
    if y.numel() != n_nodes:
        raise ValueError(f"y length {y.numel()} != n_nodes {n_nodes}")
    train_mask = data.train_mask.reshape(-1).to(torch.bool)
    val_mask = data.val_mask.reshape(-1).to(torch.bool)
    test_mask = data.test_mask.reshape(-1).to(torch.bool)
    if train_mask.numel() != n_nodes or val_mask.numel() != n_nodes or test_mask.numel() != n_nodes:
        raise ValueError("train/val/test masks must match num_nodes")

    key = stream_key or str(getattr(data, "safe_name", getattr(data, "stream_label", "stream")))
    repartition_stats: dict[str, int] | None = None
    if repartition_obvious:
        train_mask, val_mask, test_mask, repartition_stats = repartition_obvious_stream_masks(
            train_mask,
            val_mask,
            test_mask,
            y,
            seed=int(mask_seed),
            stream_key=key,
        )

    # Undirected edge label: both endpoints are target members.
    edge_y = ((y[pairs[0]] == 1) & (y[pairs[1]] == 1)).to(torch.long)
    edge_train_mask = train_mask[pairs[0]] & train_mask[pairs[1]]
    edge_val_mask = val_mask[pairs[0]] | val_mask[pairs[1]]
    edge_test_mask = test_mask[pairs[0]] | test_mask[pairs[1]]

    payload: dict[str, Any] = {
        "x": x,
        "edge_x": edge_x,
        "y": y,
        "edge_y": edge_y,
        "mask": torch.ones(n_nodes, dtype=torch.bool),  # alive for operators
        "train_mask": train_mask,
        "val_mask": val_mask,
        "test_mask": test_mask,
        "edge_train_mask": edge_train_mask,
        "edge_val_mask": edge_val_mask,
        "edge_test_mask": edge_test_mask,
        "incidence_node": incidence_node,
        "incidence_edge": incidence_edge,
        "incidence_nnz": torch.tensor(2 * n_edges, dtype=torch.int64),
        "node_degree_inv": node_degree.clamp_min(1).reciprocal(),
        "edge_degree_inv": torch.full((n_edges,), 0.5, dtype=torch.float32),
        "n_nodes": torch.tensor(n_nodes, dtype=torch.int64),
        "n_edges": torch.tensor(n_edges, dtype=torch.int64),
    }
    if repartition_stats is not None:
        payload["repartition_stats"] = repartition_stats
    return payload


def train_class_weights(y: torch.Tensor, train_mask: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency class weights on the supervised train pool."""
    labels = y[train_mask.to(torch.bool)]
    counts = torch.bincount(labels, minlength=OUT_DIM).to(torch.float32).clamp_min(1.0)
    weights = counts.sum() / (OUT_DIM * counts)
    return weights


def edge_train_class_weights(
    edge_y: torch.Tensor, edge_train_mask: torch.Tensor
) -> torch.Tensor:
    """Inverse-frequency weights on supervised edges (both endpoints in train_mask)."""
    labels = edge_y[edge_train_mask.to(torch.bool)]
    if labels.numel() == 0:
        return torch.ones(OUT_DIM, dtype=torch.float32)
    counts = torch.bincount(labels, minlength=OUT_DIM).to(torch.float32).clamp_min(1.0)
    return counts.sum() / (OUT_DIM * counts)


def aggregate_class_weights(payloads: Sequence[dict[str, torch.Tensor]]) -> torch.Tensor:
    counts = torch.zeros(OUT_DIM, dtype=torch.float32)
    for p in payloads:
        labels = p["y"][p["train_mask"].to(torch.bool)]
        counts += torch.bincount(labels, minlength=OUT_DIM).to(torch.float32)
    counts = counts.clamp_min(1.0)
    return counts.sum() / (OUT_DIM * counts)


def aggregate_edge_class_weights(payloads: Sequence[dict[str, torch.Tensor]]) -> torch.Tensor:
    counts = torch.zeros(OUT_DIM, dtype=torch.float32)
    for p in payloads:
        labels = p["edge_y"][p["edge_train_mask"].to(torch.bool)]
        if labels.numel() == 0:
            continue
        counts += torch.bincount(labels, minlength=OUT_DIM).to(torch.float32)
    counts = counts.clamp_min(1.0)
    return counts.sum() / (OUT_DIM * counts)


class MultiStreamGraphDataset(Dataset):
    """
    One sample per MOCK graph. Split selects which node/edge masks feed the loss.

    Train: CE on ``train_mask`` / ``edge_train_mask``.
    Val/test: discovery evaluation on ``val_mask`` / ``test_mask`` (and edge
    counterparts); AUROC uses the unlabeled pool ``~train_mask``.
    """

    def __init__(
        self,
        payloads: Sequence[dict[str, torch.Tensor]],
        stream_names: Sequence[str],
        *,
        split: str,
        include_incidence: bool = True,
        include_edge_features: bool = True,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train|val|test; got {split!r}")
        if len(payloads) != len(stream_names):
            raise ValueError("payloads and stream_names must have the same length")
        self.payloads = list(payloads)
        self.stream_names = list(stream_names)
        self.split = split
        self.include_incidence = include_incidence
        self.include_edge_features = include_edge_features
        self._node_loss_key = f"{split}_mask"
        self._edge_loss_key = f"edge_{split}_mask"

    def __len__(self) -> int:
        return len(self.payloads)

    def __getitem__(self, index: int) -> int:
        if index < 0 or index >= len(self.payloads):
            raise IndexError(index)
        return index

    def collate_samples(self, indices: list[int]) -> dict[str, torch.Tensor]:
        if len(indices) != 1:
            raise ValueError(
                f"stream graphs have different sizes; batch_size must be 1, got {len(indices)}"
            )
        p = self.payloads[indices[0]]
        batch: dict[str, torch.Tensor] = {
            "x": p["x"].unsqueeze(0),
            "y": p["y"].unsqueeze(0),
            "edge_y": p["edge_y"].unsqueeze(0),
            "mask": p["mask"].unsqueeze(0),
            "loss_mask": p[self._node_loss_key].unsqueeze(0),
            "edge_loss_mask": p[self._edge_loss_key].unsqueeze(0),
            "train_mask": p["train_mask"].unsqueeze(0),
            "val_mask": p["val_mask"].unsqueeze(0),
            "test_mask": p["test_mask"].unsqueeze(0),
            "edge_train_mask": p["edge_train_mask"].unsqueeze(0),
            "edge_val_mask": p["edge_val_mask"].unsqueeze(0),
            "edge_test_mask": p["edge_test_mask"].unsqueeze(0),
        }
        if self.include_edge_features:
            batch["edge_x"] = p["edge_x"].unsqueeze(0)
        if self.include_incidence:
            batch["incidence_node"] = p["incidence_node"].unsqueeze(0)
            batch["incidence_edge"] = p["incidence_edge"].unsqueeze(0)
            batch["incidence_nnz"] = p["incidence_nnz"].unsqueeze(0)
            batch["node_degree_inv"] = p["node_degree_inv"].unsqueeze(0)
            batch["edge_degree_inv"] = p["edge_degree_inv"].unsqueeze(0)
        return batch


# Backward-compatible alias used by older single-graph call sites.
class SingleStreamGraphDataset(MultiStreamGraphDataset):
    def __init__(
        self,
        payload: dict[str, torch.Tensor],
        *,
        split: str,
        include_incidence: bool = True,
        include_edge_features: bool = True,
        stream_name: str = "stream",
    ) -> None:
        super().__init__(
            [payload],
            [stream_name],
            split=split,
            include_incidence=include_incidence,
            include_edge_features=include_edge_features,
        )
