"""Load sibling ``attention_temperature.pyc`` bytecode as this module.

Adds hierarchical-cache γ estimation and **train-batch** γ estimation (live
star/kNN or cached incidence) on top of the star/kNN bytecode helpers.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PYC = Path(__file__).resolve().with_name("attention_temperature.pyc")
if not _PYC.is_file():
    raise ImportError(f"Missing bytecode backend {_PYC}")

_spec = importlib.util.spec_from_file_location("_cpen_attn_temp_bc", _PYC)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load {_PYC}")
_bc = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_cpen_attn_temp_bc", _bc)
_spec.loader.exec_module(_bc)

_PUBLIC = __name__
for _name, _value in vars(_bc).items():
    if _name.startswith("__"):
        continue
    if isinstance(_value, type) or callable(_value):
        try:
            _value.__module__ = _PUBLIC
        except (AttributeError, TypeError):
            pass
    globals()[_name] = _value

_orig_estimate_attention_gammas_from_cache = _bc.estimate_attention_gammas_from_cache


def materialize_batch_incidence(
    batch: dict[str, torch.Tensor],
    *,
    live_star_radius: float | None = None,
    live_graph_k: int | None = None,
    live_edge_features: str = "logdot-dp",
    live_centroid_weight: str | None = None,
    live_dense_pairs: bool = False,
) -> dict[str, torch.Tensor]:
    """Ensure ``incidence`` + node mask exist (live star/kNN or dense-from-COO)."""
    out = dict(batch)
    if "x_raw" in out:
        x_raw = out.pop("x_raw")
        mask = out.get("mask")
        if mask is None:
            mask = out.get("node_mask")
        if mask is None:
            raise ValueError("live graph batch requires mask or node_mask with x_raw")
        if live_star_radius is not None:
            from cpen.graphs.graph_star import build_star_radius_graph

            edge_x, incidence, _ = build_star_radius_graph(
                x_raw,
                radius=float(live_star_radius),
                mask=mask,
                edge_features=live_edge_features,
                centroid_weight=live_centroid_weight,
            )
            out["edge_x"] = edge_x
            out["incidence"] = incidence
            out["mask"] = mask.bool()
        elif live_graph_k is not None:
            from cpen.utils.graphs import build_knn_graph

            _, edge_x, incidence = build_knn_graph(
                x_raw, k=int(live_graph_k), mask=mask
            )
            out["edge_x"] = edge_x
            out["incidence"] = incidence
            out["mask"] = mask.bool()
        elif live_dense_pairs:
            raise ValueError(
                "dense-pair batches have no incidence; cannot estimate attention γ"
            )
        else:
            raise ValueError(
                "batch has x_raw but no live_star_radius / live_graph_k for γ estimate"
            )

    if "incidence" not in out and "incidence_node" in out:
        from cpen.graphs.sparse_incidence import (
            coo_from_batch_tensors,
            dense_incidence_from_coo,
        )

        node_mask = out.get("node_mask", out.get("mask"))
        if node_mask is None:
            raise ValueError("COO batch requires node_mask or mask to infer N")
        edge_x = out.get("edge_x")
        num_nodes = int(node_mask.size(-1))
        if edge_x is not None:
            num_edges = int(edge_x.size(1))
        else:
            num_edges = int(out["incidence_edge"].max().item()) + 1
        coo = coo_from_batch_tensors(
            incidence_node=out["incidence_node"],
            incidence_edge=out["incidence_edge"],
            incidence_nnz=out["incidence_nnz"],
            num_edges=num_edges,
            num_nodes=num_nodes,
        )
        out["incidence"] = dense_incidence_from_coo(coo, dtype=torch.bool)
    return out


def estimate_attention_gammas_from_batches(
    batches,
    *,
    max_jets: int = 512,
    all_to_all_particle_attention: bool = False,
    hyperedge_m22_only: bool = False,
    drop_pairwise_edges: bool = False,
    live_star_radius: float | None = None,
    live_graph_k: int | None = None,
    live_edge_features: str = "logdot-dp",
    live_centroid_weight: str | None = None,
) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Mean nonempty support row degrees from the first train batches."""
    degree_lists: dict[str, list[np.ndarray]] = {k: [] for k in ("11", "12", "21", "22")}
    n_seen = 0
    for raw in batches:
        if n_seen >= int(max_jets):
            break
        if not isinstance(raw, dict):
            raise TypeError(f"expected dict batch, got {type(raw)!r}")
        batch = materialize_batch_incidence(
            raw,
            live_star_radius=live_star_radius,
            live_graph_k=live_graph_k,
            live_edge_features=live_edge_features,
            live_centroid_weight=live_centroid_weight,
        )
        if "incidence" not in batch:
            raise ValueError("batch is missing incidence after materialize")
        node_mask = batch.get("node_mask", batch.get("mask"))
        if node_mask is None:
            raise ValueError("batch requires node_mask or mask")
        node_mask = node_mask.bool()
        edge_mask = batch["edge_mask"].bool() if "edge_mask" in batch else None
        s_bool = batch["incidence"].bool()

        take = min(int(node_mask.size(0)), int(max_jets) - n_seen)
        if take <= 0:
            break
        if take < int(node_mask.size(0)):
            node_mask = node_mask[:take]
            s_bool = s_bool[:take]
            if edge_mask is not None:
                edge_mask = edge_mask[:take]
            for key in ("edge_x", "edge_type", "edge_mask", "x"):
                if key in batch and torch.is_tensor(batch[key]):
                    batch[key] = batch[key][:take]

        if drop_pairwise_edges:
            from cpen.utils.graph_hierarchical import drop_pairwise_edges_from_batch

            packed = {
                "x": batch.get(
                    "x", torch.zeros(s_bool.size(0), s_bool.size(-1), 1)
                ),
                "edge_x": batch["edge_x"],
                "edge_type": batch.get(
                    "edge_type",
                    torch.zeros(s_bool.shape[:2], dtype=torch.long),
                ),
                "edge_mask": edge_mask
                if edge_mask is not None
                else torch.ones(s_bool.shape[:2], dtype=torch.bool),
                "incidence": s_bool,
                "node_mask": node_mask,
            }
            packed = drop_pairwise_edges_from_batch(packed)
            s_bool = packed["incidence"].bool()
            edge_mask = packed["edge_mask"].bool()
            node_mask = packed["node_mask"].bool()

        m_11, m_21, m_12, m_22_full = _build_attention_supports(  # noqa: F821
            s_bool,
            node_mask,
            all_to_all_particle_attention=all_to_all_particle_attention,
        )
        m_22 = (
            hyperedge_m22_support(s_bool, edge_mask) if hyperedge_m22_only else m_22_full
        )
        for key, support in (("11", m_11), ("12", m_12), ("21", m_21), ("22", m_22)):
            deg = support_row_degrees(support).reshape(-1).detach().cpu().numpy()  # noqa: F821
            degree_lists[key].append(deg[deg >= 1].astype(np.float64))
        n_seen += int(node_mask.size(0))

    if n_seen == 0:
        raise ValueError("no train batches available for attention-γ estimation")
    return _summarize_degree_lists(degree_lists)


def estimate_attention_gammas_from_datamodule(
    datamodule,
    *,
    max_jets: int = 512,
    batch_size: int | None = None,
    all_to_all_particle_attention: bool = False,
    hyperedge_m22_only: bool = False,
    drop_pairwise_edges: bool = False,
    live_star_radius: float | None = None,
    live_graph_k: int | None = None,
    live_edge_features: str = "logdot-dp",
    live_centroid_weight: str | None = None,
) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Pull the first train batches from ``datamodule`` and estimate γ."""
    from torch.utils.data import DataLoader

    datamodule.setup("fit")
    train = getattr(datamodule, "_train", None)
    if train is None:
        raise RuntimeError("datamodule.setup('fit') did not populate the train split")

    bs = int(batch_size or getattr(datamodule, "batch_size", 32) or 32)
    bs = max(1, min(bs, int(max_jets)))
    # Always single-process: γ only needs a few batches, and streaming Iterable
    # datasets should not spin ROOT worker pools just for temperature estimate.
    loader_kwargs: dict[str, Any] = {
        "batch_size": bs,
        "num_workers": 0,
        "shuffle": False,
        "pin_memory": False,
    }
    collate_fn = getattr(train, "collate_samples", None)
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    loader = DataLoader(train, **loader_kwargs)
    return estimate_attention_gammas_from_batches(
        loader,
        max_jets=int(max_jets),
        all_to_all_particle_attention=all_to_all_particle_attention,
        hyperedge_m22_only=hyperedge_m22_only,
        drop_pairwise_edges=drop_pairwise_edges,
        live_star_radius=live_star_radius,
        live_graph_k=live_graph_k,
        live_edge_features=live_edge_features,
        live_centroid_weight=live_centroid_weight,
    )


def hyperedge_m22_support(
    s_bool: torch.Tensor,
    edge_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """All-to-all among incidence-degree > 2 (CAPEN-Llama-att ``M_{22}``)."""
    is_hyper = s_bool.to(dtype=torch.float32).sum(dim=-1) > 2
    if edge_mask is not None:
        is_hyper = is_hyper & edge_mask.to(device=is_hyper.device, dtype=torch.bool)
    return is_hyper.unsqueeze(-1) & is_hyper.unsqueeze(-2)


def _summarize_degree_lists(
    degree_lists: dict[str, list[np.ndarray]],
) -> tuple[Any, dict[str, dict[str, Any]]]:
    summaries: dict[str, dict[str, Any]] = {}
    means: dict[str, float] = {}
    for key, chunks in degree_lists.items():
        vals = np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
        if vals.size == 0:
            summaries[key] = {
                "n_rows": 0,
                "mean": 1.0,
                "std": 0.0,
                "min": 1.0,
                "max": 1.0,
                "median": 1.0,
                "degrees": vals,
            }
            means[key] = 1.0
        else:
            summaries[key] = {
                "n_rows": int(vals.size),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "median": float(np.median(vals)),
                "degrees": vals,
            }
            means[key] = float(vals.mean())
    gammas = AttentionGammas(  # noqa: F821 — loaded from bytecode
        gamma_11=means["11"],
        gamma_12=means["12"],
        gamma_21=means["21"],
        gamma_22=means["22"],
    )
    return gammas, summaries


def estimate_attention_gammas_from_hier_payload(
    payload: dict[str, torch.Tensor],
    indices: np.ndarray | torch.Tensor,
    *,
    batch_size: int = 16,
    all_to_all_particle_attention: bool = True,
    hyperedge_m22_only: bool = True,
    drop_pairwise_edges: bool = False,
    drop_knn_edges: bool | None = None,
) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Mean nonempty row degrees of the supports CAPEN-Llama-att actually uses."""
    if drop_knn_edges is not None:
        drop_pairwise_edges = bool(drop_pairwise_edges or drop_knn_edges)
    pick = np.asarray(indices)
    if pick.size == 0:
        raise ValueError("indices must be non-empty")
    degree_lists: dict[str, list[np.ndarray]] = {k: [] for k in ("11", "12", "21", "22")}
    for start in range(0, len(pick), int(batch_size)):
        idx = pick[start : start + int(batch_size)]
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        if "node_mask" in payload:
            node_mask = payload["node_mask"][idx_t].bool()
        else:
            node_mask = payload["mask"][idx_t].bool()
        edge_mask = payload["edge_mask"][idx_t].bool() if "edge_mask" in payload else None
        if "incidence" in payload:
            s_bool = payload["incidence"][idx_t].bool()
        else:
            s_bool = _dense_incidence_from_coo_payload(payload, idx_t)  # noqa: F821
        batch: dict[str, torch.Tensor] | None = None
        if drop_pairwise_edges:
            from cpen.utils.graph_hierarchical import drop_pairwise_edges_from_batch

            batch = {
                "x": payload["x"][idx_t],
                "edge_x": payload["edge_x"][idx_t],
                "edge_type": payload["edge_type"][idx_t]
                if "edge_type" in payload
                else torch.zeros(s_bool.shape[:2], dtype=torch.long),
                "edge_mask": edge_mask
                if edge_mask is not None
                else torch.ones(s_bool.shape[:2], dtype=torch.bool),
                "incidence": s_bool,
                "node_mask": node_mask,
            }
            batch = drop_pairwise_edges_from_batch(batch)
            s_bool = batch["incidence"].bool()
            edge_mask = batch["edge_mask"].bool()
            node_mask = batch["node_mask"].bool()

        m_11, m_21, m_12, m_22_full = _build_attention_supports(  # noqa: F821
            s_bool,
            node_mask,
            all_to_all_particle_attention=all_to_all_particle_attention,
        )
        m_22 = (
            hyperedge_m22_support(s_bool, edge_mask) if hyperedge_m22_only else m_22_full
        )
        for key, support in (("11", m_11), ("12", m_12), ("21", m_21), ("22", m_22)):
            deg = support_row_degrees(support).reshape(-1).detach().cpu().numpy()  # noqa: F821
            degree_lists[key].append(deg[deg >= 1].astype(np.float64))
        del node_mask, edge_mask, s_bool, m_11, m_21, m_12, m_22, m_22_full, batch
    return _summarize_degree_lists(degree_lists)


def estimate_attention_gammas_from_hier_cache(
    *,
    data_root,
    split: str = "train",
    k: int,
    eps: float = 0.08,
    min_samples: int = 2,
    n_virtual_nodes: int = 1,
    n_virtual_edges: int = 1,
    max_dbscan_edges: int = 32,
    num_particles: int,
    n_jets: int = 512,
    batch_size: int = 16,
    seed: int = 42,
    all_to_all_particle_attention: bool = True,
    hyperedge_m22_only: bool = True,
    drop_pairwise_edges: bool = False,
    drop_knn_edges: bool | None = None,
) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Estimate attention γ from a hierarchical TopTagging cache (mmap)."""
    from cpen.utils.hier_graph_cache import (
        hier_graph_cache_available,
        missing_hier_graph_cache_message,
        processed_hier_split_dir,
    )

    kw = dict(
        k=int(k),
        eps=float(eps),
        min_samples=int(min_samples),
        n_virtual_nodes=int(n_virtual_nodes),
        n_virtual_edges=int(n_virtual_edges),
        max_dbscan_edges=int(max_dbscan_edges),
        num_particles=int(num_particles),
    )
    if not hier_graph_cache_available(data_root=data_root, split=split, **kw):
        raise FileNotFoundError(
            missing_hier_graph_cache_message(data_root=data_root, split=split, **kw)
        )
    cache_dir = processed_hier_split_dir(data_root, **kw)
    payload = torch.load(
        cache_dir / f"{split}.pt",
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    n_total = int(payload["labels"].size(0)) if "labels" in payload else int(
        payload["node_mask"].size(0)
    )
    rng = np.random.default_rng(int(seed))
    pick = rng.choice(n_total, size=min(int(n_jets), n_total), replace=False)
    return estimate_attention_gammas_from_hier_payload(
        payload,
        pick,
        batch_size=int(batch_size),
        all_to_all_particle_attention=all_to_all_particle_attention,
        hyperedge_m22_only=hyperedge_m22_only,
        drop_pairwise_edges=drop_pairwise_edges,
        drop_knn_edges=drop_knn_edges,
    )


def estimate_attention_gammas_from_cache(  # type: ignore[no-untyped-def]
    *,
    data_root,
    split="train",
    num_particles,
    n_jets=512,
    batch_size=16,
    seed=42,
    radius=None,
    graph_construction=None,
    all_to_all_particle_attention=False,
    hyperedge_m22_only=False,
    drop_pairwise_edges=False,
    drop_knn_edges=None,
    k=None,
    eps=0.08,
    min_samples=2,
    n_virtual_nodes=1,
    n_virtual_edges=1,
    max_dbscan_edges=32,
):
    """Star / kNN bytecode estimator, plus hierarchical caches."""
    construction = str(graph_construction or "")
    hier = construction.startswith("hier_") or k is not None
    if hier and radius is None:
        if k is None:
            from cpen.utils.graph_hierarchical import parse_hierarchical_construction_tag

            parsed = parse_hierarchical_construction_tag(construction)
            k = parsed["k"]
            eps = parsed["eps"]
            min_samples = parsed["min_samples"]
            n_virtual_nodes = parsed["n_virtual_nodes"]
            n_virtual_edges = parsed["n_virtual_edges"]
            max_dbscan_edges = parsed["max_dbscan_edges"]
        return estimate_attention_gammas_from_hier_cache(
            data_root=data_root,
            split=split,
            k=int(k),
            eps=float(eps),
            min_samples=int(min_samples),
            n_virtual_nodes=int(n_virtual_nodes),
            n_virtual_edges=int(n_virtual_edges),
            max_dbscan_edges=int(max_dbscan_edges),
            num_particles=int(num_particles),
            n_jets=int(n_jets),
            batch_size=int(batch_size),
            seed=int(seed),
            all_to_all_particle_attention=all_to_all_particle_attention,
            hyperedge_m22_only=hyperedge_m22_only,
            drop_pairwise_edges=drop_pairwise_edges,
            drop_knn_edges=drop_knn_edges,
        )
    return _orig_estimate_attention_gammas_from_cache(
        data_root=data_root,
        split=split,
        num_particles=num_particles,
        n_jets=n_jets,
        batch_size=batch_size,
        seed=seed,
        radius=radius,
        graph_construction=graph_construction,
        all_to_all_particle_attention=all_to_all_particle_attention,
    )
