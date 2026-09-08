"""Load sibling ``attention_temperature.pyc`` bytecode as this module.

Adds hierarchical-cache γ estimation (CAPEN-Llama-att supports, optional
hyperedge-only ``M_{22}``, optional drop-pairwise) on top of the star/kNN bytecode.
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
