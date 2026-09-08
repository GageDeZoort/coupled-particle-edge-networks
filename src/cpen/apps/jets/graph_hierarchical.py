"""Hierarchical jet graph construction for CAPEN / class-token readout.

Combines three layers of structure plus optional virtual nodes/edges:

1. **kNN 2-edges** (local) in :math:`(\\eta,\\phi)` with distance :math:`\\Delta R`.
2. **DBSCAN hyperedges** (structural patches) in the same :math:`\\Delta R` metric.
3. **Virtual nodes** — extra *nodes* (not edges) appended after particles; each is
   linked to every particle by a pairwise **vn_link** edge.
4. **Virtual hyperedges** — each covers every (active) particle.

Virtual nodes carry ``node_type`` only (never ``edge_type``). Edge type IDs
label relations among the incidence rows (kNN / DBSCAN / vn_link / virtual hyper).

Node layout (for downstream MHSA pooling over :math:`[X; E]`)::

    [ particles (N) | virtual nodes (n_virtual_nodes) ]

Edge layout (concatenated; ``edge_type`` discriminates relations)::

    [ kNN (N·k) | DBSCAN (max_dbscan) | virtual↔particle (n_vn·N) | virtual hypers (n_ve) ]

Edge *features* are ParT pairwise kinematics ``(ln Δ, ln k_T, ln z, ln m²)``
(:func:`cpen.apps.jets.part_kin.build_part_interaction_features`), not the
legacy Minkowski-dot + ``Δp`` vector, then L2-scaled so ``||e||^2 = 4``.
2-edges use the two endpoints; hyperedges and virtual links use the leading
1→2 split (hardest member vs remainder 4-sum), so the pre-L2 ``ln m²`` is the
invariant mass of that component.

Defaults match the DBSCAN study: ``eps=0.08``, ``min_samples=2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from sklearn.cluster import DBSCAN

from cpen.utils.graph_star import (
    delta_r_torch,
    eta_phi_from_four_vectors_torch,
)
from cpen.utils.graphs import particle_mask_from_four_vectors
from cpen.utils.part_kin import (
    N_PART_INT_FEATURES,
    build_part_interaction_features,
    leading_split_four_vectors,
)
from cpen.utils.sparse_incidence import finalize_incidence_storage

# ---------------------------------------------------------------------------
# Edge / node type IDs (stable; safe to embed)
# ---------------------------------------------------------------------------

EDGE_TYPE_KNN: Final[int] = 0
EDGE_TYPE_DBSCAN: Final[int] = 1
EDGE_TYPE_VN_LINK: Final[int] = 2  # pairwise edge: virtual *node* ↔ particle
EDGE_TYPE_VIRTUAL_HYPER: Final[int] = 3  # hyperedge over all particles

NODE_TYPE_PARTICLE: Final[int] = 0
NODE_TYPE_VIRTUAL: Final[int] = 1

EDGE_TYPE_NAMES: Final[dict[int, str]] = {
    EDGE_TYPE_KNN: "knn",
    EDGE_TYPE_DBSCAN: "dbscan",
    EDGE_TYPE_VN_LINK: "vn_link",
    EDGE_TYPE_VIRTUAL_HYPER: "virtual_hyper",
}


@dataclass(frozen=True)
class HierarchicalGraph:
    """Batched hierarchical jet graph ready for CAPEN + typed readout.

    Shapes use ``B`` batch, ``N`` particle slots, ``V`` virtual nodes,
    ``N_tot = N + V``, ``M`` total hyperedge/edge slots.
    """

    # Node channels
    x: torch.Tensor  # (B, N_tot, 4) raw four-vectors (virtuals: jet aggregate)
    node_type: torch.Tensor  # (B, N_tot) long
    node_mask: torch.Tensor  # (B, N_tot) bool — valid nodes
    particle_mask: torch.Tensor  # (B, N) bool — active particles only
    z: torch.Tensor  # (B, N_tot) pooling prior (pT frac on particles; uniform on virtuals)

    # Edge / hyperedge channels
    edge_x: torch.Tensor  # (B, M, 4) ParT-int features, L2-scaled
    edge_type: torch.Tensor  # (B, M) long
    edge_mask: torch.Tensor  # (B, M) bool — nonempty incidence rows
    incidence: torch.Tensor  # (B, M, N_tot) float {0,1}

    # Slice sizes (fixed by hparams; useful for type embeddings / debugging)
    n_particles: int
    n_virtual_nodes: int
    n_knn_edges: int
    n_dbscan_edges: int
    n_virtual_node_edges: int
    n_virtual_hyperedges: int

    # Construction hparams
    k: int
    eps: float
    min_samples: int

    def num_nodes(self) -> int:
        return int(self.x.size(1))

    def num_edges(self) -> int:
        return int(self.edge_x.size(1))

    def to_cache_payload(self) -> dict[str, torch.Tensor]:
        """Tensors for a future ``processed/hier_…/n{N}/{split}.pt`` cache."""
        payload = finalize_incidence_storage(self.incidence)
        payload.update(
            {
                "x": self.x,
                "edge_x": self.edge_x,
                "edge_type": self.edge_type,
                "edge_mask": self.edge_mask,
                "node_type": self.node_type,
                "node_mask": self.node_mask,
                "particle_mask": self.particle_mask,
                "mask": self.particle_mask,  # CAPEN legacy particle mask
                "z": self.z,
            }
        )
        return payload


def hierarchical_construction_tag(
    *,
    k: int,
    eps: float = 0.08,
    min_samples: int = 2,
    n_virtual_nodes: int = 1,
    n_virtual_edges: int = 1,
    max_dbscan_edges: int = 32,
) -> str:
    """Directory-safe tag, e.g. ``hier_k8_eps0p08_ms2_vn1_ve1_mdb32_pint``."""
    eps_tok = f"{eps:g}".replace(".", "p")
    return (
        f"hier_k{int(k)}_eps{eps_tok}_ms{int(min_samples)}"
        f"_vn{int(n_virtual_nodes)}_ve{int(n_virtual_edges)}_mdb{int(max_dbscan_edges)}"
        f"_pint"
    )


def parse_hierarchical_construction_tag(tag: str) -> dict[str, int | float]:
    """Inverse of :func:`hierarchical_construction_tag` (suffix after ``mdb`` is ignored)."""
    import re

    match = re.match(
        r"^hier_k(\d+)_eps([0-9p]+)_ms(\d+)_vn(\d+)_ve(\d+)_mdb(\d+)(?:_.*)?$",
        str(tag).strip(),
    )
    if match is None:
        raise ValueError(f"not a hierarchical construction tag: {tag!r}")
    return {
        "k": int(match.group(1)),
        "eps": float(match.group(2).replace("p", ".")),
        "min_samples": int(match.group(3)),
        "n_virtual_nodes": int(match.group(4)),
        "n_virtual_edges": int(match.group(5)),
        "max_dbscan_edges": int(match.group(6)),
    }


def _pairwise_delta_r_np(eta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    n = len(eta)
    d = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        deta = eta[i] - eta
        dphi = np.abs(phi[i] - phi)
        dphi = np.minimum(dphi, 2.0 * np.pi - dphi)
        d[i] = np.hypot(deta, dphi)
    return d


def _dbscan_labels_delta_r(
    eta: np.ndarray,
    phi: np.ndarray,
    *,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    """DBSCAN labels on active particles; empty → all noise."""
    n = len(eta)
    if n < int(min_samples):
        return np.full(n, -1, dtype=np.int64)
    dist = _pairwise_delta_r_np(eta, phi)
    return DBSCAN(
        eps=float(eps),
        min_samples=int(min_samples),
        metric="precomputed",
    ).fit_predict(dist)


def build_hierarchical_graph(
    x_raw: torch.Tensor,
    *,
    k: int,
    eps: float = 0.08,
    min_samples: int = 2,
    n_virtual_nodes: int = 1,
    n_virtual_edges: int = 1,
    max_dbscan_edges: int = 32,
    mask: torch.Tensor | None = None,
) -> HierarchicalGraph:
    """
    Build a batched hierarchical jet graph.

    Parameters
    ----------
    x_raw:
        Particle four-vectors ``(B, N, 4)`` as ``[E, px, py, pz]``.
    k:
        Directed kNN degree in :math:`\\Delta R` (includes self as nearest).
    eps, min_samples:
        DBSCAN hyperparameters in :math:`\\Delta R` space.
    n_virtual_nodes:
        Count of virtual nodes appended after particles. Each gets a pairwise
        edge to every particle slot (inactive slots → empty edge).
    n_virtual_edges:
        Count of global hyperedges; each covers all *active* particles.
    max_dbscan_edges:
        Fixed pad for DBSCAN hyperedge slots (extras stay empty).
    mask:
        Optional ``(B, N)`` particle activity mask.
    """
    if x_raw.dim() != 3 or x_raw.size(-1) != 4:
        raise ValueError(f"x_raw must be (B, N, 4); got {tuple(x_raw.shape)}")
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if max_dbscan_edges < 0 or n_virtual_nodes < 0 or n_virtual_edges < 0:
        raise ValueError("edge/node counts must be non-negative")

    if mask is None:
        mask = particle_mask_from_four_vectors(x_raw)
    mask = mask.bool()

    batch, n_particles, _ = x_raw.shape
    device = x_raw.device
    dtype = x_raw.dtype
    n_virt = int(n_virtual_nodes)
    n_tot = n_particles + n_virt

    eta, phi = eta_phi_from_four_vectors_torch(x_raw)
    pt = torch.hypot(x_raw[..., 1], x_raw[..., 2])

    # ---- node features: particles + virtuals (jet 4-sum) ----
    jet_sum = (x_raw * mask.unsqueeze(-1).to(dtype)).sum(dim=1)  # (B, 4)
    if n_virt > 0:
        x_virt = jet_sum.unsqueeze(1).expand(batch, n_virt, 4).clone()
        x = torch.cat([x_raw, x_virt], dim=1)
    else:
        x = x_raw

    node_type = torch.zeros(batch, n_tot, dtype=torch.long, device=device)
    if n_virt > 0:
        node_type[:, n_particles:] = NODE_TYPE_VIRTUAL

    node_mask = torch.zeros(batch, n_tot, dtype=torch.bool, device=device)
    node_mask[:, :n_particles] = mask
    if n_virt > 0:
        # Virtual nodes are always present when requested.
        node_mask[:, n_particles:] = True

    # Pooling prior z: pT fractions on particles; equal share on virtuals.
    z = torch.zeros(batch, n_tot, dtype=dtype, device=device)
    pt_masked = pt * mask.to(dtype)
    pt_sum = pt_masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    z[:, :n_particles] = pt_masked / pt_sum
    if n_virt > 0:
        z[:, n_particles:] = 1.0 / float(n_virt)

    # ---- allocate edge buffers ----
    n_knn = n_particles * int(k)
    n_db = int(max_dbscan_edges)
    n_vn_edges = n_virt * n_particles
    n_ve = int(n_virtual_edges)
    m_total = n_knn + n_db + n_vn_edges + n_ve

    incidence = torch.zeros(batch, m_total, n_tot, dtype=dtype, device=device)
    edge_x = torch.zeros(batch, m_total, N_PART_INT_FEATURES, dtype=dtype, device=device)
    edge_type = torch.zeros(batch, m_total, dtype=torch.long, device=device)
    edge_mask = torch.zeros(batch, m_total, dtype=torch.bool, device=device)

    # =====================================================================
    # 1) Directed ΔR-kNN among particles  → slots [0, n_knn)
    # =====================================================================
    eta1, phi1 = eta.unsqueeze(-1), phi.unsqueeze(-1)
    eta2, phi2 = eta.unsqueeze(-2), phi.unsqueeze(-2)
    dr = delta_r_torch(eta1, phi1, eta2, phi2)
    pair_ok = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    dr = dr.masked_fill(~pair_ok, float("inf"))
    eye = torch.eye(n_particles, device=device, dtype=torch.bool).unsqueeze(0)
    dr = torch.where(eye & mask.unsqueeze(-1), torch.zeros_like(dr), dr)

    knn_k = min(int(k), n_particles)
    knn_dist, knn_idx = dr.topk(knn_k, dim=-1, largest=False)  # (B, N, knn_k)
    finite = torch.isfinite(knn_dist) & mask.unsqueeze(-1)

    src = (
        torch.arange(n_particles, device=device)
        .view(1, n_particles, 1)
        .expand(batch, -1, knn_k)
    )
    dst = knn_idx
    # Flat edge ids for the first knn_k neighbor slots of each particle.
    e_ids = (
        torch.arange(n_particles, device=device).view(1, n_particles, 1) * int(k)
        + torch.arange(knn_k, device=device).view(1, 1, knn_k)
    ).expand(batch, -1, -1)

    b_ids = (
        torch.arange(batch, device=device).view(batch, 1, 1).expand(batch, n_particles, knn_k)
    )
    flat_b = b_ids[finite]
    flat_e = e_ids[finite]
    flat_s = src[finite]
    flat_d = dst[finite]
    if flat_e.numel():
        incidence[flat_b, flat_e, flat_s] = 1.0
        incidence[flat_b, flat_e, flat_d] = 1.0
        edge_mask[flat_b, flat_e] = True
        edge_type[flat_b, flat_e] = EDGE_TYPE_KNN
        edge_x[flat_b, flat_e] = build_part_interaction_features(
            x_raw[flat_b, flat_s], x_raw[flat_b, flat_d]
        )

    # =====================================================================
    # 2) DBSCAN hyperedges  → slots [n_knn, n_knn + n_db)
    # =====================================================================
    db_base = n_knn
    for b in range(batch):
        active = mask[b].nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            continue
        eta_b = eta[b, active].detach().cpu().numpy()
        phi_b = phi[b, active].detach().cpu().numpy()
        labels = _dbscan_labels_delta_r(
            eta_b, phi_b, eps=eps, min_samples=min_samples
        )
        n_clusters = int(labels.max() + 1) if (labels >= 0).any() else 0
        n_write = min(n_clusters, n_db)
        for c in range(n_write):
            members_local = np.flatnonzero(labels == c)
            if members_local.size == 0:
                continue
            members = active[torch.as_tensor(members_local, device=device, dtype=torch.long)]
            e = db_base + c
            incidence[b, e, members] = 1.0
            edge_mask[b, e] = True
            edge_type[b, e] = EDGE_TYPE_DBSCAN
            hardest, remainder = leading_split_four_vectors(
                x_raw[b], members, pt=pt[b]
            )
            edge_x[b, e] = build_part_interaction_features(
                hardest.unsqueeze(0), remainder.unsqueeze(0)
            ).squeeze(0)

    # =====================================================================
    # 3) Pairwise links: each *virtual node* ↔ every particle
    #     (edges get EDGE_TYPE_VN_LINK; virtual nodes themselves stay nodes)
    #     slots [n_knn+n_db, n_knn+n_db+n_vn_edges)
    # =====================================================================
    vn_base = n_knn + n_db
    if n_virt > 0:
        # e(v,p) = vn_base + v*N + p
        v_ids = torch.arange(n_virt, device=device).view(n_virt, 1).expand(n_virt, n_particles)
        p_ids = torch.arange(n_particles, device=device).view(1, n_particles).expand(n_virt, n_particles)
        e_ids_vp = vn_base + v_ids * n_particles + p_ids  # (V, N)
        v_nodes = n_particles + v_ids  # (V, N)

        for b in range(batch):
            active_p = mask[b]  # (N,)
            if not active_p.any():
                continue
            keep = active_p.view(1, n_particles).expand(n_virt, n_particles)
            ee = e_ids_vp[keep]
            pp = p_ids[keep]
            vv = v_nodes[keep]
            incidence[b, ee, pp] = 1.0
            incidence[b, ee, vv] = 1.0
            edge_mask[b, ee] = True
            edge_type[b, ee] = EDGE_TYPE_VN_LINK
            # 1→2 of this particle vs the rest of the jet (same m² = jet mass).
            rest = jet_sum[b].unsqueeze(0) - x_raw[b, pp]
            edge_x[b, ee] = build_part_interaction_features(x_raw[b, pp], rest)

    # =====================================================================
    # 4) Virtual hyperedges over all active particles
    # =====================================================================
    ve_base = n_knn + n_db + n_vn_edges
    if n_ve > 0:
        has = mask.any(dim=-1)
        for v in range(n_ve):
            e = ve_base + v
            edge_type[:, e] = EDGE_TYPE_VIRTUAL_HYPER
            incidence[has, e, :n_particles] = mask[has].to(dtype)
            edge_mask[has, e] = True
        for b in range(batch):
            if not bool(has[b]):
                continue
            active = mask[b].nonzero(as_tuple=True)[0]
            hardest, remainder = leading_split_four_vectors(
                x_raw[b], active, pt=pt[b]
            )
            feat = build_part_interaction_features(
                hardest.unsqueeze(0), remainder.unsqueeze(0)
            ).squeeze(0)
            edge_x[b, ve_base : ve_base + n_ve] = feat

    return HierarchicalGraph(
        x=x,
        node_type=node_type,
        node_mask=node_mask,
        particle_mask=mask,
        z=z,
        edge_x=edge_x,
        edge_type=edge_type,
        edge_mask=edge_mask,
        incidence=incidence,
        n_particles=n_particles,
        n_virtual_nodes=n_virt,
        n_knn_edges=n_knn,
        n_dbscan_edges=n_db,
        n_virtual_node_edges=n_vn_edges,
        n_virtual_hyperedges=n_ve,
        k=int(k),
        eps=float(eps),
        min_samples=int(min_samples),
    )


def build_hierarchical_graph_single(
    x_raw: torch.Tensor,
    **kwargs,
) -> HierarchicalGraph:
    """Single-jet wrapper; ``x_raw`` may be ``(N, 4)`` or ``(1, N, 4)``."""
    if x_raw.dim() == 2:
        x_raw = x_raw.unsqueeze(0)
    return build_hierarchical_graph(x_raw, **kwargs)


def _dense_incidence_from_coo(
    node_idx: torch.Tensor,
    edge_idx: torch.Tensor,
    nnz: torch.Tensor,
    n_edges: int,
    n_nodes: int,
) -> torch.Tensor:
    """Unpack padded COO ``(B, max_nnz)`` into bool incidence ``(B, M, N)``."""
    batch, max_nnz = node_idx.shape
    inc = torch.zeros(batch, n_edges, n_nodes, dtype=torch.bool, device=node_idx.device)
    if max_nnz == 0 or n_edges == 0 or n_nodes == 0:
        return inc
    valid = torch.arange(max_nnz, device=node_idx.device).unsqueeze(0) < nnz.to(
        device=node_idx.device, dtype=torch.long
    ).unsqueeze(1)
    if not bool(valid.any()):
        return inc
    b_ids, k_ids = valid.nonzero(as_tuple=True)
    inc[
        b_ids,
        edge_idx[b_ids, k_ids].to(torch.long),
        node_idx[b_ids, k_ids].to(torch.long),
    ] = True
    return inc


def _batch_n_nodes(batch: dict[str, torch.Tensor]) -> int:
    if "incidence" in batch and batch["incidence"] is not None:
        return int(batch["incidence"].size(-1))
    if "x" in batch:
        return int(batch["x"].size(1))
    if "node_mask" in batch:
        return int(batch["node_mask"].size(-1))
    if "mask" in batch:
        return int(batch["mask"].size(-1))
    return 0


def _batch_incidence(
    batch: dict[str, torch.Tensor], n_edges: int, n_nodes: int
) -> torch.Tensor | None:
    if "incidence" in batch and batch["incidence"] is not None:
        return batch["incidence"].to(torch.bool)
    if "incidence_edge" in batch and "incidence_node" in batch and "incidence_nnz" in batch:
        return _dense_incidence_from_coo(
            batch["incidence_node"],
            batch["incidence_edge"],
            batch["incidence_nnz"],
            n_edges,
            n_nodes,
        )
    return None


def _compact_edge_axis(
    batch: dict[str, torch.Tensor], keep: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Keep ``keep`` edge slots and compact the edge axis (pad to batch max)."""
    if keep.dim() != 2:
        raise ValueError(f"keep must be (batch, n_edges); got {tuple(keep.shape)}")
    if bool(keep.all()):
        return batch

    batch_size, n_old = keep.shape
    n_keep = keep.sum(dim=-1)
    max_keep = int(n_keep.max().item()) if batch_size > 0 else 0
    device = keep.device
    n_nodes = _batch_n_nodes(batch)
    edge_type = batch.get("edge_type")
    incidence = _batch_incidence(batch, n_old, n_nodes)

    if max_keep == 0:
        if "edge_x" in batch:
            batch["edge_x"] = batch["edge_x"].new_zeros(
                batch_size, 0, batch["edge_x"].size(-1)
            )
        if edge_type is not None:
            batch["edge_type"] = edge_type.new_zeros(batch_size, 0)
        batch["edge_mask"] = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)
        inc = torch.zeros(batch_size, 0, n_nodes, dtype=torch.bool, device=device)
        batch["incidence"] = inc
        fields = finalize_incidence_storage(inc)
        for key, value in fields.items():
            if key != "incidence":
                batch[key] = value.to(device=device)
        return batch

    order = keep.to(torch.int64).argsort(dim=-1, descending=True)[:, :max_keep]
    kept_ok = torch.arange(max_keep, device=device).unsqueeze(0) < n_keep.unsqueeze(1)

    def _gather_edge_dim(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.size(1) != n_old:
            return tensor
        if tensor.dim() == 2:
            out = torch.gather(tensor, 1, order)
            if tensor.dtype == torch.bool:
                return out & kept_ok
            return torch.where(kept_ok, out, torch.zeros_like(out))
        gather_index = order.view(batch_size, max_keep, *([1] * (tensor.dim() - 2)))
        gather_index = gather_index.expand(-1, -1, *tensor.shape[2:])
        out = torch.gather(tensor, 1, gather_index)
        mask = kept_ok.view(batch_size, max_keep, *([1] * (tensor.dim() - 2)))
        if tensor.dtype == torch.bool:
            return out & mask
        return torch.where(mask, out, torch.zeros_like(out))

    if "edge_x" in batch:
        batch["edge_x"] = _gather_edge_dim(batch["edge_x"])
    if edge_type is not None:
        batch["edge_type"] = _gather_edge_dim(edge_type)
    if "edge_mask" in batch and batch["edge_mask"] is not None:
        batch["edge_mask"] = _gather_edge_dim(batch["edge_mask"].to(torch.bool)) & kept_ok
    else:
        batch["edge_mask"] = kept_ok

    if incidence is not None:
        incidence = _gather_edge_dim(incidence) & kept_ok.unsqueeze(-1)
        batch["incidence"] = incidence
        fields = finalize_incidence_storage(incidence)
        for key, value in fields.items():
            if key == "incidence":
                continue
            batch[key] = value.to(device=device)

    return batch


def drop_knn_edges_from_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop ``EDGE_TYPE_KNN`` rows and compact the edge axis.

    vn_link 2-edges are kept. Prefer :func:`drop_pairwise_edges_from_batch` for
    ``--hyperedge-only`` (any incidence degree ``<= 2``).
    """
    if "edge_type" not in batch or "edge_x" not in batch:
        return batch
    edge_type = batch["edge_type"]
    if edge_type.dim() != 2:
        raise ValueError(f"edge_type must be (batch, n_edges); got {tuple(edge_type.shape)}")
    return _compact_edge_axis(batch, edge_type != EDGE_TYPE_KNN)


def drop_pairwise_edges_from_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop every 2-edge (incidence degree ``<= 2``) and compact the edge axis.

    Train-time counterpart to rebuilding caches without pairwise slots: kNN,
    vn_link, and 2-member DBSCAN rows are removed. Remaining rows are true
    hyperedges (DBSCAN with ``>= 3`` members and virtual hypers), matching
    CAPEN-Llama-att ``M_{22}`` / class-token edge KV. Does not drop virtual
    *nodes* from ``X``.
    """
    if "edge_x" not in batch:
        return batch
    n_old = int(batch["edge_x"].size(1))
    n_nodes = _batch_n_nodes(batch)
    incidence = _batch_incidence(batch, n_old, n_nodes)
    if incidence is None:
        # Fall back to typed pairwise slots when incidence is missing.
        if "edge_type" not in batch:
            return batch
        edge_type = batch["edge_type"]
        keep = (edge_type != EDGE_TYPE_KNN) & (edge_type != EDGE_TYPE_VN_LINK)
        return _compact_edge_axis(batch, keep)
    keep = incidence.sum(dim=-1) > 2
    return _compact_edge_axis(batch, keep)


__all__ = [
    "EDGE_TYPE_KNN",
    "EDGE_TYPE_DBSCAN",
    "EDGE_TYPE_VN_LINK",
    "EDGE_TYPE_VIRTUAL_HYPER",
    "NODE_TYPE_PARTICLE",
    "NODE_TYPE_VIRTUAL",
    "EDGE_TYPE_NAMES",
    "HierarchicalGraph",
    "hierarchical_construction_tag",
    "parse_hierarchical_construction_tag",
    "build_hierarchical_graph",
    "build_hierarchical_graph_single",
    "drop_knn_edges_from_batch",
    "drop_pairwise_edges_from_batch",
]
