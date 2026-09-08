"""Sparse COO incidence operators for CPEN (replaces dense bmm on S)."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class BatchedIncidenceCOO:
    """Padded batch of incidence COO entries with shape ``(B, M, N)``."""

    node_idx: torch.Tensor  # (B, max_nnz) int
    edge_idx: torch.Tensor  # (B, max_nnz) int
    nnz: torch.Tensor  # (B,) int
    num_edges: int
    num_nodes: int
    # Lazily materialized flat COO + scatter offsets, reused across every
    # operator application in a forward pass. Building these requires a
    # data-dependent boolean mask (one device sync); caching amortizes that
    # single sync over the ~24 operator applications per forward instead of
    # paying it each time.
    _cache: dict = field(default_factory=dict, compare=False, hash=False, repr=False)

    def flat_indices(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(batch, edge, node)`` triples for valid COO entries."""
        cached = self._cache.get("flat")
        if cached is None:
            cached = _flat_coo(self.node_idx, self.edge_idx, self.nnz)
            self._cache["flat"] = cached
        return cached

    def edge_scatter_index(self) -> torch.Tensor:
        """Flat scatter target ``batch * num_edges + edge`` for S applications."""
        idx = self._cache.get("edge_scatter")
        if idx is None:
            batch_ids, edge_ids, _ = self.flat_indices()
            idx = batch_ids * self.num_edges + edge_ids
            self._cache["edge_scatter"] = idx
        return idx

    def node_scatter_index(self) -> torch.Tensor:
        """Flat scatter target ``batch * num_nodes + node`` for S^T applications."""
        idx = self._cache.get("node_scatter")
        if idx is None:
            batch_ids, _, node_ids = self.flat_indices()
            idx = batch_ids * self.num_nodes + node_ids
            self._cache["node_scatter"] = idx
        return idx


def compute_incidence_degrees(
    incidence: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Inverse node and edge degrees from incidence ``(B, M, N)`` (bool or float).

    node_degree_inv[b, j] = 1 / sum_e S[b, e, j]
    edge_degree_inv[b, e] = 1 / sum_j S[b, e, j]
    """
    if incidence.dtype == torch.bool:
        edge_degree = incidence.sum(dim=2, dtype=torch.float32).clamp_min(eps)
        node_degree = incidence.sum(dim=1, dtype=torch.float32).clamp_min(eps)
    else:
        edge_degree = incidence.sum(dim=2).clamp_min(eps)
        node_degree = incidence.sum(dim=1).clamp_min(eps)
    return node_degree.pow(-1.0), edge_degree.pow(-1.0)


def pack_incidence_coo(incidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pack bool/float incidence ``(B, M, N)`` into padded COO arrays.

    Returns ``node_idx (B, max_nnz)``, ``edge_idx (B, max_nnz)``, ``nnz (B,)``.
    """
    if incidence.dim() != 3:
        raise ValueError(f"incidence must be (batch, n_edges, n_particles); got {tuple(incidence.shape)}")

    inc = incidence.to(torch.bool)
    batch_size, n_edges, n_nodes = inc.shape
    nnz = inc.sum(dim=(1, 2)).to(torch.int32)
    max_nnz = int(nnz.max().item()) if batch_size > 0 else 0
    if max_nnz == 0:
        empty_n = torch.zeros(batch_size, 0, dtype=torch.int16, device=inc.device)
        return empty_n, empty_n.clone(), nnz

    node_padded = torch.zeros(batch_size, max_nnz, dtype=torch.int16, device=inc.device)
    edge_padded = torch.zeros(batch_size, max_nnz, dtype=torch.int16, device=inc.device)
    batch_i, edge_i, node_i = inc.nonzero(as_tuple=True)
    offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=inc.device)
    offsets[1:] = nnz.to(torch.long).cumsum(0)
    within = torch.arange(batch_i.numel(), device=inc.device) - offsets[batch_i]
    node_padded[batch_i, within] = node_i.to(torch.int16)
    edge_padded[batch_i, within] = edge_i.to(torch.int16)
    return node_padded, edge_padded, nnz


def finalize_incidence_storage(incidence: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build bool incidence + padded COO + precomputed degrees for cache storage."""
    inc_bool = incidence.to(torch.bool)
    node_idx, edge_idx, nnz = pack_incidence_coo(inc_bool)
    node_degree_inv, edge_degree_inv = compute_incidence_degrees(inc_bool)
    return {
        "incidence": inc_bool,
        "incidence_node": node_idx,
        "incidence_edge": edge_idx,
        "incidence_nnz": nnz,
        "node_degree_inv": node_degree_inv.to(torch.float32),
        "edge_degree_inv": edge_degree_inv.to(torch.float32),
    }


def coo_from_dense_incidence(incidence: torch.Tensor) -> BatchedIncidenceCOO:
    """Build :class:`BatchedIncidenceCOO` from dense bool/float incidence on the fly."""
    node_idx, edge_idx, nnz = pack_incidence_coo(incidence)
    return BatchedIncidenceCOO(
        node_idx=node_idx,
        edge_idx=edge_idx,
        nnz=nnz,
        num_edges=incidence.size(-2),
        num_nodes=incidence.size(-1),
    )


def coo_from_batch_tensors(
    *,
    incidence_node: torch.Tensor,
    incidence_edge: torch.Tensor,
    incidence_nnz: torch.Tensor,
    num_edges: int,
    num_nodes: int,
) -> BatchedIncidenceCOO:
    return BatchedIncidenceCOO(
        node_idx=incidence_node,
        edge_idx=incidence_edge,
        nnz=incidence_nnz,
        num_edges=num_edges,
        num_nodes=num_nodes,
    )


def _flat_coo(
    node_idx: torch.Tensor,
    edge_idx: torch.Tensor,
    nnz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, max_nnz = node_idx.shape
    device = node_idx.device
    batch_ids = torch.arange(batch_size, device=device).unsqueeze(1).expand(batch_size, max_nnz)
    valid = torch.arange(max_nnz, device=device).unsqueeze(0) < nnz.to(device).unsqueeze(1)
    return batch_ids[valid], edge_idx[valid].to(torch.long), node_idx[valid].to(torch.long)


def apply_t12_unnorm(
    h: torch.Tensor,
    coo: BatchedIncidenceCOO,
) -> torch.Tensor:
    """Unnormalized T_12 = S h, particle (B, N, D) -> edge (B, M, D)."""
    batch_ids, edge_ids, node_ids = coo.flat_indices()
    if batch_ids.numel() == 0:
        return h.new_zeros(h.size(0), coo.num_edges, h.size(-1))
    gathered = h[batch_ids, node_ids]
    out = h.new_zeros(h.size(0) * coo.num_edges, h.size(-1))
    out.index_add_(0, coo.edge_scatter_index(), gathered)
    return out.view(h.size(0), coo.num_edges, h.size(-1))


def apply_t21_unnorm(
    h: torch.Tensor,
    coo: BatchedIncidenceCOO,
) -> torch.Tensor:
    """Unnormalized T_21 = S^T h, edge (B, M, D) -> particle (B, N, D)."""
    batch_ids, edge_ids, node_ids = coo.flat_indices()
    if batch_ids.numel() == 0:
        return h.new_zeros(h.size(0), coo.num_nodes, h.size(-1))
    gathered = h[batch_ids, edge_ids]
    out = h.new_zeros(h.size(0) * coo.num_nodes, h.size(-1))
    out.index_add_(0, coo.node_scatter_index(), gathered)
    return out.view(h.size(0), coo.num_nodes, h.size(-1))


def apply_t12(
    h: torch.Tensor,
    coo: BatchedIncidenceCOO,
    edge_degree_inv: torch.Tensor,
) -> torch.Tensor:
    """T_12 = D_E^{-1} S h."""
    return apply_t12_unnorm(h, coo) * edge_degree_inv.unsqueeze(-1)


def apply_t21(
    h: torch.Tensor,
    coo: BatchedIncidenceCOO,
    node_degree_inv: torch.Tensor,
) -> torch.Tensor:
    """T_21 = D_X^{-1} S^T h."""
    return apply_t21_unnorm(h, coo) * node_degree_inv.unsqueeze(-1)


def apply_t22_unnorm(
    h: torch.Tensor,
    coo: BatchedIncidenceCOO,
) -> torch.Tensor:
    """Unnormalized T_22 = S S^T h."""
    return apply_t12_unnorm(apply_t21_unnorm(h, coo), coo)


def apply_t22(
    h: torch.Tensor,
    coo: BatchedIncidenceCOO,
    node_degree_inv: torch.Tensor,
    edge_degree_inv: torch.Tensor,
) -> torch.Tensor:
    """T_22 = D_E^{-1} S D_X^{-1} S^T h."""
    lowered = apply_t21(h, coo, node_degree_inv)
    return apply_t12(lowered, coo, edge_degree_inv)


def apply_t11_sts_unnorm(
    h: torch.Tensor,
    coo: BatchedIncidenceCOO,
) -> torch.Tensor:
    """Unnormalized T_11 = S^T S h."""
    return apply_t21_unnorm(apply_t12_unnorm(h, coo), coo)


def apply_t11_sts(
    h: torch.Tensor,
    coo: BatchedIncidenceCOO,
    node_degree_inv: torch.Tensor,
    edge_degree_inv: torch.Tensor,
) -> torch.Tensor:
    """T_11 = D_X^{-1} S^T D_E^{-1} S h."""
    return apply_t21(apply_t12(h, coo, edge_degree_inv), coo, node_degree_inv)


def attach_sparse_incidence_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Add padded COO + precomputed degrees when only dense incidence is present."""
    if "incidence_node" in batch or "incidence" not in batch:
        return batch
    fields = finalize_incidence_storage(batch["incidence"])
    device = batch["incidence"].device
    for key, value in fields.items():
        batch[key] = value.to(device)
    return batch


def beta_pooling_weights(
    alpha: torch.Tensor,
    coo: BatchedIncidenceCOO,
) -> torch.Tensor:
    """Edge pooling weights beta = normalize(S alpha)."""
    batch_ids, edge_ids, node_ids = coo.flat_indices()
    beta = alpha.new_zeros(alpha.size(0) * coo.num_edges)
    if batch_ids.numel() > 0:
        beta.index_add_(0, coo.edge_scatter_index(), alpha[batch_ids, node_ids])
    beta = beta.view(alpha.size(0), coo.num_edges)
    return beta / beta.sum(dim=-1, keepdim=True).clamp_min(1e-12)


# ---------------------------------------------------------------------------
# Operator backends
#
# At small graph sizes (N, M ~ 100) the COO scatter/gather path is dominated by
# ``index_add_`` atomic kernels (especially slow in bf16), which profile at
# ~85% of GPU time while the actual linears are ~5%. A dense batched matmul with
# the incidence matrix S is only ~0.6 GFLOP/step at this scale (microseconds on
# an A100) and its cost is independent of edge count, so it is far faster here.
# Both backends expose the same primitives so CPEN is backend-agnostic.
# ---------------------------------------------------------------------------


class SparseIncidenceOps:
    """COO scatter/gather incidence operators (best for large, sparse graphs)."""

    def __init__(self, coo: BatchedIncidenceCOO) -> None:
        self.coo = coo
        self.num_edges = coo.num_edges
        self.num_nodes = coo.num_nodes

    def t12_unnorm(self, h: torch.Tensor) -> torch.Tensor:
        return apply_t12_unnorm(h, self.coo)

    def t21_unnorm(self, h: torch.Tensor) -> torch.Tensor:
        return apply_t21_unnorm(h, self.coo)

    def t22_unnorm(self, h: torch.Tensor) -> torch.Tensor:
        return apply_t22_unnorm(h, self.coo)

    def t11_sts_unnorm(self, h: torch.Tensor) -> torch.Tensor:
        return apply_t11_sts_unnorm(h, self.coo)

    def t12(self, h: torch.Tensor, edge_degree_inv: torch.Tensor) -> torch.Tensor:
        return apply_t12(h, self.coo, edge_degree_inv)

    def t21(self, h: torch.Tensor, node_degree_inv: torch.Tensor) -> torch.Tensor:
        return apply_t21(h, self.coo, node_degree_inv)

    def t22(
        self, h: torch.Tensor, node_degree_inv: torch.Tensor, edge_degree_inv: torch.Tensor
    ) -> torch.Tensor:
        return apply_t22(h, self.coo, node_degree_inv, edge_degree_inv)

    def t11_sts(
        self, h: torch.Tensor, node_degree_inv: torch.Tensor, edge_degree_inv: torch.Tensor
    ) -> torch.Tensor:
        return apply_t11_sts(h, self.coo, node_degree_inv, edge_degree_inv)

    def beta_pooling(self, alpha: torch.Tensor) -> torch.Tensor:
        return beta_pooling_weights(alpha, self.coo)


def dense_incidence_from_coo(
    coo: BatchedIncidenceCOO,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Reconstruct dense incidence S ``(B, M, N)`` from padded COO on device.

    A single scatter, far cheaper than reading the dense incidence from disk
    (~10 KB/jet) every batch. Lets training skip the dense-incidence cache read
    and rebuild S on-GPU for the dense operator backend.
    """
    device = coo.node_idx.device
    batch = coo.node_idx.size(0)
    S = torch.zeros(batch, coo.num_edges, coo.num_nodes, dtype=dtype, device=device)
    batch_ids, edge_ids, node_ids = coo.flat_indices()
    if batch_ids.numel() > 0:
        S[batch_ids, edge_ids, node_ids] = 1
    return S


class DenseIncidenceOps:
    """Dense ``bmm`` incidence operators (best for small graphs, N, M ~ 100)."""

    def __init__(self, incidence: torch.Tensor, *, dtype: torch.dtype) -> None:
        # incidence: (B, M, N) bool/float. Cast once to compute dtype.
        self.S = incidence.to(dtype)
        self.St = self.S.transpose(1, 2).contiguous()
        self.num_edges = incidence.size(-2)
        self.num_nodes = incidence.size(-1)

    @classmethod
    def from_coo(
        cls,
        coo: BatchedIncidenceCOO,
        *,
        dtype: torch.dtype,
    ) -> "DenseIncidenceOps":
        """Build dense operators by reconstructing S from padded COO tensors."""
        return cls(dense_incidence_from_coo(coo, dtype=dtype), dtype=dtype)

    def t12_unnorm(self, h: torch.Tensor) -> torch.Tensor:
        return torch.bmm(self.S, h)

    def t21_unnorm(self, h: torch.Tensor) -> torch.Tensor:
        return torch.bmm(self.St, h)

    def t22_unnorm(self, h: torch.Tensor) -> torch.Tensor:
        return self.t12_unnorm(self.t21_unnorm(h))

    def t11_sts_unnorm(self, h: torch.Tensor) -> torch.Tensor:
        return self.t21_unnorm(self.t12_unnorm(h))

    def t12(self, h: torch.Tensor, edge_degree_inv: torch.Tensor) -> torch.Tensor:
        return self.t12_unnorm(h) * edge_degree_inv.unsqueeze(-1)

    def t21(self, h: torch.Tensor, node_degree_inv: torch.Tensor) -> torch.Tensor:
        return self.t21_unnorm(h) * node_degree_inv.unsqueeze(-1)

    def t22(
        self, h: torch.Tensor, node_degree_inv: torch.Tensor, edge_degree_inv: torch.Tensor
    ) -> torch.Tensor:
        return self.t12(self.t21(h, node_degree_inv), edge_degree_inv)

    def t11_sts(
        self, h: torch.Tensor, node_degree_inv: torch.Tensor, edge_degree_inv: torch.Tensor
    ) -> torch.Tensor:
        return self.t21(self.t12(h, edge_degree_inv), node_degree_inv)

    def beta_pooling(self, alpha: torch.Tensor) -> torch.Tensor:
        beta = torch.bmm(self.S, alpha.unsqueeze(-1)).squeeze(-1)
        return beta / beta.sum(dim=-1, keepdim=True).clamp_min(1e-12)
