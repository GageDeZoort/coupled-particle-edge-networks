"""
Spectral coordinate utilities for Wave-Induced Rotary Encodings (WIRE).

Coordinates are graph-derived inputs (typically nontrivial Laplacian
eigenvectors). Eigen-decomposition is detached by default — do not backprop
through it. Laplacian eigenvectors have sign / degenerate-basis ambiguities;
optional sign canonicalization and training-time sign augmentation address
part of that without resolving within-degenerate rotations.
"""

from __future__ import annotations

import warnings

import torch


def _adjacency_from_edge_index(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros(num_nodes, num_nodes, dtype=dtype, device=edge_index.device)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape (2, E); got {tuple(edge_index.shape)}")
    adj = torch.zeros(num_nodes, num_nodes, dtype=dtype, device=edge_index.device)
    src, dst = edge_index[0].long(), edge_index[1].long()
    adj[src, dst] = 1.0
    adj[dst, src] = 1.0
    adj.fill_diagonal_(0.0)
    return adj


def _laplacian(adj: torch.Tensor, *, normalized: bool) -> torch.Tensor:
    """Build symmetric Laplacian from a dense nonnegative adjacency matrix."""
    deg = adj.sum(dim=-1)
    if not normalized:
        return torch.diag(deg) - adj
    # Normalized L = I - D^{-1/2} A D^{-1/2}; isolated nodes keep L_ii = 0.
    inv_sqrt = torch.zeros_like(deg)
    nonzero = deg > 0
    inv_sqrt[nonzero] = deg[nonzero].rsqrt()
    d_mat = torch.diag(inv_sqrt)
    eye = torch.eye(adj.size(0), dtype=adj.dtype, device=adj.device)
    return eye - d_mat @ adj @ d_mat


def _canonicalize_sign(vectors: torch.Tensor) -> torch.Tensor:
    """Flip each column so its largest-magnitude entry is positive."""
    if vectors.numel() == 0:
        return vectors
    out = vectors.clone()
    for k in range(out.size(1)):
        col = out[:, k]
        idx = int(col.abs().argmax())
        if col[idx] < 0:
            out[:, k] = -col
    return out


def _standardize_columns(vectors: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    if vectors.numel() == 0:
        return vectors
    mean = vectors.mean(dim=0, keepdim=True)
    centered = vectors - mean
    rms = centered.pow(2).mean(dim=0, keepdim=True).sqrt().clamp_min(eps)
    return centered / rms


def compute_wire_coordinates(
    edge_index: torch.Tensor,
    num_nodes: int,
    num_frequencies: int,
    *,
    normalized_laplacian: bool = True,
    eigenvalue_tolerance: float = 1e-8,
    standardize: bool = True,
    canonicalize_sign: bool = True,
    warn_on_pad: bool = True,
) -> torch.Tensor:
    """
    Nontrivial Laplacian eigenvectors as WIRE coordinates for one graph.

    Returns ``[num_nodes, num_frequencies]``. Zero modes with
    ``lambda <= eigenvalue_tolerance`` are dropped (important for disconnected
    graphs). If fewer than ``num_frequencies`` nontrivial modes exist, remaining
    columns are zero-padded and a warning is issued.

    The returned tensor is detached (no autograd through the eigensolver).
    Sign canonicalization forces the largest-magnitude entry of each retained
    eigenvector to be positive; it does **not** resolve arbitrary basis
    rotations within degenerate eigenspaces.
    """
    if num_frequencies < 1:
        raise ValueError(f"num_frequencies must be >= 1; got {num_frequencies}")
    if num_nodes < 1:
        raise ValueError(f"num_nodes must be >= 1; got {num_nodes}")

    device = edge_index.device
    adj = _adjacency_from_edge_index(edge_index, num_nodes)
    lap = _laplacian(adj, normalized=normalized_laplacian)
    # Symmetric eigensolver; use float64 for stability on small graphs.
    evals, evecs = torch.linalg.eigh(lap)
    order = torch.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    nontrivial = evals > eigenvalue_tolerance
    evecs_nt = evecs[:, nontrivial]
    n_keep = min(num_frequencies, int(evecs_nt.size(1)))
    coords = torch.zeros(num_nodes, num_frequencies, dtype=torch.float32, device=device)
    if n_keep > 0:
        chosen = evecs_nt[:, :n_keep].to(dtype=torch.float32)
        if canonicalize_sign:
            chosen = _canonicalize_sign(chosen)
        if standardize:
            chosen = _standardize_columns(chosen)
        coords[:, :n_keep] = chosen
    if n_keep < num_frequencies and warn_on_pad:
        warnings.warn(
            f"WIRE coordinates: only {n_keep} nontrivial Laplacian modes for "
            f"num_nodes={num_nodes}; padding {num_frequencies - n_keep} zero columns.",
            stacklevel=2,
        )
    return coords.detach()


def particle_adjacency_from_incidence(
    incidence: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Dense particle co-occurrence adjacency from incidence ``S`` (B, N_e, N_p).

    ``A = 1[S^T S > 0]`` with zeros on the diagonal and padded rows/cols cleared.
    """
    s = incidence.to(dtype=torch.float32)
    adj = torch.bmm(s.transpose(1, 2), s) > 0.0
    n = adj.size(-1)
    eye = torch.eye(n, device=adj.device, dtype=torch.bool)
    adj = adj & ~eye
    if mask is not None:
        alive = mask.to(dtype=torch.bool)
        adj = adj & alive.unsqueeze(-1) & alive.unsqueeze(-2)
    return adj


def compute_wire_coordinates_batched(
    adjacency: torch.Tensor,
    mask: torch.Tensor,
    num_frequencies: int,
    *,
    normalized_laplacian: bool = True,
    eigenvalue_tolerance: float = 1e-8,
    standardize: bool = True,
    canonicalize_sign: bool = True,
    warn_on_pad: bool = True,
) -> torch.Tensor:
    """
    Per-graph WIRE coordinates for a padded batch.

    Parameters
    ----------
    adjacency:
        ``(B, N, N)`` bool/float particle adjacency (no self-loops).
    mask:
        ``(B, N)`` True for real nodes.

    Returns
    -------
    coords:
        ``(B, N, m)`` float32; padded slots are zero. Each graph is solved
        independently — never eigh the block-diagonal batch graph as one system.
    """
    if adjacency.dim() != 3:
        raise ValueError(f"adjacency must be (B, N, N); got {tuple(adjacency.shape)}")
    batch, n_nodes, n2 = adjacency.shape
    if n_nodes != n2:
        raise ValueError(f"adjacency must be square; got {tuple(adjacency.shape)}")
    if mask.shape != (batch, n_nodes):
        raise ValueError(f"mask shape {tuple(mask.shape)} != {(batch, n_nodes)}")

    out = torch.zeros(batch, n_nodes, num_frequencies, dtype=torch.float32, device=adjacency.device)
    for b in range(batch):
        alive = mask[b].to(dtype=torch.bool)
        idx = alive.nonzero(as_tuple=False).squeeze(-1)
        n_alive = int(idx.numel())
        if n_alive == 0:
            continue
        # Build COO edge_index on the induced subgraph of live nodes.
        sub_adj = adjacency[b].index_select(0, idx).index_select(1, idx).to(dtype=torch.float64)
        sub_adj = ((sub_adj + sub_adj.transpose(0, 1)) > 0).to(dtype=torch.float64)
        sub_adj.fill_diagonal_(0.0)
        src, dst = sub_adj.nonzero(as_tuple=True)
        edge_index = torch.stack([src, dst], dim=0)
        coords_sub = compute_wire_coordinates(
            edge_index,
            n_alive,
            num_frequencies,
            normalized_laplacian=normalized_laplacian,
            eigenvalue_tolerance=eigenvalue_tolerance,
            standardize=standardize,
            canonicalize_sign=canonicalize_sign,
            warn_on_pad=warn_on_pad,
        )
        out[b, idx] = coords_sub.to(device=out.device)
    return out.detach()


def augment_wire_coordinate_signs(
    coordinates: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Randomly flip each coordinate column with a graph-shared sign in ``{-1,+1}``.

    ``coordinates`` is ``(B, N, m)`` (or ``(N, m)``). One sign is drawn per
    graph and frequency; never flip individual nodes independently.
    """
    if coordinates.dim() == 2:
        coords = coordinates.unsqueeze(0)
        squeeze = True
        if mask is not None and mask.dim() == 1:
            mask_b = mask.unsqueeze(0)
        else:
            mask_b = mask
    elif coordinates.dim() == 3:
        coords = coordinates
        squeeze = False
        mask_b = mask
    else:
        raise ValueError(f"coordinates must be (N, m) or (B, N, m); got {tuple(coordinates.shape)}")

    batch, _, n_freq = coords.shape
    device = coords.device
    signs = torch.empty(batch, 1, n_freq, device=device, dtype=coords.dtype)
    # Bernoulli(0.5) -> ±1
    bits = torch.randint(0, 2, (batch, 1, n_freq), device=device, generator=generator)
    signs.copy_(bits.mul(2).sub(1).to(dtype=coords.dtype))
    out = coords * signs
    if mask_b is not None:
        out = out * mask_b.to(dtype=out.dtype).unsqueeze(-1)
    return out.squeeze(0) if squeeze else out
