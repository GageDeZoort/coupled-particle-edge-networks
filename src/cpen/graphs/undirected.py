"""Undirected edge helpers shared by Pascal / stream graph caches."""

from __future__ import annotations

import torch


def canonical_undirected_edges(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    *,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deduplicate directed PyG edges into unordered pairs, averaging features."""
    src, dst = edge_index.to(torch.long)
    keep = src != dst
    src, dst, attrs = src[keep], dst[keep], edge_attr[keep].to(torch.float32)
    lo, hi = torch.minimum(src, dst), torch.maximum(src, dst)
    keys = lo * num_nodes + hi
    unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)

    features = attrs.new_zeros((unique_keys.numel(), attrs.size(-1)))
    features.index_add_(0, inverse, attrs)
    counts = attrs.new_zeros(unique_keys.numel())
    counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=attrs.dtype))
    features = features / counts.clamp_min(1).unsqueeze(-1)
    pairs = torch.stack(
        (unique_keys.div(num_nodes, rounding_mode="floor"), unique_keys.remainder(num_nodes)),
        dim=0,
    )
    return pairs, features
