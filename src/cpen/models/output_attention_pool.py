"""Writeup class-token attentional pooling for coupled node--edge models.

Independent full-width (``H=1``) pools of \(X^{(L)}\) and \(E^{(L)}\), fused as
\(\frac{1}{\sqrt{2}}(P_1+P_2)\), then per-class unembedding
\(z_a=\frac{1}{D\sigma_U}P_{a,:}\cdot w_a\) with \(\sigma_U=D^{-1/2}\).

``SupportAttention`` supplies the \(\alpha_A=1\) scales: \(W_{QKVO}\sim\mathcal N(0,1)\),
projections \(1/\sqrt{D}\), logits \(1/D\) when ``heads=1``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from cpen.models.capen import SupportAttention


def default_output_attention_heads(width: int, heads: int | None = None) -> int:
    """Unused by the writeup pool (always one full-width head). Kept for callers."""
    del width, heads
    return 1


class OutputAttentionPool(nn.Module):
    """\(C\) class tokens pool nodes and edges independently, then unembed.

    ``node_only=True`` (BaselineTransformer) pools \(X^{(L)}\) only, with the
    same \(\alpha_A=1\) scales and per-class unembed. It does **not** apply
    \(\frac{1}{\sqrt{2}}\) — that factor is a two-stream fuse, not a readout
    temperature.
    """

    def __init__(
        self,
        width: int,
        n_classes: int,
        *,
        heads: int | None = None,  # noqa: ARG002 — writeup is a single full-width head
        dropout: float = 0.0,
        node_only: bool = False,
    ) -> None:
        super().__init__()
        d = int(width)
        c = int(n_classes)
        if d < 1 or c < 1:
            raise ValueError(f"width and n_classes must be >= 1; got {d}, {c}")
        self.width = d
        self.n_classes = c
        self.heads = 1
        self.node_only = bool(node_only)
        sigma = 1.0 / math.sqrt(d)
        self.class_tokens = nn.Parameter(torch.randn(c, d) * sigma)
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        drop = float(dropout or 0.0)
        self.attn_x = SupportAttention(d, heads=1, dropout=drop)
        self.attn_e = SupportAttention(d, heads=1, dropout=drop)
        self.class_unembed = nn.Parameter(torch.randn(c, d) * sigma)
        self._readout_scale = 1.0 / (d * sigma)
        self._fuse_scale = 0.5**0.5

    def _unused_edge_pool(self, h_e: torch.Tensor | None) -> torch.Tensor:
        unused = sum((p * 0.0).sum() for p in self.attn_e.parameters())
        if h_e is not None:
            unused = unused + (h_e * 0.0).sum()
        return unused

    def forward(
        self,
        h_x: torch.Tensor,
        h_e: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, n_nodes, _ = h_x.shape
        queries = self.ln(self.class_tokens).unsqueeze(0).expand(batch, -1, -1)
        z_x = self.ln(h_x)
        if mask is None:
            node_ok = torch.ones(batch, n_nodes, device=h_x.device, dtype=torch.bool)
        else:
            node_ok = mask.to(device=h_x.device, dtype=torch.bool)
        supp_x = node_ok.unsqueeze(1).expand(batch, self.n_classes, -1)
        pooled_x = self.attn_x(queries, z_x, supp_x)
        if self.node_only:
            logits = (pooled_x * self.class_unembed).sum(dim=-1) * self._readout_scale
            return logits + self._unused_edge_pool(h_e)
        if h_e is None:
            raise ValueError("OutputAttentionPool requires h_e unless node_only=True")
        n_edges = h_e.size(1)
        z_e = self.ln(h_e)
        if edge_mask is None:
            edge_ok = torch.ones(batch, n_edges, device=h_e.device, dtype=torch.bool)
        else:
            edge_ok = edge_mask.to(device=h_e.device, dtype=torch.bool)
        supp_e = edge_ok.unsqueeze(1).expand(batch, self.n_classes, -1)
        pooled = self._fuse_scale * (pooled_x + self.attn_e(queries, z_e, supp_e))
        return (pooled * self.class_unembed).sum(dim=-1) * self._readout_scale
