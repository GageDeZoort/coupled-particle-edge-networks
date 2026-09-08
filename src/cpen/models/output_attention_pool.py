"""Class-token MHSA graph readout (optional; ``--readout-mode attn``).

Residual \(X^{(L)},E^{(L)}\) are keys/values with no LayerNorm. \(C\) class
tokens are LN'd queries. ``SupportAttention`` uses the CAPEN-Llama scales
(\(W_{QKVO}\sim\mathcal{N}(0,1)\), proj \(1/\sqrt{D}\), scores \(1/d_h\)).
Logits are the μP linear map \(z_c=U_c w_c/\sqrt{D}\), \(w_c\sim\mathcal{N}(0,D^{-1})\).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from cpen.models.capen import SupportAttention


def default_output_attention_heads(width: int, heads: int | None = None) -> int:
    """Prefer CLI ``heads`` when it divides \(D\); else 16/8/4/2/1."""
    width = int(width)
    if heads is not None and int(heads) > 1:
        h = int(heads)
        if width % h != 0:
            raise ValueError(f"width {width} is not divisible by heads {h}")
        return h
    for candidate in (16, 8, 4, 2, 1):
        if width % candidate == 0:
            return candidate
    return 1


class OutputAttentionPool(nn.Module):
    """\(C\) class tokens attend to every live node and edge."""

    def __init__(
        self,
        width: int,
        n_classes: int,
        *,
        heads: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d = int(width)
        c = int(n_classes)
        if d < 1 or c < 1:
            raise ValueError(f"width and n_classes must be >= 1; got {d}, {c}")
        self.width = d
        self.n_classes = c
        self.heads = default_output_attention_heads(d, heads)
        sigma = 1.0 / math.sqrt(d)
        self.class_tokens = nn.Parameter(torch.randn(c, d) * sigma)
        self.query_ln = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = SupportAttention(d, heads=self.heads, dropout=float(dropout or 0.0))
        self.class_decoder = nn.Linear(d, 1, bias=False)
        nn.init.normal_(self.class_decoder.weight, mean=0.0, std=sigma)
        self._readout_scale = 1.0 / (d * sigma)

    def forward(
        self,
        h_x: torch.Tensor,
        h_e: torch.Tensor,
        mask: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, n_nodes, _ = h_x.shape
        n_edges = h_e.size(1)
        queries = self.query_ln(self.class_tokens).unsqueeze(0).expand(batch, -1, -1)
        kv = torch.cat([h_x, h_e], dim=1)
        if mask is None:
            node_ok = torch.ones(batch, n_nodes, device=h_x.device, dtype=torch.bool)
        else:
            node_ok = mask.to(device=h_x.device, dtype=torch.bool)
        if edge_mask is None:
            edge_ok = torch.ones(batch, n_edges, device=h_e.device, dtype=torch.bool)
        else:
            edge_ok = edge_mask.to(device=h_e.device, dtype=torch.bool)
        live = torch.cat([node_ok, edge_ok], dim=-1)
        support = live.unsqueeze(1).expand(batch, self.n_classes, -1)
        pooled = self.attn(queries, kv, support)
        return self.class_decoder(pooled).squeeze(-1) * self._readout_scale
