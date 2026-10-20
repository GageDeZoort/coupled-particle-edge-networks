"""Writeup class-token attentional pooling for coupled node--edge models.

CompleteP classification is non-negotiable:

\[
\mathrm{logit}_a = \frac{1}{D\sigma}\, X_{\mathrm{out},a}\cdot w_a,
\qquad w_a\sim\mathcal N(0,\sigma^2),\quad\sigma=D^{-1/2}.
\]

Attentional pooling only builds the **feature** \(X_{\mathrm{out}}\) (Θ(1) in
width). It does **not** replace the readout. Structure:

1. Trunk-style \(\alpha_A=1\) MHSA (\(W_O\!:D\to D\), ×\(1/\sqrt{D}\)) pools
   each stream → Θ(1) class features \(P_X,P_E\).
2. Two-stream fuse \(X_{\mathrm{out}}=\frac{1}{\sqrt{2}}(P_X+P_E)\) (fuse only).
3. CompleteP unembed \(\frac{1}{D\sigma}\,X_{\mathrm{out}}W\).

Folding the head into \(W_O\!:D\to 1\) (α_A=1 init) is **not** the same map as
\(W\sim\mathcal N(0,\sigma^2)\) with ×\(1/(D\sigma)\), and breaks width-flat
output \(\Delta\rho\). Do not omit step 3.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from cpen.models.capen import SupportAttention


def default_output_attention_heads(width: int, heads: int | None = None) -> int:
    """Pool head count: match trunk heads when provided and valid."""
    d = int(width)
    if heads is not None and int(heads) >= 1 and d % int(heads) == 0:
        return int(heads)
    # Fall back to d_head=32 when possible (CompleteP JetClass default).
    if d >= 32 and d % 32 == 0:
        return d // 32
    return 1


class OutputAttentionPool(nn.Module):
    """\(C\) class tokens → Θ(1) features → CompleteP unembed.

    ``node_only=True`` (BaselineTransformer) pools \(X^{(L)}\) only and skips
    the \(\frac{1}{\sqrt{2}}\) fuse (one stream).

    Pool attention matches the trunk (\(\alpha_A=1\), concat, \(W_O\!:D\to D\)).
    Classification uses a separate \(w_a\sim\mathcal N(0,\sigma^2)\) with
    forward ×\(1/(D\sigma)\).
    """

    def __init__(
        self,
        width: int,
        n_classes: int,
        *,
        heads: int | None = None,
        dropout: float = 0.0,
        node_only: bool = False,
    ) -> None:
        super().__init__()
        d = int(width)
        c = int(n_classes)
        if d < 1 or c < 1:
            raise ValueError(f"width and n_classes must be >= 1; got {d}, {c}")
        h = default_output_attention_heads(d, heads)
        if d % h != 0:
            raise ValueError(f"width={d} must be divisible by pool heads={h}")
        self.width = d
        self.n_classes = c
        self.heads = h
        self.head_dim = d // h
        self.node_only = bool(node_only)
        sigma = 1.0 / math.sqrt(d)
        self.class_tokens = nn.Parameter(torch.randn(c, d) * sigma)
        self.ln = nn.LayerNorm(d, elementwise_affine=False)
        drop = float(dropout or 0.0)
        # α_A=1 feature pool: W_O : D→D, ×1/√D → Θ(1) class features.
        self.attn_x = SupportAttention(d, heads=h, dropout=drop, head_combine="concat")
        self.attn_e = SupportAttention(d, heads=h, dropout=drop, head_combine="concat")
        # CompleteP head: W ~ N(0, σ²), forward × 1/(Dσ) = 1/√D.
        self.class_unembed = nn.Parameter(torch.randn(c, d) * sigma)
        self._readout_scale = 1.0 / (d * sigma)
        self._fuse_scale = 0.5**0.5

    def _unused_edge_pool(self, h_e: torch.Tensor | None) -> torch.Tensor:
        unused = sum((p * 0.0).sum() for p in self.attn_e.parameters())
        if h_e is not None:
            unused = unused + (h_e * 0.0).sum()
        return unused

    def pool_streams(
        self,
        h_x: torch.Tensor,
        h_e: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Return ``(pooled_x, pooled_e, fused)`` each ``(B, C, D)`` features.

        ``pooled_e`` is ``None`` when ``node_only``. ``fused`` is
        \(X_{\mathrm{out}}\) — Θ(1) features before the CompleteP unembed.
        """
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
            return pooled_x, None, pooled_x
        if h_e is None:
            raise ValueError("OutputAttentionPool requires h_e unless node_only=True")
        n_edges = h_e.size(1)
        z_e = self.ln(h_e)
        if edge_mask is None:
            edge_ok = torch.ones(batch, n_edges, device=h_e.device, dtype=torch.bool)
        else:
            edge_ok = edge_mask.to(device=h_e.device, dtype=torch.bool)
        supp_e = edge_ok.unsqueeze(1).expand(batch, self.n_classes, -1)
        pooled_e = self.attn_e(queries, z_e, supp_e)
        fused = self._fuse_scale * (pooled_x + pooled_e)
        return pooled_x, pooled_e, fused

    def forward(
        self,
        h_x: torch.Tensor,
        h_e: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _pooled_x, _pooled_e, x_out = self.pool_streams(
            h_x, h_e, mask=mask, edge_mask=edge_mask
        )
        # logit = 1/(Dσ) X_out W,  W ~ N(0, σ²).
        logits = (x_out * self.class_unembed).sum(dim=-1) * self._readout_scale
        if self.node_only:
            return logits + self._unused_edge_pool(h_e)
        return logits
