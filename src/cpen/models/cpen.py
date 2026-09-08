"""Coupled particle-edge network (CPEN) with graph incidence operators.

The core implementation currently lives in ``cpen.pyc`` (source was lost in a
restore). This module loads that bytecode and extends ``CPEN`` so ``dropout``
is applied in the transformer-like MLP residual branches:

    h ← h + scale * Dropout( MLP( LN(h) ) )

With ``residual_structure='standard'`` there is no separate MLP block, so
dropout has no effect (argument is still accepted for CLI uniformity).

This wrapper is CPEN-only. CAPEN / CAPEN-Llama are separate modules and keep
their existing attention-only dropout (``SupportAttention``); they do not
import or subclass anything from here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn

_PYC = Path(__file__).resolve().with_name("cpen.pyc")
if not _PYC.is_file():
    raise ImportError(f"Missing bytecode backend {_PYC}")

_spec = importlib.util.spec_from_file_location("cpen.models._cpen_bc", _PYC)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load {_PYC}")
_bc = importlib.util.module_from_spec(_spec)
# Register before exec so the bytecode module can resolve itself if needed.
sys.modules.setdefault("cpen.models._cpen_bc", _bc)
_spec.loader.exec_module(_bc)

# Re-export the bytecode public API; ``CPEN`` is overridden below.
for _name, _value in vars(_bc).items():
    if _name.startswith("__"):
        continue
    globals()[_name] = _value

_BaseCPEN = _bc.CPEN


class CPEN(_BaseCPEN):
    """Bytecode CPEN + optional MLP dropout (transformer-like residuals).

    ``x_only_node_readout`` (Pascal study): decode ``X^{(L)}`` with ``W_X`` and
    optionally ``E^{(L)}`` with ``W_E``, skipping the stock node-mode mix
    ``(z_X + T_{21} z_E)/√2`` that requires ``decoder_e`` to match node classes.
    """

    def __init__(  # type: ignore[no-untyped-def]
        self,
        *args,
        dropout: float = 0.0,
        output_attention_heads: int | None = None,
        **kwargs,
    ):
        requested = str(kwargs.get("readout_mode", "graph") or "graph")
        if requested == "attn":
            kwargs = dict(kwargs)
            kwargs["readout_mode"] = "graph"
        super().__init__(*args, **kwargs)
        p = float(dropout)
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout must be in [0, 1); got {p}")
        self.dropout = p
        self.mlp_dropout = nn.Dropout(p)
        self.x_only_node_readout = False
        # None = stock graph/node readout. True/False = MNIST pooled z_X vs mix.
        self.x_only_graph_readout = None
        self.output_attention_pool = None
        if requested == "attn":
            from cpen.models.output_attention_pool import OutputAttentionPool

            self.output_attention_pool = OutputAttentionPool(
                width=int(self.D),
                n_classes=int(self.out_dim),
                heads=output_attention_heads,
                dropout=p,
            )

    def _mlp_block(self, h: torch.Tensor, w1: nn.Module, w2: nn.Module) -> torch.Tensor:
        """W2 GELU(W1 h) with muP scales, then Dropout (train-time only)."""
        return self.mlp_dropout(super()._mlp_block(h, w1, w2))

    def _forward_hidden(  # type: ignore[no-untyped-def]
        self,
        x: torch.Tensor,
        edge_x: torch.Tensor,
        incidence: torch.Tensor | None = None,
        *,
        incidence_node: torch.Tensor | None = None,
        incidence_edge: torch.Tensor | None = None,
        incidence_nnz: torch.Tensor | None = None,
        node_degree_inv: torch.Tensor | None = None,
        edge_degree_inv: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        z: torch.Tensor | None = None,  # noqa: ARG002
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode + residual layers; same operator order as bytecode ``forward``."""
        if x.dim() != 3:
            raise ValueError(
                f"Expected x shape (batch, n_particles, n_features); got {tuple(x.shape)}"
            )
        num_edges = edge_x.size(-2)
        num_nodes = x.size(-2)
        h_x = self.encoder_x(x) * self._encoder_x_scale
        h_e = self.encoder_e(edge_x) * self._encoder_e_scale
        if self.operator_backend == "dense":
            if incidence is None:
                coo = self._resolve_incidence(
                    incidence=None,
                    incidence_node=incidence_node,
                    incidence_edge=incidence_edge,
                    incidence_nnz=incidence_nnz,
                    num_edges=num_edges,
                    num_nodes=num_nodes,
                )
                ops = DenseIncidenceOps.from_coo(coo, dtype=h_x.dtype)
            else:
                ops = DenseIncidenceOps(incidence, dtype=h_x.dtype)
            if self.operator_normalization == "degree" and (
                node_degree_inv is None or edge_degree_inv is None
            ):
                if incidence is None:
                    raise ValueError(
                        "degree-normalized dense backend without cached degrees "
                        "requires the dense incidence tensor"
                    )
                node_degree_inv, edge_degree_inv = sp.compute_incidence_degrees(incidence)
        else:
            coo = self._resolve_incidence(
                incidence=incidence,
                incidence_node=incidence_node,
                incidence_edge=incidence_edge,
                incidence_nnz=incidence_nnz,
                num_edges=num_edges,
                num_nodes=num_nodes,
            )
            ops = SparseIncidenceOps(coo)
            if self.operator_normalization == "degree" and (
                node_degree_inv is None or edge_degree_inv is None
            ):
                if incidence is None:
                    raise ValueError(
                        "degree-normalized operators require cached "
                        "node_degree_inv and edge_degree_inv"
                    )
                node_degree_inv, edge_degree_inv = sp.compute_incidence_degrees(incidence)
        if self.operator_normalization == "degree" and mask is not None:
            node_degree_inv = node_degree_inv * mask.to(dtype=node_degree_inv.dtype)
        transformer_like = self.residual_structure == "transformer-like"
        self._record_probe("encode", h_x, h_e, mask)
        f_kw = dict(
            x_raw=x,
            ops=ops,
            node_degree_inv=node_degree_inv,
            edge_degree_inv=edge_degree_inv,
        )
        for layer_idx in range(self.depth):
            h_x = (
                h_x
                + self._f_block(self._norm(h_x), layer_idx, (1, 1), **f_kw)
                + self._f_block(self._norm(h_e), layer_idx, (2, 1), **f_kw)
            )
            if transformer_like:
                h_x = h_x + self._mlp_block(
                    self._norm(h_x),
                    self.mlp_x_w1[layer_idx],
                    self.mlp_x_w2[layer_idx],
                ) * self._mlp_residual_scale
            h_e = (
                h_e
                + self._f_block(self._norm(h_x), layer_idx, (1, 2), **f_kw)
                + self._f_block(self._norm(h_e), layer_idx, (2, 2), **f_kw)
            )
            if transformer_like:
                h_e = h_e + self._mlp_block(
                    self._norm(h_e),
                    self.mlp_e_w1[layer_idx],
                    self.mlp_e_w2[layer_idx],
                ) * self._mlp_residual_scale
            self._record_probe(f"layer{layer_idx}", h_x, h_e, mask)
        return h_x, h_e

    @staticmethod
    def _mask_mean(tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return tokens.mean(dim=1)
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _graph_logits(
        self,
        h_x: torch.Tensor,
        h_e: torch.Tensor,
        mask: torch.Tensor | None,
        edge_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        z_x = self.decoder_x(h_x) * self._readout_scale
        pooled_x = self._mask_mean(z_x, mask)
        if getattr(self, "x_only_graph_readout", False):
            return pooled_x + (self.decoder_e.weight * 0).sum() + (h_e * 0).sum()
        z_e = self.decoder_e(h_e) * self._readout_scale
        pooled_e = self._mask_mean(z_e, edge_mask)
        return (pooled_x + pooled_e) * (0.5**0.5)

    def _hidden_kwargs(self, kwargs: dict) -> dict:
        return {
            key: kwargs[key]
            for key in (
                "incidence_node",
                "incidence_edge",
                "incidence_nnz",
                "node_degree_inv",
                "edge_degree_inv",
                "mask",
                "z",
            )
            if key in kwargs
        }

    def forward(self, x, edge_x, incidence=None, **kwargs):  # type: ignore[no-untyped-def]
        if getattr(self, "output_attention_pool", None) is not None:
            if getattr(self, "operator_mode", None) == "pe-basis":
                raise ValueError("output attention pooling is not implemented for pe-basis")
            h_x, h_e = self._forward_hidden(
                x, edge_x, incidence, **self._hidden_kwargs(kwargs)
            )
            logits = self.output_attention_pool(
                h_x, h_e, kwargs.get("mask"), kwargs.get("edge_mask")
            )
            unused = (self.decoder_x.weight * 0).sum() + (self.decoder_e.weight * 0).sum()
            return logits + unused
        if getattr(self, "x_only_graph_readout", None) is not None:
            if getattr(self, "operator_mode", None) == "pe-basis":
                raise ValueError("x_only_graph_readout is not implemented for pe-basis")
            h_x, h_e = self._forward_hidden(
                x, edge_x, incidence, **self._hidden_kwargs(kwargs)
            )
            return self._graph_logits(
                h_x, h_e, kwargs.get("mask"), kwargs.get("edge_mask")
            )
        if not getattr(self, "x_only_node_readout", False):
            return super().forward(x, edge_x, incidence, **kwargs)
        if getattr(self, "operator_mode", None) == "pe-basis":
            raise ValueError("x_only_node_readout is not implemented for pe-basis")
        h_x, h_e = self._forward_hidden(x, edge_x, incidence, **kwargs)
        z_x = self.decoder_x(h_x) * self._readout_scale
        if self.readout_mode == "node+edge":
            return z_x, self.decoder_e(h_e) * self._readout_scale
        return z_x + (self.decoder_e.weight * 0).sum()


__all__ = [n for n in globals() if not n.startswith("_")]
