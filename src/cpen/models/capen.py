"""Coupled attentional particle-edge network (CAPEN).

The core implementation currently lives in ``capen.pyc`` (source was lost in a
restore). This module loads that bytecode and extends ``CAPEN`` with an
optional identity edge–edge path:

    ``identity_m22=True`` → skip ``SupportAttention`` for relation ``22`` and
    set ``f_22 = target`` (the LayerNorm'd edge states).

    ``incidence_m22=True`` → ``f_22 = S S^T h_e`` via two incidence matmuls
    (node gather then edge scatter). Same residual layout, no ``(B, M, M)``
    attention. Mutually exclusive with ``identity_m22``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn

_PYC = Path(__file__).resolve().with_name("capen.pyc")
if not _PYC.is_file():
    raise ImportError(f"Missing bytecode backend {_PYC}")

_spec = importlib.util.spec_from_file_location("cpen.models._capen_bc", _PYC)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load {_PYC}")
_bc = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("cpen.models._capen_bc", _bc)
_spec.loader.exec_module(_bc)

for _name, _value in vars(_bc).items():
    if _name.startswith("__"):
        continue
    globals()[_name] = _value

_BaseCAPEN = _bc.CAPEN


def supports_without_m22(
    s_bool: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``m_11, m_21, m_12`` as in bytecode ``_build_supports``, no ``(B, M, M)`` ``m_22``.

    Bytecode ``_build_supports`` is a staticmethod (first arg is ``s_bool``, not
    ``self``). ``identity_m22`` / ``incidence_m22`` never read the line-graph
    attention mask, so skip it.
    """
    s_float = s_bool.to(torch.float32)
    m_11 = torch.bmm(s_float.transpose(1, 2), s_float) > 0.0
    n_particles = s_bool.size(-1)
    eye_x = torch.eye(n_particles, device=s_bool.device, dtype=torch.bool)
    m_11 = m_11 | eye_x
    m_12 = s_bool
    m_21 = s_bool.transpose(1, 2)
    m_22 = s_bool.new_zeros(0)
    return m_11, m_21, m_12, m_22


def incidence_ss_t(s: torch.Tensor, h_e: torch.Tensor) -> torch.Tensor:
    """``(S S^T) h_e`` without forming ``S S^T``: ``S (S^T h_e)``.

    ``s`` is ``(B, M, N)`` incidence (bool or float); ``h_e`` is ``(B, M, D)``.
    """
    s_f = s.to(dtype=h_e.dtype)
    return torch.bmm(s_f, torch.bmm(s_f.transpose(1, 2), h_e))


class CAPEN(_BaseCAPEN):
    """Bytecode CAPEN + optional linear / identity ``f_22`` (skip edge–edge attention)."""

    def __init__(
        self,
        *args,
        identity_m22: bool = False,
        incidence_m22: bool = False,
        **kwargs,
    ):  # type: ignore[no-untyped-def]
        requested = str(kwargs.get("readout_mode", "graph") or "graph")
        if requested == "node+edge":
            # Bytecode only accepts graph/node; restore after super().__init__.
            kwargs = dict(kwargs)
            kwargs["readout_mode"] = "node"
        super().__init__(*args, **kwargs)
        if requested == "node+edge":
            self.readout_mode = "node+edge"
        self.identity_m22 = bool(identity_m22)
        self.incidence_m22 = bool(incidence_m22)
        if self.identity_m22 and self.incidence_m22:
            raise ValueError("identity_m22 and incidence_m22 are mutually exclusive")
        self._pending_s: torch.Tensor | None = None
        # Pascal study: decode X with W_X only (skip stock (z_X + S^T z_E)/√2).
        self.x_only_node_readout = False

    def _skip_dense_m22(self) -> bool:
        return bool(self.identity_m22 or self.incidence_m22)

    def _token_readout(
        self, h_x: torch.Tensor, h_e: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z_x = self.decoder_x(h_x) * self._readout_scale
        if self.readout_mode == "node+edge":
            return z_x, self.decoder_e(h_e) * self._readout_scale
        # Node-only: last-layer f_12 / f_22 / mlp_e write h_e after h_x is
        # already updated, so they would be DDP-unused without this dummy.
        return z_x + (self.decoder_e.weight * 0).sum() + (h_e * 0).sum()

    def _forward_hidden(  # type: ignore[no-untyped-def]
        self,
        x: torch.Tensor,
        edge_x: torch.Tensor,
        incidence: torch.Tensor | None = None,
        *,
        incidence_node: torch.Tensor | None = None,
        incidence_edge: torch.Tensor | None = None,
        incidence_nnz: torch.Tensor | None = None,
        node_degree_inv: torch.Tensor | None = None,  # noqa: ARG002
        edge_degree_inv: torch.Tensor | None = None,  # noqa: ARG002
        mask: torch.Tensor | None = None,
        z: torch.Tensor | None = None,  # noqa: ARG002
        wire_coordinates: torch.Tensor | None = None,  # noqa: ARG002
        **_ignored: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode + residual layers (same order as bytecode ``CAPEN.forward``)."""
        if x.dim() != 3:
            raise ValueError(
                f"Expected x shape (batch, n_particles, n_features); got {tuple(x.shape)}"
            )
        if edge_x.dim() != 3:
            raise ValueError(
                f"CAPEN expects sparse edge features (batch, n_edges, m_0); got {tuple(edge_x.shape)}"
            )
        num_edges = edge_x.size(-2)
        num_nodes = x.size(-2)
        s_bool = self._resolve_dense_incidence(
            incidence=incidence,
            incidence_node=incidence_node,
            incidence_edge=incidence_edge,
            incidence_nnz=incidence_nnz,
            num_edges=num_edges,
            num_nodes=num_nodes,
            device=x.device,
        )
        if self.incidence_m22:
            self._pending_s = s_bool.to(dtype=torch.float32)
        m_11, m_21, m_12, m_22 = (
            supports_without_m22(s_bool)
            if self._skip_dense_m22()
            else self._build_supports(s_bool, mask=mask)
        )
        h_x = self.encoder_x(x) * self._encoder_x_scale
        h_e = self.encoder_e(edge_x) * self._encoder_e_scale
        self._record_probe("encode", h_x, h_e, mask)
        attn_s = self._attn_residual_scale
        mlp_s = self._mlp_residual_scale
        for layer_idx in range(self.depth):
            x_ln = self.ln(h_x)
            e_ln = self.ln(h_e)
            f_11 = self._run_attn(
                self.attn_11[layer_idx], x_ln, x_ln, m_11, relation="11",
                probe=self._attn_probe_entry(layer_idx, "11"),
            )
            f_21 = self._run_attn(
                self.attn_21[layer_idx], x_ln, e_ln, m_21, relation="21",
                probe=self._attn_probe_entry(layer_idx, "21"),
            )
            f_22 = self._run_attn(
                self.attn_22[layer_idx], e_ln, e_ln, m_22, relation="22",
                probe=self._attn_probe_entry(layer_idx, "22"),
            )
            f_12 = self._run_attn(
                self.attn_12[layer_idx], e_ln, x_ln, m_12, relation="12",
                probe=self._attn_probe_entry(layer_idx, "12"),
            )
            h_x = h_x + attn_s * (f_11 + f_21)
            h_e = h_e + attn_s * (f_22 + f_12)
            h_x = h_x + mlp_s * self._mlp(
                h_x, self.mlp_x_w1[layer_idx], self.mlp_x_w2[layer_idx]
            )
            h_e = h_e + mlp_s * self._mlp(
                h_e, self.mlp_e_w1[layer_idx], self.mlp_e_w2[layer_idx]
            )
            self._record_probe(f"layer{layer_idx}", h_x, h_e, mask)
        return h_x, h_e

    def forward(self, x, edge_x, incidence=None, **kwargs):  # type: ignore[no-untyped-def]
        token_wise = bool(getattr(self, "x_only_node_readout", False)) or (
            getattr(self, "readout_mode", None) == "node+edge"
        )
        try:
            if token_wise:
                h_x, h_e = self._forward_hidden(x, edge_x, incidence, **kwargs)
                return self._token_readout(h_x, h_e)
            if self.incidence_m22:
                num_edges = edge_x.size(-2)
                num_nodes = x.size(-2)
                s_bool = self._resolve_dense_incidence(
                    incidence=incidence,
                    incidence_node=kwargs.get("incidence_node"),
                    incidence_edge=kwargs.get("incidence_edge"),
                    incidence_nnz=kwargs.get("incidence_nnz"),
                    num_edges=num_edges,
                    num_nodes=num_nodes,
                    device=x.device,
                )
                self._pending_s = s_bool.to(dtype=torch.float32)
            return super().forward(x, edge_x, incidence, **kwargs)
        finally:
            self._pending_s = None

    def _run_attn(
        self,
        module: nn.Module,
        target: torch.Tensor,
        source: torch.Tensor,
        support: torch.Tensor,
        *,
        relation: str,
        probe: dict | None = None,
        target_coordinates: torch.Tensor | None = None,
        source_coordinates: torch.Tensor | None = None,
        wire: nn.Module | None = None,
    ) -> torch.Tensor:
        if self.identity_m22 and relation == "22":
            unused = sum((p * 0.0).sum() for p in module.parameters())
            return target + unused
        if self.incidence_m22 and relation == "22":
            unused = sum((p * 0.0).sum() for p in module.parameters())
            if self._pending_s is None:
                raise RuntimeError("incidence_m22 requires stashed incidence S")
            return incidence_ss_t(self._pending_s, target) + unused
        return super()._run_attn(
            module,
            target,
            source,
            support,
            relation=relation,
            probe=probe,
            target_coordinates=target_coordinates,
            source_coordinates=source_coordinates,
            wire=wire,
        )


__all__ = [n for n in globals() if not n.startswith("_")]
