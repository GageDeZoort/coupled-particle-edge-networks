"""CAPEN-Llama (WIRE / SwiGLU / optional all-to-all particle attention).

Loads ``capen_llama.pyc`` after the ``capen.py`` wrapper so ``CAPENLlama``
subclasses the identity / incidence-``m22``-aware ``CAPEN``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

# Ensure wrapped CAPEN is registered before the Llama bytecode imports it.
from cpen.models.capen import CAPEN as _  # noqa: F401
from cpen.models.capen import supports_without_m22

_PYC = Path(__file__).resolve().with_name("capen_llama.pyc")
if not _PYC.is_file():
    raise ImportError(f"Missing bytecode backend {_PYC}")

_spec = importlib.util.spec_from_file_location("cpen.models._capen_llama_bc", _PYC)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load {_PYC}")
_bc = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("cpen.models._capen_llama_bc", _bc)
_spec.loader.exec_module(_bc)

for _name, _value in vars(_bc).items():
    if _name.startswith("__"):
        continue
    globals()[_name] = _value

_BaseLlama = globals()["CAPENLlama"]


class CAPENLlama(_BaseLlama):
    """Bytecode CAPEN-Llama + Pascal token-wise node / node+edge readout."""

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
        wire_coordinates: torch.Tensor | None = None,
        **_ignored: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """SwiGLU / WIRE trunk; same residual order as bytecode ``forward``."""
        if x.dim() != 3:
            raise ValueError(
                f"Expected x shape (batch, n_particles, n_features); got {tuple(x.shape)}"
            )
        if edge_x.dim() != 3:
            raise ValueError(
                f"CAPEN-Llama expects sparse edge features (batch, n_edges, m_0); "
                f"got {tuple(edge_x.shape)}"
            )
        num_nodes = x.size(1)
        num_edges = edge_x.size(1)
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
        wire_coords = None
        if bool(getattr(self, "use_wire", False)):
            wire_coords = self._particle_wire_coordinates(
                s_bool, mask, wire_coordinates=wire_coordinates
            )
        h_x = self.encoder_x(x) * self._encoder_x_scale
        h_e = self.encoder_e(edge_x) * self._encoder_e_scale
        self._record_probe("encode", h_x, h_e, mask)
        attn_s = self._attn_residual_scale
        mlp_s = self._mlp_residual_scale
        for layer_idx in range(self.depth):
            x_ln = self.ln(h_x)
            e_ln = self.ln(h_e)
            wire = None
            if wire_coords is not None and hasattr(self, "wire_11"):
                wire = self.wire_11[layer_idx]
            f_11 = self._run_attn(
                self.attn_11[layer_idx],
                x_ln,
                x_ln,
                m_11,
                relation="11",
                probe=self._attn_probe_entry(layer_idx, "11"),
                target_coordinates=wire_coords,
                source_coordinates=wire_coords,
                wire=wire,
            )
            f_21 = self._run_attn(
                self.attn_21[layer_idx],
                x_ln,
                e_ln,
                m_21,
                relation="21",
                probe=self._attn_probe_entry(layer_idx, "21"),
            )
            f_22 = self._run_attn(
                self.attn_22[layer_idx],
                e_ln,
                e_ln,
                m_22,
                relation="22",
                probe=self._attn_probe_entry(layer_idx, "22"),
            )
            f_12 = self._run_attn(
                self.attn_12[layer_idx],
                e_ln,
                x_ln,
                m_12,
                relation="12",
                probe=self._attn_probe_entry(layer_idx, "12"),
            )
            h_x = h_x + attn_s * (f_11 + f_21)
            h_e = h_e + attn_s * (f_22 + f_12)
            h_x = h_x + mlp_s * self._mlp(
                h_x,
                self.mlp_x_gate[layer_idx],
                self.mlp_x_up[layer_idx],
                self.mlp_x_down[layer_idx],
            )
            h_e = h_e + mlp_s * self._mlp(
                h_e,
                self.mlp_e_gate[layer_idx],
                self.mlp_e_up[layer_idx],
                self.mlp_e_down[layer_idx],
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
            if bool(getattr(self, "incidence_m22", False)):
                num_edges = edge_x.size(1)
                num_nodes = x.size(1)
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


__all__ = [n for n in globals() if not n.startswith("_")]
