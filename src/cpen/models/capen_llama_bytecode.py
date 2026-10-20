"""Reconstructed CAPEN-Llama backend (replaces lost ``capen_llama.pyc``).

SwiGLU residuals + optional all-to-all particle attention + WIRE hooks.
Subclasses the reconstructed :class:`cpen.models.capen.CAPEN`.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

# Capen wrapper must be importable (registers reconstructed CAPEN).
from cpen.models.capen import CAPEN, AttentionGammas, SupportAttention  # noqa: F401


class CAPENLlama(CAPEN):
    """CAPEN trunk with SwiGLU MLPs and optional all-to-all M11 / WIRE."""

    def __init__(
        self,
        n_features: int,
        n_edge_features: int,
        out_dim: int,
        depth: int,
        width: int,
        heads: int = 4,
        *,
        dropout: float = 0.0,
        normalization: str = "uniform",
        energy_index: int = 0,
        optimizer: str = "adam",
        readout_mode: str = "graph",
        attention_normalization: str = "none",
        attention_gammas: AttentionGammas | None = None,
        use_wire: bool = False,
        wire_coordinate_dim: int = 8,
        wire_share_across_heads: bool = False,
        wire_frequency_init_std: float | None = None,
        wire_sign_augmentation: bool = False,
        wire_normalized_laplacian: bool = True,
        all_to_all_particle_attention: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            n_features=n_features,
            n_edge_features=n_edge_features,
            out_dim=out_dim,
            depth=depth,
            width=width,
            heads=heads,
            dropout=dropout,
            normalization=normalization,
            energy_index=energy_index,
            optimizer=optimizer,
            readout_mode=readout_mode,
            attention_normalization=attention_normalization,
            attention_gammas=attention_gammas,
            use_wire=False,  # set below after modules exist
            wire_coordinate_dim=wire_coordinate_dim,
            wire_share_across_heads=wire_share_across_heads,
            wire_frequency_init_std=wire_frequency_init_std,
            wire_sign_augmentation=wire_sign_augmentation,
            wire_normalized_laplacian=wire_normalized_laplacian,
            all_to_all_particle_attention=all_to_all_particle_attention,
            **kwargs,
        )
        # Replace GELU MLP stacks with SwiGLU (gate / up / down).
        del self.mlp_x_w1, self.mlp_x_w2, self.mlp_e_w1, self.mlp_e_w2
        self.mlp_x_gate = nn.ModuleList(
            [nn.Linear(self.D, 4 * self.D, bias=False) for _ in range(self.depth)]
        )
        self.mlp_x_up = nn.ModuleList(
            [nn.Linear(self.D, 4 * self.D, bias=False) for _ in range(self.depth)]
        )
        self.mlp_x_down = nn.ModuleList(
            [nn.Linear(4 * self.D, self.D, bias=False) for _ in range(self.depth)]
        )
        self.mlp_e_gate = nn.ModuleList(
            [nn.Linear(self.D, 4 * self.D, bias=False) for _ in range(self.depth)]
        )
        self.mlp_e_up = nn.ModuleList(
            [nn.Linear(self.D, 4 * self.D, bias=False) for _ in range(self.depth)]
        )
        self.mlp_e_down = nn.ModuleList(
            [nn.Linear(4 * self.D, self.D, bias=False) for _ in range(self.depth)]
        )
        for mod in (
            self.mlp_x_gate,
            self.mlp_x_up,
            self.mlp_x_down,
            self.mlp_e_gate,
            self.mlp_e_up,
            self.mlp_e_down,
        ):
            for lin in mod:
                nn.init.normal_(lin.weight, mean=0.0, std=1.0)

        self.use_wire = bool(use_wire)
        self.wire_11 = None
        if self.use_wire:
            if self.head_dim % 2 != 0:
                raise ValueError(
                    f"WIRE requires even head_dim; got head_dim={self.head_dim}"
                )
            from cpen.models.wire import WIRERotaryEmbedding

            self.wire_11 = nn.ModuleList(
                [
                    WIRERotaryEmbedding(
                        self.head_dim,
                        self.heads,
                        coordinate_dim=int(wire_coordinate_dim),
                        share_across_heads=bool(wire_share_across_heads),
                        frequency_init_std=wire_frequency_init_std,
                    )
                    for _ in range(self.depth)
                ]
            )

    def _build_supports(
        self,
        s_bool: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        m_11, m_21, m_12, m_22 = CAPEN._build_supports(s_bool, mask=mask)
        if not self.all_to_all_particle_attention:
            return m_11, m_21, m_12, m_22
        batch, _, n = s_bool.shape
        if mask is None:
            node_ok = torch.ones(batch, n, device=s_bool.device, dtype=torch.bool)
        else:
            node_ok = mask.to(device=s_bool.device, dtype=torch.bool)
        m_11 = node_ok.unsqueeze(1) & node_ok.unsqueeze(2)
        return m_11, m_21, m_12, m_22

    def _mlp(  # type: ignore[override]
        self,
        h: torch.Tensor,
        gate: nn.Module,
        up: nn.Module | None = None,
        down: nn.Module | None = None,
    ) -> torch.Tensor:
        """SwiGLU: down(silu(gate(LN h)) * up(LN h)).

        Also accepts legacy ``(h, w1, w2)`` if ``up is None`` (should not happen
        on Llama).
        """
        if up is None or down is None:
            # Fallback GELU path (should not be used on Llama).
            assert isinstance(gate, nn.Module)
            return super()._mlp(h, gate, up)  # type: ignore[arg-type]
        z = self.ln(h)
        return down(
            torch.nn.functional.silu(gate(z) * self._mlp_w1_scale)
            * (up(z) * self._mlp_w1_scale)
        ) * self._mlp_w2_scale

    def _particle_wire_coordinates(
        self,
        s_bool: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        wire_coordinates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if wire_coordinates is not None:
            return wire_coordinates
        from cpen.utils.wire_coordinates import (
            compute_wire_coordinates_batched,
            particle_adjacency_from_incidence,
        )

        node_mask = (
            mask
            if mask is not None
            else s_bool.new_ones(s_bool.shape[0], s_bool.shape[-1], dtype=torch.bool)
        )
        adj = particle_adjacency_from_incidence(s_bool, mask=node_mask)
        return compute_wire_coordinates_batched(
            adj,
            node_mask,
            num_frequencies=int(self.wire_coordinate_dim),
            normalized_laplacian=bool(self.wire_normalized_laplacian),
            canonicalize_sign=True,
        )

    def forward(
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
        z: torch.Tensor | None = None,
        wire_coordinates: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x (B,N,F); got {tuple(x.shape)}")
        if edge_x.dim() != 3:
            raise ValueError(f"Expected edge_x (B,M,F); got {tuple(edge_x.shape)}")
        num_edges = edge_x.size(1)
        num_nodes = x.size(1)
        s_bool = self._resolve_dense_incidence(
            incidence=incidence,
            incidence_node=incidence_node,
            incidence_edge=incidence_edge,
            incidence_nnz=incidence_nnz,
            num_edges=num_edges,
            num_nodes=num_nodes,
            device=x.device,
        )
        m_11, m_21, m_12, m_22 = self._build_supports(s_bool, mask=mask)
        wire_coords = None
        if self.use_wire:
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
            rotary = self.wire_11[layer_idx] if self.wire_11 is not None else None
            f_11 = self._run_attn(
                self.attn_11[layer_idx],
                x_ln,
                x_ln,
                m_11,
                relation="11",
                probe=self._attn_probe_entry(layer_idx, "11"),
                target_coordinates=wire_coords,
                source_coordinates=wire_coords,
                wire=rotary,
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

        z_x = self.decoder_x(h_x) * self._readout_scale
        z_e = self.decoder_e(h_e) * self._readout_scale
        if self.readout_mode == "node":
            s_f = s_bool.to(dtype=z_e.dtype)
            z_from_e = torch.bmm(s_f.transpose(1, 2), z_e)
            out = (z_x + z_from_e) * (0.5**0.5)
            if mask is not None:
                out = out * mask.to(dtype=out.dtype).unsqueeze(-1)
            return out
        if z is not None and self.normalization == "energy-weights":
            w = z.to(dtype=z_x.dtype)
            if mask is not None:
                w = w * mask.to(dtype=w.dtype)
            pooled_x = (z_x * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1).clamp_min(1e-12).unsqueeze(-1)
        else:
            pooled_x = self._mask_mean(z_x, mask)
        if edge_mask is None:
            edge_mask = s_bool.any(dim=-1)
        pooled_e = self._mask_mean(z_e, edge_mask)
        return (pooled_x + pooled_e) * (0.5**0.5)
