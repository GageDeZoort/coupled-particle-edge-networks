"""CAPEN-Llama-att: class-token MHSA readout + hyperedge-only edge–edge attention.

Extends :class:`~cpen.models.capen_llama.CAPENLlama` without changing the default
CAPEN-Llama behaviour.

* **Readout** — two learned class tokens (QCD, top/jets) query a multi-head
  attention over ``KV = [h_x; h_e]``. Logits are
  ``1/(D·σ) · W · MHSA`` with the same μP readout scale as CAPEN
  (``σ = D^{-1/2}`` ⇒ ``1/√D``).
* **M₂₂** — edge–edge attention runs only among hyperedges (incidence degree
  ``> 2``). Pairwise kNN / vn_link slots keep identity ``f_22``. Implemented as
  gather→attend→scatter so we never materialize dense ``M×M`` attention scores
  over padded edge slots (critical for hierarchical caches with ``M∼10³``).
  ``--hyperedge-only`` (``ignore_knn_edges``) additionally drops those pairwise
  rows from the edge stream before ``M_{12}/M_{21}``.
"""

from __future__ import annotations

import torch
from torch import nn

from cpen.models.capen import SupportAttention, supports_without_m22
from cpen.models.capen_llama import CAPENLlama


class CAPENLlamaAtt(CAPENLlama):
    """CAPEN-Llama trunk with class-token MHSA graph readout."""

    def __init__(
        self,
        *args,
        n_class_tokens: int = 2,
        hyperedge_m22_only: bool = True,
        ignore_knn_edges: bool = False,
        **kwargs,
    ):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        if int(n_class_tokens) < 1:
            raise ValueError(f"n_class_tokens must be >= 1; got {n_class_tokens}")
        self.n_class_tokens = int(n_class_tokens)
        self.hyperedge_m22_only = bool(hyperedge_m22_only)
        self.ignore_knn_edges = bool(ignore_knn_edges)
        d = int(self.D)
        scale = d**-0.5
        self.class_tokens = nn.Parameter(torch.randn(self.n_class_tokens, d) * scale)
        self.readout_attn = SupportAttention(
            d,
            heads=int(self.heads),
            dropout=float(getattr(self, "dropout", 0.0) or 0.0),
        )
        # Collapse each class token to one logit; μP scale applied in forward.
        self.class_decoder = nn.Linear(d, 1, bias=False)
        self._pending_edge_type: torch.Tensor | None = None
        self._pending_edge_mask: torch.Tensor | None = None
        self._pending_is_hyper: torch.Tensor | None = None

    def _build_supports(
        self,
        s_bool: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if bool(getattr(self, "identity_m22", False)) or bool(
            getattr(self, "incidence_m22", False)
        ):
            self._pending_is_hyper = None
            return supports_without_m22(s_bool)
        m_11, m_21, m_12, m_22 = super()._build_supports(s_bool, mask=mask)
        if not self.hyperedge_m22_only:
            self._pending_is_hyper = None
            return m_11, m_21, m_12, m_22

        # Degree > 2 ⇒ hyperedge (DBSCAN / virtual hyper); 2-edges stay identity-only.
        deg = s_bool.to(dtype=torch.float32).sum(dim=-1)
        is_hyper = deg > 2
        edge_mask = self._pending_edge_mask
        if edge_mask is not None:
            is_hyper = is_hyper & edge_mask.to(device=is_hyper.device, dtype=torch.bool)
        self._pending_is_hyper = is_hyper
        # Cheap placeholder: identity only. Real M22 runs on the hyper subset in
        # ``_run_attn`` (avoids allocating bool/float ``M×M`` attention).
        eye = torch.eye(m_22.size(-1), device=m_22.device, dtype=torch.bool)
        m_22 = eye.unsqueeze(0).expand(m_22.size(0), -1, -1).contiguous()
        return m_11, m_21, m_12, m_22

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
        if (
            relation == "22"
            and self.hyperedge_m22_only
            and not bool(getattr(self, "identity_m22", False))
            and not bool(getattr(self, "incidence_m22", False))
        ):
            return self._run_attn_22_hyper_subset(
                module,
                target,
                source,
                probe=probe,
            )
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

    def _run_attn_22_hyper_subset(
        self,
        module: nn.Module,
        target: torch.Tensor,
        source: torch.Tensor,
        *,
        probe: dict | None = None,
    ) -> torch.Tensor:
        """``f_22`` among hyperedges only; identity on pairwise / pad slots."""
        is_hyper = self._pending_is_hyper
        if is_hyper is None:
            unused = sum((p * 0.0).sum() for p in module.parameters())
            return target + unused

        is_hyper = is_hyper.to(device=target.device, dtype=torch.bool)
        batch, n_edges, width = target.shape
        counts = is_hyper.sum(dim=-1)
        max_h = int(counts.max().item()) if batch > 0 else 0
        if max_h <= 0:
            unused = sum((p * 0.0).sum() for p in module.parameters())
            return target + unused

        # Padded gather indices (B, H_max); invalid slots reuse index 0 but are masked.
        idx = target.new_zeros(batch, max_h, dtype=torch.long)
        hyp_ok = torch.zeros(batch, max_h, dtype=torch.bool, device=target.device)
        for b in range(batch):
            ix = is_hyper[b].nonzero(as_tuple=False).flatten()
            n = int(ix.numel())
            if n == 0:
                continue
            idx[b, :n] = ix
            hyp_ok[b, :n] = True

        gather_idx = idx.unsqueeze(-1).expand(batch, max_h, width)
        tgt_h = torch.gather(target, 1, gather_idx)
        src_h = torch.gather(source, 1, gather_idx)
        support_h = hyp_ok.unsqueeze(-1) & hyp_ok.unsqueeze(-2)

        # Bypass this override's hyper path via super() → CAPEN wrapper → bytecode.
        out_h = super()._run_attn(
            module,
            tgt_h,
            src_h,
            support_h,
            relation="22",
            probe=probe,
        )

        out = target.clone()
        for b in range(batch):
            n = int(counts[b].item())
            if n > 0:
                out[b, idx[b, :n]] = out_h[b, :n]
        return out

    def _class_token_readout(
        self,
        h_x: torch.Tensor,
        h_e: torch.Tensor,
        *,
        node_mask: torch.Tensor | None,
        edge_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """``logits = 1/(D σ) · Linear(MHSA(Q=G, KV=[X;E]))`` → ``(B, Nc)``."""
        batch = h_x.size(0)
        n_nodes = h_x.size(1)
        n_edges = h_e.size(1)
        tokens = self.class_tokens.unsqueeze(0).expand(batch, -1, -1)
        kv = torch.cat([h_x, h_e], dim=1)

        if node_mask is None:
            node_ok = torch.ones(batch, n_nodes, device=h_x.device, dtype=torch.bool)
        else:
            node_ok = node_mask.to(device=h_x.device, dtype=torch.bool)
        if edge_mask is None:
            edge_ok = torch.ones(batch, n_edges, device=h_e.device, dtype=torch.bool)
        else:
            edge_ok = edge_mask.to(device=h_e.device, dtype=torch.bool)
        # Prefer hyperedge-only edge KV when the mask is available (matches M22 policy).
        if self._pending_is_hyper is not None:
            edge_ok = edge_ok & self._pending_is_hyper.to(
                device=edge_ok.device, dtype=torch.bool
            )
        kv_ok = torch.cat([node_ok, edge_ok], dim=-1)
        support = kv_ok.unsqueeze(1).expand(batch, self.n_class_tokens, -1)

        q = self.ln(tokens)
        k = self.ln(kv)
        pooled = self.readout_attn(q, k, support)
        logits = self.class_decoder(pooled).squeeze(-1) * self._readout_scale
        # Parent energy-pool heads are unused here; keep them in the DDP graph
        # (find_unused_parameters=False). Same pattern as identity f_22.
        unused = (self.decoder_x.weight * 0).sum() + (self.decoder_e.weight * 0).sum()
        return logits + unused

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
        z: torch.Tensor | None = None,  # noqa: ARG002 — unused; class-token readout
        wire_coordinates: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,  # noqa: ARG002 — optional; degree drives M22
        edge_mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"Expected x shape (batch, n_particles, n_features); got {tuple(x.shape)}"
            )
        if edge_x.dim() != 3:
            raise ValueError(
                f"CAPEN expects sparse edge features (batch, n_edges, m_0); got {tuple(edge_x.shape)}"
            )

        node_mask_eff = node_mask if node_mask is not None else mask
        self._pending_edge_type = edge_type
        self._pending_edge_mask = edge_mask
        self._pending_is_hyper = None
        try:
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
            if bool(getattr(self, "incidence_m22", False)):
                self._pending_s = s_bool.to(dtype=torch.float32)
            m_11, m_21, m_12, m_22 = self._build_supports(s_bool, mask=node_mask_eff)

            wire_coords = None
            if bool(getattr(self, "use_wire", False)):
                wire_coords = self._particle_wire_coordinates(
                    s_bool,
                    node_mask_eff,
                    wire_coordinates=wire_coordinates,
                )

            h_x = self.encoder_x(x) * self._encoder_x_scale
            h_e = self.encoder_e(edge_x) * self._encoder_e_scale
            self._record_probe("encode", h_x, h_e, node_mask_eff)

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
                attn_s = self._attn_residual_scale
                h_x = h_x + attn_s * (f_11 + f_21)
                h_e = h_e + attn_s * (f_22 + f_12)
                mlp_s = self._mlp_residual_scale
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
                self._record_probe(f"layer{layer_idx}", h_x, h_e, node_mask_eff)

            return self._class_token_readout(
                h_x,
                h_e,
                node_mask=node_mask_eff,
                edge_mask=edge_mask,
            )
        finally:
            self._pending_edge_type = None
            self._pending_edge_mask = None
            self._pending_is_hyper = None
            self._pending_s = None


__all__ = ["CAPENLlamaAtt"]
