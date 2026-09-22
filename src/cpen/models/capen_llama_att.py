"""CAPEN-Llama-att: writeup class-token pooling + hyperedge-only edge–edge attention.

Extends :class:`~cpen.models.capen_llama.CAPENLlama` without changing the default
CAPEN-Llama behaviour.

* **Readout** — :class:`~cpen.models.output_attention_pool.OutputAttentionPool`:
  class tokens pool \(X^{(L)}\) and \(E^{(L)}\) independently (full-width
  \(\alpha_A=1\) attention), fuse \(\frac{1}{\sqrt{2}}(P_1+P_2)\), then
  per-class unembed \(z_a=\frac{1}{D\sigma_U}P_a\cdot w_a\).
* **M₂₂** — edge–edge attention runs only among hyperedges (incidence degree
  ``> 2``). Pairwise kNN / vn_link slots keep identity ``f_22``. Implemented as
  gather→attend→scatter so we never materialize dense ``M×M`` attention scores
  over padded edge slots (critical for hierarchical caches with ``M∼10³``).
  ``--hyperedge-only`` (``ignore_knn_edges``) additionally drops those pairwise
  rows from the edge stream before ``M_{12}/M_{21}``.
* **RoPE** — ``--use-rope`` applies axial :math:`(\\eta,\\phi)` rotary embeddings
  on relation 11 only (jet-centered :math:`(\\Delta\\eta,\\Delta\\phi)` from
  four-vectors). Mutually exclusive with ``--use-wire``.
* **BaselineTransformer** — same Llama MLPs, CompleteP scales, and node
  class-token pool, but ``m11_only``: no 12/21/22, no edge residual, no
  \(\frac{1}{\sqrt{2}}\) fuse.
"""

from __future__ import annotations

import torch
from torch import nn

from cpen.models.capen import supports_without_m22
from cpen.models.capen_llama import CAPENLlama
from cpen.models.output_attention_pool import OutputAttentionPool


class CAPENLlamaAtt(CAPENLlama):
    """CAPEN-Llama trunk with class-token MHSA graph readout."""

    def __init__(
        self,
        *args,
        n_class_tokens: int = 2,
        hyperedge_m22_only: bool = True,
        ignore_knn_edges: bool = False,
        m11_only: bool = False,
        uniform_output_pool: bool = False,
        **kwargs,
    ):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        if int(n_class_tokens) < 1:
            raise ValueError(f"n_class_tokens must be >= 1; got {n_class_tokens}")
        self.n_class_tokens = int(n_class_tokens)
        self.hyperedge_m22_only = bool(hyperedge_m22_only)
        self.ignore_knn_edges = bool(ignore_knn_edges)
        self.m11_only = bool(m11_only)
        self.uniform_output_pool = bool(uniform_output_pool)
        if self.m11_only and bool(getattr(self, "use_wire", False)):
            raise ValueError(
                "m11_only / BaselineTransformer cannot use WIRE (needs a graph Laplacian)"
            )
        if self.uniform_output_pool:
            self.output_pool = None
        else:
            self.output_pool = OutputAttentionPool(
                int(self.D),
                self.n_class_tokens,
                dropout=float(getattr(self, "dropout", 0.0) or 0.0),
                node_only=self.m11_only,
            )
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

    @staticmethod
    def _mask_mean(tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return tokens.mean(dim=1)
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _uniform_output_readout(
        self,
        h_x: torch.Tensor,
        h_e: torch.Tensor | None,
        *,
        node_mask: torch.Tensor | None,
        edge_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mask-uniform mean of decoded X and E, fused ``(z_X+z_E)/√2``."""
        z_x = self.decoder_x(h_x) * self._readout_scale
        pooled_x = self._mask_mean(z_x, node_mask)
        if h_e is None or bool(getattr(self, "m11_only", False)):
            unused = (self.decoder_e.weight * 0).sum()
            if h_e is not None:
                unused = unused + (h_e * 0).sum()
            return pooled_x + unused
        z_e = self.decoder_e(h_e) * self._readout_scale
        pooled_e = self._mask_mean(z_e, edge_mask)
        return (pooled_x + pooled_e) * (0.5**0.5)

    def _class_token_readout(
        self,
        h_x: torch.Tensor,
        h_e: torch.Tensor | None,
        *,
        node_mask: torch.Tensor | None,
        edge_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Class-token pool, or uniform mean pool when ``uniform_output_pool``."""
        if self.uniform_output_pool:
            return self._uniform_output_readout(
                h_x, h_e, node_mask=node_mask, edge_mask=edge_mask
            )
        logits = self.output_pool(h_x, h_e, mask=node_mask, edge_mask=edge_mask)
        unused = (self.decoder_x.weight * 0).sum() + (self.decoder_e.weight * 0).sum()
        return logits + unused

    def _unused_edge_modules(self) -> torch.Tensor:
        """Keep 12/21/22 + edge MLP/encoder in the loss graph for DDP."""
        prefixes = (
            "encoder_e",
            "attn_12",
            "attn_21",
            "attn_22",
            "mlp_e_",
        )
        acc = None
        for name, param in self.named_parameters():
            if any(name == p or name.startswith(p) for p in prefixes):
                term = (param * 0.0).sum()
                acc = term if acc is None else acc + term
        if acc is None:
            return next(self.parameters()).new_zeros(())
        return acc

    def _particle_all_to_all_support(
        self,
        *,
        batch: int,
        n_particles: int,
        mask: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        if mask is None:
            node_ok = torch.ones(batch, n_particles, device=device, dtype=torch.bool)
        else:
            node_ok = mask.to(device=device, dtype=torch.bool)
        return node_ok.unsqueeze(1) & node_ok.unsqueeze(2)

    def _forward_m11_only(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
        rope_coordinates: torch.Tensor | None = None,
        x_raw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Llama-style particle transformer: LN → M11 → SwiGLU, then node pool."""
        if x.dim() != 3:
            raise ValueError(
                f"Expected x shape (batch, n_particles, n_features); got {tuple(x.shape)}"
            )
        node_mask_eff = node_mask if node_mask is not None else mask
        m_11 = self._particle_all_to_all_support(
            batch=int(x.size(0)),
            n_particles=int(x.size(1)),
            mask=node_mask_eff,
            device=x.device,
        )
        rope_coords = None
        if bool(getattr(self, "use_rope", False)):
            rope_coords = self._particle_rope_coordinates(
                rope_coordinates=rope_coordinates,
                x_raw=x_raw,
                mask=node_mask_eff,
            )

        h_x = self.encoder_x(x) * self._encoder_x_scale
        h_e_dummy = h_x.new_zeros(h_x.size(0), 1, h_x.size(-1))
        self._record_probe("encode", h_x, h_e_dummy, node_mask_eff)

        for layer_idx in range(self.depth):
            x_ln = self.ln(h_x)
            rot_coords, rotary = self._relation_11_rotary(
                layer_idx, rope_coords=rope_coords, wire_coords=None
            )
            f_11 = self._run_attn(
                self.attn_11[layer_idx],
                x_ln,
                x_ln,
                m_11,
                relation="11",
                probe=self._attn_probe_entry(layer_idx, "11"),
                target_coordinates=rot_coords,
                source_coordinates=rot_coords,
                wire=rotary,
            )
            attn_s = self._attn_residual_scale
            h_x = h_x + attn_s * f_11
            mlp_s = self._mlp_residual_scale
            h_x = h_x + mlp_s * self._mlp(
                h_x,
                self.mlp_x_gate[layer_idx],
                self.mlp_x_up[layer_idx],
                self.mlp_x_down[layer_idx],
            )
            self._record_probe(f"layer{layer_idx}", h_x, h_e_dummy, node_mask_eff)

        logits = self._class_token_readout(
            h_x,
            h_e_dummy,
            node_mask=node_mask_eff,
            edge_mask=h_e_dummy.new_zeros(h_e_dummy.shape[:2], dtype=torch.bool),
        )
        return logits + self._unused_edge_modules()

    def forward(
        self,
        x: torch.Tensor,
        edge_x: torch.Tensor | None = None,
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
        rope_coordinates: torch.Tensor | None = None,
        x_raw: torch.Tensor | None = None,
        edge_type: torch.Tensor | None = None,  # noqa: ARG002 — optional; degree drives M22
        edge_mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bool(getattr(self, "m11_only", False)):
            return self._forward_m11_only(
                x,
                mask=mask,
                node_mask=node_mask,
                rope_coordinates=rope_coordinates,
                x_raw=x_raw,
            )
        if x.dim() != 3:
            raise ValueError(
                f"Expected x shape (batch, n_particles, n_features); got {tuple(x.shape)}"
            )
        if edge_x is None or edge_x.dim() != 3:
            raise ValueError(
                f"CAPEN expects sparse edge features (batch, n_edges, m_0); got {None if edge_x is None else tuple(edge_x.shape)}"
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
            rope_coords = None
            if bool(getattr(self, "use_rope", False)):
                rope_coords = self._particle_rope_coordinates(
                    rope_coordinates=rope_coordinates,
                    x_raw=x_raw,
                    mask=node_mask_eff,
                )

            h_x = self.encoder_x(x) * self._encoder_x_scale
            h_e = self.encoder_e(edge_x) * self._encoder_e_scale
            self._record_probe("encode", h_x, h_e, node_mask_eff)

            for layer_idx in range(self.depth):
                x_ln = self.ln(h_x)
                e_ln = self.ln(h_e)
                rot_coords, rotary = self._relation_11_rotary(
                    layer_idx, rope_coords=rope_coords, wire_coords=wire_coords
                )
                f_11 = self._run_attn(
                    self.attn_11[layer_idx],
                    x_ln,
                    x_ln,
                    m_11,
                    relation="11",
                    probe=self._attn_probe_entry(layer_idx, "11"),
                    target_coordinates=rot_coords,
                    source_coordinates=rot_coords,
                    wire=rotary,
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


class BaselineTransformer(CAPENLlamaAtt):
    """Edgeless particle transformer: M11 + Llama MLPs + node class-token pool."""

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        kwargs["m11_only"] = True
        super().__init__(*args, **kwargs)


__all__ = ["CAPENLlamaAtt", "BaselineTransformer"]
