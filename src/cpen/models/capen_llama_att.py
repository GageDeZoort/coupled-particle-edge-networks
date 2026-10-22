"""CAPEN-Llama-att: writeup class-token pooling + hyperedge-only edge–edge attention.

Extends :class:`~cpen.models.capen_llama.CAPENLlama` without changing the default
CAPEN-Llama behaviour.

* **Readout** — :class:`~cpen.models.output_attention_pool.OutputAttentionPool`:
  class-token \(\alpha_A=1\) pools build Θ(1) features \(X_{\mathrm{out}}\)
  (fuse \(\frac{1}{\sqrt{2}}(P_X+P_E)\)), then the CompleteP head
  \(\mathrm{logit}_a=\frac{1}{D\sigma}X_{\mathrm{out},a}\cdot w_a\) with
  \(w_a\sim\mathcal N(0,\sigma^2)\). Output \(\Delta\rho\) must stay \(O(1)\)
  in width under Adam \(\eta=\eta_0/\sqrt{D}\).
* **M₂₂** — edge–edge attention runs only among hyperedges (incidence degree
  ``> 2``). Pairwise kNN / vn_link slots keep identity ``f_22``. Implemented as
  gather→attend→scatter so we never materialize dense ``M×M`` attention scores
  over padded edge slots (critical for hierarchical caches with ``M∼10³``).
  ``--hyperedge-only`` (``ignore_knn_edges``) additionally drops those pairwise
  rows from the edge stream before ``M_{12}/M_{21}``.
* **Typed edge streams** (``typed_edge_streams``) — dual CompleteP encoders
  (star kin7 → √(D/7), kNN part-int → √(D/4), no pad). Routing (§1 in
  ``scans/jets/TYPED_EDGE_ATTENTION_TARGET.md``): particles get M11+M21 from
  **all** edges; hyperedges (``deg>2``) get M12+M22; non-hyper edges (kNN +
  tiny stars) get M12 only with ``f_22=0`` (no LN self-copy, no knn↔knn).
  Edge residual: hyper ``1/(√2 L)``, non-hyper ``1/L``.
* **RoPE** — ``--use-rope`` applies axial :math:`(\\eta,\\phi)` rotary embeddings
  on relation 11 only (jet-centered :math:`(\\Delta\\eta,\\Delta\\phi)` from
  four-vectors). Mutually exclusive with ``--use-wire``.
* **BaselineTransformer** — separate particle-only class (not ``m11_only`` on
  this trunk): Llama MLPs, CompleteP scales, node class-token pool, single-branch
  residual \(\alpha_A^{\mathrm{res}}=1/L\) (no \(\sqrt{2}\)).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn

from cpen.models.capen import supports_without_m22
from cpen.models.capen_llama import CAPENLlama
from cpen.models.output_attention_pool import OutputAttentionPool

EdgeBiasM11 = Literal["none", "dynamic", "static"]


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
        typed_edge_streams: bool = False,
        n_star_edge_features: int = 7,
        n_knn_edge_features: int = 4,
        edge_bias_m11: EdgeBiasM11 = "none",
        m21: bool = True,
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
        self.typed_edge_streams = bool(typed_edge_streams)
        self.n_star_edge_features = int(n_star_edge_features)
        self.n_knn_edge_features = int(n_knn_edge_features)
        mode = str(edge_bias_m11 or "none").lower()
        if mode not in {"none", "dynamic", "static"}:
            raise ValueError(
                f"edge_bias_m11 must be none|dynamic|static; got {edge_bias_m11!r}"
            )
        self.edge_bias_m11: EdgeBiasM11 = mode  # type: ignore[assignment]
        self.m21 = bool(m21)
        if self.m11_only and bool(getattr(self, "use_wire", False)):
            raise ValueError(
                "m11_only / BaselineTransformer cannot use WIRE (needs a graph Laplacian)"
            )
        if self.typed_edge_streams and self.m11_only:
            raise ValueError("typed_edge_streams is incompatible with m11_only")
        if self.edge_bias_m11 != "none" and not self.typed_edge_streams:
            raise ValueError("edge_bias_m11 requires typed_edge_streams=True")
        if not self.m21 and not self.typed_edge_streams:
            raise ValueError("m21=False is only supported with typed_edge_streams")
        if self.uniform_output_pool:
            self.output_pool = None
        else:
            self.output_pool = OutputAttentionPool(
                int(self.D),
                self.n_class_tokens,
                heads=int(self.heads),
                dropout=float(getattr(self, "dropout", 0.0) or 0.0),
                node_only=self.m11_only,
            )
        # Keep bytecode CAPEN CompleteP readout scale: 1/(D σ) = 1/√D.
        # (Do not override to 1.0 — that breaks O(1) output-update μP.)
        self._pending_edge_type: torch.Tensor | None = None
        self._pending_edge_mask: torch.Tensor | None = None
        self._pending_is_hyper: torch.Tensor | None = None
        self._pending_is_star: torch.Tensor | None = None
        self._pending_is_knn: torch.Tensor | None = None
        self._pending_m22_full: torch.Tensor | None = None
        self._attn_residual_scale_single = 1.0 / float(max(int(self.depth), 1))
        if self.typed_edge_streams:
            # CompleteP: W ~ N(0, σ²) with σ = D^{-1/2}, same as encoder_x / encoder_e.
            # Do not use getattr(init_std, 1.0) — that is N(0,1) and breaks width-stable Δρ.
            d = int(self.D)
            sigma = 1.0 / math.sqrt(d)
            n_s = int(self.n_star_edge_features)
            n_k = int(self.n_knn_edge_features)
            if n_s < 1 or n_k < 1:
                raise ValueError(
                    f"typed edge fan-ins must be >=1; got star={n_s} knn={n_k}"
                )
            self.encoder_e_star = nn.Linear(n_s, d, bias=False)
            self.encoder_e_knn = nn.Linear(n_k, d, bias=False)
            nn.init.normal_(self.encoder_e_star.weight, mean=0.0, std=sigma)
            nn.init.normal_(self.encoder_e_knn.weight, mean=0.0, std=sigma)
            self._encoder_e_star_scale = math.sqrt(d / float(n_s))
            self._encoder_e_knn_scale = math.sqrt(d / float(n_k))
            # Drop unused shared encoder so it is not in P / corr groups / DDP.
            del self.encoder_e
            if self.edge_bias_m11 != "none":
                # Readout rule: W_b ~ N(0, σ²), forward × 1/(Dσ) = 1/√D.
                self.bias_11 = nn.ModuleList(
                    [
                        nn.Linear(d, int(self.heads), bias=False)
                        for _ in range(int(self.depth))
                    ]
                )
                for lin in self.bias_11:
                    nn.init.normal_(lin.weight, mean=0.0, std=sigma)
                self._bias_scale = 1.0 / math.sqrt(d)
            if not self.m21:
                # Drop M21 params from P/DDP. Particle residual uses 1/L in forward
                # (n_src=1); edge residuals stay typed §1 (hyper 1/(√2L), else 1/L).
                del self.attn_21

    def _gamma_for(self, relation: str) -> float:
        # Typed §1 M22 is sparse hyper↔hyper gather/scatter — not dense all-to-all.
        # Skip γ₂₂ in the model itself (do not rely on sweep_common name-clamp → 1).
        if relation == "22" and bool(getattr(self, "typed_edge_streams", False)):
            return 1.0
        return super()._gamma_for(relation)

    def _build_supports(
        self,
        s_bool: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if bool(getattr(self, "identity_m22", False)) or bool(
            getattr(self, "incidence_m22", False)
        ):
            self._pending_is_hyper = None
            self._pending_m22_full = None
            return supports_without_m22(s_bool)
        m_11, m_21, m_12, m_22 = super()._build_supports(s_bool, mask=mask)
        if self.typed_edge_streams:
            # Hyperedges (deg>2) only for M22; placeholder identity avoids a
            # dense all-edge M22 in the generic path. M12/M21 stay full
            # incidence (all edges).
            deg = s_bool.to(dtype=torch.float32).sum(dim=-1)
            is_hyper = deg > 2
            edge_mask = self._pending_edge_mask
            if edge_mask is not None:
                is_hyper = is_hyper & edge_mask.to(
                    device=is_hyper.device, dtype=torch.bool
                )
            self._pending_is_hyper = is_hyper
            self._pending_m22_full = None
            eye = torch.eye(m_22.size(-1), device=m_22.device, dtype=torch.bool)
            m_22 = eye.unsqueeze(0).expand(m_22.size(0), -1, -1).contiguous()
            return m_11, m_21, m_12, m_22
        if not self.hyperedge_m22_only:
            self._pending_is_hyper = None
            self._pending_m22_full = None
            return m_11, m_21, m_12, m_22

        # Degree > 2 ⇒ hyperedge (DBSCAN / virtual hyper); 2-edges stay identity-only.
        deg = s_bool.to(dtype=torch.float32).sum(dim=-1)
        is_hyper = deg > 2
        edge_mask = self._pending_edge_mask
        if edge_mask is not None:
            is_hyper = is_hyper & edge_mask.to(device=is_hyper.device, dtype=torch.bool)
        self._pending_is_hyper = is_hyper
        self._pending_m22_full = None
        # Cheap placeholder: identity only. Real M22 runs on the hyper subset in
        # ``_run_attn`` (avoids allocating bool/float ``M×M`` attention).
        eye = torch.eye(m_22.size(-1), device=m_22.device, dtype=torch.bool)
        m_22 = eye.unsqueeze(0).expand(m_22.size(0), -1, -1).contiguous()
        return m_11, m_21, m_12, m_22

    def _m11_edge_bias(
        self,
        *,
        layer_idx: int,
        e_ln: torch.Tensor,
        e_ln0: torch.Tensor | None,
        knn_dst: torch.Tensor,
        knn_valid: torch.Tensor,
        n_star: int,
        n_particles: int,
    ) -> torch.Tensor:
        """Build (B, H, N, N) M11 logit bias from live kNN edge slots."""
        if self.edge_bias_m11 == "none":
            raise RuntimeError("_m11_edge_bias called with edge_bias_m11=none")
        z_src = e_ln0 if self.edge_bias_m11 == "static" else e_ln
        if z_src is None:
            raise RuntimeError("static edge bias requires e_ln0")
        z = z_src[:, n_star:, :]  # (B, N*k, D)
        bsz = int(z.size(0))
        k = int(knn_dst.size(-1))
        if z.size(1) != n_particles * k:
            raise ValueError(
                f"kNN edge count {z.size(1)} != N*k={n_particles * k}"
            )
        u = self.bias_11[layer_idx](z) * self._bias_scale  # (B, N*k, H)
        u = u.view(bsz, n_particles, k, int(self.heads)).permute(0, 3, 1, 2)
        u = u * knn_valid.unsqueeze(1).to(dtype=u.dtype)
        idx = knn_dst.unsqueeze(1).expand(bsz, int(self.heads), n_particles, k)
        bias = u.new_zeros(bsz, int(self.heads), n_particles, n_particles)
        bias.scatter_add_(-1, idx, u)
        return bias

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
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            relation == "22"
            and self.typed_edge_streams
            and not bool(getattr(self, "identity_m22", False))
            and not bool(getattr(self, "incidence_m22", False))
        ):
            # Typed forward builds f_22 explicitly; this path should not run.
            raise RuntimeError("typed_edge_streams builds M22 via subset helpers")
        if (
            relation == "22"
            and self.hyperedge_m22_only
            and not self.typed_edge_streams
            and not bool(getattr(self, "identity_m22", False))
            and not bool(getattr(self, "incidence_m22", False))
        ):
            return self._run_attn_22_member_subset(
                module,
                target,
                source,
                member_mask=self._pending_is_hyper,
                full_support=None,
                zeros_elsewhere=False,
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
            attn_bias=attn_bias,
        )

    def _run_attn_22_member_subset(
        self,
        module: nn.Module,
        target: torch.Tensor,
        source: torch.Tensor,
        *,
        member_mask: torch.Tensor | None,
        full_support: torch.Tensor | None = None,
        zeros_elsewhere: bool = False,
        probe: dict | None = None,
    ) -> torch.Tensor:
        """``f_22`` among ``member_mask`` rows; identity or zeros elsewhere.

        ``full_support=None`` → all-to-all among members (star block).
        ``full_support`` → gather the incidence M22 sub-block (knn block).
        """
        if member_mask is None:
            unused = sum((p * 0.0).sum() for p in module.parameters())
            base = target.new_zeros(target.shape) if zeros_elsewhere else target
            return base + unused

        member_mask = member_mask.to(device=target.device, dtype=torch.bool)
        batch, n_edges, width = target.shape
        counts = member_mask.sum(dim=-1)
        max_h = int(counts.max().item()) if batch > 0 else 0
        if max_h <= 0:
            unused = sum((p * 0.0).sum() for p in module.parameters())
            base = target.new_zeros(target.shape) if zeros_elsewhere else target
            return base + unused

        idx = target.new_zeros(batch, max_h, dtype=torch.long)
        mem_ok = torch.zeros(batch, max_h, dtype=torch.bool, device=target.device)
        for b in range(batch):
            ix = member_mask[b].nonzero(as_tuple=False).flatten()
            n = int(ix.numel())
            if n == 0:
                continue
            idx[b, :n] = ix
            mem_ok[b, :n] = True

        gather_idx = idx.unsqueeze(-1).expand(batch, max_h, width)
        tgt_h = torch.gather(target, 1, gather_idx)
        src_h = torch.gather(source, 1, gather_idx)
        if full_support is None:
            support_h = mem_ok.unsqueeze(-1) & mem_ok.unsqueeze(-2)
        else:
            supp = full_support.to(device=target.device, dtype=torch.bool)
            support_h = torch.zeros(
                batch, max_h, max_h, dtype=torch.bool, device=target.device
            )
            for b in range(batch):
                n = int(counts[b].item())
                if n <= 0:
                    continue
                ix = idx[b, :n]
                support_h[b, :n, :n] = supp[b][ix][:, ix]

        out_h = super()._run_attn(
            module,
            tgt_h,
            src_h,
            support_h,
            relation="22",
            probe=probe,
        )

        out = target.new_zeros(target.shape) if zeros_elsewhere else target.clone()
        out_h = out_h.to(dtype=out.dtype)
        for b in range(batch):
            n = int(counts[b].item())
            if n > 0:
                out[b, idx[b, :n]] = out_h[b, :n]
        return out

    # Back-compat name used by older call sites / docs.
    def _run_attn_22_hyper_subset(
        self,
        module: nn.Module,
        target: torch.Tensor,
        source: torch.Tensor,
        *,
        probe: dict | None = None,
    ) -> torch.Tensor:
        return self._run_attn_22_member_subset(
            module,
            target,
            source,
            member_mask=self._pending_is_hyper,
            full_support=None,
            zeros_elsewhere=False,
            probe=probe,
        )

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
        edge_x_star: torch.Tensor | None = None,
        edge_x_knn: torch.Tensor | None = None,
        knn_dst: torch.Tensor | None = None,
        knn_valid: torch.Tensor | None = None,
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
        typed = bool(self.typed_edge_streams)
        if typed:
            if edge_x_star is None or edge_x_knn is None:
                raise ValueError(
                    "typed_edge_streams requires edge_x_star and edge_x_knn"
                )
            if edge_x_star.dim() != 3 or edge_x_knn.dim() != 3:
                raise ValueError(
                    "edge_x_star / edge_x_knn must be (batch, n_edges, n_features)"
                )
        elif edge_x is None or edge_x.dim() != 3:
            raise ValueError(
                f"CAPEN expects sparse edge features (batch, n_edges, m_0); got {None if edge_x is None else tuple(edge_x.shape)}"
            )
        if self.edge_bias_m11 != "none":
            if knn_dst is None or knn_valid is None:
                raise ValueError(
                    "edge_bias_m11 requires knn_dst and knn_valid from the graph builder"
                )

        node_mask_eff = node_mask if node_mask is not None else mask
        self._pending_edge_type = edge_type
        self._pending_edge_mask = edge_mask
        self._pending_is_hyper = None
        self._pending_is_star = None
        self._pending_is_knn = None
        self._pending_m22_full = None
        try:
            num_nodes = x.size(1)
            if typed:
                n_star = int(edge_x_star.size(1))
                n_knn = int(edge_x_knn.size(1))
                num_edges = n_star + n_knn
            else:
                n_star = 0
                n_knn = 0
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

            if typed:
                is_star = torch.zeros(
                    s_bool.size(0), num_edges, dtype=torch.bool, device=x.device
                )
                is_star[:, :n_star] = True
                is_knn = ~is_star
                live = s_bool.any(dim=-1)
                if edge_mask is not None:
                    live = live & edge_mask.to(device=live.device, dtype=torch.bool)
                self._pending_is_star = is_star & live
                self._pending_is_knn = is_knn & live
                if edge_mask is None:
                    edge_mask = live

            self._pending_edge_mask = edge_mask
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
            if typed:
                h_star = (
                    self.encoder_e_star(edge_x_star) * self._encoder_e_star_scale
                )
                h_knn = self.encoder_e_knn(edge_x_knn) * self._encoder_e_knn_scale
                h_e = torch.cat([h_star, h_knn], dim=1)
            else:
                h_e = self.encoder_e(edge_x) * self._encoder_e_scale
            self._record_probe("encode", h_x, h_e, node_mask_eff)
            e_ln0 = self.ln(h_e) if self.edge_bias_m11 == "static" else None

            for layer_idx in range(self.depth):
                x_ln = self.ln(h_x)
                e_ln = self.ln(h_e)
                rot_coords, rotary = self._relation_11_rotary(
                    layer_idx, rope_coords=rope_coords, wire_coords=wire_coords
                )
                attn_bias_11 = None
                probe11 = self._attn_probe_entry(layer_idx, "11")
                if self.edge_bias_m11 != "none":
                    assert knn_dst is not None and knn_valid is not None
                    attn_bias_11 = self._m11_edge_bias(
                        layer_idx=layer_idx,
                        e_ln=e_ln,
                        e_ln0=e_ln0,
                        knn_dst=knn_dst,
                        knn_valid=knn_valid,
                        n_star=n_star,
                        n_particles=num_nodes,
                    )
                    if probe11 is not None:
                        gathered = torch.gather(
                            attn_bias_11,
                            -1,
                            knn_dst.unsqueeze(1).expand(
                                attn_bias_11.size(0),
                                attn_bias_11.size(1),
                                num_nodes,
                                knn_dst.size(-1),
                            ),
                        )
                        m = knn_valid.unsqueeze(1).expand_as(gathered)
                        if bool(m.any()):
                            probe11["bias11_rms"] = (
                                gathered[m].float().pow(2).mean().sqrt()
                            )
                        else:
                            probe11["bias11_rms"] = gathered.new_zeros(())
                f_11 = self._run_attn(
                    self.attn_11[layer_idx],
                    x_ln,
                    x_ln,
                    m_11,
                    relation="11",
                    probe=probe11,
                    target_coordinates=rot_coords,
                    source_coordinates=rot_coords,
                    wire=rotary,
                    attn_bias=attn_bias_11,
                )
                if self.m21:
                    f_21 = self._run_attn(
                        self.attn_21[layer_idx],
                        x_ln,
                        e_ln,
                        m_21,
                        relation="21",
                        probe=self._attn_probe_entry(layer_idx, "21"),
                    )
                else:
                    f_21 = torch.zeros_like(f_11)
                if typed:
                    # §1: hyper↔hyper M22 only; f_22=0 on knn / tiny stars.
                    # M12 on all live edges; M21 unmasked into particles.
                    f_22 = self._run_attn_22_member_subset(
                        self.attn_22[layer_idx],
                        e_ln,
                        e_ln,
                        member_mask=self._pending_is_hyper,
                        full_support=None,
                        zeros_elsewhere=True,
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
                    hyper = self._pending_is_hyper
                    assert hyper is not None
                    live = (
                        edge_mask
                        if edge_mask is not None
                        else torch.ones_like(hyper)
                    )
                    live = live.to(device=hyper.device, dtype=torch.bool)
                    hyper_w = hyper.unsqueeze(-1).to(dtype=f_12.dtype)
                    nonhyper_w = ((~hyper) & live).unsqueeze(-1).to(dtype=f_12.dtype)
                    attn_s = self._attn_residual_scale
                    attn_s1 = self._attn_residual_scale_single
                    # Particles: 1/(√2 L) with M21, else 1/L (single branch).
                    attn_sx = attn_s if self.m21 else attn_s1
                    h_x = h_x + attn_sx * (f_11 + f_21)
                    # Hyper: two edge branches → 1/(√2 L). Non-hyper: M12 only → 1/L.
                    h_e = (
                        h_e
                        + attn_s * hyper_w * (f_22 + f_12)
                        + attn_s1 * nonhyper_w * f_12
                    )
                else:
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
            self._pending_is_star = None
            self._pending_is_knn = None
            self._pending_m22_full = None
            self._pending_s = None


class BaselineTransformer(CAPENLlamaAtt):
    """Edgeless particle transformer: M11 + Llama MLPs + node class-token pool.

    Residual attention scale is \(1/L\) (one branch into the particle stream).
    The coupled CAPEN trunk keeps \(1/(\\sqrt{2}\,L)\) because two attn branches
    feed each stream — that \(\sqrt{2}\) must not leak onto this baseline.
    """

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        kwargs["m11_only"] = True
        super().__init__(*args, **kwargs)
        depth = max(int(self.depth), 1)
        self._attn_residual_scale = 1.0 / float(depth)
        self._mlp_residual_scale = 1.0 / float(depth)


__all__ = ["CAPENLlamaAtt", "BaselineTransformer"]
