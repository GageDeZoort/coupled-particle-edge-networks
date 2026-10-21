"""Reconstructed CAPEN / SupportAttention backend (replaces lost ``capen.pyc``).

Recovered from wrappers, CompleteP docs, and the unit-test suite after the
gitignored bytecode was deleted. Behaviour is pinned by ``tests/test_capen.py``
and ``tests/test_wire.py``; treat deviations as bugs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from cpen.models.base import adam_lr


@dataclass(frozen=True)
class AttentionGammas:
    """Mean support-row degrees used as attention temperatures."""

    gamma_11: float = 1.0
    gamma_12: float = 1.0
    gamma_21: float = 1.0
    gamma_22: float = 1.0

    @classmethod
    def ones(cls) -> AttentionGammas:
        return cls()

    def as_dict(self) -> dict[str, float]:
        """Match ``OperatorGammas.as_dict`` for bytecode dump/tag paths."""
        return {
            "gamma_11": float(self.gamma_11),
            "gamma_12": float(self.gamma_12),
            "gamma_21": float(self.gamma_21),
            "gamma_22": float(self.gamma_22),
        }


def attention_diversity_stats(
    weights: torch.Tensor,
    support: torch.Tensor,
) -> dict[str, torch.Tensor] | None:
    """Per-head entropy / effective width / max weight over supported sources.

    ``weights``: ``(B, H, T, S)``; ``support``: ``(B, T, S)`` bool.
    """
    if weights.numel() == 0:
        return None
    supp = support.unsqueeze(1).to(dtype=weights.dtype)
    w = weights * supp
    # Renormalize nonempty rows.
    row = w.sum(dim=-1, keepdim=True)
    nonempty = (row.squeeze(-1) > 0)
    w = torch.where(row > 0, w / row.clamp_min(1e-12), w)
    # Entropy in nats / log(|support|), clipped to [0, 1].
    log_w = torch.where(w > 0, w.clamp_min(1e-12).log(), torch.zeros_like(w))
    ent = -(w * log_w).sum(dim=-1)
    n_src = supp.sum(dim=-1).clamp_min(1.0)
    ent_norm = (ent / n_src.clamp_min(1.0).log().clamp_min(1e-12)).clamp(0.0, 1.0)
    eff = torch.exp(ent).clamp_max(n_src)
    mx = w.max(dim=-1).values
    # Average over batch + targets that have support, keep heads.
    mask = nonempty  # (B, H, T)
    def _mean_heads(x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, T) → (H,)
        out = []
        for h in range(x.size(1)):
            m = mask[:, h]
            if m.any():
                out.append(x[:, h][m].mean())
            else:
                out.append(x.new_zeros(()))
        return torch.stack(out)

    return {
        "entropy": _mean_heads(ent_norm),
        "eff_width": _mean_heads(eff),
        "max_weight": _mean_heads(mx),
    }


class SupportAttention(nn.Module):
    """Masked multi-head attention over an explicit boolean support.

    CompleteP ``α_A = 1``: ``W_{QKVO} ∼ N(0,1)``, projections ``× 1/√D``,
    logits ``× 1/d_head``. Optional WIRE rotates Q/K only.
    """

    def __init__(
        self,
        width: int,
        heads: int = 1,
        dropout: float = 0.0,
        *,
        head_combine: str = "concat",
        out_dim: int | None = None,
    ) -> None:
        super().__init__()
        d = int(width)
        h = int(heads)
        if d < 1 or h < 1 or d % h != 0:
            raise ValueError(f"width={d} must be positive and divisible by heads={h}")
        combine = str(head_combine or "concat").lower()
        if combine not in {"concat", "mean"}:
            raise ValueError(f"head_combine must be 'concat' or 'mean'; got {head_combine!r}")
        p = float(dropout)
        if not 0.0 <= p < 1.0:
            raise ValueError(f"attention dropout must be in [0, 1); got {p}")
        odim = d if out_dim is None else int(out_dim)
        if odim < 1:
            raise ValueError(f"out_dim must be >= 1; got {odim}")
        self.width = d
        self.out_dim = odim
        self.heads = h
        self.head_dim = d // h
        self.dropout = p
        self.head_combine = combine
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        # concat: WO maps D→out_dim on concatenated heads; mean: WO maps d_h→out_dim.
        # Trunk uses out_dim=D; class-token pool uses out_dim=1 (logits, no extra unembed).
        self.wo = nn.Linear(self.head_dim if combine == "mean" else d, odim, bias=False)
        for lin in (self.wq, self.wk, self.wv, self.wo):
            nn.init.normal_(lin.weight, mean=0.0, std=1.0)
        self._proj_scale = 1.0 / math.sqrt(d)
        self._logit_scale = 1.0 / float(self.head_dim)
        # concat WO: ×1/√D = 1/(Dσ); mean WO: ×1/√d_h so outputs stay O(1).
        self._out_scale = (
            1.0 / math.sqrt(self.head_dim) if combine == "mean" else self._proj_scale
        )

    def _drop_support(
        self, support: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        """Bernoulli drop on supported entries; keep ≥1 source on nonempty rows."""
        # support: (B, T, S); logits: (B, H, T, S) — broadcast mask to heads.
        b, h, t, s = logits.shape
        base = support.unsqueeze(1).expand(b, h, t, s)
        if (not self.training) or self.dropout <= 0.0:
            return base
        keep = torch.rand(b, h, t, s, device=logits.device) >= self.dropout
        mask = base & keep
        # Rows that were nonempty in base but are now empty → restore one source.
        was = base.any(dim=-1)
        now_empty = was & ~mask.any(dim=-1)
        if now_empty.any():
            # Pick first supported index per (b,h,t).
            # base is bool; use max of (base * arange) inverted — first True.
            idx = base.to(torch.long).argmax(dim=-1)  # first True works if ≥1
            # For rows that had support, argmax of bool gives first True in PyTorch? 
            # Actually True>False so argmax finds last True. Use cumsum trick.
            first = (base.to(torch.long).cumsum(dim=-1) == 1) & base
            # first is one-hot at first True.
            restore = first & now_empty.unsqueeze(-1)
            mask = mask | restore
        # Never attend outside original support.
        return mask & base

    def _attention_weights(
        self, logits: torch.Tensor, support: torch.Tensor
    ) -> torch.Tensor:
        """Softmax over sources with support masking + optional dropout."""
        mask = self._drop_support(support, logits)
        neg = torch.finfo(logits.dtype).min
        # NaN logits → treat as -inf inside support so fallback stays in-support.
        safe = torch.where(torch.isfinite(logits), logits, logits.new_full((), neg))
        masked = safe.masked_fill(~mask, neg)
        # All-masked rows (empty support): softmax of all -inf → nan; fix below.
        weights = torch.softmax(masked, dim=-1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        # Renormalize nonempty rows (dropout can zero some mass).
        row = weights.sum(dim=-1, keepdim=True)
        weights = torch.where(row > 0, weights / row.clamp_min(1e-12), weights)
        # Empty original support → zeros (already).
        empty = ~support.any(dim=-1).unsqueeze(1)
        weights = weights.masked_fill(empty.unsqueeze(-1), 0.0)
        return weights

    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        support: torch.Tensor,
        *,
        target_coordinates: torch.Tensor | None = None,
        source_coordinates: torch.Tensor | None = None,
        wire: nn.Module | None = None,
        logit_temperature: float = 1.0,
    ) -> torch.Tensor:
        if target.dim() != 3 or source.dim() != 3:
            raise ValueError("target/source must be (batch, tokens, width)")
        batch, n_t, _ = target.shape
        n_s = source.size(1)
        scale = self._proj_scale
        q = (self.wq(target) * scale).view(batch, n_t, self.heads, self.head_dim)
        k = (self.wk(source) * scale).view(batch, n_s, self.heads, self.head_dim)
        v = (self.wv(source) * scale).view(batch, n_s, self.heads, self.head_dim)
        if wire is not None and target_coordinates is not None and source_coordinates is not None:
            q, k = wire(q, k, target_coordinates, source_coordinates)
        logits = torch.einsum("bihd,bjhd->bhij", q, k) * self._logit_scale
        temp = float(logit_temperature)
        if temp != 1.0:
            logits = logits / temp
        weights = self._attention_weights(logits, support.bool())
        ctx = torch.einsum("bhij,bjhd->bihd", weights, v)
        if self.head_combine == "mean":
            # Mean over heads, ×√H so variance stays Θ(1) as H grows with width.
            ctx = ctx.mean(dim=2) * math.sqrt(float(self.heads))
        else:
            ctx = ctx.reshape(batch, n_t, self.width)
        return self.wo(ctx) * self._out_scale


class CAPEN(nn.Module):
    """Coupled attentional particle–edge network (GELU MLP residuals)."""

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
        all_to_all_particle_attention: bool = False,
        **_ignored: Any,
    ) -> None:
        super().__init__()
        del _ignored  # accept forward-compat kwargs from Llama / att constructors
        if readout_mode not in {"graph", "node"}:
            raise ValueError(f"readout_mode must be graph|node; got {readout_mode!r}")
        if normalization not in {"uniform", "energy-weights"}:
            raise ValueError(f"normalization must be uniform|energy-weights; got {normalization!r}")
        p = float(dropout)
        if not 0.0 <= p < 1.0:
            raise ValueError(f"attention dropout must be in [0, 1); got {p}")
        if int(width) % int(heads) != 0:
            raise ValueError(f"heads must divide width; got width={width}, heads={heads}")

        self.n_features = int(n_features)
        self.n_edge_features = int(n_edge_features)
        self.out_dim = int(out_dim)
        self.depth = int(depth)
        self.width = int(width)
        self.D = int(width)
        self.heads = int(heads)
        self.head_dim = self.D // self.heads
        self.dropout = p
        self.normalization = normalization
        self.energy_index = int(energy_index)
        self.optimizer = str(optimizer).lower()
        self.readout_mode = readout_mode
        self.attention_normalization = str(attention_normalization or "none")
        self.all_to_all_particle_attention = bool(all_to_all_particle_attention)
        self.use_wire = False  # CAPEN base ignores WIRE; Llama enables it
        self.wire_11 = None
        self.wire_coordinate_dim = int(wire_coordinate_dim)
        self.wire_share_across_heads = bool(wire_share_across_heads)
        self.wire_frequency_init_std = wire_frequency_init_std
        self.wire_sign_augmentation = bool(wire_sign_augmentation)
        self.wire_normalized_laplacian = bool(wire_normalized_laplacian)

        sigma = 1.0 / math.sqrt(self.D)
        # Encoder (CompleteP / ParticleOnly): W ~ N(0, σ²), forward × √(D/n_in).
        # (Not N(0,1): that makes ρ∝D and breaks width-stable output Δρ.)
        self.encoder_x = nn.Linear(self.n_features, self.D, bias=False)
        self.encoder_e = nn.Linear(self.n_edge_features, self.D, bias=False)
        nn.init.normal_(self.encoder_x.weight, mean=0.0, std=sigma)
        nn.init.normal_(self.encoder_e.weight, mean=0.0, std=sigma)
        self._encoder_x_scale = math.sqrt(self.D / self.n_features)
        self._encoder_e_scale = math.sqrt(self.D / self.n_edge_features)

        self.ln = nn.LayerNorm(self.D, elementwise_affine=False)
        self.attn_11 = nn.ModuleList(
            [SupportAttention(self.D, self.heads, dropout=p) for _ in range(self.depth)]
        )
        self.attn_12 = nn.ModuleList(
            [SupportAttention(self.D, self.heads, dropout=p) for _ in range(self.depth)]
        )
        self.attn_21 = nn.ModuleList(
            [SupportAttention(self.D, self.heads, dropout=p) for _ in range(self.depth)]
        )
        self.attn_22 = nn.ModuleList(
            [SupportAttention(self.D, self.heads, dropout=p) for _ in range(self.depth)]
        )

        # GELU MLP: D → 4D → D.
        self.mlp_x_w1 = nn.ModuleList(
            [nn.Linear(self.D, 4 * self.D, bias=False) for _ in range(self.depth)]
        )
        self.mlp_x_w2 = nn.ModuleList(
            [nn.Linear(4 * self.D, self.D, bias=False) for _ in range(self.depth)]
        )
        self.mlp_e_w1 = nn.ModuleList(
            [nn.Linear(self.D, 4 * self.D, bias=False) for _ in range(self.depth)]
        )
        self.mlp_e_w2 = nn.ModuleList(
            [nn.Linear(4 * self.D, self.D, bias=False) for _ in range(self.depth)]
        )
        for lin in list(self.mlp_x_w1) + list(self.mlp_x_w2) + list(self.mlp_e_w1) + list(self.mlp_e_w2):
            nn.init.normal_(lin.weight, mean=0.0, std=1.0)
        self._mlp_w1_scale = 1.0 / math.sqrt(self.D)
        self._mlp_w2_scale = 1.0 / math.sqrt(4 * self.D)
        self._attn_residual_scale = 1.0 / (math.sqrt(2.0) * self.depth)
        self._mlp_residual_scale = 1.0 / float(self.depth)

        # Decoder: W ~ N(0, σ²), forward × 1/(D σ) = 1/√D.
        self.decoder_x = nn.Linear(self.D, self.out_dim, bias=False)
        self.decoder_e = nn.Linear(self.D, self.out_dim, bias=False)
        nn.init.normal_(self.decoder_x.weight, mean=0.0, std=sigma)
        nn.init.normal_(self.decoder_e.weight, mean=0.0, std=sigma)
        self._readout_scale = 1.0 / (self.D * sigma)

        gammas = attention_gammas or AttentionGammas.ones()
        self.register_buffer("attn_gamma_11", torch.tensor(float(gammas.gamma_11)))
        self.register_buffer("attn_gamma_12", torch.tensor(float(gammas.gamma_12)))
        self.register_buffer("attn_gamma_21", torch.tensor(float(gammas.gamma_21)))
        self.register_buffer("attn_gamma_22", torch.tensor(float(gammas.gamma_22)))

        self._feature_probe: list | None = None
        self._attention_probe: list | None = None
        self.identity_m22 = False
        self.incidence_m22 = False
        self._pending_s: torch.Tensor | None = None

    def get_lr(self, eta_0: float, corr: float = 1.0) -> float:
        if self.optimizer in {"adam", "adamw"}:
            return adam_lr(eta_0, self.width, corr=corr)
        raise ValueError(f"Unknown optimizer {self.optimizer!r}")

    def get_weight_decay(self, lambda_0: float) -> float:
        return float(lambda_0) * math.sqrt(self.width)

    def _record_probe(
        self,
        stage: str,
        h_x: torch.Tensor,
        h_e: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> None:
        probe = getattr(self, "_feature_probe", None)
        if probe is None:
            return
        probe.append((stage, h_x.detach(), h_e.detach(), mask))

    def _attn_probe_entry(self, layer_idx: int, relation: str) -> dict | None:
        probe = getattr(self, "_attention_probe", None)
        if probe is None:
            return None
        entry: dict = {
            "layer": int(layer_idx),
            "stage": f"layer{int(layer_idx)}",
            "relation": str(relation),
        }
        probe.append(entry)
        return entry

    @staticmethod
    def _build_supports(
        s_bool: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Incidence → (m_11, m_21, m_12, m_22) with self-loops on 11 and 22."""
        s_float = s_bool.to(torch.float32)
        m_11 = torch.bmm(s_float.transpose(1, 2), s_float) > 0.0
        n_particles = s_bool.size(-1)
        eye_x = torch.eye(n_particles, device=s_bool.device, dtype=torch.bool)
        m_11 = m_11 | eye_x
        if mask is not None:
            node_ok = mask.to(device=s_bool.device, dtype=torch.bool)
            m_11 = m_11 & node_ok.unsqueeze(1) & node_ok.unsqueeze(2)
        m_12 = s_bool
        m_21 = s_bool.transpose(1, 2)
        m_22 = torch.bmm(s_float, s_float.transpose(1, 2)) > 0.0
        n_edges = s_bool.size(-2)
        eye_e = torch.eye(n_edges, device=s_bool.device, dtype=torch.bool)
        m_22 = m_22 | eye_e
        return m_11, m_21, m_12, m_22

    def _resolve_dense_incidence(
        self,
        *,
        incidence: torch.Tensor | None,
        incidence_node: torch.Tensor | None,
        incidence_edge: torch.Tensor | None,
        incidence_nnz: torch.Tensor | None,
        num_edges: int,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        if incidence is not None:
            return incidence.to(device=device, dtype=torch.bool)
        if incidence_node is None or incidence_edge is None or incidence_nnz is None:
            raise ValueError("Need dense incidence or COO (incidence_node/edge/nnz)")
        batch = int(incidence_nnz.size(0))
        s = torch.zeros(batch, num_edges, num_nodes, dtype=torch.bool, device=device)
        for b in range(batch):
            nnz = int(incidence_nnz[b].item())
            if nnz <= 0:
                continue
            # test_capen: nonzero(incidence[b]) → (edge, node); passed as
            # incidence_node=node_idx, incidence_edge=edge_idx.
            n_idx = incidence_node[b, :nnz].long()
            e_idx = incidence_edge[b, :nnz].long()
            s[b, e_idx, n_idx] = True
        return s

    def _gamma_for(self, relation: str) -> float:
        """Degree temperature for sparse incidence relations only.

        All-to-all supports have mean row-degree ≈ #tokens, so dividing logits
        by γ collapses softmax to uniform. Skip γ on:

        * ``11`` when ``all_to_all_particle_attention`` (dense particle MHSA)
        * ``22`` when ``hyperedge_m22_only`` (all-to-all among hyperedges)

        Keep γ on ``12`` / ``21`` (incidence ``S`` / ``Sᵀ`` row degrees).
        """
        if self.attention_normalization != "gamma":
            return 1.0
        if relation == "11" and bool(getattr(self, "all_to_all_particle_attention", False)):
            return 1.0
        if relation == "22" and bool(getattr(self, "hyperedge_m22_only", False)):
            return 1.0
        buf = getattr(self, f"attn_gamma_{relation}", None)
        if buf is None:
            return 1.0
        return float(buf.item() if torch.is_tensor(buf) else buf)

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
        temp = self._gamma_for(relation)
        out = module(
            target,
            source,
            support,
            target_coordinates=target_coordinates,
            source_coordinates=source_coordinates,
            wire=wire,
            logit_temperature=temp,
        )
        if probe is not None:
            # Diagnostic second pass: diversity + numerical pathology stats.
            # Tests require entropy/eff_width/max_weight shapes when probe is on.
            with torch.no_grad():
                scale = module._proj_scale
                b, n_t, _ = target.shape
                n_s = source.size(1)
                q = (module.wq(target) * scale).view(b, n_t, module.heads, module.head_dim)
                k = (module.wk(source) * scale).view(b, n_s, module.heads, module.head_dim)
                v = (module.wv(source) * scale).view(b, n_s, module.heads, module.head_dim)
                if (
                    wire is not None
                    and target_coordinates is not None
                    and source_coordinates is not None
                ):
                    q, k = wire(q, k, target_coordinates, source_coordinates)
                logits = torch.einsum("bihd,bjhd->bhij", q, k) * module._logit_scale
                if temp != 1.0:
                    logits = logits / temp
                supp = support.bool()
                weights = module._attention_weights(logits, supp)
                stats = attention_diversity_stats(weights, supp)
                if stats is not None:
                    probe.update(stats)
                # Mean support size over nonempty target rows.
                row_deg = supp.sum(dim=-1).to(dtype=torch.float32)
                nonempty = supp.any(dim=-1)
                if nonempty.any():
                    probe["support_size"] = float(row_deg[nonempty].mean().item())
                else:
                    probe["support_size"] = 0.0

                # Per-head pathology proxies (H,).
                def _rms_heads(t: torch.Tensor) -> torch.Tensor:
                    # t: (B, N, H, Dh) → (H,)
                    return t.float().pow(2).mean(dim=(0, 1, 3)).sqrt()

                probe["q_rms"] = _rms_heads(q)
                probe["k_rms"] = _rms_heads(k)
                probe["v_rms"] = _rms_heads(v)
                ctx = torch.einsum("bhij,bjhd->bihd", weights, v)
                probe["ctx_rms"] = ctx.float().pow(2).mean(dim=(0, 1, 3)).sqrt()
                # Combined WO output is not per-head; store a scalar broadcast.
                out_rms = float(out.float().pow(2).mean().sqrt().item())
                probe["out_rms"] = out.new_full((module.heads,), out_rms)

                supp_h = supp.unsqueeze(1)  # (B,1,T,S)
                logits_f = logits.float()
                # Mask unsupported entries when reporting logit extremes.
                neg = torch.finfo(torch.float32).min
                logits_m = logits_f.masked_fill(~supp_h, neg)
                # (B,H,T) → mean over nonempty → (H,)
                nonempty_h = nonempty.unsqueeze(1).expand_as(logits_m[..., 0])
                abs_logits = logits_m.abs()
                abs_logits = abs_logits.masked_fill(~supp_h, 0.0)

                def _mean_over_rows(x_bht: torch.Tensor) -> torch.Tensor:
                    out_h = []
                    for h in range(x_bht.size(1)):
                        m = nonempty_h[:, h]
                        out_h.append(
                            x_bht[:, h][m].mean() if m.any() else x_bht.new_zeros(())
                        )
                    return torch.stack(out_h)

                probe["logit_abs_max"] = abs_logits.amax(dim=(0, 2, 3))
                probe["logit_abs_mean"] = _mean_over_rows(
                    abs_logits.sum(dim=-1)
                    / supp_h.to(abs_logits.dtype).sum(dim=-1).clamp_min(1.0)
                )
                probe["n_nan_logits"] = (~torch.isfinite(logits_f) & supp_h).sum(
                    dim=(0, 2, 3)
                ).to(torch.float32)
                probe["n_inf_logits"] = (
                    torch.isinf(logits_f) & supp_h
                ).sum(dim=(0, 2, 3)).to(torch.float32)
                probe["n_nan_weights"] = (
                    ~torch.isfinite(weights.float()) & supp_h
                ).sum(dim=(0, 2, 3)).to(torch.float32)
                # Softmax row mass on nonempty supported rows (should be ≈1).
                row_sum = (weights.float() * supp_h.to(torch.float32)).sum(dim=-1)
                mass_err = (row_sum - 1.0).abs()
                mass_err = mass_err.masked_fill(~nonempty_h, 0.0)
                probe["row_mass_err_max"] = mass_err.amax(dim=(0, 2))
                probe["row_mass_err_mean"] = _mean_over_rows(mass_err)
                probe["logit_temperature"] = float(temp)
                probe["n_nan_out"] = float((~torch.isfinite(out)).sum().item())
                probe["n_inf_out"] = float(torch.isinf(out.float()).sum().item())
        return out

    def _mlp(self, h: torch.Tensor, w1: nn.Module, w2: nn.Module) -> torch.Tensor:
        return w2(torch.nn.functional.gelu(w1(self.ln(h)) * self._mlp_w1_scale)) * self._mlp_w2_scale

    @staticmethod
    def _mask_mean(tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return tokens.mean(dim=1)
        weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _pool_weights(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor | None:
        if self.normalization != "energy-weights":
            return mask
        # Energy channel as weights; fall back to mask.
        e = x[..., self.energy_index].clamp_min(0.0)
        if mask is not None:
            e = e * mask.to(dtype=e.dtype)
        return e

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
        edge_mask: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x (B,N,F); got {tuple(x.shape)}")
        if edge_x.dim() != 3:
            raise ValueError(f"Expected edge_x (B,M,F); got {tuple(edge_x.shape)}")
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
        m_11, m_21, m_12, m_22 = self._build_supports(s_bool, mask=mask)
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

        z_x = self.decoder_x(h_x) * self._readout_scale
        z_e = self.decoder_e(h_e) * self._readout_scale
        if self.readout_mode == "node":
            # Stock node mix: (z_X + S^T z_E) / √2.
            s_f = s_bool.to(dtype=z_e.dtype)
            z_from_e = torch.bmm(s_f.transpose(1, 2), z_e)
            out = (z_x + z_from_e) * (0.5**0.5)
            if mask is not None:
                out = out * mask.to(dtype=out.dtype).unsqueeze(-1)
            return out

        # Graph readout: (pool_x + pool_e) / √2.
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
