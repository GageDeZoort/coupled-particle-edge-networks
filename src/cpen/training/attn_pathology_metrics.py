"""Row-aware attention pathology metrics (support-size stratified).

Used by the CAPEN probe path. Single-key rows (support size 1) make
``max_weight=1`` / ``entropy=0`` by construction — they must not drive
``collapse`` flags. Logit pathology uses per-row spread (max−min on the
supported keys), not a batch-global ``|logit|`` max (softmax is shift-invariant).
"""

from __future__ import annotations

import torch


def compute_attention_probe_metrics(
    weights: torch.Tensor,
    support: torch.Tensor,
    logits: torch.Tensor,
) -> dict[str, torch.Tensor | float]:
    """Per-head metrics from ``(B,H,T,S)`` weights / logits and ``(B,T,S)`` support.

    Returns tensors of shape ``(H,)`` unless noted as scalar float.
    """
    if weights.numel() == 0:
        return {}
    w = weights.float()
    logits_f = logits.float()
    supp = support.bool()
    b, h, t, s = w.shape
    supp_h = supp.unsqueeze(1)  # (B,1,T,S)
    row_deg = supp.sum(dim=-1).to(dtype=torch.float32)  # (B,T)
    nonempty = row_deg > 0  # (B,T)

    # Renormalize on support (defensive; Softmax path should already).
    w = w * supp_h.to(dtype=w.dtype)
    row = w.sum(dim=-1, keepdim=True)
    w = torch.where(row > 0, w / row.clamp_min(1e-12), w)

    log_w = torch.where(w > 0, w.clamp_min(1e-12).log(), torch.zeros_like(w))
    ent = -(w * log_w).sum(dim=-1)  # (B,H,T) nats
    n_src = row_deg.unsqueeze(1).clamp_min(1.0)  # (B,1,T)
    ent_norm = (ent / n_src.log().clamp_min(1e-12)).clamp(0.0, 1.0)
    mx = w.max(dim=-1).values  # (B,H,T)

    # Per-row logit spread on supported keys (softmax-invariant).
    neg = torch.finfo(torch.float32).min
    logits_m = logits_f.masked_fill(~supp_h, neg)
    logits_lo = logits_f.masked_fill(~supp_h, torch.finfo(torch.float32).max)
    row_max = logits_m.amax(dim=-1)  # (B,H,T)
    row_min = logits_lo.amin(dim=-1)
    spread = (row_max - row_min).masked_fill(~nonempty.unsqueeze(1), 0.0)

    # Self-as-top key (square attentions only).
    self_top = None
    if t == s:
        top = w.argmax(dim=-1)  # (B,H,T)
        idx = torch.arange(t, device=w.device).view(1, 1, t).expand(b, h, t)
        self_top = ((top == idx) & nonempty.unsqueeze(1)).to(dtype=torch.float32)

    buckets = {
        "k1": row_deg == 1,
        "k2": row_deg == 2,
        "kge3": row_deg >= 3,
        "kge2": row_deg >= 2,  # multi-key: used for collapse
        "all": nonempty,
    }

    def _mean_heads(x_bht: torch.Tensor, row_mask_bt: torch.Tensor) -> torch.Tensor:
        out = []
        for hi in range(h):
            m = row_mask_bt  # (B,T)
            if m.any():
                out.append(x_bht[:, hi][m].mean())
            else:
                out.append(x_bht.new_zeros(()))
        return torch.stack(out)

    def _max_heads(x_bht: torch.Tensor, row_mask_bt: torch.Tensor) -> torch.Tensor:
        out = []
        for hi in range(h):
            m = row_mask_bt
            if m.any():
                out.append(x_bht[:, hi][m].max())
            else:
                out.append(x_bht.new_zeros(()))
        return torch.stack(out)

    def _frac_rows(row_mask_bt: torch.Tensor) -> float:
        n = float(nonempty.sum().item())
        if n <= 0:
            return 0.0
        return float((row_mask_bt & nonempty).sum().item()) / n

    out: dict[str, torch.Tensor | float] = {
        # Legacy all-row means (includes trivial k=1).
        "entropy": _mean_heads(ent_norm, nonempty),
        "eff_width": _mean_heads(torch.exp(ent).clamp_max(n_src), nonempty),
        "max_weight": _mean_heads(mx, nonempty),
        # Multi-key (≥2) — collapse must use these.
        "entropy_multikey": _mean_heads(ent_norm, buckets["kge2"]),
        "max_weight_multikey": _mean_heads(mx, buckets["kge2"]),
        "n_rows_multikey": float(buckets["kge2"].sum().item()),
        # Support-size stratified.
        "frac_rows_k1": _frac_rows(buckets["k1"]),
        "frac_rows_k2": _frac_rows(buckets["k2"]),
        "frac_rows_kge3": _frac_rows(buckets["kge3"]),
        "entropy_k1": _mean_heads(ent_norm, buckets["k1"]),
        "entropy_k2": _mean_heads(ent_norm, buckets["k2"]),
        "entropy_kge3": _mean_heads(ent_norm, buckets["kge3"]),
        "max_weight_k1": _mean_heads(mx, buckets["k1"]),
        "max_weight_k2": _mean_heads(mx, buckets["k2"]),
        "max_weight_kge3": _mean_heads(mx, buckets["kge3"]),
        # Logit spread (per-row max−min).
        "logit_spread_mean": _mean_heads(spread, nonempty),
        "logit_spread_max": _max_heads(spread, nonempty),
        "logit_spread_mean_multikey": _mean_heads(spread, buckets["kge2"]),
        "logit_spread_max_multikey": _max_heads(spread, buckets["kge2"]),
        # Keep absolute extremes for diagnostics only (not for explode flag).
        "logit_abs_max": logits_f.abs().masked_fill(~supp_h, 0.0).amax(dim=(0, 2, 3)),
        "support_size": float(row_deg[nonempty].mean().item()) if nonempty.any() else 0.0,
        "n_rows_nonempty": float(nonempty.sum().item()),
    }
    if self_top is not None:
        out["self_top_frac"] = _mean_heads(self_top, nonempty)
        out["self_top_frac_multikey"] = _mean_heads(self_top, buckets["kge2"])
    else:
        out["self_top_frac"] = w.new_full((h,), float("nan"))
        out["self_top_frac_multikey"] = w.new_full((h,), float("nan"))
    return out


__all__ = ["compute_attention_probe_metrics"]
