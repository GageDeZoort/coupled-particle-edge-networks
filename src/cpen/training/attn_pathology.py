"""Per-head attention pathology diagnostics for CAPEN trunks.

Enable ``model._attention_probe = []`` (and optionally ``_feature_probe``),
run a forward, then classify each ``(layer, relation, head)`` for numerical
pathologies: NaN/Inf, softmax collapse / near-uniform mass, dead QKV, logit
spread explosions, and broken row mass.

Collapse uses **multi-key** (support ≥ 2) max-weight only — single-key rows
are trivial (weight 1, entropy 0) and must not trip the flag. Logit pathology
uses per-row **spread** (max−min on supported keys), not a batch-global
``|logit|`` max (softmax is shift-invariant).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class PathologyThresholds:
    """Flags trip when a metric crosses these (tunable for sweeps)."""

    collapse_max_weight: float = 0.98
    uniform_entropy: float = 0.98
    uniform_min_support: float = 2.5
    dead_rms: float = 1e-6
    # Per-row logit spread (max−min on supported keys), not |logit| abs max.
    logit_spread_max: float = 40.0
    # Legacy absolute threshold kept for reporting only (not used for flags).
    logit_abs_max: float = 40.0
    row_mass_err: float = 1e-3
    feature_rms_hi: float = 50.0
    feature_rms_lo: float = 1e-6
    # Require this many multi-key rows before collapse can fire.
    collapse_min_multikey_rows: float = 1.0


DEFAULT_THRESHOLDS = PathologyThresholds()


@dataclass
class HeadFinding:
    layer: int
    relation: str
    head: int
    flags: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)


def _as_float_list(value: Any, *, n_heads: int) -> list[float]:
    if value is None:
        return [float("nan")] * n_heads
    if isinstance(value, torch.Tensor):
        t = value.detach().float().cpu().reshape(-1)
        if t.numel() == 1:
            return [float(t.item())] * n_heads
        if t.numel() != n_heads:
            raise ValueError(f"expected {n_heads} head stats, got {tuple(value.shape)}")
        return [float(x) for x in t.tolist()]
    if isinstance(value, (int, float)):
        return [float(value)] * n_heads
    raise TypeError(f"unsupported probe value type: {type(value)!r}")


_HEAD_METRIC_KEYS = (
    "entropy",
    "eff_width",
    "max_weight",
    "entropy_multikey",
    "max_weight_multikey",
    "entropy_k1",
    "entropy_k2",
    "entropy_kge3",
    "max_weight_k1",
    "max_weight_k2",
    "max_weight_kge3",
    "logit_spread_mean",
    "logit_spread_max",
    "logit_spread_mean_multikey",
    "logit_spread_max_multikey",
    "logit_abs_max",
    "logit_abs_mean",
    "self_top_frac",
    "self_top_frac_multikey",
    "q_rms",
    "k_rms",
    "v_rms",
    "ctx_rms",
    "n_nan_logits",
    "n_inf_logits",
    "n_nan_weights",
    "row_mass_err_max",
    "row_mass_err_mean",
)


def classify_attention_entry(
    entry: dict[str, Any],
    *,
    thresholds: PathologyThresholds = DEFAULT_THRESHOLDS,
) -> list[HeadFinding]:
    """Turn one probe dict (one layer×relation) into per-head findings."""
    entropy = entry.get("entropy")
    if entropy is None:
        return []
    if isinstance(entropy, torch.Tensor):
        n_heads = int(entropy.numel())
    else:
        raise TypeError("probe entry missing tensor entropy")

    metrics = {
        key: _as_float_list(entry.get(key), n_heads=n_heads) for key in _HEAD_METRIC_KEYS
    }
    support_size = float(entry.get("support_size", float("nan")))
    n_rows_multikey = float(entry.get("n_rows_multikey", float("nan")))
    frac_k1 = float(entry.get("frac_rows_k1", float("nan")))
    frac_k2 = float(entry.get("frac_rows_k2", float("nan")))
    frac_kge3 = float(entry.get("frac_rows_kge3", float("nan")))
    n_nan_out = float(entry.get("n_nan_out", 0.0) or 0.0)
    n_inf_out = float(entry.get("n_inf_out", 0.0) or 0.0)
    layer = int(entry.get("layer", -1))
    relation = str(entry.get("relation", "?"))

    findings: list[HeadFinding] = []
    thr = thresholds
    for h in range(n_heads):
        flags: list[str] = []
        row = {k: metrics[k][h] for k in metrics}
        row["support_size"] = support_size
        row["n_rows_multikey"] = n_rows_multikey
        row["frac_rows_k1"] = frac_k1
        row["frac_rows_k2"] = frac_k2
        row["frac_rows_kge3"] = frac_kge3
        row["n_nan_out"] = n_nan_out
        row["n_inf_out"] = n_inf_out

        if row["n_nan_logits"] > 0:
            flags.append("nan_logits")
        if row["n_inf_logits"] > 0:
            flags.append("inf_logits")
        if row["n_nan_weights"] > 0:
            flags.append("nan_weights")
        if n_nan_out > 0:
            flags.append("nan_out")
        if n_inf_out > 0:
            flags.append("inf_out")

        # Collapse: multi-key rows only (k≥2). Skip if no multi-key mass.
        mw_mk = row["max_weight_multikey"]
        if (
            n_rows_multikey >= thr.collapse_min_multikey_rows
            and mw_mk == mw_mk  # not NaN
            and mw_mk >= thr.collapse_max_weight
        ):
            flags.append("collapse")

        # Near-uniform: prefer multi-key entropy when available.
        ent_u = row["entropy_multikey"]
        if ent_u != ent_u:
            ent_u = row["entropy"]
        if ent_u >= thr.uniform_entropy and support_size >= thr.uniform_min_support:
            flags.append("near_uniform")

        if row["q_rms"] < thr.dead_rms:
            flags.append("dead_q")
        if row["k_rms"] < thr.dead_rms:
            flags.append("dead_k")
        if row["v_rms"] < thr.dead_rms:
            flags.append("dead_v")
        if row["ctx_rms"] < thr.dead_rms:
            flags.append("dead_ctx")

        # Logit explode: per-row spread (multi-key preferred).
        spread = row["logit_spread_max_multikey"]
        if spread != spread:
            spread = row["logit_spread_max"]
        if spread == spread and spread >= thr.logit_spread_max:
            flags.append("logit_explode")

        if row["row_mass_err_max"] > thr.row_mass_err:
            flags.append("row_mass_bad")
        # self_top_* is NaN on bipartite relations (T≠S) by design — not a fault.
        for key, val in row.items():
            if key.startswith("n_") or key.startswith("frac_") or key.startswith(
                "self_top"
            ):
                continue
            if val != val or val in (float("inf"), float("-inf")):
                flags.append(f"nonfinite_{key}")

        findings.append(
            HeadFinding(
                layer=layer,
                relation=relation,
                head=h,
                flags=sorted(set(flags)),
                metrics=row,
            )
        )
    return findings


def classify_feature_probe(
    probe: Sequence[tuple],
    *,
    thresholds: PathologyThresholds = DEFAULT_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Summarize residual-stream RMS / finiteness at each recorded stage."""
    rows: list[dict[str, Any]] = []
    thr = thresholds
    for item in probe:
        stage, h_x, h_e = item[0], item[1], item[2]
        mask = item[3] if len(item) > 3 else None
        hx = h_x.detach().float()
        he = h_e.detach().float()
        if mask is not None:
            w = mask.to(dtype=hx.dtype).unsqueeze(-1)
            rms_x = float(
                ((hx.pow(2) * w).sum() / w.sum().clamp_min(1.0) / hx.size(-1)).sqrt()
            )
        else:
            rms_x = float(hx.pow(2).mean().sqrt())
        rms_e = float(he.pow(2).mean().sqrt())
        flags: list[str] = []
        if not torch.isfinite(hx).all():
            flags.append("nonfinite_h_x")
        if not torch.isfinite(he).all():
            flags.append("nonfinite_h_e")
        if rms_x > thr.feature_rms_hi or rms_e > thr.feature_rms_hi:
            flags.append("rms_explode")
        if rms_x < thr.feature_rms_lo:
            flags.append("rms_vanish_x")
        if rms_e < thr.feature_rms_lo:
            flags.append("rms_vanish_e")
        rows.append(
            {
                "stage": str(stage),
                "rms_x": rms_x,
                "rms_e": rms_e,
                "flags": flags,
            }
        )
    return rows


def findings_to_rows(findings: Iterable[HeadFinding]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for f in findings:
        row = {
            "layer": f.layer,
            "relation": f.relation,
            "head": f.head,
            "flags": "|".join(f.flags),
            "n_flags": len(f.flags),
        }
        row.update(f.metrics)
        rows.append(row)
    return rows


def summarize_findings(findings: Sequence[HeadFinding]) -> dict[str, Any]:
    flag_counts: dict[str, int] = {}
    flagged_heads = 0
    for f in findings:
        if f.flags:
            flagged_heads += 1
        for flag in f.flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    worst = sorted(findings, key=lambda f: (-len(f.flags), f.layer, f.relation, f.head))
    return {
        "n_heads_total": len(findings),
        "n_heads_flagged": flagged_heads,
        "flag_counts": dict(sorted(flag_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "worst": [
            {
                "layer": f.layer,
                "relation": f.relation,
                "head": f.head,
                "flags": f.flags,
                "max_weight": f.metrics.get("max_weight"),
                "max_weight_multikey": f.metrics.get("max_weight_multikey"),
                "entropy": f.metrics.get("entropy"),
                "entropy_multikey": f.metrics.get("entropy_multikey"),
                "logit_spread_max": f.metrics.get("logit_spread_max"),
                "logit_spread_max_multikey": f.metrics.get("logit_spread_max_multikey"),
                "self_top_frac": f.metrics.get("self_top_frac"),
                "frac_rows_k1": f.metrics.get("frac_rows_k1"),
                "q_rms": f.metrics.get("q_rms"),
                "k_rms": f.metrics.get("k_rms"),
            }
            for f in worst[:16]
            if f.flags
        ],
    }


def enable_attention_probe(model: nn.Module) -> list:
    probe: list = []
    model._attention_probe = probe  # type: ignore[attr-defined]
    return probe


def disable_attention_probe(model: nn.Module) -> None:
    model._attention_probe = None  # type: ignore[attr-defined]


def enable_feature_probe(model: nn.Module) -> list:
    probe: list = []
    model._feature_probe = probe  # type: ignore[attr-defined]
    return probe


def disable_feature_probe(model: nn.Module) -> None:
    model._feature_probe = None  # type: ignore[attr-defined]


__all__ = [
    "PathologyThresholds",
    "DEFAULT_THRESHOLDS",
    "HeadFinding",
    "classify_attention_entry",
    "classify_feature_probe",
    "findings_to_rows",
    "summarize_findings",
    "enable_attention_probe",
    "disable_attention_probe",
    "enable_feature_probe",
    "disable_feature_probe",
]
