"""Unit tests for attention-head pathology classification + row metrics."""

from __future__ import annotations

import torch

from cpen.models.capen import CAPEN
from cpen.training.attn_pathology import (
    PathologyThresholds,
    classify_attention_entry,
    classify_feature_probe,
    summarize_findings,
)
from cpen.training.attn_pathology_metrics import compute_attention_probe_metrics


def _tiny_model(**kwargs):
    defaults = dict(
        n_features=3,
        n_edge_features=4,
        out_dim=2,
        depth=1,
        width=16,
        heads=4,
        readout_mode="graph",
    )
    defaults.update(kwargs)
    torch.manual_seed(1)
    return CAPEN(**defaults)


def _tiny_batch():
    torch.manual_seed(0)
    n, m = 4, 2
    x = torch.randn(1, n, 3)
    edge_x = torch.randn(1, m, 4)
    incidence = torch.zeros(1, m, n, dtype=torch.bool)
    incidence[0, 0, 0] = incidence[0, 0, 1] = True
    incidence[0, 1, 1] = incidence[0, 1, 2] = True
    mask = torch.ones(1, n, dtype=torch.bool)
    return {"x": x, "edge_x": edge_x, "incidence": incidence, "mask": mask}


def test_classify_uses_multikey_collapse_and_logit_spread():
    n_heads = 3
    entry = {
        "layer": 0,
        "relation": "11",
        "entropy": torch.tensor([0.1, 0.99, 0.5]),
        "eff_width": torch.tensor([1.1, 5.0, 2.0]),
        "max_weight": torch.tensor([0.99, 0.2, 0.5]),  # legacy all-row (incl k=1)
        # Multi-key: head0 not collapsed; head2 has huge spread.
        "max_weight_multikey": torch.tensor([0.5, 0.2, 0.5]),
        "entropy_multikey": torch.tensor([0.5, 0.99, 0.5]),
        "logit_spread_max": torch.tensor([1.0, 1.0, 80.0]),
        "logit_spread_max_multikey": torch.tensor([1.0, 1.0, 80.0]),
        "logit_abs_max": torch.tensor([100.0, 1.0, 1.0]),  # must NOT trip explode
        "logit_abs_mean": torch.tensor([0.5, 0.5, 10.0]),
        "q_rms": torch.tensor([1.0, 1.0, 1e-9]),
        "k_rms": torch.tensor([1.0, 1.0, 1.0]),
        "v_rms": torch.tensor([1.0, 1.0, 1.0]),
        "ctx_rms": torch.tensor([1.0, 1.0, 1.0]),
        "n_nan_logits": torch.tensor([0.0, 2.0, 0.0]),
        "n_inf_logits": torch.zeros(n_heads),
        "n_nan_weights": torch.zeros(n_heads),
        "row_mass_err_max": torch.tensor([0.0, 0.0, 0.05]),
        "row_mass_err_mean": torch.zeros(n_heads),
        "support_size": 6.0,
        "n_rows_multikey": 10.0,
        "n_nan_out": 0.0,
        "n_inf_out": 0.0,
    }
    findings = classify_attention_entry(entry)
    assert len(findings) == 3
    # Head0: legacy max_weight=0.99 but multikey=0.5 → no collapse.
    assert "collapse" not in findings[0].flags
    assert "nan_logits" in findings[1].flags
    assert "near_uniform" in findings[1].flags
    assert "dead_q" in findings[2].flags
    assert "logit_explode" in findings[2].flags
    assert "row_mass_bad" in findings[2].flags
    # Absolute |logit| alone must not flag head0.
    assert "logit_explode" not in findings[0].flags
    summary = summarize_findings(findings)
    assert summary["n_heads_flagged"] == 3


def test_single_key_rows_do_not_drive_collapse_metric():
    """k=1 rows → weight 1 / entropy 0; multikey stats exclude them."""
    # One query, two keys; second query has only self (k=1).
    # weights (B=1,H=1,T=2,S=2)
    w = torch.zeros(1, 1, 2, 2)
    w[0, 0, 0, 0] = 0.6
    w[0, 0, 0, 1] = 0.4
    w[0, 0, 1, 1] = 1.0  # single-key row
    supp = torch.tensor([[[True, True], [False, True]]])
    logits = torch.tensor([[[[1.0, 0.0], [0.0, 5.0]]]])  # (1,1,2,2)
    m = compute_attention_probe_metrics(w, supp, logits)
    assert float(m["frac_rows_k1"]) == 0.5
    assert float(m["max_weight_k1"]) == 1.0
    # Multi-key max weight is 0.6, not polluted by the trivial 1.0 row.
    assert abs(float(m["max_weight_multikey"]) - 0.6) < 1e-5
    # Collapse classify must use multikey.
    entry = {
        "layer": 0,
        "relation": "22",
        "entropy": m["entropy"],
        "max_weight": m["max_weight"],
        "max_weight_multikey": m["max_weight_multikey"],
        "entropy_multikey": m["entropy_multikey"],
        "logit_spread_max": m["logit_spread_max"],
        "logit_spread_max_multikey": m["logit_spread_max_multikey"],
        "q_rms": torch.ones(1),
        "k_rms": torch.ones(1),
        "v_rms": torch.ones(1),
        "ctx_rms": torch.ones(1),
        "n_nan_logits": torch.zeros(1),
        "n_inf_logits": torch.zeros(1),
        "n_nan_weights": torch.zeros(1),
        "row_mass_err_max": torch.zeros(1),
        "row_mass_err_mean": torch.zeros(1),
        "support_size": float(m["support_size"]),
        "n_rows_multikey": float(m["n_rows_multikey"]),
        "n_nan_out": 0.0,
        "n_inf_out": 0.0,
    }
    findings = classify_attention_entry(entry)
    assert "collapse" not in findings[0].flags


def test_self_top_frac_on_square_attention():
    w = torch.zeros(1, 1, 2, 2)
    w[0, 0, 0, 0] = 0.9
    w[0, 0, 0, 1] = 0.1
    w[0, 0, 1, 0] = 0.8  # top is other token
    w[0, 0, 1, 1] = 0.2
    supp = torch.ones(1, 2, 2, dtype=torch.bool)
    logits = torch.zeros(1, 1, 2, 2)
    m = compute_attention_probe_metrics(w, supp, logits)
    # Row0 self-top, row1 not → 0.5
    assert abs(float(m["self_top_frac"]) - 0.5) < 1e-5


def test_probe_pathology_fields_on_real_forward():
    model = _tiny_model()
    batch = _tiny_batch()
    model.eval()
    model._attention_probe = []
    model._feature_probe = []
    try:
        out = model(
            batch["x"], batch["edge_x"], batch["incidence"], mask=batch["mask"]
        )
    finally:
        probe = model._attention_probe
        feat = model._feature_probe
        model._attention_probe = None
        model._feature_probe = None

    assert torch.isfinite(out).all()
    assert len(probe) == 4  # 1 layer × 4 relations
    for entry in probe:
        assert entry["entropy"].shape == (4,)
        assert "q_rms" in entry
        assert "logit_spread_max" in entry
        assert "max_weight_multikey" in entry
        assert "frac_rows_k1" in entry
        assert "n_nan_logits" in entry
        assert "row_mass_err_max" in entry
        assert float(entry["n_nan_logits"].sum()) == 0.0
        assert float(entry["row_mass_err_max"].max()) < 1e-4
        findings = classify_attention_entry(
            entry,
            thresholds=PathologyThresholds(
                uniform_entropy=1.01,
                collapse_max_weight=1.01,
                logit_spread_max=1e9,
            ),
        )
        for f in findings:
            assert "nan_logits" not in f.flags
            assert "row_mass_bad" not in f.flags

    feat_rows = classify_feature_probe(feat)
    assert feat_rows
    assert all("rms_x" in r for r in feat_rows)
