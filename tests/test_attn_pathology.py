"""Unit tests for attention-head pathology classification."""

from __future__ import annotations

import torch

from cpen.models.capen import CAPEN
from cpen.training.attn_pathology import (
    PathologyThresholds,
    classify_attention_entry,
    classify_feature_probe,
    summarize_findings,
)


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


def test_classify_detects_collapse_and_nan():
    n_heads = 3
    entry = {
        "layer": 0,
        "relation": "11",
        "entropy": torch.tensor([0.1, 0.99, 0.5]),
        "eff_width": torch.tensor([1.1, 5.0, 2.0]),
        "max_weight": torch.tensor([0.99, 0.2, 0.5]),
        "q_rms": torch.tensor([1.0, 1.0, 1e-9]),
        "k_rms": torch.tensor([1.0, 1.0, 1.0]),
        "v_rms": torch.tensor([1.0, 1.0, 1.0]),
        "ctx_rms": torch.tensor([1.0, 1.0, 1.0]),
        "logit_abs_max": torch.tensor([1.0, 1.0, 80.0]),
        "logit_abs_mean": torch.tensor([0.5, 0.5, 10.0]),
        "n_nan_logits": torch.tensor([0.0, 2.0, 0.0]),
        "n_inf_logits": torch.zeros(n_heads),
        "n_nan_weights": torch.zeros(n_heads),
        "row_mass_err_max": torch.tensor([0.0, 0.0, 0.05]),
        "row_mass_err_mean": torch.zeros(n_heads),
        "support_size": 6.0,
        "n_nan_out": 0.0,
        "n_inf_out": 0.0,
    }
    findings = classify_attention_entry(entry)
    assert len(findings) == 3
    assert "collapse" in findings[0].flags
    assert "nan_logits" in findings[1].flags
    assert "near_uniform" in findings[1].flags
    assert "dead_q" in findings[2].flags
    assert "logit_explode" in findings[2].flags
    assert "row_mass_bad" in findings[2].flags
    summary = summarize_findings(findings)
    assert summary["n_heads_flagged"] == 3
    assert summary["flag_counts"]["collapse"] == 1


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
        assert "logit_abs_max" in entry
        assert "n_nan_logits" in entry
        assert "row_mass_err_max" in entry
        assert float(entry["n_nan_logits"].sum()) == 0.0
        assert float(entry["row_mass_err_max"].max()) < 1e-4
        findings = classify_attention_entry(
            entry,
            thresholds=PathologyThresholds(
                # Fresh init can look near-uniform; don't trip on that here.
                uniform_entropy=1.01,
                collapse_max_weight=1.01,
            ),
        )
        for f in findings:
            assert "nan_logits" not in f.flags
            assert "row_mass_bad" not in f.flags

    feat_rows = classify_feature_probe(feat)
    assert feat_rows
    assert all("rms_x" in r for r in feat_rows)
