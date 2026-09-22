"""Batch-based attention-γ estimation (no on-disk star/hier cache)."""

from __future__ import annotations

import torch
import pytest

from cpen.graphs.graph_star import build_star_radius_graph
from cpen.utils.attention_temperature import (
    estimate_attention_gammas_from_batches,
    materialize_batch_incidence,
)


def _toy_jet_batch(batch: int = 4, n: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.zeros(batch, n, 4)
    mask = torch.zeros(batch, n, dtype=torch.bool)
    for b in range(batch):
        n_active = 12 if b % 2 == 0 else 10
        mask[b, :n_active] = True
        for i in range(n_active):
            phi = 0.15 * i
            pt = 20.0 + i
            x[b, i, 1] = pt * torch.cos(torch.tensor(phi))
            x[b, i, 2] = pt * torch.sin(torch.tensor(phi))
            x[b, i, 3] = 5.0
            x[b, i, 0] = torch.sqrt(
                x[b, i, 1] ** 2 + x[b, i, 2] ** 2 + x[b, i, 3] ** 2
            )
    return x, mask


def test_materialize_live_star_from_x_raw() -> None:
    x_raw, mask = _toy_jet_batch()
    batch = {"x_raw": x_raw, "mask": mask, "y": torch.zeros(x_raw.size(0), dtype=torch.long)}
    out = materialize_batch_incidence(batch, live_star_radius=0.2)
    assert "incidence" in out
    assert out["incidence"].shape[-2:] == (x_raw.size(1), x_raw.size(1))
    assert "x_raw" not in out


def test_estimate_gammas_from_star_batches() -> None:
    x_raw, mask = _toy_jet_batch(batch=8, n=16)
    edge_x, incidence, _ = build_star_radius_graph(
        x_raw, radius=0.2, mask=mask, edge_features="logdot-dp"
    )
    batches = [
        {
            "x": torch.randn(4, 16, 7),
            "mask": mask[:4],
            "edge_x": edge_x[:4],
            "incidence": incidence[:4],
            "y": torch.zeros(4, dtype=torch.long),
        },
        {
            "x": torch.randn(4, 16, 7),
            "mask": mask[4:],
            "edge_x": edge_x[4:],
            "incidence": incidence[4:],
            "y": torch.zeros(4, dtype=torch.long),
        },
    ]
    gammas, summaries = estimate_attention_gammas_from_batches(
        batches,
        max_jets=6,
        all_to_all_particle_attention=True,
    )
    assert gammas.gamma_11 > 1.0
    assert gammas.gamma_12 > 0.0
    assert gammas.gamma_21 > 0.0
    assert gammas.gamma_22 > 0.0
    assert summaries["11"]["n_rows"] > 0
    # max_jets=6 → only first batch fully + 2 from second.
    assert summaries["11"]["n_rows"] <= 6 * 16


def test_estimate_gammas_from_x_raw_batches() -> None:
    x_raw, mask = _toy_jet_batch(batch=4, n=12)
    batches = [{"x_raw": x_raw, "mask": mask, "y": torch.zeros(4, dtype=torch.long)}]
    gammas, _ = estimate_attention_gammas_from_batches(
        batches,
        max_jets=4,
        live_star_radius=0.15,
        all_to_all_particle_attention=True,
    )
    assert gammas.gamma_11 >= 1.0
    assert gammas.gamma_12 >= 1.0


def test_estimate_gammas_m11_only_skips_star_graph() -> None:
    x_raw, mask = _toy_jet_batch(batch=4, n=12)
    batches = [{"x_raw": x_raw, "mask": mask, "y": torch.zeros(4, dtype=torch.long)}]
    gammas, summaries = estimate_attention_gammas_from_batches(
        batches,
        max_jets=4,
        all_to_all_particle_attention=True,
        m11_only=True,
    )
    # All-to-all M11 degree is the particle count (~10–12 in the toy jets).
    assert gammas.gamma_11 > 1.0
    assert summaries["11"]["n_rows"] > 0
    # No 12/21/22 supports → dummy γ = 1.
    assert gammas.gamma_12 == pytest.approx(1.0)
    assert gammas.gamma_21 == pytest.approx(1.0)
    assert gammas.gamma_22 == pytest.approx(1.0)
    assert summaries["12"]["n_rows"] == 0
