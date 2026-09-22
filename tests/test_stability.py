"""Feature-norm probe for CPEN / CAPEN μP stability diagnostics."""

from __future__ import annotations

import math

import torch

from cpen.models.cpen import CPEN
from cpen.training.stability import (
    build_stability_model,
    disable_probe,
    enable_probe,
    forward_probe,
    last_layer_stage,
    masked_msq,
    run_one_architecture,
    _ce_step,
)


def _toy_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    n, m = 4, 2
    x = torch.randn(2, n, 7)
    edge_x = torch.randn(2, m, 4)
    incidence = torch.zeros(2, m, n, dtype=torch.bool)
    incidence[:, 0, 0] = incidence[:, 0, 1] = True
    incidence[:, 1, 1] = incidence[:, 1, 2] = True
    mask = torch.ones(2, n, dtype=torch.bool)
    z = torch.rand(2, n)
    z = z / z.sum(dim=-1, keepdim=True)
    y = torch.tensor([0, 1], dtype=torch.long)
    return {
        "x": x,
        "edge_x": edge_x,
        "incidence": incidence,
        "mask": mask,
        "z": z,
        "y": y,
    }


def test_masked_msq_unit_tokens() -> None:
    z = torch.ones(1, 3, 5)
    mask = torch.tensor([[True, True, False]])
    assert math.isclose(masked_msq(z, mask), 1.0, rel_tol=1e-6)


def test_cpen_probe_stages() -> None:
    model = CPEN(
        n_features=7,
        n_edge_features=4,
        out_dim=2,
        depth=2,
        width=16,
        operators="incidence",
        residual_structure="transformer-like",
        layer_norm=True,
        operator_backend="dense",
    )
    batch = _toy_batch()
    snaps, logits = forward_probe(model, batch)
    assert logits.shape == (2, 2)
    assert set(snaps) == {"encode", "layer0", "layer1", "decode"}
    assert snaps["decode"].h_x.shape[-1] == 2
    assert model._feature_probe is None
    rho = masked_msq(snaps["encode"].h_x, snaps["encode"].node_mask)
    assert math.isfinite(rho)
    assert 0.01 < rho < 100.0


def test_run_one_architecture_adam_moves_features() -> None:
    batch = _toy_batch()
    df = run_one_architecture(
        "cpen",
        depth=2,
        width=16,
        batch=batch,
        eta_0=0.25,
        n_adam_steps=2,
        seed=0,
        device=torch.device("cpu"),
    )
    last = df.loc[df["stage"].eq(last_layer_stage(2)) & df["token"].eq("particle")]
    assert len(last) == 1
    assert float(last["dmsq_1"].iloc[0]) > 0.0
    assert math.isfinite(float(last["msq_init"].iloc[0]))
    decode = df.loc[df["stage"].eq("decode") & df["token"].eq("particle")]
    assert len(decode) == 1
    assert math.isfinite(float(decode["msq_init"].iloc[0]))


def test_build_capen_and_probe() -> None:
    batch = _toy_batch()
    model = build_stability_model(
        "capen",
        depth=2,
        width=16,
        n_features=7,
        n_edge_features=4,
        heads=4,
        seed=0,
    )
    snaps, logits = forward_probe(model, batch)
    assert logits.shape == (2, 2)
    assert "encode" in snaps and "layer1" in snaps and "decode" in snaps
    assert snaps["decode"].h_x.shape[-1] == 2
    assert model._feature_probe is None


def test_ce_step_does_not_grow_probe() -> None:
    batch = _toy_batch()
    model = build_stability_model(
        "cpen",
        depth=2,
        width=16,
        n_features=7,
        n_edge_features=4,
        seed=0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    enable_probe(model)
    _ce_step(model, batch, opt)
    _ce_step(model, batch, opt)
    assert model._feature_probe is None
    disable_probe(model)


def test_decode_snap_reconstructs_model_logits() -> None:
    """ℓ=L+1 must be the model's own readout, not a per-token decode.

    Both families read out ``(z_X + z_E)/√2`` after pooling (CPEN α-pools with
    the energy weights, CAPEN mask-means), so the streams are separable only if
    the reconstruction is exact.
    """
    batch = _toy_batch()
    for family in ("cpen", "capen", "capen-llama"):
        model = build_stability_model(
            family,
            depth=2,
            width=16,
            n_features=7,
            n_edge_features=4,
            heads=4,
            seed=0,
        )
        snaps, logits = forward_probe(model, batch)
        assert "decode" in snaps, family
        snap = snaps["decode"]
        assert snap.h_x.shape == (2, 1, 2), (family, snap.h_x.shape)
        recon = (snap.h_x[:, 0, :] + snap.h_e[:, 0, :]) / math.sqrt(2.0)
        assert torch.allclose(recon, logits, atol=1e-5), family


def test_last_layer_frame_keeps_encoder_first_last_decode() -> None:
    import pandas as pd

    from cpen.training.stability import last_layer_frame

    df = pd.DataFrame(
        {
            "family": ["cpen"] * 8,
            "depth": [2] * 8,
            "width": [16] * 8,
            "stage": [
                "encode",
                "encode",
                "layer0",
                "layer0",
                "layer1",
                "layer1",
                "decode",
                "decode",
            ],
            "token": ["particle", "edge"] * 4,
        }
    )
    kept = last_layer_frame(df)
    assert set(kept["stage"]) == {"encode", "layer0", "layer1", "decode"}
    assert set(kept["stage_kind"]) == {"encode", "first", "last", "decode"}


def test_plot_stability_vs_params_grid_and_token_legend() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import pandas as pd

    from cpen.training.stability import plot_stability_vs_params

    rows = []
    for batch_id in (0, 1, 2):
        for n_params, depth, width in ((1_000, 2, 16), (8_000, 4, 32)):
            last = f"layer{depth - 1}"
            for family in ("cpen", "capen"):
                for stage in ("encode", "layer0", last, "decode"):
                    for token in ("particle", "edge"):
                        rows.append(
                            {
                                "family": family,
                                "depth": depth,
                                "width": width,
                                "n_params": n_params,
                                "stage": stage,
                                "token": token,
                                "n_adam_steps": 3,
                                "batch_id": batch_id,
                                "msq_init": 1.0 + 0.05 * batch_id,
                                "msq_1": 1.05 + 0.05 * batch_id,
                                "dmsq_1": 0.01 + 0.002 * batch_id,
                                "msq_T": 1.1 + 0.05 * batch_id,
                                "dmsq_T": 0.02 + 0.002 * batch_id,
                            }
                        )
    figs = plot_stability_vs_params(
        pd.DataFrame(rows),
        n_adam_steps=3,
        stem=None,
        show=False,
    )
    assert isinstance(figs, list) and len(figs) == 2
    fig = figs[0]
    assert fig.get_axes()[0].get_gridspec().get_geometry() == (3, 3)
    legend = fig.legends[0]
    labels = [t.get_text() for t in legend.get_texts()]
    assert "particle" in labels and "edge" in labels
    assert r"$\rho$" in labels and r"$\Delta\rho$" in labels
    assert r"$\ell=1$" in labels and r"$\ell=L$" in labels
    n_err = sum(
        1 for ax in fig.axes for c in ax.containers if type(c).__name__ == "ErrorbarContainer"
    )
    assert n_err > 0
    matplotlib.pyplot.close("all")


def _pascal_toy_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "x": torch.randn(2, 4, 14),
        "edge_x": torch.randn(2, 3, 2),
        "y": torch.tensor([[0, 0, 1, 2], [3, 3, 3, 4]], dtype=torch.long),
        "mask": torch.ones(2, 4, dtype=torch.bool),
        "edge_mask": torch.ones(2, 3, dtype=torch.bool),
        "n_edges": torch.tensor([3, 3], dtype=torch.int32),
        "incidence_node": torch.tensor(
            [[0, 1, 1, 2, 2, 3], [0, 1, 1, 2, 2, 3]], dtype=torch.int16
        ),
        "incidence_edge": torch.tensor(
            [[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]], dtype=torch.int16
        ),
        "incidence_nnz": torch.tensor([6, 6], dtype=torch.int32),
        "node_degree_inv": torch.tensor(
            [[1.0, 0.5, 0.5, 1.0], [1.0, 0.5, 0.5, 1.0]]
        ),
        "edge_degree_inv": torch.full((2, 3), 0.5),
    }


def test_pascal_node_edge_decode_and_adam() -> None:
    from cpen.training.stability import run_one_architecture

    batch = _pascal_toy_batch()
    df = run_one_architecture(
        "cpen",
        depth=2,
        width=16,
        batch=batch,
        eta_0=0.25,
        n_adam_steps=2,
        seed=0,
        device=torch.device("cpu"),
        objective="node+edge",
        out_dim=21,
        batch_id=0,
    )
    decode = df.loc[df["stage"].eq("decode") & df["token"].eq("particle")]
    assert len(decode) == 1
    assert math.isfinite(float(decode["msq_init"].iloc[0]))
    first = df.loc[df["stage"].eq("layer0") & df["token"].eq("particle")]
    last = df.loc[df["stage"].eq("layer1") & df["token"].eq("particle")]
    assert len(first) == 1 and len(last) == 1
    assert float(last["dmsq_1"].iloc[0]) > 0.0
    assert int(decode["batch_id"].iloc[0]) == 0
    assert str(decode["objective"].iloc[0]) == "node+edge"


def test_pascal_decode_snap_is_tokenwise() -> None:
    from cpen.training.stability import build_stability_model, forward_probe

    batch = _pascal_toy_batch()
    model = build_stability_model(
        "cpen",
        depth=2,
        width=16,
        n_features=14,
        n_edge_features=2,
        out_dim=21,
        readout_mode="node+edge",
        seed=0,
    )
    snaps, logits = forward_probe(model, batch)
    assert isinstance(logits, tuple) and len(logits) == 2
    assert logits[0].shape == (2, 4, 21)
    assert logits[1].shape == (2, 3, 2)
    assert "decode" in snaps
    assert snaps["decode"].h_x.shape == (2, 4, 21)
    assert snaps["decode"].h_e.shape == (2, 3, 2)


def test_ladder_accepts_batch_list() -> None:
    from cpen.training.stability import run_stability_ladder

    batches = [_toy_batch(), _toy_batch()]
    batches[1]["x"] = batches[1]["x"] + 0.1
    df = run_stability_ladder(
        batches,
        families=("cpen",),
        archs=((2, 16),),
        eta_0=0.25,
        n_adam_steps=1,
        seed=0,
        device=torch.device("cpu"),
        quiet=True,
        objective="graph",
    )
    assert set(df["batch_id"]) == {0, 1}
    assert df["stage"].eq("decode").any()
