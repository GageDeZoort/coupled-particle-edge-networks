#!/usr/bin/env python3
"""Load a pretrained CAPEN-Llama-att ckpt and probe attention-head pathologies.

Default target is the frozen L4/D256 JetClass cosine init used for TopTag FT:

  /scratch/.../toptagging_finetune/init_ckpts/L4D256-cosine-s781250-frozen.ckpt

Builds live star+kNN graphs (k★=7, R★=0.1, noloop) on a **partial HDF5 slice**
of TopTagging val (no full-file RAM load), records per-head entropy / max-weight /
QKV RMS / logit extremes / NaN counts, and writes CSV + JSON summaries.

Examples
--------
Login-node smoke::

  python scans/jets/probe_attn_pathology.py --device cpu --n-batches 2 --batch-size 2

GPU node (recommended)::

  python scans/jets/probe_attn_pathology.py --device cuda --n-batches 16 --batch-size 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cpen.apps.jets.part_kin import N_PART_KIN_FEATURES
from cpen.apps.jets.toptagging import (
    MAX_NUM_PARTICLES,
    N_PARTICLE_FEATURES,
    preprocess_particle_features,
    raw_hdf5_path,
)
from cpen.graphs.graph_star import PART_INTERACTION_PREPROCESS
from cpen.lit_models.lit_cpen_toptagging import LitCPENTopTagging
from cpen.training.attn_pathology import (
    DEFAULT_THRESHOLDS,
    PathologyThresholds,
    classify_attention_entry,
    classify_feature_probe,
    disable_attention_probe,
    disable_feature_probe,
    enable_attention_probe,
    enable_feature_probe,
    findings_to_rows,
    summarize_findings,
)
from scans.common.sweep_common import build_model, jetclass_live_graph_kwargs


DEFAULT_CKPT = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune/"
    "init_ckpts/L4D256-cosine-s781250-frozen.ckpt"
)
DEFAULT_OUT = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune/"
    "attn_pathology"
)
DEFAULT_DATA = Path("/scratch/gpfs/BHANIN/jdezoort/datasets/toptagging")


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-batches", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--knn-k", type=int, default=7)
    p.add_argument("--star-r", type=float, default=0.1)
    p.add_argument(
        "--num-particles",
        type=int,
        default=80,
        help="Pad/truncate (pretrain/FT use 80; HDF5 stores up to 200).",
    )
    p.add_argument(
        "--collapse-max-weight",
        type=float,
        default=DEFAULT_THRESHOLDS.collapse_max_weight,
    )
    p.add_argument(
        "--logit-abs-max",
        type=float,
        default=DEFAULT_THRESHOLDS.logit_abs_max,
    )
    p.add_argument("--tag", default="L4D256-cosine", help="Filename stem tag")
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Skip HDF5; use random jets (unit-test / offline smoke).",
    )
    return p.parse_args()


def _load_val_slice(
    data_root: Path,
    *,
    n_jets: int,
    num_particles: int,
    seed: int,
) -> list[dict[str, torch.Tensor]]:
    """Partial-read TopTagging val.h5 (contiguous head + shuffle) via pandas.

    Avoids ``RawTopTaggingSplit``'s full-file RAM load. Uses a contiguous
    ``start:stop`` window then subsamples — enough for pathology probes.
    """
    path = raw_hdf5_path(str(data_root), "val")
    if not path.is_file():
        raise FileNotFoundError(path)
    need = int(n_jets)
    # Read a modest window (4× need, capped) so we can shuffle without scanning
    # all 403k rows.
    window = min(max(need * 4, need), 4096)
    df = pd.read_hdf(path, key="table", start=0, stop=window)
    data = np.asarray(df)
    rng = np.random.default_rng(seed)
    if data.shape[0] > need:
        pick = rng.choice(data.shape[0], size=need, replace=False)
        data = data[pick]
    total_pf = MAX_NUM_PARTICLES * N_PARTICLE_FEATURES
    samples: list[dict[str, torch.Tensor]] = []
    for row in data:
        particles = row[:total_pf].reshape(MAX_NUM_PARTICLES, N_PARTICLE_FEATURES)[
            :num_particles
        ].astype(np.float32)
        label = int(row[-1])
        x_raw, x_model, mask = preprocess_particle_features(
            torch.from_numpy(particles)
        )
        samples.append(
            {
                "x": x_model,
                "x_raw": x_raw,
                "mask": mask.to(dtype=torch.bool),
                "y": torch.tensor(label, dtype=torch.long),
            }
        )
    return samples


def _synthetic_jets(n_jets: int, num_particles: int, seed: int) -> list[dict[str, torch.Tensor]]:
    rng = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(n_jets):
        # Rough positive-energy four-vectors; enough for graph + forward.
        px = torch.randn(num_particles, generator=rng)
        py = torch.randn(num_particles, generator=rng)
        pz = torch.randn(num_particles, generator=rng)
        e = (px.pow(2) + py.pow(2) + pz.pow(2)).sqrt() + 0.1
        particles = torch.stack([e, px, py, pz], dim=-1)
        # Zero-pad a few trailing slots to exercise masks.
        n_live = max(8, num_particles - 10)
        particles[n_live:] = 0
        x_raw, x_model, mask = preprocess_particle_features(particles)
        samples.append(
            {
                "x": x_model,
                "x_raw": x_raw,
                "mask": mask.to(dtype=torch.bool),
                "y": torch.tensor(0, dtype=torch.long),
            }
        )
    return samples


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "x": torch.stack([b["x"] for b in batch], dim=0),
        "x_raw": torch.stack([b["x_raw"] for b in batch], dim=0),
        "mask": torch.stack([b["mask"] for b in batch], dim=0),
        "y": torch.stack([b["y"] for b in batch], dim=0),
    }


def _make_args(cli: argparse.Namespace):
    """Minimal namespace for build_model / live-graph kwargs."""
    args = argparse.Namespace(
        model="capen-llama-att",
        operators="incidence",
        normalization="uniform",
        energy_index=0,
        optimizer="adam",
        residual_structure="transformer-like",
        attention_normalization="gamma",
        layer_norm=True,
        dropout=0.0,
        readout_mode="graph",
        dataset="toptagging",
        toptagging_live=True,
        use_wire=False,
        use_rope=False,
        identity_m22=False,
        incidence_m22=False,
        all_edge_m22=False,
        hyperedge_only=False,
        uniform_output_pool=False,
        live_star_kin7=True,
        no_star_hyperedges=False,
        star_radius=float(cli.star_r),
        live_knn_k=int(cli.knn_k),
        live_knn_include_self=False,
        live_star_seed_min_pt=0.0,
        live_edge_features="part-interaction",
        jetclass_edge_features="part-interaction",
        part_interaction_preprocess=PART_INTERACTION_PREPROCESS.get(
            float(cli.star_r), PART_INTERACTION_PREPROCESS[0.1]
        ),
        gamma_11=1.0,
        gamma_12=1.0,
        gamma_21=1.0,
        gamma_22=1.0,
        corr=1.0,
        n_edge_features=7,
    )
    return args


def main() -> None:
    cli = _cli()
    device = torch.device(cli.device)
    ckpt = cli.ckpt.expanduser().resolve()
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    torch.manual_seed(cli.seed)
    n_jets = int(cli.n_batches) * int(cli.batch_size)
    if cli.synthetic:
        samples = _synthetic_jets(n_jets, cli.num_particles, cli.seed)
        data_note = "synthetic"
    else:
        samples = _load_val_slice(
            cli.data_root,
            n_jets=n_jets,
            num_particles=cli.num_particles,
            seed=cli.seed,
        )
        data_note = f"toptagging-val-slice n={len(samples)}"

    args = _make_args(cli)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = payload["state_dict"]
    step = int(payload.get("global_step", 0) or 0)
    args.gamma_11 = float(sd["model.attn_gamma_11"])
    args.gamma_12 = float(sd["model.attn_gamma_12"])
    args.gamma_21 = float(sd["model.attn_gamma_21"])
    args.gamma_22 = float(sd["model.attn_gamma_22"])
    out_dim = int(sd["model.output_pool.class_tokens"].shape[0])

    model = build_model(
        args,
        n_features=N_PART_KIN_FEATURES,
        n_edge_features=7,
        out_dim=out_dim,
        depth=int(cli.depth),
        width=int(cli.width),
        heads=int(cli.heads),
    )
    live = jetclass_live_graph_kwargs(args)
    lit = LitCPENTopTagging(
        model=model,
        eta_0=0.1,
        model_name="capen-llama-att",
        optimizer="adam",
        scheduler="none",
        dataset="toptagging",
        depth=int(cli.depth),
        width=int(cli.width),
        heads=int(cli.heads),
        corr=1.0,
        **live,
    )
    missing, unexpected = lit.load_state_dict(sd, strict=False)
    lit.eval()
    lit.to(device)
    inner = lit._unwrap_model(lit.model)

    thr = PathologyThresholds(
        collapse_max_weight=float(cli.collapse_max_weight),
        logit_abs_max=float(cli.logit_abs_max),
    )

    all_findings = []
    feature_rows: list[dict] = []
    n_done = 0
    with torch.no_grad():
        for start in range(0, len(samples), cli.batch_size):
            if n_done >= cli.n_batches:
                break
            batch = _collate(samples[start : start + cli.batch_size])
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            batch = lit.on_after_batch_transfer(batch, 0)
            attn_probe = enable_attention_probe(inner)
            feat_probe = enable_feature_probe(inner)
            try:
                logits = lit.forward(batch["x"], **lit._model_kwargs(batch))
            finally:
                disable_attention_probe(inner)
                disable_feature_probe(inner)

            if not torch.isfinite(logits).all():
                print(f"[warn] batch {n_done}: non-finite logits")

            for entry in attn_probe:
                all_findings.extend(classify_attention_entry(entry, thresholds=thr))
            feature_rows.extend(classify_feature_probe(feat_probe, thresholds=thr))
            n_done += 1
            print(
                f"[probe] batch {n_done}/{cli.n_batches}  "
                f"N={int(batch['x'].size(0))} M={int(batch['edge_x'].size(1))}  "
                f"attn_entries={len(attn_probe)}"
            )

    head_rows = findings_to_rows(all_findings)
    df = pd.DataFrame(head_rows)
    if df.empty:
        raise SystemExit("no attention probe rows collected")

    group_cols = ["layer", "relation", "head"]
    metric_cols = [
        c
        for c in df.columns
        if c not in group_cols + ["flags", "n_flags"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    agg = df.groupby(group_cols, sort=True)[metric_cols].mean().reset_index()
    flag_union = (
        df.groupby(group_cols)["flags"]
        .apply(
            lambda s: "|".join(
                sorted({f for cell in s for f in str(cell).split("|") if f})
            )
        )
        .reset_index(name="flags")
    )
    agg = agg.merge(flag_union, on=group_cols)
    agg["n_flags"] = agg["flags"].apply(lambda s: 0 if not s else len(s.split("|")))

    summary = summarize_findings(all_findings)
    feat_flagged = [r for r in feature_rows if r["flags"]]

    out_dir = cli.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"attn_pathology_{cli.tag}_b{cli.n_batches}"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    agg.to_csv(csv_path, index=False)

    report = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ckpt": str(ckpt),
        "global_step": step,
        "device": str(device),
        "data": data_note,
        "n_batches": n_done,
        "batch_size": cli.batch_size,
        "graph": {"knn_k": cli.knn_k, "star_r": cli.star_r, "star_kin7": True},
        "model": {
            "depth": cli.depth,
            "width": cli.width,
            "heads": cli.heads,
            "out_dim": out_dim,
        },
        "load_state_dict": {
            "n_missing": len(missing),
            "n_unexpected": len(unexpected),
            "missing_sample": list(missing)[:12],
            "unexpected_sample": list(unexpected)[:12],
        },
        "thresholds": {
            "collapse_max_weight": thr.collapse_max_weight,
            "uniform_entropy": thr.uniform_entropy,
            "dead_rms": thr.dead_rms,
            "logit_abs_max": thr.logit_abs_max,
            "row_mass_err": thr.row_mass_err,
        },
        "gammas": {
            "11": args.gamma_11,
            "12": args.gamma_12,
            "21": args.gamma_21,
            "22": args.gamma_22,
        },
        "summary": summary,
        "feature_probe_flagged": feat_flagged[:32],
        "csv": str(csv_path),
    }
    json_path.write_text(json.dumps(report, indent=2, default=str) + "\n")

    print(
        json.dumps(
            {"summary": summary, "csv": str(csv_path), "json": str(json_path)},
            indent=2,
        )
    )
    if summary["n_heads_flagged"] == 0:
        print("[probe] no pathology flags tripped (at current thresholds)")
    else:
        print(
            f"[probe] flagged {summary['n_heads_flagged']}/{summary['n_heads_total']} "
            f"head-instances across batches; top flags: {summary['flag_counts']}"
        )
    print(f"[probe] wrote {csv_path}")
    print(f"[probe] wrote {json_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
