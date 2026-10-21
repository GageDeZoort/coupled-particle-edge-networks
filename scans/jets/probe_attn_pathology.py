#!/usr/bin/env python3
"""Load a pretrained CAPEN-Llama-att ckpt and probe attention-head pathologies.

Default data is **JetClass val** (kin7), not TopTagging / not isotropic synthetic
jets. Synthetic jets scatter over η–φ and produce mostly single-key hyperedges,
which trivially trip collapse (max_weight=1, entropy=0).

Metrics (see ``attn_pathology_metrics``):
  * entropy / max-weight stratified by support size (k=1, 2, ≥3)
  * collapse uses multi-key (k≥2) max-weight only
  * logit pathology uses per-row spread (max−min), not batch |logit| max
  * self_top_frac on square attentions (M11 / M22)

Examples
--------
  python scans/jets/probe_attn_pathology.py \\
    --ckpt .../last.ckpt --typed-edge-streams \\
    --jetclass-raw-root /projects/j/jdezoort/jetclass \\
    --n-jets 256 --batch-size 16 --device cuda
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

from cpen.apps.jets.jetclass_stream import JetClassSubsetDataset
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
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "attn_pathology"
)
DEFAULT_JETCLASS = Path("/projects/j/jdezoort/jetclass")
DEFAULT_TOPTAG = Path("/scratch/gpfs/BHANIN/jdezoort/datasets/toptagging")


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--jetclass-raw-root",
        type=Path,
        default=DEFAULT_JETCLASS,
        help="JetClass ROOT root (expects val_5M/). Default for paper probes.",
    )
    p.add_argument(
        "--toptagging-root",
        type=Path,
        default=DEFAULT_TOPTAG,
        help="Only used with --data toptagging (legacy).",
    )
    p.add_argument(
        "--data",
        choices=("jetclass", "toptagging", "synthetic"),
        default="jetclass",
        help="Probe jets source. Prefer jetclass (default).",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--n-jets",
        type=int,
        default=256,
        help="Number of validation jets (≥256 recommended).",
    )
    p.add_argument(
        "--n-batches",
        type=int,
        default=None,
        help="Override batch count; default = ceil(n_jets / batch_size).",
    )
    p.add_argument("--batch-size", type=int, default=16)
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
        help="Pad/truncate (JetClass typed pretrain uses 80).",
    )
    p.add_argument(
        "--collapse-max-weight",
        type=float,
        default=DEFAULT_THRESHOLDS.collapse_max_weight,
    )
    p.add_argument(
        "--logit-spread-max",
        type=float,
        default=DEFAULT_THRESHOLDS.logit_spread_max,
        help="Flag logit_explode when per-row (max−min) spread exceeds this.",
    )
    p.add_argument(
        "--logit-abs-max",
        type=float,
        default=DEFAULT_THRESHOLDS.logit_abs_max,
        help="Legacy; kept for report fields only (does not trip flags).",
    )
    p.add_argument("--tag", default="L4D256-cosine", help="Filename stem tag")
    p.add_argument(
        "--typed-edge-streams",
        action="store_true",
        dest="typed_edge_streams",
        help=(
            "Match --typed-edge-streams training: dual encoders + §1 routing "
            "(scans/jets/TYPED_EDGE_ATTENTION_TARGET.md)."
        ),
    )
    p.add_argument(
        "--synthetic",
        action="store_true",
        help="Alias for --data synthetic (offline smoke only; not for paper).",
    )
    return p.parse_args()


def _load_jetclass_val(
    data_root: Path,
    *,
    n_jets: int,
    num_particles: int,
) -> list[dict[str, torch.Tensor]]:
    """Class-balanced JetClass val subset (kin7) held in RAM."""
    ds = JetClassSubsetDataset(
        data_root,
        "val",
        n_jets=int(n_jets),
        num_particles=int(num_particles),
        feature_config="kin7",
    )
    samples: list[dict[str, torch.Tensor]] = []
    for i in range(len(ds)):
        item = ds[i]
        samples.append(
            {
                "x": item["x"],
                "x_raw": item["x_raw"],
                "mask": item["mask"].to(dtype=torch.bool),
                "y": item["y"].to(dtype=torch.long),
            }
        )
    return samples


def _load_toptag_val_slice(
    data_root: Path,
    *,
    n_jets: int,
    num_particles: int,
    seed: int,
) -> list[dict[str, torch.Tensor]]:
    """Legacy TopTagging val.h5 slice (prefer JetClass for paper probes)."""
    path = raw_hdf5_path(str(data_root), "val")
    if not path.is_file():
        raise FileNotFoundError(path)
    need = int(n_jets)
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


def _qk_weight_norms(state_dict: dict) -> dict[str, float]:
    """Frobenius norms of attn W_Q / W_K per relation×layer (from ckpt)."""
    out: dict[str, float] = {}
    for rel in ("11", "12", "21", "22"):
        for kind in ("wq", "wk"):
            key_prefix = f"model.attn_{rel}."
            # ModuleList: model.attn_11.0.wq.weight
            norms = []
            for k, v in state_dict.items():
                if f"attn_{rel}." in k and f".{kind}.weight" in k:
                    norms.append(float(v.detach().float().norm().item()))
            if norms:
                out[f"attn_{rel}_{kind}_norm_mean"] = float(np.mean(norms))
                out[f"attn_{rel}_{kind}_norm_max"] = float(np.max(norms))
                out[f"attn_{rel}_{kind}_norm_by_layer"] = norms
    return out


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
        typed_edge_streams=bool(getattr(cli, "typed_edge_streams", False)),
    )
    return args


def main() -> None:
    cli = _cli()
    device = torch.device(cli.device)
    ckpt = cli.ckpt.expanduser().resolve()
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    torch.manual_seed(cli.seed)
    data_mode = "synthetic" if cli.synthetic else str(cli.data)
    n_jets = int(cli.n_jets)
    if cli.n_batches is not None:
        n_jets = max(n_jets, int(cli.n_batches) * int(cli.batch_size))
    if data_mode == "synthetic":
        samples = _synthetic_jets(n_jets, cli.num_particles, cli.seed)
        data_note = f"synthetic n={len(samples)} (NOT for paper)"
    elif data_mode == "toptagging":
        samples = _load_toptag_val_slice(
            cli.toptagging_root,
            n_jets=n_jets,
            num_particles=cli.num_particles,
            seed=cli.seed,
        )
        data_note = f"toptagging-val-slice n={len(samples)}"
    else:
        samples = _load_jetclass_val(
            cli.jetclass_raw_root,
            n_jets=n_jets,
            num_particles=cli.num_particles,
        )
        data_note = f"jetclass-val-kin7 n={len(samples)}"

    n_batches = (
        int(cli.n_batches)
        if cli.n_batches is not None
        else max(1, (len(samples) + int(cli.batch_size) - 1) // int(cli.batch_size))
    )

    args = _make_args(cli)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = payload["state_dict"]
    step = int(payload.get("global_step", 0) or 0)
    args.gamma_11 = float(sd["model.attn_gamma_11"])
    args.gamma_12 = float(sd["model.attn_gamma_12"])
    args.gamma_21 = float(sd["model.attn_gamma_21"])
    args.gamma_22 = float(sd["model.attn_gamma_22"])
    out_dim = int(sd["model.output_pool.class_tokens"].shape[0])
    qk_norms = _qk_weight_norms(sd)

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
        logit_spread_max=float(cli.logit_spread_max),
        logit_abs_max=float(cli.logit_abs_max),
    )

    all_findings = []
    feature_rows: list[dict] = []
    n_done = 0
    with torch.no_grad():
        for start in range(0, len(samples), cli.batch_size):
            if n_done >= n_batches:
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
            n_edges = int(batch["edge_x"].size(1)) if batch.get("edge_x") is not None else -1
            if batch.get("edge_x_star") is not None and batch.get("edge_x_knn") is not None:
                n_edges = int(batch["edge_x_star"].size(1) + batch["edge_x_knn"].size(1))
            print(
                f"[probe] batch {n_done}/{n_batches}  "
                f"N={int(batch['x'].size(0))} M={n_edges}  "
                f"attn_entries={len(attn_probe)}"
                + (
                    " typed"
                    if bool(getattr(cli, "typed_edge_streams", False))
                    else ""
                )
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
    stem = f"attn_pathology_{cli.tag}_n{len(samples)}"
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    agg.to_csv(csv_path, index=False)

    # Relation-level rollup of stratified metrics (mean over heads×batches).
    rel_rollup: dict[str, dict[str, float]] = {}
    if not df.empty:
        for rel, g in df.groupby("relation"):
            rel_rollup[str(rel)] = {
                "frac_rows_k1": float(g["frac_rows_k1"].mean())
                if "frac_rows_k1" in g
                else float("nan"),
                "frac_rows_kge3": float(g["frac_rows_kge3"].mean())
                if "frac_rows_kge3" in g
                else float("nan"),
                "entropy_multikey": float(g["entropy_multikey"].mean())
                if "entropy_multikey" in g
                else float("nan"),
                "max_weight_multikey": float(g["max_weight_multikey"].mean())
                if "max_weight_multikey" in g
                else float("nan"),
                "logit_spread_max_multikey": float(g["logit_spread_max_multikey"].mean())
                if "logit_spread_max_multikey" in g
                else float("nan"),
                "self_top_frac": float(g["self_top_frac"].mean())
                if "self_top_frac" in g
                else float("nan"),
                "self_top_frac_multikey": float(g["self_top_frac_multikey"].mean())
                if "self_top_frac_multikey" in g
                else float("nan"),
                "q_rms": float(g["q_rms"].mean()) if "q_rms" in g else float("nan"),
                "k_rms": float(g["k_rms"].mean()) if "k_rms" in g else float("nan"),
            }

    report = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ckpt": str(ckpt),
        "global_step": step,
        "device": str(device),
        "data": data_note,
        "n_jets": len(samples),
        "n_batches": n_done,
        "batch_size": cli.batch_size,
        "graph": {
            "knn_k": cli.knn_k,
            "star_r": cli.star_r,
            "star_kin7": True,
            "typed_edge_streams": bool(getattr(cli, "typed_edge_streams", False)),
        },
        "model": {
            "depth": cli.depth,
            "width": cli.width,
            "heads": cli.heads,
            "out_dim": out_dim,
            "typed_edge_streams": bool(getattr(cli, "typed_edge_streams", False)),
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
            "logit_spread_max": thr.logit_spread_max,
            "logit_abs_max": thr.logit_abs_max,
            "row_mass_err": thr.row_mass_err,
        },
        "gammas": {
            "11": args.gamma_11,
            "12": args.gamma_12,
            "21": args.gamma_21,
            "22": args.gamma_22,
        },
        "qk_weight_norms": qk_norms,
        "relation_rollup": rel_rollup,
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
