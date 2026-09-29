#!/usr/bin/env python3
"""Re-evaluate graph-k sweep arms from best.ckpt on the fixed 25k val set.

Paper Plot 1 numbers must come from a clean checkpoint val pass, not the
mid-train val_check_interval=500 echoes (epoch-end @1954 re-logs @1500).

Examples:
  # Discover arms that have best.ckpt
  python scans/jets/eval_graph_k_ckpt.py --discover \\
    --manifest /scratch/.../logs/graph_k_val_eval/manifest.txt

  # Eval one run dir (Slurm array task)
  python scans/jets/eval_graph_k_ckpt.py \\
    --run-dir /scratch/.../sweep_lr/capen-llama-att_...knn10...R0p125 \\
    --out-csv paper/data/graph_k_val_from_ckpt.csv
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "scans"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

SWEEP_ROOT_DEFAULT = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "jetclass_output/adam/sweep_lr"
)
RAW_ROOT_DEFAULT = Path("/projects/j/jdezoort/jetclass")
DATA_ROOT_DEFAULT = Path("/scratch/gpfs/BHANIN/jdezoort/datasets/jetclass")

KNN_RE = re.compile(r"knn(\d+)")
GSTAR_RE = re.compile(r"gstar([0-9p]+)")
SEED_RE = re.compile(r"seed(\d+)")
RUN_TAG_RE = re.compile(r"graph-[kR]-b\d+-g\d+-s\d+")


def _parse_gstar(tag: str) -> float:
    return float(tag.replace("p", "."))


def parse_run_dir(run_dir: Path, *, allow_graph_r: bool = False) -> dict[str, Any]:
    name = run_dir.name
    km = KNN_RE.search(name)
    sm = GSTAR_RE.search(name)
    if km is None or sm is None:
        raise ValueError(f"cannot parse knn/gstar from {name}")
    is_k = "graph-k" in name and "s2000" in name
    is_r = allow_graph_r and "graph-R" in name and "s2000" in name and "knn10" in name
    if not (is_k or is_r):
        raise ValueError(f"not a matched graph-k/graph-R s2000 arm: {name}")
    smatch = SEED_RE.search(name)
    return {
        "run_dir": run_dir,
        "k": int(km.group(1)),
        "star_r": _parse_gstar(sm.group(1)),
        "seed": int(smatch.group(1)) if smatch else 0,
        "ckpt": run_dir / "best.ckpt",
        "parquet": Path(str(run_dir) + ".parquet"),
        "kind": "graph-R" if is_r else "graph-k",
    }


def discover_arms(
    sweep_root: Path,
    *,
    require_ckpt: bool = True,
    include_graph_r: bool = False,
    graph_k_only: bool = False,
    graph_r_only: bool = False,
) -> list[dict[str, Any]]:
    arms: list[dict[str, Any]] = []
    for d in sorted(sweep_root.iterdir()):
        if not d.is_dir():
            continue
        try:
            arm = parse_run_dir(d, allow_graph_r=include_graph_r or graph_r_only)
        except ValueError:
            continue
        if graph_k_only and arm.get("kind") != "graph-k":
            continue
        if graph_r_only and arm.get("kind") != "graph-R":
            continue
        if require_ckpt and not arm["ckpt"].is_file():
            continue
        arms.append(arm)
    arms.sort(key=lambda a: (a["star_r"], a["k"], a.get("seed", 0), a.get("kind", "")))
    return arms


def _parquet_test_roc(parquet: Path) -> float | None:
    import pandas as pd

    if not parquet.is_file():
        return None
    df = pd.read_parquet(parquet)
    if "test_roc_auc" not in df.columns:
        return None
    te = df.dropna(subset=["test_roc_auc"])
    if te.empty:
        return None
    return float(te.iloc[-1]["test_roc_auc"])


def _parquet_gammas(parquet: Path) -> dict[str, float] | None:
    import pandas as pd

    if not parquet.is_file():
        return None
    df = pd.read_parquet(parquet)
    cols = ["attn_gamma_11", "attn_gamma_12", "attn_gamma_21", "attn_gamma_22"]
    if any(c not in df.columns for c in cols):
        return None
    row = df.iloc[0]
    if any(pd.isna(row[c]) for c in cols):
        return None
    return {c: float(row[c]) for c in cols}


def _build_args(
    *,
    k: int,
    star_r: float,
    n_gpus: int,
    batch_size: int,
    n_val: int,
    seed: int,
    gammas: dict[str, float] | None,
) -> argparse.Namespace:
    from scans.common.sweep_common import add_common_args, configure_graph_args

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    argv = [
        "--mode",
        "sweep_lr",
        "--model",
        "capen-llama-att",
        "--operators",
        "incidence",
        "--jetclass-stream",
        "--jetclass-raw-root",
        str(RAW_ROOT_DEFAULT),
        "--jetclass-features",
        "kin7",
        "--jetclass-edge-features",
        "part-interaction",
        "--live-knn-k",
        str(k),
        "--star-radius",
        str(star_r),
        "--optimizer",
        "adam",
        "--t-epoch",
        "1",
        "--residual-structure",
        "transformer-like",
        "--etas",
        "0.25",
        "--epochs",
        "1",
        "--n-train",
        "1000000",
        "--data-root",
        str(DATA_ROOT_DEFAULT),
        "--depth",
        "4",
        "--width",
        "256",
        "--heads",
        "8",
        "--dropout",
        "0.0",
        "--root",
        str(SWEEP_ROOT_DEFAULT.parent.parent),
        "--batch-size",
        str(batch_size),
        "--n-gpus",
        str(n_gpus),
        "--num-workers",
        "4",
        "--normalization",
        "uniform",
        "--attention-normalization",
        "gamma",
        "--num-particles",
        "80",
        "--n-val",
        str(n_val),
        "--seed",
        str(seed),
        "--layer-norm",
        "--no-torch-compile",
    ]
    if gammas is not None:
        argv.extend(
            [
                "--gamma-11",
                str(gammas["attn_gamma_11"]),
                "--gamma-12",
                str(gammas["attn_gamma_12"]),
                "--gamma-21",
                str(gammas["attn_gamma_21"]),
                "--gamma-22",
                str(gammas["attn_gamma_22"]),
            ]
        )
    args = parser.parse_args(argv)
    args.dataset = "jetclass"
    configure_graph_args(args)
    return args


def _append_csv_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0)
        existing = f.read()
        write_header = len(existing.strip()) == 0
        f.seek(0, 2)
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


CSV_FIELDS = [
    "k",
    "star_r",
    "seed",
    "ckpt",
    "run_dir",
    "global_step",
    "val_loss",
    "val_acc",
    "val_roc_auc",
    "val_bg_rejection",
    "val_bg_rejection_0p3",
    "test_roc_auc",
]


def eval_one(
    run_dir: Path,
    *,
    out_csv: Path,
    out_json: Path | None = None,
    n_gpus: int = 1,
    batch_size: int = 128,
    n_val: int = 25000,
    seed: int = 0,
) -> dict[str, Any]:
    import lightning as L
    import torch

    from cpen.lit_models.lit_cpen_jetclass import LitCPENJetClass
    from cpen.utils.jetclass import N_CLASSES
    from scans.common.sweep_common import (
        build_model,
        create_jetclass_datamodule,
        jetclass_live_graph_kwargs,
        run_options_from_args,
    )

    arm = parse_run_dir(run_dir, allow_graph_r=True)
    ckpt: Path = arm["ckpt"]
    if not ckpt.is_file():
        raise FileNotFoundError(f"missing best.ckpt under {run_dir}")

    gammas = _parquet_gammas(arm["parquet"])
    args = _build_args(
        k=arm["k"],
        star_r=arm["star_r"],
        n_gpus=n_gpus,
        batch_size=batch_size,
        n_val=n_val,
        seed=seed,
        gammas=gammas,
    )
    L.seed_everything(seed, workers=True)

    data_root = args.data_root or args.root
    dm = create_jetclass_datamodule(args, data_root=data_root)

    n_features = 7  # kin7
    n_edge_features = 4  # part-interaction
    out_dim = int(N_CLASSES)
    model = build_model(
        args,
        n_features=n_features,
        n_edge_features=n_edge_features,
        out_dim=out_dim,
        depth=int(args.depth),
        width=int(args.width),
        heads=int(args.heads),
    )
    live = jetclass_live_graph_kwargs(args)
    lit = LitCPENJetClass(
        model,
        eta_0=0.25,
        model_name="capen-llama-att",
        optimizer="adam",
        scheduler="none",
        dataset="jetclass",
        depth=int(args.depth),
        width=int(args.width),
        heads=int(args.heads),
        t_epoch=1.0,
        run_options=run_options_from_args(args),
        **live,
    )

    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    global_step = int(payload.get("global_step", -1))

    trainer = L.Trainer(
        accelerator="gpu" if n_gpus > 0 else "cpu",
        devices=max(n_gpus, 1) if n_gpus > 0 else 1,
        precision="bf16-mixed" if n_gpus > 0 else "32-true",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    metrics_list = trainer.validate(lit, datamodule=dm, ckpt_path=str(ckpt))
    metrics = metrics_list[0] if metrics_list else {}

    def _get(name: str) -> float | None:
        v = metrics.get(name)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    row = {
        "k": arm["k"],
        "star_r": arm["star_r"],
        "seed": arm.get("seed", 0),
        "ckpt": str(ckpt),
        "run_dir": str(run_dir),
        "global_step": global_step,
        "val_loss": _get("val_loss"),
        "val_acc": _get("val_acc"),
        "val_roc_auc": _get("val_roc_auc"),
        "val_bg_rejection": _get("val_bg_rejection"),
        "val_bg_rejection_0p3": _get("val_bg_rejection_0p3"),
        "test_roc_auc": _parquet_test_roc(arm["parquet"]),
    }
    _append_csv_row(out_csv, row, CSV_FIELDS)
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(row, indent=2) + "\n")
    print(
        f"[eval] k={row['k']} R★={row['star_r']} step={row['global_step']} "
        f"val_roc={row['val_roc_auc']} test_roc={row['test_roc_auc']}"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, default=SWEEP_ROOT_DEFAULT)
    parser.add_argument(
        "--discover",
        action="store_true",
        help="List arms with best.ckpt and optionally write --manifest",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Write discovered run dirs (one per line) for Slurm array",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--manifest-index",
        type=int,
        default=None,
        help="Select --run-dir as line N (0-based) of --manifest",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=REPO_ROOT / "paper" / "data" / "graph_k_val_from_ckpt.csv",
    )
    parser.add_argument(
        "--out-json-dir",
        type=Path,
        default=Path(
            "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
            "logs/graph_k_val_eval"
        ),
    )
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-val", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-graph-r",
        action="store_true",
        help="Also discover graph-R (k=10 × R★) s2000 arms",
    )
    parser.add_argument(
        "--graph-r-only",
        action="store_true",
        help="Discover only graph-R s2000 arms (Plot 2 extras)",
    )
    args = parser.parse_args()

    if args.discover:
        arms = discover_arms(
            args.sweep_root,
            include_graph_r=args.include_graph_r,
            graph_r_only=args.graph_r_only,
        )
        for a in arms:
            kind = a.get("kind", "graph-k")
            print(
                f"k={a['k']:<2} R★={a['star_r']:<5} seed={a.get('seed', 0)} "
                f"[{kind}] {a['run_dir']}"
            )
        if args.manifest is not None:
            args.manifest.parent.mkdir(parents=True, exist_ok=True)
            args.manifest.write_text(
                "\n".join(str(a["run_dir"]) for a in arms) + ("\n" if arms else "")
            )
            print(f"wrote {args.manifest} ({len(arms)} arms)")
        return

    run_dir = args.run_dir
    if args.manifest_index is not None:
        if args.manifest is None:
            raise SystemExit("--manifest-index requires --manifest")
        lines = [
            ln.strip()
            for ln in args.manifest.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if args.manifest_index < 0 or args.manifest_index >= len(lines):
            raise SystemExit(
                f"manifest index {args.manifest_index} out of range "
                f"(0..{len(lines) - 1})"
            )
        run_dir = Path(lines[args.manifest_index])
    if run_dir is None:
        raise SystemExit("pass --run-dir or --manifest-index, or --discover")

    arm = parse_run_dir(run_dir, allow_graph_r=True)
    r_tag = f"{arm['star_r']:g}".replace(".", "p")
    out_json = (
        args.out_json_dir / f"knn{arm['k']}_R{r_tag}_seed{arm.get('seed', 0)}.json"
    )
    eval_one(
        run_dir,
        out_csv=args.out_csv,
        out_json=out_json,
        n_gpus=args.n_gpus,
        batch_size=args.batch_size,
        n_val=args.n_val,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
