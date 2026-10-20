#!/usr/bin/env python3
"""Hyperparameter-transfer proof: AdamW WFC val ROC vs η₀ at D∈{256,384,512}.

Working point: k=10, R★=0.125, t_epoch=10, d_head=32, 50k steps.

  python scans/jets/plot_hparam_xfer_eta.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "jetclass_output/adamw/sweep_lr"
)
OUT_CSV = REPO_ROOT / "paper" / "data" / "hparam_xfer_wfc_eta.csv"
FIG_DIR = REPO_ROOT / "paper" / "figures"

NAME_RE = re.compile(
    r"^capen-llama-att_"
    r"(?P<L>\d+)_(?P<D>\d+)_(?P<eta>[0-9p]+)_t(?P<t>[0-9p]+)_"
)


def _f(tag: str) -> float:
    return float(tag.replace("p", "."))


def _roc_at_or_near(pq: Path, target: int = 50000) -> tuple[float | None, int | None]:
    if not pq.is_file():
        return None, None
    df = pd.read_parquet(pq)
    if "val_roc_auc" not in df.columns or "global_step" not in df.columns:
        return None, None
    sub = df.dropna(subset=["val_roc_auc"]).copy()
    if sub.empty:
        return None, None
    # Prefer last val with step >= 0.95 * target; else best overall
    near = sub[sub["global_step"] >= int(0.95 * target)]
    if not near.empty:
        row = near.sort_values("global_step").iloc[-1]
    else:
        row = sub.loc[sub["val_roc_auc"].idxmax()]
    return float(row["val_roc_auc"]), int(row["global_step"])


def collect(*, target: int = 50000) -> pd.DataFrame:
    rows = []
    for d in sorted(SWEEP.iterdir()):
        if not d.is_dir() or "hxfer-wfc" not in d.name:
            continue
        m = NAME_RE.match(d.name)
        if m is None:
            continue
        width = int(m.group("D"))
        eta = _f(m.group("eta"))
        t_epoch = _f(m.group("t"))
        roc, step = _roc_at_or_near(Path(str(d) + ".parquet"), target=target)
        if roc is None:
            continue
        rows.append(
            {
                "width": width,
                "heads": width // 32,
                "eta0": eta,
                "t_epoch": t_epoch,
                "val_roc_auc": roc,
                "step": step,
                "run_dir": str(d),
            }
        )
    if not rows:
        raise SystemExit(f"no hxfer-wfc rows under {SWEEP}")
    df = pd.DataFrame(rows)
    df = df.sort_values(["width", "eta0"]).drop_duplicates(
        subset=["width", "eta0"], keep="last"
    )
    return df.reset_index(drop=True)


def plot(df: pd.DataFrame, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    markers = {256: "o", 384: "s", 512: "^"}
    for width, sub in df.groupby("width"):
        sub = sub.sort_values("eta0")
        ax.plot(
            sub["eta0"],
            sub["val_roc_auc"],
            marker=markers.get(int(width), "o"),
            linewidth=1.5,
            label=rf"$D$={width} ($H$={width // 32})",
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\eta_0$")
    ax.set_ylabel("val ROC AUC")
    ax.set_title(
        r"AdamW transfer @ $t_{\mathrm{epoch}}{=}10$, "
        r"$k{=}10$, $R^\star{=}0.125$ (50k steps)"
    )
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in (".pdf", ".png"):
        path = out_stem.with_suffix(ext)
        fig.savefig(path, dpi=200 if ext == ".png" else None, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-csv", type=Path, default=OUT_CSV)
    parser.add_argument(
        "--out-fig", type=Path, default=FIG_DIR / "fig_hparam_xfer_wfc_eta"
    )
    parser.add_argument("--target", type=int, default=50000)
    args = parser.parse_args()
    df = collect(target=args.target)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(df)} cells)")
    for w, sub in df.groupby("width"):
        best = sub.loc[sub["val_roc_auc"].idxmax()]
        print(
            f"  D={w}: best η₀={best['eta0']:g} roc={best['val_roc_auc']:.5f} "
            f"@step={best['step']}"
        )
    plot(df, args.out_fig)


if __name__ == "__main__":
    main()
