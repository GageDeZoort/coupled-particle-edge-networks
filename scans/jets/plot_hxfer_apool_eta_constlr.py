#!/usr/bin/env python3
"""Preliminary hparam-transfer figure: attn Adam const-LR η₀×D @ 10k (v2stream).

  python scans/jets/plot_hxfer_apool_eta_constlr.py
  python scans/jets/plot_hxfer_apool_eta_constlr.py --csv paper/data/hxfer_apool_eta_constlr.csv
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(REPO))
from paper.style.nature_plots import nature_color, save_paper_figure, use_nature

SWEEP = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "jetclass_output/adam/sweep_lr"
)
CSV_OUT = REPO / "paper" / "data" / "hxfer_apool_eta_constlr.csv"
FIG_STEM = "fig_hparam_transfer_proof_apool10k"
TAG = "v2stream"
MAX_STEPS = 10000
ETA_GRID = [0.05, 0.1, 0.25, 0.5, 1.0]
WIDTHS = [512, 384, 256]
NAME_RE = re.compile(r"^capen-llama-att_(?P<L>\d+)_(?P<D>\d+)_(?P<eta>[0-9p]+)_")


def _eta(tag: str) -> float:
    return float(tag.replace("p", "."))


def collect_from_sweep(target: int = MAX_STEPS) -> pd.DataFrame:
    rows = []
    for pq in sorted(SWEEP.glob(f"*hxfer-apool-constlr*{TAG}.parquet")):
        m = NAME_RE.match(pq.name)
        if m is None:
            continue
        width = int(m.group("D"))
        eta0 = _eta(m.group("eta"))
        df = pd.read_parquet(pq, columns=["global_step", "val_roc_auc"])
        sub = df.dropna(subset=["val_roc_auc"])
        if sub.empty:
            continue
        near = sub[sub["global_step"] >= int(0.95 * target)]
        if not near.empty:
            row = near.sort_values("global_step").iloc[-1]
            done = True
        else:
            row = sub.loc[sub["val_roc_auc"].idxmax()]
            done = False
        rows.append(
            dict(
                width=width,
                eta0=eta0,
                step=int(row["global_step"]),
                val_roc_auc=float(row["val_roc_auc"]),
                best_val_roc_auc=float(sub["val_roc_auc"].max()),
                done=done,
                tag=TAG,
                run=pq.stem,
            )
        )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, stem: str) -> None:
    use_nature(fontsize=8)
    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    markers = {512: "^", 384: "s", 256: "o"}
    # Plot larger widths first so D=256 sits on top if overlapping.
    for i, width in enumerate(WIDTHS):
        sub = df[df["width"] == width].sort_values("eta0")
        if sub.empty:
            continue
        c = nature_color(i)
        if "done" not in sub.columns:
            sub = sub.assign(done=True)
        done = sub[sub["done"]]
        mid = sub[~sub["done"]]
        if len(done):
            ax.plot(
                done["eta0"],
                done["val_roc_auc"],
                marker=markers[width],
                ms=4.5,
                lw=1.0,
                color=c,
                label=rf"$D={width}$",
                zorder=3,
            )
        elif len(mid):
            # width with only mid-run cells — still legend
            ax.plot([], [], marker=markers[width], ms=4.5, lw=1.0, color=c, label=rf"$D={width}$")
        if len(mid):
            ax.plot(
                mid["eta0"],
                mid["val_roc_auc"],
                marker=markers[width],
                ms=4.5,
                lw=0.0,
                color=c,
                markerfacecolor="white",
                markeredgewidth=1.0,
                label="_nolegend_",
                zorder=4,
            )

    ax.set_xscale("log")
    ax.set_xticks(ETA_GRID)
    ax.set_xticklabels([str(e) for e in ETA_GRID])
    ax.set_xlabel(r"base learning rate $\eta_0$")
    ax.set_ylabel("Val. ROC-AUC")
    ax.set_title(
        r"attn Adam const-LR @ 10k  ($k^\star{=}6$, $R_\star{=}0.1$)",
        loc="left",
        pad=3,
    )
    ax.legend(loc="lower right", handlelength=1.2)
    if "done" not in df.columns:
        df = df.assign(done=True)
    n_done = int(df["done"].sum())
    ax.text(
        0.02,
        0.98,
        f"preliminary — {n_done}/{len(WIDTHS)*len(ETA_GRID)} cells @10k"
        + (" (open = mid-run)" if (~df["done"]).any() else ""),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.5,
        color="0.4",
    )
    # Tight y-window around finished cells only (ignore mid-run lows).
    y = df.loc[df["done"], "val_roc_auc"].astype(float)
    if len(y):
        ax.set_ylim(float(y.min()) - 0.0012, float(y.max()) + 0.0008)

    written = save_paper_figure(fig, stem, formats=("png", "pdf", "eps"))
    for p in written:
        print(f"wrote {p}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=None, help="Optional prebuilt CSV")
    ap.add_argument("--stem", default=FIG_STEM)
    ap.add_argument("--target", type=int, default=MAX_STEPS)
    args = ap.parse_args()

    if args.csv is not None and args.csv.is_file():
        df = pd.read_csv(args.csv)
        if "done" not in df.columns:
            df["done"] = df["step"] >= int(0.95 * args.target)
    else:
        df = collect_from_sweep(target=args.target)
        if df.empty:
            raise SystemExit(f"no {TAG} cells under {SWEEP}")
        if "done" not in df.columns:
            df["done"] = df["step"] >= int(0.95 * args.target)

    df = df.sort_values(["width", "eta0"]).drop_duplicates(
        subset=["width", "eta0"], keep="last"
    )
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"wrote {CSV_OUT} ({len(df)} rows)")
    for w, sub in df.groupby("width"):
        best = sub.loc[sub["val_roc_auc"].idxmax()]
        flag = "" if bool(best.get("done", True)) else " (mid-run)"
        print(
            f"  D={w}: best η₀={best['eta0']:g} roc={best['val_roc_auc']:.5f} "
            f"@step={int(best['step'])}{flag}"
        )
    plot(df, args.stem)


if __name__ == "__main__":
    main()
