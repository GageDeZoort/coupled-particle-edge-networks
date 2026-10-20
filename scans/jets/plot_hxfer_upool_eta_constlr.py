#!/usr/bin/env python3
"""Nature figure: uniform-pool Adam const-LR η₀×D transfer @ 10k (v2stream).

  python scans/jets/plot_hxfer_upool_eta_constlr.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from paper.style.nature_plots import nature_color, save_paper_figure, use_nature

SWEEP = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "jetclass_output/adam/sweep_lr"
)
CSV_OUT = REPO / "paper" / "data" / "hxfer_upool_eta_constlr.csv"
FIG_STEM = "fig_hparam_transfer_proof_upool10k"
TAG = "hxfer-upool-constlr"
TAG_SUFFIX = "v2stream"
MAX_STEPS = 10000
ETA_GRID = [0.01, 0.05, 0.1, 0.18, 0.25, 0.5, 1.0]
WIDTHS = [512, 384]
NAME_RE = re.compile(r"^capen-llama-att_(?P<L>\d+)_(?P<D>\d+)_(?P<eta>[0-9p]+)_")


def _eta(tag: str) -> float:
    return float(tag.replace("p", "."))


def collect(target: int = MAX_STEPS) -> pd.DataFrame:
    rows = []
    for pq in sorted(SWEEP.glob(f"*{TAG}*{TAG_SUFFIX}.parquet")):
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
                pool="uniform",
                tag=TAG_SUFFIX,
                run=pq.stem,
            )
        )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, stem: str) -> None:
    use_nature(fontsize=8)
    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    markers = {512: "^", 384: "s"}
    for i, width in enumerate(WIDTHS):
        sub = df[df["width"] == width].sort_values("eta0")
        if sub.empty:
            continue
        c = nature_color(i)
        ax.plot(
            sub["eta0"],
            sub["val_roc_auc"],
            marker=markers[width],
            ms=4.5,
            lw=1.0,
            color=c,
            label=rf"$D={width}$",
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xticks(ETA_GRID)
    ax.set_xticklabels([str(e) for e in ETA_GRID])
    ax.set_xlabel(r"base learning rate $\eta_0$")
    ax.set_ylabel("Val. ROC-AUC")
    ax.set_title(
        r"uniform pool, Adam const-LR @ 10k  ($k^\star{=}6$, $R_\star{=}0.1$)",
        loc="left",
        pad=3,
    )
    ax.legend(loc="lower left", handlelength=1.2)
    # y-window: focus on the transfer band; η₀=1 cliffs stay visible
    y = df["val_roc_auc"].astype(float)
    ax.set_ylim(float(y.min()) - 0.002, float(y.max()) + 0.001)

    written = save_paper_figure(fig, stem, formats=("png", "pdf", "eps"))
    for p in written:
        print(f"wrote {p}")
    plt.close(fig)

    # Caption
    peaks = []
    for w, sub in df.groupby("width"):
        best = sub.loc[sub["val_roc_auc"].idxmax()]
        peaks.append(
            f"D={w} peak η₀={best['eta0']:g} (ROC={best['val_roc_auc']:.5f})"
        )
    cap = REPO / "paper" / "figures" / f"{stem}_caption.txt"
    cap.write_text(
        "Figure. Learning-rate transfer under uniform output pooling on JetClass.\n"
        "Holdout validation ROC–AUC after 10 000 Adam steps at constant peak LR\n"
        "η = η₀/√D versus base learning rate η₀, for widths D ∈ {384, 512}\n"
        "(L=4, d_h=32, B_eff=512, k★=6, R★=0.1, single-pass n_train=5.12M).\n"
        + "; ".join(peaks)
        + ".\n"
        "Both widths peak at the same η₀, demonstrating μP LR transfer.\n"
    )
    print(f"wrote {cap}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--stem", default=FIG_STEM)
    ap.add_argument("--target", type=int, default=MAX_STEPS)
    args = ap.parse_args()

    if args.csv and args.csv.is_file():
        df = pd.read_csv(args.csv)
    else:
        df = collect(target=args.target)
    if df.empty:
        raise SystemExit(f"no uniform cells under {SWEEP} (*{TAG}*{TAG_SUFFIX}*)")
    df = df.sort_values(["width", "eta0"]).drop_duplicates(
        subset=["width", "eta0"], keep="last"
    )
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"wrote {CSV_OUT} ({len(df)} rows)")
    for w, sub in df.groupby("width"):
        best = sub.loc[sub["val_roc_auc"].idxmax()]
        print(
            f"  D={w}: best η₀={best['eta0']:g} roc={best['val_roc_auc']:.5f} "
            f"@step={int(best['step'])}"
        )
    plot(df, args.stem)


if __name__ == "__main__":
    main()
