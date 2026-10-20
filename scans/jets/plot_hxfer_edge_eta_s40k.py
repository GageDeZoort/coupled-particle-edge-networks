#!/usr/bin/env python3
"""Hyperparameter-transfer figure: Adam η₀×(L,D) @ 40k, frozen OP (k★=7, R★=0.1).

Latest panel-(a) evidence from the edge-η campaigns:
  L4/D256  tag edge-b64-g4-s40000-dh32-k7-R0p1
  L6/D512  tag edge-L6D512-b64-g4-s40000-dh32-k7-R0p1

  python scans/jets/plot_hxfer_edge_eta_s40k.py
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
LOGDIR = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs"
)
CSV_OUT = REPO / "paper" / "data" / "hxfer_edge_eta_s40k.csv"
FIG_STEM = "fig_hparam_transfer_proof_edge_eta_s40k"
MAX_STEPS = 40000

# (L, D, run_tag, log_glob)
CAMPAIGNS = [
    (4, 256, "edge-b64-g4-s40000-dh32-k7-R0p1", "jc_edge_k7_s40k_gt_*.out"),
    (6, 512, "edge-L6D512-b64-g4-s40000-dh32-k7-R0p1", "jc_edge_L6D512_s40k_gt_*.out"),
]
ETA_GRID = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0]
NAME_RE = re.compile(r"^capen-llama-att_(?P<L>\d+)_(?P<D>\d+)_(?P<eta>[0-9p]+)_")


def _eta(tag: str) -> float:
    return float(tag.replace("p", "."))


def _from_parquet(L: int, D: int, tag: str) -> list[dict]:
    rows = []
    for pq in sorted(SWEEP.glob(f"capen-llama-att_{L}_{D}_*{tag}*.parquet")):
        m = NAME_RE.match(pq.name)
        if m is None:
            continue
        eta0 = _eta(m.group("eta"))
        df = pd.read_parquet(pq, columns=["global_step", "val_roc_auc", "val_loss"])
        sub = df.dropna(subset=["val_roc_auc"])
        if sub.empty:
            continue
        near = sub[sub["global_step"] >= int(0.95 * MAX_STEPS)]
        if not near.empty:
            row = near.sort_values("global_step").iloc[-1]
            done = True
        else:
            row = sub.loc[sub["global_step"].idxmax()]
            done = int(row["global_step"]) >= MAX_STEPS - 100
        loss = row["val_loss"] if "val_loss" in row and row["val_loss"] == row["val_loss"] else None
        rows.append(
            dict(
                depth=L,
                width=D,
                model=f"L{L}/D{D}",
                eta0=eta0,
                step=int(row["global_step"]),
                val_roc_auc=float(row["val_roc_auc"]),
                val_loss=float(loss) if loss is not None else None,
                best_val_roc_auc=float(sub["val_roc_auc"].max()),
                done=done,
                source="parquet",
                tag=tag,
                run=pq.stem,
            )
        )
    return rows


def _from_logs(L: int, D: int, tag: str, log_glob: str) -> list[dict]:
    """Fallback / upgrade mid-run cells from Slurm val lines."""
    rows = []
    for lp in LOGDIR.glob(log_glob):
        try:
            text = lp.read_text(errors="ignore")
        except Exception:
            continue
        if tag.split("-")[0] not in text and "max_steps=40000" not in text[:5000]:
            # still accept if eta0 banner present
            pass
        m = re.search(r"eta0=([0-9.]+)", text[:5000])
        if m is None:
            continue
        eta0 = float(m.group(1))
        last = None
        for m2 in re.finditer(
            r"\[val\].*?step=(\d+).*?val_roc_auc=([0-9.eE+-]+)"
            r"(?:.*?val_loss=([0-9.eE+-]+))?",
            text,
        ):
            last = (
                int(m2.group(1)),
                float(m2.group(2)),
                float(m2.group(3)) if m2.group(3) else None,
            )
        if last is None:
            continue
        step, auc, loss = last
        rows.append(
            dict(
                depth=L,
                width=D,
                model=f"L{L}/D{D}",
                eta0=eta0,
                step=step,
                val_roc_auc=auc,
                val_loss=loss,
                best_val_roc_auc=auc,
                done=step >= MAX_STEPS - 100,
                source="log",
                tag=tag,
                run=lp.name,
            )
        )
    return rows


def collect() -> pd.DataFrame:
    rows: list[dict] = []
    for L, D, tag, log_glob in CAMPAIGNS:
        rows.extend(_from_parquet(L, D, tag))
        by_key = {(r["depth"], r["width"], r["eta0"]): r for r in rows if r["tag"] == tag}
        for lr in _from_logs(L, D, tag, log_glob):
            key = (lr["depth"], lr["width"], lr["eta0"])
            if key not in by_key:
                rows.append(lr)
                by_key[key] = lr
            elif lr["step"] > by_key[key]["step"]:
                by_key[key].update(
                    step=lr["step"],
                    val_roc_auc=lr["val_roc_auc"],
                    val_loss=lr["val_loss"],
                    done=lr["done"],
                    source="log+parquet" if by_key[key]["source"] == "parquet" else "log",
                )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, stem: str) -> None:
    use_nature(fontsize=8)
    fig, ax = plt.subplots(figsize=(3.6, 2.45))
    series = [
        ("L4/D256", 4, 256, "o"),
        ("L6/D512", 6, 512, "^"),
    ]
    for i, (label, L, D, marker) in enumerate(series):
        sub = df[(df["depth"] == L) & (df["width"] == D)].sort_values("eta0")
        if sub.empty:
            continue
        c = nature_color(i)
        done = sub[sub["done"]]
        mid = sub[~sub["done"]]
        if len(done):
            ax.plot(
                done["eta0"],
                done["val_roc_auc"],
                marker=marker,
                ms=4.5,
                lw=1.0,
                color=c,
                label=label,
                zorder=3,
            )
        elif len(mid):
            ax.plot(
                [],
                [],
                marker=marker,
                ms=4.5,
                lw=1.0,
                color=c,
                label=label,
            )
        if len(mid):
            ax.plot(
                mid["eta0"],
                mid["val_roc_auc"],
                marker=marker,
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
    ax.set_xticklabels([f"{e:g}" for e in ETA_GRID], fontsize=6.5)
    ax.set_xlabel(r"base learning rate $\eta_0$")
    ax.set_ylabel("Val. ROC-AUC")
    ax.set_title(
        r"Adam @ 40k  ($k^\star{=}7$, $R_\star{=}0.1$, noloop, $B{=}256$)",
        loc="left",
        pad=3,
    )
    ax.legend(loc="lower left", handlelength=1.2)
    y = df["val_roc_auc"].astype(float)
    ax.set_ylim(float(y.min()) - 0.002, float(y.max()) + 0.0015)

    written = save_paper_figure(fig, stem, formats=("png", "pdf", "eps"))
    for p in written:
        print(f"wrote {p}")
    plt.close(fig)

    peaks = []
    for (L, D), sub in df.groupby(["depth", "width"]):
        best = sub.loc[sub["val_roc_auc"].idxmax()]
        status = "done" if bool(sub["done"].all()) else f"{int(sub['done'].sum())}/{len(sub)} done"
        peaks.append(
            f"L{L}/D{D} peak η₀={best['eta0']:g} (ROC={best['val_roc_auc']:.5f} @ {int(best['step'])}; {status})"
        )
    n_done = int(df["done"].sum())
    cap = REPO / "paper" / "figures" / f"{stem}_caption.txt"
    cap.write_text(
        "Figure. Learning-rate transfer under the frozen graph-construction OP.\n"
        "Holdout validation ROC–AUC after up to 40 000 Adam steps at constant peak LR\n"
        "η = η₀/√D versus base learning rate η₀, for (L, D) ∈ {(4, 256), (6, 512)}\n"
        "(d_h=32, B_eff=256, k★=7, R★=0.1, noloop, single-pass stream).\n"
        f"Open markers = mid-run (<95% of 40k). {n_done}/{len(df)} cells finished.\n"
        + "; ".join(peaks)
        + ".\n"
    )
    print(f"wrote {cap}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--stem", default=FIG_STEM)
    args = ap.parse_args()

    if args.csv and args.csv.is_file():
        df = pd.read_csv(args.csv)
    else:
        df = collect()
    if df.empty:
        raise SystemExit("no edge-η 40k cells found")
    df = df.sort_values(["depth", "width", "eta0"]).drop_duplicates(
        subset=["depth", "width", "eta0"], keep="last"
    )
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"wrote {CSV_OUT} ({len(df)} rows)")
    for (L, D), sub in df.groupby(["depth", "width"]):
        best = sub.loc[sub["val_roc_auc"].idxmax()]
        print(
            f"  L{L}/D{D}: best η₀={best['eta0']:g} roc={best['val_roc_auc']:.5f} "
            f"@step={int(best['step'])} done={int(sub['done'].sum())}/{len(sub)}"
        )
    plot(df, args.stem)


if __name__ == "__main__":
    main()
