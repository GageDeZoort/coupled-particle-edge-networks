#!/usr/bin/env python3
"""Quick Plot 1 from training-log val_roc (parquet), mean±std over seeds."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path("/home/jdezoort/coupled-particle-edge-networks")
ROOT = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "jetclass_output/adam/sweep_lr"
)
FIG = REPO / "paper" / "figures" / "fig_graph_k_vs_roc"
CSV_OUT = REPO / "paper" / "data" / "graph_k_val_from_trainlog.csv"
AGG_OUT = REPO / "paper" / "data" / "graph_k_val_from_trainlog_agg.csv"

KNN_RE = re.compile(r"knn(\d+)")
GSTAR_RE = re.compile(r"gstar([0-9p]+)")
SEED_RE = re.compile(r"seed(\d+)")


def main() -> None:
    rows = []
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir() or "graph-k" not in d.name:
            continue
        if "s2k" not in d.name and "s2000" not in d.name:
            continue
        km = KNN_RE.search(d.name)
        sm = GSTAR_RE.search(d.name)
        sd = SEED_RE.search(d.name)
        if not (km and sm and sd):
            continue
        pq_path = Path(str(d) + ".parquet")
        if not pq_path.is_file():
            continue
        try:
            table = pq.read_table(
                pq_path, columns=["global_step", "val_roc_auc", "val_loss"]
            )
        except Exception as exc:
            print(f"skip {d.name}: {exc}")
            continue
        steps = table.column("global_step").to_pylist()
        rocs = table.column("val_roc_auc").to_pylist()
        losses = table.column("val_loss").to_pylist()
        pairs = [
            (s, r, lo)
            for s, r, lo in zip(steps, rocs, losses)
            if r is not None and np.isfinite(r)
        ]
        if not pairs:
            continue
        last_s, last_r, _ = pairs[-1]
        with_loss = [
            (s, r, lo) for s, r, lo in pairs if lo is not None and np.isfinite(lo)
        ]
        best = min(with_loss, key=lambda x: x[2]) if with_loss else None
        rows.append(
            {
                "k": int(km.group(1)),
                "star_r": float(sm.group(1).replace("p", ".")),
                "seed": int(sd.group(1)),
                "val_roc_auc": float(last_r),
                "global_step": int(last_s),
                "val_roc_auc_bestloss": float(best[1]) if best else float(last_r),
                "best_step": int(best[0]) if best else int(last_s),
                "run_dir": str(d),
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values(["star_r", "k", "seed"])
        .drop_duplicates(["star_r", "k", "seed"], keep="last")
    )
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    print(f"per-seed rows: {len(df)} -> {CSV_OUT}")

    agg_rows = []
    for (r, k), sub in df.groupby(["star_r", "k"]):
        rocs = sub["val_roc_auc"].to_numpy(dtype=float)
        n = len(rocs)
        agg_rows.append(
            {
                "k": int(k),
                "star_r": float(r),
                "n_seeds": n,
                "val_roc_auc_mean": float(np.mean(rocs)),
                "val_roc_auc_std": float(np.std(rocs, ddof=1)) if n > 1 else 0.0,
                "seeds": ",".join(str(int(s)) for s in sorted(sub["seed"])),
                "steps": ",".join(str(int(s)) for s in sub["global_step"]),
            }
        )
    agg = pd.DataFrame(agg_rows).sort_values(["star_r", "k"])
    agg.to_csv(AGG_OUT, index=False)

    for r, sub in agg.groupby("star_r"):
        print(f"\nR★={r:g}:")
        for _, row in sub.iterrows():
            if row["n_seeds"] > 1:
                print(
                    f"  k={int(row['k']):<2} {row['val_roc_auc_mean']:.5f} "
                    f"± {row['val_roc_auc_std']:.5f}  "
                    f"n={int(row['n_seeds'])} [{row['seeds']}] @steps {row['steps']}"
                )
            else:
                print(
                    f"  k={int(row['k']):<2} {row['val_roc_auc_mean']:.5f}  "
                    f"n=1 [{row['seeds']}] @step {row['steps']}"
                )

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    markers = ["o", "s", "^", "D"]
    for i, r in enumerate(sorted(agg["star_r"].unique())):
        sub = agg[agg["star_r"] == r].sort_values("k")
        yerr = np.where(
            sub["n_seeds"].to_numpy() > 1,
            sub["val_roc_auc_std"].to_numpy(),
            np.nan,
        )
        ax.errorbar(
            sub["k"],
            sub["val_roc_auc_mean"],
            yerr=yerr,
            marker=markers[i % len(markers)],
            linewidth=1.5,
            capsize=3,
            label=rf"$R^\star$={r:g}",
        )
    ax.set_xlabel(r"kNN $k$")
    ax.set_ylabel("val ROC AUC")
    ax.set_title(r"Graph construction: val ROC vs $k$ (train-log mean $\pm$ std)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(FIG.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {FIG.with_suffix('.png')}")
    print(f"wrote {FIG.with_suffix('.pdf')}")
    print(
        "NOTE: last logged val_roc from training parquet "
        "(usually epoch-end ~1954), NOT best.ckpt re-eval."
    )


if __name__ == "__main__":
    main()
