#!/usr/bin/env python3
"""Aggregate ckpt-val CSV/JSON and plot paper ROC vs k (mean ± std over seeds).

  python scans/jets/plot_graph_k_vs_roc.py
  python scans/jets/plot_graph_k_vs_roc.py --csv paper/data/graph_k_val_from_ckpt.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "paper" / "data" / "graph_k_val_from_ckpt.csv"
DEFAULT_JSON_DIR = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "logs/graph_k_val_eval"
)
FIG_DIR = REPO_ROOT / "paper" / "figures"
SEED_RE = re.compile(r"seed(\d+)")


def _seed_from_row(row: dict | pd.Series) -> int:
    if "seed" in row and pd.notna(row["seed"]):
        try:
            return int(row["seed"])
        except (TypeError, ValueError):
            pass
    for key in ("run_dir", "ckpt"):
        val = row.get(key) if isinstance(row, dict) else row.get(key, None)
        if isinstance(val, str):
            m = SEED_RE.search(val)
            if m:
                return int(m.group(1))
    return 0


def _load_rows(csv_path: Path, json_dir: Path | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if csv_path.is_file() and csv_path.stat().st_size > 0:
        frames.append(pd.read_csv(csv_path))
    if json_dir is not None and json_dir.is_dir():
        rows = []
        for p in sorted(json_dir.glob("knn*_R*.json")):
            rows.append(pd.read_json(p, typ="series").to_dict())
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        raise SystemExit(f"no eval rows in {csv_path} or {json_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["val_roc_auc", "k", "star_r"])
    df["k"] = df["k"].astype(int)
    df["star_r"] = df["star_r"].astype(float)
    df["seed"] = [_seed_from_row(r) for _, r in df.iterrows()]
    # One row per (k, star_r, seed); prefer newest
    df = df.sort_values(["star_r", "k", "seed"]).drop_duplicates(
        subset=["k", "star_r", "seed"], keep="last"
    )
    return df.reset_index(drop=True)


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (star_r, k), sub in df.groupby(["star_r", "k"], sort=True):
        rocs = sub["val_roc_auc"].astype(float).to_numpy()
        n = len(rocs)
        mean = float(np.mean(rocs))
        std = float(np.std(rocs, ddof=1)) if n > 1 else 0.0
        rows.append(
            {
                "k": int(k),
                "star_r": float(star_r),
                "n_seeds": n,
                "val_roc_auc_mean": mean,
                "val_roc_auc_std": std,
                "seeds": ",".join(str(int(s)) for s in sorted(sub["seed"].unique())),
            }
        )
    return pd.DataFrame(rows).sort_values(["star_r", "k"]).reset_index(drop=True)


def plot_roc_vs_k(agg: pd.DataFrame, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    radii = sorted(agg["star_r"].unique())
    markers = ["o", "s", "^", "D", "v"]
    for i, r in enumerate(radii):
        sub = agg[agg["star_r"] == r].sort_values("k")
        yerr = sub["val_roc_auc_std"].to_numpy()
        # Hide error bars when only one seed (std=0 by construction)
        yerr_plot = np.where(sub["n_seeds"].to_numpy() > 1, yerr, np.nan)
        ax.errorbar(
            sub["k"],
            sub["val_roc_auc_mean"],
            yerr=yerr_plot,
            marker=markers[i % len(markers)],
            linewidth=1.5,
            capsize=3,
            label=rf"$R^\star$={r:g}",
        )
    ax.set_xlabel(r"kNN $k$")
    ax.set_ylabel("val ROC AUC")
    ax.set_title(r"Graph construction: val ROC vs $k$ (mean $\pm$ std)")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pdf = out_stem.with_suffix(".pdf")
    png = out_stem.with_suffix(".png")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Per-seed paper CSV",
    )
    parser.add_argument(
        "--out-agg-csv",
        type=Path,
        default=REPO_ROOT / "paper" / "data" / "graph_k_val_from_ckpt_agg.csv",
    )
    parser.add_argument(
        "--out-fig",
        type=Path,
        default=FIG_DIR / "fig_graph_k_vs_roc",
    )
    args = parser.parse_args()

    df = _load_rows(args.csv, args.json_dir)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(df)} seed-arms)")

    agg = aggregate_seeds(df)
    args.out_agg_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.out_agg_csv, index=False)
    print(f"wrote {args.out_agg_csv} ({len(agg)} aggregated points)")

    for r, sub in agg.groupby("star_r"):
        print(f"\nR★={r:g}:")
        for _, row in sub.sort_values("k").iterrows():
            if int(row["n_seeds"]) > 1:
                print(
                    f"  k={int(row['k']):<2} "
                    f"val_roc={row['val_roc_auc_mean']:.5f} "
                    f"± {row['val_roc_auc_std']:.5f} "
                    f"(n={int(row['n_seeds'])} seeds [{row['seeds']}])"
                )
            else:
                print(
                    f"  k={int(row['k']):<2} "
                    f"val_roc={row['val_roc_auc_mean']:.5f} "
                    f"(n=1 seed [{row['seeds']}])"
                )

    plot_roc_vs_k(agg, args.out_fig)
    print(
        "\nCaption: mean ± sample std of val ROC from best.ckpt "
        "(min val_loss) on fixed 25k val across seeds; "
        "error bars omitted when n=1."
    )


if __name__ == "__main__":
    main()
