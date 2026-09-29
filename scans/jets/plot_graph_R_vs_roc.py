#!/usr/bin/env python3
"""Paper Plot 2: val ROC vs R★ at fixed k=10 (ckpt eval + graph-k anchors).

  python scans/jets/plot_graph_R_vs_roc.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "paper" / "data" / "graph_R_val_from_ckpt.csv"
K_CSV = REPO_ROOT / "paper" / "data" / "graph_k_val_from_ckpt.csv"
DEFAULT_JSON_DIR = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "logs/graph_R_val_eval"
)
K_JSON_DIR = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "logs/graph_k_val_eval"
)
FIG_DIR = REPO_ROOT / "paper" / "figures"
FIXED_K = 10


def _load_json_dir(json_dir: Path) -> pd.DataFrame:
    rows = []
    if not json_dir.is_dir():
        return pd.DataFrame()
    for p in sorted(json_dir.glob("knn*_R*.json")):
        rows.append(pd.read_json(p, typ="series").to_dict())
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def load_k10(csv_path: Path, json_dir: Path, k_csv: Path, k_json: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in (csv_path, k_csv):
        if path.is_file() and path.stat().st_size > 0:
            frames.append(pd.read_csv(path))
    for d in (json_dir, k_json):
        df = _load_json_dir(d)
        if not df.empty:
            frames.append(df)
    if not frames:
        raise SystemExit(f"no eval rows in {csv_path}/{json_dir} or k anchors")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["val_roc_auc", "k", "star_r"])
    df["k"] = df["k"].astype(int)
    df["star_r"] = df["star_r"].astype(float)
    df = df[df["k"] == FIXED_K].copy()
    df = df.sort_values("star_r").drop_duplicates(subset=["star_r"], keep="last")
    return df.reset_index(drop=True)


def plot_roc_vs_r(df: pd.DataFrame, out_stem: Path) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    sub = df.sort_values("star_r")
    ax.plot(
        sub["star_r"],
        sub["val_roc_auc"],
        marker="o",
        linewidth=1.5,
        color="C0",
        label=f"$k$={FIXED_K}",
    )
    ax.set_xlabel(r"star radius $R^\star$")
    ax.set_ylabel("val ROC AUC")
    ax.set_title(rf"Graph construction: val ROC vs $R^\star$ ($k$={FIXED_K})")
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
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--k-csv", type=Path, default=K_CSV)
    parser.add_argument("--k-json-dir", type=Path, default=K_JSON_DIR)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-fig", type=Path, default=FIG_DIR / "fig_graph_R_vs_roc")
    args = parser.parse_args()

    df = load_k10(args.csv, args.json_dir, args.k_csv, args.k_json_dir)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(df)} radii @ k={FIXED_K})")
    for _, row in df.sort_values("star_r").iterrows():
        print(f"  R★={row['star_r']:g}  val_roc={row['val_roc_auc']:.5f}")
    plot_roc_vs_r(df, args.out_fig)


if __name__ == "__main__":
    main()
