#!/usr/bin/env python
"""
Pre-build TopTagging graph caches for offline SLURM training.

Optional optimization only — training defaults to live on-the-fly graphs from
local HDF5 and does not require this step.

Run on a high-memory login node if you want faster IO during training.
Graph caches are written under:

  {data-root}/processed/{knn}/n{num_particles}/train.pt
  {data-root}/processed/{knn}/n{num_particles}/val.pt
  {data-root}/processed/{knn}/n{num_particles}/test.pt

Example:
  python scans/jets/build_toptagging_graphs.py \\
    --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/toptagging \\
    --graph-construction 8-NN \\
    --n-train 25000 --n-val 25000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cpen.utils.graph_cache import build_all_graph_caches, processed_split_dir
from cpen.utils.toptagging import ensure_raw_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-build TopTagging graph dataset caches")
    parser.add_argument(
        "--data-root",
        required=True,
        help="Dataset root, e.g. /scratch/gpfs/BHANIN/jdezoort/datasets/toptagging",
    )
    parser.add_argument(
        "--graph-construction",
        default="8-NN",
        help='kNN graph spec, e.g. "8-NN" or "12-NN"',
    )
    parser.add_argument("--num-particles", type=int, default=128)
    parser.add_argument("--n-train", type=int, default=None, help="Optional train subsample")
    parser.add_argument("--n-val", type=int, default=None, help="Optional val subsample")
    parser.add_argument("--n-test", type=int, default=None, help="Optional test subsample")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download jetnet HDF5 from Zenodo if missing (use on login node)",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Only ensure raw jetnet HDF5 files exist; skip graph cache build",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild graph caches even if existing metadata matches",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    print(f"data-root: {data_root}", flush=True)
    print(f"graph-construction: {args.graph_construction}", flush=True)

    if args.download or args.raw_only:
        print("Ensuring raw jetnet HDF5 splits exist...", flush=True)
        ensure_raw_splits(
            str(data_root),
            num_particles=args.num_particles,
            download=args.download,
        )
        print("Raw splits ready.", flush=True)

    if args.raw_only:
        return

    print("Building graph caches (uses local HDF5; no Zenodo if raw files exist)...", flush=True)
    out_dir = build_all_graph_caches(
        data_root=str(data_root),
        graph_construction=args.graph_construction,
        num_particles=args.num_particles,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.data_seed,
        download=False,
        rebuild=args.rebuild,
    )
    print(f"Graph caches ready under {out_dir}", flush=True)


if __name__ == "__main__":
    main()
