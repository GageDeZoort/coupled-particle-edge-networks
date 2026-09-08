#!/usr/bin/env python
"""
Pre-build TopTagging star-$R$ hypergraph caches (PyG-compatible, mmap-friendly).

Caches are written under::

  {data-root}/processed/star-r{R}/n{num_particles}/train.pt
  {data-root}/processed/star-r{R}/n{num_particles}/val.pt
  {data-root}/processed/star-r{R}/n{num_particles}/test.pt

Example::

  python scans/jets/build_toptagging_star_graphs.py \\
    --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/toptagging \\
    --radius 0.25 \\
    --num-particles 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cpen.utils.star_graph_cache import (
    build_all_star_graph_caches,
    processed_star_split_dir,
    star_radius_tag,
)
from cpen.utils.toptagging import ensure_raw_splits


def _print_config_banner(args: argparse.Namespace, *, out_dir: Path) -> None:
    line = "=" * 72
    print(line, flush=True)
    print("  TOP TAGGING — STAR-R HYPERGRAPH CACHE BUILD", flush=True)
    print(line, flush=True)
    print(f"  data-root      : {args.data_root}", flush=True)
    print(f"  star-R radius  : {args.radius:g}", flush=True)
    print(f"  num-particles  : {args.num_particles}", flush=True)
    print(f"  cache tag      : {star_radius_tag(args.radius)}", flush=True)
    print(f"  output dir     : {out_dir}", flush=True)
    if args.n_train or args.n_val or args.n_test:
        print(
            f"  subsample      : train={args.n_train} val={args.n_val} test={args.n_test}",
            flush=True,
        )
    else:
        print("  subsample      : full splits (train / val / test)", flush=True)
    if args.rebuild:
        print("  rebuild        : yes (overwrite matching caches)", flush=True)
    print(line, flush=True)
    print(flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-build TopTagging star-R hypergraph caches for CPEN / PyG training"
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Dataset root, e.g. /scratch/gpfs/BHANIN/jdezoort/datasets/toptagging",
    )
    parser.add_argument(
        "--radius",
        type=float,
        required=True,
        help="Delta-R in (eta, phi) for star hyperedge neighborhoods",
    )
    parser.add_argument("--num-particles", type=int, default=128)
    parser.add_argument("--n-train", type=int, default=None, help="Optional train subsample")
    parser.add_argument("--n-val", type=int, default=None, help="Optional val subsample")
    parser.add_argument("--n-test", type=int, default=None, help="Optional test subsample")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download jetnet HDF5 from Zenodo if missing (login node only)",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Only ensure raw HDF5 splits exist; skip graph cache build",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild caches even when metadata matches",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal logging (no tqdm / progress banners)",
    )
    parser.add_argument(
        "--wire-coordinate-dim",
        type=int,
        default=None,
        help=(
            "If set, bake WIRE Laplacian eigenvectors into the primary "
            "{split}.pt (shape [n_jets, N, m]). Recommended when training "
            "with --use-wire."
        ),
    )
    parser.add_argument(
        "--wire-combinatorial-laplacian",
        dest="wire_normalized_laplacian",
        action="store_false",
        help="Use combinatorial Laplacian for baked WIRE coords",
    )
    parser.set_defaults(wire_normalized_laplacian=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    show_progress = not args.quiet

    out_dir = processed_star_split_dir(
        data_root,
        radius=args.radius,
        num_particles=args.num_particles,
    )

    if show_progress:
        _print_config_banner(args, out_dir=out_dir)
    else:
        print(f"data-root: {data_root}", flush=True)
        print(f"star-R radius: {args.radius:g}", flush=True)
        print(f"num-particles: {args.num_particles}", flush=True)

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

    t0 = time.perf_counter()
    build_all_star_graph_caches(
        data_root=str(data_root),
        radius=args.radius,
        num_particles=args.num_particles,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        seed=args.data_seed,
        rebuild=args.rebuild,
        show_progress=show_progress,
        wire_coordinate_dim=args.wire_coordinate_dim,
        wire_normalized_laplacian=args.wire_normalized_laplacian,
    )
    elapsed = time.perf_counter() - t0

    print(flush=True)
    print(f"Finished in {elapsed / 60:.1f} min ({elapsed:.0f}s).", flush=True)
    train_hint = f"Train with: --star-radius {args.radius:g}"
    if args.wire_coordinate_dim is not None:
        train_hint += (
            f" --use-wire --wire-coordinate-dim {args.wire_coordinate_dim}"
        )
    print(train_hint, flush=True)


if __name__ == "__main__":
    main()
