#!/usr/bin/env python
"""
Build JetClassLite particle archives (ParT-full features).

Samples stratified jets from the official JetClass ROOT splits and writes
TopTagging-style torch archives under::

  {cache-root}/lite/n{num_particles}/{train,val,test}.pt

Defaults match a Lite scale of 5M / 250k / 1M train/val/test.

Example::

  python scans/jets/build_jetclass_lite.py \\
    --source-root /tigress/jdezoort/jetclass \\
    --cache-root /scratch/gpfs/BHANIN/jdezoort/datasets/jetclass
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

from cpen.utils.jetclass import (
    DEFAULT_LITE_COUNTS,
    DEFAULT_NUM_PARTICLES,
    build_all_lite_archives,
    lite_archive_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build JetClassLite ParT-full particle archives")
    parser.add_argument(
        "--source-root",
        default="/tigress/jdezoort/jetclass",
        help="Official JetClass ROOT root (contains train_100M/, val_5M/, test_20M/)",
    )
    parser.add_argument(
        "--cache-root",
        required=True,
        help="Where to write lite archives, e.g. /scratch/.../datasets/jetclass",
    )
    parser.add_argument("--num-particles", type=int, default=DEFAULT_NUM_PARTICLES)
    parser.add_argument("--n-train", type=int, default=DEFAULT_LITE_COUNTS["train"])
    parser.add_argument("--n-val", type=int, default=DEFAULT_LITE_COUNTS["val"])
    parser.add_argument("--n-test", type=int, default=DEFAULT_LITE_COUNTS["test"])
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = lite_archive_dir(args.cache_root, num_particles=args.num_particles)
    print("=" * 72, flush=True)
    print("  JETCLASS LITE — PARTICLE ARCHIVE BUILD (ParT full)", flush=True)
    print("=" * 72, flush=True)
    print(f"  source-root   : {args.source_root}", flush=True)
    print(f"  cache-root    : {args.cache_root}", flush=True)
    print(f"  num-particles : {args.num_particles}", flush=True)
    print(
        f"  counts        : train={args.n_train:,} val={args.n_val:,} test={args.n_test:,}",
        flush=True,
    )
    print(f"  output dir    : {out}", flush=True)
    print("=" * 72, flush=True)

    t0 = time.perf_counter()
    build_all_lite_archives(
        source_root=args.source_root,
        cache_root=args.cache_root,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        num_particles=args.num_particles,
        seed=args.data_seed,
        rebuild=args.rebuild,
    )
    elapsed = time.perf_counter() - t0
    print(f"Finished in {elapsed / 60:.1f} min ({elapsed:.0f}s).", flush=True)
    print("Next: build_jetclass_lite_star_graphs.py --cache-root ... --radius 0.15", flush=True)


if __name__ == "__main__":
    main()
