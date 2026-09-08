#!/usr/bin/env python
"""
Pre-build JetClassLite star-$R$ hypergraph caches from Lite particle archives.

Requires Lite archives (``build_jetclass_lite`` / ``--build-lite`` / class shards).
Writes::

  {cache-root}/processed/star-r{R}/n{num_particles}/{train,val,test}.pt

Examples::

  # Full one-shot build (long; prefer high-mem cpu QOS)
  python scans/jets/build_jetclass_lite_star_graphs.py \\
    --cache-root /scratch/gpfs/BHANIN/jdezoort/datasets/jetclass \\
    --radius 0.15 --build-lite --rebuild

  # cputest-sized pieces (see build_jetclass_lite_cputest.slurm):
  python ... --stage lite-shard --splits train --lite-class-shard 0 --rebuild
  python ... --stage lite-merge --splits train --rebuild
  python ... --stage lite --splits val,test --rebuild
  python ... --stage graphs --splits train --rebuild
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
    N_CLASSES,
    build_all_lite_archives,
    build_or_load_lite_class_shard,
    merge_lite_class_shards,
)
from cpen.utils.jetclass_star_cache import build_all_jetclass_star_caches
from cpen.utils.star_graph_cache import processed_star_split_dir, star_radius_tag


def _parse_splits(raw: str | None) -> list[str] | None:
    if raw is None or raw.strip() == "" or raw.strip().lower() == "all":
        return None
    splits = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [s for s in splits if s not in {"train", "val", "test"}]
    if unknown:
        raise ValueError(f"Unknown splits {unknown}; expected train|val|test")
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build JetClassLite star-R graph caches")
    parser.add_argument(
        "--source-root",
        default="/tigress/jdezoort/jetclass",
        help="Official JetClass ROOT root (lite sampling)",
    )
    parser.add_argument(
        "--cache-root",
        required=True,
        help="JetClass cache root (lite archives + processed graphs)",
    )
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--num-particles", type=int, default=DEFAULT_NUM_PARTICLES)
    parser.add_argument(
        "--stage",
        choices=("all", "lite", "lite-shard", "lite-merge", "graphs"),
        default="all",
        help=(
            "Pipeline stage. Use lite-shard/lite-merge to break train lite into "
            "1h cputest jobs; graphs builds star caches from existing lite archives."
        ),
    )
    parser.add_argument(
        "--splits",
        default="all",
        help="Comma-separated subset: train,val,test (default: all)",
    )
    parser.add_argument(
        "--lite-class-shard",
        type=int,
        default=None,
        help="With --stage lite-shard: class index in [0, 9] (train recommended)",
    )
    parser.add_argument(
        "--build-lite",
        action="store_true",
        help="Deprecated alias: with --stage all, also build lite archives",
    )
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-val", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = _parse_splits(args.splits)
    stage = args.stage
    if args.build_lite and stage not in {"all", "lite"}:
        raise ValueError("--build-lite is only valid with --stage all|lite")

    # --stage all: graphs always; lite only if --build-lite (legacy BUILD_LITE=1).
    do_lite = stage == "lite" or (stage == "all" and args.build_lite)
    do_graphs = stage in {"all", "graphs"}
    do_shard = stage == "lite-shard"
    do_merge = stage == "lite-merge"

    out_dir = processed_star_split_dir(
        args.cache_root,
        radius=args.radius,
        num_particles=args.num_particles,
    )
    show = not args.quiet
    if show:
        print("=" * 72, flush=True)
        print("  JETCLASS LITE — CACHE BUILD", flush=True)
        print("=" * 72, flush=True)
        print(f"  cache-root    : {args.cache_root}", flush=True)
        print(f"  stage         : {stage}", flush=True)
        print(f"  splits        : {splits or ['train', 'val', 'test']}", flush=True)
        print(f"  star-R radius : {args.radius:g}", flush=True)
        print(f"  num-particles : {args.num_particles}", flush=True)
        print(f"  cache tag     : {star_radius_tag(args.radius)}", flush=True)
        print(f"  output dir    : {out_dir}", flush=True)
        print("=" * 72, flush=True)

    n_train = DEFAULT_LITE_COUNTS["train"] if args.n_train is None else args.n_train
    n_val = DEFAULT_LITE_COUNTS["val"] if args.n_val is None else args.n_val
    n_test = DEFAULT_LITE_COUNTS["test"] if args.n_test is None else args.n_test
    counts = {"train": n_train, "val": n_val, "test": n_test}
    active_splits = splits or ["train", "val", "test"]

    t0 = time.perf_counter()

    if do_shard:
        if args.lite_class_shard is None:
            raise ValueError("--stage lite-shard requires --lite-class-shard INT")
        if args.lite_class_shard < 0 or args.lite_class_shard >= N_CLASSES:
            raise ValueError(
                f"--lite-class-shard must be in [0, {N_CLASSES - 1}]; "
                f"got {args.lite_class_shard}"
            )
        for split in active_splits:
            build_or_load_lite_class_shard(
                source_root=args.source_root,
                cache_root=args.cache_root,
                split=split,
                class_idx=args.lite_class_shard,
                n_jets=counts[split],
                num_particles=args.num_particles,
                seed=args.data_seed,
                rebuild=args.rebuild,
            )

    if do_merge:
        for split in active_splits:
            merge_lite_class_shards(
                cache_root=args.cache_root,
                split=split,
                n_jets=counts[split],
                num_particles=args.num_particles,
                seed=args.data_seed,
                source_root=args.source_root,
                rebuild=args.rebuild,
            )

    if do_lite:
        if show:
            print(
                f"  lite counts   : train={n_train:,} val={n_val:,} test={n_test:,}",
                flush=True,
            )
        build_all_lite_archives(
            source_root=args.source_root,
            cache_root=args.cache_root,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            num_particles=args.num_particles,
            seed=args.data_seed,
            rebuild=args.rebuild,
            splits=active_splits,
        )

    if do_graphs:
        build_all_jetclass_star_caches(
            cache_root=args.cache_root,
            radius=args.radius,
            num_particles=args.num_particles,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            seed=args.data_seed,
            rebuild=args.rebuild,
            show_progress=show,
            splits=active_splits,
        )

    elapsed = time.perf_counter() - t0
    print(f"Finished in {elapsed / 60:.1f} min ({elapsed:.0f}s).", flush=True)
    if do_graphs:
        print(f"Train with: --dataset jetclass --star-radius {args.radius:g}", flush=True)


if __name__ == "__main__":
    main()
