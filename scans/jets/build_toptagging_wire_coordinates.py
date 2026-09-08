#!/usr/bin/env python
"""
Augment an existing TopTagging star-$R$ cache with baked-in WIRE coordinates.

Writes ``wire_coordinates`` into the primary ``{split}.pt`` (same Lightning
datamodule path) and updates ``meta.json``. Prefer building WIRE together with
the star cache via::

  python scans/jets/build_toptagging_star_graphs.py ... --wire-coordinate-dim 8

This script is for migrating caches that already exist without rebuilding
graphs from raw HDF5.

Example::

  python scans/jets/build_toptagging_wire_coordinates.py \\
    --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/toptagging \\
    --radius 0.15 \\
    --num-particles 128 \\
    --wire-coordinate-dim 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cpen.utils.star_graph_cache import (
    processed_star_split_dir,
    star_cache_meta,
    star_radius_tag,
)
from cpen.utils.wire_coordinates import (
    compute_wire_coordinates_batched,
    particle_adjacency_from_incidence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bake WIRE coords into existing TopTagging star {split}.pt files"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--num-particles", type=int, default=128)
    parser.add_argument("--wire-coordinate-dim", type=int, default=8)
    parser.add_argument(
        "--wire-combinatorial-laplacian",
        dest="wire_normalized_laplacian",
        action="store_false",
    )
    parser.set_defaults(wire_normalized_laplacian=True)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to augment",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Recompute even if wire_coordinates already present with matching m",
    )
    return parser.parse_args()


def _augment_split(
    *,
    cache_dir: Path,
    split: str,
    coordinate_dim: int,
    normalized_laplacian: bool,
    chunk_size: int,
    rebuild: bool,
) -> None:
    cache_path = cache_dir / f"{split}.pt"
    meta_path = cache_dir / "meta.json"
    if not cache_path.is_file():
        raise FileNotFoundError(f"Missing star cache {cache_path}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing star meta {meta_path}")

    print(f"[wire] loading {cache_path}", flush=True)
    payload = torch_load(cache_path)
    if "incidence" not in payload or "mask" not in payload:
        raise KeyError(f"{cache_path} missing incidence/mask")

    existing = payload.get("wire_coordinates")
    if (
        existing is not None
        and existing.size(-1) == coordinate_dim
        and not rebuild
    ):
        print(f"[wire] {split}: already has m={coordinate_dim}; skipping", flush=True)
    else:
        n_jets = int(payload["mask"].size(0))
        chunks = []
        t0 = time.perf_counter()
        for start in range(0, n_jets, chunk_size):
            end = min(start + chunk_size, n_jets)
            adj = particle_adjacency_from_incidence(
                payload["incidence"][start:end],
                mask=payload["mask"][start:end],
            )
            chunks.append(
                compute_wire_coordinates_batched(
                    adj,
                    payload["mask"][start:end],
                    coordinate_dim,
                    normalized_laplacian=normalized_laplacian,
                    warn_on_pad=False,
                ).cpu()
            )
            if end == n_jets or end % (chunk_size * 20) == 0:
                rate = end / max(time.perf_counter() - t0, 1e-6)
                print(f"  [{split}] {end:,}/{n_jets:,} ({rate:,.0f} jets/s)", flush=True)
        payload["wire_coordinates"] = torch.cat(chunks, dim=0)
        print(f"[wire] writing {cache_path}", flush=True)
        torch_save(payload, cache_path)

    meta = json.loads(meta_path.read_text())
    meta.update(
        {
            "wire_coordinate_dim": int(coordinate_dim),
            "wire_normalized_laplacian": bool(normalized_laplacian),
            "wire_standardize": True,
            "wire_canonicalize_sign": True,
            "wire_source": "particle-cooccurrence-STS",
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[wire] updated {meta_path}", flush=True)


def torch_load(path: Path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def torch_save(obj, path: Path) -> None:
    import torch

    torch.save(obj, path)


def main() -> None:
    args = parse_args()
    cache_dir = processed_star_split_dir(
        args.data_root,
        radius=args.radius,
        num_particles=args.num_particles,
    )
    print("=" * 72, flush=True)
    print("  TOP TAGGING — BAKE WIRE INTO PRIMARY STAR CACHE", flush=True)
    print("=" * 72, flush=True)
    print(f"  cache dir : {cache_dir}", flush=True)
    print(f"  tag       : {star_radius_tag(args.radius)}", flush=True)
    print(f"  wire m    : {args.wire_coordinate_dim}", flush=True)
    print("=" * 72, flush=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    t0 = time.perf_counter()
    for split in splits:
        _augment_split(
            cache_dir=cache_dir,
            split=split,
            coordinate_dim=args.wire_coordinate_dim,
            normalized_laplacian=args.wire_normalized_laplacian,
            chunk_size=args.chunk_size,
            rebuild=args.rebuild,
        )
    # Sanity: meta helper shape for docs
    _ = star_cache_meta(
        radius=args.radius,
        num_particles=args.num_particles,
        max_jets=None,
        seed=42,
        wire_coordinate_dim=args.wire_coordinate_dim,
        wire_normalized_laplacian=args.wire_normalized_laplacian,
    )
    elapsed = time.perf_counter() - t0
    print(flush=True)
    print(f"Finished in {elapsed / 60:.1f} min.", flush=True)
    print(
        f"Train with: --star-radius {args.radius:g} --use-wire "
        f"--wire-coordinate-dim {args.wire_coordinate_dim}",
        flush=True,
    )


if __name__ == "__main__":
    main()
