#!/usr/bin/env python
"""
Shard / merge TopTagging hierarchical graph caches for Della CPU QOS ``test`` (~1h).

Materializing full splits with DBSCAN + virtual structure is multi-hour and
RAM-heavy if done monolithically. This script:

  1. Builds index-range **shards** (``shards/{split}/shard_XXX.pt``)
  2. **Merges** shards into the canonical ``{split}.pt`` + ``meta.json``

Example (one shard)::

  python scans/jets/build_toptagging_hier_cputest.py \\
    --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/toptagging \\
    --stage shard --split train --shard-index 0 --n-shards 20 \\
    --k 8 --eps 0.08 --num-particles 128 --rebuild

Prefer the submit helper::

  bash scans/jets/submit_hier_graphs_cputest.sh
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cpen.utils.graph_hierarchical import hierarchical_construction_tag
from cpen.utils.hier_graph_cache import (
    PAYLOAD_KEYS,
    LiveHierJetDataset,
    hier_cache_meta,
    materialize_hier_batch,
    processed_hier_split_dir,
    write_hier_cache_split_count,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", required=True)
    p.add_argument(
        "--stage",
        choices=("shard", "merge"),
        required=True,
        help="shard: build one index range; merge: concat shards → {split}.pt",
    )
    p.add_argument("--split", required=True, choices=("train", "val", "test"))
    p.add_argument("--k", type=int, default=8, help="Directed ΔR-kNN degree")
    p.add_argument("--eps", type=float, default=0.08, help="DBSCAN eps in ΔR")
    p.add_argument("--min-samples", type=int, default=2)
    p.add_argument("--n-virtual-nodes", type=int, default=1)
    p.add_argument("--n-virtual-edges", type=int, default=1)
    p.add_argument("--max-dbscan-edges", type=int, default=32)
    p.add_argument("--num-particles", type=int, default=128)
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--shard-index", type=int, default=None, help="0-based shard id")
    p.add_argument("--n-shards", type=int, default=None, help="Total shards for this split")
    p.add_argument(
        "--build-batch-size",
        type=int,
        default=64,
        help="Jets per build_hierarchical_graph call (DBSCAN is per-jet in-batch)",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=2000,
        help="Log progress every this many jets within the shard",
    )
    p.add_argument("--rebuild", action="store_true")
    return p.parse_args()


def shard_dir(cache_dir: Path, split: str) -> Path:
    return cache_dir / "shards" / split


def shard_path(cache_dir: Path, split: str, shard_index: int) -> Path:
    return shard_dir(cache_dir, split) / f"shard_{shard_index:03d}.pt"


def shard_meta_path(cache_dir: Path, split: str, shard_index: int) -> Path:
    return shard_dir(cache_dir, split) / f"shard_{shard_index:03d}.json"


def index_range(n_jets: int, shard_index: int, n_shards: int) -> tuple[int, int]:
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1; got {n_shards}")
    if shard_index < 0 or shard_index >= n_shards:
        raise ValueError(f"shard_index must be in [0, {n_shards}); got {shard_index}")
    base = n_jets // n_shards
    rem = n_jets % n_shards
    if shard_index < rem:
        start = shard_index * (base + 1)
        end = start + base + 1
    else:
        start = rem * (base + 1) + (shard_index - rem) * base
        end = start + base
    return start, end


def _hier_kwargs(args: argparse.Namespace) -> dict:
    return dict(
        k=args.k,
        eps=args.eps,
        min_samples=args.min_samples,
        n_virtual_nodes=args.n_virtual_nodes,
        n_virtual_edges=args.n_virtual_edges,
        max_dbscan_edges=args.max_dbscan_edges,
    )


def _cache_dir(args: argparse.Namespace) -> Path:
    return processed_hier_split_dir(
        args.data_root,
        num_particles=args.num_particles,
        **_hier_kwargs(args),
    )


# Sparse COO side-channels are padded to batch-max nnz; widths differ across chunks.
_VARIABLE_WIDTH_KEYS = ("incidence_node", "incidence_edge")


def cat_payloads(chunks: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not chunks:
        raise ValueError("no payload chunks to concatenate")
    out: dict[str, torch.Tensor] = {}
    for key in PAYLOAD_KEYS:
        if key in _VARIABLE_WIDTH_KEYS:
            width = max(int(c[key].size(1)) for c in chunks)
            padded: list[torch.Tensor] = []
            for c in chunks:
                t = c[key]
                if t.size(1) < width:
                    pad = t.new_zeros(t.size(0), width - t.size(1))
                    t = torch.cat([t, pad], dim=1)
                padded.append(t)
            out[key] = torch.cat(padded, dim=0)
        else:
            out[key] = torch.cat([c[key] for c in chunks], dim=0)
    return out


def build_shard(args: argparse.Namespace) -> None:
    if args.shard_index is None or args.n_shards is None:
        raise SystemExit("--stage shard requires --shard-index and --n-shards")

    cache_dir = _cache_dir(args)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_pt = shard_path(cache_dir, args.split, args.shard_index)
    out_meta = shard_meta_path(cache_dir, args.split, args.shard_index)

    if out_pt.is_file() and out_meta.is_file() and not args.rebuild:
        print(f"[skip] shard exists: {out_pt}", flush=True)
        return

    ds = LiveHierJetDataset(
        data_root=args.data_root,
        split=args.split,
        num_particles=args.num_particles,
        max_jets=None,
        seed=args.data_seed,
    )
    n_jets = len(ds)
    start, end = index_range(n_jets, args.shard_index, args.n_shards)
    n_shard = end - start
    tag = hierarchical_construction_tag(**_hier_kwargs(args))
    print(
        f"[shard] split={args.split} {tag} "
        f"shard={args.shard_index}/{args.n_shards} jets=[{start}:{end}) "
        f"n={n_shard} / {n_jets}",
        flush=True,
    )
    if n_shard <= 0:
        raise SystemExit(f"Empty shard range for index {args.shard_index}")

    t0 = time.time()
    chunks: list[dict[str, torch.Tensor]] = []
    bs = args.build_batch_size
    done = 0
    hier_kw = _hier_kwargs(args)
    for batch_start in range(start, end, bs):
        batch_end = min(batch_start + bs, end)
        samples = [ds[i] for i in range(batch_start, batch_end)]
        chunks.append(materialize_hier_batch(samples, **hier_kw))
        done += batch_end - batch_start
        if done == n_shard or done % args.progress_every < bs:
            rate = done / max(time.time() - t0, 1e-6)
            print(
                f"[shard] progress {done}/{n_shard}  ({rate:.1f} jets/s)",
                flush=True,
            )

    print("[shard] concatenating chunks…", flush=True)
    payload = cat_payloads(chunks)
    del chunks
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_pt)

    meta = {
        "stage": "shard",
        "split": args.split,
        "shard_index": args.shard_index,
        "n_shards": args.n_shards,
        "start": start,
        "end": end,
        "n_jets": n_shard,
        "num_particles": args.num_particles,
        "seed": args.data_seed,
        **hier_cache_meta(
            **hier_kw,
            num_particles=args.num_particles,
            max_jets=None,
            seed=args.data_seed,
        ),
        "payload_keys": list(PAYLOAD_KEYS),
        "shapes": {key: list(payload[key].shape) for key in PAYLOAD_KEYS},
    }
    out_meta.write_text(json.dumps(meta, indent=2) + "\n")
    elapsed = time.time() - t0
    print(
        f"[shard] wrote {out_pt}  ({n_shard} jets in {elapsed / 60:.1f} min)",
        flush=True,
    )


def merge_shards(args: argparse.Namespace) -> None:
    if args.n_shards is None:
        raise SystemExit("--stage merge requires --n-shards")

    cache_dir = _cache_dir(args)
    out_pt = cache_dir / f"{args.split}.pt"
    meta_path = cache_dir / "meta.json"

    if out_pt.is_file() and meta_path.is_file() and not args.rebuild:
        print(f"[skip] merged split exists: {out_pt}", flush=True)
        return

    paths = [shard_path(cache_dir, args.split, i) for i in range(args.n_shards)]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(
            "Missing shard files:\n  " + "\n  ".join(str(p) for p in missing)
        )

    print(f"[merge] loading {len(paths)} shards (mmap) → {out_pt}", flush=True)
    t0 = time.time()
    shards = [
        torch.load(p, map_location="cpu", mmap=True, weights_only=False) for p in paths
    ]
    for i, sh in enumerate(shards):
        for key in PAYLOAD_KEYS:
            if key not in sh:
                raise KeyError(f"shard {i} missing key {key!r}")

    payload = cat_payloads(list(shards))
    n_jets = int(payload["labels"].shape[0])
    print(f"[merge] concatenated n_jets={n_jets}; saving…", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_pt)

    hier_kw = _hier_kwargs(args)
    meta = hier_cache_meta(
        **hier_kw,
        num_particles=args.num_particles,
        max_jets=None,
        seed=args.data_seed,
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    write_hier_cache_split_count(cache_dir=cache_dir, split=args.split, n_jets=n_jets)
    elapsed = time.time() - t0
    print(
        f"[merge] wrote {out_pt} (n_jets={n_jets}, shards={args.n_shards}) "
        f"in {elapsed / 60:.1f} min",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    tag = hierarchical_construction_tag(**_hier_kwargs(args))
    print(f"data-root: {args.data_root}", flush=True)
    print(f"stage: {args.stage}  split: {args.split}", flush=True)
    print(
        f"tag: {tag}  num-particles: {args.num_particles}  "
        f"k={args.k} eps={args.eps} vn={args.n_virtual_nodes} ve={args.n_virtual_edges}",
        flush=True,
    )
    if args.stage == "shard":
        build_shard(args)
    else:
        merge_shards(args)


if __name__ == "__main__":
    main()
