#!/usr/bin/env python
"""
Shard / merge TopTagging kNN graph caches for Della CPU QOS ``test`` (~1h).

Materializing the full train split (~1.21M jets, n=128, k=8) is multi-hour and
RAM-heavy if done monolithically. This script:

  1. Builds index-range **shards** (``shards/{split}/shard_XXX.pt``)
  2. **Merges** shards into the canonical ``{split}.pt`` + ``meta.json`` that
     ``CachedGraphJetDataset`` already expects

Example (one shard)::

  python scans/jets/build_toptagging_knn_cputest.py \\
    --data-root /scratch/gpfs/BHANIN/jdezoort/datasets/toptagging \\
    --stage shard --split train --shard-index 0 --n-shards 10 \\
    --graph-construction 8-NN --num-particles 128 --rebuild

Prefer the submit helper::

  bash scans/jets/submit_knn_graphs_cputest.sh
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

from cpen.utils.graph_cache import (
    LiveGraphJetDataset,
    cache_meta,
    finalize_incidence_storage,
    materialize_graph_batch,
    processed_split_dir,
    write_graph_cache_split_count,
)
from cpen.utils.graphs import parse_graph_construction


PAYLOAD_KEYS = (
    "x",
    "edge_x",
    "incidence",
    "incidence_node",
    "incidence_edge",
    "incidence_nnz",
    "node_degree_inv",
    "edge_degree_inv",
    "mask",
    "z",
    "labels",
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
    p.add_argument("--graph-construction", default="8-NN")
    p.add_argument("--num-particles", type=int, default=128)
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--shard-index", type=int, default=None, help="0-based shard id")
    p.add_argument("--n-shards", type=int, default=None, help="Total shards for this split")
    p.add_argument(
        "--build-batch-size",
        type=int,
        default=256,
        help="Jets per materialize_graph_batch call (matches monolithic builder)",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=5000,
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
    # Contiguous partitions; last shard absorbs the remainder.
    base = n_jets // n_shards
    rem = n_jets % n_shards
    # First `rem` shards get base+1 jets.
    if shard_index < rem:
        start = shard_index * (base + 1)
        end = start + base + 1
    else:
        start = rem * (base + 1) + (shard_index - rem) * base
        end = start + base
    return start, end


def assemble_payload(samples_out: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    x = torch.stack([o["x"] for o in samples_out])
    edge_x = torch.stack([o["edge_x"] for o in samples_out])
    incidence = torch.stack([o["incidence"] for o in samples_out])
    mask = torch.stack([o["mask"] for o in samples_out])
    z = torch.stack([o["z"] for o in samples_out])
    labels = torch.stack([o["y"] for o in samples_out]).view(-1).to(torch.int64)
    fin = finalize_incidence_storage(incidence)
    return {
        "x": x,
        "edge_x": edge_x,
        **fin,
        "mask": mask,
        "z": z,
        "labels": labels,
    }


def build_shard(args: argparse.Namespace) -> None:
    if args.shard_index is None or args.n_shards is None:
        raise SystemExit("--stage shard requires --shard-index and --n-shards")

    cache_dir = processed_split_dir(
        args.data_root,
        graph_construction=args.graph_construction,
        num_particles=args.num_particles,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_pt = shard_path(cache_dir, args.split, args.shard_index)
    out_meta = shard_meta_path(cache_dir, args.split, args.shard_index)

    if out_pt.is_file() and out_meta.is_file() and not args.rebuild:
        print(f"[skip] shard exists: {out_pt}", flush=True)
        return

    k = parse_graph_construction(args.graph_construction)
    # Full split in the live dataset (no max_jets); we slice by index range.
    ds = LiveGraphJetDataset(
        data_root=args.data_root,
        split=args.split,
        graph_construction=args.graph_construction,
        num_particles=args.num_particles,
        max_jets=None,
        seed=args.data_seed,
    )
    n_jets = len(ds)
    start, end = index_range(n_jets, args.shard_index, args.n_shards)
    n_shard = end - start
    print(
        f"[shard] split={args.split} {args.graph_construction} "
        f"shard={args.shard_index}/{args.n_shards} jets=[{start}:{end}) "
        f"n={n_shard} / {n_jets}",
        flush=True,
    )
    if n_shard <= 0:
        raise SystemExit(f"Empty shard range for index {args.shard_index}")

    t0 = time.time()
    outs: list[dict[str, torch.Tensor]] = []
    bs = args.build_batch_size
    done = 0
    for batch_start in range(start, end, bs):
        batch_end = min(batch_start + bs, end)
        samples = [ds[i] for i in range(batch_start, batch_end)]
        outs.extend(materialize_graph_batch(samples, k=k))
        done += batch_end - batch_start
        if done == n_shard or done % args.progress_every < bs:
            rate = done / max(time.time() - t0, 1e-6)
            print(
                f"[shard] progress {done}/{n_shard}  ({rate:.1f} jets/s)",
                flush=True,
            )

    print("[shard] stacking + finalize…", flush=True)
    payload = assemble_payload(outs)
    del outs
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
        "graph_construction": args.graph_construction,
        "num_particles": args.num_particles,
        "seed": args.data_seed,
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

    cache_dir = processed_split_dir(
        args.data_root,
        graph_construction=args.graph_construction,
        num_particles=args.num_particles,
    )
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

    payload = {key: torch.cat([sh[key] for sh in shards], dim=0) for key in PAYLOAD_KEYS}
    n_jets = int(payload["labels"].shape[0])
    print(f"[merge] concatenated n_jets={n_jets}; saving…", flush=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_pt)

    meta = cache_meta(
        graph_construction=args.graph_construction,
        num_particles=args.num_particles,
        max_jets=None,
        seed=args.data_seed,
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    write_graph_cache_split_count(
        data_root=args.data_root,
        split=args.split,
        graph_construction=args.graph_construction,
        num_particles=args.num_particles,
        n_jets=n_jets,
    )
    elapsed = time.time() - t0
    print(
        f"[merge] wrote {out_pt} (n_jets={n_jets}, shards={args.n_shards}) "
        f"in {elapsed / 60:.1f} min",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    print(f"data-root: {args.data_root}", flush=True)
    print(f"stage: {args.stage}  split: {args.split}", flush=True)
    print(
        f"graph-construction: {args.graph_construction}  "
        f"num-particles: {args.num_particles}",
        flush=True,
    )
    if args.stage == "shard":
        build_shard(args)
    else:
        merge_shards(args)


if __name__ == "__main__":
    main()
