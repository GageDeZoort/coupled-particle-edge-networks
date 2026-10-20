#!/usr/bin/env python3
"""Stream Plot-1 train shards and measure mean star support |S| per jet.

Uses DataLoader (same pattern as ``stream_knn_edge_counts.py``) so peak RSS
stays bounded. Matches ``build_star_radius_graph`` neighbour selection.

Per-jet scalar = mean |S| over active centers. CSV error bars = sample s.d.
over all jets in the shards examined (pooled across equal-sized shards).

Usage::

    python scans/jets/stream_star_support_counts.py
    python scans/jets/stream_star_support_counts.py --max-jets 8000   # smoke

Writes
  paper/data/graph_geometry_star_shards.csv
  paper/data/graph_geometry_star.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from cpen.apps.jets.jetclass_stream import JetClassStreamDataset  # noqa: E402
from cpen.graphs.graph_star import (  # noqa: E402
    delta_r_torch,
    eta_phi_from_four_vectors_torch,
)

DATA = REPO / "paper" / "data"
DEFAULT_RAW = Path("/projects/j/jdezoort/jetclass")
R_GRID = [0.05, 0.1, 0.15, 0.2, 0.25]
SHARD_JETS = 384_000
NUM_PARTICLES = 80


@torch.no_grad()
def mean_star_support(mask: torch.Tensor, dr: torch.Tensor, radius: float) -> torch.Tensor:
    """``(B,)`` mean |S| over active centers (self-inclusive radius ball)."""
    pair_valid = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    within = (dr < float(radius)) & pair_valid
    eye = torch.eye(mask.size(1), dtype=torch.bool, device=mask.device).unsqueeze(0)
    support = within | (eye & mask.unsqueeze(-1))
    sizes = support.sum(-1).float()
    n_act = mask.float().sum(-1).clamp_min(1.0)
    return (sizes * mask.float()).sum(-1) / n_act


def stream_shard(
    *,
    data_root: Path,
    shard: int,
    n_jets: int,
    batch_size: int,
    num_workers: int,
    radii: list[float],
) -> dict[float, tuple[float, float, int]]:
    skip = shard * SHARD_JETS
    ds = JetClassStreamDataset(
        data_root,
        "train",
        n_jets=n_jets,
        num_particles=NUM_PARTICLES,
        feature_config="kin7",
        shuffle_buffer=1,
        shuffle_seed=0,
        skip_jets=skip,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )
    sums = {r: 0.0 for r in radii}
    sumsq = {r: 0.0 for r in radii}
    n_seen = 0
    t0 = time.time()
    for batch in loader:
        x_raw = batch["x_raw"]
        mask = batch["mask"].bool()
        eta, phi = eta_phi_from_four_vectors_torch(x_raw)
        dr = delta_r_torch(
            eta.unsqueeze(-1), phi.unsqueeze(-1), eta.unsqueeze(-2), phi.unsqueeze(-2)
        )
        for r in radii:
            s = mean_star_support(mask, dr, r)
            sums[r] += float(s.sum().item())
            sumsq[r] += float((s * s).sum().item())
        n_seen += int(mask.size(0))
        if n_seen % 50_000 < batch_size:
            rate = n_seen / max(time.time() - t0, 1e-6)
            print(
                f"  shard={shard} jets={n_seen:,}/{n_jets:,}  ({rate:,.0f} jets/s)",
                flush=True,
            )
    if n_seen == 0:
        raise RuntimeError(f"shard {shard}: no jets streamed")
    print(f"shard={shard} done n={n_seen:,} in {time.time() - t0:.1f}s", flush=True)
    out = {}
    for r in radii:
        mu = sums[r] / n_seen
        var = (sumsq[r] / n_seen - mu * mu) * n_seen / max(n_seen - 1, 1)
        out[r] = (mu, float(np.sqrt(max(var, 0.0))), n_seen)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--radii", type=float, nargs="+", default=R_GRID)
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--shard-jets", type=int, default=SHARD_JETS)
    ap.add_argument("--max-jets", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--out-shards", type=Path, default=DATA / "graph_geometry_star_shards.csv")
    ap.add_argument("--out-agg", type=Path, default=DATA / "graph_geometry_star.csv")
    args = ap.parse_args()

    n_jets = int(args.max_jets) if args.max_jets is not None else int(args.shard_jets)
    radii = [float(r) for r in args.radii]
    print(
        f"data_root={args.data_root} shards={args.shards} "
        f"n_jets/shard={n_jets:,} radii={radii} batch={args.batch_size} "
        f"workers={args.num_workers}",
        flush=True,
    )

    rows: list[dict] = []
    if args.out_shards.is_file():
        prev = pd.read_csv(args.out_shards)
        keep = prev[~prev["shard"].isin(args.shards)]
        if len(keep):
            rows.extend(keep.to_dict(orient="records"))
            print(f"resume: kept {len(keep)} rows", flush=True)

    for shard in args.shards:
        stats = stream_shard(
            data_root=args.data_root,
            shard=shard,
            n_jets=n_jets,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            radii=radii,
        )
        for r, (mu, sigma, n) in stats.items():
            rows.append(
                dict(
                    star_r=r,
                    shard=shard,
                    n_jets=n,
                    train_skip=shard * SHARD_JETS,
                    mean_support=mu,
                    support_std=sigma,
                )
            )
            print(
                f"  R={r:g} shard={shard} mean_|S|={mu:.4f} std_jet={sigma:.4f}",
                flush=True,
            )
        pd.DataFrame(rows).sort_values(["star_r", "shard"]).to_csv(
            args.out_shards, index=False
        )
        print(f"checkpoint {args.out_shards} ({len(rows)} rows)", flush=True)

    shard_df = pd.DataFrame(rows)
    agg_rows = []
    for r, sub in shard_df.groupby("star_r"):
        n_tot = int(sub["n_jets"].sum())
        sum_x = float((sub["mean_support"] * sub["n_jets"]).sum())
        sum_x2 = float(
            (
                sub["support_std"] ** 2 * (sub["n_jets"] - 1).clip(lower=0)
                + sub["n_jets"] * sub["mean_support"] ** 2
            ).sum()
        )
        mu = sum_x / n_tot
        var = (sum_x2 / n_tot - mu * mu) * n_tot / max(n_tot - 1, 1)
        agg_rows.append(
            dict(
                star_r=float(r),
                mean_support=mu,
                support_std=float(np.sqrt(max(var, 0.0))),
                n_shards=int(len(sub)),
                n_jets_per_shard=int(sub["n_jets"].iloc[0]),
                source="plot1_train_shards_star_support_jetstd",
            )
        )
    agg = pd.DataFrame(agg_rows).sort_values("star_r").reset_index(drop=True)
    args.out_agg.parent.mkdir(parents=True, exist_ok=True)
    args.out_shards.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.out_agg, index=False)
    print(f"wrote {args.out_agg}", flush=True)
    print(agg.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
