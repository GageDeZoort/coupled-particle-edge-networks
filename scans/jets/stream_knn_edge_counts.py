#!/usr/bin/env python3
"""Stream the Plot-1 train shards and count directed ΔR-kNN edges per jet.

Matches the live graph used in training (``build_delta_r_knn_edges`` / self-
inclusive top-k in Δη–Δφ). The default ``--mode mask`` uses the exact closed
form ``N·min(k,N)`` (verified ≡ ΔR-kNN on the first batch); ``--mode knn``
runs the full top-k construction.

The three disjoint 384k-jet shards are the same windows as Plot 1::

    skip = shard * 384_000,  n_jets = 384_000,  P = 80

For each k ∈ {2,4,…,16} we record the mean and jet-to-jet s.d. of edges/jet
on each shard (multiplicity variation), then pool across shards for the twin
axis (mean of means; RMS of within-shard jet stds as the error bar).

Usage::

    python scans/jets/stream_knn_edge_counts.py
    python scans/jets/stream_knn_edge_counts.py --max-jets 8000   # smoke
    python scans/jets/stream_knn_edge_counts.py --mode knn

Writes
  paper/data/graph_geometry_knn_shards.csv   # per (k, shard)
  paper/data/graph_geometry_knn.csv          # aggregated mean ± std across shards
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

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
    particle_mask_from_four_vectors,
)

DATA = REPO / "paper" / "data"
DEFAULT_RAW = Path("/projects/j/jdezoort/jetclass")
K_GRID = [2, 4, 6, 8, 10, 12, 14, 16]
N_SHARD = 3
SHARD_JETS = 384_000
NUM_PARTICLES = 80


@torch.no_grad()
def count_delta_r_knn_edges(
    x_raw: torch.Tensor,
    *,
    k: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``(B,)`` active directed ΔR-kNN slots per jet (self-inclusive).

    Same neighbour selection as ``build_delta_r_knn_edges``; no edge features.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    if mask is None:
        mask = particle_mask_from_four_vectors(x_raw)
    batch, n_particles, _ = x_raw.shape
    knn_k = min(int(k), n_particles)
    eta, phi = eta_phi_from_four_vectors_torch(x_raw)
    dr = delta_r_torch(
        eta.unsqueeze(-1), phi.unsqueeze(-1), eta.unsqueeze(-2), phi.unsqueeze(-2)
    )
    pair_ok = mask.unsqueeze(-1) & mask.unsqueeze(-2)
    dr = dr.masked_fill(~pair_ok, float("inf"))
    eye = torch.eye(n_particles, device=x_raw.device, dtype=torch.bool).unsqueeze(0)
    dr = torch.where(eye & mask.unsqueeze(-1), torch.zeros_like(dr), dr)
    knn_dist, _ = dr.topk(knn_k, dim=-1, largest=False)
    finite = torch.isfinite(knn_dist) & mask.unsqueeze(-1)
    return finite.sum(dim=(-1, -2)).to(torch.float64)


@torch.no_grad()
def count_edges_from_mask(mask: torch.Tensor, k: int) -> torch.Tensor:
    """Exact closed form for self-inclusive ΔR-kNN: ``N · min(k, N)`` per jet."""
    n = mask.to(torch.float64).sum(dim=-1)
    return n * torch.minimum(n, torch.full_like(n, float(k)))


def verify_mask_equals_knn(x_raw: torch.Tensor, mask: torch.Tensor, ks: list[int]) -> None:
    for k in ks:
        knn = count_delta_r_knn_edges(x_raw, k=k, mask=mask)
        fast = count_edges_from_mask(mask, k)
        diff = (knn - fast).abs().max().item()
        if diff > 0:
            raise RuntimeError(f"mask≠knn at k={k}: max|Δ|={diff}")
    print(f"verified mask ≡ ΔR-kNN on first batch (B={x_raw.size(0)}) for ks={ks}", flush=True)


def stream_shard(
    *,
    data_root: Path,
    shard: int,
    n_jets: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    ks: list[int],
    mode: str,
    verify: bool,
) -> dict[int, tuple[float, float]]:
    """Return {k: (mean, std)} of per-jet edge counts for one train shard.

    Streams once, accumulates multiplicity of particle counts ``N``, then
    evaluates ``N·min(k,N)`` offline (exact for ΔR-kNN). ``std`` is the
    jet-to-jet sample s.d. of that edge count (not the s.d. of shard means).
    """
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
    # hist[n] = number of jets with exactly n real particles (n=0..P).
    hist = np.zeros(NUM_PARTICLES + 1, dtype=np.int64)
    n_seen = 0
    verified = False
    knn_sum = {k: 0.0 for k in ks} if mode == "knn" else None
    knn_sumsq = {k: 0.0 for k in ks} if mode == "knn" else None
    t0 = time.time()
    for batch in loader:
        mask = batch["mask"]
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask)
        mask = mask.bool()
        if verify and not verified:
            x_raw = batch["x_raw"]
            if not torch.is_tensor(x_raw):
                x_raw = torch.as_tensor(x_raw)
            n = min(128, mask.size(0))
            verify_mask_equals_knn(x_raw[:n], mask[:n], ks)
            verified = True
        n_real = mask.sum(dim=-1).cpu().numpy().astype(np.int64)
        for n in n_real:
            hist[int(n)] += 1
        if mode == "knn":
            x_raw = batch["x_raw"].to(device)
            m_dev = mask.to(device)
            for k in ks:
                n_e = count_delta_r_knn_edges(x_raw, k=k, mask=m_dev).cpu().numpy()
                knn_sum[k] += float(n_e.sum())
                knn_sumsq[k] += float((n_e * n_e).sum())
        n_seen += int(mask.size(0))
        if n_seen % 50_000 < batch_size:
            rate = n_seen / max(time.time() - t0, 1e-6)
            print(
                f"  shard={shard} jets={n_seen:,}/{n_jets:,}  ({rate:,.0f} jets/s)",
                flush=True,
            )
    if n_seen == 0:
        raise RuntimeError(f"shard {shard}: no jets streamed")
    print(
        f"shard={shard} done n={n_seen:,} in {time.time() - t0:.1f}s",
        flush=True,
    )
    if mode == "knn":
        out = {}
        for k in ks:
            mu = knn_sum[k] / n_seen
            var = knn_sumsq[k] / n_seen - mu * mu
            # unbiased sample variance
            var = var * n_seen / max(n_seen - 1, 1)
            out[k] = (mu, float(np.sqrt(max(var, 0.0))))
        return out
    out = {}
    ns = np.arange(NUM_PARTICLES + 1, dtype=np.float64)
    w = hist.astype(np.float64)
    for k in ks:
        edges = ns * np.minimum(ns, float(k))
        mu = float((w * edges).sum() / n_seen)
        second = float((w * edges * edges).sum() / n_seen)
        var = (second - mu * mu) * n_seen / max(n_seen - 1, 1)
        out[k] = (mu, float(np.sqrt(max(var, 0.0))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--ks", type=int, nargs="+", default=K_GRID)
    ap.add_argument("--n-shards", type=int, default=N_SHARD)
    ap.add_argument(
        "--shards",
        type=int,
        nargs="+",
        default=None,
        help="Explicit shard indices (default: 0..n-shards-1).",
    )
    ap.add_argument("--shard-jets", type=int, default=SHARD_JETS)
    ap.add_argument(
        "--max-jets",
        type=int,
        default=None,
        help="Cap jets per shard (smoke tests). Default: full shard-jets.",
    )
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--mode",
        choices=("mask", "knn"),
        default="mask",
        help="mask: exact N·min(k,N) closed form (default). knn: full ΔR top-k.",
    )
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip first-batch mask≡knn check.",
    )
    ap.add_argument(
        "--out-shards",
        type=Path,
        default=DATA / "graph_geometry_knn_shards.csv",
    )
    ap.add_argument(
        "--out-agg",
        type=Path,
        default=DATA / "graph_geometry_knn.csv",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    n_jets = int(args.max_jets) if args.max_jets is not None else int(args.shard_jets)
    ks = [int(k) for k in args.ks]
    print(
        f"data_root={args.data_root}  device={device}  mode={args.mode}  "
        f"shards={args.shards or list(range(args.n_shards))}  "
        f"n_jets/shard={n_jets:,}  ks={ks}",
        flush=True,
    )

    shard_ids = args.shards if args.shards is not None else list(range(args.n_shards))
    rows = []
    # Resume: keep existing rows for shards we are not recomputing.
    if args.out_shards.is_file() and args.shards is not None:
        prev = pd.read_csv(args.out_shards)
        keep = prev[~prev["shard"].isin(shard_ids)]
        rows.extend(keep.to_dict(orient="records"))
        print(f"resume: kept {len(keep)} rows from {args.out_shards}", flush=True)

    for shard in shard_ids:
        stats = stream_shard(
            data_root=args.data_root,
            shard=shard,
            n_jets=n_jets,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            ks=ks,
            mode=args.mode,
            verify=(not args.no_verify) and shard == shard_ids[0],
        )
        for k, (mu, sigma) in stats.items():
            rows.append(
                dict(
                    k=k,
                    shard=shard,
                    n_jets=n_jets,
                    train_skip=shard * SHARD_JETS,
                    n_edges_mean=mu,
                    n_edges_std=sigma,
                )
            )
            print(
                f"  k={k:2d} shard={shard} mean_edges={mu:.4f}  std_jet={sigma:.4f}",
                flush=True,
            )
        # Checkpoint after each shard.
        pd.DataFrame(rows).sort_values(["k", "shard"]).to_csv(args.out_shards, index=False)
        print(f"checkpoint {args.out_shards} ({len(rows)} rows)", flush=True)

    shard_df = pd.DataFrame(rows).sort_values(["k", "shard"]).reset_index(drop=True)
    args.out_shards.parent.mkdir(parents=True, exist_ok=True)
    shard_df.to_csv(args.out_shards, index=False)
    print(f"wrote {args.out_shards} ({len(shard_df)} rows)", flush=True)

    agg_rows = []
    for k, sub in shard_df.groupby("k"):
        means = sub["n_edges_mean"].to_numpy(dtype=np.float64)
        # Prefer jet-level stds if present; else fall back to std of shard means.
        if "n_edges_std" in sub.columns and sub["n_edges_std"].notna().all():
            # Pool equal-sized shards: RMS of within-shard jet stds.
            jet_std = float(np.sqrt(np.mean(np.square(sub["n_edges_std"].to_numpy(dtype=np.float64)))))
        else:
            jet_std = float(means.std(ddof=1)) if len(means) > 1 else 0.0
        agg_rows.append(
            dict(
                k=int(k),
                n_edges_mean=float(means.mean()),
                n_edges_std=jet_std,
                n_shards=int(len(means)),
                n_jets_per_shard=n_jets,
                source=f"plot1_train_shards_deltaR_knn_{args.mode}_jetstd",
            )
        )
    agg = pd.DataFrame(agg_rows).sort_values("k").reset_index(drop=True)
    agg.to_csv(args.out_agg, index=False)
    print(f"wrote {args.out_agg}", flush=True)
    print(agg.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
