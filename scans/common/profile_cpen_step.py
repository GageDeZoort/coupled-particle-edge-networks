"""Profile a single CPEN train step to locate the real bottleneck.

Run on a GPU node:
    PYTHONPATH=src python scans/common/profile_cpen_step.py \
        --width 256 --depth 4 --batch-size 128 --num-particles 100 \
        --num-edges 100 --edge-degree 8 --iters 30

Reports:
  * forward-only and forward+backward wall time (CUDA-synchronized)
  * torch.profiler kernel breakdown (top ops by CUDA time)
  * isolated timing of the sparse incidence operators per step

This is synthetic (no data loading) so it isolates model compute + sparse ops.
Compare its per-step time against the observed ~0.8 s/step: if this is far
faster, the bottleneck is data loading / host overhead, not the model.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cpen.models.cpen import CPEN
from cpen.utils import sparse_incidence as sp
from cpen.utils.operator_gamma import OperatorGammas


def build_star_incidence(
    batch: int, num_edges: int, num_nodes: int, edge_degree: int, device
) -> torch.Tensor:
    """Random bool incidence (B, M, N) with ~edge_degree nodes per edge."""
    inc = torch.zeros(batch, num_edges, num_nodes, dtype=torch.bool, device=device)
    for b in range(batch):
        for e in range(num_edges):
            nodes = torch.randperm(num_nodes, device=device)[:edge_degree]
            inc[b, e, nodes] = True
    return inc


def synchronize(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-particles", type=int, default=100)
    ap.add_argument("--num-edges", type=int, default=100)
    ap.add_argument("--edge-degree", type=int, default=8)
    ap.add_argument("--n-features", type=int, default=4)
    ap.add_argument("--n-edge-features", type=int, default=4)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--residual", default="transformer-like")
    ap.add_argument("--norm", default="gamma", choices=["gamma", "degree"])
    ap.add_argument("--backend", default="dense", choices=["dense", "sparse"])
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[profile] device={device} precision={args.precision}")
    torch.manual_seed(0)

    gamma = OperatorGammas(gamma_11=14.4, gamma_12=2.0, gamma_21=2.0, gamma_22=14.6)
    model = CPEN(
        n_features=args.n_features,
        n_edge_features=args.n_edge_features,
        out_dim=2,
        depth=args.depth,
        width=args.width,
        operators="incidence",
        normalization="energy-weights",
        residual_structure=args.residual,
        operator_normalization=args.norm,
        operator_backend=args.backend,
        gamma_rs=gamma if args.norm == "gamma" else None,
    ).to(device)
    print(f"[profile] operator_backend={args.backend} norm={args.norm} residual={args.residual}")

    B, N, M = args.batch_size, args.num_particles, args.num_edges
    x = torch.randn(B, N, args.n_features, device=device)
    edge_x = torch.randn(B, M, args.n_edge_features, device=device)
    z = torch.rand(B, N, device=device)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    y = torch.randint(0, 2, (B,), device=device)

    inc = build_star_incidence(B, M, N, args.edge_degree, device)
    fields = sp.finalize_incidence_storage(inc)
    kw = dict(
        incidence=fields["incidence"].to(device),
        incidence_node=fields["incidence_node"].to(device),
        incidence_edge=fields["incidence_edge"].to(device),
        incidence_nnz=fields["incidence_nnz"].to(device),
        node_degree_inv=fields["node_degree_inv"].to(device),
        edge_degree_inv=fields["edge_degree_inv"].to(device),
        mask=mask,
        z=z,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if (device.type == "cuda" and args.precision == "bf16")
        else torch.autocast("cpu", enabled=False)
    )

    def step(backward: bool):
        opt.zero_grad(set_to_none=True)
        with autocast:
            out = model(x, edge_x, **kw)
            loss = torch.nn.functional.cross_entropy(out, y)
        if backward:
            loss.backward()
            opt.step()
        return loss

    for _ in range(args.warmup):
        step(True)
    synchronize(device)

    # forward-only
    synchronize(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.iters):
            with autocast:
                model(x, edge_x, **kw)
    synchronize(device)
    fwd = (time.perf_counter() - t0) / args.iters

    # forward + backward
    t0 = time.perf_counter()
    for _ in range(args.iters):
        step(True)
    synchronize(device)
    fb = (time.perf_counter() - t0) / args.iters

    print(f"[profile] forward-only : {fwd*1e3:8.2f} ms/step")
    print(f"[profile] fwd+bwd+opt  : {fb*1e3:8.2f} ms/step")

    # sparse ops in isolation (one t22 = S D S^T, the heaviest chain)
    h_edge = torch.randn(B, M, args.width, device=device)
    coo = sp.coo_from_batch_tensors(
        incidence_node=kw["incidence_node"],
        incidence_edge=kw["incidence_edge"],
        incidence_nnz=kw["incidence_nnz"],
        num_edges=M,
        num_nodes=N,
    )
    _ = coo.flat_indices()  # warm cache
    synchronize(device)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        sp.apply_t22(h_edge, coo, kw["node_degree_inv"], kw["edge_degree_inv"])
    synchronize(device)
    print(f"[profile] one t22 chain: {(time.perf_counter()-t0)/args.iters*1e3:8.2f} ms")

    if device.type == "cuda":
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(10):
                step(True)
            synchronize(device)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
