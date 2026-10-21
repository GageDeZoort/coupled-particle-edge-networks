#!/usr/bin/env python3
"""One-step Adam μP feature stability: ρ / Δρ on particle and edge states.

Measures, for streams \(X\) (particles) and \(E\) (edges),

\[
  \\rho = \\|Z\\|_F^2 / (N_{\\mathrm{live}}\\, d),\\qquad
  \\Delta\\rho = \\|\\Delta Z\\|_F^2 / (N_{\\mathrm{live}}\\, d)
\]

where \(d=D\) after the encoder / residual blocks and \(d=C\) (class dim) at
the token-wise decoder outputs. Stages:

  * \(\\ell=0\)   — post-encoder \(X^{(0)}, E^{(0)}\)
  * \(\\ell=L\)   — after the last residual block \(X^{(L)}, E^{(L)}\)
  * \(\\ell=L+1\) — token-wise decoder outputs (× readout scale)

Each point averages ``n_seeds × n_batches`` trials. Sweep \(\\eta_0\) so
hyperparameter transfer vs width is visible.

  python scans/jets/mup_one_step_transfer_probe.py --synthetic
  python scans/jets/plot_stability_nature.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from cpen.graphs.graph_star import (  # noqa: E402
    build_star_kin7_plus_knn_graph,
    build_star_plus_knn_graph,
    robust_part_interaction_preprocess,
)
from cpen.models.base import adam_lr  # noqa: E402
from cpen.models.capen_llama_att import CAPENLlamaAtt  # noqa: E402
from cpen.training.stability import (  # noqa: E402
    ADAM_EPS,
    disable_probe,
    enable_probe,
    last_layer_stage,
    masked_msq,
    n_trainable,
    parse_probe,
)

DEFAULT_ARCHS: tuple[tuple[int, int], ...] = (
    (2, 128),
    (4, 256),
    (6, 512),
    (8, 1024),
)
DEFAULT_CELLS: tuple[tuple[int, float], ...] = ((12, 0.10),)
DEFAULT_ETAS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)

# (stage_key, paper label)
STAGES: tuple[tuple[str, str], ...] = (
    ("encode", "ell0"),
    ("last", "ellL"),
    ("output", "ellLp1"),
)
STREAMS: tuple[str, ...] = ("x", "e")


def _parse_archs(raw: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        l_s, d_s = tok.split(":")
        out.append((int(l_s), int(d_s)))
    return out


def _parse_cells(raw: str) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k_s, r_s = tok.split(":")
        out.append((int(k_s), float(r_s)))
    return out


def _parse_etas(raw: str) -> list[float]:
    return [float(t.strip()) for t in raw.split(",") if t.strip()]


def _synthetic_batch(
    *,
    batch_size: int,
    n_particles: int,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    e = torch.rand(batch_size, n_particles, generator=g) * 50 + 1.0
    direction = torch.randn(batch_size, n_particles, 3, generator=g)
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    p = direction * (e.unsqueeze(-1) * 0.9)
    x_raw = torch.cat([e.unsqueeze(-1), p], dim=-1)
    lo = max(8, n_particles // 4)
    hi = max(lo + 1, min(n_particles, max(lo + 1, (3 * n_particles) // 4)))
    n_live = torch.randint(lo, hi + 1, (batch_size,), generator=g)
    mask = torch.zeros(batch_size, n_particles, dtype=torch.bool)
    for b in range(batch_size):
        mask[b, : int(n_live[b])] = True
    x_raw = x_raw * mask.unsqueeze(-1)
    x = torch.randn(batch_size, n_particles, 7, generator=g) * mask.unsqueeze(-1)
    y = torch.randint(0, 2, (batch_size,), generator=g)
    if batch_size >= 2:
        y[0] = 0
        y[1] = 1
    return {
        "x_raw": x_raw.to(device),
        "x": x.to(device),
        "mask": mask.to(device),
        "node_mask": mask.to(device),
        "y": y.to(device),
    }


def _estimate_preprocess(
    x_raw: torch.Tensor,
    mask: torch.Tensor,
    *,
    k: int,
    radius: float,
    typed: bool = False,
) -> tuple[tuple[float, float], ...]:
    if typed:
        _star, knn_x, _inc = build_star_kin7_plus_knn_graph(
            x_raw.cpu(),
            radius=float(radius),
            k=int(k),
            mask=mask.cpu(),
            typed_features=True,
            standardize_knn=False,
            include_self=False,
        )
        live = knn_x.abs().sum(dim=-1) > 0
        if not live.any():
            raise RuntimeError(f"no live knn edges for k={k} R={radius}")
        return robust_part_interaction_preprocess(knn_x[live])
    edge_x, _ = build_star_plus_knn_graph(
        x_raw.cpu(),
        radius=float(radius),
        k=int(k),
        mask=mask.cpu(),
        edge_features="part-interaction",
        standardize=False,
    )
    live = edge_x.abs().sum(dim=-1) > 0
    if not live.any():
        raise RuntimeError(f"no live edges for k={k} R={radius}")
    return robust_part_interaction_preprocess(edge_x[live])


def _materialize_graph(
    batch: dict[str, torch.Tensor],
    *,
    k: int,
    radius: float,
    preprocess: tuple[tuple[float, float], ...],
    typed: bool = False,
) -> dict[str, torch.Tensor]:
    out = dict(batch)
    if typed:
        star_x, knn_x, incidence = build_star_kin7_plus_knn_graph(
            batch["x_raw"],
            radius=float(radius),
            k=int(k),
            mask=batch["mask"],
            part_interaction_preprocess=preprocess,
            typed_features=True,
            include_self=False,
        )
        edge_mask = incidence.bool().any(dim=-1)
        out["edge_x_star"] = star_x
        out["edge_x_knn"] = knn_x
        out["edge_x"] = None  # typed forward ignores shared edge_x
        out["incidence"] = incidence
        out["edge_mask"] = edge_mask
        return out
    edge_x, incidence = build_star_plus_knn_graph(
        batch["x_raw"],
        radius=float(radius),
        k=int(k),
        mask=batch["mask"],
        edge_features="part-interaction",
        part_interaction_preprocess=preprocess,
    )
    edge_mask = edge_x.abs().sum(dim=-1) > 0
    out["edge_x"] = edge_x
    out["incidence"] = incidence
    out["edge_mask"] = edge_mask
    return out


def _build_model(
    *,
    depth: int,
    width: int,
    n_features: int,
    n_edge_features: int,
    out_dim: int,
    heads: int,
    uniform_pool: bool,
    seed: int,
    typed: bool = False,
) -> CAPENLlamaAtt:
    torch.manual_seed(int(seed))
    return CAPENLlamaAtt(
        n_features=n_features,
        n_edge_features=n_edge_features,
        out_dim=out_dim,
        depth=depth,
        width=width,
        heads=heads,
        dropout=0.0,
        optimizer="adam",
        readout_mode="graph",
        use_wire=False,
        all_to_all_particle_attention=True,
        uniform_output_pool=bool(uniform_pool),
        n_class_tokens=int(out_dim),
        typed_edge_streams=bool(typed),
        hyperedge_m22_only=not bool(typed),
        n_star_edge_features=7,
        n_knn_edge_features=4,
    )


def _forward_kwargs(batch: dict[str, torch.Tensor]) -> dict:
    keys = (
        "incidence",
        "mask",
        "node_mask",
        "edge_mask",
        "edge_type",
        "rope_coordinates",
        "wire_coordinates",
        "x_raw",
        "edge_x_star",
        "edge_x_knn",
    )
    return {k: batch[k] for k in keys if k in batch and batch[k] is not None}


def _edge_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor | None:
    if batch.get("edge_mask") is not None:
        return batch["edge_mask"].bool()
    inc = batch.get("incidence")
    if inc is not None:
        return inc.bool().any(dim=-1)
    return None


def _token_decode(
    model: CAPENLlamaAtt,
    h_x: torch.Tensor,
    h_e: torch.Tensor,
    *,
    node_mask: torch.Tensor | None,
    edge_mask: torch.Tensor | None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor | None]]:
    """ℓ=L+1 streams: attn pool states, or uniform token-wise decoders."""
    if not bool(getattr(model, "uniform_output_pool", False)) and model.output_pool is not None:
        pool = model.output_pool
        pooled_x, pooled_e, fused = pool.pool_streams(
            h_x, h_e, mask=node_mask, edge_mask=edge_mask
        )
        # Class tokens are the token axis; no pad mask.
        # Pool returns (B, C, D) Θ(1) features; CompleteP head is separate.
        out = {
            "output_x": (pooled_x.detach(), None),
            "output_e": (
                pooled_e.detach() if pooled_e is not None else pooled_x.detach(),
                None,
            ),
            "output_fused": (fused.detach(), None),
        }
        logits = (fused * pool.class_unembed).sum(dim=-1) * float(pool._readout_scale)
        # (B, C) → (B, 1, C) so masked_msq sees class dim as feature width.
        out["output_logits"] = (logits.detach().unsqueeze(1), None)
        return out

    scale = float(getattr(model, "_readout_scale", 1.0))
    z_x = model.decoder_x(h_x) * scale
    z_e = model.decoder_e(h_e) * scale
    return {
        "output_x": (z_x.detach(), node_mask),
        "output_e": (z_e.detach(), edge_mask),
    }


def snapshot_xe(
    model: CAPENLlamaAtt,
    batch: dict[str, torch.Tensor],
) -> dict[str, tuple[torch.Tensor, torch.Tensor | None]]:
    """No-grad forward → CPU (tensor, mask) for encode / last / output × {x,e}."""
    node_mask = batch.get("node_mask", batch.get("mask"))
    edge_mask = _edge_mask(batch)
    probe = enable_probe(model)
    try:
        model.eval()
        with torch.no_grad():
            model(batch["x"], batch.get("edge_x"), **_forward_kwargs(batch))
        snaps = parse_probe(probe, node_mask=node_mask, edge_mask=edge_mask)
    finally:
        disable_probe(model)

    enc = snaps["encode"]
    last = snaps[last_layer_stage(int(model.depth))]
    out_bufs = _token_decode(
        model,
        last.h_x,
        last.h_e,
        node_mask=node_mask if isinstance(node_mask, torch.Tensor) else enc.node_mask,
        edge_mask=edge_mask if isinstance(edge_mask, torch.Tensor) else enc.edge_mask,
    )

    def _cpu(z: torch.Tensor, m: torch.Tensor | None):
        return (
            z.detach().float().cpu(),
            None if m is None else m.detach().cpu(),
        )

    result = {
        "encode_x": _cpu(enc.h_x, enc.node_mask),
        "encode_e": _cpu(enc.h_e, enc.edge_mask),
        "last_x": _cpu(last.h_x, last.node_mask),
        "last_e": _cpu(last.h_e, last.edge_mask),
    }
    for key, (z, m) in out_bufs.items():
        result[key] = _cpu(z, m)
    # Always expose output_{x,e} for the Nature grid; extras are optional channels.
    return result


def _rho(pair: tuple[torch.Tensor, torch.Tensor | None]) -> float:
    z, m = pair
    return masked_msq(z, m)


def _drho(
    a: tuple[torch.Tensor, torch.Tensor | None],
    b: tuple[torch.Tensor, torch.Tensor | None],
) -> float:
    z0, m0 = a
    z1, m1 = b
    m = m0 if m0 is not None else m1
    return masked_msq(z1 - z0, m)


CHANNEL_KEYS = tuple(f"{stage}_{stream}" for stage, _ in STAGES for stream in STREAMS)


def run_one(
    *,
    depth: int,
    width: int,
    batch: dict[str, torch.Tensor],
    eta_0: float,
    uniform_pool: bool,
    k: int,
    radius: float,
    seed: int,
    heads: int,
    typed: bool = False,
) -> list[dict]:
    device = batch["x"].device
    out_dim = max(2, int(batch["y"].max().item()) + 1 if batch["y"].numel() else 2)
    if typed:
        n_edge_features = 7  # unused after dual encoders; keep for ctor
    else:
        n_edge_features = int(batch["edge_x"].size(-1))
    model = _build_model(
        depth=depth,
        width=width,
        n_features=int(batch["x"].size(-1)),
        n_edge_features=n_edge_features,
        out_dim=out_dim,
        heads=heads,
        uniform_pool=uniform_pool,
        seed=seed,
        typed=typed,
    ).to(device)

    lr = adam_lr(eta_0, width)
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=ADAM_EPS)

    buf0 = snapshot_xe(model, batch)

    model.train()
    opt.zero_grad(set_to_none=True)
    edge_x = batch.get("edge_x")
    logits = model(batch["x"], edge_x, **_forward_kwargs(batch))
    loss = F.cross_entropy(logits, batch["y"])
    loss.backward()
    opt.step()

    buf1 = snapshot_xe(model, batch)

    base = {
        "depth": depth,
        "width": width,
        "heads": int(heads),
        "d_head": int(width) // int(heads),
        "n_params": n_trainable(model),
        "eta_0": float(eta_0),
        "lr": lr,
        "k": int(k),
        "radius": float(radius),
        "uniform_pool": bool(uniform_pool),
        "typed_edge_streams": bool(typed),
        "seed": int(seed),
        "batch_id": int(batch.get("_batch_id", 0)),
        "loss_1": float(loss.detach()),
    }
    rows = []
    for stage, paper in STAGES:
        for stream in STREAMS:
            key = f"{stage}_{stream}"
            if key not in buf0 or key not in buf1:
                continue
            rows.append(
                {
                    **base,
                    "stage": stage,
                    "stream": stream,
                    "ell": paper,
                    "channel": key,
                    "msq_0": _rho(buf0[key]),
                    "msq_1": _rho(buf1[key]),
                    "dmsq_1": _drho(buf0[key], buf1[key]),
                }
            )
    # Optional attn-only diagnostics (fused pool / logits).
    for key, stage, stream in (
        ("output_fused", "output", "fused"),
        ("output_logits", "output", "logits"),
    ):
        if key in buf0 and key in buf1:
            rows.append(
                {
                    **base,
                    "stage": stage,
                    "stream": stream,
                    "ell": "ellLp1",
                    "channel": key,
                    "msq_0": _rho(buf0[key]),
                    "msq_1": _rho(buf1[key]),
                    "dmsq_1": _drho(buf0[key], buf1[key]),
                }
            )

    model.cpu()
    del model, opt, buf0, buf1
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


AGG_KEYS = (
    "depth",
    "width",
    "k",
    "radius",
    "uniform_pool",
    "channel",
    "stage",
    "stream",
    "ell",
    "eta_0",
)


def aggregate_trials(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    g = (
        df.groupby(list(AGG_KEYS), as_index=False)
        .agg(
            msq_0=("msq_0", "mean"),
            msq_0_std=("msq_0", "std"),
            msq_1=("msq_1", "mean"),
            msq_1_std=("msq_1", "std"),
            dmsq_1=("dmsq_1", "mean"),
            dmsq_1_std=("dmsq_1", "std"),
            loss_1=("loss_1", "mean"),
            n_trials=("msq_0", "count"),
            n_params=("n_params", "first"),
            lr=("lr", "first"),
        )
        .sort_values(["k", "radius", "uniform_pool", "eta_0", "width", "channel"])
    )
    n = g["n_trials"].clip(lower=1).astype(float)
    g["msq_0_sem"] = g["msq_0_std"].fillna(0.0) / np.sqrt(n)
    g["msq_1_sem"] = g["msq_1_std"].fillna(0.0) / np.sqrt(n)
    g["dmsq_1_sem"] = g["dmsq_1_std"].fillna(0.0) / np.sqrt(n)
    return g


def _write_outputs(trials: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    trials_path = out.with_name(out.stem + "_trials.csv")
    trials.to_csv(trials_path, index=False)
    agg = aggregate_trials(trials)
    agg.to_csv(out, index=False)
    legacy = _REPO / "paper/data/mup_one_step_probe.csv"
    if out.resolve() != legacy.resolve():
        agg.to_csv(legacy, index=False)
    print(
        f"    checkpoint {out} ({len(agg)} agg rows from {len(trials)} trials; "
        f"widths={sorted(agg.width.unique()) if len(agg) else []}; "
        f"etas={sorted(agg.eta_0.unique()) if len(agg) else []})",
        flush=True,
    )


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=_REPO / "paper/data/stability_one_step.csv")
    p.add_argument("--archs", type=str, default="2:128,4:256,6:512,8:1024")
    p.add_argument(
        "--cells",
        type=str,
        default="12:0.10",
        help="(k,R*) cells; default operating point only (η₀ sweep is expensive)",
    )
    p.add_argument(
        "--etas",
        type=str,
        default="0.01,0.05,0.1,0.25,0.5,1.0",
        help="comma-separated η₀ values (Adam lr = η₀/√D)",
    )
    p.add_argument(
        "--heads",
        type=int,
        default=None,
        help="fixed head count (legacy). Prefer --d-head so H=D/d_h.",
    )
    p.add_argument(
        "--d-head",
        type=int,
        default=None,
        help="fixed head dim; sets H=D/d_h per width (must divide). Paper default: 32.",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-particles", type=int, default=128)
    p.add_argument("--seed", type=int, default=0, help="base seed; trials use seed+i")
    p.add_argument("--n-seeds", type=int, default=10, help="model init seeds per point")
    p.add_argument("--n-batches", type=int, default=10, help="synthetic jet batches per point")
    p.add_argument(
        "--same-data-shard",
        action="store_true",
        help="Fix synthetic batch seed across model seeds (vary init only)",
    )
    p.add_argument("--synthetic", action="store_true")
    p.add_argument(
        "--typed-edge-streams",
        action="store_true",
        help=(
            "Probe CAPEN-Llama-att typed dual encoders + §1 routing on the "
            "kin7 star + part-int kNN operating-point graph (no pad)."
        ),
    )
    p.add_argument("--pools", type=str, default="attn")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(list(argv) if argv is not None else None)

    device = torch.device(args.device)
    archs = _parse_archs(args.archs)
    cells = _parse_cells(args.cells)
    etas = _parse_etas(args.etas)
    n_seeds = max(1, int(args.n_seeds))
    n_batches = max(1, int(args.n_batches))
    pools = []
    for tok in args.pools.split(","):
        tok = tok.strip().lower()
        if tok in {"attn", "attention", "attentional"}:
            pools.append(False)
        elif tok in {"uniform", "mean", "linear"}:
            pools.append(True)
        else:
            raise SystemExit(f"unknown pool {tok!r}")

    if not args.synthetic:
        raise SystemExit("Pass --synthetic (JetClass stream loader not wired here yet).")

    d_head = int(args.d_head) if args.d_head is not None else None
    fixed_heads = int(args.heads) if args.heads is not None else None
    if d_head is not None and fixed_heads is not None:
        raise SystemExit("Pass only one of --d-head / --heads")
    if d_head is None and fixed_heads is None:
        fixed_heads = 8  # legacy default for older slurm scripts
    if d_head is not None and d_head <= 0:
        raise SystemExit(f"--d-head must be positive; got {d_head}")
    for depth, width in archs:
        if d_head is not None and int(width) % int(d_head) != 0:
            raise SystemExit(
                f"width D={width} not divisible by d_h={d_head} (L={depth})"
            )
        if fixed_heads is not None and int(width) % int(fixed_heads) != 0:
            raise SystemExit(
                f"width D={width} not divisible by heads={fixed_heads} (L={depth})"
            )

    def _heads_for(width: int) -> int:
        if d_head is not None:
            return int(width) // int(d_head)
        return int(fixed_heads)

    n_cfg = len(cells) * len(pools) * len(archs) * len(etas)
    head_msg = f"d_h={d_head} → H=D/{d_head}" if d_head is not None else f"heads={fixed_heads}"
    print(
        f"[probe] n_seeds={n_seeds} n_batches={n_batches} → {n_seeds * n_batches} trials/point; "
        f"same_data_shard={bool(args.same_data_shard)}; {head_msg}; "
        f"{n_cfg} configs (cells×pools×archs×etas); "
        f"stages={[s for s, _ in STAGES]}; streams={list(STREAMS)}; etas={etas}",
        flush=True,
    )

    ref_raw = _synthetic_batch(
        batch_size=args.batch_size,
        n_particles=args.n_particles,
        device=device,
        seed=int(args.seed),
    )
    typed = bool(args.typed_edge_streams)
    if typed:
        print(
            "[probe] typed-edge-streams: kin7 star + part-int kNN, dual encoders, §1 routing",
            flush=True,
        )
    preprocess_by_cell: dict[tuple[int, float], tuple] = {}
    for k, radius in cells:
        preprocess_by_cell[(k, radius)] = _estimate_preprocess(
            ref_raw["x_raw"],
            ref_raw["mask"],
            k=k,
            radius=radius,
            typed=typed,
        )
        print(
            f"[probe] k={k} R={radius:g} preprocess={preprocess_by_cell[(k, radius)]}",
            flush=True,
        )

    trial_rows: list[dict] = []
    for k, radius in cells:
        preprocess = preprocess_by_cell[(k, radius)]
        for uniform in pools:
            for depth, width in archs:
                for eta0 in etas:
                    heads = _heads_for(width)
                    print(
                        f"  L={depth} D={width} H={heads} d_h={width // heads} η₀={eta0:g} "
                        f"pool={'uniform' if uniform else 'attn'} "
                        f"typed={typed} ({n_seeds}×{n_batches}) …",
                        flush=True,
                    )
                    for seed_i in range(n_seeds):
                        model_seed = int(args.seed) + seed_i
                        for batch_i in range(n_batches):
                            batch_seed = int(args.seed) + 10_000 + batch_i * 97
                            if not args.same_data_shard:
                                batch_seed += seed_i * 13
                            raw = _synthetic_batch(
                                batch_size=args.batch_size,
                                n_particles=args.n_particles,
                                device=device,
                                seed=batch_seed,
                            )
                            batch = _materialize_graph(
                                raw,
                                k=k,
                                radius=radius,
                                preprocess=preprocess,
                                typed=typed,
                            )
                            batch["_batch_id"] = batch_i
                            trial_rows.extend(
                                run_one(
                                    depth=depth,
                                    width=width,
                                    batch=batch,
                                    eta_0=eta0,
                                    uniform_pool=uniform,
                                    k=k,
                                    radius=radius,
                                    seed=model_seed,
                                    heads=heads,
                                    typed=typed,
                                )
                            )
                            del raw, batch
                    _write_outputs(pd.DataFrame(trial_rows), args.out)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    print(
        f"wrote {args.out} + {args.out.with_name(args.out.stem + '_trials.csv')} "
        f"({len(trial_rows)} trials)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
