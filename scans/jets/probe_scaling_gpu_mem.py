#!/usr/bin/env python3
"""Single-GPU memory probe for CAPEN-llama-att (live star-R, bf16, Adam).

Mirrors a real unique-pass step: graph construction + fwd + bwd + optimizer.
Prints trainable params and peak CUDA memory. Exits 0 even on OOM so the
log records the failure; OOM is marked oom=1.

  PYTHONPATH=src python scans/jets/probe_scaling_gpu_mem.py --width 512 --batch-size 512
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cpen.training.scaling_compute import _raw_batch, build_lit


def _gib(n: int) -> float:
    return n / (1024**3)


def probe(*, depth: int, width: int, heads: int, batch_size: int, num_particles: int) -> dict:
    if not torch.cuda.is_available():
        raise SystemExit("need a GPU")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    lit = build_lit(depth=depth, width=width, heads=heads)
    n_params = sum(p.numel() for p in lit.parameters() if p.requires_grad)
    lit.train()
    lit.to(device)
    opt = torch.optim.Adam(
        [p for p in lit.parameters() if p.requires_grad],
        lr=1e-4,
        eps=1e-14,
    )

    raw = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in _raw_batch(lit, batch_size=batch_size, num_particles=num_particles).items()
    }

    torch.cuda.synchronize()
    after_load = torch.cuda.max_memory_allocated()

    batch = lit.on_after_batch_transfer(
        {k: v.clone() if torch.is_tensor(v) else v for k, v in raw.items()}, 0
    )
    opt.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x, y = lit._unpack_batch(batch)
        logits = lit.forward(x, **lit._model_kwargs(batch))
        loss = F.cross_entropy(logits, y)
    loss.backward()
    opt.step()
    torch.cuda.synchronize()

    return {
        "ok": True,
        "oom": False,
        "depth": depth,
        "width": width,
        "heads": heads,
        "d_head": width // heads,
        "n_params": n_params,
        "batch_size": batch_size,
        "num_particles": num_particles,
        "gpu": props.name,
        "gpu_total_gib": round(_gib(props.total_memory), 2),
        "peak_allocated_gib": round(_gib(torch.cuda.max_memory_allocated()), 2),
        "peak_reserved_gib": round(_gib(torch.cuda.max_memory_reserved()), 2),
        "after_load_gib": round(_gib(after_load), 2),
        "loss": float(loss.detach().float().cpu()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--heads", type=int, default=None, help="default width/32")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-particles", type=int, default=80)
    p.add_argument("--json-out", type=str, default=None)
    args = p.parse_args()
    heads = args.heads if args.heads is not None else args.width // 32
    if args.width % heads != 0:
        raise SystemExit(f"width {args.width} not divisible by heads {heads}")

    print(
        f"[probe] L={args.depth} D={args.width} H={heads} "
        f"d_head={args.width // heads} B={args.batch_size} P={args.num_particles}",
        flush=True,
    )
    try:
        rec = probe(
            depth=args.depth,
            width=args.width,
            heads=heads,
            batch_size=args.batch_size,
            num_particles=args.num_particles,
        )
    except torch.cuda.OutOfMemoryError as exc:
        rec = {
            "ok": False,
            "oom": True,
            "depth": args.depth,
            "width": args.width,
            "heads": heads,
            "batch_size": args.batch_size,
            "error": str(exc).split("\n")[0],
        }
        if torch.cuda.is_available():
            rec["peak_allocated_gib"] = round(_gib(torch.cuda.max_memory_allocated()), 2)
            rec["gpu_total_gib"] = round(
                _gib(torch.cuda.get_device_properties(0).total_memory), 2
            )
        print("[probe] OOM", flush=True)
    except Exception:
        traceback.print_exc()
        rec = {"ok": False, "oom": False, "error": "exception"}
        raise SystemExit(1) from None
    print(json.dumps(rec, indent=2), flush=True)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rec, indent=2) + "\n")
    # Always exit 0 so a width ladder can continue after OOM.
    raise SystemExit(0)


if __name__ == "__main__":
    main()
