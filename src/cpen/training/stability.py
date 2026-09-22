"""Forward / one-step backward feature-norm diagnostics for CPEN and CAPEN.

The μP / coupled-network theory says hidden states stay Θ(1) in the
coordinate

    ρ_t^{(ℓ)} = ||Z_t^{(ℓ)}||_F² / (N_t D)

and that a transferred Adam step (η = η₀ / √D) moves those states by an
amount that does not blow up with width. ``t`` indexes the particle (X) and
edge (E) streams.

Enable the bytecode ``_feature_probe`` list, run several batches through a
depth/width ladder, and plot ρ and Δρ vs parameter count with error bands.
Jets use a graph-level CE on ``(z_X + z_E)/√2``. PascalVOC-SP uses simultaneous
node CE and boundary-edge CE (``readout_mode='node+edge'``).
"""

from __future__ import annotations

import gc
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from cpen.models.base import adam_lr
from cpen.training.transfer_plots import DEFAULT_FIG_DIR, savefig

ADAM_EPS = 1e-14

# After a CUDA OOM, IPython keeps the traceback frame (and every tensor it
# referenced). That is why a later L=2 forward can fail with 9 GiB already
# allocated in this process.
_STALE_CUDA_GIB = 2.0

# (L, D) = (2, 128), (3, 192), …  matches D = 64 L.
DEFAULT_ARCHS: tuple[tuple[int, int], ...] = tuple(
    [(2, 128), (3, 192), (4, 256), (5, 384), (6, 512)]
)
DEFAULT_N_ADAM_STEPS = 1000
DEFAULT_N_BATCHES = 4
PASCAL_OUT_DIM = 21
PASCAL_N_EDGE_CLASSES = 2

FORWARD_KW = (
    "incidence",
    "incidence_node",
    "incidence_edge",
    "incidence_nnz",
    "node_degree_inv",
    "edge_degree_inv",
    "mask",
    "z",
    "wire_coordinates",
    "rope_coordinates",
    "edge_type",
    "edge_mask",
    "node_mask",
)


@dataclass(frozen=True)
class ProbeSnap:
    """Detached hidden states at one residual stage."""

    stage: str
    h_x: torch.Tensor
    h_e: torch.Tensor
    node_mask: torch.Tensor | None
    edge_mask: torch.Tensor | None


def cuda_mem_line() -> str:
    if not torch.cuda.is_available():
        return "cuda: unavailable"
    free, total = torch.cuda.mem_get_info()
    alloc = torch.cuda.memory_allocated()
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    return (
        f"{name}  allocated={alloc / 2**30:.2f} GiB  "
        f"used={(total - free) / 2**30:.2f}/{total / 2**30:.2f} GiB"
    )


def release_cuda() -> None:
    """Drop IPython's last exception frames, then free cached CUDA blocks."""
    for name in ("last_traceback", "last_type", "last_value", "last_exc"):
        if hasattr(sys, name):
            setattr(sys, name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def n_trainable(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def last_layer_stage(depth: int) -> str:
    return f"layer{int(depth) - 1}"


def _mask_mean(tokens: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return tokens.mean(dim=1)
    w = mask.to(dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


def _decode_candidates(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    snap: ProbeSnap,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per-stream graph logits ``(z_X, z_E)``, each ``(B, C)``.

    Both families read out as ``(z_X + z_E)/√2`` with the decoder applied
    *after* pooling, so the streams are separable. Pooling differs (CPEN
    α-pools with the energy weights; CAPEN mask-means), hence candidates.
    """
    scale = getattr(model, "_readout_scale", 1.0)
    h_x, h_e = snap.h_x, snap.h_e
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    needed = ("_particle_alpha", "_pool_particles", "_pool_edges", "_resolve_incidence")
    if all(hasattr(model, name) for name in needed):
        try:
            from cpen.graphs.sparse_incidence import SparseIncidenceOps

            coo = model._resolve_incidence(
                incidence=batch.get("incidence"),
                incidence_node=batch.get("incidence_node"),
                incidence_edge=batch.get("incidence_edge"),
                incidence_nnz=batch.get("incidence_nnz"),
                num_edges=int(h_e.size(1)),
                num_nodes=int(h_x.size(1)),
            )
            ops = SparseIncidenceOps(coo)
            alpha = model._particle_alpha(batch["x"], batch.get("z"))
            out.append(
                (
                    model.decoder_x(model._pool_particles(alpha, h_x)) * scale,
                    model.decoder_e(model._pool_edges(ops, alpha, h_e)) * scale,
                )
            )
        except Exception:
            pass
    out.append(
        (
            model.decoder_x(_mask_mean(h_x, snap.node_mask)) * scale,
            model.decoder_e(_mask_mean(h_e, snap.edge_mask)) * scale,
        )
    )
    return out


def _logits_close(a: torch.Tensor, b: torch.Tensor, *, tol: float) -> bool:
    if a.shape != b.shape:
        return False
    scale = max(1.0, float(b.detach().abs().max()))
    return float((a - b).abs().max()) <= tol * scale


def attach_decode_snap(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    snaps: dict[str, ProbeSnap],
    logits: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    *,
    tol: float = 1e-4,
) -> dict[str, ProbeSnap]:
    """Add the readout stage (paper ``ℓ = L+1``).

    Graph classification (jets): the decoder acts after pooling, so ``ℓ=L+1``
    is a ``(B, C)`` object, not a per-token state: ``ρ = ||Z||_F²/(B C)``.
    The reconstruction is only kept when ``(z_X + z_E)/√2`` reproduces the
    model's own logits.

    Node+edge (Pascal): both heads are token-wise. ``ℓ=L+1`` is
    ``(B, N, C_X)`` / ``(B, M, C_E)`` with the same masks as the last residual,
    kept only when ``decoder_{x,e}(h^{(L)})`` matches the model outputs.
    """
    depth = getattr(model, "depth", None)
    if depth is None or not hasattr(model, "decoder_x") or not hasattr(model, "decoder_e"):
        return snaps
    snap = snaps.get(last_layer_stage(int(depth)))
    if snap is None:
        return snaps
    readout = str(getattr(model, "readout_mode", "graph") or "graph")
    scale = getattr(model, "_readout_scale", 1.0)
    if readout == "node+edge":
        if not isinstance(logits, tuple) or len(logits) != 2:
            return snaps
        z_x, z_e = logits
        pred_x = model.decoder_x(snap.h_x) * scale
        pred_e = model.decoder_e(snap.h_e) * scale
        if not _logits_close(pred_x, z_x, tol=tol) or not _logits_close(pred_e, z_e, tol=tol):
            return snaps
        out = dict(snaps)
        out["decode"] = ProbeSnap(
            "decode", z_x.detach(), z_e.detach(), snap.node_mask, snap.edge_mask
        )
        return out
    if getattr(model, "readout_mode", "graph") != "graph":
        return snaps
    if isinstance(logits, tuple) or logits.dim() != 2:
        return snaps
    for z_x, z_e in _decode_candidates(model, batch, snap):
        if z_x.shape != logits.shape or z_e.shape != logits.shape:
            continue
        recon = (z_x + z_e) / 2.0**0.5
        if not _logits_close(recon, logits, tol=tol):
            continue
        out = dict(snaps)
        out["decode"] = ProbeSnap(
            "decode", z_x.unsqueeze(1), z_e.unsqueeze(1), None, None
        )
        return out
    return snaps


def masked_msq(z: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    """||Z||_F² / (N_t D) over valid tokens (batch as one token pool)."""
    zf = z.detach().float()
    if zf.dim() != 3:
        raise ValueError(f"expected (batch, tokens, D); got {tuple(zf.shape)}")
    width = zf.size(-1)
    sq = zf.square()
    if mask is None:
        return float(sq.mean())
    w = mask.detach().to(dtype=zf.dtype)
    if w.shape != zf.shape[:2]:
        raise ValueError(f"mask shape {tuple(w.shape)} != tokens {tuple(zf.shape[:2])}")
    n_valid = float(w.sum().clamp_min(1.0))
    return float((sq * w.unsqueeze(-1)).sum() / (n_valid * width))


def edge_mask_from_batch(batch: dict[str, torch.Tensor], n_edges: int) -> torch.Tensor | None:
    if batch.get("edge_mask") is not None:
        return batch["edge_mask"].bool()
    inc = batch.get("incidence")
    if inc is not None:
        return inc.bool().any(dim=-1)
    return None


def parse_probe(
    probe: Sequence[tuple],
    *,
    node_mask: torch.Tensor | None,
    edge_mask: torch.Tensor | None,
) -> dict[str, ProbeSnap]:
    """Index probe tuples ``(stage, h_x, h_e, mask)`` by stage name."""
    out: dict[str, ProbeSnap] = {}
    for item in probe:
        stage, h_x, h_e, probe_mask = item[:4]
        nm = probe_mask if probe_mask is not None else node_mask
        em = edge_mask
        if em is None:
            em = h_e.detach().abs().sum(dim=-1) > 0
        out[str(stage)] = ProbeSnap(str(stage), h_x, h_e, nm, em)
    return out


def enable_probe(model: nn.Module) -> list:
    probe: list = []
    model._feature_probe = probe  # type: ignore[attr-defined]
    return probe


def disable_probe(model: nn.Module) -> None:
    """Stop recording. ``None`` is the bytecode no-op sentinel."""
    model._feature_probe = None  # type: ignore[attr-defined]


def _forward_kwargs(
    batch: dict[str, torch.Tensor],
    model: nn.Module | None = None,
) -> dict[str, torch.Tensor]:
    out = {
        key: batch[key]
        for key in FORWARD_KW
        if key in batch and batch[key] is not None
    }
    if model is None:
        return out
    fn = getattr(model, "_forward_hidden", None)
    if fn is None:
        return out
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return out
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return out
    allowed = set(sig.parameters)
    return {key: value for key, value in out.items() if key in allowed}


def forward_probe(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[dict[str, ProbeSnap], torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
    probe = enable_probe(model)
    try:
        with torch.no_grad():
            logits = model(batch["x"], batch["edge_x"], **_forward_kwargs(batch, model))
        node_mask = batch.get("node_mask", batch.get("mask"))
        edge_mask = edge_mask_from_batch(batch, int(batch["edge_x"].size(1)))
        snaps = parse_probe(probe, node_mask=node_mask, edge_mask=edge_mask)
        snaps = attach_decode_snap(model, batch, snaps, logits)
        return snaps, logits
    finally:
        disable_probe(model)


def _snaps_cpu(snaps: dict[str, ProbeSnap]) -> dict[str, ProbeSnap]:
    """Host copies so Adam steps do not pin layer states on GPU."""
    out: dict[str, ProbeSnap] = {}
    for stage, snap in snaps.items():
        nm = None if snap.node_mask is None else snap.node_mask.detach().cpu()
        em = None if snap.edge_mask is None else snap.edge_mask.detach().cpu()
        out[stage] = ProbeSnap(
            snap.stage,
            snap.h_x.detach().cpu(),
            snap.h_e.detach().cpu(),
            nm,
            em,
        )
    return out


def snap_metrics(snaps: dict[str, ProbeSnap]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for stage, snap in snaps.items():
        rows[stage] = {
            "particle": masked_msq(snap.h_x, snap.node_mask),
            "edge": masked_msq(snap.h_e, snap.edge_mask),
        }
    return rows


def delta_metrics(
    before: dict[str, ProbeSnap],
    after: dict[str, ProbeSnap],
) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for stage in before:
        if stage not in after:
            continue
        a, b = before[stage], after[stage]
        rows[stage] = {
            "particle": masked_msq(b.h_x - a.h_x, a.node_mask),
            "edge": masked_msq(b.h_e - a.h_e, a.edge_mask),
        }
    return rows


def build_stability_model(
    family: str,
    *,
    depth: int,
    width: int,
    n_features: int,
    n_edge_features: int,
    heads: int = 8,
    out_dim: int = 2,
    readout_mode: str = "graph",
    n_edge_classes: int = PASCAL_N_EDGE_CLASSES,
    seed: int = 0,
) -> nn.Module:
    """Construct CPEN / CAPEN / CAPEN-Llama with the jets or Pascal layout."""
    family = family.lower().replace("_", "-")
    readout_mode = str(readout_mode or "graph")
    torch.manual_seed(int(seed))
    if family == "cpen":
        from cpen.models.cpen import CPEN

        model = CPEN(
            n_features=n_features,
            n_edge_features=n_edge_features,
            out_dim=out_dim,
            depth=depth,
            width=width,
            operators="incidence",
            normalization="energy-weights",
            residual_structure="transformer-like",
            operator_normalization="degree",
            operator_backend="sparse",
            layer_norm=True,
            dropout=0.0,
            optimizer="adam",
            readout_mode=readout_mode,
        )
    elif family == "capen":
        from cpen.models.capen import CAPEN

        model = CAPEN(
            n_features=n_features,
            n_edge_features=n_edge_features,
            out_dim=out_dim,
            depth=depth,
            width=width,
            heads=heads,
            dropout=0.0,
            optimizer="adam",
            readout_mode=readout_mode,
        )
    elif family in {"capen-llama", "capenllama"}:
        from cpen.models.capen_llama import CAPENLlama

        model = CAPENLlama(
            n_features=n_features,
            n_edge_features=n_edge_features,
            out_dim=out_dim,
            depth=depth,
            width=width,
            heads=heads,
            dropout=0.0,
            optimizer="adam",
            readout_mode=readout_mode,
            use_wire=False,
            all_to_all_particle_attention=True,
        )
    else:
        raise ValueError(f"unknown family {family!r}")
    if readout_mode == "node+edge":
        if hasattr(model, "x_only_node_readout"):
            model.x_only_node_readout = True
        from cpen.apps.pascal.lit_cpen_pascal import _resize_edge_decoder

        _resize_edge_decoder(model, n_edge_classes=int(n_edge_classes))
    return model


def _infer_out_dim(batch: dict[str, torch.Tensor], default: int = 2) -> int:
    y = batch.get("y")
    if y is None:
        return default
    valid = y[y >= 0] if y.dtype in (torch.int32, torch.int64, torch.long) else y
    if valid.numel() == 0:
        return default
    return int(valid.max()) + 1


def _pascal_node_edge_loss(
    output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Node CE + boundary-edge CE, matching ``LitCPENPascal`` with ``hard-ce``."""
    from cpen.apps.pascal.lit_cpen_pascal import (
        PASCAL_BOUNDARY_POS_WEIGHT,
        PASCAL_CLASS_WEIGHTS,
        boundary_edge_targets,
    )

    if not isinstance(output, tuple) or len(output) != 2:
        raise ValueError("node+edge objective expects (node_logits, edge_logits)")
    node_logits, edge_logits = output
    mask = batch["mask"].to(torch.bool)
    node_logits = node_logits[mask]
    targets = batch["y"][mask].to(torch.long)
    weights = torch.tensor(
        PASCAL_CLASS_WEIGHTS, device=node_logits.device, dtype=node_logits.dtype
    )
    loss = F.cross_entropy(node_logits, targets, weight=weights)
    edge_y, edge_mask = boundary_edge_targets(batch)
    edge_logits = edge_logits[edge_mask]
    edge_y = edge_y[edge_mask]
    if edge_y.numel() > 0:
        edge_w = torch.tensor(
            [1.0, PASCAL_BOUNDARY_POS_WEIGHT],
            device=edge_logits.device,
            dtype=edge_logits.dtype,
        )
        loss = loss + F.cross_entropy(edge_logits, edge_y, weight=edge_w)
    return loss


def _objective_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    opt: torch.optim.Optimizer,
    *,
    objective: str,
) -> float:
    # Must not record probes here: each step would otherwise retain every
    # layer's (B, M, D) edge states for the rest of the run.
    disable_probe(model)
    opt.zero_grad(set_to_none=True)
    logits = model(batch["x"], batch["edge_x"], **_forward_kwargs(batch, model))
    if objective == "graph":
        loss = F.cross_entropy(logits, batch["y"])
    elif objective == "node+edge":
        loss = _pascal_node_edge_loss(logits, batch)
    else:
        raise ValueError(f"unknown objective {objective!r}")
    loss.backward()
    opt.step()
    return float(loss.detach())


def _ce_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    opt: torch.optim.Optimizer,
) -> float:
    """Graph-level CE step (jets). Kept for tests."""
    return _objective_step(model, batch, opt, objective="graph")


def run_one_architecture(
    family: str,
    *,
    depth: int,
    width: int,
    batch: dict[str, torch.Tensor],
    eta_0: float,
    n_adam_steps: int,
    heads: int = 8,
    seed: int = 0,
    device: torch.device | None = None,
    objective: str = "graph",
    out_dim: int | None = None,
    readout_mode: str | None = None,
    n_edge_classes: int = PASCAL_N_EDGE_CLASSES,
    batch_id: int = 0,
) -> pd.DataFrame:
    """Init norms, 1-step Adam Δ, then T-step feature and cumulative Δ."""
    device = device or batch["x"].device
    n_features = int(batch["x"].size(-1))
    n_edge_features = int(batch["edge_x"].size(-1))
    objective = str(objective or "graph")
    if readout_mode is None:
        readout_mode = "node+edge" if objective == "node+edge" else "graph"
    if out_dim is None:
        out_dim = PASCAL_OUT_DIM if objective == "node+edge" else _infer_out_dim(batch)
    model = build_stability_model(
        family,
        depth=depth,
        width=width,
        n_features=n_features,
        n_edge_features=n_edge_features,
        heads=heads,
        out_dim=int(out_dim),
        readout_mode=str(readout_mode),
        n_edge_classes=n_edge_classes,
        seed=seed,
    ).to(device)
    model.train()
    try:
        return _run_one_architecture_body(
            model,
            family=family,
            depth=depth,
            width=width,
            batch=batch,
            eta_0=eta_0,
            n_adam_steps=n_adam_steps,
            heads=heads,
            device=device,
            objective=objective,
            batch_id=batch_id,
        )
    finally:
        disable_probe(model)
        try:
            model.cpu()
        except Exception:
            pass
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _run_one_architecture_body(
    model: nn.Module,
    *,
    family: str,
    depth: int,
    width: int,
    batch: dict[str, torch.Tensor],
    eta_0: float,
    n_adam_steps: int,
    heads: int,
    device: torch.device,
    objective: str,
    batch_id: int,
) -> pd.DataFrame:
    n_params = n_trainable(model)
    lr = adam_lr(eta_0, width)
    opt = torch.optim.Adam(model.parameters(), lr=lr, eps=ADAM_EPS)

    snaps0, _ = forward_probe(model, batch)
    m0 = snap_metrics(snaps0)
    snaps0 = _snaps_cpu(snaps0)
    m1 = m0
    d1 = {stage: {"particle": 0.0, "edge": 0.0} for stage in m0}
    mt = m0
    dt = d1
    loss_t = float("nan")
    n_steps = int(n_adam_steps)
    if n_steps >= 1:
        loss_t = _objective_step(model, batch, opt, objective=objective)
        snaps1, _ = forward_probe(model, batch)
        m1 = snap_metrics(snaps1)
        snaps1 = _snaps_cpu(snaps1)
        d1 = delta_metrics(snaps0, snaps1)
        mt, dt = m1, d1
        del snaps1
        for _ in range(n_steps - 1):
            loss_t = _objective_step(model, batch, opt, objective=objective)
        if n_steps > 1:
            snaps_t, _ = forward_probe(model, batch)
            mt = snap_metrics(snaps_t)
            dt = delta_metrics(snaps0, _snaps_cpu(snaps_t))
            del snaps_t
    del snaps0

    rows = []
    for stage, init in m0.items():
        for token in ("particle", "edge"):
            rows.append(
                {
                    "family": family,
                    "depth": depth,
                    "width": width,
                    "heads": heads if family != "cpen" else pd.NA,
                    "n_params": n_params,
                    "eta_0": eta_0,
                    "lr": lr,
                    "n_adam_steps": int(n_adam_steps),
                    "objective": objective,
                    "batch_id": int(batch_id),
                    "stage": stage,
                    "token": token,
                    "msq_init": init[token],
                    "msq_1": m1.get(stage, {}).get(token, float("nan")),
                    "dmsq_1": d1.get(stage, {}).get(token, float("nan")),
                    "msq_T": mt.get(stage, {}).get(token, float("nan")),
                    "dmsq_T": dt.get(stage, {}).get(token, float("nan")),
                    "loss_T": loss_t,
                }
            )
    del opt
    return pd.DataFrame(rows)


def _as_batch_list(
    batch: dict[str, torch.Tensor] | Sequence[dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    if isinstance(batch, dict) and "x" in batch:
        return [batch]
    return list(batch)


def run_stability_ladder(
    batch: dict[str, torch.Tensor] | Sequence[dict[str, torch.Tensor]],
    *,
    families: Sequence[str] = ("cpen", "capen"),
    archs: Sequence[tuple[int, int]] = DEFAULT_ARCHS,
    eta_0: float = 0.25,
    n_adam_steps: int = DEFAULT_N_ADAM_STEPS,
    heads: int = 8,
    seed: int = 0,
    device: torch.device | None = None,
    quiet: bool = False,
    objective: str = "graph",
    out_dim: int | None = None,
    readout_mode: str | None = None,
    n_edge_classes: int = PASCAL_N_EDGE_CLASSES,
) -> pd.DataFrame:
    """Sweep families × (L, D) over one or more batches."""
    batches = _as_batch_list(batch)
    if not batches:
        raise ValueError("run_stability_ladder needs at least one batch")
    device = device or batches[0]["x"].device
    release_cuda()
    if device.type == "cuda":
        alloc_gib = torch.cuda.memory_allocated(device) / 2**30
        if not quiet:
            print(f"[stability] {cuda_mem_line()}")
        if alloc_gib > _STALE_CUDA_GIB:
            raise RuntimeError(
                f"This process already has {alloc_gib:.2f} GiB allocated on CUDA "
                f"(need a near-empty GPU for the ladder). Kernel → Restart, then "
                f"run from the top. Leftover ipykernel PIDs from earlier OOM runs "
                f"will also pin the device — check nvidia-smi and kill those."
            )
    frames: list[pd.DataFrame] = []
    n_batches = len(batches)
    for batch_id, one in enumerate(batches):
        for family in families:
            for depth, width in archs:
                if not quiet:
                    print(
                        f"[stability] batch {batch_id + 1}/{n_batches}  "
                        f"{family:12s}  L={depth} D={width}  T={n_adam_steps}  "
                        f"obj={objective}"
                    )
                frames.append(
                    run_one_architecture(
                        family,
                        depth=int(depth),
                        width=int(width),
                        batch=one,
                        eta_0=eta_0,
                        n_adam_steps=n_adam_steps,
                        heads=heads,
                        seed=seed,
                        device=device,
                        objective=objective,
                        out_dim=out_dim,
                        readout_mode=readout_mode,
                        n_edge_classes=n_edge_classes,
                        batch_id=batch_id,
                    )
                )
    return pd.concat(frames, ignore_index=True)


def load_jets_knn_batch(
    data_root: str | Path,
    *,
    n_jets: int = 16,
    num_particles: int = 128,
    k: int = 8,
    split: str = "train",
    seed: int = 0,
    pool: int = 4096,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Sample jets from a short HDF5 prefix, stack, and build directed kNN graphs.

    ``pool`` rows are read (not the full 1.2M train split) so the notebook can
    start without paging the whole table into RAM.
    """
    import numpy as np
    import pandas as pd

    from cpen.apps.jets.toptagging import (
        MAX_NUM_PARTICLES,
        N_PARTICLE_FEATURES,
        preprocess_particle_features,
        raw_hdf5_path,
    )
    from cpen.graphs.graphs import build_knn_graph
    from cpen.graphs.sparse_incidence import attach_sparse_incidence_batch

    device = device or torch.device("cpu")
    hdf5_path = raw_hdf5_path(data_root, split)
    n_want = int(n_jets)
    n_pool = max(int(pool), n_want)
    table = pd.read_hdf(hdf5_path, key="table", start=0, stop=n_pool)
    data = np.asarray(table)
    n_avail = int(data.shape[0])
    feat = MAX_NUM_PARTICLES * N_PARTICLE_FEATURES
    particles = data[:, :feat].reshape(-1, MAX_NUM_PARTICLES, N_PARTICLE_FEATURES)[
        :, : int(num_particles)
    ]
    labels = data[:, -1]
    g = torch.Generator().manual_seed(int(seed))
    n = min(n_want, n_avail)
    perm = torch.randperm(n_avail, generator=g)[:n]
    xs_raw, xs, masks, ys = [], [], [], []
    for i in perm.tolist():
        part = torch.from_numpy(particles[i].astype(np.float32))
        x_raw, x_model, mask = preprocess_particle_features(part)
        xs_raw.append(x_raw)
        xs.append(x_model)
        masks.append(mask)
        ys.append(int(labels[i]))
    x_raw = torch.stack(xs_raw, dim=0)
    x = torch.stack(xs, dim=0)
    mask = torch.stack(masks, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    pt = torch.hypot(x_raw[..., 1], x_raw[..., 2]) * mask.to(x_raw.dtype)
    z = pt / pt.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    _, edge_x, incidence = build_knn_graph(x_raw, k=int(k), mask=mask)
    batch = {
        "x": x,
        "x_raw": x_raw,
        "edge_x": edge_x,
        "incidence": incidence,
        "mask": mask,
        "z": z,
        "y": y,
    }
    batch = attach_sparse_incidence_batch(batch)
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def load_jets_knn_batches(
    data_root: str | Path,
    *,
    n_jets: int = 16,
    n_batches: int = DEFAULT_N_BATCHES,
    num_particles: int = 128,
    k: int = 8,
    split: str = "train",
    seed: int = 0,
    pool: int = 4096,
    device: torch.device | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Independent TopTagging minibatches (different ``seed`` per batch)."""
    return [
        load_jets_knn_batch(
            data_root,
            n_jets=n_jets,
            num_particles=num_particles,
            k=k,
            split=split,
            seed=int(seed) + i,
            pool=pool,
            device=device,
        )
        for i in range(int(n_batches))
    ]


def load_pascal_batch(
    data_root: str | Path,
    *,
    n_graphs: int = 8,
    split: str = "train",
    seed: int = 0,
    offset: int = 0,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """One padded PascalVOC-SP minibatch from the incidence-v2 cache."""
    from cpen.apps.pascal.pascal_graph_cache import CachedPascalGraphDataset

    device = device or torch.device("cpu")
    ds = CachedPascalGraphDataset(data_root=data_root, split=split)
    n_avail = len(ds)
    n_want = int(n_graphs)
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n_avail, generator=g)
    start = int(offset) * n_want
    if start >= n_avail:
        raise ValueError(f"Pascal batch offset {offset} is past split size {n_avail}")
    idx = perm[start : start + n_want].tolist()
    if len(idx) < n_want:
        raise ValueError(
            f"need {n_want} Pascal graphs from offset {offset}; only {len(idx)} remain"
        )
    batch = ds.collate_samples(idx)
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def load_pascal_batches(
    data_root: str | Path,
    *,
    n_graphs: int = 8,
    n_batches: int = DEFAULT_N_BATCHES,
    split: str = "train",
    seed: int = 0,
    device: torch.device | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Disjoint PascalVOC-SP minibatches from one permutation of the split."""
    return [
        load_pascal_batch(
            data_root,
            n_graphs=n_graphs,
            split=split,
            seed=seed,
            offset=i,
            device=device,
        )
        for i in range(int(n_batches))
    ]


def _family_pretty(name: str) -> str:
    key = str(name).lower().replace("_", "-")
    return {
        "cpen": "CPEN",
        "capen": "CAPEN",
        "capen-llama": "CAPEN-Llama",
    }.get(key, str(name))


def _expand_ylim(ax: plt.Axes, *, frac: float = 0.32, log_frac: float = 0.07) -> None:
    """Pad y-limits so traces are not flush with the frame.

    Log axes here can span four decades (hidden ρ vs pooled logits), so they
    take a much smaller fractional pad than linear ones.
    """
    lo, hi = ax.get_ylim()
    if not (hi > lo) or not (lo < float("inf") and hi < float("inf")):
        return
    if ax.get_yscale() == "log":
        if lo <= 0:
            return
        ratio = (hi / lo) ** log_frac
        ax.set_ylim(lo / ratio, hi * ratio)
        return
    pad = (hi - lo) * frac
    ax.set_ylim(lo - pad, hi + pad)


_TOKEN_COLOR = {"particle": "C0", "edge": "C1"}
_TOKEN_MARKER = {"particle": "o", "edge": "s"}
_Q_LS = {"rho": "-", "drho": "--"}
_Q_LABEL = {"rho": r"$\rho$", "drho": r"$\Delta\rho$"}
_ROW_KINDS = (
    ("encode",),
    ("first", "last"),
    ("decode",),
)
_ROW_TITLE = (
    r"encoder $\ell=0$",
    r"residuals $\ell=1,L$",
    r"decoder $\ell=L{+}1$",
)
_FIGSIZE_GRID = (7.6, 6.6)
_FIGSIZE_LAYERS = (3.4, 2.25)
_MARKER_MS = 3.2


def _stage_kind(stage: str, depth: int) -> str | None:
    if stage == "encode":
        return "encode"
    if stage == "decode":
        return "decode"
    depth = int(depth)
    if stage == "layer0" and depth > 1:
        return "first"
    if stage == last_layer_stage(depth):
        return "last"
    return None


def last_layer_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Keep encoder, first residual, last residual, and decoder."""
    if df.empty:
        return df
    kinds = [
        _stage_kind(str(stage), int(depth))
        for stage, depth in zip(df["stage"], df["depth"])
    ]
    out = df.copy()
    out["stage_kind"] = kinds
    return out.loc[out["stage_kind"].notna()].reset_index(drop=True)


def _mean_std(
    sub: pd.DataFrame, xcol: str, ycol: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sub.empty or ycol not in sub.columns:
        return np.array([]), np.array([]), np.array([])
    grouped = sub.groupby(xcol, sort=True)[ycol]
    x = grouped.mean().index.to_numpy(dtype=float)
    y = grouped.mean().to_numpy(dtype=float)
    std = grouped.std(ddof=1).to_numpy(dtype=float)
    std = np.where(np.isfinite(std), std, 0.0)
    return x, y, std


def _draw_mean_band(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    *,
    token: str,
    quantity: str,
    filled: bool = True,
    ls: str | None = None,
    label: str | None = None,
) -> None:
    if x.size == 0:
        return
    color = _TOKEN_COLOR[token]
    marker = _TOKEN_MARKER[token]
    ls = _Q_LS[quantity] if ls is None else ls
    mfc = color if filled else "none"
    lo = np.clip(y - yerr, np.maximum(y * 1e-6, np.finfo(float).tiny), None)
    hi = y + yerr
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0, zorder=1)
    ax.errorbar(
        x,
        y,
        yerr=np.vstack([y - lo, np.clip(hi - y, 0.0, None)]),
        color=color,
        marker=marker,
        ls=ls,
        ms=_MARKER_MS,
        lw=1.15,
        capsize=2.0,
        capthick=0.7,
        elinewidth=0.7,
        markerfacecolor=mfc,
        markeredgecolor=color,
        markeredgewidth=0.7,
        zorder=2,
        label=label,
    )


def _stability_legend_handles(*, include_residual: bool) -> tuple[list[Line2D], list[str]]:
    handles: list[Line2D] = []
    labels: list[str] = []
    for token in ("particle", "edge"):
        handles.append(
            Line2D(
                [0],
                [0],
                color=_TOKEN_COLOR[token],
                marker=_TOKEN_MARKER[token],
                ls="None",
                ms=5,
                markerfacecolor=_TOKEN_COLOR[token],
                markeredgecolor=_TOKEN_COLOR[token],
            )
        )
        labels.append(token)
    handles.append(Line2D([0], [0], color="none", lw=0, marker="None"))
    labels.append("")
    for quantity, ls in _Q_LS.items():
        handles.append(Line2D([0], [0], color="0.25", ls=ls, lw=1.6, marker="None"))
        labels.append(_Q_LABEL[quantity])
    if include_residual:
        handles.append(Line2D([0], [0], color="none", lw=0, marker="None"))
        labels.append("")
        handles.append(
            Line2D(
                [0],
                [0],
                color="0.25",
                marker="o",
                ls="None",
                ms=5,
                markerfacecolor="none",
                markeredgecolor="0.25",
            )
        )
        labels.append(r"$\ell=1$")
        handles.append(
            Line2D(
                [0],
                [0],
                color="0.25",
                marker="o",
                ls="None",
                ms=5,
                markerfacecolor="0.25",
                markeredgecolor="0.25",
            )
        )
        labels.append(r"$\ell=L$")
    return handles, labels


def _plot_stability_grid(
    plot_df: pd.DataFrame,
    *,
    family: str,
    n_adam_steps: int,
    figsize: tuple[float, float],
    stem: str | None,
    fig_dir: Path | None,
    show: bool,
) -> plt.Figure:
    tokens = ("particle", "edge")
    col_specs: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
        ("Init", (("msq_init", "rho"),)),
        ("After 1 Adam step", (("msq_1", "rho"), ("dmsq_1", "drho"))),
        (
            rf"After $T={n_adam_steps}$ Adam steps",
            (("msq_T", "rho"), ("dmsq_T", "drho")),
        ),
    )
    fam_df = plot_df.loc[plot_df["family"].eq(family)]
    fig, axes = plt.subplots(3, 3, figsize=figsize, dpi=200, sharex=True)
    include_residual = "first" in set(fam_df["stage_kind"])
    for row, kinds in enumerate(_ROW_KINDS):
        for col, (title, series) in enumerate(col_specs):
            ax = axes[row, col]
            for kind in kinds:
                filled = kind != "first"
                for token in tokens:
                    sub = fam_df.loc[
                        fam_df["token"].eq(token) & fam_df["stage_kind"].eq(kind)
                    ]
                    for ycol, quantity in series:
                        x, y, yerr = _mean_std(sub, "n_params", ycol)
                        _draw_mean_band(
                            ax, x, y, yerr, token=token, quantity=quantity, filled=filled
                        )
            ax.set_yscale("log")
            ax.set_xscale("log")
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(_ROW_TITLE[row])
            if row == 2:
                ax.set_xlabel(r"$\#$ params")
            _expand_ylim(ax)

    n_batches = int(fam_df["batch_id"].nunique()) if "batch_id" in fam_df.columns else 1
    handles, labels = _stability_legend_handles(include_residual=include_residual)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(handles),
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.4,
    )
    fig.suptitle(
        rf"{_family_pretty(family)}  (mean $\pm$ 1 std, $n={n_batches}$ batches)",
        y=1.08,
        fontsize=9,
    )
    fig.tight_layout()
    if stem:
        savefig(fig, stem, fig_dir=fig_dir or DEFAULT_FIG_DIR)
    if show:
        plt.show()
    return fig


def plot_stability_vs_params(
    df: pd.DataFrame,
    *,
    n_adam_steps: int | None = None,
    figsize: tuple[float, float] | None = None,
    stem: str | None = "jets_cpen_capen_stability",
    fig_dir: Path | None = None,
    show: bool = True,
    family: str | None = None,
) -> plt.Figure | list[plt.Figure] | None:
    """3×3 grid vs ``#`` params: rows are encoder / first+last residual / decoder.

    Columns are init, 1 Adam step, and T Adam steps. Particle traces are blue
    circles; edge traces are orange squares. Solid = ρ, dashed = Δρ. Open
    markers on the residual row are ℓ=1; filled markers are ℓ=L. Bands are
    mean ± 1 std across batches. One figure per family.
    """
    plot_df = last_layer_frame(df)
    if plot_df.empty:
        print("No stability rows to plot.")
        return None
    if n_adam_steps is None:
        n_adam_steps = int(plot_df["n_adam_steps"].iloc[0])
    if figsize is None:
        figsize = _FIGSIZE_GRID
    families = list(dict.fromkeys(plot_df["family"]))
    if family is not None:
        families = [family]
    figs: list[plt.Figure] = []
    for fam in families:
        fam_stem = None if stem is None else (
            stem if len(families) == 1 else f"{stem}_{str(fam).replace('-', '_')}"
        )
        figs.append(
            _plot_stability_grid(
                plot_df,
                family=str(fam),
                n_adam_steps=int(n_adam_steps),
                figsize=figsize,
                stem=fam_stem,
                fig_dir=fig_dir,
                show=show,
            )
        )
    if not figs:
        return None
    return figs[0] if len(figs) == 1 else figs


def plot_layerwise_msq(
    df: pd.DataFrame,
    *,
    depth: int,
    width: int,
    ycol: str = "msq_init",
    stem: str | None = None,
    fig_dir: Path | None = None,
    show: bool = True,
) -> plt.Figure | None:
    """ρ vs layer index at one architecture (sanity that depth does not explode)."""
    sub = df.loc[df["depth"].eq(depth) & df["width"].eq(width)].copy()
    if sub.empty:
        print(f"No rows for L={depth}, D={width}.")
        return None

    def _layer_idx(stage: str) -> int:
        if stage == "encode":
            return 0
        if stage == "decode":
            return int(depth) + 1
        if stage.startswith("layer"):
            return int(stage.replace("layer", "")) + 1
        return -1

    sub["ell"] = sub["stage"].map(_layer_idx)
    sub = sub.loc[sub["ell"] >= 0]
    fig, ax = plt.subplots(figsize=_FIGSIZE_LAYERS, dpi=200)
    families = list(dict.fromkeys(sub["family"]))
    fam_ls = {"cpen": "-", "capen": ":", "capen-llama": "-."}
    for fam in families:
        for token in ("particle", "edge"):
            sw = sub.loc[sub["family"].eq(fam) & sub["token"].eq(token)]
            x, y, yerr = _mean_std(sw, "ell", ycol)
            if x.size == 0:
                continue
            _draw_mean_band(
                ax,
                x,
                y,
                yerr,
                token=token,
                quantity="rho",
                filled=True,
                ls=fam_ls.get(str(fam).lower().replace("_", "-"), "-"),
                label=f"{_family_pretty(fam)}, {token}",
            )
    ax.set_xlabel(r"layer ($\ell=0$ encoder, $\ell=L$ last residual, $\ell=L{+}1$ decoder)")
    ylabel = {
        "msq_init": r"init $\rho$",
        "dmsq_1": r"1-step $\Delta\rho$",
        "msq_1": r"after 1 step $\rho$",
        "msq_T": rf"after $T$ $\rho$",
        "dmsq_T": rf"after $T$ $\Delta\rho$",
    }.get(ycol, ycol)
    ax.set_ylabel(ylabel)
    ax.set_title(rf"$L={depth}$, $D={width}$")
    _expand_ylim(ax, frac=0.12)
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    if stem:
        savefig(fig, stem, fig_dir=fig_dir or DEFAULT_FIG_DIR)
    if show:
        plt.show()
    return fig
