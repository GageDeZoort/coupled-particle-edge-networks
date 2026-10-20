"""Minimal model registry (``registry.pyc`` was also lost with the backends)."""

from __future__ import annotations

import math
from typing import Any


def build_model(
    args: Any,
    *,
    n_features: int,
    n_edge_features: int,
    out_dim: int,
    depth: int,
    width: int,
    heads: int = 1,
) -> Any:
    model_name = str(getattr(args, "model", "capen") or "capen")
    common = dict(
        n_features=n_features,
        n_edge_features=n_edge_features,
        out_dim=out_dim,
        depth=depth,
        width=width,
        heads=heads,
        dropout=float(getattr(args, "dropout", 0.0) or 0.0),
        normalization=getattr(args, "normalization", "uniform"),
        energy_index=int(getattr(args, "energy_index", 0) or 0),
        optimizer=getattr(args, "optimizer", "adam"),
        readout_mode=getattr(args, "readout_mode", "graph"),
    )
    if model_name == "capen":
        from cpen.models.capen import CAPEN

        return CAPEN(**common)
    if model_name == "capen-llama":
        from cpen.models.capen_llama import CAPENLlama

        return CAPENLlama(
            **common,
            all_to_all_particle_attention=True,
            use_wire=bool(getattr(args, "use_wire", False)),
        )
    if model_name in {"capen-llama-att", "baseline-transformer"}:
        from cpen.models.capen_llama_att import BaselineTransformer, CAPENLlamaAtt

        cls = BaselineTransformer if model_name == "baseline-transformer" else CAPENLlamaAtt
        return cls(
            **common,
            n_class_tokens=int(out_dim),
            all_to_all_particle_attention=True,
            uniform_output_pool=bool(getattr(args, "uniform_output_pool", False)),
        )
    if model_name == "cpen":
        from cpen.models.cpen import CPEN

        return CPEN(
            n_features=n_features,
            n_edge_features=n_edge_features,
            out_dim=out_dim,
            depth=depth,
            width=width,
            operators=getattr(args, "operators", "incidence"),
            normalization=getattr(args, "normalization", "uniform"),
            energy_index=int(getattr(args, "energy_index", 0) or 0),
            optimizer=getattr(args, "optimizer", "adam"),
            readout_mode=getattr(args, "readout_mode", "graph"),
            dropout=float(getattr(args, "dropout", 0.0) or 0.0),
        )
    if model_name in {"particle-only", "particle_only"}:
        from cpen.models.particle_only import ParticleOnlyNetwork

        return ParticleOnlyNetwork(
            n_features=n_features,
            out_dim=out_dim,
            depth=depth,
            width=width,
            normalization=getattr(args, "normalization", "uniform"),
            energy_index=int(getattr(args, "energy_index", 0) or 0),
            optimizer=getattr(args, "optimizer", "adam"),
            readout_mode=getattr(args, "readout_mode", "graph"),
        )
    raise ValueError(f"Unknown model {model_name!r}")


def count_trainable_parameters(model: Any) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def maybe_compile_model(model: Any, args: Any) -> Any:
    """Optionally wrap ``model`` with ``torch.compile`` (default on; ``--no-torch-compile``)."""
    if not bool(getattr(args, "torch_compile", True)):
        return model
    import torch

    return torch.compile(model)


def weight_decay_for_run(
    args: Any,
    *,
    width: int,
    n_train: int | None = None,
    t_epoch: float | None = None,
    eta_0: float | None = None,
) -> float:
    """Return AdamW weight decay, or 0 for Adam.

    When ``optimizer=adamw`` and ``t_epoch`` / ``n_train`` / ``eta_0`` are set
    (the ``train_one_run`` path), use the CompleteP t-epoch schedule:

        λ = λ₀ √D,  λ₀ = B / (t_epoch · η₀ · n_train)

    Otherwise fall back to ``args.weight_decay`` (×√D for AdamW if > 0).
    """
    from cpen.models.base import t_epoch_weight_decay

    if getattr(args, "optimizer", "adam") != "adamw":
        return float(getattr(args, "weight_decay", 0.0) or 0.0)

    if (
        t_epoch is not None
        and n_train is not None
        and eta_0 is not None
        and int(n_train) > 0
        and float(t_epoch) > 0
        and float(eta_0) > 0
    ):
        return t_epoch_weight_decay(
            eta_0=float(eta_0),
            width=int(width),
            batch_size=int(getattr(args, "batch_size", 1) or 1),
            n_gpus=int(getattr(args, "n_gpus", 1) or 1),
            n_train=int(n_train),
            t_epoch=float(t_epoch),
        )

    wd = float(getattr(args, "weight_decay", 0.0) or 0.0)
    if wd > 0:
        return wd * math.sqrt(width)
    return wd
