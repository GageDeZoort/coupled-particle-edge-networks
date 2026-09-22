"""Warm-start helpers: load a pretrained Lightning ckpt into a new Lit module.

Used by stream mock→real fine-tunes and JetClass→TopTagging fine-tunes.
Weights only — optimizer / global_step always start fresh on the FT run.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from cpen.utils.log_utils import log_info

# Per-layer ModuleLists in CPEN / CAPEN-style coupled nets (depth = len).
_LAYER_MODULELISTS = (
    "f11_weights",
    "f12_weights",
    "f21_weights",
    "f22_weights",
    "mlp_x_w1",
    "mlp_x_w2",
    "mlp_e_w1",
    "mlp_e_w2",
)

# Readout / class-head parameter name fragments (JetClass 10-way ≠ TopTagging 2-way).
_HEAD_NAME_RE = re.compile(
    r"(^|\.)(decoder_x|decoder_e|decoder|output_attention_pool|class_tokens|"
    r"class_unembed|attn_x|attn_e)($|\.)"
)


def is_readout_param(name: str) -> bool:
    """True for class-token pool / unembed / decoder heads."""
    return _HEAD_NAME_RE.search(name) is not None


def load_init_checkpoint(
    lit_module,
    path: str | Path,
    *,
    strict: bool = False,
    skip_readout: bool = True,
    min_load_frac: float = 0.5,
) -> list[str]:
    """Copy compatible weights from ``path`` into ``lit_module``.

    Accepts a Lightning ``.ckpt`` (``state_dict`` under the usual key) or a raw
    ``state_dict``. Returns the list of missing keys after the load.

    When ``skip_readout`` is True (default), decoder / class-token / unembed
    tensors are never copied — the FT model keeps its freshly initialized head
    (required for JetClass 10-way → TopTagging 2-way).
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"--init-from checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint payload type: {type(state)!r}")

    current = lit_module.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped_head: list[str] = []
    skipped_shape: list[str] = []
    for k, v in state.items():
        if k not in current:
            continue
        if skip_readout and is_readout_param(k):
            skipped_head.append(k)
            continue
        if current[k].shape != v.shape:
            skipped_shape.append(k)
            continue
        filtered[k] = v

    missing, unexpected = lit_module.load_state_dict(filtered, strict=False)
    n_cur = len(current)
    frac = float(len(filtered)) / max(float(n_cur), 1.0)
    log_info(
        f"[init-from] loaded {len(filtered)}/{n_cur} tensors ({100 * frac:.1f}%) "
        f"from {ckpt_path.name} "
        f"(skip_head={len(skipped_head)}, skip_shape={len(skipped_shape)}, "
        f"missing_after={len(missing)}, unexpected={len(unexpected)}, strict={strict})"
    )
    if frac < float(min_load_frac):
        raise RuntimeError(
            f"init-from loaded only {100 * frac:.1f}% of parameters "
            f"(min_load_frac={min_load_frac}); refusing to continue"
        )
    if strict and (missing or unexpected or skipped_shape):
        raise RuntimeError(
            f"strict init-from failed: missing={missing[:8]}… "
            f"unexpected={unexpected[:8]}… skipped_shape={skipped_shape[:8]}…"
        )
    return list(missing)


def read_ckpt_eta0(path: str | Path) -> float | None:
    """Best-effort ``eta_0`` from a Lightning ckpt (hparams / metadata)."""
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        return None
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for key in ("hyper_parameters", "hparams"):
        hp = payload.get(key)
        if isinstance(hp, dict) and "eta_0" in hp:
            try:
                return float(hp["eta_0"])
            except (TypeError, ValueError):
                pass
    # Some runs stash static_metadata under callbacks / custom keys.
    for key, val in payload.items():
        if isinstance(val, dict) and "eta_0" in val:
            try:
                return float(val["eta_0"])
            except (TypeError, ValueError):
                continue
    return None


def read_ckpt_global_step(path: str | Path) -> int:
    """``global_step`` from a Lightning ckpt, or 0."""
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        return 0
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    try:
        return int(payload.get("global_step", 0) or 0)
    except (TypeError, ValueError):
        return 0


def freeze_except_last_blocks(model: nn.Module, n_blocks: int = 1) -> dict[str, Any]:
    """Freeze encoder + early blocks; train the last ``n_blocks`` + decoders / readout.

    CPEN stores depth as ModuleList entries ``0 … L-1``. With ``n_blocks=1`` only
    layer ``L-1`` (and decoder / output pool) receive gradients.
    """
    n_blocks = int(n_blocks)
    if n_blocks <= 0:
        raise ValueError(f"n_blocks must be >= 1; got {n_blocks}")

    for p in model.parameters():
        p.requires_grad = False

    depth = None
    for attr in _LAYER_MODULELISTS:
        modlist = getattr(model, attr, None)
        if isinstance(modlist, nn.ModuleList) and len(modlist) > 0:
            depth = len(modlist)
            break
    if depth is None:
        raise ValueError(
            "freeze_except_last_blocks: model has no CPEN-style layer ModuleLists "
            f"(looked for {_LAYER_MODULELISTS})"
        )
    n = min(n_blocks, depth)
    start = depth - n
    trainable_blocks = list(range(start, depth))
    for attr in _LAYER_MODULELISTS:
        modlist = getattr(model, attr, None)
        if not isinstance(modlist, nn.ModuleList):
            continue
        for i in trainable_blocks:
            for p in modlist[i].parameters():
                p.requires_grad = True

    for name in ("decoder_x", "decoder_e", "decoder", "output_attention_pool"):
        head = getattr(model, name, None)
        if head is None:
            continue
        if isinstance(head, nn.Parameter):
            head.requires_grad = True
            continue
        for p in head.parameters():
            p.requires_grad = True

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    info = {
        "n_blocks": n,
        "trainable_block_indices": trainable_blocks,
        "n_trainable_params": int(n_train),
        "n_total_params": int(n_total),
        "trainable_frac": float(n_train) / max(float(n_total), 1.0),
    }
    log_info(
        f"[freeze] train last {n}/{depth} block(s) + readout | "
        f"trainable {n_train:,}/{n_total:,} ({100 * info['trainable_frac']:.1f}%) "
        f"indices={trainable_blocks}"
    )
    return info


def assert_ft_root_not_pretrain(root: str | Path) -> None:
    """Refuse FT ``--root`` values that resolve under a JetClass pretrain tree."""
    text = str(Path(root).expanduser().resolve()).lower()
    forbidden = ("jetclass_output", "jetclass_scaling_law", "jetclass_runs")
    # Allow ".../toptagging_finetune" even if a parent path string is noisy.
    if "toptagging_finetune" in text:
        return
    for frag in forbidden:
        if frag in text:
            raise ValueError(
                f"Fine-tune --root must not point at a JetClass pretrain tree "
                f"(found {frag!r} in {root}). "
                "Use e.g. .../cpen_runs/toptagging_finetune"
            )
