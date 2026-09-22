"""Load a mock-pretrained Lightning checkpoint into a stream Lit module."""

from __future__ import annotations

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


def load_init_checkpoint(lit_module, path: str | Path, *, strict: bool = False) -> list[str]:
    """Copy compatible weights from ``path`` into ``lit_module``.

    Accepts a Lightning ``.ckpt`` (``state_dict`` under the usual key) or a raw
    ``state_dict``. Returns the list of missing keys after the load.
    """
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"--init-from checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint payload type: {type(state)!r}")

    current = lit_module.state_dict()
    filtered = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
    skipped = sorted(set(state) - set(filtered))
    missing, unexpected = lit_module.load_state_dict(filtered, strict=False)
    log_info(
        f"[init-from] loaded {len(filtered)}/{len(current)} tensors from {ckpt_path.name} "
        f"(skipped_shape_or_absent={len(skipped)}, missing_after={len(missing)}, "
        f"unexpected={len(unexpected)}, strict={strict})"
    )
    if strict and (missing or unexpected or skipped):
        raise RuntimeError(
            f"strict init-from failed: missing={missing[:8]}… "
            f"unexpected={unexpected[:8]}… skipped={skipped[:8]}…"
        )
    return list(missing)


def freeze_except_last_blocks(model: nn.Module, n_blocks: int = 1) -> dict[str, Any]:
    """Freeze encoder + early blocks; train the last ``n_blocks`` + decoders.

    CPEN stores depth as ModuleList entries ``0 … L-1``. With ``n_blocks=1`` only
    layer ``L-1`` (and ``decoder_x`` / ``decoder_e``) receive gradients.
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

    for name in ("decoder_x", "decoder_e", "decoder"):
        head = getattr(model, name, None)
        if head is None:
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
        f"[freeze] train last {n}/{depth} block(s) + decoders | "
        f"trainable {n_train:,}/{n_total:,} ({100 * info['trainable_frac']:.1f}%) "
        f"indices={trainable_blocks}"
    )
    return info
