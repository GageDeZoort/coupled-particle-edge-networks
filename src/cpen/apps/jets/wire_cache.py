"""Precomputed WIRE spectral-coordinate sidecars for star-$R$ caches.

Stored beside the parent star graph cache so existing ``train.pt`` / ``val.pt``
/ ``test.pt`` stay untouched::

  {data_root}/processed/star-r{R}/n{N}/wire-m{m}-norm/
    meta.json
    train.pt
    val.pt
    test.pt

Each ``{split}.pt`` holds ``wire_coordinates`` with shape ``[n_jets, N, m]``
aligned to the parent star-cache jet order. Sign augmentation is *not* baked
in — it remains a train-time option.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from cpen.utils.log_utils import log_info
from cpen.utils.star_graph_cache import (
    load_star_graph_cache_mmap,
    open_star_graph_cache,
    processed_star_split_dir,
    star_radius_tag,
)
from cpen.utils.wire_coordinates import (
    compute_wire_coordinates_batched,
    particle_adjacency_from_incidence,
)


def wire_cache_tag(
    *,
    coordinate_dim: int,
    normalized_laplacian: bool = True,
) -> str:
    lap = "norm" if normalized_laplacian else "comb"
    return f"wire-m{int(coordinate_dim)}-{lap}"


def processed_wire_cache_dir(
    data_root: str | Path,
    *,
    radius: float,
    num_particles: int,
    coordinate_dim: int,
    normalized_laplacian: bool = True,
) -> Path:
    return (
        processed_star_split_dir(
            data_root,
            radius=radius,
            num_particles=num_particles,
        )
        / wire_cache_tag(
            coordinate_dim=coordinate_dim,
            normalized_laplacian=normalized_laplacian,
        )
    )


def wire_cache_meta(
    *,
    radius: float,
    num_particles: int,
    coordinate_dim: int,
    normalized_laplacian: bool,
    standardize: bool,
    canonicalize_sign: bool,
    n_jets: int | None = None,
    split: str | None = None,
) -> dict:
    meta = {
        "kind": "wire-coordinates-v1",
        "source": "particle-cooccurrence-STS",
        "parent_construction": "star-radius",
        "radius": float(radius),
        "graph_construction": f"star-R={radius:g}",
        "num_particles": int(num_particles),
        "wire_coordinate_dim": int(coordinate_dim),
        "wire_normalized_laplacian": bool(normalized_laplacian),
        "wire_standardize": bool(standardize),
        "wire_canonicalize_sign": bool(canonicalize_sign),
        "parent_star_tag": star_radius_tag(radius),
        "wire_tag": wire_cache_tag(
            coordinate_dim=coordinate_dim,
            normalized_laplacian=normalized_laplacian,
        ),
    }
    if split is not None:
        meta["split"] = split
    if n_jets is not None:
        meta["n_jets"] = int(n_jets)
    return meta


def _meta_matches(meta: dict, expected: dict) -> bool:
    ignore = {"split", "n_jets"}
    a = {k: v for k, v in meta.items() if k not in ignore}
    b = {k: v for k, v in expected.items() if k not in ignore}
    return a == b


def missing_wire_cache_message(
    *,
    data_root: str,
    split: str,
    radius: float,
    num_particles: int,
    coordinate_dim: int,
    normalized_laplacian: bool = True,
) -> str:
    cache_dir = processed_wire_cache_dir(
        data_root,
        radius=radius,
        num_particles=num_particles,
        coordinate_dim=coordinate_dim,
        normalized_laplacian=normalized_laplacian,
    )
    return (
        f"Missing precomputed WIRE coordinate cache for split={split!r} at {cache_dir}/.\n"
        "Build once (login / CPU node is fine):\n"
        "  python scans/testing/build_toptagging_wire_coordinates.py \\\n"
        f"    --data-root {data_root} \\\n"
        f"    --radius {radius:g} \\\n"
        f"    --num-particles {num_particles} \\\n"
        f"    --wire-coordinate-dim {coordinate_dim}"
        + ("" if normalized_laplacian else " \\\n    --wire-combinatorial-laplacian")
        + "\n"
    )


def wire_cache_available(
    *,
    data_root: str,
    split: str,
    radius: float,
    num_particles: int,
    coordinate_dim: int,
    normalized_laplacian: bool = True,
    standardize: bool = True,
    canonicalize_sign: bool = True,
) -> bool:
    cache_dir = processed_wire_cache_dir(
        data_root,
        radius=radius,
        num_particles=num_particles,
        coordinate_dim=coordinate_dim,
        normalized_laplacian=normalized_laplacian,
    )
    cache_path = cache_dir / f"{split}.pt"
    meta_path = cache_dir / "meta.json"
    if not cache_path.is_file() or not meta_path.is_file():
        return False
    meta = json.loads(meta_path.read_text())
    expected = wire_cache_meta(
        radius=radius,
        num_particles=num_particles,
        coordinate_dim=coordinate_dim,
        normalized_laplacian=normalized_laplacian,
        standardize=standardize,
        canonicalize_sign=canonicalize_sign,
    )
    return _meta_matches(meta, expected)


def build_or_load_wire_coordinate_cache(
    *,
    data_root: str,
    split: str,
    radius: float,
    num_particles: int,
    coordinate_dim: int,
    normalized_laplacian: bool = True,
    standardize: bool = True,
    canonicalize_sign: bool = True,
    max_jets: int | None = None,
    seed: int = 42,
    rebuild: bool = False,
    chunk_size: int = 256,
    show_progress: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Build WIRE coords from the parent star-cache incidence, or load if present.

    Uses the dense ``incidence`` tensor already stored in the star ``{split}.pt``
    (not loaded during normal training). Coordinates are detached float32.
    """
    cache_dir = processed_wire_cache_dir(
        data_root,
        radius=radius,
        num_particles=num_particles,
        coordinate_dim=coordinate_dim,
        normalized_laplacian=normalized_laplacian,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}.pt"
    meta_path = cache_dir / "meta.json"

    expected = wire_cache_meta(
        radius=radius,
        num_particles=num_particles,
        coordinate_dim=coordinate_dim,
        normalized_laplacian=normalized_laplacian,
        standardize=standardize,
        canonicalize_sign=canonicalize_sign,
        split=split,
    )

    if cache_path.is_file() and meta_path.is_file() and not rebuild:
        meta = json.loads(meta_path.read_text())
        if _meta_matches(meta, expected):
            log_info(f"[wire] split={split} loading existing cache {cache_path}")
            return load_star_graph_cache_mmap(cache_path)

    # Parent star cache (full split; max_jets only affects training subsample).
    star_payload, _ = open_star_graph_cache(
        data_root=data_root,
        split=split,
        radius=radius,
        num_particles=num_particles,
        max_jets=None,
        seed=seed,
    )
    if "incidence" not in star_payload:
        raise KeyError(
            f"Star cache for split={split!r} has no dense 'incidence' field; "
            "rebuild the star graph cache."
        )
    incidence = star_payload["incidence"]
    mask = star_payload["mask"]
    n_jets = int(incidence.size(0))
    expected["n_jets"] = n_jets

    if show_progress:
        log_info(
            f"[wire] split={split} computing coords m={coordinate_dim} "
            f"n_jets={n_jets} -> {cache_path}"
        )

    chunks: list[torch.Tensor] = []
    t0 = time.perf_counter()
    for start in range(0, n_jets, chunk_size):
        end = min(start + chunk_size, n_jets)
        inc = incidence[start:end]
        m = mask[start:end]
        adj = particle_adjacency_from_incidence(inc, mask=m)
        coords = compute_wire_coordinates_batched(
            adj,
            m,
            coordinate_dim,
            normalized_laplacian=normalized_laplacian,
            standardize=standardize,
            canonicalize_sign=canonicalize_sign,
            warn_on_pad=False,
        )
        chunks.append(coords.cpu())
        if show_progress and ((end % (chunk_size * 20) == 0) or end == n_jets):
            elapsed = time.perf_counter() - t0
            rate = end / max(elapsed, 1e-6)
            log_info(f"  [{split}] {end:,}/{n_jets:,} jets ({rate:,.0f} jets/s)")

    payload = {"wire_coordinates": torch.cat(chunks, dim=0)}
    if payload["wire_coordinates"].shape != (n_jets, num_particles, coordinate_dim):
        raise RuntimeError(
            f"WIRE cache shape mismatch: got {tuple(payload['wire_coordinates'].shape)}, "
            f"expected {(n_jets, num_particles, coordinate_dim)}"
        )

    log_info(f"  writing {cache_path} ...")
    torch.save(payload, cache_path)
    meta_path.write_text(json.dumps(expected, indent=2))
    elapsed = time.perf_counter() - t0
    log_info(
        f"  ✓ {split} WIRE cache done in {elapsed:.1f}s "
        f"({n_jets / max(elapsed, 1e-6):,.0f} jets/s) "
        f"| shape={tuple(payload['wire_coordinates'].shape)}"
    )
    del star_payload
    return payload


def build_all_wire_coordinate_caches(
    *,
    data_root: str,
    radius: float,
    num_particles: int,
    coordinate_dim: int,
    normalized_laplacian: bool = True,
    standardize: bool = True,
    canonicalize_sign: bool = True,
    n_train: int | None = None,
    n_val: int | None = None,
    n_test: int | None = None,
    seed: int = 42,
    rebuild: bool = False,
    chunk_size: int = 256,
    show_progress: bool = True,
) -> Path:
    del n_train, n_val, n_test  # full parent splits; subsample remains a train-time choice
    out = processed_wire_cache_dir(
        data_root,
        radius=radius,
        num_particles=num_particles,
        coordinate_dim=coordinate_dim,
        normalized_laplacian=normalized_laplacian,
    )
    for split in ("train", "val", "test"):
        build_or_load_wire_coordinate_cache(
            data_root=data_root,
            split=split,
            radius=radius,
            num_particles=num_particles,
            coordinate_dim=coordinate_dim,
            normalized_laplacian=normalized_laplacian,
            standardize=standardize,
            canonicalize_sign=canonicalize_sign,
            seed=seed,
            rebuild=rebuild,
            chunk_size=chunk_size,
            show_progress=show_progress,
        )
    if show_progress:
        log_info(f"All WIRE coordinate caches ready under {out}")
    return out


def open_wire_coordinate_cache(
    *,
    data_root: str,
    split: str,
    radius: float,
    num_particles: int,
    coordinate_dim: int,
    normalized_laplacian: bool = True,
    standardize: bool = True,
    canonicalize_sign: bool = True,
) -> dict[str, torch.Tensor]:
    cache_dir = processed_wire_cache_dir(
        data_root,
        radius=radius,
        num_particles=num_particles,
        coordinate_dim=coordinate_dim,
        normalized_laplacian=normalized_laplacian,
    )
    cache_path = cache_dir / f"{split}.pt"
    meta_path = cache_dir / "meta.json"
    if not cache_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            missing_wire_cache_message(
                data_root=data_root,
                split=split,
                radius=radius,
                num_particles=num_particles,
                coordinate_dim=coordinate_dim,
                normalized_laplacian=normalized_laplacian,
            )
        )
    meta = json.loads(meta_path.read_text())
    expected = wire_cache_meta(
        radius=radius,
        num_particles=num_particles,
        coordinate_dim=coordinate_dim,
        normalized_laplacian=normalized_laplacian,
        standardize=standardize,
        canonicalize_sign=canonicalize_sign,
    )
    if not _meta_matches(meta, expected):
        raise FileNotFoundError(
            f"WIRE cache metadata mismatch at {meta_path}. "
            f"Expected {expected}, found {meta}."
        )
    log_info(f"[wire] split={split} mmap {cache_path}")
    return load_star_graph_cache_mmap(cache_path)
