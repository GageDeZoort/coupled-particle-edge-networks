"""Frozen galaxy → cells → physical cuts → GMM → blobs → graphs pipeline.

Cut knobs were chosen on ``train/0000`` and are not refit. Train galaxy 0000
is skipped by default when iterating array tasks.
"""
from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

from cpen.apps.streams.preprocess.blob_graph import (
    BLOB_HYPER_BG_MIN,
    BLOB_HYPER_CONTRAST,
    BLOB_HYPER_K,
    BLOB_HYPER_N,
    BLOB_KNN_K,
    build_and_save_graphs,
)
from cpen.apps.streams.preprocess.cells_io import write_cell_parquets
from cpen.apps.streams.preprocess.config import (
    DEFAULT_GALAXY_ROOT,
    DEFAULT_PIPELINE_ROOT,
    IMPUTE_VARS,
    MIN_MOCK_STREAM_STARS,
    SPLITS,
    TRAIN_VARS,
    TUNE_GALAXY_ID,
    TUNE_SPLIT,
)
from cpen.apps.streams.preprocess.des_footprint_centers import DES_FOOTPRINT_CENTERS
from cpen.apps.streams.preprocess.galaxy import (
    add_is_train_column,
    drop_tiny_mock_streams,
    load_galaxy,
    prepare_mock_cells,
    prepare_real_table,
    tiny_mock_stream_labels,
)
from cpen.apps.streams.preprocess.geometry import assign_cells
from cpen.apps.streams.preprocess.gmm import (
    BLOB_N_MAX,
    BLOB_SPLIT_METHOD,
    GMM_MAX_FIT_SAMPLES,
    GMM_N_COMPONENTS,
    GMM_N_INIT,
    GMM_WIDTH_MAX,
    PM_FLOOR_MAS_YR,
    assign_gmm_to_cells,
    load_blobs,
    load_gmm_cells,
    save_blobs,
    save_gmm_cells,
    split_gmm_blobs,
    split_oversized_blobs,
)
from cpen.apps.streams.preprocess.physical import apply_physical_cuts_to_cells

_GALAXY_RE = re.compile(
    r"^(?P<split>train|val|test)_galaxy_(?P<galaxy_id>\d+)_with_background\.parquet$"
)

CELLS_SUBDIR = "cells"
PHYS_SUBDIR = "cells_phys"
GMM_SUBDIR = f"cells_gmm_k{GMM_N_COMPONENTS}"
BLOBS_SUBDIR = "blobs"
GRAPHS_SUBDIR = "graphs"


@dataclass(frozen=True)
class GalaxySpec:
    split: str
    galaxy_id: str
    input_parquet: Path
    out_dir: Path

    @property
    def tag(self) -> str:
        return f"{self.split}/{self.galaxy_id}"


def parse_galaxy_parquet_name(path) -> tuple[str, str]:
    name = Path(path).name
    m = _GALAXY_RE.match(name)
    if not m:
        raise ValueError(
            f"Expected '{{split}}_galaxy_{{id}}_with_background.parquet'; got {name}"
        )
    return m.group("split"), m.group("galaxy_id")


def galaxy_parquet(galaxy_root, split: str, galaxy_id: str) -> Path:
    root = Path(galaxy_root)
    gid = f"{int(galaxy_id):04d}"
    path = root / split / f"{split}_galaxy_{gid}_with_background.parquet"
    return path


def list_galaxy_ids(
    galaxy_root,
    split: str,
    *,
    skip_tune: bool = True,
) -> list[str]:
    root = Path(galaxy_root) / split
    ids = []
    for fp in sorted(root.glob(f"{split}_galaxy_*_with_background.parquet")):
        _, gid = parse_galaxy_parquet_name(fp)
        if skip_tune and split == TUNE_SPLIT and gid == TUNE_GALAXY_ID:
            continue
        ids.append(gid)
    return ids


def galaxy_out_dir(out_root, split: str, galaxy_id: str) -> Path:
    return Path(out_root) / split / f"{int(galaxy_id):04d}"


def resolve_galaxy(
    *,
    galaxy_root=DEFAULT_GALAXY_ROOT,
    out_root=DEFAULT_PIPELINE_ROOT,
    split: str | None = None,
    galaxy_id: str | None = None,
    array_task: int | None = None,
    input_parquet=None,
    out_dir=None,
    skip_tune: bool = True,
) -> GalaxySpec:
    """Resolve one galaxy from an explicit id, an array index, or an input path."""
    if input_parquet is not None:
        ip = Path(input_parquet)
        sp, gid = parse_galaxy_parquet_name(ip)
        split = split or sp
        galaxy_id = galaxy_id or gid
        input_parquet = ip
    if split is None:
        raise ValueError("Need --split or an input parquet name that encodes the split")
    split = str(split)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}; got {split!r}")
    if galaxy_id is None:
        if array_task is None:
            raise ValueError("Need --galaxy-id, --array-task, or --input")
        ids = list_galaxy_ids(galaxy_root, split, skip_tune=skip_tune)
        if not ids:
            raise FileNotFoundError(f"No galaxies for split={split} under {galaxy_root}")
        if int(array_task) < 0 or int(array_task) >= len(ids):
            raise IndexError(
                f"array task {array_task} out of range for {split} "
                f"({len(ids)} galaxies; ids={ids})"
            )
        galaxy_id = ids[int(array_task)]
    galaxy_id = f"{int(galaxy_id):04d}"
    if input_parquet is None:
        input_parquet = galaxy_parquet(galaxy_root, split, galaxy_id)
    if not Path(input_parquet).is_file():
        raise FileNotFoundError(input_parquet)
    dest = Path(out_dir) if out_dir is not None else galaxy_out_dir(out_root, split, galaxy_id)
    return GalaxySpec(
        split=split,
        galaxy_id=galaxy_id,
        input_parquet=Path(input_parquet),
        out_dir=dest,
    )


def add_galaxy_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--galaxy-root",
        default=DEFAULT_GALAXY_ROOT,
        help="Root with train/ val/ test/ galaxy parquets",
    )
    parser.add_argument(
        "--out-root",
        default=DEFAULT_PIPELINE_ROOT,
        help="Pipeline output root ({out-root}/{split}/{galaxy_id}/)",
    )
    parser.add_argument("--split", choices=SPLITS, default=None)
    parser.add_argument("--galaxy-id", default=None, help="e.g. 0001")
    parser.add_argument(
        "--array-task",
        type=int,
        default=None,
        help="Index into the split's galaxy list (SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Explicit galaxy parquet (overrides galaxy-root/id)",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Explicit galaxy output directory (overrides out-root/split/id)",
    )
    parser.add_argument(
        "--include-tune-galaxy",
        action="store_true",
        help="Include train/0000 in array listings (default: skip; cuts were frozen there)",
    )
    parser.add_argument("--rebuild", action="store_true", help="Overwrite existing stage outputs")
    parser.add_argument("--list", action="store_true", help="Print galaxy ids for --split and exit")
    parser.add_argument("--no-verbose", action="store_true")
    return parser


def spec_from_args(args: argparse.Namespace) -> GalaxySpec:
    return resolve_galaxy(
        galaxy_root=args.galaxy_root,
        out_root=args.out_root,
        split=args.split,
        galaxy_id=args.galaxy_id,
        array_task=args.array_task,
        input_parquet=args.input,
        out_dir=args.outdir,
        skip_tune=not bool(args.include_tune_galaxy),
    )


def maybe_list_and_exit(args: argparse.Namespace) -> bool:
    if not args.list:
        return False
    split = args.split or "train"
    ids = list_galaxy_ids(
        args.galaxy_root, split, skip_tune=not bool(args.include_tune_galaxy)
    )
    print(f"{split}: {len(ids)} galaxies")
    for i, gid in enumerate(ids):
        print(f"  {i:3d}  {gid}")
    print(f"sbatch --array=0-{max(len(ids) - 1, 0)}  SPLIT={split}")
    return True


def write_stage_meta(spec: GalaxySpec, stage: str, payload: dict) -> Path:
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    path = spec.out_dir / "pipeline_meta.yaml"
    meta = {}
    if path.is_file():
        meta = yaml.safe_load(path.read_text()) or {}
    meta["split"] = spec.split
    meta["galaxy_id"] = spec.galaxy_id
    meta["input"] = str(spec.input_parquet)
    meta[stage] = payload
    path.write_text(yaml.safe_dump(meta, sort_keys=False))
    return path


def read_composition(spec: GalaxySpec) -> str:
    """``mock`` (train) or ``real`` (S5 recovery). Default mock if unset."""
    path = spec.out_dir / "pipeline_meta.yaml"
    if not path.is_file():
        return "mock"
    meta = yaml.safe_load(path.read_text()) or {}
    c = meta.get("composition")
    if c:
        return str(c)
    cells = meta.get("cells") or {}
    if cells.get("composition"):
        return str(cells["composition"])
    return "mock"


def _maybe_prepare_mock_cells(spec: GalaxySpec, cells: dict, *, verbose: bool = True) -> dict:
    """Drop real S5 members and demote tiny mocks — skipped for composition=real."""
    if read_composition(spec) == "real":
        if verbose:
            print("composition=real: keep S5 members; skip tiny-mock demotion")
        return cells
    return prepare_mock_cells(cells, min_stream_stars=MIN_MOCK_STREAM_STARS, verbose=verbose)


def _done_path(directory: Path) -> Path:
    return Path(directory) / ".done"


def _stage_done(directory: Path, rebuild: bool) -> bool:
    directory = Path(directory)
    if rebuild and directory.exists():
        shutil.rmtree(directory)
        return False
    return _done_path(directory).is_file()


def _mark_done(directory: Path) -> None:
    Path(directory).mkdir(parents=True, exist_ok=True)
    _done_path(directory).write_text("ok\n")


def load_stage_cells(spec: GalaxySpec, subdir: str) -> dict:
    cells = load_gmm_cells(spec.out_dir, subdir=subdir)
    if not cells:
        raise FileNotFoundError(
            f"Missing {spec.out_dir / subdir} — run the previous pipeline stage first"
        )
    return cells


def run_stage_cli(run_fn, *, description: str) -> None:
    parser = argparse.ArgumentParser(description=description)
    add_galaxy_args(parser)
    args = parser.parse_args()
    if maybe_list_and_exit(args):
        return
    spec = spec_from_args(args)
    print(f"{spec.tag}  {spec.input_parquet}")
    print(f"  → {spec.out_dir}")
    run_fn(spec, rebuild=bool(args.rebuild), verbose=not bool(args.no_verbose))


def run_cells(
    spec: GalaxySpec,
    *,
    rebuild: bool = False,
    verbose: bool = True,
    circle_radius_deg: float = 5.0,
    pm_cut_min: float | None = None,
    keep_real_streams: bool = False,
    composition: str = "mock",
    centers=None,
) -> Path:
    """Stage 0: load galaxy → composition filter → DES cells.

    ``composition='mock'`` (default): drop catalogued S5 members, demote mock
    streams with <100 stars, keep injected mocks as the positive class.

    ``composition='real'``: drop injected mocks, keep catalogued S5 members,
    and alias ``is_mock_stream ← is_real_stream`` so later stages / graph
    labels / plotters treat S5 membership as the positive class. Tiny-mock
    demotion is skipped (real streams can be smaller than 100 stars).
    """
    import numpy as np

    composition = str(composition or "mock").strip().lower()
    if composition not in {"mock", "real"}:
        raise ValueError(f"composition must be 'mock' or 'real'; got {composition!r}")

    out = spec.out_dir / CELLS_SUBDIR
    if _stage_done(out, rebuild):
        if verbose:
            print(f"Skip cells (exists): {out}")
        return out
    df = load_galaxy(spec.input_parquet, feature_cols_to_impute=IMPUTE_VARS)
    tiny: set[str] = set()
    if composition == "real":
        df = prepare_real_table(df, verbose=verbose)
    else:
        df = add_is_train_column(
            df,
            verbose=verbose,
            keep_real_streams=bool(keep_real_streams),
            mock_train_frac=1.0,
        )
        tiny = tiny_mock_stream_labels(df, min_stars=MIN_MOCK_STREAM_STARS)
        df = drop_tiny_mock_streams(df, tiny_labels=tiny, verbose=verbose)
    if pm_cut_min is not None:
        pm_mag = np.hypot(df["pmra_gaia"].to_numpy(), df["pmdec_gaia"].to_numpy())
        keep = pm_mag >= float(pm_cut_min)
        n_before = len(df)
        df = df.loc[keep].copy()
        if verbose:
            print(
                f"PM cut |pm| >= {pm_cut_min} mas/yr: "
                f"{n_before:,} → {len(df):,} ({n_before - len(df):,} removed)"
            )
    if centers is None:
        centers = DES_FOOTPRINT_CENTERS
    df, n_centers = assign_cells(
        df, centers=centers, circle_radius_deg=circle_radius_deg, verbose=verbose,
    )
    keep_cols = (
        list(TRAIN_VARS)
        + ["ra_des"]
        + [
            "stream_label",
            "stream_s5",
            "is_train",
            "is_mock",
            "is_mock_stream",
            "is_real_stream",
            "stream_id",
            "split",
        ]
    )
    keep_cols = [c for c in keep_cols if c in df.columns]
    cell_cols = [f"in_cell_{j + 1}" for j in range(n_centers)]
    df_out = df[keep_cols + cell_cols].copy()
    spec.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = spec.out_dir / "checkpoint.parquet"
    df_out.to_parquet(ckpt, index=False)
    write_cell_parquets(df_out, centers, spec.out_dir, n_centers, verbose=verbose)
    _mark_done(out)
    write_stage_meta(spec, "cells", {
        "subdir": CELLS_SUBDIR,
        "circle_radius_deg": float(circle_radius_deg),
        "n_centers": int(n_centers),
        "n_stars": int(len(df_out)),
        "n_tiny_streams": int(len(tiny)),
        "pm_cut_min": pm_cut_min,
        "keep_real_streams": bool(keep_real_streams) or composition == "real",
        "composition": composition,
        "checkpoint": str(ckpt),
    })
    # Top-level so stages 1–4 can skip mock-only filters without extra flags.
    meta_path = spec.out_dir / "pipeline_meta.yaml"
    meta = yaml.safe_load(meta_path.read_text()) or {}
    meta["composition"] = composition
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False))
    if verbose:
        print(f"Wrote checkpoint → {ckpt}")
        print(f"Wrote cells      → {out}")
        print(f"composition      → {composition}")
    return out


def run_physical_cuts(spec: GalaxySpec, *, rebuild: bool = False, verbose: bool = True) -> Path:
    out = spec.out_dir / PHYS_SUBDIR
    if _stage_done(out, rebuild):
        if verbose:
            print(f"Skip physical cuts (exists): {out}")
        return out
    cells = load_stage_cells(spec, CELLS_SUBDIR)
    cells = _maybe_prepare_mock_cells(spec, cells, verbose=verbose)
    pos = "S5" if read_composition(spec) == "real" else "mocks"
    phys = apply_physical_cuts_to_cells(cells, verbose=verbose, positive_name=pos)
    save_gmm_cells(phys, spec.out_dir, subdir=PHYS_SUBDIR)
    _mark_done(out)
    write_stage_meta(spec, "physical", {
        "subdir": PHYS_SUBDIR,
        "n_cells": len(phys),
        "n_stars": int(sum(len(v) for v in phys.values())),
    })
    if verbose:
        print(f"Wrote physical cells → {out}")
    return out


def run_gmm(
    spec: GalaxySpec,
    *,
    rebuild: bool = False,
    verbose: bool = True,
    n_components: int = GMM_N_COMPONENTS,
    n_init: int = GMM_N_INIT,
    max_fit_samples: int = GMM_MAX_FIT_SAMPLES,
) -> Path:
    subdir = f"cells_gmm_k{int(n_components)}"
    out = spec.out_dir / subdir
    if _stage_done(out, rebuild):
        if verbose:
            print(f"Skip GMM (exists): {out}")
        return out
    cells = load_stage_cells(spec, PHYS_SUBDIR)
    cells = _maybe_prepare_mock_cells(spec, cells, verbose=verbose)
    assigned, _fits, occ = assign_gmm_to_cells(
        cells,
        n_components=n_components,
        verbose=verbose,
        n_init=n_init,
        max_fit_samples=max_fit_samples,
        pm_floor=PM_FLOOR_MAS_YR,
        random_state=0,
    )
    save_gmm_cells(assigned, spec.out_dir, subdir=subdir)
    _mark_done(out)
    n_mock = int(occ.n_mock.sum()) if occ is not None and len(occ) and "n_mock" in occ.columns else 0
    write_stage_meta(spec, "gmm", {
        "subdir": subdir,
        "n_components": int(n_components),
        "n_init": int(n_init),
        "max_fit_samples": int(max_fit_samples),
        "pm_floor": float(PM_FLOOR_MAS_YR),
        "n_cells": len(assigned),
        "n_mock": n_mock,
    })
    if verbose:
        print(f"Wrote GMM cells → {out}")
    return out


def run_split_blobs(
    spec: GalaxySpec,
    *,
    rebuild: bool = False,
    verbose: bool = True,
    width_max: float = GMM_WIDTH_MAX,
    n_max: int = BLOB_N_MAX,
    method: str = BLOB_SPLIT_METHOD,
) -> Path:
    out = spec.out_dir / BLOBS_SUBDIR
    if _stage_done(out, rebuild):
        if verbose:
            print(f"Skip blob split (exists): {out}")
        return out
    cells = load_stage_cells(spec, GMM_SUBDIR)
    cells = _maybe_prepare_mock_cells(spec, cells, verbose=verbose)
    blobs = split_gmm_blobs(cells, width_max=width_max)
    blobs = split_oversized_blobs(blobs, n_max=n_max, method=method, verbose=verbose)
    save_blobs(blobs, spec.out_dir, subdir=BLOBS_SUBDIR)
    _mark_done(out)
    ns = [len(v) for v in blobs.values()]
    write_stage_meta(spec, "blobs", {
        "subdir": BLOBS_SUBDIR,
        "width_max": float(width_max),
        "n_max": int(n_max),
        "method": method,
        "n_blobs": len(blobs),
        "max_n": int(max(ns) if ns else 0),
        "n_stars": int(sum(ns)),
    })
    if verbose:
        print(f"Wrote {len(blobs):,} blobs → {out}")
    return out


def run_build_graphs(
    spec: GalaxySpec,
    *,
    rebuild: bool = False,
    verbose: bool = True,
) -> Path:
    out = spec.out_dir / GRAPHS_SUBDIR
    if _stage_done(out, rebuild):
        if verbose:
            print(f"Skip graphs (exists): {out}")
        return out
    blobs = load_blobs(spec.out_dir, subdir=BLOBS_SUBDIR)
    if not blobs:
        raise FileNotFoundError(
            f"Missing {spec.out_dir / BLOBS_SUBDIR} — run stage 3 first"
        )
    tab = build_and_save_graphs(
        blobs,
        spec.out_dir,
        subdir=GRAPHS_SUBDIR,
        k=BLOB_KNN_K,
        n_hyper=BLOB_HYPER_N,
        k_ball=BLOB_HYPER_K,
        contrast=BLOB_HYPER_CONTRAST,
        bg_min=BLOB_HYPER_BG_MIN,
        verbose=verbose,
    )
    _mark_done(out)
    write_stage_meta(spec, "graphs", {
        "subdir": GRAPHS_SUBDIR,
        "n_graphs": int(len(tab)),
        "knn_k": int(BLOB_KNN_K),
        "hyper_n": int(BLOB_HYPER_N),
        "hyper_k": int(BLOB_HYPER_K),
        "hyper_contrast": float(BLOB_HYPER_CONTRAST),
    })
    return out
