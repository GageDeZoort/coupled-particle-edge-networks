#!/usr/bin/env python
"""Run stages 0→4 on a mock-less, S5-restored galaxy (default: train/0000).

Same cells → frozen CMD/π cuts → PM GMM → width cut → blob graphs as the mock
training pipeline, but:

- injected mock stars are dropped
- catalogued real/S5 members that training removed are kept
- ``is_mock_stream`` is aliased to S5 membership (graph ``y`` = known stream)
- outputs go to ``galaxies_real/`` so mock graphs are untouched

Each stage skips if its ``.done`` marker exists unless ``--rebuild``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cpen.apps.streams.preprocess.config import DEFAULT_REAL_PIPELINE_ROOT
from cpen.apps.streams.preprocess.pipeline import (
    add_galaxy_args,
    maybe_list_and_exit,
    run_build_graphs,
    run_cells,
    run_gmm,
    run_physical_cuts,
    run_split_blobs,
    spec_from_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_galaxy_args(parser)
    parser.set_defaults(
        out_root=DEFAULT_REAL_PIPELINE_ROOT,
        split="train",
        galaxy_id="0000",
        include_tune_galaxy=True,
    )
    parser.add_argument("--circle-radius", type=float, default=5.0)
    parser.add_argument(
        "--stop-after",
        choices=("cells", "physical", "gmm", "blobs", "graphs"),
        default="graphs",
        help="Run through this stage (inclusive). Default: graphs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if maybe_list_and_exit(args):
        return
    spec = spec_from_args(args)
    stop = args.stop_after
    rebuild = bool(args.rebuild)
    verbose = not bool(args.no_verbose)
    print(f"{spec.tag}  {spec.input_parquet}")
    print(f"  → {spec.out_dir}  composition=real  stop-after={stop}")

    run_cells(
        spec,
        rebuild=rebuild,
        verbose=verbose,
        circle_radius_deg=float(args.circle_radius),
        composition="real",
    )
    if stop == "cells":
        return
    run_physical_cuts(spec, rebuild=rebuild, verbose=verbose)
    if stop == "physical":
        return
    run_gmm(spec, rebuild=rebuild, verbose=verbose)
    if stop == "gmm":
        return
    run_split_blobs(spec, rebuild=rebuild, verbose=verbose)
    if stop == "blobs":
        return
    run_build_graphs(spec, rebuild=rebuild, verbose=verbose)


if __name__ == "__main__":
    main()
