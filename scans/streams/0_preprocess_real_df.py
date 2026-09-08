#!/usr/bin/env python
"""Stage 0 (real-data track): galaxy parquet → DES 5° cells.

Load DES+mock galaxy ``train/0000``, **drop injected mocks**, **keep catalogued
S5 members**, alias ``is_mock_stream ← is_real_stream`` so later stages and
graph labels treat known streams as the positive class, then assign overlapping
5° cells.

Writes to ``galaxies_real/`` (not the mock training tree). Frozen CMD / π / GMM
knobs are still those fit on mock 0000 — this stage does not refit them.

Example
-------
::

    python scans/streams/0_preprocess_real_df.py
    python scans/streams/0_preprocess_real_df.py --split train --galaxy-id 0000 --rebuild
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
    run_cells,
    spec_from_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Real-data cells: drop mocks, keep S5 members, write cells under "
            "galaxies_real/."
        )
    )
    add_galaxy_args(parser)
    parser.set_defaults(
        out_root=DEFAULT_REAL_PIPELINE_ROOT,
        split="train",
        galaxy_id="0000",
        include_tune_galaxy=True,
    )
    parser.add_argument("--circle-radius", type=float, default=5.0)
    parser.add_argument(
        "--pm-cut-min",
        type=float,
        default=None,
        metavar="MAS_YR",
        help="Cut stars with |pm| < this (mas/yr). Default: no cut.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if maybe_list_and_exit(args):
        return
    spec = spec_from_args(args)
    print(f"{spec.tag}  {spec.input_parquet}")
    print(f"  → {spec.out_dir}  (composition=real)")
    run_cells(
        spec,
        rebuild=bool(args.rebuild),
        verbose=not bool(args.no_verbose),
        circle_radius_deg=float(args.circle_radius),
        pm_cut_min=args.pm_cut_min,
        composition="real",
    )


if __name__ == "__main__":
    main()
