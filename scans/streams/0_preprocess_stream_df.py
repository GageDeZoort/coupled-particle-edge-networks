#!/usr/bin/env python
"""Stage 0: galaxy parquet → DES 5° cells.

Load DES+mock galaxy, drop real-stream members, demote mock streams with
<100 stars to field, assign overlapping 5° cells, write per-cell parquets.

Example
-------
::

    python scans/streams/0_preprocess_stream_df.py --split train --galaxy-id 0001
    python scans/streams/0_preprocess_stream_df.py --split train --array-task 0
    python scans/streams/0_preprocess_stream_df.py --split train --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cpen.apps.streams.preprocess.pipeline import (
    add_galaxy_args,
    maybe_list_and_exit,
    run_cells,
    spec_from_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign stars to DES footprint cells; write checkpoint + cells/."
    )
    add_galaxy_args(parser)
    parser.add_argument("--circle-radius", type=float, default=5.0)
    parser.add_argument(
        "--pm-cut-min",
        type=float,
        default=None,
        metavar="MAS_YR",
        help="Cut stars with |pm| < this (mas/yr). Default: no cut.",
    )
    parser.add_argument(
        "--keep-real-streams",
        action="store_true",
        help="Keep catalogued real (S5) stream members (default: drop).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if maybe_list_and_exit(args):
        return
    spec = spec_from_args(args)
    print(f"{spec.tag}  {spec.input_parquet}")
    print(f"  → {spec.out_dir}")
    run_cells(
        spec,
        rebuild=bool(args.rebuild),
        verbose=not bool(args.no_verbose),
        circle_radius_deg=float(args.circle_radius),
        pm_cut_min=args.pm_cut_min,
        keep_real_streams=bool(args.keep_real_streams),
    )


if __name__ == "__main__":
    main()
