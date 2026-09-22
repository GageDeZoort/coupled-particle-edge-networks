"""Cell-level train / discovery splits for real-stream blob fine-tuning.

A cell is a **train cell** if it contains enough members of any *train* stream.
All other cells (holdout-only, empty, or background-only) are **test / discovery**.
That yields a preponderance of test cells — the intended recovery setup.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from cpen.apps.streams.preprocess.config import TEST_REAL_STREAMS

CELLS_SUBDIR = "cells"
BACKGROUND = "Background"


def parse_stream_name_list(raw: str | Sequence[str] | None) -> set[str] | None:
    """Comma/space-separated stream names, or ``None`` if unset / ``auto``."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple, set)):
        names = {str(x).strip() for x in raw if str(x).strip()}
        return names or None
    text = str(raw).strip()
    if not text or text.lower() in {"auto", "default", "none"}:
        return None
    return {p for p in re.split(r"[,\s]+", text) if p}


@dataclass(frozen=True)
class CellStreamOccupancy:
    """Per-cell stream member counts for one galaxy."""

    galaxy_id: str
    # cell_id -> {stream_label: n_members}
    cell_streams: dict[int, dict[str, int]]
    # stream_label -> total members across cells (overlapping cells double-count)
    stream_counts: dict[str, int]

    @property
    def stream_names(self) -> set[str]:
        return set(self.stream_counts)

    def cells_with_streams(self, names: Iterable[str], *, min_stars: int = 1) -> set[int]:
        wanted = {str(n) for n in names}
        out: set[int] = set()
        for cid, counts in self.cell_streams.items():
            n = sum(int(counts.get(s, 0)) for s in wanted)
            if n >= int(min_stars):
                out.add(int(cid))
        return out


def scan_cell_stream_occupancy(
    galaxy_dir: str | Path,
    *,
    galaxy_id: str | None = None,
) -> CellStreamOccupancy:
    """Read ``cells/cell_*.parquet`` and tally S5 / positive-stream members."""
    gdir = Path(galaxy_dir)
    gid = str(galaxy_id or gdir.name)
    cells_dir = gdir / CELLS_SUBDIR
    if not cells_dir.is_dir():
        raise FileNotFoundError(f"Missing cells dir {cells_dir}")

    cell_streams: dict[int, dict[str, int]] = {}
    stream_counts: dict[str, int] = defaultdict(int)
    for path in sorted(cells_dir.glob("cell_*.parquet")):
        cid = int(path.stem.split("_")[1])
        # stream_label is always present; positive flags depend on composition.
        try:
            df = pd.read_parquet(
                path, columns=["stream_label", "is_mock_stream", "is_real_stream"]
            )
        except (ValueError, KeyError, OSError):
            try:
                df = pd.read_parquet(path, columns=["stream_label", "is_mock_stream"])
            except (ValueError, KeyError, OSError):
                df = pd.read_parquet(path, columns=["stream_label"])
        if "is_mock_stream" in df.columns:
            pos = df["is_mock_stream"].to_numpy(bool)
        elif "is_real_stream" in df.columns:
            pos = df["is_real_stream"].to_numpy(bool)
        else:
            pos = df["stream_label"].astype(str).ne(BACKGROUND).to_numpy()
        labels = df.loc[pos, "stream_label"].astype(str)
        vc = {k: int(v) for k, v in labels.value_counts().items() if k != BACKGROUND}
        cell_streams[cid] = vc
        for name, n in vc.items():
            stream_counts[name] += int(n)
    return CellStreamOccupancy(
        galaxy_id=gid,
        cell_streams=cell_streams,
        stream_counts=dict(stream_counts),
    )


def resolve_train_holdout_streams(
    occupancy: CellStreamOccupancy,
    *,
    train_streams: set[str] | None = None,
    holdout_streams: set[str] | None = None,
) -> tuple[set[str], set[str]]:
    """Resolve train vs holdout stream names.

    Defaults: holdout = ``TEST_REAL_STREAMS`` ∩ present; train = all other
    present streams. Explicit ``train_streams`` wins; holdout becomes the rest
    (or the intersection with ``holdout_streams`` when both are set).
    """
    present = set(occupancy.stream_names)
    if not present:
        raise ValueError(f"No stream members found in galaxy {occupancy.galaxy_id}")

    if train_streams is not None:
        train = {s for s in train_streams if s in present}
        if not train:
            raise ValueError(
                f"None of the requested train streams appear in galaxy "
                f"{occupancy.galaxy_id}: {sorted(train_streams)}; "
                f"present={sorted(present)}"
            )
        if holdout_streams is not None:
            holdout = {s for s in holdout_streams if s in present and s not in train}
        else:
            holdout = present - train
        return train, holdout

    holdout_default = (
        set(holdout_streams) if holdout_streams is not None else set(TEST_REAL_STREAMS)
    )
    holdout = holdout_default & present
    train = present - holdout
    if not train:
        raise ValueError(
            f"Holdout streams leave no train streams in galaxy {occupancy.galaxy_id}: "
            f"holdout={sorted(holdout)} present={sorted(present)}"
        )
    return train, holdout


@dataclass(frozen=True)
class CellSplit:
    train_streams: frozenset[str]
    holdout_streams: frozenset[str]
    train_cells: frozenset[int]
    val_cells: frozenset[int]
    test_cells: frozenset[int]
    occupancy: CellStreamOccupancy

    @property
    def supervised_cells(self) -> frozenset[int]:
        return self.train_cells | self.val_cells


def assign_cells_by_train_streams(
    occupancy: CellStreamOccupancy,
    *,
    train_streams: set[str] | None = None,
    holdout_streams: set[str] | None = None,
    min_train_stream_stars: int = 1,
    val_frac: float = 0.15,
    seed: int = 0,
) -> CellSplit:
    """Train cells = cells with train-stream members; all others → test.

    A fraction ``val_frac`` of train cells is carved out for validation (early
    stopping) so discovery test cells stay unlabeled / unused for model selection.
    """
    import numpy as np

    train, holdout = resolve_train_holdout_streams(
        occupancy,
        train_streams=train_streams,
        holdout_streams=holdout_streams,
    )
    train_cells = occupancy.cells_with_streams(train, min_stars=min_train_stream_stars)
    all_cells = set(occupancy.cell_streams)
    test_cells = all_cells - train_cells
    if not train_cells:
        raise ValueError(
            f"No train cells with ≥{min_train_stream_stars} members of "
            f"{sorted(train)} in galaxy {occupancy.galaxy_id}"
        )

    val_frac = float(val_frac)
    if val_frac < 0.0 or val_frac >= 1.0:
        raise ValueError(f"val_frac must be in [0, 1); got {val_frac}")
    train_list = sorted(train_cells)
    rng = np.random.default_rng(int(seed))
    n_val = int(round(val_frac * len(train_list)))
    if val_frac > 0.0 and len(train_list) >= 2:
        n_val = max(1, min(n_val, len(train_list) - 1))
    else:
        n_val = 0
    if n_val:
        val_idx = set(rng.choice(len(train_list), n_val, replace=False).tolist())
        val_cells = {train_list[i] for i in val_idx}
        train_cells = set(train_list) - val_cells
    else:
        val_cells = set()

    return CellSplit(
        train_streams=frozenset(train),
        holdout_streams=frozenset(holdout),
        train_cells=frozenset(train_cells),
        val_cells=frozenset(val_cells),
        test_cells=frozenset(test_cells),
        occupancy=occupancy,
    )


def partition_manifest_by_cells(
    man: pd.DataFrame,
    split: CellSplit,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Slice a graph manifest into train / val / test by ``cell_id``."""
    if "cell_id" not in man.columns:
        raise KeyError("manifest needs cell_id for cell-stream splits")
    cell = man["cell_id"].astype(int)
    train = man[cell.isin(split.train_cells)].copy()
    val = man[cell.isin(split.val_cells)].copy()
    test = man[cell.isin(split.test_cells)].copy()
    return train, val, test


def describe_cell_split(split: CellSplit) -> str:
    occ = split.occupancy
    return (
        f"galaxy={occ.galaxy_id} | "
        f"train_streams({len(split.train_streams)}): "
        f"{', '.join(sorted(split.train_streams))} | "
        f"holdout({len(split.holdout_streams)}): "
        f"{', '.join(sorted(split.holdout_streams)) or '(none)'} | "
        f"cells train/val/test="
        f"{len(split.train_cells)}/{len(split.val_cells)}/{len(split.test_cells)} "
        f"(of {len(occ.cell_streams)})"
    )
