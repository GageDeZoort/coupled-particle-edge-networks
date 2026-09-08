import numpy as np
import pandas as pd
from pathlib import Path

from cpen.apps.streams.preprocess.config import BACKGROUND, IMPUTE_VARS


def blob_key_ids(key) -> tuple[int, int, int]:
    """``(cell_id, gmm_label, blob_part)``. ``blob_part`` is 0 when the key is a 2-tuple."""
    cid = int(key[0])
    j = int(key[1])
    part = int(key[2]) if len(key) > 2 else 0
    return cid, j, part


def blob_stem(key) -> str:
    cid, j, part = blob_key_ids(key)
    return f"blob_c{cid:04d}_g{j:02d}_p{part:02d}"


def parse_blob_stem(stem: str) -> tuple[int, int, int]:
    """Parse ``blob_cXXXX_gYY_pZZ`` → ``(cell_id, gmm_label, blob_part)``."""
    name = Path(stem).stem
    parts = name.split("_")
    if len(parts) < 4 or not parts[0] == "blob":
        raise ValueError(f"not a blob stem: {stem!r}")
    cid = int(parts[1][1:])
    gmm = int(parts[2][1:])
    part = int(parts[3][1:])
    return cid, gmm, part


def hdr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def sub(title):
    print(f"\n-- {title}")


def print_galaxy_stats(df, max_show=10):
    hdr("Galaxy Dataset Overview")
    print(f"Total rows        : {len(df):,}")
    print(f"Total columns     : {df.shape[1]}")
    print(f"Columns           : {list(df.columns)}")

    # slice on the mock stars, print stats
    sub("Mock stars (is_mock = True)")
    mock = df[df.is_mock == True]

    print(f"Rows              : {len(mock):,}")
    print(f"Unique stream_id  : {mock.stream_id.nunique()}")

    sid_sample = np.sort(mock.stream_id.unique())[:max_show]
    print(
        f"stream_id sample  : {sid_sample}"
        + (" ..." if mock.stream_id.nunique() > max_show else "")
    )

    if "Stream" in mock.columns:
        print(
            f"Stream labels     : {mock.Stream.unique()[:max_show]}"
            + (" ..." if mock.Stream.nunique() > max_show else "")
        )

    if "stream_s5" in mock.columns:
        print(
            f"stream_s5 labels  : {mock.stream_s5.unique()[:max_show]}"
            + (" ..." if mock.stream_s5.nunique() > max_show else "")
        )

    # slice on the real stars, print stats
    sub("Real stars (is_mock = False)")
    real = df[df.is_mock == False]

    print(f"Rows              : {len(real):,}")
    print(f"Unique stream_id  : {real.stream_id.nunique()}")

    if "Stream" in real.columns:
        print(
            f"Stream labels     : {real.Stream.unique()[:max_show]}"
            + (" ..." if real.Stream.nunique() > max_show else "")
        )

    if "stream_s5" in real.columns:
        print(
            f"stream_s5 labels  : {real.stream_s5.unique()[:max_show]}"
            + (" ..." if real.stream_s5.nunique() > max_show else "")
        )

    # sanity checks
    sub("Sanity checks")
    print(f"Remaining NaNs    : {df[IMPUTE_VARS].isna().sum().sum():,}")
    print(
        "Background rows   : "
        f"{(df.get('stream_s5') == BACKGROUND).sum() if 'stream_s5' in df else 'N/A'}"
    )


def print_stream_train_summary(df):
    """Per-``stream_label`` star counts and mock-stream fraction.

    Prefer ``is_mock_stream`` when present; fall back to legacy ``is_train``.
    """
    frac_col = "is_mock_stream" if "is_mock_stream" in df.columns else "is_train"
    g = (
        df.groupby("stream_label")
          .agg(
              n_stars=("stream_label", "size"),
              mock_frac=(frac_col, "mean"),
          )
          .sort_values("n_stars", ascending=False)
    )

    print("\n=== Stream / mock-stream Summary ===")
    for s, row in g.iterrows():
        print(
            f"{s:20s} | N = {int(row.n_stars):7d} | mock_frac = {row.mock_frac:7.3f}"
        )


def _in_cell_columns(df):
    cols = [c for c in df.columns if str(c).startswith("in_cell_")]

    def _cell_key(name):
        try:
            return int(str(name).split("_")[-1])
        except ValueError:
            return str(name)

    return sorted(cols, key=_cell_key)


def print_mock_stream_cell_occupancy(df, top_k=8, min_frac=0.01):
    """Print how each mock stream is spread across DES footprint cells.

    For every ``is_mock_stream`` label, reports:
      - total star count
      - number of cells containing ≥1 member
      - per-cell fractions (stars in cell / stars in stream)

    Cell circles overlap, so fractions can sum to > 1.
    """
    if "is_mock_stream" not in df.columns:
        raise KeyError("df must have is_mock_stream")
    if "stream_label" not in df.columns:
        raise KeyError("df must have stream_label")

    cell_cols = _in_cell_columns(df)
    if not cell_cols:
        raise KeyError("df has no in_cell_* columns")

    mock = df.loc[df["is_mock_stream"].to_numpy(bool)].copy()
    if len(mock) == 0:
        print("\n=== Mock-stream cell occupancy ===\n(no mock-stream stars)")
        return None

    rows = []
    print("\n=== Mock-stream cell occupancy ===")
    print(
        f"(n_mock_streams={mock['stream_label'].nunique()}, "
        f"n_mock_stars={len(mock):,}, n_cell_cols={len(cell_cols)}; "
        "fractions may sum >1 because cells overlap)\n"
    )

    for label, part in mock.groupby("stream_label", sort=False):
        n_stars = len(part)
        fracs = []
        for col in cell_cols:
            n_in = int(part[col].to_numpy(bool).sum())
            if n_in == 0:
                continue
            cid = int(str(col).split("_")[-1])
            frac = n_in / n_stars
            fracs.append((cid, n_in, frac))

        fracs.sort(key=lambda t: (-t[2], t[0]))
        n_cells = len(fracs)
        max_frac = fracs[0][2] if fracs else 0.0
        sum_frac = float(sum(f for _, _, f in fracs))
        n_unassigned = int((~part[cell_cols].any(axis=1)).sum()) if cell_cols else n_stars

        shown = [(cid, n_in, frac) for cid, n_in, frac in fracs if frac >= min_frac][:top_k]
        frac_str = ", ".join(
            f"c{cid}:{frac:.1%}({n_in})" for cid, n_in, frac in shown
        )
        if len(fracs) > len(shown):
            frac_str += f", …(+{len(fracs) - len(shown)} more)"

        print(
            f"{str(label):20s} | N={n_stars:5d} | cells={n_cells:3d} | "
            f"max={max_frac:5.1%} | Σfrac={sum_frac:5.2f} | "
            f"outside={n_unassigned:4d}"
        )
        if frac_str:
            print(f"{'':20s}   fractions: {frac_str}")

        rows.append(
            {
                "stream_label": label,
                "n_stars": n_stars,
                "n_cells": n_cells,
                "max_cell_frac": max_frac,
                "sum_cell_frac": sum_frac,
                "n_outside_cells": n_unassigned,
                "cell_fractions": {cid: frac for cid, _, frac in fracs},
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["n_cells", "n_stars"], ascending=[False, False]
    ).reset_index(drop=True)

    print("\n-- sparsity snapshot (sorted by n_cells desc) --")
    display_cols = [
        "stream_label", "n_stars", "n_cells", "max_cell_frac", "sum_cell_frac", "n_outside_cells"
    ]
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(summary[display_cols].to_string(index=False))

    return summary
