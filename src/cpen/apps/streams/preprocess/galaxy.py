import numpy as np
import pandas as pd

from cpen.apps.streams.preprocess.config import (
    BACKGROUND,
    IMPUTE_VARS,
    MERGE_TO_AAU,
    MIN_MOCK_STREAM_STARS,
    PSF_G_UPPER,
    PSF_R_UPPER,
    TEST_REAL_STREAMS,
)


def load_galaxy(indir, feature_cols_to_impute=None):
    df = pd.read_parquet(indir)

    # impute NaNs
    if feature_cols_to_impute is None:
        feature_cols_to_impute = IMPUTE_VARS
    for c in feature_cols_to_impute:
        print(f"Imputing NaNs with 0      : {c}")
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].fillna(0.0)

    # drop mock stars with unrealistic photometry
    print("\nDropping unrealistic photometry")
    print(f"Initial mock count       : {len(df[df.is_mock==True])}")
    print(f"Initial DES  count       : {len(df[df.is_mock==False])}")
    df = df[
        (df.psf_mag_aper_8_r_corrected_des < PSF_R_UPPER) &
        (df.psf_mag_aper_8_g_corrected_des < PSF_G_UPPER)
    ]
    print(f"Final mock count       : {len(df[df.is_mock==True])}")
    print(f"Final DES  count       : {len(df[df.is_mock==False])}")
    return df


def ensure_is_mock_stream(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``is_mock_stream`` exists (derive from ``is_mock``/``stream_id`` or label)."""
    if "is_mock_stream" in df.columns:
        return df
    out = df.copy()
    if "is_mock" in out.columns and "stream_id" in out.columns:
        sid = pd.to_numeric(out["stream_id"], errors="coerce").fillna(-999).astype("int64")
        out["is_mock_stream"] = out["is_mock"].astype(bool) & (sid.to_numpy() > 0)
    elif "stream_label" in out.columns:
        out["is_mock_stream"] = out["stream_label"].astype(str).str.startswith("MOCK_")
    else:
        raise ValueError(
            "Cannot derive is_mock_stream: need is_mock+stream_id or stream_label"
        )
    return out


def drop_real_stream_members(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Keep DES background + mocks; remove catalogued real-stream members.

    Training on mocks against a DES sky that still contains labelled real streams
    lets the network treat those streams as ignorable background. For mock-only
    supervision we therefore drop ``is_real_stream`` rows (or the equivalent
    ``~is_mock & stream_label != Background`` when the flag is absent).
    """
    out = df
    if "is_real_stream" in out.columns:
        real = out["is_real_stream"].to_numpy(bool)
    elif "is_mock" in out.columns and "stream_label" in out.columns:
        real = (~out["is_mock"].to_numpy(bool)) & (
            out["stream_label"].astype(str) != BACKGROUND
        )
    elif "is_mock" in out.columns and "stream_s5" in out.columns:
        s5 = out["stream_s5"].astype("string").str.strip().fillna(BACKGROUND)
        s5 = s5.replace({"-999": BACKGROUND})
        s5 = s5.where(~s5.isin(MERGE_TO_AAU), "AAU")
        real = (~out["is_mock"].to_numpy(bool)) & (s5.to_numpy() != BACKGROUND)
    else:
        raise ValueError(
            "drop_real_stream_members needs is_real_stream, or is_mock+stream_label/stream_s5"
        )
    n_drop = int(real.sum())
    out = out.loc[~real].copy()
    if "is_real_stream" in out.columns:
        out["is_real_stream"] = False
    if verbose:
        print(f"Dropped real-stream members: {n_drop:,}  →  N={len(out):,}")
    return out


def drop_mock_members(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Remove injected mock stars; keep DES background + catalogued real streams."""
    if "is_mock" not in df.columns:
        raise ValueError("drop_mock_members needs is_mock")
    mock = df["is_mock"].to_numpy(bool)
    n_drop = int(mock.sum())
    out = df.loc[~mock].copy()
    if "is_mock_stream" in out.columns:
        out["is_mock_stream"] = False
    if verbose:
        print(f"Dropped mock members: {n_drop:,}  →  N={len(out):,}")
    return out


def prepare_real_table(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """DES background + S5 real streams; mocks removed.

    Downstream graph code keys off ``is_mock_stream`` as the positive class, so
    after this call that column is aliased to ``is_real_stream`` (catalogued
    S5 members). Training never sees these tables — they are the recovery set.
    """
    out = add_is_train_column(df, keep_real_streams=True, mock_train_frac=1.0, verbose=False)
    out = drop_mock_members(out, verbose=verbose)
    out["is_mock_stream"] = out["is_real_stream"].to_numpy(bool)
    out["is_train"] = out["is_real_stream"].to_numpy(bool)
    if verbose:
        n_real = int(out["is_real_stream"].sum())
        print(f"Real-data table: N={len(out):,}  S5 members={n_real:,}")
        counts = out.loc[out["is_real_stream"], "stream_label"].astype(str).value_counts()
        for name, n in counts.items():
            print(f"  {name:16s} {n:,}")
    return out


def stream_retention_table(
    df: pd.DataFrame,
    keep: np.ndarray,
    *,
    label_col: str = "stream_label",
    target_col: str = "is_mock_stream",
) -> pd.DataFrame:
    """Per-stream keep counts for a boolean mask (physical cuts, width cut, …)."""
    labels = df[label_col].astype(str).to_numpy()
    target = (
        df[target_col].to_numpy(bool)
        if target_col in df.columns
        else np.zeros(len(df), dtype=bool)
    )
    keep = np.asarray(keep, dtype=bool)
    rows = []
    for name in pd.Index(labels).value_counts().index:
        m = labels == str(name)
        n0 = int(m.sum())
        n1 = int((m & keep).sum())
        rows.append({
            "stream_label": str(name),
            "is_target": bool(target[m].any()) and str(name) != BACKGROUND,
            "n_before": n0,
            "n_after": n1,
            "keep_frac": float(n1 / n0) if n0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values(
        ["is_target", "n_before"], ascending=[False, False]
    ).reset_index(drop=True)


def stream_cutflow_table(
    df: pd.DataFrame,
    stages: list[tuple[str, np.ndarray]],
    *,
    label_col: str = "stream_label",
    target_col: str = "is_mock_stream",
    include_background: bool = False,
) -> pd.DataFrame:
    """Sequential keep counts per ``stream_label``.

    Each ``stages`` entry is ``(name, keep_mask)`` with ``keep_mask`` aligned
    to ``df``. Masks are AND-ed in order (pipeline cutflow, not independent).
    """
    labels = df[label_col].astype(str).to_numpy()
    target = (
        df[target_col].to_numpy(bool)
        if target_col in df.columns
        else np.ones(len(df), dtype=bool)
    )
    names = [str(n) for n in pd.Index(labels[target]).value_counts().index]
    if not include_background:
        names = [n for n in names if n != BACKGROUND]
    if include_background and BACKGROUND not in names:
        names.append(BACKGROUND)

    running = np.ones(len(df), dtype=bool)
    rows = []
    for name in names:
        m = labels == name
        row = {"stream_label": name, "n_input": int(m.sum())}
        live = running.copy()
        for stage_name, keep in stages:
            live = live & np.asarray(keep, dtype=bool)
            n = int((m & live).sum())
            row[stage_name] = n
            row[f"{stage_name}_frac"] = float(n / m.sum()) if m.sum() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def mock_stream_star_counts(
    df: pd.DataFrame,
    ra_col: str = "ra_des",
    dec_col: str = "dec_des",
    round_deg: int = 6,
) -> pd.Series:
    """Galaxy-wide unique star counts per ``MOCK_*`` label.

    If ``ra_col``/``dec_col`` exist, collapse overlapping cell copies of the
    same star by rounded sky position. Otherwise a raw ``value_counts``.
    """
    df = ensure_is_mock_stream(df)
    mock = df["is_mock_stream"].to_numpy(bool)
    if not mock.any():
        return pd.Series(dtype=int)
    if "stream_label" not in df.columns:
        raise ValueError("mock_stream_star_counts needs stream_label")
    labels = df.loc[mock, "stream_label"].astype(str)
    if ra_col in df.columns and dec_col in df.columns:
        ra = np.round(df.loc[mock, ra_col].to_numpy(float), int(round_deg))
        dec = np.round(df.loc[mock, dec_col].to_numpy(float), int(round_deg))
        uniq = pd.DataFrame({"lab": labels.to_numpy(), "ra": ra, "dec": dec}).drop_duplicates()
        return uniq["lab"].value_counts()
    return labels.value_counts()


def tiny_mock_stream_labels(
    df: pd.DataFrame,
    min_stars: int = MIN_MOCK_STREAM_STARS,
) -> set[str]:
    """Labels of mock streams with fewer than ``min_stars`` unique members."""
    counts = mock_stream_star_counts(df)
    if not len(counts):
        return set()
    return set(counts[counts < int(min_stars)].index.astype(str))


def tiny_mock_stream_labels_from_frames(
    frames,
    min_stars: int = MIN_MOCK_STREAM_STARS,
) -> set[str]:
    """Same as ``tiny_mock_stream_labels``, pooling several cell tables."""
    parts = []
    for df in frames:
        d = ensure_is_mock_stream(df)
        mock = d["is_mock_stream"].to_numpy(bool)
        if not mock.any() or "stream_label" not in d.columns:
            continue
        cols = [c for c in ("stream_label", "ra_des", "dec_des") if c in d.columns]
        parts.append(d.loc[mock, cols])
    if not parts:
        return set()
    cat = pd.concat(parts, ignore_index=True)
    cat["is_mock_stream"] = True
    return tiny_mock_stream_labels(cat, min_stars=min_stars)


def drop_tiny_mock_streams(
    df: pd.DataFrame,
    min_stars: int = MIN_MOCK_STREAM_STARS,
    tiny_labels: set[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Reclassify mock streams with ``< min_stars`` members as field.

    Stars stay in the table (no holes); ``is_mock_stream`` becomes False and
    ``stream_label`` becomes ``Background``. If ``stream_id`` is present it is
    set to ``-999`` so a later ``ensure_is_mock_stream`` rebuild stays consistent.
    Pass ``tiny_labels`` (from the galaxy table) when applying to overlapping
    cells, so occupancy is not counted per pointing.
    """
    out = ensure_is_mock_stream(df)
    if "stream_label" not in out.columns:
        raise ValueError("drop_tiny_mock_streams needs stream_label")
    counts = None
    if tiny_labels is None:
        counts = mock_stream_star_counts(out)
        tiny_labels = set(counts[counts < int(min_stars)].index.astype(str)) if len(counts) else set()
    else:
        tiny_labels = {str(s) for s in tiny_labels}
    if not tiny_labels:
        if verbose:
            print(f"No mock streams with <{int(min_stars)} stars")
        return out
    mock = out["is_mock_stream"].to_numpy(bool)
    labels = out["stream_label"].astype(str)
    demote = mock & labels.isin(tiny_labels)
    n_stars = int(demote.sum())
    if n_stars == 0:
        if verbose:
            print(
                f"No rows to demote for {len(tiny_labels)} tiny mock streams "
                f"(<{int(min_stars)} stars)"
            )
        return out
    out = out.copy()
    out.loc[demote, "is_mock_stream"] = False
    out.loc[demote, "stream_label"] = BACKGROUND
    if "stream_id" in out.columns:
        out.loc[demote, "stream_id"] = -999
    if verbose:
        print(
            f"Demoted {len(tiny_labels)} mock streams with <{int(min_stars)} stars "
            f"({n_stars:,} stars → field)"
        )
        names = sorted(tiny_labels)
        if counts is not None:
            names = sorted(tiny_labels, key=lambda s: int(counts.get(s, 0)))
            detail = ", ".join(f"{s} ({int(counts[s])})" for s in names[:24])
        else:
            detail = ", ".join(names[:24])
        extra = f" … +{len(names) - 24}" if len(names) > 24 else ""
        print(f"  {detail}{extra}")
    return out


def prepare_mock_table(
    df: pd.DataFrame,
    min_stream_stars: int = MIN_MOCK_STREAM_STARS,
    tiny_labels: set[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Drop real-stream members, ensure mock flags, demote tiny mock streams."""
    out = drop_real_stream_members(df, verbose=verbose)
    out = ensure_is_mock_stream(out)
    if tiny_labels is None:
        tiny_labels = tiny_mock_stream_labels(out, min_stars=min_stream_stars)
    return drop_tiny_mock_streams(
        out, min_stars=min_stream_stars, tiny_labels=tiny_labels, verbose=verbose
    )


def prepare_mock_cells(
    cell_dfs: dict,
    min_stream_stars: int = MIN_MOCK_STREAM_STARS,
    tiny_labels: set[str] | None = None,
    verbose: bool = True,
) -> dict:
    """``prepare_mock_table`` on every cell, with galaxy-wide tiny-stream labels."""
    if tiny_labels is None:
        tiny_labels = tiny_mock_stream_labels_from_frames(
            cell_dfs.values(), min_stars=min_stream_stars
        )
    out = {}
    n_demote = 0
    for cid, df in cell_dfs.items():
        try:
            d = drop_real_stream_members(df, verbose=False)
        except ValueError:
            d = df
        d = ensure_is_mock_stream(d)
        before = int(d["is_mock_stream"].sum())
        d = drop_tiny_mock_streams(
            d, min_stars=min_stream_stars, tiny_labels=tiny_labels, verbose=False
        )
        n_demote += before - int(d["is_mock_stream"].sum())
        out[int(cid)] = d
    if verbose:
        print(
            f"Cell tables: demoted {n_demote:,} tiny-stream members → field  "
            f"({len(tiny_labels)} streams with <{int(min_stream_stars)} stars)"
        )
    return out


def add_is_train_column(
    df: pd.DataFrame,
    seed: int = 0,
    mock_train_frac: float = 0.8,
    verbose: bool = True,
    *,
    keep_real_streams: bool = False,
) -> pd.DataFrame:
    """Label streams and set ``is_train``.

    Default composition after this call is **DES background + mocks** (real
    stream members removed). Pass ``keep_real_streams=True`` only if you
    explicitly want catalogued S5 streams left in the table.
    """
    out = df.copy()

    # --- clean real labels ---
    s5 = out["stream_s5"].astype("string").str.strip()
    s5 = s5.fillna(BACKGROUND).replace({"-999": BACKGROUND})
    s5 = s5.where(~s5.isin(MERGE_TO_AAU), "AAU")
    out["stream_s5"] = s5
    out = out.loc[~((~out["is_mock"].astype(bool)) & (out["stream_s5"] == "Test"))].copy()

    is_mock = out["is_mock"].astype(bool)
    real_stream = (~is_mock) & (out["stream_s5"] != BACKGROUND)

    # --- canonical label ---
    stream_label = pd.Series(index=out.index, dtype="object")
    sid = out.loc[is_mock, "stream_id"].astype("int64")
    stream_label.loc[is_mock] = np.where(sid == -999, BACKGROUND, "MOCK_" + sid.astype(str))
    stream_label.loc[~is_mock] = out.loc[~is_mock, "stream_s5"].astype(str)
    out["stream_label"] = stream_label
    out["is_real_stream"] = real_stream

    # --- split (real only; unused once reals are dropped), train flag ---
    out["split"] = out["split"].astype("string")
    out.loc[real_stream, "split"] = np.where(
        out.loc[real_stream, "stream_label"].isin(TEST_REAL_STREAMS), "test", "train"
    )
    out.loc[~real_stream, "split"] = "none"

    out["is_train"] = False
    out.loc[real_stream & (out["split"] == "train"), "is_train"] = True

    # --- mock stream subsample ---
    out["is_mock_stream"] = is_mock & (out["stream_id"].astype("int64") > 0)
    rng = np.random.default_rng(seed)
    mock_ids = out.loc[out["is_mock_stream"], "stream_id"].unique()
    n_train = max(int(np.floor(mock_train_frac * len(mock_ids))), 1) if len(mock_ids) else 0
    train_mock_ids = set(
        rng.choice(mock_ids, size=n_train, replace=False).tolist()
    ) if n_train else set()
    out.loc[out["is_mock_stream"] & out["stream_id"].isin(train_mock_ids), "is_train"] = True

    # --- optional convenience ---
    out["ra_des_sin"] = np.sin(np.deg2rad(out["ra_des"].to_numpy()))
    out["ra_des_cos"] = np.cos(np.deg2rad(out["ra_des"].to_numpy()))

    if not keep_real_streams:
        out = drop_real_stream_members(out, verbose=verbose)
        # After the drop, only mock members carry is_train.
        out["is_train"] = out["is_mock_stream"] & out["stream_id"].isin(train_mock_ids)
        out["split"] = "none"
        out["is_real_stream"] = False

    if verbose:
        def _hdr(t):
            print("\n" + "="*72 + f"\n{t}\n" + "="*72)
        is_mock = out["is_mock"].astype(bool)
        _hdr("Split / Train Sanity")
        print(f"N rows                : {len(out):,}")
        print(f"mock / DES            : {is_mock.sum():,} / {(~is_mock).sum():,}")
        print(f"mock stream members   : {out['is_mock_stream'].sum():,}")
        print(f"real stream members   : {int(out['is_real_stream'].sum()):,} (expect 0)")
        print(
            f"DES background        : "
            f"{(((~is_mock) & (out['stream_label']==BACKGROUND)).sum()):,}"
        )
        print("\nTrain stars breakdown:")
        print(
            f"  train mock-stream    : "
            f"{(out['is_mock_stream'] & out['is_train']).sum():,}"
        )
        print(
            f"  train DES background : "
            f"{(((~is_mock) & (out['stream_label']==BACKGROUND) & out['is_train']).sum()):,}"
        )

    return out
