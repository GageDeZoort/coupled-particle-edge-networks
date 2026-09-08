"""Load and plot hyperparameter-transfer sweep runs from Slurm job IDs.

Notebook-facing workflow::

    from cpen.utils.transfer_plots import JobSpec, TransferStudy

    jobs = [
        JobSpec("12259080_0", "lr_scan", "η0=0.005, D=128, h=4"),
        ...
    ]
    study = TransferStudy.from_jobs(jobs, d_ref=128)
    study.summary("lr_scan")          # includes depth / width / heads
    study.plot_lr_scan()              # one color per (L, D, h)
    study.plot_train_curves(depth=4, width=256)
    study.plot_width_scaling()        # series split when L or h vary

Subsequent experiment cells can build another ``TransferStudy`` from a
different job list (or call the free functions on a filtered frame).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# (depth, width, heads); missing / N/A dimensions use -1.
ModelArchKey = tuple[int, int, int]

# ---------------------------------------------------------------------------
# Defaults (Della / this repo)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIG_DIR = REPO_ROOT / "notebooks" / "figures"
DEFAULT_OUT_ROOT = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_output/adam/sweep_lr"
)
DEFAULT_SLURM_DIRS: tuple[Path, ...] = (
    REPO_ROOT / "scans" / "testing",
    REPO_ROOT / "scans" / "testing" / "job_output",
    Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs"),
)
PASCAL_OUT_ROOT = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/pascal_output/adam/sweep_lr"
)
PASCAL_SLURM_DIRS: tuple[Path, ...] = (
    REPO_ROOT / "scans" / "pascal",
) + DEFAULT_SLURM_DIRS

_RUN_RE = re.compile(r"^RUN:\s*(.+)$", re.M)
_HEADS_RE = re.compile(r"_h(\d+)(?:_|$)")

FIGSIZE_TWIN = (5.2, 2.2)
FIGSIZE_FAMILY = (5.4, 2.15)


@dataclass(frozen=True)
class JobSpec:
    """One Slurm array task to load from its ``slurm-<job_id>.out`` RUN line."""

    job_id: str
    role: str = "run"  # e.g. "lr_scan", "width_scale", or any label you invent
    note: str = ""


# ---------------------------------------------------------------------------
# Style / I/O helpers
# ---------------------------------------------------------------------------


def setup_mpl_style(*, styles: Sequence[str] | None = None) -> None:
    """Apply scienceplots styles (nature look, no-latex for Della)."""
    try:
        import scienceplots  # noqa: F401
    except ImportError:
        return
    plt.style.use(list(styles or ("science", "nature", "no-latex")))


def savefig(
    fig: plt.Figure,
    stem: str,
    *,
    fig_dir: Path | None = None,
    exts: Sequence[str] = ("pdf", "png"),
    dpi: int = 300,
) -> list[Path]:
    """Save *fig* under ``notebooks/figures/{stem}.{ext}``."""
    out_dir = Path(fig_dir or DEFAULT_FIG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in exts:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
        print("saved", path)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _job_id_candidates(job_id: str) -> tuple[str, ...]:
    """Accept ``12345`` or array-task ``12345_0``."""
    job_id = (job_id or "").strip()
    if not job_id:
        return ()
    if job_id.isdigit():
        return (job_id, f"{job_id}_0")
    return (job_id,)


def resolve_run_name(
    job_id: str,
    *,
    slurm_dirs: Sequence[Path] | None = None,
) -> str | None:
    """Parse ``RUN: <name>`` from the first matching ``slurm-<job_id>.out``."""
    dirs = tuple(Path(p) for p in (slurm_dirs or DEFAULT_SLURM_DIRS))
    for candidate in _job_id_candidates(job_id):
        for directory in dirs:
            log = directory / f"slurm-{candidate}.out"
            if not log.is_file():
                continue
            match = _RUN_RE.search(log.read_text(errors="replace"))
            if match:
                return match.group(1).strip()
    return None


def heads_from_run_name(run: str) -> int | None:
    match = _HEADS_RE.search(run)
    return int(match.group(1)) if match else None


def load_job(
    spec: JobSpec,
    *,
    out_root: Path | None = None,
    slurm_dirs: Sequence[Path] | None = None,
    quiet: bool = False,
) -> pd.DataFrame | None:
    """Load one job's parquet metrics; return ``None`` if log/parquet missing."""
    root = Path(out_root or DEFAULT_OUT_ROOT)
    run = resolve_run_name(spec.job_id, slurm_dirs=slurm_dirs)
    if run is None:
        if not quiet:
            print(f"[skip] {spec.job_id}: no log / RUN yet")
        return None
    parquet = root / f"{run}.parquet"
    if not parquet.is_file():
        if not quiet:
            print(f"[skip] {spec.job_id}: missing {parquet.name}")
        return None
    df = pd.read_parquet(parquet).copy()
    df["job_id"] = spec.job_id
    df["role"] = spec.role
    df["note"] = spec.note
    df["run_name"] = run
    df["model_family"] = _model_family_series(df)
    if "heads" not in df.columns or df["heads"].isna().all():
        heads = heads_from_run_name(run)
        df["heads"] = heads if heads is not None else pd.NA
    if not quiet:
        depth = df["depth"].iloc[0] if "depth" in df.columns else "?"
        print(
            f"[ok] {spec.job_id}: η0={df['eta_0'].iloc[0]:g}  "
            f"L={depth}  D={int(df['width'].iloc[0])}  "
            f"h={df['heads'].iloc[0]}  rows={len(df)}"
        )
    return df


def load_jobs(
    jobs: Iterable[JobSpec],
    *,
    out_root: Path | None = None,
    slurm_dirs: Sequence[Path] | None = None,
    quiet: bool = False,
) -> pd.DataFrame:
    """Load and concatenate jobs (deduped by ``job_id``, first role wins)."""
    seen: set[str] = set()
    frames: list[pd.DataFrame] = []
    for spec in jobs:
        job_id = (spec.job_id or "").strip()
        if not job_id or job_id in seen:
            continue
        seen.add(job_id)
        frame = load_job(
            JobSpec(job_id, spec.role, spec.note),
            out_root=out_root,
            slurm_dirs=slurm_dirs,
            quiet=quiet,
        )
        if frame is not None and not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if not quiet:
        depths = (
            sorted(df["depth"].dropna().unique())
            if "depth" in df.columns
            else []
        )
        heads = (
            sorted(df["heads"].dropna().unique())
            if "heads" in df.columns
            else []
        )
        print(
            f"\n{df['job_id'].nunique()} jobs · {len(df)} rows · "
            f"η0={sorted(df['eta_0'].unique())} · "
            f"L={depths} · "
            f"D={sorted(df['width'].unique())} · "
            f"h={heads}"
        )
    return df


def job_status_table(
    jobs: Iterable[JobSpec],
    df: pd.DataFrame | None = None,
    *,
    out_root: Path | None = None,
    slurm_dirs: Sequence[Path] | None = None,
) -> pd.DataFrame:
    """Per-job availability of Slurm log, RUN name, parquet, and load status."""
    root = Path(out_root or DEFAULT_OUT_ROOT)
    dirs = tuple(Path(p) for p in (slurm_dirs or DEFAULT_SLURM_DIRS))
    loaded = set() if df is None or df.empty else set(df["job_id"].unique())
    rows = []
    for spec in jobs:
        run = resolve_run_name(spec.job_id, slurm_dirs=dirs)
        parquet = root / f"{run}.parquet" if run else None
        rows.append(
            {
                "job_id": spec.job_id,
                "role": spec.role,
                "note": spec.note,
                "slurm_log": any(
                    (d / f"slurm-{jid}.out").is_file()
                    for jid in _job_id_candidates(spec.job_id)
                    for d in dirs
                ),
                "run_name": run,
                "parquet": bool(parquet and parquet.is_file()),
                "loaded": spec.job_id in loaded,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metric slices
# ---------------------------------------------------------------------------


def epoch_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["record_type"].eq("epoch")].dropna(subset=["val_loss"]).copy()


def step_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["record_type"].eq("step")].copy()


def best_val_per_run(df: pd.DataFrame) -> pd.DataFrame:
    ep = epoch_rows(df)
    if ep.empty:
        return ep
    idx = ep.groupby("job_id")["val_loss"].idxmin()
    sort_cols = [c for c in ("depth", "width", "heads", "eta_0") if c in ep.columns]
    return ep.loc[idx].sort_values(sort_cols, na_position="last").reset_index(drop=True)


def min_losses_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per ``job_id`` with *minimum* train and val loss over epoch records.

    Architecture / ``η0`` fields are taken from that job's rows (first nonempty).
    ``train_loss`` and ``val_loss`` minima are independent (possibly different epochs).
    """
    ep = epoch_rows(df)
    if ep.empty:
        return ep

    meta_cols = [
        c
        for c in (
            "job_id",
            "role",
            "note",
            "eta_0",
            "depth",
            "width",
            "heads",
            "run_name",
        )
        if c in ep.columns
    ]
    meta = ep.groupby("job_id", as_index=False).first()[meta_cols]

    mins = ep.groupby("job_id", as_index=False).agg(
        **{
            k: (k, "min")
            for k in ("train_loss", "val_loss")
            if k in ep.columns
        }
    )
    out = meta.merge(mins, on="job_id", how="left")
    sort_cols = [c for c in ("depth", "width", "heads", "eta_0") if c in out.columns]
    return out.sort_values(sort_cols, na_position="last").reset_index(drop=True)


def final_test_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ``job_id`` with final test metrics (last nonempty ROC AUC)."""
    if "test_roc_auc" in df.columns and df["test_roc_auc"].notna().any():
        te = df.dropna(subset=["test_roc_auc"]).copy()
    elif "test_f1" in df.columns and df["test_f1"].notna().any():
        te = df.dropna(subset=["test_f1"]).copy()
    else:
        return df.iloc[0:0].copy()
    if te.empty:
        return te
    idx = te.groupby("job_id")["global_step"].idxmax()
    cols = [
        "job_id",
        "role",
        "note",
        "eta_0",
        "depth",
        "width",
        "heads",
        "n_trainable_params",
        "epochs",
        "epoch",
        "learning_rate",
        "train_loss",
        "val_loss",
        "test_roc_auc",
        "test_bg_rejection",
        "test_bg_rejection_0p3",
        "test_acc",
        "val_roc_auc",
        "val_f1",
        "test_f1",
        "val_edge_auroc",
        "test_edge_auroc",
        "run_name",
    ]
    cols = [c for c in cols if c in te.columns]
    sort_cols = [c for c in ("depth", "width", "heads", "eta_0") if c in cols]
    return (
        te.loc[idx, cols]
        .sort_values(sort_cols, na_position="last")
        .reset_index(drop=True)
    )


def _arch_int(row: pd.Series, col: str) -> int:
    if col not in row.index or pd.isna(row[col]):
        return -1
    return int(row[col])


def model_size_key(row: pd.Series) -> ModelArchKey:
    """Group key for overlaying curves: ``(depth, width, heads)``."""
    return (_arch_int(row, "depth"), int(row["width"]), _arch_int(row, "heads"))


def model_size_label(
    depth: int | ModelArchKey,
    width: int | None = None,
    heads: int | None = None,
    *,
    show_depth: bool = True,
    show_width: bool = True,
    show_heads: bool = True,
) -> str:
    """Pretty label; pass either ``(L, D, h)`` or explicit ints."""
    if isinstance(depth, tuple):
        depth, width, heads = depth
    assert width is not None and heads is not None
    parts: list[str] = []
    if show_depth and depth >= 0:
        parts.append(rf"$L={depth}$")
    if show_width:
        parts.append(rf"$D={width}$")
    if show_heads and heads >= 0:
        parts.append(rf"$h={heads}$")
    return ", ".join(parts) if parts else rf"$D={width}$"


def arch_label_flags(df: pd.DataFrame) -> dict[str, bool]:
    """Which architecture dims vary (always show width; others if present/varying)."""
    def _nunique(col: str) -> int:
        if col not in df.columns:
            return 0
        return int(df[col].dropna().nunique())

    n_depth, n_heads = _nunique("depth"), _nunique("heads")
    # Show a dimension if it exists and either varies or is the only arch signal
    # besides width. Always show depth/heads when the column is present and not all-NA.
    return {
        "show_depth": n_depth >= 1,
        "show_width": True,
        "show_heads": n_heads >= 1,
    }


def mask_model_size(df: pd.DataFrame, key: ModelArchKey) -> pd.Series:
    depth, width, heads = key
    mask = df["width"].astype(int).eq(width)
    if "depth" in df.columns:
        depth_vals = pd.to_numeric(df["depth"], errors="coerce").fillna(-1).astype(int)
        mask &= depth_vals.eq(depth) if depth >= 0 else df["depth"].isna()
    if "heads" in df.columns:
        heads_vals = pd.to_numeric(df["heads"], errors="coerce").fillna(-1).astype(int)
        mask &= heads_vals.eq(heads) if heads >= 0 else df["heads"].isna()
    return mask


def pick_eta0_star(
    df: pd.DataFrame,
    *,
    prefer_test: bool = True,
) -> dict[ModelArchKey, float]:
    """
    Best ``η0`` per ``(depth, width, heads)``.

    Uses max test ROC AUC when available, else min val loss.
    """
    test = final_test_per_run(df) if prefer_test else pd.DataFrame()
    best = best_val_per_run(df)
    if not test.empty:
        metrics, score_col, maximize = test, "test_roc_auc", True
    elif not best.empty:
        metrics, score_col, maximize = best, "val_loss", False
    else:
        return {}

    tagged = metrics.copy()
    tagged["_arch"] = tagged.apply(model_size_key, axis=1)
    stars: dict[ModelArchKey, float] = {}
    for key, sub in tagged.groupby("_arch", sort=True):
        star = sub.loc[sub[score_col].idxmax() if maximize else sub[score_col].idxmin()]
        stars[key] = float(star["eta_0"])
    return stars


def filter_role(df: pd.DataFrame, role: str | None) -> pd.DataFrame:
    if role is None or df.empty or "role" not in df.columns:
        return df
    return df.loc[df["role"].eq(role)].copy()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _color_cycle(n: int):
    if hasattr(mpl, "color_sequences"):
        cmap = mpl.color_sequences["tab10"]
    else:
        cmap = plt.cm.tab10.colors
    return [cmap[i % len(cmap)] for i in range(n)]


def plot_lr_scan(
    df: pd.DataFrame,
    *,
    eta0_star: dict[ModelArchKey, float] | None = None,
    figsize: tuple[float, float] | None = None,
    stem: str | None = "transfer_lr_scan_test",
    fig_dir: Path | None = None,
    show: bool = True,
) -> plt.Figure | None:
    """
    2×2 panels vs ``η0`` (one color per architecture):

    - test ROC AUC / background rejection (or val proxies)
    - *minimum* train loss / *minimum* val loss over the run
    """
    test = final_test_per_run(df)
    best = best_val_per_run(df)
    if test.empty and best.empty:
        print("Waiting for metrics…")
        return None

    use_test = not test.empty
    metrics = test if use_test else best
    y_roc = "test_roc_auc" if use_test else "val_loss"
    y_rej = (
        "test_bg_rejection"
        if use_test and "test_bg_rejection" in metrics.columns
        else ("val_acc" if "val_acc" in metrics.columns else y_roc)
    )
    loss_src = min_losses_per_run(df)
    if eta0_star is None:
        eta0_star = pick_eta0_star(df)

    label_kwargs = arch_label_flags(metrics)
    size_keys = sorted(
        {model_size_key(row) for _, row in metrics.iterrows()},
        key=lambda ldh: (ldh[0], ldh[1], ldh[2]),
    )
    print(
        "LR-scan series:",
        ", ".join(model_size_label(k, **label_kwargs) for k in size_keys),
    )

    if figsize is None:
        figsize = (FIGSIZE_TWIN[0], FIGSIZE_TWIN[1] * 1.85)
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=400)
    ax_roc, ax_rej = axes[0, 0], axes[0, 1]
    ax_tr, ax_va = axes[1, 0], axes[1, 1]

    ax_roc.set_ylabel("Test ROC AUC" if use_test else "Best val loss")
    ax_rej.set_ylabel(
        r"Test $1/\varepsilon_B$ ($\varepsilon_S=0.5$)"
        if use_test and y_rej == "test_bg_rejection"
        else ("Val accuracy" if y_rej == "val_acc" else y_rej)
    )
    ax_tr.set_ylabel("Min train loss")
    ax_va.set_ylabel("Min val loss")

    colors = _color_cycle(len(size_keys))
    for i, key in enumerate(size_keys):
        color = colors[i]
        sub = metrics.loc[mask_model_size(metrics, key)].sort_values("eta_0")
        if sub.empty:
            continue
        label = model_size_label(key, **label_kwargs)
        if y_roc in sub.columns:
            ax_roc.plot(sub["eta_0"], sub[y_roc], "o-", color=color, label=label)
        if y_rej in sub.columns:
            ax_rej.plot(sub["eta_0"], sub[y_rej], "o-", color=color, label=label)

        if not loss_src.empty:
            loss_sub = loss_src.loc[mask_model_size(loss_src, key)].sort_values("eta_0")
            if not loss_sub.empty and "train_loss" in loss_sub.columns:
                ax_tr.plot(
                    loss_sub["eta_0"],
                    loss_sub["train_loss"],
                    "o-",
                    color=color,
                    label=label,
                )
            if not loss_sub.empty and "val_loss" in loss_sub.columns:
                ax_va.plot(
                    loss_sub["eta_0"],
                    loss_sub["val_loss"],
                    "o-",
                    color=color,
                    label=label,
                )

        star = eta0_star.get(key)
        if star is not None:
            for ax in (ax_roc, ax_rej, ax_tr, ax_va):
                ax.axvline(star, ls="--", color=color, alpha=0.45, lw=1.0)

    for ax in (ax_roc, ax_rej, ax_tr, ax_va):
        ax.set_xscale("log")
        ax.set_xlabel(r"$\eta_0$")
        ax.legend(loc="best", title="architecture", fontsize=8)

    ax_roc.set_title(r"Test ROC AUC vs $\eta_0$" if use_test else r"Val loss vs $\eta_0$")
    ax_rej.set_title(
        r"Background rejection vs $\eta_0$"
        if use_test and y_rej == "test_bg_rejection"
        else y_rej
    )
    ax_tr.set_title(r"Min train loss vs $\eta_0$")
    ax_va.set_title(r"Min val loss vs $\eta_0$")

    fig.tight_layout()
    if stem:
        savefig(fig, stem, fig_dir=fig_dir)
    if show:
        plt.show()
    return fig


def plot_train_curves(
    df: pd.DataFrame,
    *,
    depth: int | None = None,
    width: int | None = None,
    heads: int | None = None,
    figsize: tuple[float, float] = FIGSIZE_TWIN,
    stem: str | None = "transfer_lr_scan_curves",
    fig_dir: Path | None = None,
    show: bool = True,
) -> plt.Figure | None:
    """Train-loss (steps) and val-loss (epochs) for each ``η0`` in *df*."""
    if df.empty:
        print("No rows to plot.")
        return None
    plot_df = df
    filters: list[str] = []
    if depth is not None and "depth" in plot_df.columns:
        plot_df = plot_df.loc[plot_df["depth"].astype(int).eq(int(depth))]
        filters.append(rf"$L={int(depth)}$")
    if width is not None:
        plot_df = plot_df.loc[plot_df["width"].astype(int).eq(int(width))]
        filters.append(rf"$D={int(width)}$")
    if heads is not None and "heads" in plot_df.columns:
        plot_df = plot_df.loc[plot_df["heads"].fillna(-1).astype(int).eq(int(heads))]
        filters.append(rf"$h={int(heads)}$")
    if plot_df.empty:
        print(f"No rows matching depth={depth}, width={width}, heads={heads}.")
        return None

    job_ids = sorted(
        plot_df["job_id"].unique(),
        key=lambda j: float(plot_df.loc[plot_df["job_id"].eq(j), "eta_0"].iloc[0]),
    )
    title = ", ".join(filters) if filters else model_size_label(
        model_size_key(plot_df.iloc[0]), **arch_label_flags(plot_df)
    )
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for job_id in job_ids:
        sub = plot_df.loc[plot_df["job_id"].eq(job_id)]
        eta = float(sub["eta_0"].iloc[0])
        label = rf"$\eta_0={eta:g}$"
        steps = step_rows(sub).sort_values("global_step")
        if not steps.empty and "train_loss" in steps.columns:
            axes[0].plot(steps["global_step"], steps["train_loss"], label=label)
        epochs = epoch_rows(sub).sort_values("epoch")
        if not epochs.empty:
            axes[1].plot(epochs["epoch"] + 1, epochs["val_loss"], "o-", label=label)

    axes[0].set(xlabel="Step", ylabel="Train loss", title=rf"Train ({title})")
    axes[1].set(xlabel="Epoch", ylabel="Val loss", title=rf"Val ({title})")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    if stem:
        savefig(fig, stem, fig_dir=fig_dir)
    if show:
        plt.show()
    return fig


def plot_width_scaling(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = FIGSIZE_TWIN,
    stem_width: str | None = "transfer_width_scaling_test",
    stem_params: str | None = "transfer_params_scaling_test",
    fig_dir: Path | None = None,
    show: bool = True,
) -> list[plt.Figure]:
    """Test metrics vs width / params, with separate series when $L$ or $h$ vary."""
    test = final_test_per_run(df)
    if test.empty:
        print("Width-scale plots appear once jobs have test metrics.")
        return []

    figs: list[plt.Figure] = []
    label_kwargs = arch_label_flags(test)

    # --- vs width: one series per (depth, heads, eta) ---
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    series_keys: list[tuple[ModelArchKey, float]] = []
    for _, row in test.iterrows():
        depth, _width, heads = model_size_key(row)
        series_keys.append(((depth, -1, heads), float(row["eta_0"])))
    # unique preserving sort
    uniq = sorted(set(series_keys), key=lambda kh: (kh[0][0], kh[0][2], kh[1]))
    colors = _color_cycle(len(uniq))
    for i, ((depth, _, heads), eta) in enumerate(uniq):
        color = colors[i]
        mask = test["eta_0"].astype(float).eq(eta)
        if "depth" in test.columns:
            if depth < 0:
                mask &= test["depth"].isna()
            else:
                mask &= test["depth"].fillna(-1).astype(int).eq(depth)
        if "heads" in test.columns:
            if heads < 0:
                mask &= test["heads"].isna()
            else:
                mask &= test["heads"].fillna(-1).astype(int).eq(heads)
        sub = test.loc[mask].sort_values("width")
        if sub.empty:
            continue
        arch = model_size_label(
            depth, int(sub["width"].iloc[0]), heads,
            show_depth=label_kwargs["show_depth"],
            show_width=False,
            show_heads=label_kwargs["show_heads"],
        )
        label = rf"{arch}, $\eta_0={eta:g}$" if arch else rf"$\eta_0={eta:g}$"
        axes[0].plot(sub["width"], sub["test_roc_auc"], "o-", color=color, label=label)
        if "test_bg_rejection" in sub.columns:
            axes[1].plot(
                sub["width"], sub["test_bg_rejection"], "o-", color=color, label=label
            )
    axes[0].set(xlabel="Width $D$", ylabel="Test ROC AUC", title="Width scaling")
    axes[1].set(
        xlabel="Width $D$",
        ylabel=r"Test $1/\varepsilon_B$ ($\varepsilon_S=0.5$)",
        title="Width scaling",
    )
    for ax in axes:
        ax.legend(loc="best", fontsize=8)
        ax.set_xticks(sorted(test["width"].unique()))
    fig.tight_layout()
    if stem_width:
        savefig(fig, stem_width, fig_dir=fig_dir)
    if show:
        plt.show()
    figs.append(fig)

    if "n_trainable_params" not in test.columns or test["n_trainable_params"].isna().all():
        return figs

    # --- vs #params: one series per (depth, heads, eta) ---
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for i, ((depth, _, heads), eta) in enumerate(uniq):
        color = colors[i]
        mask = test["eta_0"].astype(float).eq(eta)
        if "depth" in test.columns:
            if depth < 0:
                mask &= test["depth"].isna()
            else:
                mask &= test["depth"].fillna(-1).astype(int).eq(depth)
        if "heads" in test.columns:
            if heads < 0:
                mask &= test["heads"].isna()
            else:
                mask &= test["heads"].fillna(-1).astype(int).eq(heads)
        sub = test.loc[mask].sort_values("n_trainable_params")
        if sub.empty:
            continue
        arch = model_size_label(
            depth, int(sub["width"].iloc[0]), heads,
            show_depth=label_kwargs["show_depth"],
            show_width=False,
            show_heads=label_kwargs["show_heads"],
        )
        label = rf"{arch}, $\eta_0={eta:g}$" if arch else rf"$\eta_0={eta:g}$"
        axes[0].plot(
            sub["n_trainable_params"], sub["test_roc_auc"], "o-", color=color, label=label
        )
        if "test_bg_rejection" in sub.columns:
            axes[1].plot(
                sub["n_trainable_params"],
                sub["test_bg_rejection"],
                "o-",
                color=color,
                label=label,
            )
    axes[0].set(
        xlabel="Trainable parameters",
        ylabel="Test ROC AUC",
        title="Parameter scaling",
    )
    axes[1].set(
        xlabel="Trainable parameters",
        ylabel=r"Test $1/\varepsilon_B$ ($\varepsilon_S=0.5$)",
        title="Parameter scaling",
    )
    for ax in axes:
        ax.set_xscale("log")
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if stem_params:
        savefig(fig, stem_params, fig_dir=fig_dir)
    if show:
        plt.show()
    figs.append(fig)
    return figs


# ---------------------------------------------------------------------------
# Pascal readout (node vs node+edge, node macro-F1)
# ---------------------------------------------------------------------------

_FAMILY_STYLE = {
    "node": {"ls": "-", "marker": "o"},
    "node+edge": {"ls": "--", "marker": "s"},
}

_MODEL_PANELS = (
    ("cpen", "CPEN"),
    ("capen-llama", "CAPEN-Llama"),
)

PASCAL_FAMILY_METRICS: tuple[tuple[str, str, str], ...] = (
    ("val_f1_best", r"Best val node macro-F1", "pascal_readout_lr_f1"),
    ("val_acc_best", r"Best val accuracy", "pascal_readout_lr_val_acc"),
    ("test_acc", r"Test accuracy", "pascal_readout_lr_test_acc"),
    ("test_f1", r"Test node macro-F1", "pascal_readout_lr_test_f1"),
    ("train_loss_min", r"Min train loss", "pascal_readout_lr_train_loss"),
)


def normalize_model_family(name: object) -> str:
    text = str(name or "").strip().lower().replace("_", "-")
    if text in {"capen-llama", "capenllama"}:
        return "capen-llama"
    if text.startswith("capen-llama"):
        return "capen-llama"
    if text.startswith("cpen"):
        return "cpen"
    if text.startswith("capen"):
        return "capen"
    return text


def _model_family_series(df: pd.DataFrame) -> pd.Series:
    if "model" in df.columns and df["model"].notna().any():
        return df["model"].map(normalize_model_family)
    if "run_name" in df.columns:
        return df["run_name"].map(normalize_model_family)
    return pd.Series(["cpen"] * len(df), index=df.index)


def _model_family_label(name: str) -> str:
    for key, label in _MODEL_PANELS:
        if key == name:
            return label
    return name


def pascal_readout_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per job: best ``val_f1``, last epoch, ``test_f1`` if written."""
    ep = epoch_rows(df)
    if ep.empty or "val_f1" not in ep.columns:
        return ep.iloc[0:0].copy()
    ep = ep.dropna(subset=["val_f1"])
    if ep.empty:
        return ep
    idx = ep.groupby("job_id")["val_f1"].idxmax()
    best = ep.loc[idx].copy()
    last_epoch = ep.groupby("job_id")["epoch"].max()
    best["epoch_last"] = best["job_id"].map(last_epoch)
    best["val_f1_best"] = best["val_f1"]
    test = final_test_per_run(df)
    if not test.empty:
        keyed = test.drop_duplicates("job_id").set_index("job_id")
        for c in ("test_f1", "test_acc", "test_edge_auroc"):
            if c in keyed.columns:
                best[c] = best["job_id"].map(keyed[c])
    if "val_edge_auroc" in ep.columns:
        best_edge = ep.groupby("job_id")["val_edge_auroc"].max()
        best["val_edge_auroc"] = best["job_id"].map(best_edge)
    if "val_acc" in ep.columns:
        best["val_acc_best"] = best["job_id"].map(ep.groupby("job_id")["val_acc"].max())
    if "train_loss" in ep.columns:
        best["train_loss_min"] = best["job_id"].map(
            ep.groupby("job_id")["train_loss"].min()
        )
    if "model_family" not in best.columns:
        best["model_family"] = _model_family_series(best)
    else:
        best["model_family"] = best["model_family"].map(normalize_model_family)
    planned = pd.to_numeric(best.get("epochs"), errors="coerce")
    last = pd.to_numeric(best["epoch_last"], errors="coerce")
    best["finished"] = (
        best["test_f1"].notna() if "test_f1" in best.columns else False
    )
    if planned.notna().any():
        best["finished"] = best["finished"] | (last + 1 >= planned)
    cols = [
        "job_id",
        "role",
        "note",
        "model_family",
        "eta_0",
        "depth",
        "width",
        "epoch_last",
        "epochs",
        "finished",
        "val_f1_best",
        "val_acc_best",
        "train_loss_min",
        "test_f1",
        "test_acc",
        "val_edge_auroc",
        "test_edge_auroc",
        "run_name",
    ]
    cols = [c for c in cols if c in best.columns]
    sort_cols = [
        c for c in ("model_family", "depth", "width", "role", "eta_0") if c in cols
    ]
    return best[cols].sort_values(sort_cols, na_position="last").reset_index(drop=True)


def plot_pascal_readout_lr(
    df: pd.DataFrame,
    *,
    y: str = "val_f1_best",
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    stem: str | None = "pascal_readout_lr_f1",
    fig_dir: Path | None = None,
    show: bool = True,
    dpi: int = 300,
) -> plt.Figure | None:
    """One metric vs ``η0``: CPEN (left) and CAPEN-Llama (right)."""
    summary = pascal_readout_summary(df)
    if summary.empty or y not in summary.columns:
        print(f"Waiting for Pascal {y}…")
        return None
    if figsize is None:
        figsize = FIGSIZE_FAMILY
    panels = [
        (key, title)
        for key, title in _MODEL_PANELS
        if key in set(summary.get("model_family", pd.Series(dtype=str)))
    ]
    if not panels:
        panels = [("cpen", "CPEN")]
        summary = summary.copy()
        summary["model_family"] = "cpen"
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=figsize, dpi=dpi, sharey=True)
    if n == 1:
        axes = [axes]
    label_kwargs = {"show_depth": True, "show_width": True, "show_heads": False}
    ld_keys = sorted(
        {
            (int(row["depth"]), int(row["width"]))
            for _, row in summary.iterrows()
            if pd.notna(row.get("depth")) and pd.notna(row.get("width"))
        }
    )
    roles = [f for f in ("node", "node+edge") if f in set(summary["role"])]
    colors = _color_cycle(len(ld_keys))
    color_of = dict(zip(ld_keys, colors))
    y_name = ylabel or y.replace("_", " ")

    for ax, (family, title) in zip(axes, panels):
        fam_df = summary.loc[summary["model_family"].eq(family)]
        for key in ld_keys:
            color = color_of[key]
            arch = model_size_label(key[0], key[1], -1, **label_kwargs)
            for role in roles:
                style = _FAMILY_STYLE.get(role, {"ls": "-", "marker": "o"})
                sub = fam_df.loc[
                    fam_df["depth"].astype(int).eq(key[0])
                    & fam_df["width"].astype(int).eq(key[1])
                    & fam_df["role"].eq(role)
                ].dropna(subset=[y]).sort_values("eta_0")
                if sub.empty:
                    continue
                ax.plot(
                    sub["eta_0"],
                    sub[y],
                    color=color,
                    label=f"{arch}, {role}",
                    ms=4,
                    **style,
                )
        ax.set_xscale("log")
        ax.set_xlabel(r"$\eta_0$")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=6, handlelength=1.6)
    axes[0].set_ylabel(y_name)
    fig.tight_layout()
    if stem:
        savefig(fig, stem, fig_dir=fig_dir)
    if show:
        plt.show()
    return fig


def plot_pascal_readout_suite(
    df: pd.DataFrame,
    *,
    fig_dir: Path | None = None,
    show: bool = True,
) -> list[plt.Figure]:
    """CPEN | CAPEN-Llama η0 scans for F1, acc, and train loss."""
    figs: list[plt.Figure] = []
    for y, ylabel, stem in PASCAL_FAMILY_METRICS:
        fig = plot_pascal_readout_lr(
            df, y=y, ylabel=ylabel, stem=stem, fig_dir=fig_dir, show=show
        )
        if fig is not None:
            figs.append(fig)
    return figs


def plot_pascal_f1_curves(
    df: pd.DataFrame,
    *,
    figsize: tuple[float, float] | None = None,
    stem: str | None = "pascal_readout_val_f1_curves",
    fig_dir: Path | None = None,
    show: bool = True,
) -> plt.Figure | None:
    """Val node macro-F1 vs epoch, one panel per architecture."""
    ep = epoch_rows(df)
    if ep.empty or "val_f1" not in ep.columns:
        print("No val_f1 epoch rows.")
        return None
    ep = ep.dropna(subset=["val_f1"])
    label_kwargs = {"show_depth": True, "show_width": True, "show_heads": False}
    size_keys = sorted(
        {
            (int(row["depth"]), int(row["width"]))
            for _, row in ep.iterrows()
            if pd.notna(row.get("depth")) and pd.notna(row.get("width"))
        }
    )
    if "model_family" not in ep.columns:
        ep["model_family"] = _model_family_series(ep)
    else:
        ep["model_family"] = ep["model_family"].map(normalize_model_family)
    model_rows = [k for k, _ in _MODEL_PANELS if k in set(ep["model_family"])]
    if not model_rows:
        model_rows = ["cpen"]
    n = max(len(size_keys), 1)
    n_r = len(model_rows)
    if figsize is None:
        figsize = (FIGSIZE_FAMILY[0] * 0.58 * n, FIGSIZE_FAMILY[1] * 0.95 * n_r)
    fig, axes = plt.subplots(n_r, n, figsize=figsize, dpi=300, sharey=True, squeeze=False)
    etas = sorted(ep["eta_0"].dropna().unique())
    eta_colors = dict(zip(etas, _color_cycle(len(etas))))
    families = [f for f in ("node", "node+edge") if f in set(ep["role"])]
    for row_i, model_key in enumerate(model_rows):
        model_ep = ep.loc[ep["model_family"].eq(model_key)]
        for col_i, key in enumerate(size_keys):
            ax = axes[row_i][col_i]
            sub = model_ep.loc[
                model_ep["depth"].astype(int).eq(key[0])
                & model_ep["width"].astype(int).eq(key[1])
            ]
            for fam in families:
                style = _FAMILY_STYLE.get(fam, {"ls": "-", "marker": None})
                for eta in etas:
                    curve = sub.loc[
                        sub["role"].eq(fam) & sub["eta_0"].eq(eta)
                    ].sort_values("epoch")
                    if curve.empty:
                        continue
                    ax.plot(
                        curve["epoch"] + 1,
                        curve["val_f1"],
                        color=eta_colors[eta],
                        ls=style["ls"],
                        lw=1.0,
                        label=rf"{fam}, $\eta_0={eta:g}$",
                    )
            ax.set_xlabel("Epoch")
            title = model_size_label(key[0], key[1], -1, **label_kwargs)
            ax.set_title(f"{_model_family_label(model_key)}, {title}")
            ax.legend(loc="best", fontsize=5, handlelength=1.4)
        axes[row_i][0].set_ylabel("Val node macro-F1")
    fig.tight_layout()
    if stem:
        savefig(fig, stem, fig_dir=fig_dir)
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Notebook facade
# ---------------------------------------------------------------------------


class TransferStudy:
    """Bundle of loaded jobs with convenience summaries and plots."""

    def __init__(
        self,
        df: pd.DataFrame,
        jobs: Sequence[JobSpec] | None = None,
        *,
        d_ref: int = 128,
        out_root: Path | None = None,
        slurm_dirs: Sequence[Path] | None = None,
        fig_dir: Path | None = None,
    ) -> None:
        self.df = df
        self.jobs = list(jobs or [])
        self.d_ref = int(d_ref)
        self.out_root = Path(out_root or DEFAULT_OUT_ROOT)
        self.slurm_dirs = tuple(Path(p) for p in (slurm_dirs or DEFAULT_SLURM_DIRS))
        self.fig_dir = Path(fig_dir or DEFAULT_FIG_DIR)

    @classmethod
    def from_jobs(
        cls,
        jobs: Sequence[JobSpec],
        *,
        d_ref: int = 128,
        out_root: Path | None = None,
        slurm_dirs: Sequence[Path] | None = None,
        fig_dir: Path | None = None,
        quiet: bool = False,
    ) -> TransferStudy:
        df = load_jobs(
            jobs, out_root=out_root, slurm_dirs=slurm_dirs, quiet=quiet
        )
        if df.empty:
            raise RuntimeError("No job parquets loaded yet.")
        return cls(
            df,
            jobs,
            d_ref=d_ref,
            out_root=out_root,
            slurm_dirs=slurm_dirs,
            fig_dir=fig_dir,
        )

    def by_role(self, role: str) -> pd.DataFrame:
        return filter_role(self.df, role)

    def summary(self, role: str | None = None) -> pd.DataFrame:
        """Final test metrics (fallback: best-val) for a role or all jobs."""
        df = filter_role(self.df, role)
        test = final_test_per_run(df)
        if not test.empty:
            return test
        return best_val_per_run(df)

    def eta0_star(self, role: str = "lr_scan") -> dict[ModelArchKey, float]:
        return pick_eta0_star(self.by_role(role))

    def print_eta0_star(self, role: str = "lr_scan") -> dict[ModelArchKey, float]:
        stars = self.eta0_star(role)
        metrics = self.summary(role)
        if metrics.empty:
            print(f"No metrics for role={role!r}.")
            return stars
        score_col = "test_roc_auc" if "test_roc_auc" in metrics.columns else "val_loss"
        label_kwargs = arch_label_flags(metrics)
        print(f"η0★ by model architecture ({role}):")
        for key, eta in stars.items():
            sub = metrics.loc[mask_model_size(metrics, key)]
            if sub.empty or score_col not in sub.columns:
                print(f"  {model_size_label(key, **label_kwargs)}: η0★={eta:g}")
                continue
            star = sub.loc[(sub["eta_0"] - eta).abs().idxmin()]
            print(
                f"  {model_size_label(key, **label_kwargs)}: η0★={eta:g}  "
                f"(job {star['job_id']}, {score_col}={star[score_col]:.4f})"
            )
        return stars

    def status(self) -> pd.DataFrame:
        jobs = self.jobs or [
            JobSpec(jid, str(role))
            for jid, role in (
                self.df.groupby("job_id")["role"].first().items()
                if not self.df.empty
                else []
            )
        ]
        return job_status_table(
            jobs,
            self.df,
            out_root=self.out_root,
            slurm_dirs=self.slurm_dirs,
        )

    def plot_lr_scan(
        self,
        role: str = "lr_scan",
        *,
        stem: str | None = "transfer_lr_scan_test",
        show: bool = True,
    ) -> plt.Figure | None:
        return plot_lr_scan(
            self.by_role(role),
            eta0_star=self.eta0_star(role),
            stem=stem,
            fig_dir=self.fig_dir,
            show=show,
        )

    def plot_train_curves(
        self,
        role: str = "lr_scan",
        *,
        depth: int | None = None,
        width: int | None = None,
        heads: int | None = None,
        stem: str | None = "transfer_lr_scan_curves",
        show: bool = True,
    ) -> plt.Figure | None:
        return plot_train_curves(
            self.by_role(role),
            depth=depth,
            width=self.d_ref if width is None else width,
            heads=heads,
            stem=stem,
            fig_dir=self.fig_dir,
            show=show,
        )

    def plot_width_scaling(
        self,
        role: str = "width_scale",
        *,
        stem_width: str | None = "transfer_width_scaling_test",
        stem_params: str | None = "transfer_params_scaling_test",
        show: bool = True,
    ) -> list[plt.Figure]:
        return plot_width_scaling(
            self.by_role(role),
            stem_width=stem_width,
            stem_params=stem_params,
            fig_dir=self.fig_dir,
            show=show,
        )

    def pascal_summary(self) -> pd.DataFrame:
        return pascal_readout_summary(self.df)

    def plot_pascal_readout(
        self,
        *,
        y: str = "val_f1_best",
        ylabel: str | None = None,
        stem: str | None = "pascal_readout_lr_f1",
        show: bool = True,
    ) -> plt.Figure | None:
        return plot_pascal_readout_lr(
            self.df, y=y, ylabel=ylabel, stem=stem, fig_dir=self.fig_dir, show=show
        )

    def plot_pascal_readout_suite(self, *, show: bool = True) -> list[plt.Figure]:
        return plot_pascal_readout_suite(self.df, fig_dir=self.fig_dir, show=show)

    def plot_pascal_f1_curves(
        self,
        *,
        stem: str | None = "pascal_readout_val_f1_curves",
        show: bool = True,
    ) -> plt.Figure | None:
        return plot_pascal_f1_curves(
            self.df, stem=stem, fig_dir=self.fig_dir, show=show
        )


def eta0_star_for_ref(
    stars: dict[ModelArchKey, float],
    d_ref: int,
    *,
    depth: int | None = None,
    heads: int | None = None,
) -> float | None:
    """Pick a reference-architecture ``η0★`` (prefer matching $D$, then $L$/$h$)."""
    candidates = [(k, v) for k, v in stars.items() if k[1] == d_ref]
    if depth is not None:
        candidates = [(k, v) for k, v in candidates if k[0] == depth]
    if heads is not None:
        candidates = [(k, v) for k, v in candidates if k[2] == heads]
    if candidates:
        return candidates[0][1]
    if stars:
        return next(iter(stars.values()))
    return None
