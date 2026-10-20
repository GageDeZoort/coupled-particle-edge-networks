#!/usr/bin/env python3
"""Nature-ready μP feature-stability figures: ρ / Δρ of X and E.

Layout (2×3): rows = particle \(X\) / edge \(E\); columns =
encoder \(\\ell=0\), last residual \(\\ell=L\), decoder \(\\ell=L+1\).

Stems are pool-tagged:
  ``fig_stability_attn`` / ``fig_stability_attn_rho`` / ``fig_stability_attn_combo``
  ``fig_stability_uniform`` / ``fig_stability_uniform_rho`` / ``fig_stability_uniform_combo``

  python scans/jets/plot_stability_nature.py --pool attn
  python scans/jets/plot_stability_nature.py --pool uniform \\
      --csv paper/data/stability_one_step_uniform.csv
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from paper.style.nature_plots import save_paper_figure, use_nature

DATA = REPO / "paper" / "data"
FIG_DIR = REPO / "paper" / "figures"
CSV_CANON = DATA / "stability_one_step.csv"
CSV_TRIALS = DATA / "stability_one_step_trials.csv"
CSV_LEGACY = DATA / "mup_one_step_probe.csv"
CSV_UNIFORM = DATA / "stability_one_step_uniform.csv"
FIG_STEM = "fig_stability"
OP_K, OP_R = 12, 0.1

STAGE_COLS = (
    ("encode", r"encoder $\ell=0$"),
    ("last", r"last residual $\ell=L$"),
    ("output", r"output $\ell=L+1$"),
)
STREAM_ROWS = (
    ("x", r"particle $X$"),
    ("e", r"edge $E$"),
)

AGG_KEYS = (
    "depth",
    "width",
    "k",
    "radius",
    "uniform_pool",
    "channel",
    "stage",
    "stream",
    "eta_0",
)


def _ensure_csv() -> Path:
    if CSV_CANON.is_file():
        return CSV_CANON
    if CSV_TRIALS.is_file():
        return CSV_TRIALS
    if CSV_LEGACY.is_file():
        shutil.copy2(CSV_LEGACY, CSV_CANON)
        return CSV_CANON
    raise FileNotFoundError(f"missing {CSV_CANON}")


def _aggregate_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if "msq_0_sem" in df.columns:
        return df
    trial_cols = {"seed", "batch_id"}
    if not trial_cols.intersection(df.columns):
        out = df.copy()
        out["msq_0_sem"] = 0.0
        out["dmsq_1_sem"] = 0.0
        out["n_trials"] = 1
        return out
    keys = [c for c in AGG_KEYS if c in df.columns]
    g = (
        df.groupby(keys, as_index=False)
        .agg(
            msq_0=("msq_0", "mean"),
            msq_0_std=("msq_0", "std"),
            dmsq_1=("dmsq_1", "mean"),
            dmsq_1_std=("dmsq_1", "std"),
            n_trials=("msq_0", "count"),
            n_params=("n_params", "first") if "n_params" in df.columns else ("msq_0", "count"),
            lr=("lr", "first") if "lr" in df.columns else ("msq_0", "count"),
        )
    )
    n = g["n_trials"].clip(lower=1).astype(float)
    g["msq_0_sem"] = g["msq_0_std"].fillna(0.0) / np.sqrt(n)
    g["dmsq_1_sem"] = g["dmsq_1_std"].fillna(0.0) / np.sqrt(n)
    return g


def _ensure_stage_stream(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill stage/stream from channel if needed (new probe writes both)."""
    out = df.copy()
    if "stage" not in out.columns or "stream" not in out.columns:
        if "channel" not in out.columns:
            raise ValueError(f"CSV needs channel or stage/stream columns")
        # Legacy QKV channels are not plottable with this figure — require re-run.
        ok = out["channel"].astype(str).str.match(
            r"^(encode|last|output)_(x|e|particle|edge)$"
        )
        if not bool(ok.all()):
            raise ValueError(
                "CSV has legacy QKV/MLP channels; re-run "
                "scans/jets/mup_one_step_transfer_probe.py"
            )
        parts = out["channel"].astype(str).str.rsplit("_", n=1, expand=True)
        out["stage"] = parts[0]
        stream = parts[1].replace({"particle": "x", "edge": "e"})
        out["stream"] = stream
    else:
        out["stream"] = out["stream"].replace({"particle": "x", "edge": "e"})
    return out


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "uniform_pool" in df.columns and df.uniform_pool.dtype == object:
        df["uniform_pool"] = df["uniform_pool"].map(
            lambda x: str(x).lower() in {"1", "true", "yes"}
        )
    df = _ensure_stage_stream(df)
    return _aggregate_if_needed(df)


def _op(df: pd.DataFrame) -> pd.DataFrame:
    if "k" in df.columns and "radius" in df.columns:
        sub = df[(df.k == OP_K) & np.isclose(df.radius, OP_R)]
        if not sub.empty:
            return sub.copy()
    return df.copy()


def _eta_colors(etas: list[float]) -> dict[float, tuple]:
    if not etas:
        return {}
    # Log-spaced η₀ → sequential colormap (light→dark = small→large).
    norm = Normalize(vmin=0, vmax=max(len(etas) - 1, 1))
    try:
        cmap = plt.colormaps["viridis"]
    except (AttributeError, KeyError):
        cmap = cm.get_cmap("viridis")
    return {eta: cmap(norm(i)) for i, eta in enumerate(etas)}


def _style_x(ax, widths: list[float] | None = None) -> None:
    ax.set_xscale("log", base=2)
    if widths:
        ticks = sorted({float(w) for w in widths})
    else:
        ticks = [64, 128, 256, 384, 512, 768, 1024, 1536, 2048]
    # Avoid label pile-up on dense ladders: keep major paper ticks.
    prefer = {64, 128, 256, 512, 1024, 2048}
    show = [t for t in ticks if int(t) in prefer] or ticks
    # Always include endpoints present in data.
    if ticks:
        if ticks[0] not in show:
            show = [ticks[0], *show]
        if ticks[-1] not in show:
            show = [*show, ticks[-1]]
        show = sorted(set(show))
    ax.set_xticks(show)
    ax.set_xticklabels([str(int(t)) if float(t).is_integer() else f"{t:g}" for t in show])
    ax.set_xlabel(r"width $D$")
    if ticks:
        pad = 1.08
        ax.set_xlim(ticks[0] / pad, ticks[-1] * pad)

# Particle vs edge (Wong palette); η₀ on Δρ via linestyle (color stays stream).
STREAM_STYLE = {
    "x": {"color": "#0072B2", "label": r"particle $X$"},
    "e": {"color": "#D55E00", "label": r"edge $E$"},
}
# Working η₀ window for Δρ: solid / long dash / short dash (not dotted).
DRHO_ETA_LINESTYLES: dict[float, str | tuple] = {
    1.0: "-",
    0.5: (0, (6.0, 2.0)),
    0.25: (0, (2.0, 1.25)),
}


def _eta_linestyle(eta: float) -> str | tuple:
    for k, ls in DRHO_ETA_LINESTYLES.items():
        if np.isclose(float(eta), k):
            return ls
    return "-"


def _plot_grid(
    df: pd.DataFrame,
    *,
    ycol: str,
    semcol: str,
    ylabel: str,
    stem: str,
    uniform_pool: bool | None = None,
    ylim: tuple[float, float] | None = None,
) -> list[Path]:
    op = _op(df)
    if "uniform_pool" in op.columns and uniform_pool is not None:
        op = op[op.uniform_pool == bool(uniform_pool)].copy()
    elif "uniform_pool" in op.columns and (~op.uniform_pool).any():
        # Default: attentional when both present.
        op = op[~op.uniform_pool].copy()
    if op.empty:
        raise RuntimeError(f"no rows to plot (uniform_pool={uniform_pool})")

    etas = sorted(float(e) for e in op["eta_0"].unique())
    colors = _eta_colors(etas)
    widths = sorted(float(w) for w in op["width"].unique())

    use_nature(fontsize=8)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.2, 4.0),
        sharex=True,
        gridspec_kw={"hspace": 0.28, "wspace": 0.32},
    )

    for r, (stream, row_lab) in enumerate(STREAM_ROWS):
        for c, (stage, col_lab) in enumerate(STAGE_COLS):
            ax = axes[r, c]
            sub = op[(op.stream == stream) & (op.stage == stage)]
            for eta in etas:
                s = sub[np.isclose(sub.eta_0, eta)].sort_values("width")
                if s.empty:
                    continue
                y = s[ycol].to_numpy(dtype=float)
                x = s["width"].to_numpy(dtype=float)
                ax.plot(
                    x,
                    y,
                    "-o",
                    color=colors[eta],
                    ms=3.0,
                    lw=0.9,
                    zorder=3,
                    label=rf"$\eta_0={eta:g}$" if (r == 0 and c == 2) else None,
                )
            if r == 0:
                ax.set_title(col_lab, loc="left", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{row_lab}\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
            _style_x(ax, widths)
            ax.set_yscale("log")
            if ylim is not None:
                ax.set_ylim(*ylim)
            if r == 0:
                ax.set_xlabel("")

    handles, labels = axes[0, 2].get_legend_handles_labels()
    if handles:
        axes[0, 2].legend(
            handles,
            labels,
            loc="best",
            fontsize=6,
            frameon=False,
            handlelength=1.4,
            labelspacing=0.25,
        )

    written = save_paper_figure(fig, stem, formats=("png", "pdf", "eps"))
    plt.close(fig)
    return written


def _plot_rho_drho_combined(
    df: pd.DataFrame,
    *,
    stem: str,
    uniform_pool: bool | None = None,
    drho_etas: tuple[float, ...] | None = None,
) -> list[Path]:
    """2×3 paper combo (width now; depth rows land later).

    Row 0 — pre-step ρ vs \(D\): color = stream (\(X\) / \(E\)).
    Row 1 — one-step Δρ vs \(D\): color = stream, linestyle = η₀.

    Columns: encode / last residual / output. No error bars (SEM ≪ marker).
    """
    from matplotlib.lines import Line2D

    op = _op(df)
    if "uniform_pool" in op.columns and uniform_pool is not None:
        op = op[op.uniform_pool == bool(uniform_pool)].copy()
    elif "uniform_pool" in op.columns and (~op.uniform_pool).any():
        op = op[~op.uniform_pool].copy()
    if op.empty:
        raise RuntimeError(f"no rows for combined plot (uniform_pool={uniform_pool})")

    all_etas = sorted(float(e) for e in op["eta_0"].unique())
    eta_keys = list(drho_etas) if drho_etas is not None else list(DRHO_ETA_LINESTYLES)
    drho_eta_list = [e for e in eta_keys if any(np.isclose(e, a) for a in all_etas)]
    if not drho_eta_list:
        drho_eta_list = all_etas
    rho_eta = all_etas[0] if all_etas else None
    widths = sorted(float(w) for w in op["width"].unique())

    use_nature(fontsize=7)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(7.2, 3.8),
        sharex=True,
        gridspec_kw={"hspace": 0.28, "wspace": 0.30},
    )

    for c, (stage, col_lab) in enumerate(STAGE_COLS):
        # --- ρ (pre-step): one curve per stream ---
        ax_r = axes[0, c]
        for stream, sty in STREAM_STYLE.items():
            if rho_eta is None:
                continue
            s = op[
                (op.stream == stream)
                & (op.stage == stage)
                & np.isclose(op.eta_0, rho_eta)
            ].sort_values("width")
            if s.empty:
                continue
            y = s["msq_0"].to_numpy(dtype=float)
            x = s["width"].to_numpy(dtype=float)
            ax_r.plot(
                x,
                y,
                "-o",
                color=sty["color"],
                ms=3.0,
                lw=0.95,
                zorder=3,
                label=sty["label"] if c == 2 else None,
            )
        ax_r.set_title(col_lab, loc="left", fontsize=8)
        ax_r.set_ylabel(r"$\rho$ (pre-step)" if c == 0 else r"$\rho$")
        ax_r.set_yscale("log")
        _style_x(ax_r, widths)
        ax_r.set_xlabel("")

        # --- Δρ: color = stream, linestyle = η₀ ---
        ax_d = axes[1, c]
        for stream, sty in STREAM_STYLE.items():
            for eta in drho_eta_list:
                s = op[
                    (op.stream == stream)
                    & (op.stage == stage)
                    & np.isclose(op.eta_0, eta)
                ].sort_values("width")
                if s.empty:
                    continue
                y = s["dmsq_1"].to_numpy(dtype=float)
                x = s["width"].to_numpy(dtype=float)
                ax_d.plot(
                    x,
                    y,
                    linestyle=_eta_linestyle(eta),
                    marker="o",
                    color=sty["color"],
                    ms=2.8,
                    lw=1.05,
                    zorder=3,
                )
        ax_d.set_ylabel(r"$\Delta\rho$")
        ax_d.set_yscale("log")
        _style_x(ax_d, widths)

    stream_handles = [
        Line2D(
            [0], [0], color=sty["color"], marker="o", lw=0.95, ms=3.5, label=sty["label"]
        )
        for sty in STREAM_STYLE.values()
    ]
    eta_handles = [
        Line2D(
            [0],
            [0],
            color="0.2",
            linestyle=_eta_linestyle(eta),
            marker="o",
            lw=1.05,
            ms=3.0,
            label=rf"$\eta_0={eta:g}$",
        )
        for eta in drho_eta_list
    ]
    axes[0, 2].legend(
        handles=stream_handles,
        loc="upper right",
        fontsize=5.5,
        frameon=False,
        handlelength=1.3,
        title="stream",
        title_fontsize=5.5,
    )
    axes[1, 2].legend(
        handles=eta_handles,
        loc="best",
        fontsize=5.5,
        frameon=False,
        handlelength=2.0,
        title=r"$\Delta\rho$ $\eta_0$",
        title_fontsize=5.5,
    )

    written = save_paper_figure(fig, stem, formats=("png", "pdf", "eps"))
    plt.close(fig)
    return written


def write_caption(path: Path, df: pd.DataFrame, *, pool: str, stem: str) -> None:
    n = int(df["n_trials"].max()) if "n_trials" in df.columns else 1
    etas = sorted(float(e) for e in df["eta_0"].unique()) if "eta_0" in df.columns else []
    widths = sorted(int(w) for w in df["width"].unique()) if "width" in df.columns else []
    depths = sorted(int(d) for d in df["depth"].unique()) if "depth" in df.columns else []
    eta_str = ", ".join(f"{e:g}" for e in etas)
    w_str = ",".join(str(w) for w in widths)
    L_str = ",".join(str(d) for d in depths) if depths else "?"
    d_head = None
    if "d_head" in df.columns and df["d_head"].notna().any():
        d_head = int(df["d_head"].dropna().iloc[0])
    elif "heads" in df.columns and widths:
        # infer if constant d_h across rows
        hs = df.dropna(subset=["heads"]) if "heads" in df.columns else df
        if not hs.empty and widths:
            dh = {int(r.width) // int(r.heads) for r in hs.itertuples() if int(r.heads)}
            if len(dh) == 1:
                d_head = next(iter(dh))
    dh_str = f"d_h={d_head}" if d_head is not None else "d_h=?"
    lines = [
        f"Fig. {stem}. One-step Adam feature norms/updates for CAPEN-Llama-att",
        rf"(η = η₀/√D) at k★={OP_K}, R★={OP_R}. Pool={pool}. L∈{{{L_str}}}, {dh_str}.",
        rf"D∈{{{w_str}}}. η₀ ∈ {{{eta_str}}}.",
        f"Each point is mean over {n} trials (model seeds × data shards); "
        f"SEM ≪ marker size so error bars are omitted.",
        f"Combo: row1 pre-step ρ (color=X/E); row2 Δρ (color=stream, linestyle=η₀∈{{0.25,0.5,1}}).",
        f"Depth-scaling rows will append below when ready.",
        f"Files: {stem}.png (Δρ), {stem}_rho.png (ρ), {stem}_combo.png.",
        r"Columns: encoder (ℓ=0), last residual (ℓ=L), readout (ℓ=L+1).",
        r"Δρ = ||ΔZ||_F² / (N_live · d); ρ = ||Z||_F² / (N_live · d)",
        r"with d = D (hidden) or C (readout).",
        "Source: paper/data/stability_one_step_*paper*.csv.",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


def _pool_stem(base: str, pool_name: str) -> str:
    """Ensure stem ends with _attn or _uniform."""
    tag = f"_{pool_name}"
    if base.endswith(tag):
        return base
    # Strip a mistaken opposite tag, then append.
    for other in ("_attn", "_uniform"):
        if other != tag and base.endswith(other):
            base = base[: -len(other)]
            break
    return f"{base}{tag}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument(
        "--stem",
        type=str,
        default=FIG_STEM,
        help="Base stem; _attn / _uniform is appended automatically",
    )
    ap.add_argument(
        "--pool",
        type=str,
        default="attn",
        choices=("attn", "uniform", "both"),
        help="Which pool mode(s) to plot from the CSV",
    )
    args = ap.parse_args()

    if args.csv is not None:
        csv_path = args.csv
    elif args.pool == "uniform":
        csv_path = CSV_UNIFORM if CSV_UNIFORM.is_file() else _ensure_csv()
    else:
        csv_path = _ensure_csv()
    df = _load(csv_path)

    if args.pool == "both":
        pool_jobs: list[tuple[bool | None, str]] = [(False, "attn"), (True, "uniform")]
    elif args.pool == "uniform":
        pool_jobs = [(True, "uniform")]
    else:
        pool_jobs = [(False, "attn")]

    written: list[Path] = []
    for uniform_flag, pool_name in pool_jobs:
        stem = _pool_stem(args.stem, pool_name)
        sub = df
        if "uniform_pool" in df.columns and uniform_flag is not None:
            cand = df[df.uniform_pool == bool(uniform_flag)]
            if cand.empty and args.pool != "both":
                cand = df
                uniform_flag = None
            sub = cand
        if sub.empty:
            print(f"skip pool={pool_name}: no rows")
            continue
        # Standalone Δρ: working η₀ window only (matches combo).
        drho = sub[sub["eta_0"].apply(lambda e: any(np.isclose(float(e), t) for t in (0.25, 0.5, 1.0)))]
        if drho.empty:
            drho = sub
        written += _plot_grid(
            drho,
            ycol="dmsq_1",
            semcol="dmsq_1_sem",
            ylabel=r"$\Delta\rho$",
            stem=stem,
            uniform_pool=uniform_flag,
        )
        # Standalone ρ: pre-step; one η₀ curve (η₀-independent).
        rho = sub[np.isclose(sub["eta_0"], float(sorted(sub["eta_0"].unique())[0]))]
        written += _plot_grid(
            rho,
            ycol="msq_0",
            semcol="msq_0_sem",
            ylabel=r"$\rho$ (pre-step)",
            stem=f"{stem}_rho",
            uniform_pool=uniform_flag,
        )
        written += _plot_rho_drho_combined(
            sub, stem=f"{stem}_combo", uniform_pool=uniform_flag
        )
        write_caption(
            FIG_DIR / f"{stem}_caption.txt", sub, pool=pool_name, stem=stem
        )

    for p in written:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
