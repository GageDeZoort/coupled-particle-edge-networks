"""Nature Machine Intelligence figure style for this repo.

Usage::

    from paper.style.nature_plots import use_nature, save_paper_figure
    use_nature()
    fig, ax = plt.subplots()
    ...
    save_paper_figure(fig, "paper/figures/fig1_ablation")

All paper plots should go through ``use_nature`` + ``save_paper_figure`` so
PNG (review), PDF, and EPS (production) stay in lockstep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt

# Colorblind-friendly qualitative palette (Wong / Nature-ish).
NATURE_COLORS: Sequence[str] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
    "#F0E442",  # yellow
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIG_DIR = REPO_ROOT / "paper" / "figures"


def use_nature(*, fontsize: int = 8) -> None:
    """Apply scienceplots Nature style with a few journal-friendly overrides."""
    # Import registers the style library with matplotlib.
    import scienceplots  # noqa: F401

    plt.style.use(["science", "nature", "no-latex"])
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.size": fontsize,
            "axes.labelsize": fontsize,
            "axes.titlesize": fontsize,
            "xtick.labelsize": fontsize - 1,
            "ytick.labelsize": fontsize - 1,
            "legend.fontsize": fontsize - 1,
            "legend.frameon": False,
            "axes.grid": False,
            "pdf.fonttype": 42,  # editable text in Illustrator
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_paper_figure(
    fig: mpl.figure.Figure,
    stem: str | Path,
    *,
    formats: Sequence[str] = ("png", "pdf", "eps"),
    fig_dir: Path | None = None,
) -> list[Path]:
    """Write ``stem.{png,pdf,eps}`` under ``paper/figures`` (or ``fig_dir``)."""
    out_dir = Path(fig_dir) if fig_dir is not None else DEFAULT_FIG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_path = Path(stem)
    if stem_path.suffix:
        stem_path = stem_path.with_suffix("")
    base = out_dir / stem_path.name
    written: list[Path] = []
    for fmt in formats:
        path = base.with_suffix(f".{fmt}")
        fig.savefig(path, format=fmt)
        written.append(path)
    return written


def nature_color(i: int) -> str:
    return NATURE_COLORS[i % len(NATURE_COLORS)]
