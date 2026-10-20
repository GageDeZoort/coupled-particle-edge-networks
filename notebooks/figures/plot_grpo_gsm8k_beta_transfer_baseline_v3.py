#!/usr/bin/env python3
"""GRPO GSM8K β transfer: Baseline vs CompleteP.

Baseline (proper_scaling=False):
  /scratch/.../jgdezoort/.../grpo_gsm8k_beta_xfer_baseline_d{512,1024}_*_v3/
CompleteP (proper_scaling=True):
  /scratch/.../jdezoort/.../grpo_gsm8k_beta_xfer_d{512,1024}_*_v1/

Final eval_reward = last eval in trainer_state;
train reward = all_results.json "reward".
Encoding: color by family, solid/dashed by width (d=1024/512), circle markers.
Each panel has its own padded y-limits from that panel's series only
(Baseline and CompleteP share a scale within each metric).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASELINE_ROOT = Path(
    "/scratch/gpfs/BHANIN/jgdezoort/posttrain_scaling/outputs"
)
COMPLETEP_ROOT = Path(
    "/scratch/gpfs/BHANIN/jdezoort/posttrain_scaling/outputs"
)
FIG_DIR = Path(__file__).resolve().parent
OUT_PRIMARY = FIG_DIR / "grpo_gsm8k_beta_transfer_baseline_vs_completep"
OUT_LEGACY = FIG_DIR / "grpo_gsm8k_beta_transfer_baseline_v3"
BETA0_X = 1e-5  # plot β=0 at this log-x position

_BASELINE_RE = re.compile(
    r"grpo_gsm8k_beta_xfer_baseline_d(\d+)_lr1e-4_eps0\.2_beta([^_]+)_v3$"
)
_COMPLETEP_RE = re.compile(
    r"grpo_gsm8k_beta_xfer_d(\d+)_lr1e-4_eps0\.2_beta([^_]+)_v1$"
)

# Color by family; linestyle by width; all markers are circles
COLORS = {"Baseline": "#1f6f8b", "CompleteP": "#c45c26"}
LINESTYLES = {512: "--", 1024: "-"}


def _latest_trainer_state(run_dir: Path) -> Path | None:
    candidates = list(run_dir.glob("checkpoint-*/trainer_state.json"))
    if (run_dir / "trainer_state.json").exists():
        candidates.append(run_dir / "trainer_state.json")
    if not candidates:
        return None
    best, best_len = None, -1
    for c in candidates:
        try:
            n = len(json.load(open(c)).get("log_history", []))
        except Exception:
            n = -1
        if n > best_len:
            best, best_len = c, n
    return best


def _extract_one(run_dir: Path, family: str, dim: int, beta: float) -> dict:
    ts_path = _latest_trainer_state(run_dir)
    eval_reward, eval_step = None, None
    if ts_path is not None:
        log = json.load(open(ts_path)).get("log_history", [])
        evals = [
            (e.get("step"), e["eval_reward"])
            for e in log
            if "eval_reward" in e
        ]
        if evals:
            eval_step, eval_reward = evals[-1]

    all_res = {}
    ar = run_dir / "all_results.json"
    if ar.exists():
        all_res = json.load(open(ar))
    train_reward = all_res.get("reward")
    if train_reward is None and ts_path is not None:
        log = json.load(open(ts_path)).get("log_history", [])
        trains = [
            (e.get("step"), e["reward"])
            for e in log
            if "reward" in e and "eval_reward" not in e
        ]
        if trains:
            train_reward = trains[-1][1]

    return {
        "dir": run_dir.name,
        "family": family,
        "dim": dim,
        "beta": beta,
        "eval_reward": float(eval_reward) if eval_reward is not None else np.nan,
        "eval_step": eval_step,
        "train_reward": float(train_reward) if train_reward is not None else np.nan,
    }


def extract_runs() -> list[dict]:
    rows: list[dict] = []

    for d in sorted(BASELINE_ROOT.glob("grpo_gsm8k_beta_xfer_baseline_d*_v3")):
        m = _BASELINE_RE.search(d.name)
        if not m:
            continue
        rows.append(_extract_one(d, "Baseline", int(m.group(1)), float(m.group(2))))

    for d in sorted(COMPLETEP_ROOT.glob("grpo_gsm8k_beta_xfer_d*_v1")):
        m = _COMPLETEP_RE.search(d.name)
        if not m:
            continue
        # Skip accidental baseline-named dirs if any under CompleteP root
        if "baseline" in d.name:
            continue
        rows.append(_extract_one(d, "CompleteP", int(m.group(1)), float(m.group(2))))

    return rows


def _x_for_beta(beta: float) -> float:
    return BETA0_X if beta == 0.0 else beta


def _panel_ylim(
    rows: list[dict], key: str, pad_frac: float = 0.06
) -> tuple[float, float]:
    vals = []
    for r in rows:
        v = r[key]
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return 0.0, 1.0
    lo, hi = min(vals), max(vals)
    span = hi - lo
    pad = pad_frac * span if span > 0 else 0.05
    return lo - pad, hi + pad


def plot(rows: list[dict]) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), constrained_layout=True)
    panels = [
        (axes[0], "eval_reward", "Final eval reward"),
        (axes[1], "train_reward", "Final train reward"),
    ]

    # Legend order: solid (d=1024) then dashed (d=512) within each family
    series = [
        ("Baseline", 1024),
        ("Baseline", 512),
        ("CompleteP", 1024),
        ("CompleteP", 512),
    ]

    for ax, key, ylabel in panels:
        for family, dim in series:
            sub = sorted(
                [r for r in rows if r["family"] == family and r["dim"] == dim],
                key=lambda r: r["beta"],
            )
            if not sub:
                continue
            xs = [_x_for_beta(r["beta"]) for r in sub]
            ys = [r[key] for r in sub]
            color = COLORS[family]
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=LINESTYLES[dim],
                marker="o",
                markersize=6.5,
                linewidth=1.7,
                markerfacecolor="white",
                markeredgewidth=1.5,
                markeredgecolor=color,
                label=f"{family} d={dim}",
            )

        ax.set_xscale("log")
        ax.set_xlabel(r"$\beta$")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*_panel_ylim(rows, key))
        ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.65)
        ax.legend(frameon=False, loc="best")
        ax.axvline(BETA0_X, color="#bbbbbb", linewidth=0.8, linestyle=":", zorder=0)

        tick_betas = [BETA0_X, 1e-3, 1e-2, 1e-1, 1.0]
        tick_labels = ["0", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$1$"]
        ax.set_xticks(tick_betas)
        ax.set_xticklabels(tick_labels)

    fig.suptitle(
        r"GRPO GSM8K $\beta$ transfer: Baseline vs CompleteP",
        fontsize=13,
        fontweight="semibold",
        y=1.06,
    )
    fig.text(
        0.5,
        1.005,
        r"lr$=$1e-4 · eps$=$0.2 · solid=$d$=1024 · dashed=$d$=512 · color=family · own y per panel",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#444444",
    )

    png_path = OUT_PRIMARY.with_suffix(".png")
    pdf_path = OUT_PRIMARY.with_suffix(".pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)

    # Keep open-tab / legacy filename in sync
    for suf in (".png", ".pdf"):
        shutil.copy2(OUT_PRIMARY.with_suffix(suf), OUT_LEGACY.with_suffix(suf))

    return png_path, pdf_path


def main() -> None:
    rows = extract_runs()
    if not rows:
        raise SystemExit("No GRPO β-transfer runs found.")

    print("family      dim      beta         eval        train")
    print("-" * 56)
    for r in sorted(rows, key=lambda x: (x["family"], x["dim"], x["beta"])):
        print(
            f"{r['family']:<10} {r['dim']:4d} {r['beta']:10g} "
            f"{r['eval_reward']:10.6f} {r['train_reward']:10.6f}"
        )
    n_b = sum(1 for r in rows if r["family"] == "Baseline")
    n_c = sum(1 for r in rows if r["family"] == "CompleteP")
    print(f"\nTotal: {len(rows)}  (Baseline={n_b}, CompleteP={n_c})")
    for key, label in (
        ("eval_reward", "eval"),
        ("train_reward", "train"),
    ):
        lo, hi = _panel_ylim(rows, key)
        print(f"{label} y-limits: [{lo:.4f}, {hi:.4f}]")

    png_path, pdf_path = plot(rows)
    for p in (
        png_path,
        pdf_path,
        OUT_LEGACY.with_suffix(".png"),
        OUT_LEGACY.with_suffix(".pdf"),
    ):
        size = p.stat().st_size
        print(f"{p}  ({size:,} bytes / {size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
