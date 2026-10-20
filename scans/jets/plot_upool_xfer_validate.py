#!/usr/bin/env python3
"""Debug plot: uniform-pool η₀×D transfer validation (1000-step gputest).

Collects val_roc_auc from run parquets / logs under the upool-xfer tag and
writes a simple η₀→ROC curve for D=256 vs D=512.

  python scans/jets/plot_upool_xfer_validate.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law")
LOG_DIR = ROOT / "logs"
FIG_DIR = REPO / "paper" / "figures"
DATA_DIR = REPO / "paper" / "data"
TAG_SUB = "upool-xfer-b96-g4-s1000-dh32-k10-R0p1"
WIDTHS = (256, 512)
ETAS = (0.05, 0.1, 0.25, 0.5, 1.0)


def _eta_tag(eta: float) -> str:
    return f"{eta:g}".replace(".", "p")


def _from_parquet(width: int, eta: float) -> dict | None:
    et = _eta_tag(eta)
    paths = sorted(ROOT.glob(f"*_{width}_{et}_*{TAG_SUB}*.parquet"))
    best = None
    for p in paths:
        try:
            df = pd.read_parquet(p, columns=["global_step", "val_roc_auc", "val_acc"])
        except Exception:
            continue
        sub = df.dropna(subset=["val_roc_auc"])
        if sub.empty:
            continue
        row = sub.loc[sub["val_roc_auc"].idxmax()]
        cand = {
            "width": width,
            "eta_0": eta,
            "step": int(row["global_step"]),
            "val_roc_auc": float(row["val_roc_auc"]),
            "val_acc": float(row["val_acc"]) if "val_acc" in row and pd.notna(row["val_acc"]) else float("nan"),
            "source": p.name,
        }
        if best is None or cand["val_roc_auc"] > best["val_roc_auc"]:
            best = cand
    return best


def _from_logs(width: int, eta: float) -> dict | None:
    """Fallback: parse jc_upool_xfer_*.out for D/η matching lines."""
    et = _eta_tag(eta)
    best = None
    for p in sorted(LOG_DIR.glob("jc_upool_xfer_*.out")):
        text = p.read_text(errors="ignore")
        if f"D={width}" not in text and f"--width {width}" not in text:
            continue
        if f"eta0={eta:g}" not in text and f"etas {eta:g}" not in text and f"_{width}_{et}_" not in text:
            # still allow if parquet path printed
            if f"_{width}_{et}_" not in text and f"eta0={eta}" not in text:
                continue
        for m in re.finditer(
            r"\[val\].*?step=(\d+).*?val_roc_auc=([0-9.]+)", text
        ):
            step, roc = int(m.group(1)), float(m.group(2))
            cand = {
                "width": width,
                "eta_0": eta,
                "step": step,
                "val_roc_auc": roc,
                "val_acc": float("nan"),
                "source": p.name,
            }
            if best is None or roc > best["val_roc_auc"]:
                best = cand
    return best


def collect() -> pd.DataFrame:
    rows = []
    for w in WIDTHS:
        for eta in ETAS:
            row = _from_parquet(w, eta) or _from_logs(w, eta)
            if row is None:
                rows.append(
                    {
                        "width": w,
                        "eta_0": eta,
                        "step": None,
                        "val_roc_auc": float("nan"),
                        "val_acc": float("nan"),
                        "source": "",
                        "status": "missing",
                    }
                )
            else:
                rows.append({**row, "status": "ok"})
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, stem: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    markers = {256: "o", 512: "s"}
    for w in WIDTHS:
        sub = df[df.width == w].sort_values("eta_0")
        ok = sub.dropna(subset=["val_roc_auc"])
        if ok.empty:
            continue
        ax.plot(
            ok["eta_0"],
            ok["val_roc_auc"],
            marker=markers.get(int(w), "o"),
            lw=1.2,
            ms=5,
            label=rf"$D={w}$",
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"$\eta_0$ (Adam $\eta=\eta_0/\sqrt{D}$)")
    ax.set_ylabel("best val ROC-AUC @ ≤1000 steps")
    ax.set_title("uniform-pool transfer validation (debug)")
    ax.legend(frameon=False)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    out = FIG_DIR / f"{stem}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stem", default="fig_upool_xfer_validate_debug")
    args = ap.parse_args()
    df = collect()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv = DATA_DIR / "upool_xfer_validate.csv"
    df.to_csv(csv, index=False)
    print(f"wrote {csv}")
    print(df.to_string(index=False))
    out = plot(df, args.stem)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
