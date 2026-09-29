"""Collect JetClass metrics used by Nature-style paper figures."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ADAM_SWEEP = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/jetclass_output/adam/sweep_lr")
JET_RUNS = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_runs/jetclass_output/adam/sweep_lr")
LOGS = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs")
OUT = Path(__file__).resolve().parent

# Canonical M11 vs edges (L=3) — archived 2026-09-28; not matched to B=64 recipe.
# See paper/archive/graph_construction_2026-09-28/data/edges_vs_m11_1m.csv
M11_VS_EDGES: list = []


def collect_edges_vs_m11() -> pd.DataFrame:
    """Placeholder: old L=3 edges/M11 table archived (batch not matched to B=64 recipe)."""
    print("[warn] edges_vs_m11: rematch at B=64 not started; writing empty CSV")
    return pd.DataFrame(
        columns=["model", "eta0", "val_acc", "val_roc_auc", "val_bg_rejection"]
    )


def collect_graph_ablation() -> pd.DataFrame:
    """Matched B=64 / 15625-step / part-int graph ablations (ablate-Rmig-b64)."""
    rows = []
    tag = "ablate-Rmig-b64"

    def add(pqf: Path, family: str, variant: str, panel: str, **extra) -> None:
        df = pq.read_table(pqf).to_pandas()
        v = df.dropna(subset=["val_roc_auc"]).sort_values("global_step")
        if len(v) == 0:
            return
        r = v.iloc[-1]
        rows.append(dict(
            panel=panel, family=family, variant=variant,
            global_step=int(r.global_step), val_acc=float(r.val_acc),
            val_roc_auc=float(r.val_roc_auc),
            val_bg_rejection=float(r.val_bg_rejection), path=str(pqf),
            **extra,
        ))

    # kNN + no-star at fixed R★=0.2
    panel_k = [
        ("connectivity", "kNN $k=2$", "knn2-D1M-s15p625k", False),
        ("connectivity", "kNN $k=4$", "knn4-D1M-s15p625k", False),
        ("connectivity", "kNN $k=6$", "knn6-D1M-s15p625k", False),
        ("connectivity", "kNN $k=8$", "knn8-D1M-s15p625k", False),
        ("connectivity", "kNN $k=10$", "knn10-D1M-s15p625k", False),
        ("star graph", "no star", "knn6-nostar-D1M-s15p625k", True),
    ]
    for family, variant, token, want_nostar in panel_k:
        hits = sorted(ADAM_SWEEP.glob(
            f"capen-llama-att_4_256_*gstar0p2_stream-kin7-partint*{token}*seed0-{tag}.parquet"
        ))
        if want_nostar:
            hits = [h for h in hits if "nostar" in h.name]
        else:
            hits = [h for h in hits if "nostar" not in h.name]
        if not hits:
            print(f"[warn] missing {variant}"); continue
        add(hits[0], family, variant, "graph_matched")

    # Star radius at fixed k=6 + part-int
    star_r = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    for r_star in star_r:
        r_tag = f"{r_star:g}".replace(".", "p")
        hits = sorted(ADAM_SWEEP.glob(
            f"capen-llama-att_4_256_*gstar{r_tag}_stream-kin7-partint*knn6-D1M-s15p625k*seed0-{tag}.parquet"
        ))
        hits = [h for h in hits if "nostar" not in h.name and "knn10" not in h.name]
        if not hits:
            print(f"[warn] missing R★={r_star}"); continue
        add(hits[0], "star radius", f"R={r_star:g}", "star_radius", r_star=r_star)

    return pd.DataFrame(rows)


# Width → attention heads used in the L4 Adam transfer sweep (d_head=32).
TRANSFER_HEADS = {256: 8, 384: 12, 512: 16, 768: 24}


def _row_at_step(df: pd.DataFrame, step: int, require: str) -> pd.Series:
    v = df.dropna(subset=[require]).sort_values("global_step")
    return v.iloc[(v.global_step - step).abs().argmin()]


def collect_transfer_adam_45k() -> pd.DataFrame:
    rows = []

    def add_parquet(pqf: Path, width: int) -> None:
        df = pq.read_table(pqf).to_pandas()
        r_val = _row_at_step(df, 45000, "val_roc_auc")
        r_tr = _row_at_step(df, 45000, "train_loss")
        rows.append(dict(
            width=width, heads=TRANSFER_HEADS[width],
            eta0=float(r_val.eta_0), global_step=int(r_val.global_step),
            train_loss=float(r_tr.train_loss),
            val_acc=float(r_val.val_acc), val_roc_auc=float(r_val.val_roc_auc),
            val_bg_rejection=float(r_val.val_bg_rejection),
            val_bg_rejection_0p3=(
                float(r_val.val_bg_rejection_0p3)
                if "val_bg_rejection_0p3" in r_val.index else float("nan")
            ),
            source="parquet", path=str(pqf),
        ))

    for pqf in sorted(ADAM_SWEEP.glob("capen-llama-att_4_256_*Dfull-s45k*L4D256-gputest*.parquet")):
        if "corr" in pqf.name:
            continue
        add_parquet(pqf, 256)
    for width, pat in [
        (512, "capen-llama-att_4_512_0p25_*L4D512-gputest*.parquet"),
        (768, "capen-llama-att_4_768_0p25_*L4D768-gputest*.parquet"),
    ]:
        hits = sorted(ADAM_SWEEP.glob(pat))
        if not hits:
            print(f"[warn] missing D={width}"); continue
        add_parquet(hits[0], width)

    # D=384: train/val from logs; rejection was never persisted (see README).
    log_map = {
        "0.01": "jc_L4D384_gputest_14262377.out", "0.025": "jc_L4D384_gputest_14262777.out",
        "0.05": "jc_L4D384_gputest_14243368.out", "0.1": "jc_L4D384_gputest_14244173.out",
        "0.25": "jc_L4D384_gputest_14243369.out", "0.5": "jc_L4D384_gputest_14264044.out",
        "1": "jc_L4D384_gputest_14289134.out", "2.5": "jc_L4D384_gputest_14289135.out",
    }
    val_re = re.compile(
        r"\[val\].*?step=(\d+) val_loss=([0-9.]+) val_acc=([0-9.]+) val_roc_auc=([0-9.]+)"
    )
    train_re = re.compile(r"\[train\].*?step=(\d+).*?train_loss=([0-9.]+)")
    for eta_s, fname in log_map.items():
        path = LOGS / fname
        if not path.exists():
            print(f"[warn] missing log {fname}"); continue
        text = path.read_text(errors="ignore")
        vals = [(int(s), float(acc), float(roc)) for s, _, acc, roc in val_re.findall(text)]
        trains = [(int(s), float(loss)) for s, loss in train_re.findall(text)]
        if not vals or not trains:
            continue
        s, acc, roc = min(vals, key=lambda t: abs(t[0] - 45000))
        _, train_loss = min(trains, key=lambda t: abs(t[0] - 45000))
        rows.append(dict(
            width=384, heads=TRANSFER_HEADS[384],
            eta0=float(eta_s), global_step=s, train_loss=train_loss,
            val_acc=acc, val_roc_auc=roc,
            val_bg_rejection=float("nan"), val_bg_rejection_0p3=float("nan"),
            source="log", path=str(path),
        ))
    return pd.DataFrame(rows).sort_values(["width", "eta0"]).reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    edges = collect_edges_vs_m11()
    graph = collect_graph_ablation()
    xfer = collect_transfer_adam_45k()
    edges.to_csv(OUT / "edges_vs_m11_1m.csv", index=False)
    graph.to_csv(OUT / "graph_ablation_1m.csv", index=False)
    xfer.to_csv(OUT / "transfer_adam_45k.csv", index=False)
    print("edges_vs_m11", len(edges))
    print("graph_ablation", len(graph))
    print(graph[["panel", "family", "variant", "val_roc_auc"]].to_string(index=False))
    print("transfer", len(xfer))
    print(xfer[["width", "heads", "eta0", "train_loss", "val_roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
