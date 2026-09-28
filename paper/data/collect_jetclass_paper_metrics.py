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

# Canonical M11 vs edges (L=3, D=256, 1M, 1 epoch) from paper viz / jetclass_runs.
M11_VS_EDGES = [
    # model, eta0, val_acc, val_roc_auc, val_bg_rejection
    ("edges", 0.01, 0.565920, 0.906638, 111.381),
    ("edges", 0.05, 0.598080, 0.921038, 124.559),
    ("edges", 0.10, 0.610480, 0.925185, 121.763),
    ("edges", 0.25, 0.610360, 0.926301, 120.656),
    ("edges", 0.50, 0.615240, 0.926166, 130.398),
    ("edges", 1.00, 0.601840, 0.923868, 130.405),
    ("edges", 2.50, 0.557400, 0.906729, 119.866),
    ("m11", 0.01, 0.554880, 0.902637, 90.730),
    ("m11", 0.05, 0.584600, 0.915338, 105.613),
    ("m11", 0.10, 0.595120, 0.919282, 103.411),
    ("m11", 0.25, 0.590920, 0.920145, 109.923),
    ("m11", 0.50, 0.598880, 0.921783, 91.770),
    ("m11", 1.00, 0.575600, 0.913964, 94.475),
    ("m11", 2.50, 0.531600, 0.897734, 87.788),
]


def collect_edges_vs_m11() -> pd.DataFrame:
    rows = [
        dict(model=m, eta0=e, val_acc=a, val_roc_auc=r, val_bg_rejection=j)
        for m, e, a, r, j in M11_VS_EDGES
    ]
    return pd.DataFrame(rows)


def collect_graph_ablation() -> pd.DataFrame:
    """Graph-construction ablations only (no RoPE / pooling / pT-cut)."""
    rows = []

    def add(pqf: Path, family: str, variant: str, panel: str) -> None:
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
        ))

    # Matched 2-epoch B=256: kNN + nostar + default knn6
    panel_b = [
        ("connectivity", "kNN $k=2$", "knn2-D1M-s3p907k"),
        ("connectivity", "kNN $k=4$", "knn4-D1M-s3p907k"),
        ("connectivity", "kNN $k=6$", "knn6-D1M-s3p907k"),
        ("connectivity", "kNN $k=8$", "knn8-D1M-s3p907k"),
        ("star graph", "no star", "knn6-nostar-D1M-s3p907k"),
    ]
    for family, variant, token in panel_b:
        hits = sorted(ADAM_SWEEP.glob(f"capen-llama-att_4_256_*{token}*ablate*.parquet"))
        if "k=6" in variant:
            hits = [h for h in hits if "nostar" not in h.name]
        if not hits:
            print(f"[warn] missing {variant}"); continue
        add(hits[0], family, variant, "graph_matched")

    # Matched 1-epoch star-radius sweep (no part-int; same otherwise).
    # R★=0.2 is the "no pairwise part-int" run at gstar0p2.
    star_r = [
        (0.10, "gstar0p1_stream-kin7-p80-D1M-s1p954k-seed0-ablate"),
        (0.15, "gstar0p15_stream-kin7-p80-D1M-s1p954k-seed0-ablate"),
        (0.20, "gstar0p2_stream-kin7-p80-D1M-s1p954k-seed0-ablate"),
        (0.30, "gstar0p3_stream-kin7-p80-D1M-s1p954k-seed0-ablate"),
    ]
    for r_star, token in star_r:
        hits = sorted(ADAM_SWEEP.glob(f"capen-llama-att_4_256_*{token}*.parquet"))
        hits = [h for h in hits
                if "unipool" not in h.name and "logdot" not in h.name
                and "fullm22" not in h.name and "partint" not in h.name
                and "rope" not in h.name]
        if not hits:
            print(f"[warn] missing R★={r_star}"); continue
        df = pq.read_table(hits[0]).to_pandas()
        v = df.dropna(subset=["val_roc_auc"]).sort_values("global_step")
        if len(v) == 0:
            continue
        r = v.iloc[-1]
        rows.append(dict(
            panel="star_radius", family="star radius",
            variant=f"R={r_star:g}", r_star=r_star,
            global_step=int(r.global_step), val_acc=float(r.val_acc),
            val_roc_auc=float(r.val_roc_auc),
            val_bg_rejection=float(r.val_bg_rejection), path=str(hits[0]),
        ))

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
