#!/bin/bash
# Full pull of Lambda graph-construction v1lam panels (a)+(b):
#   run dirs, parquets, best/last checkpoints → Della SWEEP + organized tree.
#
#   bash scans/jets/pull_graph_construction_v1lam_full.sh
#
set -euo pipefail

HOST="${LAMBDA_HOST:-ubuntu@193.122.155.21}"
CM="${LAMBDA_CM:-$HOME/.cache/cm-lambda-193}"
RSH=(ssh -o BatchMode=yes -o ConnectTimeout=30)
if [[ -S "$CM" ]]; then
  RSH+=(-S "$CM")
fi

LOCAL_SWEEP="${LOCAL_SWEEP:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/jetclass_output/adam/sweep_lr}"
ORG_ROOT="${ORG_ROOT:-/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/graph_construction_lambda_v1lam}"
PAPER_ORG="${PAPER_ORG:-$HOME/coupled-particle-edge-networks/paper/data/graph_construction_lambda_v1lam}"

mkdir -p "$LOCAL_SWEEP" \
  "$ORG_ROOT/k_R0p1"/{runs,parquets,checkpoints,logs} \
  "$ORG_ROOT/R_k6"/{runs,parquets,checkpoints,logs} \
  "$PAPER_ORG/k_R0p1"/{summaries,parquets,checkpoints} \
  "$PAPER_ORG/R_k6"/{summaries,parquets,checkpoints}

pull_one() {
  local remote_root="$1" glob="$2" label="$3"
  echo "==== pull $label ===="
  echo "rsync $HOST:$remote_root → $LOCAL_SWEEP ($glob)"
  rsync -aP -e "${RSH[*]}" \
    --include="$glob" --include="${glob}/**" --exclude='*' \
    "$HOST:$remote_root" "$LOCAL_SWEEP/"
}

# Panel (a)
pull_one "~/runs/graph_k/jetclass_output/adam/sweep_lr/" \
  "*graph-k-b64-g8-s1000-R0p1*v1lam*" "panel_a_k"

# Panel (b)
pull_one "~/runs/graph_R/jetclass_output/adam/sweep_lr/" \
  "*graph-R-b64-g8-s1000-k6*v1lam*" "panel_b_R"

# Also pull stdout logs
echo "==== pull logs ===="
rsync -aP -e "${RSH[*]}" \
  "$HOST:~/logs/graph_k/" "$ORG_ROOT/k_R0p1/logs/" || true
rsync -aP -e "${RSH[*]}" \
  "$HOST:~/logs/graph_R/" "$ORG_ROOT/R_k6/logs/" || true

echo "==== organize into $ORG_ROOT ===="
python3 - <<'PY'
import csv, shutil
from pathlib import Path

SWEEP = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/jetclass_output/adam/sweep_lr")
ORG = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/graph_construction_lambda_v1lam")
PAPER = Path.home() / "coupled-particle-edge-networks/paper/data/graph_construction_lambda_v1lam"

def organize(glob_pat: str, sub: str):
    rows = []
    dest_runs = ORG / sub / "runs"
    dest_pq = ORG / sub / "parquets"
    dest_ckpt = ORG / sub / "checkpoints"
    paper_pq = PAPER / sub / "parquets"
    paper_ckpt = PAPER / sub / "checkpoints"
    for d in sorted(SWEEP.glob(glob_pat)):
        if not d.is_dir():
            continue
        # symlink run dir into org tree
        link = dest_runs / d.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(d)
        pq = Path(str(d) + ".parquet")
        if pq.is_file():
            pq_link = dest_pq / pq.name
            if pq_link.is_symlink() or pq_link.exists():
                pq_link.unlink()
            pq_link.symlink_to(pq)
            paper_pq_link = paper_pq / pq.name
            if paper_pq_link.is_symlink() or paper_pq_link.exists():
                paper_pq_link.unlink()
            paper_pq_link.symlink_to(pq)
        for ck in ("best.ckpt", "last.ckpt"):
            src = d / ck
            if src.is_file():
                ck_link = dest_ckpt / f"{d.name}__{ck}"
                if ck_link.is_symlink() or ck_link.exists():
                    ck_link.unlink()
                ck_link.symlink_to(src)
                pck = paper_ckpt / f"{d.name}__{ck}"
                if pck.is_symlink() or pck.exists():
                    pck.unlink()
                pck.symlink_to(src)
        rows.append({
            "run": d.name,
            "run_dir": str(d),
            "parquet": str(pq) if pq.is_file() else "",
            "best_ckpt": str(d / "best.ckpt") if (d / "best.ckpt").is_file() else "",
            "last_ckpt": str(d / "last.ckpt") if (d / "last.ckpt").is_file() else "",
        })
    man = PAPER / sub / "summaries" / "artifact_manifest.csv"
    man.parent.mkdir(parents=True, exist_ok=True)
    with man.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "run_dir", "parquet", "best_ckpt", "last_ckpt"])
        w.writeheader()
        w.writerows(rows)
    print(f"{sub}: {len(rows)} runs → {man}")

organize("*graph-k-b64-g8-s1000-R0p1*v1lam*", "k_R0p1")
organize("*graph-R-b64-g8-s1000-k6*v1lam*", "R_k6")
PY

echo "==== counts ===="
echo -n "k runs: "; ls "$ORG_ROOT/k_R0p1/runs" 2>/dev/null | wc -l
echo -n "k ckpts: "; ls "$ORG_ROOT/k_R0p1/checkpoints" 2>/dev/null | wc -l
echo -n "R runs: "; ls "$ORG_ROOT/R_k6/runs" 2>/dev/null | wc -l
echo -n "R ckpts: "; ls "$ORG_ROOT/R_k6/checkpoints" 2>/dev/null | wc -l
echo done
