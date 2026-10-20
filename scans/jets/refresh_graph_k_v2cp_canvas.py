#!/usr/bin/env python3
"""Regenerate canvases/graph-k-completep-v2-live.canvas.tsx from Slurm + parquets."""
from __future__ import annotations

import json
import math
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

JOB = "15281965"
KS = (2, 3, 4, 5, 6, 7, 8, 10)
N_SHARD = 3
N_INIT = 2
N_PER_K = N_SHARD * N_INIT
N_TOT = len(KS) * N_PER_K  # 48
MAX_STEPS = 2500
ETA0 = 0.1
BATCH = 128

SWEEP = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "jetclass_output/adam/sweep_lr"
)
LOGDIR = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs")
OUT = Path(
    "/home/jdezoort/.cursor/projects/home-jdezoort/canvases/"
    "graph-k-completep-v2-live.canvas.tsx"
)

# Archived v1lam mean ROC @ k (for overlay) — from live canvas snapshot
V1LAM_MEAN = {
    2: 0.9376,
    3: 0.9382,
    4: 0.9399,
    5: 0.9389,
    6: 0.9398,
    7: 0.9398,
    8: 0.9400,
    10: 0.9398,
}


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, timeout=45)
    except Exception:
        return ""


def js(obj) -> str:
    return json.dumps(obj)


def task_meta(task: int) -> dict:
    k_idx = task // N_PER_K
    rem = task % N_PER_K
    shard = rem // N_INIT
    seed = rem % N_INIT
    return {"task": task, "k": KS[k_idx], "shard": shard, "seed": seed}


def main() -> None:
    acct: dict[str, dict] = {}
    for line in run(
        [
            "sacct",
            "-j",
            JOB,
            "-n",
            "-X",
            "--format=JobID%18,State%14,Elapsed",
        ]
    ).splitlines():
        p = line.split()
        if len(p) >= 3:
            acct[p[0]] = {"state": p[1], "elapsed": p[2]}

    queue: dict[str, dict] = {}
    for line in run(
        ["squeue", "-j", JOB, "-h", "-o", "%i %T %M %R"]
    ).splitlines():
        parts = line.split(None, 3)
        if len(parts) >= 3:
            queue[parts[0]] = {
                "state": parts[1],
                "time": parts[2],
                "reason": parts[3] if len(parts) > 3 else "",
            }

    # index parquets by (k, shard, seed)
    pq_map: dict[tuple[int, int, int], Path] = {}
    for p in SWEEP.glob("*graph-k-b128-g1-s2500-R0p1*v2cp.parquet"):
        m = re.search(r"knn(\d+)", p.name)
        sm = re.search(r"shard(\d+)-seed(\d+)-v2cp", p.name)
        if not m or not sm:
            continue
        key = (int(m.group(1)), int(sm.group(1)), int(sm.group(2)))
        prev = pq_map.get(key)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            pq_map[key] = p

    cells = []
    for t in range(N_TOT):
        meta = task_meta(t)
        jid = f"{JOB}_{t}"
        q = queue.get(jid)
        a = acct.get(jid)
        if q:
            state = q["state"].lower()
            elapsed = q["time"]
            reason = q.get("reason", "")
        elif a:
            st = a["state"]
            if st.startswith("COMPLETED"):
                state = "done"
            elif "FAIL" in st or "CANCEL" in st:
                state = "failed" if "FAIL" in st else "cancelled"
            else:
                state = st.lower()
            elapsed = a["elapsed"]
            reason = ""
        else:
            # pending in array range not yet materialized
            state = "pending"
            elapsed = ""
            reason = ""

        if state == "completed":
            state = "done"
        if state == "running":
            pass
        elif state not in ("done", "failed", "cancelled", "pending"):
            if "pending" in state:
                state = "pending"

        step = None
        auc = None
        train = None
        # live step from log
        log = LOGDIR / f"jc_gk_v2cp_{JOB}_{t}.out"
        if log.exists():
            text = log.read_text(errors="ignore")
            steps = re.findall(r"global_step=(\d+)", text)
            if steps:
                step = int(steps[-1])
            if "done k=" in text and state == "running":
                state = "done"

        key = (meta["k"], meta["shard"], meta["seed"])
        path = pq_map.get(key)
        if path is not None and path.is_file():
            try:
                df = pq.read_table(
                    path, columns=["global_step", "train_loss", "val_roc_auc"]
                ).to_pandas()
                if len(df):
                    step = max(step or 0, int(df.global_step.max()))
                    vd = df[["global_step", "val_roc_auc"]].dropna()
                    if len(vd):
                        r = vd.sort_values("global_step").iloc[-1]
                        auc = float(r.val_roc_auc)
                    td = df[["global_step", "train_loss"]].dropna()
                    if len(td):
                        train = float(
                            td.sort_values("global_step").iloc[-1].train_loss
                        )
                    if step >= MAX_STEPS - 5:
                        state = "done"
            except Exception:
                pass

        cells.append(
            {
                **meta,
                "status": state,
                "step": step,
                "elapsed": elapsed,
                "reason": reason.replace("(", "").replace(")", "") if reason else "",
                "auc": auc,
                "train": train,
            }
        )

    n_done = sum(1 for c in cells if c["status"] == "done")
    n_run = sum(1 for c in cells if c["status"] == "running")
    n_fail = sum(1 for c in cells if c["status"] == "failed")
    n_pend = N_TOT - n_done - n_run - n_fail

    # per-k means (done cells with auc)
    k_rows = []
    k_mean = []
    k_cats = [str(k) for k in KS]
    for k in KS:
        vals = [
            c["auc"]
            for c in cells
            if c["k"] == k and c["auc"] is not None and c["status"] == "done"
        ]
        n_k_done = sum(1 for c in cells if c["k"] == k and c["status"] == "done")
        mean = statistics.mean(vals) if vals else None
        std = statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else None)
        k_rows.append(
            {
                "k": k,
                "n_done": n_k_done,
                "n_total": N_PER_K,
                "mean": mean,
                "std": std,
            }
        )
        k_mean.append(mean)

    v1_series = [V1LAM_MEAN.get(k) for k in KS]

    best = None
    for c in cells:
        if c["auc"] is not None and c["status"] == "done":
            if best is None or c["auc"] > best["auc"]:
                best = {
                    "auc": c["auc"],
                    "k": c["k"],
                    "shard": c["shard"],
                    "seed": c["seed"],
                }

    # cell table sorted by k, shard, seed
    cells_sorted = sorted(cells, key=lambda c: (c["k"], c["shard"], c["seed"]))
    table = []
    tones = []
    for c in cells_sorted:
        step_s = (
            f"{c['step']}/{MAX_STEPS}" if c["step"] is not None else f"0/{MAX_STEPS}"
        )
        table.append(
            [
                str(c["k"]),
                str(c["shard"]),
                str(c["seed"]),
                c["status"],
                step_s,
                "—" if c["train"] is None else f"{c['train']:.3f}",
                "—" if c["auc"] is None else f"{c['auc']:.4f}",
            ]
        )
        tones.append(
            {
                "done": "success",
                "running": "info",
                "pending": "neutral",
                "failed": "danger",
                "cancelled": "warning",
            }.get(c["status"], "neutral")
        )

    updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_js = "null" if best is None else js(best)

    # chart series: NaN for missing means
    def series_data(vals):
        return [v if v is not None else None for v in vals]

    canvas = f"""import {{
  Callout,
  Grid,
  H1,
  H2,
  LineChart,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
}} from "cursor/canvas";

const UPDATED = {js(updated)};
const JOB = {js(JOB)};
const RECIPE =
  "L4/D256 · η₀=0.1 · B=128×1 · 2500 steps · R★=0.1 · CompleteP · v2cp";
const N_DONE = {n_done};
const N_RUN = {n_run};
const N_PEND = {n_pend};
const N_FAIL = {n_fail};
const N_TOT = {N_TOT};
const BEST = {best_js};
const K_CATS: string[] = {js(k_cats)};
const K_MEAN = {js(series_data(k_mean))};
const V1_MEAN = {js(series_data(v1_series))};
const K_ROWS = {js(k_rows)};
const TABLE = {js(table)};
const TONE = {js(tones)};

function fmt(x: number | null | undefined, d = 4): string {{
  return x == null || Number.isNaN(Number(x)) ? "—" : Number(x).toFixed(d);
}}

export default function GraphKCompletePV2Live() {{
  const series = [
    {{
      name: "v2cp mean val ROC",
      data: K_MEAN.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "accent" as const,
    }},
    {{
      name: "v1lam mean (archived)",
      data: V1_MEAN.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "neutral" as const,
    }},
  ];

  return (
    <Stack gap={{20}}>
      <Stack gap={{6}}>
        <H1>Graph construction panel (a) — CompleteP v2</H1>
        <Text tone="secondary">{{RECIPE}}</Text>
        <Text tone="secondary">
          Job {{JOB}} · Della 1×A100 · updated {{UPDATED}}
        </Text>
        <Row gap={{8}} wrap>
          <Pill tone={{N_DONE === N_TOT ? "success" : "info"}}>
            {{N_DONE}}/{{N_TOT}} done
          </Pill>
          <Pill tone="info">{{N_RUN}} running</Pill>
          <Pill tone="neutral">{{N_PEND}} pending</Pill>
          {{N_FAIL > 0 ? <Pill tone="danger">{{N_FAIL}} failed</Pill> : null}}
        </Row>
      </Stack>

      <Grid columns={{4}} gap={{12}}>
        <Stat
          value={{`${{N_DONE}}/${{N_TOT}}`}}
          label="cells done"
          tone={{N_DONE === N_TOT ? "success" : "info"}}
        />
        <Stat value={{String(N_RUN)}} label="running" />
        <Stat value={{String(N_PEND)}} label="pending" />
        <Stat
          value={{BEST == null ? "—" : BEST.auc.toFixed(4)}}
          label={{
            BEST == null
              ? "Best val AUC"
              : `Best · k=${{BEST.k}} s${{BEST.shard}} seed${{BEST.seed}}`
          }}
        />
      </Grid>

      {{N_DONE < N_TOT ? (
        <Callout tone="info" title="In progress">
          Fresh CompleteP re-run of panel (a). Overlay shows archived Lambda
          v1lam means (η₀=0.25, B=512, 1000 steps) for reference — not a matched
          recipe.
        </Callout>
      ) : (
        <Callout tone="success" title="Grid complete">
          All 48 cells finished 2500 steps.
        </Callout>
      )}}

      <Stack gap={{6}}>
        <H2>Mean val ROC vs k @ R★=0.1</H2>
        <LineChart
          categories={{K_CATS}}
          series={{series as any}}
          height={{240}}
          beginAtZero={{false}}
        />
        <Text tone="secondary">
          Source: v2cp parquets (done cells only) · v1lam overlay from archived
          campaign
        </Text>
      </Stack>

      <Stack gap={{8}}>
        <H2>Per-k summary</H2>
        <Table
          headers={{["k", "done/6", "mean ROC", "std"]}}
          rows={{K_ROWS.map((m: any) => [
            String(m.k),
            `${{m.n_done}}/${{m.n_total}}`,
            fmt(m.mean),
            fmt(m.std),
          ])}}
        />
      </Stack>

      <Stack gap={{8}}>
        <H2>Cells</H2>
        <Table
          headers={{["k", "shard", "seed", "status", "step", "train", "AUC"]}}
          rows={{TABLE}}
          rowTone={{TONE as any}}
        />
      </Stack>
    </Stack>
  );
}}
"""
    OUT.write_text(canvas)
    print(f"wrote {OUT} done={n_done} run={n_run} pend={n_pend} fail={n_fail}")


if __name__ == "__main__":
    main()
