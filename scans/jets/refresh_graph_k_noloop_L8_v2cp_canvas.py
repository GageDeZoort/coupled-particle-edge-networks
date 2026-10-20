#!/usr/bin/env python3
"""Regenerate canvases/graph-k-noloop-L8D1024-completep-v2-live.canvas.tsx."""
from __future__ import annotations

import json
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

JOB = ""
KS = (1, 2, 3, 4, 5, 6, 7, 8, 10)
N_SHARD = 3
N_INIT = 2
N_PER_K = N_SHARD * N_INIT
N_TOT = len(KS) * N_PER_K
MAX_STEPS = 2500

SWEEP = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "jetclass_output/adam/sweep_lr"
)
LOGDIR = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs")
OUT = Path(
    "/home/jdezoort/.cursor/projects/home-jdezoort/canvases/"
    "graph-k-noloop-L8D1024-completep-v2-live.canvas.tsx"
)
JID_FILE = Path(__file__).with_name("graph_k_noloop_L8_v2cp_jobid.txt")

# L4 noloop means (overlay)
L4_NOLOOP_MEAN = {
    1: 0.9368,
    2: 0.9375,
    3: 0.9380,
    4: 0.9384,
    5: 0.9385,
    6: 0.9389,
    7: 0.9391,
    8: 0.9389,
    10: 0.9387,
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
    return {
        "task": task,
        "k": KS[k_idx],
        "shard": rem // N_INIT,
        "seed": rem % N_INIT,
    }


def main() -> None:
    job = JOB
    if JID_FILE.is_file():
        job = JID_FILE.read_text().strip() or job

    acct: dict[str, dict] = {}
    if job:
        for line in run(
            ["sacct", "-j", job, "-n", "-X", "--format=JobID%24,State%14,Elapsed"]
        ).splitlines():
            p = line.split()
            if len(p) >= 3:
                acct[p[0]] = {"state": p[1], "elapsed": p[2]}

    queue: dict[int, dict] = {}
    array_pending_reason = ""
    if job:
        for line in run(["squeue", "-j", job, "-h", "-o", "%i %T %M %R"]).splitlines():
            parts = line.split(None, 3)
            if len(parts) < 2:
                continue
            jid, state = parts[0], parts[1].lower()
            elapsed = parts[2] if len(parts) > 2 else ""
            reason = parts[3] if len(parts) > 3 else ""
            m_rng = re.match(rf"{re.escape(job)}_\[(\d+)-(\d+)", jid)
            if m_rng:
                lo, hi = int(m_rng.group(1)), int(m_rng.group(2))
                array_pending_reason = reason.replace("(", "").replace(")", "")
                for tid in range(lo, hi + 1):
                    queue[tid] = {"state": state, "time": elapsed, "reason": reason}
                continue
            if "_" in jid:
                try:
                    tid = int(jid.rsplit("_", 1)[-1])
                except ValueError:
                    continue
                queue[tid] = {"state": state, "time": elapsed, "reason": reason}

    pq_map: dict[tuple[int, int, int], Path] = {}
    for p in SWEEP.glob("*graph-k-noloop-L8-b32-g4-s2500-R0p1*v2cpNL8.parquet"):
        m = re.search(r"knn(\d+)", p.name)
        sm = re.search(r"shard(\d+)-seed(\d+)-v2cpNL8", p.name)
        if not m or not sm:
            continue
        key = (int(m.group(1)), int(sm.group(1)), int(sm.group(2)))
        prev = pq_map.get(key)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            pq_map[key] = p

    cells = []
    for t in range(N_TOT):
        meta = task_meta(t)
        q = queue.get(t)
        a = acct.get(f"{job}_{t}") if job else None
        if q:
            state = q["state"].lower()
            elapsed = q["time"]
        elif a:
            st = a["state"]
            if st.startswith("COMPLETED"):
                state = "done"
            elif "FAIL" in st:
                state = "failed"
            elif "CANCEL" in st:
                state = "cancelled"
            else:
                state = st.lower()
            elapsed = a["elapsed"]
        else:
            state = "pending"
            elapsed = ""

        if state == "completed":
            state = "done"
        if "pending" in state and state != "pending":
            state = "pending"

        step = train = auc = None
        log = LOGDIR / f"jc_gk_noloop_L8_{job}_{t}.out" if job else None
        if log is not None and log.is_file():
            text = log.read_text(errors="ignore")
            gs = [int(x) for x in re.findall(r"global_step=(\d+)", text)]
            if gs:
                step = gs[-1]
            tm = re.findall(r"train_loss=([0-9.eE+-]+)", text)
            if tm:
                train = float(tm[-1])
            am = re.findall(r"val_roc_auc=([0-9.eE+-]+)", text)
            if am:
                auc = float(am[-1])
            if "done L=" in text and state == "running":
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
                        auc = float(
                            vd.sort_values("global_step").iloc[-1].val_roc_auc
                        )
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
                "auc": auc,
                "train": train,
            }
        )

    n_done = sum(1 for c in cells if c["status"] == "done")
    n_run = sum(1 for c in cells if c["status"] == "running")
    n_fail = sum(1 for c in cells if c["status"] == "failed")
    n_pend = N_TOT - n_done - n_run - n_fail

    k_rows = []
    k_mean = []
    for k in KS:
        vals = [
            c["auc"]
            for c in cells
            if c["k"] == k and c["auc"] is not None and c["status"] == "done"
        ]
        n_k = sum(1 for c in cells if c["k"] == k and c["status"] == "done")
        mean = statistics.mean(vals) if vals else None
        std = statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else None)
        k_rows.append({"k": k, "n_done": n_k, "n_total": N_PER_K, "mean": mean, "std": std})
        k_mean.append(mean)

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

    table = []
    tones = []
    for c in sorted(cells, key=lambda x: (x["k"], x["shard"], x["seed"])):
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
    overlay = [L4_NOLOOP_MEAN.get(k) for k in KS]

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
const JOB = {js(job or "—")};
const QUEUE_REASON = {js(array_pending_reason or "")};
const RECIPE =
  "L8/D1024/H32 · η₀=0.1 · B=32×4 · 2500 · R★=0.1 · noloop · v2cpNL8";
const N_DONE = {n_done};
const N_RUN = {n_run};
const N_PEND = {n_pend};
const N_FAIL = {n_fail};
const N_TOT = {N_TOT};
const BEST = {js(best)};
const K_CATS: string[] = {js([str(k) for k in KS])};
const K_MEAN = {js(k_mean)};
const L4_MEAN = {js(overlay)};
const K_ROWS = {js(k_rows)};
const TABLE = {js(table)};
const TONE = {js(tones)};

function fmt(x: number | null | undefined, d = 4): string {{
  return x == null || Number.isNaN(Number(x)) ? "—" : Number(x).toFixed(d);
}}

export default function GraphKNoloopL8D1024Live() {{
  const series = [
    {{
      name: "L8/D1024 noloop",
      data: K_MEAN.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "accent" as const,
    }},
    {{
      name: "L4/D256 noloop (ref)",
      data: L4_MEAN.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "neutral" as const,
    }},
  ];

  return (
    <Stack gap={{20}}>
      <Stack gap={{6}}>
        <H1>Noloop k-sweep — L8/D1024 depth check</H1>
        <Text tone="secondary">{{RECIPE}}</Text>
        <Text tone="secondary">
          Job {{JOB}} · Della 4×A100 · updated {{UPDATED}}
          {{QUEUE_REASON ? ` · queue: ${{QUEUE_REASON}}` : ""}}
        </Text>
        <Row gap={{8}} wrap>
          <Pill tone={{N_DONE === N_TOT ? "success" : "info"}}>
            {{N_DONE}}/{{N_TOT}} done
          </Pill>
          <Pill tone="info">{{N_RUN}} running</Pill>
          <Pill tone="neutral">{{N_PEND}} pending</Pill>
        </Row>
      </Stack>

      <Grid columns={{4}} gap={{12}}>
        <Stat value={{`${{N_DONE}}/${{N_TOT}}`}} label="cells done" />
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

      {{N_RUN === 0 && N_DONE === 0 ? (
        <Callout tone="warning" title="Queued">
          Array pending{{QUEUE_REASON ? ` (${{QUEUE_REASON}})` : ""}}. Fair
          matched jets/steps vs L4 noloop; overlay = L4 means.
        </Callout>
      ) : N_DONE < N_TOT ? (
        <Callout tone="info" title="In progress">
          Testing whether L4 noloop k★≈7 holds at L8/D1024.
        </Callout>
      ) : (
        <Callout tone="success" title="Grid complete">
          All 54 cells finished.
        </Callout>
      )}}

      <Stack gap={{6}}>
        <H2>Mean val ROC vs k</H2>
        <LineChart
          categories={{K_CATS}}
          series={{series as any}}
          height={{240}}
          beginAtZero={{false}}
        />
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
