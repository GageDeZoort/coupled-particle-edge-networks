#!/usr/bin/env python3
"""Regenerate canvases/graph-k-noloop-completep-v2-live.canvas.tsx from Slurm + parquets."""
from __future__ import annotations

import json
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

JOB = "15419558"
KS = (1, 2, 3, 4, 5, 6, 7, 8, 10)
N_SHARD = 3
N_INIT = 2
N_PER_K = N_SHARD * N_INIT
N_TOT = len(KS) * N_PER_K  # 54
MAX_STEPS = 2500
BATCH = 128

SWEEP = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "jetclass_output/adam/sweep_lr"
)
LOGDIR = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs")
OUT = Path(
    "/home/jdezoort/.cursor/projects/home-jdezoort/canvases/"
    "graph-k-noloop-completep-v2-live.canvas.tsx"
)
GT_JOBIDS = Path.home() / ".cache/jc_gk_noloop_gt/submitted_jobids.txt"

# Prior CompleteP panel (a) with self-included kNN (no k=1)
SELF_INCLUDED_MEAN = {
    1: None,
    2: 0.941890,
    3: 0.942714,
    4: 0.941232,
    5: 0.939609,
    6: 0.939075,
    7: 0.939172,
    8: 0.939077,
    10: 0.938895,
}
V1LAM_MEAN = SELF_INCLUDED_MEAN


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


def parse_log_progress(text: str) -> tuple[int | None, float | None, float | None, bool]:
    """Return (step, train_loss, val_auc, finished) from a train log."""
    step = None
    # Prefer explicit train step lines; fall back to global_step heartbeats.
    train_steps = [int(x) for x in re.findall(r"\[train\] step=(\d+)", text)]
    glob_steps = [int(x) for x in re.findall(r"global_step=(\d+)", text)]
    if train_steps:
        step = train_steps[-1]
    elif glob_steps:
        step = glob_steps[-1]
    if glob_steps:
        step = max(step or 0, glob_steps[-1])

    train = None
    tm = re.findall(r"train_loss=([0-9.eE+-]+)", text)
    if tm:
        try:
            train = float(tm[-1])
        except ValueError:
            pass

    auc = None
    am = re.findall(r"val_roc_auc=([0-9.eE+-]+)", text)
    if am:
        try:
            auc = float(am[-1])
        except ValueError:
            pass

    finished = ("done k=" in text) or (step is not None and step >= MAX_STEPS - 5)
    return step, train, auc, finished


def latest_logs_for_task(task: int, a100_job: str) -> list[Path]:
    """a100 + all gputest logs for this array index, newest last."""
    paths: list[Path] = []
    a100 = LOGDIR / f"jc_gk_noloop_{a100_job}_{task}.out"
    if a100.is_file():
        paths.append(a100)
    paths.extend(sorted(LOGDIR.glob(f"jc_gk_noloop_gt_*_{task}.out")))
    # Dedup + sort by mtime
    uniq = {p.resolve(): p for p in paths}
    return sorted(uniq.values(), key=lambda p: p.stat().st_mtime)


JID_FILE = Path(__file__).with_name("graph_k_noloop_v2cp_jobid.txt")


def main() -> None:
    global JOB
    if JID_FILE.is_file():
        JOB = JID_FILE.read_text().strip() or JOB

    # --- sacct: a100 array + known gputest parents ---
    acct: dict[str, dict] = {}
    gt_parents: list[str] = []
    if GT_JOBIDS.is_file():
        gt_parents = [
            ln.strip()
            for ln in GT_JOBIDS.read_text().splitlines()
            if ln.strip().isdigit()
        ]
    # Also discover parents from log filenames
    for p in LOGDIR.glob("jc_gk_noloop_gt_*_*.out"):
        m = re.match(r"jc_gk_noloop_gt_(\d+)_\d+\.out$", p.name)
        if m and m.group(1) not in gt_parents:
            gt_parents.append(m.group(1))

    sacct_jobs = [JOB] + gt_parents
    for jid in sacct_jobs:
        for line in run(
            [
                "sacct",
                "-j",
                jid,
                "-n",
                "-X",
                "--format=JobID%24,State%14,Elapsed",
            ]
        ).splitlines():
            p = line.split()
            if len(p) >= 3:
                acct[p[0]] = {"state": p[1], "elapsed": p[2]}

    queue: dict[int, dict] = {}
    array_pending_reason = ""

    def _ingest_squeue(lines: str, *, partition: str = "") -> None:
        nonlocal array_pending_reason
        for line in lines.splitlines():
            parts = line.split(None, 3)
            if len(parts) < 2:
                continue
            jid, state = parts[0], parts[1].lower()
            elapsed = parts[2] if len(parts) > 2 else ""
            reason = parts[3] if len(parts) > 3 else ""
            tag = partition or "a100"

            m_rng = re.match(r".*_\[(\d+)-(\d+)", jid)
            if m_rng:
                lo, hi = int(m_rng.group(1)), int(m_rng.group(2))
                if partition != "gputest":
                    array_pending_reason = reason.replace("(", "").replace(")", "")
                for tid in range(lo, hi + 1):
                    prev = queue.get(tid)
                    if prev and prev.get("state") == "running" and state != "running":
                        continue
                    if prev and prev.get("part") == "gputest" and partition != "gputest":
                        continue
                    queue[tid] = {
                        "state": state,
                        "time": elapsed,
                        "reason": reason,
                        "part": tag,
                    }
                continue

            m_one = re.match(r".*_\[(\d+)\]$", jid)
            if m_one:
                tid = int(m_one.group(1))
            elif "_" in jid:
                try:
                    tid = int(jid.rsplit("_", 1)[-1])
                except ValueError:
                    continue
            else:
                continue

            prev = queue.get(tid)
            if prev and prev.get("state") == "running" and state != "running":
                continue
            if prev and prev.get("part") == "gputest" and partition != "gputest":
                continue
            queue[tid] = {
                "state": state,
                "time": elapsed,
                "reason": reason,
                "part": tag,
            }

    _ingest_squeue(
        run(["squeue", "-j", JOB, "-h", "-o", "%i %T %M %R"]),
        partition="a100",
    )
    _ingest_squeue(
        run(
            [
                "squeue",
                "-u",
                "jdezoort",
                "-n",
                "jc-gk-nl-gt",
                "-h",
                "-o",
                "%i %T %M %R",
            ]
        ),
        partition="gputest",
    )

    # index parquets by (k, shard, seed)
    pq_map: dict[tuple[int, int, int], Path] = {}
    for p in SWEEP.glob("*graph-k-noloop-b128-g1-s2500-R0p1*v2cpNL.parquet"):
        m = re.search(r"knn(\d+)", p.name)
        sm = re.search(r"shard(\d+)-seed(\d+)-v2cpNL", p.name)
        if not m or not sm:
            continue
        key = (int(m.group(1)), int(sm.group(1)), int(sm.group(2)))
        prev = pq_map.get(key)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            pq_map[key] = p

    cells = []
    for t in range(N_TOT):
        meta = task_meta(t)
        jid_a100 = f"{JOB}_{t}"
        q = queue.get(t)

        # gputest sacct may be parent_task
        a = acct.get(jid_a100)
        a_gt = None
        for parent in gt_parents:
            cand = acct.get(f"{parent}_{t}")
            if cand is not None:
                a_gt = cand
                # Prefer completed gputest over cancelled a100
                if a is None or str(cand.get("state", "")).startswith("COMPLETED"):
                    a = cand

        if q:
            state = q["state"].lower()
            elapsed = q["time"]
            reason = q.get("reason", "")
            part = q.get("part", "")
            if part == "gputest" and state in ("pending", "running"):
                state = f"{state}@gt"
        elif a:
            st = a["state"]
            if st.startswith("COMPLETED"):
                state = "done"
            elif "FAIL" in st:
                state = "failed"
            elif "CANCEL" in st:
                # a100 cancel is expected when gputest takes the cell — not terminal.
                state = "pending"
            else:
                state = st.lower()
            elapsed = a["elapsed"]
            reason = ""
            if a_gt and str(a_gt.get("state", "")).startswith("COMPLETED"):
                state = "done"
                elapsed = a_gt["elapsed"]
        else:
            state = "pending"
            elapsed = ""
            reason = array_pending_reason

        if state == "completed":
            state = "done"
        if state in ("running", "running@gt", "pending@gt"):
            pass
        elif state not in ("done", "failed", "cancelled", "pending"):
            if state.endswith("@gt"):
                pass
            elif "pending" in state:
                state = "pending"

        step = None
        auc = None
        train = None

        # Live progress from a100 + gputest logs (newest wins for step/auc).
        for log in latest_logs_for_task(t, JOB):
            try:
                text = log.read_text(errors="ignore")
            except Exception:
                continue
            s, tr, au, finished = parse_log_progress(text)
            if s is not None:
                step = max(step or 0, s)
            if tr is not None:
                train = tr
            if au is not None:
                auc = au
            if finished and state in ("running", "running@gt", "pending", "pending@gt"):
                state = "done"
                if "gt" in log.name:
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

        if step is not None and step >= MAX_STEPS - 5 and state not in ("failed",):
            state = "done"

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
    n_run = sum(1 for c in cells if c["status"] in ("running", "running@gt"))
    n_gt = sum(1 for c in cells if c["status"].endswith("@gt"))
    n_fail = sum(1 for c in cells if c["status"] == "failed")
    n_pend = N_TOT - n_done - n_run - n_fail

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
                "running@gt": "info",
                "pending": "neutral",
                "pending@gt": "warning",
                "failed": "danger",
                "cancelled": "warning",
            }.get(c["status"], "neutral")
        )

    updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_js = "null" if best is None else js(best)

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
const QUEUE_REASON = {js(array_pending_reason or "")};
const RECIPE =
  "L4/D256 · η₀=0.1 · B=128×1 · 2500 steps · R★=0.1 · CompleteP · noloop · v2cpNL";
const N_DONE = {n_done};
const N_RUN = {n_run};
const N_GT = {n_gt};
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

export default function GraphKNoloopCompletePV2Live() {{
  const series = [
    {{
      name: "v2cp mean val ROC",
      data: K_MEAN.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "accent" as const,
    }},
    {{
      name: "self-included v2cp (prior)",
      data: V1_MEAN.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "neutral" as const,
    }},
  ];

  return (
    <Stack gap={{20}}>
      <Stack gap={{6}}>
        <H1>Graph construction panel (a′) — kNN no self-loops</H1>
        <Text tone="secondary">{{RECIPE}}</Text>
        <Text tone="secondary">
          a100 {{JOB}} + gputest greedy fill (≤3) · updated {{UPDATED}}
          {{QUEUE_REASON ? ` · a100: ${{QUEUE_REASON}}` : ""}}
        </Text>
        <Row gap={{8}} wrap>
          <Pill tone={{N_DONE === N_TOT ? "success" : "info"}}>
            {{N_DONE}}/{{N_TOT}} done
          </Pill>
          <Pill tone="info">{{N_RUN}} running</Pill>
          <Pill tone="warning">{{N_GT}} on gputest</Pill>
          <Pill tone="neutral">{{N_PEND}} pending</Pill>
          {{N_FAIL > 0 ? <Pill tone="warning">{{N_FAIL}} failed</Pill> : null}}
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

      {{N_GT > 0 && N_DONE < N_TOT ? (
        <Callout tone="info" title="gputest greedy fill">
          Watchdog keeps ≤3 jc-gk-nl-gt parents on gputest (same run-tag as
          a100). Cancels matching a100 array indices whenever a gputest task
          is queued/running/done. Steps come from gt logs + parquets.
        </Callout>
      ) : N_RUN === 0 && N_DONE === 0 ? (
        <Callout tone="warning" title="Queued — no GPUs yet">
          Array {{JOB}} still PENDING
          {{QUEUE_REASON ? ` (${{QUEUE_REASON}})` : ""}}. Overlay = prior
          self-included panel (a), not a matched noloop recipe.
        </Callout>
      ) : N_DONE < N_TOT ? (
        <Callout tone="info" title="In progress">
          True-neighbor k-sweep (noloop). Overlay = prior self-included
          CompleteP panel (a) means.
        </Callout>
      ) : (
        <Callout tone="success" title="Grid complete">
          All 54 cells finished 2500 steps.
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
          Source: v2cp parquets (done cells only) · overlay: prior self-included CompleteP panel a
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
    print(
        f"wrote {OUT} done={n_done} run={n_run} gt={n_gt} pend={n_pend} fail={n_fail}"
    )


if __name__ == "__main__":
    main()
