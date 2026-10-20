#!/usr/bin/env python3
"""Regenerate canvases/graph-R-noloop-k7-completep-v2-live.canvas.tsx."""
from __future__ import annotations

import json
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

JOB = ""
RADII = (0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.2)
# Original 30-cell array used these; fill-in {0.04,0.2} is a separate 12-cell array.
CORE_RADII = (0.06, 0.08, 0.1, 0.12, 0.14)
EXT_RADII = (0.04, 0.2)
KNN_K = 7
N_SHARD = 3
N_INIT = 2
N_PER_R = N_SHARD * N_INIT
N_TOT = len(RADII) * N_PER_R  # 42 with R∈{0.04,0.2} fill-in
MAX_STEPS = 2500

SWEEP = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "jetclass_output/adam/sweep_lr"
)
LOGDIR = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs")
OUT = Path(
    "/home/jdezoort/.cursor/projects/home-jdezoort/canvases/"
    "graph-R-noloop-k7-completep-v2-live.canvas.tsx"
)
JID_FILE = Path(__file__).with_name("graph_R_noloop_v2cp_jobid.txt")
GT_JOBIDS = Path.home() / ".cache/jc_gR_noloop_gt/submitted_jobids.txt"

# Self-included panel (b) @ k=3 means — overlay only
SELF_INCL_R_MEAN = {
    0.05: 0.9406,
    0.075: 0.9421,
    0.1: 0.9427,
    0.125: 0.9424,
    0.15: 0.9414,
}


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, timeout=45)
    except Exception:
        return ""


def js(obj) -> str:
    return json.dumps(obj)


def task_meta(task: int) -> dict:
    r_idx = task // N_PER_R
    rem = task % N_PER_R
    shard = rem // N_INIT
    seed = rem % N_INIT
    return {"task": task, "R": RADII[r_idx], "shard": shard, "seed": seed}


def array_task_for(r: float, shard: int, seed: int) -> tuple[str, int]:
    """Return (array_kind, task_id) for log/queue lookup."""
    rem = shard * N_INIT + seed
    if any(abs(r - x) < 1e-9 for x in EXT_RADII):
        r_idx = next(i for i, x in enumerate(EXT_RADII) if abs(r - x) < 1e-9)
        return "ext", r_idx * N_PER_R + rem
    r_idx = next(i for i, x in enumerate(CORE_RADII) if abs(r - x) < 1e-9)
    return "core", r_idx * N_PER_R + rem


def parse_log_progress(text: str) -> tuple[int | None, float | None, float | None, bool]:
    step = None
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
    finished = ("done R=" in text) or (step is not None and step >= MAX_STEPS - 5)
    return step, train, auc, finished


def latest_logs_for_cell(r: float, shard: int, seed: int, a100_job: str) -> list[Path]:
    kind, task = array_task_for(r, shard, seed)
    paths: list[Path] = []
    if kind == "ext":
        paths.extend(sorted(LOGDIR.glob(f"jc_gR_ext_noloop_*_{task}.out")))
        paths.extend(sorted(LOGDIR.glob(f"jc_gR_ext_noloop_gt_*_{task}.out")))
    else:
        if a100_job:
            a100 = LOGDIR / f"jc_gR_noloop_{a100_job}_{task}.out"
            if a100.is_file():
                paths.append(a100)
        paths.extend(sorted(LOGDIR.glob(f"jc_gR_noloop_gt_*_{task}.out")))
    uniq = {p.resolve(): p for p in paths}
    return sorted(uniq.values(), key=lambda p: p.stat().st_mtime)


def main() -> None:
    job = JOB
    if JID_FILE.is_file():
        job = JID_FILE.read_text().strip() or job

    gt_parents: list[str] = []
    if GT_JOBIDS.is_file():
        gt_parents = [
            ln.strip()
            for ln in GT_JOBIDS.read_text().splitlines()
            if ln.strip().isdigit()
        ]
    for p in LOGDIR.glob("jc_gR_noloop_gt_*_*.out"):
        m = re.match(r"jc_gR_noloop_gt_(\d+)_\d+\.out$", p.name)
        if m and m.group(1) not in gt_parents:
            gt_parents.append(m.group(1))

    acct: dict[str, dict] = {}
    for jid in ([job] if job else []) + gt_parents:
        for line in run(
            ["sacct", "-j", jid, "-n", "-X", "--format=JobID%24,State%14,Elapsed"]
        ).splitlines():
            p = line.split()
            if len(p) >= 3:
                acct[p[0]] = {"state": p[1], "elapsed": p[2]}

    queue: dict[int, dict] = {}
    array_pending_reason = ""

    def _ingest(lines: str, *, partition: str) -> None:
        nonlocal array_pending_reason
        for line in lines.splitlines():
            parts = line.split(None, 3)
            if len(parts) < 2:
                continue
            jid, state = parts[0], parts[1].lower()
            elapsed = parts[2] if len(parts) > 2 else ""
            reason = parts[3] if len(parts) > 3 else ""
            m_rng = re.match(r".*_\[(\d+)-(\d+)", jid)
            if m_rng:
                lo, hi = int(m_rng.group(1)), int(m_rng.group(2))
                if partition != "gputest":
                    array_pending_reason = reason.replace("(", "").replace(")", "")
                for tid in range(lo, hi + 1):
                    prev = queue.get(tid)
                    if prev and prev.get("part") == "gputest" and partition != "gputest":
                        continue
                    if prev and prev.get("state") == "running" and state != "running":
                        continue
                    queue[tid] = {
                        "state": state,
                        "time": elapsed,
                        "reason": reason,
                        "part": partition,
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
            if prev and prev.get("part") == "gputest" and partition != "gputest":
                continue
            if prev and prev.get("state") == "running" and state != "running":
                continue
            queue[tid] = {
                "state": state,
                "time": elapsed,
                "reason": reason,
                "part": partition,
            }

    if job:
        _ingest(run(["squeue", "-j", job, "-h", "-o", "%i %T %M %R"]), partition="a100")
    _ingest(
        run(
            [
                "squeue",
                "-u",
                "jdezoort",
                "-n",
                "jc-gR-nl-gt",
                "-h",
                "-o",
                "%i %T %M %R",
            ]
        ),
        partition="gputest",
    )
    # Fill-in R∈{0.04,0.2} — separate array; store under negative task ids to avoid clash.
    queue_ext: dict[int, dict] = {}

    def _ingest_ext(lines: str, *, partition: str) -> None:
        for line in lines.splitlines():
            parts = line.split(None, 3)
            if len(parts) < 2:
                continue
            jid, state = parts[0], parts[1].lower()
            elapsed = parts[2] if len(parts) > 2 else ""
            reason = parts[3] if len(parts) > 3 else ""
            m_rng = re.match(r".*_\[(\d+)-(\d+)", jid)
            tids: list[int] = []
            if m_rng:
                tids = list(range(int(m_rng.group(1)), int(m_rng.group(2)) + 1))
            else:
                m_one = re.match(r".*_\[(\d+)\]$", jid)
                if m_one:
                    tids = [int(m_one.group(1))]
                elif "_" in jid:
                    try:
                        tids = [int(jid.rsplit("_", 1)[-1])]
                    except ValueError:
                        continue
            for tid in tids:
                prev = queue_ext.get(tid)
                if prev and prev.get("part") == "gputest" and partition != "gputest":
                    continue
                if prev and prev.get("state") == "running" and state != "running":
                    continue
                queue_ext[tid] = {
                    "state": state,
                    "time": elapsed,
                    "reason": reason,
                    "part": partition,
                }

    _ingest_ext(
        run(
            [
                "squeue",
                "-u",
                "jdezoort",
                "-n",
                "jc-gR-ext-nl,jc-gR-ext-nl-gt",
                "-h",
                "-o",
                "%i %T %M %R",
            ]
        ),
        partition="a100",
    )

    pq_map: dict[tuple[float, int, int], Path] = {}
    for p in SWEEP.glob(f"*graph-R-noloop-b128-g1-s2500-k{KNN_K}*v2cpNL.parquet"):
        rm = re.search(r"gstar([0-9p]+)", p.name)
        sm = re.search(r"shard(\d+)-seed(\d+)-v2cpNL", p.name)
        if not rm or not sm:
            continue
        try:
            r_val = float(rm.group(1).replace("p", "."))
        except ValueError:
            continue
        key = (r_val, int(sm.group(1)), int(sm.group(2)))
        prev = pq_map.get(key)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            pq_map[key] = p

    cells = []
    for t in range(N_TOT):
        meta = task_meta(t)
        kind, arr_t = array_task_for(meta["R"], meta["shard"], meta["seed"])
        if kind == "ext":
            q = queue_ext.get(arr_t)
            a = None
        else:
            q = queue.get(arr_t)
            a = acct.get(f"{job}_{arr_t}") if job else None
            for parent in gt_parents:
                cand = acct.get(f"{parent}_{arr_t}")
                if cand is not None and (
                    a is None or str(cand.get("state", "")).startswith("COMPLETED")
                ):
                    a = cand

        if q:
            state = q["state"].lower()
            elapsed = q["time"]
            reason = q.get("reason", "")
            if q.get("part") == "gputest" and state in ("pending", "running"):
                state = f"{state}@gt"
        elif a:
            st = a["state"]
            if st.startswith("COMPLETED"):
                state = "done"
            elif "FAIL" in st:
                state = "failed"
            elif "CANCEL" in st:
                state = "pending"
            else:
                state = st.lower()
            elapsed = a["elapsed"]
            reason = ""
        else:
            state = "pending"
            elapsed = ""
            reason = array_pending_reason if kind == "core" else ""

        if state == "completed":
            state = "done"

        step = train = auc = None
        for log in latest_logs_for_cell(meta["R"], meta["shard"], meta["seed"], job):
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
            if finished and state not in ("failed",):
                state = "done"

        key = (meta["R"], meta["shard"], meta["seed"])
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

        if step is not None and step >= MAX_STEPS - 5 and state != "failed":
            state = "done"

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
    n_run = sum(1 for c in cells if c["status"] in ("running", "running@gt"))
    n_gt = sum(1 for c in cells if c["status"].endswith("@gt"))
    n_fail = sum(1 for c in cells if c["status"] == "failed")
    n_pend = N_TOT - n_done - n_run - n_fail

    r_rows = []
    r_mean = []
    r_cats = [f"{r:g}" for r in RADII]
    for r in RADII:
        vals = [
            c["auc"]
            for c in cells
            if abs(c["R"] - r) < 1e-9
            and c["auc"] is not None
            and c["status"] == "done"
        ]
        n_r = sum(
            1 for c in cells if abs(c["R"] - r) < 1e-9 and c["status"] == "done"
        )
        mean = statistics.mean(vals) if vals else None
        std = statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else None)
        r_rows.append({"R": r, "n_done": n_r, "n_total": N_PER_R, "mean": mean, "std": std})
        r_mean.append(mean)

    # overlay at nearest old grid points (only exact matches shown)
    overlay = [SELF_INCL_R_MEAN.get(r) for r in RADII]

    best = None
    for c in cells:
        if c["auc"] is not None and c["status"] == "done":
            if best is None or c["auc"] > best["auc"]:
                best = {
                    "auc": c["auc"],
                    "R": c["R"],
                    "shard": c["shard"],
                    "seed": c["seed"],
                }

    table = []
    tones = []
    for c in sorted(cells, key=lambda x: (x["R"], x["shard"], x["seed"])):
        step_s = (
            f"{c['step']}/{MAX_STEPS}" if c["step"] is not None else f"0/{MAX_STEPS}"
        )
        table.append(
            [
                f"{c['R']:g}",
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
            }.get(c["status"], "neutral")
        )

    updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best_js = "null" if best is None else js(best)

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
  "L4/D256 · η₀=0.1 · B=128×1 · 2500 · k★=7 noloop · R∈{{0.04,0.06,0.08,0.1,0.12,0.14,0.2}} · v2cpNL";
const N_DONE = {n_done};
const N_RUN = {n_run};
const N_GT = {n_gt};
const N_PEND = {n_pend};
const N_FAIL = {n_fail};
const N_TOT = {N_TOT};
const BEST = {best_js};
const R_CATS: string[] = {js(r_cats)};
const R_MEAN = {js(r_mean)};
const OVERLAY = {js(overlay)};
const R_ROWS = {js(r_rows)};
const TABLE = {js(table)};
const TONE = {js(tones)};

function fmt(x: number | null | undefined, d = 4): string {{
  return x == null || Number.isNaN(Number(x)) ? "—" : Number(x).toFixed(d);
}}

export default function GraphRNoloopK7CompletePV2Live() {{
  const series = [
    {{
      name: "noloop k=7 mean val ROC",
      data: R_MEAN.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "accent" as const,
    }},
    {{
      name: "self-incl k=3 (prior panel b)",
      data: OVERLAY.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "neutral" as const,
    }},
  ];

  return (
    <Stack gap={{20}}>
      <Stack gap={{6}}>
        <H1>Graph construction panel (b′) — R★ @ k★=7 noloop</H1>
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
              : `Best · R=${{BEST.R}} s${{BEST.shard}} seed${{BEST.seed}}`
          }}
        />
      </Grid>

      {{N_GT > 0 && N_DONE < N_TOT ? (
        <Callout tone="info" title="gputest greedy fill">
          Watchdog ≤3 jc-gR-nl-gt parents; cancels matching a100 indices.
          Overlay = prior self-included panel (b) @ k=3 (different R grid).
        </Callout>
      ) : N_DONE < N_TOT ? (
        <Callout tone="info" title="In progress">
          Noloop R★ scan at frozen k★=7 from panel (a′).
        </Callout>
      ) : (
        <Callout tone="success" title="Grid complete">
          All 30 cells finished 2500 steps.
        </Callout>
      )}}

      <Stack gap={{6}}>
        <H2>Mean val ROC vs R★ @ k★=7 noloop</H2>
        <LineChart
          categories={{R_CATS}}
          series={{series as any}}
          height={{240}}
          beginAtZero={{false}}
        />
      </Stack>

      <Stack gap={{8}}>
        <H2>Per-R summary</H2>
        <Table
          headers={{["R★", "done/6", "mean ROC", "std"]}}
          rows={{R_ROWS.map((m: any) => [
            String(m.R),
            `${{m.n_done}}/${{m.n_total}}`,
            fmt(m.mean),
            fmt(m.std),
          ])}}
        />
      </Stack>

      <Stack gap={{8}}>
        <H2>Cells</H2>
        <Table
          headers={{["R★", "shard", "seed", "status", "step", "train", "AUC"]}}
          rows={{TABLE}}
          rowTone={{TONE as any}}
        />
      </Stack>
    </Stack>
  );
}}
"""
    OUT.write_text(canvas)
    print(f"wrote {OUT} done={n_done} run={n_run} gt={n_gt} pend={n_pend} fail={n_fail}")


if __name__ == "__main__":
    main()
