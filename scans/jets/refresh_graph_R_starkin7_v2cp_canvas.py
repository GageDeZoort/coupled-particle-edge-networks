#!/usr/bin/env python3
"""Regenerate canvases/graph-R-starkin7-k7-completep-v2-live.canvas.tsx."""
from __future__ import annotations

import json
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

RADII = (0.05, 0.075, 0.1, 0.125, 0.15, 0.2)
KNN_K = 7
N_SHARD = 3
N_INIT = 2
N_PER_R = N_SHARD * N_INIT
N_TOT = len(RADII) * N_PER_R  # 36
MAX_STEPS = 2500
# Prior panel (b′) noloop center↔centroid @ k=7
PRIOR_R_MEAN = {
    0.06: 0.9388,
    0.08: 0.9391,
    0.1: 0.9391,
    0.12: 0.9391,
    0.14: 0.9391,
    0.2: 0.9385,
}

SWEEP = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "jetclass_output/adam/sweep_lr"
)
LOGDIR = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs")
OUT = Path(
    "/home/jdezoort/.cursor/projects/home-jdezoort/canvases/"
    "graph-R-starkin7-k7-completep-v2-live.canvas.tsx"
)
JID_FILE = Path(__file__).with_name("graph_R_starkin7_noloop_v2cp_jobid.txt")


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
    return {
        "task": task,
        "R": RADII[r_idx],
        "shard": rem // N_INIT,
        "seed": rem % N_INIT,
    }


def parse_log_progress(text: str):
    step = None
    train_steps = [int(x) for x in re.findall(r"\[train\] step=(\d+)", text)]
    glob_steps = [int(x) for x in re.findall(r"global_step=(\d+)", text)]
    if train_steps:
        step = train_steps[-1]
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
    finished = ("done starkin7" in text) or (step is not None and step >= MAX_STEPS - 5)
    failed = "Traceback (most recent call last)" in text or "File name too long" in text
    return step, train, auc, finished, failed


def main() -> None:
    job = ""
    if JID_FILE.is_file():
        job = JID_FILE.read_text().strip().split()[0]

    acct: dict[str, dict] = {}
    if job:
        for line in run(
            ["sacct", "-j", job, "-n", "-X", "--format=JobID%24,State%14,Elapsed"]
        ).splitlines():
            p = line.split()
            if len(p) >= 3:
                acct[p[0]] = {"state": p[1], "elapsed": p[2]}

    queue: dict[int, dict] = {}
    reason = ""
    for line in run(
        [
            "squeue",
            "-u",
            "jdezoort",
            "-n",
            "jc-gR-sk7-gt,jc-gR-sk7",
            "-h",
            "-o",
            "%i %T %M %R",
        ]
    ).splitlines():
        parts = line.split(None, 3)
        if len(parts) < 2:
            continue
        jid, state = parts[0], parts[1].lower()
        elapsed = parts[2] if len(parts) > 2 else ""
        rsn = parts[3] if len(parts) > 3 else ""
        m_rng = re.match(r".*_\[(\d+)-(\d+)", jid)
        if m_rng:
            reason = rsn.replace("(", "").replace(")", "")
            for tid in range(int(m_rng.group(1)), int(m_rng.group(2)) + 1):
                queue[tid] = {"state": state, "time": elapsed}
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
        queue[tid] = {"state": state, "time": elapsed}

    pq_map: dict[tuple[float, int, int], Path] = {}
    for p in SWEEP.glob("*sk7*pt*gskNL.parquet"):
        rm = re.search(r"gstar([0-9p]+)", p.name)
        sm = re.search(r"skip([0-9p]+k)?|shard", p.name)
        # seed from ...-seed{S}-gskNL or seed{S} before D
        seed_m = re.search(r"seed(\d+)-gskNL", p.name) or re.search(
            r"-s2p5k-seed(\d+)-", p.name
        )
        if not rm or not seed_m:
            continue
        r_val = float(rm.group(1).replace("p", "."))
        seed = int(seed_m.group(1))
        # shard from train-skip: skip0 → 0, skip320k → 1, skip640k → 2
        skip_m = re.search(r"-skip([0-9p]+k)-", p.name)
        if skip_m:
            tok = skip_m.group(1).replace("p", ".")
            if tok.endswith("k"):
                n = float(tok[:-1]) * 1000
            else:
                n = float(tok)
            shard = int(round(n / 320000))
        else:
            shard = 0
        key = (r_val, shard, seed)
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
                state = "pending"
            else:
                state = st.lower()
            elapsed = a["elapsed"]
        else:
            state = "pending"
            elapsed = ""
        if state == "completed":
            state = "done"

        step = train = auc = None
        log_paths = []
        if job:
            log_paths.append(LOGDIR / f"jc_gR_sk7_{job}_{t}.out")
            log_paths.append(LOGDIR / f"jc_gR_sk7_gt_{job}_{t}.out")
        log_paths.extend(sorted(LOGDIR.glob(f"jc_gR_sk7_*_{t}.out")))
        log_paths.extend(sorted(LOGDIR.glob(f"jc_gR_sk7_gt_*_{t}.out")))
        for log in log_paths:
            if not log.is_file():
                continue
            try:
                text = log.read_text(errors="ignore")
            except Exception:
                continue
            s, tr, au, finished, failed = parse_log_progress(text)
            if s is not None:
                step = max(step or 0, s)
            if tr is not None:
                train = tr
            if au is not None:
                auc = au
            if finished and state != "failed":
                state = "done"
            if failed and state not in ("done", "running"):
                state = "failed"

        path = pq_map.get((meta["R"], meta["shard"], meta["seed"]))
        if path is not None and path.is_file():
            try:
                df = pq.read_table(
                    path, columns=["global_step", "train_loss", "val_roc_auc"]
                ).to_pandas()
                if len(df):
                    step = max(step or 0, int(df.global_step.max()))
                    vd = df[["global_step", "val_roc_auc"]].dropna()
                    if len(vd):
                        row = vd.sort_values("global_step").iloc[-1]
                        auc = float(row.val_roc_auc)
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

        cells.append({**meta, "status": state, "step": step, "elapsed": elapsed, "auc": auc, "train": train})

    n_done = sum(1 for c in cells if c["status"] == "done")
    n_run = sum(1 for c in cells if c["status"] == "running")
    n_fail = sum(1 for c in cells if c["status"] == "failed")
    n_pend = N_TOT - n_done - n_run - n_fail

    r_cats = [f"{r:g}" for r in RADII]
    r_mean = []
    r_rows = []
    for r in RADII:
        vals = [
            c["auc"]
            for c in cells
            if abs(c["R"] - r) < 1e-12 and c["auc"] is not None and c["status"] == "done"
        ]
        n_r = sum(1 for c in cells if abs(c["R"] - r) < 1e-12 and c["status"] == "done")
        mean = statistics.mean(vals) if vals else None
        std = statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else None)
        r_mean.append(mean)
        r_rows.append({"R": r, "n_done": n_r, "mean": mean, "std": std})

    overlay = [PRIOR_R_MEAN.get(r) for r in RADII]

    best = None
    for c in cells:
        if c["auc"] is not None and c["status"] == "done":
            if best is None or c["auc"] > best["auc"]:
                best = {"auc": c["auc"], "R": c["R"], "shard": c["shard"], "seed": c["seed"]}

    table = []
    tones = []
    for c in sorted(cells, key=lambda x: (x["R"], x["shard"], x["seed"])):
        step_s = f"{c['step']}/{MAX_STEPS}" if c["step"] is not None else f"—/{MAX_STEPS}"
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
                "pending": "neutral",
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
const QUEUE_REASON = {js(reason or "")};
const RECIPE =
  "L4/D256 · η₀=0.1 · B=128×1 · 2500 · k=7 noloop · star=kin7(Σ4-vec) · seed pT>1 GeV · R∈{{0.05…0.2}} · gskNL";
const N_DONE = {n_done};
const N_RUN = {n_run};
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

export default function GraphRStarKin7K7CompletePV2Live() {{
  const series = [
    {{
      name: "star kin7 + kNN mean val ROC",
      data: R_MEAN.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "accent" as const,
    }},
    {{
      name: "prior star part-int (nearest R)",
      data: OVERLAY.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "neutral" as const,
    }},
  ];

  return (
    <Stack gap={{20}}>
      <Stack gap={{6}}>
        <H1>Star kin7 aggregates — R★ @ k=7 noloop</H1>
        <Text tone="secondary">{{RECIPE}}</Text>
        <Text tone="secondary">
          job {{JOB}} · updated {{UPDATED}}
          {{QUEUE_REASON ? ` · ${{QUEUE_REASON}}` : ""}}
        </Text>
        <Row gap={{8}} wrap>
          <Pill tone={{N_DONE === N_TOT ? "success" : "info"}}>
            {{N_DONE}}/{{N_TOT}} done
          </Pill>
          <Pill tone="info">{{N_RUN}} running</Pill>
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
              : `Best · R=${{BEST.R}} s${{BEST.shard}} seed${{BEST.seed}}`
          }}
        />
      </Grid>

      {{N_DONE < N_TOT ? (
        <Callout tone="info" title="In progress">
          Star hyperedges: kin7 of support 4-sum vs jet; only pT&gt;1 GeV seeds.
          kNN part-interaction unchanged. Overlay = prior center↔centroid panel (nearest R).
        </Callout>
      ) : (
        <Callout tone="success" title="Grid complete">
          All {{N_TOT}} cells finished 2500 steps.
        </Callout>
      )}}

      <Stack gap={{6}}>
        <H2>Mean val ROC vs R★ (star kin7, k=7)</H2>
        <Text tone="secondary">
          Source: JetClass stream · CompleteP v2 · 3 shards × 2 seeds
        </Text>
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
            `${{m.n_done}}/6`,
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
    print(f"wrote {OUT} done={n_done} run={n_run} pend={n_pend} fail={n_fail}")


if __name__ == "__main__":
    main()
