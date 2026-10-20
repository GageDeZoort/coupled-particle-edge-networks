#!/usr/bin/env python3
"""Live canvas: star-only R★ ablation vs star+k=3 panel (b)."""
from __future__ import annotations

import json
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

JOB = "15407319"
RADII = (0.05, 0.075, 0.1, 0.125, 0.15)
N_SHARD = 3
N_INIT = 2
N_PER_R = N_SHARD * N_INIT
N_TOT = len(RADII) * N_PER_R
MAX_STEPS = 2500

# Panel (b) star+k★=3 means (CompleteP v2, finished)
K3_MEAN = {
    0.05: 0.9406,
    0.075: 0.9421,
    0.1: 0.9427,
    0.125: 0.9424,
    0.15: 0.9414,
}

SWEEP = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "jetclass_output/adam/sweep_lr"
)
LOGDIR = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs")
OUT = Path(
    "/home/jdezoort/.cursor/projects/home-jdezoort/canvases/"
    "graph-R-staronly-completep-v2-live.canvas.tsx"
)
JID_FILE = Path(__file__).with_name("graph_R_staronly_v2cp_jobid.txt")


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, timeout=45)
    except Exception:
        return ""


def js(obj) -> str:
    return json.dumps(obj)


def r_tag(r: float) -> str:
    return f"{r:g}".replace(".", "p")


def task_meta(task: int) -> dict:
    r_idx = task // N_PER_R
    rem = task % N_PER_R
    return {
        "task": task,
        "R": RADII[r_idx],
        "shard": rem // N_INIT,
        "seed": rem % N_INIT,
    }


def main() -> None:
    job = JOB
    if JID_FILE.is_file():
        job = JID_FILE.read_text().strip() or job

    acct: dict[str, dict] = {}
    for line in run(
        ["sacct", "-j", job, "-n", "-X", "--format=JobID%18,State%14,Elapsed"]
    ).splitlines():
        p = line.split()
        if len(p) >= 3:
            acct[p[0]] = {"state": p[1], "elapsed": p[2]}

    queue: dict[str, dict] = {}
    for line in run(["squeue", "-j", job, "-h", "-o", "%i %T %M %R"]).splitlines():
        parts = line.split(None, 3)
        if len(parts) >= 3:
            queue[parts[0]] = {
                "state": parts[1],
                "time": parts[2],
                "reason": parts[3] if len(parts) > 3 else "",
            }

    pq_map: dict[tuple[float, int, int], Path] = {}
    for p in SWEEP.glob("*graph-R-staronly-b128-g1-s2500*v2cpStar.parquet"):
        # reject accidental knn-tagged runs
        if re.search(r"knn\d+", p.name):
            continue
        rm = re.search(r"gstar([0-9p]+)", p.name)
        sm = re.search(r"shard(\d+)-seed(\d+)-v2cpStar", p.name)
        if not rm or not sm:
            continue
        r_val = float(rm.group(1).replace("p", "."))
        key = (r_val, int(sm.group(1)), int(sm.group(2)))
        prev = pq_map.get(key)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            pq_map[key] = p

    cells = []
    for t in range(N_TOT):
        meta = task_meta(t)
        jid = f"{job}_{t}"
        q = queue.get(jid)
        a = acct.get(jid)
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
        elif state not in ("done", "failed", "cancelled", "pending", "running"):
            if "pending" in state:
                state = "pending"

        step = auc = train = None
        log = LOGDIR / f"jc_gR_star_{job}_{t}.out"
        if log.exists():
            text = log.read_text(errors="ignore")
            steps = re.findall(r"global_step=(\d+)", text)
            if steps:
                step = int(steps[-1])
            if "done R=" in text and state == "running":
                state = "done"

        path = None
        for (rv, sh, sd), pp in pq_map.items():
            if sh == meta["shard"] and sd == meta["seed"] and abs(rv - meta["R"]) < 1e-9:
                path = pp
                break

        if path is not None and path.is_file():
            try:
                df = pq.read_table(
                    path, columns=["global_step", "train_loss", "val_roc_auc"]
                ).to_pandas()
                if len(df):
                    step = max(step or 0, int(df.global_step.max()))
                    vd = df[["global_step", "val_roc_auc"]].dropna()
                    if len(vd):
                        auc = float(vd.sort_values("global_step").iloc[-1].val_roc_auc)
                    td = df[["global_step", "train_loss"]].dropna()
                    if len(td):
                        train = float(
                            td.sort_values("global_step").iloc[-1].train_loss
                        )
                    if step >= MAX_STEPS - 5:
                        state = "done"
            except Exception:
                pass

        cells.append({**meta, "status": state, "step": step, "elapsed": elapsed, "auc": auc, "train": train})

    n_done = sum(1 for c in cells if c["status"] == "done")
    n_run = sum(1 for c in cells if c["status"] == "running")
    n_fail = sum(1 for c in cells if c["status"] == "failed")
    n_pend = N_TOT - n_done - n_run - n_fail

    r_rows = []
    r_mean = []
    r_cats = [r_tag(r) for r in RADII]
    for r in RADII:
        vals = [
            c["auc"]
            for c in cells
            if abs(c["R"] - r) < 1e-9 and c["auc"] is not None and c["status"] == "done"
        ]
        n_r = sum(1 for c in cells if abs(c["R"] - r) < 1e-9 and c["status"] == "done")
        mean = statistics.mean(vals) if vals else None
        std = statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else None)
        k3 = K3_MEAN.get(r)
        delta = (mean - k3) if mean is not None and k3 is not None else None
        r_rows.append(
            {
                "R": r,
                "n_done": n_r,
                "n_total": N_PER_R,
                "mean": mean,
                "std": std,
                "k3": k3,
                "delta": delta,
            }
        )
        r_mean.append(mean)

    k3_series = [K3_MEAN.get(r) for r in RADII]

    best = None
    for c in cells:
        if c["auc"] is not None and c["status"] == "done":
            if best is None or c["auc"] > best["auc"]:
                best = {"auc": c["auc"], "R": c["R"], "shard": c["shard"], "seed": c["seed"]}

    table = []
    tones = []
    for c in sorted(cells, key=lambda x: (x["R"], x["shard"], x["seed"])):
        step_s = f"{c['step']}/{MAX_STEPS}" if c["step"] is not None else f"0/{MAX_STEPS}"
        table.append(
            [
                r_tag(c["R"]),
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
const JOB = {js(job)};
const RECIPE =
  "L4/D256 · η₀=0.1 · B=128×1 · 2500 steps · star-only (no kNN) · v2cpStar";
const N_DONE = {n_done};
const N_RUN = {n_run};
const N_PEND = {n_pend};
const N_FAIL = {n_fail};
const N_TOT = {N_TOT};
const BEST = {best_js};
const R_CATS: string[] = {js(r_cats)};
const STAR_MEAN = {js(series_data(r_mean))};
const K3_MEAN_S = {js(series_data(k3_series))};
const R_ROWS = {js(r_rows)};
const TABLE = {js(table)};
const TONE = {js(tones)};

function fmt(x: number | null | undefined, d = 4): string {{
  return x == null || Number.isNaN(Number(x)) ? "—" : Number(x).toFixed(d);
}}

export default function GraphRStarOnlyCompletePV2Live() {{
  const series = [
    {{
      name: "star-only mean val ROC",
      data: STAR_MEAN.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "accent" as const,
    }},
    {{
      name: "star+k=3 (panel b)",
      data: K3_MEAN_S.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "neutral" as const,
    }},
  ];

  return (
    <Stack gap={{20}}>
      <Stack gap={{6}}>
        <H1>Ablation — R★ sweep without kNN</H1>
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

      <Callout tone="neutral" title="Operating point + question">
        Frozen downstream OP: (k★, R★) = (2, 0.1) from L8 k-sweep + panel (b).
        This grid drops kNN entirely — if star-only tracks or beats star+k=3,
        the kNN path is not helping under CompleteP.
      </Callout>

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

      <Stack gap={{6}}>
        <H2>Mean val ROC vs R★ — star-only vs star+k=3</H2>
        <LineChart
          categories={{R_CATS}}
          series={{series as any}}
          height={{240}}
          beginAtZero={{false}}
        />
      </Stack>

      <Stack gap={{8}}>
        <H2>Per-R summary (Δ = star-only − star+k=3)</H2>
        <Table
          headers={{["R★", "done/6", "star-only", "std", "star+k=3", "Δ"]}}
          rows={{R_ROWS.map((m: any) => [
            String(m.R),
            `${{m.n_done}}/${{m.n_total}}`,
            fmt(m.mean),
            fmt(m.std),
            fmt(m.k3),
            fmt(m.delta),
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
