#!/usr/bin/env python3
"""Live canvas: k=1 self-loop-only @ R★=0.1 (L4/D256 + L8/D1024 matched steps)."""
from __future__ import annotations

import json
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

# L4: B=128×1 × 2500; L8 fair: B=32×4 × 2500 (both 320k jets, 2500 steps)
ARCHS = (
    {
        "key": "L4/D256",
        "depth": 4,
        "width": 256,
        "batch": 128,
        "gpus": 1,
        "steps": 2500,
        "tag": "v2cpK1",
        "job_file": "graph_k1_L4_v2cp_jobid.txt",
        "log_prefix": "jc_k1_L4",
    },
    {
        "key": "L8/D1024",
        "depth": 8,
        "width": 1024,
        "batch": 32,
        "gpus": 4,
        "steps": 2500,
        "tag": "v2cpK1b",
        "job_file": "graph_k1_L8_v2cp_jobid.txt",
        "log_prefix": "jc_k1_L8",
    },
)
N_SHARD = 3
N_INIT = 2
N_PER = N_SHARD * N_INIT

REF = {"star-only": 0.9330, "k=2": 0.9419, "k=3": 0.9427}

SWEEP = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "jetclass_output/adam/sweep_lr"
)
LOGDIR = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs")
OUT = Path(
    "/home/jdezoort/.cursor/projects/home-jdezoort/canvases/"
    "graph-k1-self-completep-v2-live.canvas.tsx"
)
HERE = Path(__file__).resolve().parent


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, timeout=45)
    except Exception:
        return ""


def js(obj) -> str:
    return json.dumps(obj)


def load_job(name: str) -> str:
    p = HERE / name
    return p.read_text().strip() if p.is_file() else ""


def parse_job_states(job: str) -> dict[int, dict]:
    """Map array task id -> {state, elapsed} from squeue + sacct."""
    out: dict[int, dict] = {}
    if not job:
        return out
    for line in run(["squeue", "-j", job, "-h", "-o", "%i %T %M"]).splitlines():
        parts = line.split(None, 2)
        if len(parts) < 2:
            continue
        jid, state = parts[0], parts[1].lower()
        elapsed = parts[2] if len(parts) > 2 else ""
        if "_" in jid:
            try:
                tid = int(jid.rsplit("_", 1)[-1])
            except ValueError:
                continue
            out[tid] = {"state": state, "elapsed": elapsed}
    for line in run(
        ["sacct", "-j", job, "-n", "-X", "--format=JobID%20,State%14,Elapsed"]
    ).splitlines():
        p = line.split()
        if len(p) < 2:
            continue
        jid, st = p[0], p[1]
        if "_" not in jid:
            continue
        try:
            tid = int(jid.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if tid in out:
            continue  # prefer live queue
        if st.startswith("COMPLETED"):
            state = "done"
        elif "FAIL" in st:
            state = "failed"
        elif "CANCEL" in st:
            state = "cancelled"
        elif "PENDING" in st or "RUNNING" in st:
            state = st.lower()
        else:
            state = st.lower()
        out[tid] = {"state": state, "elapsed": p[2] if len(p) > 2 else ""}
    for d in out.values():
        if d["state"] == "completed":
            d["state"] = "done"
    return out


def find_parquet(arch: dict, shard: int, seed: int) -> Path | None:
    tag = arch["tag"]
    depth = arch["depth"]
    batch = arch["batch"]
    gpus = arch["gpus"]
    steps = arch["steps"]
    patterns = [
        f"*knn1*graph-k1self-L{depth}-b{batch}-g{gpus}-s{steps}-R0p1-shard{shard}-seed{seed}-{tag}.parquet",
        f"*graph-k1self-L{depth}-b{batch}-g{gpus}-s{steps}*shard{shard}-seed{seed}-{tag}.parquet",
    ]
    best = None
    for pat in patterns:
        for p in SWEEP.glob(pat):
            if "knn1" not in p.name:
                continue
            if best is None or p.stat().st_mtime > best.stat().st_mtime:
                best = p
    return best


def main() -> None:
    rows_summary = []
    table = []
    tones = []
    n_done = n_run = n_fail = n_pend = 0
    jobs: dict[str, str] = {}

    for arch in ARCHS:
        job = load_job(arch["job_file"])
        jobs[arch["key"]] = job or "—"
        states = parse_job_states(job)
        cells_auc = []

        for t in range(N_PER):
            shard = t // N_INIT
            seed = t % N_INIT
            st_info = states.get(t, {"state": "pending", "elapsed": ""})
            state = st_info["state"]
            step = None
            auc = None
            train = None

            log = LOGDIR / f"{arch['log_prefix']}_{job}_{t}.out" if job else None
            if log is not None and log.exists():
                text = log.read_text(errors="ignore")
                steps_f = re.findall(r"global_step=(\d+)", text)
                if steps_f:
                    step = int(steps_f[-1])
                if "done L=" in text and state == "running":
                    state = "done"

            path = find_parquet(arch, shard, seed)
            if path is not None:
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
                        if step >= arch["steps"] - 5:
                            state = "done"
                except Exception:
                    pass

            if state == "done":
                n_done += 1
                if auc is not None:
                    cells_auc.append(auc)
            elif state == "running":
                n_run += 1
            elif state == "failed":
                n_fail += 1
            else:
                n_pend += 1

            step_s = f"{step}/{arch['steps']}" if step is not None else f"0/{arch['steps']}"
            table.append(
                [
                    arch["key"],
                    str(shard),
                    str(seed),
                    state,
                    step_s,
                    "—" if train is None else f"{train:.3f}",
                    "—" if auc is None else f"{auc:.4f}",
                ]
            )
            tones.append(
                {
                    "done": "success",
                    "running": "info",
                    "pending": "neutral",
                    "failed": "warning",
                    "cancelled": "warning",
                }.get(state, "neutral")
            )

        mean = statistics.mean(cells_auc) if cells_auc else None
        std = (
            statistics.stdev(cells_auc)
            if len(cells_auc) > 1
            else (0.0 if cells_auc else None)
        )
        rows_summary.append(
            {
                "arch": arch["key"],
                "n": len(cells_auc),
                "mean": mean,
                "std": std,
                "vs_star": (mean - REF["star-only"]) if mean is not None else None,
                "vs_k2": (mean - REF["k=2"]) if mean is not None else None,
            }
        )

    n_tot = 2 * N_PER
    updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    l4_mean = rows_summary[0]["mean"]
    l8_mean = rows_summary[1]["mean"]
    cats = ["star-only", "k=1 L4", "k=1 L8", "k=2 L4", "k=3 L4"]
    means = [REF["star-only"], l4_mean, l8_mean, REF["k=2"], REF["k=3"]]

    # Avoid optional chaining / danger tone — some canvas hosts are picky.
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
const JOB_L4 = {js(jobs.get("L4/D256", "—"))};
const JOB_L8 = {js(jobs.get("L8/D1024", "—"))};
const N_DONE = {n_done};
const N_RUN = {n_run};
const N_PEND = {n_pend};
const N_FAIL = {n_fail};
const N_TOT = {n_tot};
const L4_MEAN = {js(l4_mean)};
const L8_MEAN = {js(l8_mean)};
const SUMMARY = {js(rows_summary)};
const TABLE = {js(table)};
const TONE = {js(tones)};
const CATS: string[] = {js(cats)};
const MEANS = {js(means)};

function fmt(x: number | null | undefined, d = 4): string {{
  if (x == null || Number.isNaN(Number(x))) return "—";
  return Number(x).toFixed(d);
}}

export default function GraphK1SelfCompletePV2Live() {{
  const series = [
    {{
      name: "mean val ROC @ R★=0.1",
      data: MEANS.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "accent" as const,
    }},
  ];

  return (
    <Stack gap={{20}}>
      <Stack gap={{6}}>
        <H1>Control — k=1 self-loops only</H1>
        <Text tone="secondary">
          R★=0.1 · η₀=0.1 · 320k jets · 2500 steps both arches · updated {{UPDATED}}
        </Text>
        <Text tone="secondary">
          L4 job {{JOB_L4}} (B=128×1) · L8 job {{JOB_L8}} (B=32×4, matched steps)
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

      <Callout tone="info" title="Fair comparison">
        L4: B=128×1 × 2500 steps. L8/D1024: B=32×4 × 2500 steps (same 320k
        jets and same # updates). Prior L8 recipe used B=64×4 × 1250 — fewer
        steps; cancelled and resubmitted as v2cpK1b.
      </Callout>

      <Callout tone="neutral" title="What k=1 means">
        ΔR-kNN includes self at ΔR=0, so k=1 is self-loops only (no true
        neighbors). Expect near star-only (~0.933) if neighbors are the
        signal; near k=2 (~0.942) would be surprising.
      </Callout>

      <Grid columns={{4}} gap={{12}}>
        <Stat value={{`${{N_DONE}}/${{N_TOT}}`}} label="cells done" />
        <Stat value={{fmt(L4_MEAN)}} label="L4/D256 k=1 mean" />
        <Stat value={{fmt(L8_MEAN)}} label="L8/D1024 k=1 mean" />
        <Stat value="0.9330" label="L4 star-only ref" />
      </Grid>

      <Stack gap={{6}}>
        <H2>Mean val ROC vs L4 references @ R★=0.1</H2>
        <LineChart
          categories={{CATS}}
          series={{series as any}}
          height={{220}}
          beginAtZero={{false}}
        />
        <Text tone="secondary">
          Source: v2cpK1 / v2cpK1b parquets · refs from CompleteP panel (a) /
          star-only ablation
        </Text>
      </Stack>

      <Stack gap={{8}}>
        <H2>Summary (Δ vs L4 star-only / k=2)</H2>
        <Table
          headers={{["arch", "n/6", "mean", "std", "Δ star-only", "Δ k=2"]}}
          rows={{SUMMARY.map((m: any) => [
            m.arch,
            String(m.n),
            fmt(m.mean),
            fmt(m.std),
            fmt(m.vs_star),
            fmt(m.vs_k2),
          ])}}
        />
      </Stack>

      <Stack gap={{8}}>
        <H2>Cells</H2>
        <Table
          headers={{["arch", "shard", "seed", "status", "step", "train", "AUC"]}}
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
