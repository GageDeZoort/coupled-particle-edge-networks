#!/usr/bin/env python3
"""Regenerate canvases/mup-s5000-live.canvas.tsx from Slurm + parquet state."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

AILAB = "15276242"
DELLA = "15276243"
TARGET = 5000
ETAS = [0.01, 0.05, 0.075, 0.25, 0.75]
LOGDIR = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs")
ROOT = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "jetclass_output/adam/sweep_lr"
)
OUT = Path("/home/jdezoort/.cursor/projects/home-jdezoort/canvases/mup-s5000-live.canvas.tsx")
ETA_TAG = {
    0.01: "0p01",
    0.05: "0p05",
    0.075: "0p075",
    0.25: "0p25",
    0.75: "0p75",
}


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, timeout=45)
    except Exception:
        return ""


def js(obj) -> str:
    return json.dumps(obj)


def main() -> None:
    acct: dict[str, dict] = {}
    for line in run(
        [
            "sacct",
            "-j",
            f"{AILAB},{DELLA}",
            "-n",
            "-X",
            "--format=JobID%18,Partition%10,State%12,Elapsed",
        ]
    ).splitlines():
        p = line.split()
        if len(p) >= 4:
            acct[p[0]] = {"partition": p[1], "state": p[2], "elapsed": p[3]}

    queue: dict[str, dict] = {}
    for line in run(
        ["squeue", "-j", f"{AILAB},{DELLA}", "-h", "-o", "%i %T %M %R"]
    ).splitlines():
        parts = line.split(None, 3)
        if len(parts) >= 3:
            queue[parts[0]] = {
                "state": parts[1],
                "time": parts[2],
                "reason": parts[3] if len(parts) > 3 else "",
            }

    def live_step(side: str, task: int):
        jid = AILAB if side == "ailab" else DELLA
        pref = "a8" if side == "ailab" else "gt"
        path = LOGDIR / f"jc_mup_s5000_{pref}_{jid}_{task}.out"
        if not path.exists():
            return None, False
        text = path.read_text(errors="ignore")
        steps = re.findall(r"global_step=(\d+)", text)
        return (int(steps[-1]) if steps else None), ("done L=" in text)

    def metrics(depth: int, width: int, eta: float):
        tag = ETA_TAG[eta]
        hits = [
            p
            for p in ROOT.glob(
                f"capen-llama-att_{depth}_{width}_{tag}_*mup-check-s5000*.parquet"
            )
            if "D2M" in p.name or "D2p" in p.name
        ]
        if not hits:
            return None
        path = max(hits, key=lambda x: x.stat().st_mtime)
        df = pq.read_table(
            path,
            columns=["global_step", "train_loss", "val_loss", "val_roc_auc", "val_acc"],
        ).to_pandas()
        out: dict = {"max_step": int(df.global_step.max())}
        for col, key in [
            ("train_loss", "train"),
            ("val_loss", "val"),
            ("val_roc_auc", "auc"),
            ("val_acc", "acc"),
        ]:
            d = df[["global_step", col]].dropna()
            if len(d):
                r = d.sort_values("global_step").iloc[-1]
                out[key] = float(r[col])
        vd = df[["global_step", "val_roc_auc"]].dropna().sort_values("global_step")
        out["auc_steps"] = [int(x) for x in vd.global_step.tolist()]
        out["auc_vals"] = [float(x) for x in vd.val_roc_auc.tolist()]
        return out

    def job_state(jid: str):
        if jid in queue:
            return queue[jid]["state"], queue[jid]["time"], queue[jid].get("reason", "")
        if jid in acct:
            return acct[jid]["state"], acct[jid]["elapsed"], ""
        return None, "", ""

    rows = []
    for t in range(10):
        depth, width = (4, 256) if t < 5 else (8, 1024)
        eta = ETAS[t % 5]
        ds, dt, dr = job_state(f"{DELLA}_{t}")
        als, alt, alr = job_state(f"{AILAB}_{t}")
        d_step, d_done = live_step("della", t)
        a_step, a_done = live_step("ailab", t)
        met = metrics(depth, width, eta)

        status = "pending"
        step = None
        side = "—"
        elapsed = ""
        reason = (dr or alr or "").replace("(", "").replace(")", "")

        if ds == "RUNNING":
            status, step, side, elapsed = "running", d_step, "della", dt
        elif als == "RUNNING":
            status, step, side, elapsed = "running", a_step, "ailab", alt
        elif met and met.get("max_step", 0) >= TARGET:
            status, step = "done", met["max_step"]
            if d_done or (ds and "COMPLETED" in str(ds)):
                side, elapsed = "della", dt
            elif a_done or (als and "COMPLETED" in str(als)):
                side, elapsed = "ailab", alt
        elif d_done or (ds and "COMPLETED" in str(ds)):
            status, step, side, elapsed = "done", d_step, "della", dt
        elif a_done or (als and "COMPLETED" in str(als)):
            status, step, side, elapsed = "done", a_step, "ailab", alt
        elif ds == "PENDING" or als == "PENDING":
            status = "pending"
            if ds == "PENDING" and als == "PENDING":
                side = "both"
            elif ds == "PENDING":
                side = "della"
            else:
                side = "ailab"

        rows.append(
            {
                "task": t,
                "label": f"L{depth}/D{width}",
                "eta": eta,
                "status": status,
                "side": side,
                "step": step,
                "elapsed": elapsed,
                "reason": reason,
                "train": None if not met else met.get("train"),
                "val": None if not met else met.get("val"),
                "auc": None if not met else met.get("auc"),
                "acc": None if not met else met.get("acc"),
                "auc_steps": None if not met else met.get("auc_steps"),
                "auc_vals": None if not met else met.get("auc_vals"),
            }
        )

    n_done = sum(1 for r in rows if r["status"] == "done")
    n_run = sum(1 for r in rows if r["status"] == "running")
    n_pend = sum(1 for r in rows if r["status"] == "pending")

    def chart_block(depth: int):
        cats = sorted(
            {
                s
                for r in rows
                if r["label"].startswith(f"L{depth}") and r["auc_steps"]
                for s in r["auc_steps"]
            }
        )
        if not cats:
            return [], []
        if len(cats) > 24:
            idxs = [round(i * (len(cats) - 1) / 23) for i in range(24)]
            cats = [cats[i] for i in idxs]
        series = []
        for r in rows:
            if not r["label"].startswith(f"L{depth}") or not r["auc_steps"]:
                continue
            full = dict(zip(r["auc_steps"], r["auc_vals"]))
            data = []
            last = r["auc_vals"][0]
            for c in cats:
                prev = [s for s in r["auc_steps"] if s <= c]
                if prev:
                    last = full[prev[-1]]
                data.append(float(last))
            series.append({"name": f"η={r['eta']:g}", "data": data})
        return [str(c) for c in cats], series

    c4, s4 = chart_block(4)
    c8, s8 = chart_block(8)

    def fmt(x, nd=3):
        return "—" if x is None else f"{x:.{nd}f}"

    def step_str(r):
        if r["step"] is None:
            return f"0/{TARGET}"
        return f"{r['step']}/{TARGET}"

    table = []
    tones = []
    for r in rows:
        table.append(
            [
                r["label"],
                f"{r['eta']:g}",
                r["status"],
                r["side"],
                step_str(r),
                fmt(r["train"]),
                fmt(r["val"]),
                fmt(r["auc"]),
                fmt(r["acc"]),
            ]
        )
        tones.append(
            {
                "done": "success",
                "running": "info",
                "pending": "neutral",
                "cancelled": "warning",
            }.get(r["status"], "neutral")
        )

    done_aucs = [
        (r["auc"], r["label"], r["eta"])
        for r in rows
        if r["status"] == "done" and r["auc"] is not None
    ]
    best = max(done_aucs) if done_aucs else None
    updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    best_js = "null"
    if best is not None:
        best_js = js({"auc": best[0], "label": best[1], "eta": best[2]})

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
const AILAB = {js(AILAB)};
const DELLA = {js(DELLA)};
const TARGET = {TARGET};
const N_DONE = {n_done};
const N_RUN = {n_run};
const N_PEND = {n_pend};
const N_TOT = 10;
const BEST = {best_js};
const C4: string[] = {js(c4)};
const S4 = {js(s4)};
const C8: string[] = {js(c8)};
const S8 = {js(s8)};
const TABLE = {js(table)};
const TONE = {js(tones)};

export default function MupS5000Live() {{
  const s4 = S4.map((s: any) => ({{
    name: s.name,
    data: (s.data as (number | null)[]).map((v) =>
      v == null ? null : Number(v),
    ),
  }}));
  const s8 = S8.map((s: any) => ({{
    name: s.name,
    data: (s.data as (number | null)[]).map((v) =>
      v == null ? null : Number(v),
    ),
  }}));

  return (
    <Stack gap={{20}}>
      <Stack gap={{6}}>
        <H1>μP check · 5k steps</H1>
        <Text tone="secondary">
          CompleteP · Adam · (k★,R★)=(6,0.1) · B=256 · n_train=2M · ailab↔Della
          race
        </Text>
        <Text tone="secondary">
          Jobs {{AILAB}} (ailab) / {{DELLA}} (gputest) · updated {{UPDATED}}
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
        <Stat
          value={{String(N_DONE)}}
          label="Done @ 5k"
          tone={{N_DONE === N_TOT ? "success" : "info"}}
        />
        <Stat value={{String(N_RUN)}} label="Running" />
        <Stat value={{String(N_PEND)}} label="Pending" />
        <Stat
          value={{BEST == null ? "—" : BEST.auc.toFixed(3)}}
          label={{
            BEST == null
              ? "Best val AUC"
              : `Best · ${{BEST.label}} η=${{BEST.eta}}`
          }}
        />
      </Grid>

      {{N_DONE < N_TOT ? (
        <Callout tone="info" title="In progress">
          Fresh D2M runs (prior D1M attempt capped at ~3907 steps). Race cancels
          the slower side per task.
        </Callout>
      ) : (
        <Callout tone="success" title="Grid complete">
          All 10 cells reached 5000 optimizer steps.
        </Callout>
      )}}

      {{C4.length > 0 ? (
        <Stack gap={{6}}>
          <H2>L4 / D256 — val ROC-AUC vs step</H2>
          <LineChart
            categories={{C4}}
            series={{s4 as any}}
            height={{220}}
            beginAtZero={{false}}
          />
          <Text tone="secondary">
            Source: JetClass stream parquets · D2M · s5000
          </Text>
        </Stack>
      ) : null}}

      {{C8.length > 0 ? (
        <Stack gap={{6}}>
          <H2>L8 / D1024 — val ROC-AUC vs step</H2>
          <LineChart
            categories={{C8}}
            series={{s8 as any}}
            height={{220}}
            beginAtZero={{false}}
          />
          <Text tone="secondary">
            Source: JetClass stream parquets · D2M · s5000
          </Text>
        </Stack>
      ) : null}}

      <Stack gap={{8}}>
        <H2>Cells</H2>
        <Table
          headers={{[
            "arch",
            "η₀",
            "status",
            "side",
            "step",
            "train",
            "val",
            "AUC",
            "acc",
          ]}}
          rows={{TABLE}}
          rowTone={{TONE as any}}
        />
      </Stack>
    </Stack>
  );
}}
"""
    OUT.write_text(canvas)
    print(f"wrote {OUT} done={n_done} run={n_run} pend={n_pend}")


if __name__ == "__main__":
    main()
