#!/usr/bin/env python3
"""Regenerate canvases/edge-eta-L6D512-k7-s40k-live.canvas.tsx."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

ETAS = (0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75)
MAX_STEPS = 40000
BATCH = 64
N_GPUS = 4
D_HEAD = 32
KNN_K = 7
R_TAG = "0p1"
DEPTH = 6
WIDTH = 512
RUN_TAG = f"edge-L6D512-b{BATCH}-g{N_GPUS}-s{MAX_STEPS}-dh{D_HEAD}-k{KNN_K}-R{R_TAG}"
JOB_NAME = "jc-edge-L6D512"
MODEL_PREFIX = f"capen-llama-att_{DEPTH}_{WIDTH}"

SWEEP = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "jetclass_output/adam/sweep_lr"
)
LOGDIR = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs")
OUT = Path(
    "/home/jdezoort/.cursor/projects/home-jdezoort/canvases/"
    "edge-eta-L6D512-k7-s40k-live.canvas.tsx"
)
WATCH_LOG = Path.home() / ".cache/jc_edge_L6D512_watchdogs/nohup_edge_L6D512.out"


def run(cmd: list[str], timeout: int = 40) -> str:
    try:
        return subprocess.check_output(cmd, text=True, timeout=timeout)
    except Exception:
        return ""


def js(obj) -> str:
    return json.dumps(obj)


def eta_tag(e: float) -> str:
    return f"{e:g}".replace(".", "p")


def parse_log(text: str) -> tuple[int | None, float | None, float | None, float | None, bool]:
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
    loss = None
    lm = re.findall(r"val_loss=([0-9.eE+-]+)", text)
    if lm:
        try:
            loss = float(lm[-1])
        except ValueError:
            pass
    finished = step is not None and step >= MAX_STEPS - 5
    return step, train, auc, loss, finished


def parquet_metrics(eta: float) -> tuple[int, float | None, float | None, float | None]:
    et = eta_tag(eta)
    best_step = 0
    auc = loss = acc = None
    for p in SWEEP.glob(f"{MODEL_PREFIX}_{et}_*{RUN_TAG}*"):
        pqpath = p if p.suffix == ".parquet" else p.parent / f"{p.name}.parquet"
        if not pqpath.is_file():
            continue
        try:
            import pyarrow.parquet as pq

            df = pq.read_table(
                pqpath,
                columns=["global_step", "val_roc_auc", "val_acc", "val_loss"],
            ).to_pandas()
        except Exception:
            continue
        if df.empty:
            continue
        step = int(df["global_step"].max())
        best_step = max(best_step, step)
        sub = df.dropna(subset=["val_roc_auc"])
        if sub.empty:
            continue
        last = sub.loc[sub["global_step"].idxmax()]
        if step >= best_step:
            auc = float(last["val_roc_auc"])
            if "val_loss" in last and last["val_loss"] == last["val_loss"]:
                loss = float(last["val_loss"])
            if "val_acc" in last and last["val_acc"] == last["val_acc"]:
                acc = float(last["val_acc"])
    return best_step, auc, loss, acc


def latest_log_for_eta(eta: float) -> Path | None:
    needle = f"eta0={eta:g}"
    alts = {needle, f"eta0={float(eta)}", f"eta0={eta}"}
    hits: list[Path] = []
    for p in LOGDIR.glob("jc_edge_L6D512_s40k_gt_*.out"):
        try:
            head = p.read_text(errors="ignore")[:4000]
        except Exception:
            continue
        if any(a in head for a in alts):
            hits.append(p)
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def main() -> None:
    queue: dict[int, dict] = {}
    for line in run(
        ["squeue", "-u", "jdezoort", "-n", JOB_NAME, "-h", "-o", "%i %T %M %R %P"]
    ).splitlines():
        parts = line.split(None, 4)
        if len(parts) < 2:
            continue
        jid, state = parts[0], parts[1].lower()
        elapsed = parts[2] if len(parts) > 2 else ""
        reason = parts[3] if len(parts) > 3 else ""
        part = parts[4] if len(parts) > 4 else ""
        m = re.search(r"_(\d+)$|_\[(\d+)\]$", jid)
        if not m:
            continue
        tid = int(m.group(1) or m.group(2))
        queue[tid] = {
            "state": state,
            "time": elapsed,
            "reason": reason,
            "part": part,
            "jid": jid,
        }

    submitted: dict[int, str] = {}
    if WATCH_LOG.is_file():
        for m in re.finditer(
            r"submitted task=(\d+) eta=([0-9.]+) job=(\d+)",
            WATCH_LOG.read_text(errors="ignore"),
        ):
            submitted[int(m.group(1))] = m.group(3)

    cells = []
    for task, eta in enumerate(ETAS):
        step_pq, auc_pq, loss_pq, acc_pq = parquet_metrics(eta)
        step = step_pq or None
        train = auc = loss = acc = None
        auc, loss, acc = auc_pq, loss_pq, acc_pq
        finished = step is not None and step >= MAX_STEPS - 5

        logp = latest_log_for_eta(eta)
        if logp is not None:
            s, tr, a, lo, fin = parse_log(logp.read_text(errors="ignore")[-200_000:])
            if s is not None:
                step = max(step or 0, s)
            if tr is not None:
                train = tr
            if a is not None:
                auc = a
            if lo is not None:
                loss = lo
            finished = (step is not None and step >= MAX_STEPS - 5) or fin

        q = queue.get(task)
        if q and "run" in q["state"]:
            status = "running"
        elif q and q["state"] in {"pending", "pd", "cf", "cg"}:
            status = "pending"
        elif step is not None and step >= MAX_STEPS - 5:
            status = "done"
        elif step and step > 0:
            status = "paused"
        else:
            status = "pending"

        cells.append(
            {
                "task": task,
                "eta": eta,
                "status": status,
                "step": step or 0,
                "train": train,
                "auc": auc,
                "loss": loss,
                "acc": acc,
                "jid": (q or {}).get("jid") or submitted.get(task, "—"),
                "time": (q or {}).get("time", ""),
                "part": (q or {}).get("part", ""),
            }
        )

    n_done = sum(1 for c in cells if c["status"] == "done")
    n_run = sum(1 for c in cells if c["status"] == "running")
    n_pend = sum(1 for c in cells if c["status"] == "pending")
    n_paused = sum(1 for c in cells if c["status"] == "paused")
    n_tot = len(ETAS)

    best = None
    for c in cells:
        if c["auc"] is not None:
            if best is None or c["auc"] > best["auc"]:
                best = {"auc": c["auc"], "eta": c["eta"], "step": c["step"]}

    cats = [f"{e:g}" for e in ETAS]
    auc_series = [c["auc"] if c["auc"] is not None else None for c in cells]
    step_frac = [100.0 * c["step"] / MAX_STEPS for c in cells]

    table = []
    tones = []
    for c in cells:
        table.append(
            [
                f"{c['eta']:g}",
                c["status"],
                f"{c['step']}/{MAX_STEPS}",
                f"{100.0 * c['step'] / MAX_STEPS:.0f}%",
                "—" if c["train"] is None else f"{c['train']:.3f}",
                "—" if c["loss"] is None else f"{c['loss']:.4f}",
                "—" if c["auc"] is None else f"{c['auc']:.4f}",
                str(c["jid"]),
            ]
        )
        tones.append(
            {
                "done": "success",
                "running": "info",
                "pending": "neutral",
                "paused": "warning",
            }.get(c["status"], "neutral")
        )

    updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    job_ids = sorted(
        {
            str(c["jid"]).split("_", 1)[0]
            for c in cells
            if str(c["jid"]) not in {"—", "-", "None"}
            and str(c["jid"]).split("_", 1)[0].isdigit()
        }
    )
    job_str = ",".join(job_ids) if job_ids else "—"
    eta_set = "{" + ", ".join(f"{e:g}" for e in ETAS) + "}"

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
const JOBS = {js(job_str)};
const RECIPE =
  "L6/D512/H16 · all-seed star kin7 + ΔR-kNN noloop · (k★,R★)=(7,0.1) · B=64×4 · 40k · n_train=10.24M unique · tag `{RUN_TAG}`";
const N_DONE: number = {n_done};
const N_RUN: number = {n_run};
const N_PEND: number = {n_pend};
const N_PAUSED: number = {n_paused};
const N_TOT: number = {n_tot};
const BEST = {js(best)};
const ETA_CATS: string[] = {js(cats)};
const AUC_SERIES = {js(auc_series)};
const STEP_FRAC = {js(step_frac)};
const TABLE = {js(table)};
const TONE = {js(tones)};

export default function EdgeEtaL6D512K7S40kLive() {{
  const hasAuc = AUC_SERIES.some((v) => v != null);
  const aucSeries = [
    {{
      name: "latest val ROC",
      data: AUC_SERIES.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "accent" as const,
    }},
  ];
  const progSeries = [
    {{
      name: "% of 40k steps",
      data: STEP_FRAC.map((v) => Number(v)),
      tone: "info" as const,
    }},
  ];

  return (
    <Stack gap={{20}}>
      <Stack gap={{6}}>
        <H1>Panel (c) edge η₀ — L6/D512 @ k★=7</H1>
        <Text tone="secondary">{{RECIPE}}</Text>
        <Text tone="secondary">
          gputest rolling watchdog (≤3 shared) · jobs {{JOBS}} · updated {{UPDATED}}
        </Text>
        <Row gap={{8}} wrap>
          <Pill tone={{N_DONE === N_TOT ? "success" : "info"}}>
            {{N_DONE}}/{{N_TOT}} done
          </Pill>
          <Pill tone="info">{{N_RUN}} running</Pill>
          <Pill tone="warning">{{N_PAUSED}} between slots</Pill>
          <Pill tone="neutral">{{N_PEND}} pending</Pill>
        </Row>
      </Stack>

      <Grid columns={{4}} gap={{12}}>
        <Stat
          value={{`${{N_DONE}}/${{N_TOT}}`}}
          label="η cells done"
          tone={{N_DONE === N_TOT ? "success" : "info"}}
        />
        <Stat value={{String(N_RUN)}} label="running now" />
        <Stat value={{String(N_PAUSED + N_PEND)}} label="waiting / paused" />
        <Stat
          value={{BEST == null ? "—" : BEST.auc.toFixed(4)}}
          label={{
            BEST == null
              ? "Best val AUC so far"
              : `Best · η₀=${{BEST.eta}} @ step ${{BEST.step}}`
          }}
        />
      </Grid>

      {{N_DONE < N_TOT ? (
        <Callout tone="info" title="Rolling on gputest">
          Each 59m slot resumes last.ckpt toward 40k steps. Watchdog fills free
          slots from η ∈ {{{js(eta_set)}}}. Shares MaxJobsPU≈3 with the L4/D256
          campaign. Single-pass stream: n_train = 40k × 256 = 10.24M unique jets.
        </Callout>
      ) : (
        <Callout tone="success" title="Sweep complete">
          All eight η₀ cells reached 40k steps.
        </Callout>
      )}}

      <Stack gap={{6}}>
        <H2>Progress toward 40k (%)</H2>
        <LineChart
          categories={{ETA_CATS}}
          series={{progSeries as any}}
          height={{200}}
          beginAtZero
        />
        <Text tone="secondary" size="small">
          X = η₀ · Y = percent of 40k target · Source: parquet + Slurm logs
        </Text>
      </Stack>

      {{hasAuc ? (
        <Stack gap={{6}}>
          <H2>Latest val ROC vs η₀</H2>
          <LineChart
            categories={{ETA_CATS}}
            series={{aucSeries as any}}
            height={{220}}
            beginAtZero={{false}}
          />
          <Text tone="secondary" size="small">
            Incomplete cells omit a point until the first val checkpoint lands
          </Text>
        </Stack>
      ) : null}}

      <Stack gap={{8}}>
        <H2>Cells</H2>
        <Table
          headers={{["η₀", "status", "step", "%", "train", "val loss", "val ROC", "job"]}}
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
        f"wrote {OUT} done={n_done} run={n_run} paused={n_paused} "
        f"pend={n_pend} best={best}"
    )


if __name__ == "__main__":
    main()
