#!/usr/bin/env python3
"""Live canvas: panel (c) M11 + edge η₀×D @ 2500, seeds 0 and 1.

Rewrites:
  ~/.cursor/projects/home-jdezoort/canvases/m11-eta-panel-c-live.canvas.tsx
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
CANVAS = Path(
    "/home/jdezoort/.cursor/projects/home-jdezoort/canvases/"
    "m11-eta-panel-c-live.canvas.tsx"
)
JSON_OUT = REPO / "paper" / "data" / "m11_eta_panel_c_live.json"
CSV_OUT = REPO / "paper" / "data" / "m11_eta_panel_c_live_status.csv"

ROOT = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law")
SWEEP = ROOT / "jetclass_output" / "adam" / "sweep_lr"
LOGS = ROOT / "logs"
MAX_STEPS = 2500
ETAS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
WIDTHS = (256, 384, 512)
SEEDS = (0, 1)
ET = ZoneInfo("America/New_York")

# Per (kind, seed): job names + log prefixes. Seed0 edge D384 η0.1 also on jc-edge-rerun.
ARMS = {
    ("m11", 0): dict(
        tag="m11-b96-g4-s2500-dh32",
        model="baseline-transformer",
        gt_job="jc-m11-gt",
        gt_log="jc_m11_s2500_gt",
        ailab_job="jc-m11-a512",
        ailab_log="jc_m11_s2500_ailab",
    ),
    ("m11", 1): dict(
        tag="m11-b96-g4-s2500-dh32",
        model="baseline-transformer",
        gt_job="jc-m11-gt-s1",
        gt_log="jc_m11_s2500_gt_s1",
        ailab_job="jc-m11-a512-s1",
        ailab_log="jc_m11_s2500_ailab_s1",
    ),
    ("edge", 0): dict(
        tag="edge-b96-g4-s2500-dh32-k6-R0p1",
        model="capen-llama-att",
        gt_job="jc-edge-gt",
        gt_log="jc_edge_s2500_gt",
        ailab_job="jc-edge-a512",
        ailab_log="jc_edge_s2500_ailab",
        # D=384 η₀=0.1 wiped+rerun
        alt_gt_jobs=("jc-edge-rerun",),
        alt_gt_logs=("jc_edge_s2500_gt_rerun",),
    ),
    ("edge", 1): dict(
        tag="edge-b96-g4-s2500-dh32-k6-R0p1",
        model="capen-llama-att",
        gt_job="jc-edge-gt-s1",
        gt_log="jc_edge_s2500_gt_s1",
        ailab_job="jc-edge-a512-s1",
        ailab_log="jc_edge_s2500_ailab_s1",
    ),
}


def _js(x) -> str:
    return json.dumps(x)


def _eta_tag(eta: float) -> str:
    return f"{eta:g}".replace(".", "p")


def _squeue() -> dict[str, list[dict]]:
    """job_name → list of {job_id, task, state, time, node}."""
    out: dict[str, list[dict]] = {}
    try:
        proc = subprocess.run(
            [
                "squeue",
                "-u",
                "jdezoort",
                "-h",
                "-o",
                "%i %j %T %M %R %a",
            ],
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
        )
    except Exception:
        return out

    def _add(name: str, base: str, task: int | None, state: str, elapsed: str, node: str) -> None:
        full = f"{base}_{task}" if task is not None else base
        out.setdefault(name, []).append(
            dict(
                job_id=base,
                full_id=full,
                task=task,
                state=state,
                elapsed=elapsed,
                node=node,
            )
        )

    for line in (proc.stdout or "").splitlines():
        parts = line.split(None, 5)
        if len(parts) < 4:
            continue
        jid, name, state, elapsed = parts[0], parts[1], parts[2], parts[3]
        node = parts[4] if len(parts) > 4 else ""
        m = re.match(r"^(\d+)_\[(.+)\]$", jid)
        if m:
            base, spec = m.group(1), m.group(2)
            tasks: list[int] = []
            for chunk in spec.split(","):
                chunk = chunk.strip()
                if "-" in chunk:
                    a, b = chunk.split("-", 1)
                    tasks.extend(range(int(a), int(b) + 1))
                elif chunk.isdigit():
                    tasks.append(int(chunk))
            for t in tasks:
                _add(name, base, t, state, elapsed, node)
            continue
        task = None
        base = jid
        if "_" in jid:
            base, task_s = jid.rsplit("_", 1)
            if task_s.isdigit():
                task = int(task_s)
            else:
                base = jid
        _add(name, base, task, state, elapsed, node)
    return out


def _match_queue(sq: dict, job_names: tuple[str, ...], task: int) -> dict | None:
    for name in job_names:
        for ent in sq.get(name, []):
            if ent.get("task") == task:
                return ent
    return None


def _scrape_logs(log_prefixes: tuple[str, ...], base: str | None, task: int, step: int, best_roc):
    for prefix in log_prefixes:
        paths = []
        if base:
            paths.append(LOGS / f"{prefix}_{base}_{task}.out")
        paths.extend(sorted(LOGS.glob(f"{prefix}_*_{task}.out"))[-3:])
        for log in paths:
            if not log.exists():
                continue
            try:
                text = log.read_bytes()[-200_000:].decode("utf-8", errors="ignore")
                for m in re.finditer(r"global_step=(\d+)", text):
                    step = max(step, int(m.group(1)))
                for m in re.finditer(
                    r"\[val\].*?step=(\d+).*?val_roc_auc=([0-9.]+)", text
                ):
                    step = max(step, int(m.group(1)))
                    roc = float(m.group(2))
                    best_roc = roc if best_roc is None else max(best_roc, roc)
                if "Traceback" in text and step < MAX_STEPS:
                    return step, best_roc, "failed"
            except Exception:
                pass
    return step, best_roc, None


def _scrape_cell(kind: str, seed: int, width: int, eta: float, sq: dict) -> dict:
    arm = ARMS[(kind, seed)]
    et = _eta_tag(eta)
    if width in (256, 384):
        w_idx = {256: 0, 384: 1}[width]
        task = w_idx * len(ETAS) + list(ETAS).index(eta)
        job_names = (arm["gt_job"],) + tuple(arm.get("alt_gt_jobs") or ())
        log_prefixes = (arm["gt_log"],) + tuple(arm.get("alt_gt_logs") or ())
        host = "gputest"
    else:
        task = list(ETAS).index(eta)
        job_names = (arm["ailab_job"],)
        log_prefixes = (arm["ailab_log"],)
        host = "ailab"

    step = 0
    best_roc = None
    status = "pending"
    tag = arm["tag"]
    model = arm["model"]

    # Prefer seed-tagged parquet
    pq_hits = sorted(
        SWEEP.glob(f"{model}_*_{width}_{et}_*seed{seed}*{tag}.parquet")
    )
    if not pq_hits and seed == 0:
        # legacy seed0 dirs always contain seed0 in name for these tags
        pq_hits = sorted(SWEEP.glob(f"{model}_*_{width}_{et}_*{tag}.parquet"))
        pq_hits = [p for p in pq_hits if f"seed{seed}" in p.name]

    if pq_hits:
        try:
            import pyarrow.parquet as pq

            df = pq.read_table(
                pq_hits[0], columns=["global_step", "val_roc_auc"]
            ).to_pandas()
            if len(df):
                step = max(step, int(df["global_step"].max()))
                sub = df.dropna(subset=["val_roc_auc"])
                if len(sub):
                    best_roc = float(sub["val_roc_auc"].max())
        except Exception:
            pass

    running = False
    elapsed = ""
    node = ""
    job_id = None
    ent = _match_queue(sq, job_names, task)
    if ent:
        running = ent["state"] in ("RUNNING", "COMPLETING", "CONFIGURING")
        pending = ent["state"] in ("PENDING",)
        elapsed = ent.get("elapsed") or ""
        node = ent.get("node") or ""
        job_id = ent.get("full_id")
        if running:
            status = "running"
        elif pending:
            status = "pending"

    if status == "running" or (status == "pending" and not job_id):
        base = None
        if job_id and "_" in job_id:
            base = job_id.rsplit("_", 1)[0]
        step, best_roc, fail = _scrape_logs(log_prefixes, base, task, step, best_roc)
        if fail:
            status = fail

    # Seed1 edge tasks 6-11 may not be submitted yet (gputest MaxJobs=3)
    if (
        seed == 1
        and kind == "edge"
        and width in (256, 384)
        and task >= 6
        and job_id is None
        and step == 0
    ):
        status = "queued"  # waiting on watchdog / MaxJobs

    if step >= MAX_STEPS:
        status = "done"
    elif step > 0 and status in ("pending", "queued"):
        status = (
            "running"
            if running
            else ("done" if step >= int(0.95 * MAX_STEPS) else "stopped")
        )

    return dict(
        kind=kind,
        seed=seed,
        width=width,
        eta0=eta,
        task=task,
        status=status,
        step=step,
        max_steps=MAX_STEPS,
        best_roc=best_roc,
        host=host,
        job_id=job_id,
        node=node,
        elapsed=elapsed,
        tag=tag,
    )


def scrape() -> dict:
    sq = _squeue()
    cells = []
    for kind in ("m11", "edge"):
        for seed in SEEDS:
            for w in WIDTHS:
                for e in ETAS:
                    cells.append(_scrape_cell(kind, seed, w, e, sq))

    def _counts(kind: str, seed: int | None = None) -> dict:
        sub = [
            c
            for c in cells
            if c["kind"] == kind and (seed is None or c["seed"] == seed)
        ]
        return dict(
            n_done=sum(1 for c in sub if c["status"] == "done"),
            n_run=sum(1 for c in sub if c["status"] == "running"),
            n_fail=sum(1 for c in sub if c["status"] == "failed"),
            n_queued=sum(1 for c in sub if c["status"] in ("pending", "queued")),
            n_tot=len(sub),
        )

    m11 = _counts("m11")
    edge = _counts("edge")
    m11_s0, m11_s1 = _counts("m11", 0), _counts("m11", 1)
    edge_s0, edge_s1 = _counts("edge", 0), _counts("edge", 1)
    active = next((c for c in cells if c["status"] == "running"), None)
    return {
        "cells": cells,
        "m11": m11,
        "edge": edge,
        "m11_s0": m11_s0,
        "m11_s1": m11_s1,
        "edge_s0": edge_s0,
        "edge_s1": edge_s1,
        "n_done": m11["n_done"] + edge["n_done"],
        "n_run": m11["n_run"] + edge["n_run"],
        "n_fail": m11["n_fail"] + edge["n_fail"],
        "n_tot": m11["n_tot"] + edge["n_tot"],
        "active": active,
        "m11_tag": ARMS[("m11", 0)]["tag"],
        "edge_tag": ARMS[("edge", 0)]["tag"],
    }


def render(data: dict, updated: str) -> str:
    cells = data["cells"]
    cats = [f"{e:g}" for e in ETAS]

    def series_for(kind: str, seed: int, width: int) -> list:
        out = []
        for e in ETAS:
            c = next(
                (
                    x
                    for x in cells
                    if x["kind"] == kind
                    and x["seed"] == seed
                    and x["width"] == width
                    and abs(x["eta0"] - e) < 1e-12
                ),
                None,
            )
            v = None
            if c and c.get("best_roc") is not None:
                if c["status"] == "done" or c["step"] >= int(0.95 * MAX_STEPS):
                    v = float(c["best_roc"])
                elif c["status"] == "running" and c["step"] >= 500:
                    v = float(c["best_roc"])
            out.append(v)
        return out

    def ts_num_array(vals: list) -> str:
        parts = []
        for v in vals:
            if v is None:
                parts.append("Number.NaN")
            else:
                parts.append(repr(float(v)))
        return "[" + ", ".join(parts) + "]"

    table_str: list[list[str]] = []
    for c in cells:
        roc = "-"
        if c.get("best_roc") is not None:
            roc = f"{float(c['best_roc']):.5f}"
        table_str.append(
            [
                c["kind"],
                str(c["seed"]),
                str(c["width"]),
                f"{c['eta0']:g}",
                c["host"],
                c["status"],
                f"{c['step']}/{c['max_steps']}",
                roc,
                c.get("job_id") or "-",
            ]
        )

    active = data.get("active")
    active_s = "-"
    if active:
        active_s = (
            f"{active['kind']} seed={active['seed']} D={active['width']} "
            f"eta0={active['eta0']:g} step={active['step']} ({active['host']})"
        )

    # Charts: seed0 solid story; seed1 overlay when present
    series_blocks = {}
    for kind in ("m11", "edge"):
        for seed in SEEDS:
            for w in WIDTHS:
                series_blocks[f"{kind}_s{seed}_d{w}"] = ts_num_array(
                    series_for(kind, seed, w)
                )

    m11_s0, m11_s1 = data["m11_s0"], data["m11_s1"]
    edge_s0, edge_s1 = data["edge_s0"], data["edge_s1"]

    return f"""import {{
  Callout,
  Divider,
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

// BEGIN_AUTO_DATA
const UPDATED = {_js(updated)};
const RECIPE = {_js(
    "panel (c) · M11 + edges · B=96x4=384 · Adam · 2500 steps · (k*,R*)=(6,0.1) · seeds 0+1"
)};
const M11_TAG = {_js(data["m11_tag"])};
const EDGE_TAG = {_js(data["edge_tag"])};
const ACTIVE = {_js(active_s)};
const N_DONE: number = {int(data["n_done"])};
const N_TOT: number = {int(data["n_tot"])};
const N_RUN: number = {int(data["n_run"])};
const N_FAIL: number = {int(data["n_fail"])};
const M11_S0_DONE: number = {int(m11_s0["n_done"])};
const M11_S0_TOT: number = {int(m11_s0["n_tot"])};
const M11_S1_DONE: number = {int(m11_s1["n_done"])};
const M11_S1_TOT: number = {int(m11_s1["n_tot"])};
const M11_S1_RUN: number = {int(m11_s1["n_run"])};
const M11_S1_Q: number = {int(m11_s1["n_queued"])};
const EDGE_S0_DONE: number = {int(edge_s0["n_done"])};
const EDGE_S0_TOT: number = {int(edge_s0["n_tot"])};
const EDGE_S1_DONE: number = {int(edge_s1["n_done"])};
const EDGE_S1_TOT: number = {int(edge_s1["n_tot"])};
const EDGE_S1_RUN: number = {int(edge_s1["n_run"])};
const EDGE_S1_Q: number = {int(edge_s1["n_queued"])};
const CATS: string[] = {_js(cats)};
const M11_S0_D256: number[] = {series_blocks["m11_s0_d256"]};
const M11_S0_D384: number[] = {series_blocks["m11_s0_d384"]};
const M11_S0_D512: number[] = {series_blocks["m11_s0_d512"]};
const M11_S1_D256: number[] = {series_blocks["m11_s1_d256"]};
const M11_S1_D384: number[] = {series_blocks["m11_s1_d384"]};
const M11_S1_D512: number[] = {series_blocks["m11_s1_d512"]};
const EDGE_S0_D256: number[] = {series_blocks["edge_s0_d256"]};
const EDGE_S0_D384: number[] = {series_blocks["edge_s0_d384"]};
const EDGE_S0_D512: number[] = {series_blocks["edge_s0_d512"]};
const EDGE_S1_D256: number[] = {series_blocks["edge_s1_d256"]};
const EDGE_S1_D384: number[] = {series_blocks["edge_s1_d384"]};
const EDGE_S1_D512: number[] = {series_blocks["edge_s1_d512"]};
const TABLE_ROWS: string[][] = {_js(table_str)};
// END_AUTO_DATA

export default function PanelCEtaLive() {{
  const hasM11s0 = [...M11_S0_D256, ...M11_S0_D384, ...M11_S0_D512].some((v) =>
    Number.isFinite(v),
  );
  const hasM11s1 = [...M11_S1_D256, ...M11_S1_D384, ...M11_S1_D512].some((v) =>
    Number.isFinite(v),
  );
  const hasEdges0 = [...EDGE_S0_D256, ...EDGE_S0_D384, ...EDGE_S0_D512].some((v) =>
    Number.isFinite(v),
  );
  const hasEdges1 = [...EDGE_S1_D256, ...EDGE_S1_D384, ...EDGE_S1_D512].some((v) =>
    Number.isFinite(v),
  );
  return (
    <Stack gap={{20}}>
      <Stack gap={{6}}>
        <H1>Panel (c) eta0 x D @ 2500 — M11 + edges (seeds 0/1)</H1>
        <Text tone="secondary">{{RECIPE}}</Text>
        <Text tone="secondary">
          M11={{M11_TAG}} · edge={{EDGE_TAG}} · {{UPDATED}}
        </Text>
        <Callout tone="info" title="Seed1 wave + D384 η0.1 rerun">
          Seed0 is the original arm (edge D=384 η₀=0.1 wiped and requeued as
          jc-edge-rerun). Seed1 uses the same shard/recipe with SEED=1
          (jc-*-s1 jobs). Edge seed1 tasks 6–11 wait on gputest MaxJobsPU=3.
        </Callout>
        <Row gap={{8}}>
          <Pill tone={{N_DONE === N_TOT ? "success" : "info"}}>
            {{N_DONE}}/{{N_TOT}} done
          </Pill>
          <Pill tone="info">{{N_RUN}} running</Pill>
          {{N_FAIL > 0 ? <Pill tone="deleted">{{N_FAIL}} failed</Pill> : null}}
        </Row>
        <Text>Active: {{ACTIVE}}</Text>
      </Stack>

      <Grid columns={{4}} gap={{12}}>
        <Stat value={{`${{M11_S0_DONE}}/${{M11_S0_TOT}}`}} label="M11 seed0" />
        <Stat value={{`${{M11_S1_DONE}}/${{M11_S1_TOT}}`}} label="M11 seed1" />
        <Stat value={{`${{EDGE_S0_DONE}}/${{EDGE_S0_TOT}}`}} label="edge seed0" />
        <Stat value={{`${{EDGE_S1_DONE}}/${{EDGE_S1_TOT}}`}} label="edge seed1" />
      </Grid>
      <Grid columns={{4}} gap={{12}}>
        <Stat value={{String(M11_S1_RUN)}} label="M11 s1 run" tone="info" />
        <Stat value={{String(M11_S1_Q)}} label="M11 s1 queued" />
        <Stat value={{String(EDGE_S1_RUN)}} label="edge s1 run" tone="info" />
        <Stat value={{String(EDGE_S1_Q)}} label="edge s1 queued" />
      </Grid>

      <Divider />
      <H2>Best val ROC vs eta0 — M11 seed0</H2>
      {{hasM11s0 ? (
        <LineChart
          categories={{CATS}}
          series={{[
            {{ name: "D=256", data: M11_S0_D256 }},
            {{ name: "D=384", data: M11_S0_D384, tone: "info" }},
            {{ name: "D=512", data: M11_S0_D512, tone: "success" }},
          ]}}
          height={{240}}
          beginAtZero={{false}}
        />
      ) : (
        <Callout tone="info" title="Waiting">No M11 seed0 ROC yet.</Callout>
      )}}

      <H2>Best val ROC vs eta0 — edges seed0</H2>
      {{hasEdges0 ? (
        <LineChart
          categories={{CATS}}
          series={{[
            {{ name: "D=256", data: EDGE_S0_D256 }},
            {{ name: "D=384", data: EDGE_S0_D384, tone: "info" }},
            {{ name: "D=512", data: EDGE_S0_D512, tone: "success" }},
          ]}}
          height={{240}}
          beginAtZero={{false}}
        />
      ) : (
        <Callout tone="info" title="Waiting">No edge seed0 ROC yet.</Callout>
      )}}

      <H2>Best val ROC vs eta0 — M11 seed1</H2>
      {{hasM11s1 ? (
        <LineChart
          categories={{CATS}}
          series={{[
            {{ name: "D=256", data: M11_S1_D256 }},
            {{ name: "D=384", data: M11_S1_D384, tone: "info" }},
            {{ name: "D=512", data: M11_S1_D512, tone: "success" }},
          ]}}
          height={{240}}
          beginAtZero={{false}}
        />
      ) : (
        <Callout tone="info" title="Seed1 filling">
          M11 seed1 jobs submitted (jc-m11-gt-s1 / jc-m11-a512-s1). ROC after first val.
        </Callout>
      )}}

      <H2>Best val ROC vs eta0 — edges seed1</H2>
      {{hasEdges1 ? (
        <LineChart
          categories={{CATS}}
          series={{[
            {{ name: "D=256", data: EDGE_S1_D256 }},
            {{ name: "D=384", data: EDGE_S1_D384, tone: "info" }},
            {{ name: "D=512", data: EDGE_S1_D512, tone: "success" }},
          ]}}
          height={{240}}
          beginAtZero={{false}}
        />
      ) : (
        <Callout tone="info" title="Seed1 filling">
          Edge seed1: tasks 0–5 on jc-edge-gt-s1; 6–11 via MaxJobs watchdog; D512 on jc-edge-a512-s1.
        </Callout>
      )}}

      <Table
        headers={{["kind", "seed", "D", "eta0", "host", "status", "step", "best ROC", "job"]}}
        rows={{TABLE_ROWS}}
      />
    </Stack>
  );
}}
"""


def main() -> None:
    data = scrape()
    updated = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
    CANVAS.parent.mkdir(parents=True, exist_ok=True)
    try:
        CANVAS.write_text(render(data, updated))
    except OSError:
        tmp = CANVAS.with_suffix(".canvas.tsx.tmp")
        tmp.write_text(render(data, updated))
        tmp.replace(CANVAS)
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps({"updated": updated, **data}, indent=2) + "\n")
    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "kind",
                "seed",
                "width",
                "eta0",
                "host",
                "status",
                "step",
                "max_steps",
                "best_roc",
                "job_id",
            ],
        )
        w.writeheader()
        for c in data["cells"]:
            w.writerow(
                {
                    "kind": c["kind"],
                    "seed": c["seed"],
                    "width": c["width"],
                    "eta0": c["eta0"],
                    "host": c["host"],
                    "status": c["status"],
                    "step": c["step"],
                    "max_steps": c["max_steps"],
                    "best_roc": c.get("best_roc"),
                    "job_id": c.get("job_id"),
                }
            )
    print(
        f"{updated} panel-c canvas "
        f"M11 s0 {data['m11_s0']['n_done']}/{data['m11_s0']['n_tot']} "
        f"s1 {data['m11_s1']['n_done']}/{data['m11_s1']['n_tot']} "
        f"edge s0 {data['edge_s0']['n_done']}/{data['edge_s0']['n_tot']} "
        f"s1 {data['edge_s1']['n_done']}/{data['edge_s1']['n_tot']} "
        f"run={data['n_run']} → {CANVAS.name}"
    )


if __name__ == "__main__":
    main()
