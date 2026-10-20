#!/usr/bin/env python3
"""Live canvas: D=128 uniform-pool flat→cosine η₀ sweep on Lambda.

Rewrites:
  ~/.cursor/projects/home-jdezoort/canvases/hxfer-upool-d128-flatcos-live.canvas.tsx
"""
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
CANVAS = Path(
    "/home/jdezoort/.cursor/projects/home-jdezoort/canvases/"
    "hxfer-upool-d128-flatcos-live.canvas.tsx"
)
JSON_OUT = REPO / "paper" / "data" / "hxfer_upool_d128_flatcos_live.json"
CSV_OUT = REPO / "paper" / "data" / "hxfer_upool_d128_flatcos_live_status.csv"

LAMBDA_HOST = "ubuntu@193.122.155.21"
LAMBDA_CM = Path.home() / ".cache" / "cm-lambda-193"
ETAS = (0.01, 0.05, 0.1, 0.5, 1.0)
WIDTH = 128
FLAT_STEPS = 10000
COSINE_STEPS = 10000
MAX_STEPS = FLAT_STEPS + COSINE_STEPS
# Match run_hxfer_upool_d128_flat_cosine_lambda8.sh RUN_TAG_BASE (sched suffix auto-appended).
TAG = "hxfer-d128-fc-k6-R0p1"
RUNNER = "run_hxfer_upool_d128_flat_cosine_lambda8"
LOGDIR = "/home/ubuntu/logs/hxfer_upool_d128_flatcos"
RUNROOT = "/home/ubuntu/runs/hxfer_upool_d128_flatcos/jetclass_output/adam/sweep_lr"
ET = ZoneInfo("America/New_York")


def _js(x) -> str:
    return json.dumps(x)


def _lambda_ssh(script: str, timeout: int = 120) -> str:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]
    if LAMBDA_CM.exists():
        cmd += ["-S", str(LAMBDA_CM)]
    cmd += [LAMBDA_HOST, script]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(
            (proc.stderr or proc.stdout or f"ssh rc={proc.returncode}")[:500]
        )
    return proc.stdout


def _scrape() -> dict:
    remote = r"""
python3 - <<'PY'
import json, re, glob, subprocess
from pathlib import Path

ETAS=[0.01,0.05,0.1,0.5,1.0]
WIDTH=128
FLAT=10000
COS=10000
MAX_STEPS=FLAT+COS
TAG="hxfer-d128-fc-k6-R0p1"
RUNNER="run_hxfer_upool_d128_flat_cosine_lambda8"
LOGDIR=Path.home()/"logs"/"hxfer_upool_d128_flatcos"
RUNROOT=Path.home()/"runs"/"hxfer_upool_d128_flatcos"/"jetclass_output"/"adam"/"sweep_lr"

def eta_tag(e):
    return f"{e:g}".replace(".","p")

runner=False
try:
    out=subprocess.check_output(["pgrep","-af",RUNNER], text=True)
    runner=RUNNER in out
except Exception:
    pass

cells=[]; active=None
for ei,eta in enumerate(ETAS):
    et=eta_tag(eta)
    log=LOGDIR/f"D{WIDTH}_eta{et}_R0p1.out"
    text=log.read_text(errors="ignore") if log.exists() else ""
    step=0; best=None; last=None; pts=[]; has_ckpt=False; has_pq=False
    pq_paths=sorted(glob.glob(str(RUNROOT/f"*_{WIDTH}_{et}_*{TAG}*.parquet")))
    has_pq=bool(pq_paths)
    ckpt_dirs=sorted(glob.glob(str(RUNROOT/f"*_{WIDTH}_{et}_*{TAG}*")))
    for d in ckpt_dirs:
        p=Path(d)
        if p.is_dir() and ((p/"best.ckpt").exists() or (p/"last.ckpt").exists()):
            has_ckpt=True
            break
    for m in re.finditer(r"\[val\].*?step=(\d+).*?val_roc_auc=([0-9.]+)", text):
        s,roc=int(m.group(1)), float(m.group(2))
        pts.append({"S":s,"roc":roc}); step=max(step,s)
        best=roc if best is None else max(best,roc)
        last=roc
    for m in re.finditer(r"global_step=(\d+)", text):
        step=max(step,int(m.group(1)))
    if has_pq:
        try:
            import pyarrow.parquet as pq
            df=pq.read_table(pq_paths[0], columns=["global_step","val_roc_auc"]).to_pandas()
            if len(df):
                step=max(step, int(df["global_step"].max()))
                sub=df.dropna(subset=["val_roc_auc"])
                if len(sub):
                    best=float(sub["val_roc_auc"].max())
                    last=float(sub.sort_values("global_step").iloc[-1]["val_roc_auc"])
                    pts=[{"S":int(r.global_step),"roc":float(r.val_roc_auc)} for r in sub.itertuples()]
        except Exception:
            pass
    phase="pending"
    if step<=0:
        phase="init" if text and any(k in text for k in ("GPU available","attn-gamma","Initializing distributed","[train]")) else "pending"
    elif step < FLAT:
        phase="flat"
    elif step < MAX_STEPS:
        phase="cosine"
    else:
        phase="done"
    status="pending"
    if step>=MAX_STEPS:
        status="done"
    elif phase in ("init","flat","cosine") or step>0 or has_pq:
        # Prefer running over failed while the campaign runner is still alive
        # (setup Tracebacks can appear before a FRESH relaunch).
        status="running" if runner or step>0 or has_pq else "failed"
    elif "Traceback" in text and step<=0:
        status="failed"
    elif log.exists() and text.strip():
        status="running" if runner else "stopped"
    cell={"task":ei,"width":WIDTH,"eta0":eta,"status":status,"phase":phase,
          "step":step,"max_steps":MAX_STEPS,"flat_steps":FLAT,"cosine_steps":COS,
          "best_roc":best,"last_roc":last,"has_ckpt":has_ckpt,"has_pq":has_pq,
          "curve":pts[-60:]}
    cells.append(cell)
    if status=="running" and active is None:
        active=cell

print(json.dumps({
  "runner":runner,
  "cells":cells,
  "active":active,
  "n_done":sum(1 for c in cells if c["status"]=="done"),
  "n_run":sum(1 for c in cells if c["status"]=="running"),
  "n_fail":sum(1 for c in cells if c["status"]=="failed"),
  "n_tot":len(ETAS),
  "flat_steps":FLAT,
  "cosine_steps":COS,
  "max_steps":MAX_STEPS,
}))
PY
"""
    try:
        out = _lambda_ssh(remote, timeout=150)
        return json.loads(out.strip().splitlines()[-1])
    except Exception as e:
        return {"cells": [], "error": str(e), "n_tot": len(ETAS)}


def ensure_runner(scrape: dict) -> str:
    if scrape.get("runner"):
        return "ok"
    # Only restart if nothing is done yet or some cells remain.
    n_done = int(scrape.get("n_done") or 0)
    n_tot = int(scrape.get("n_tot") or len(ETAS))
    if n_done >= n_tot:
        return "complete"
    cmd = (
        f"mkdir -p {LOGDIR} && "
        f"nohup bash ~/cpen/scans/jets/{RUNNER}.sh "
        f">{LOGDIR}/master.log 2>&1 & echo started:$!"
    )
    try:
        out = _lambda_ssh(cmd, timeout=30)
        return "restarted:" + out.strip().replace("\n", " ")[:120]
    except Exception as e:
        return f"restart_failed:{e}"


def render(scrape: dict, updated: str, ensure: str) -> str:
    cells = scrape.get("cells") or []
    active = scrape.get("active")
    cats = [f"{e:g}" for e in ETAS]
    best_data = []
    last_data = []
    for eta in ETAS:
        cell = next((c for c in cells if abs(c["eta0"] - eta) < 1e-12), None)
        b = cell.get("best_roc") if cell else None
        l = cell.get("last_roc") if cell else None
        # Show mid-run values once we have a few vals
        if cell and cell.get("step", 0) >= 500:
            best_data.append(b)
            last_data.append(l)
        elif cell and cell.get("status") == "done":
            best_data.append(b)
            last_data.append(l)
        else:
            best_data.append(None)
            last_data.append(None)

    series = [
        {"name": "best val ROC", "data": best_data, "tone": "accent"},
        {"name": "last val ROC", "data": last_data, "tone": "info"},
    ]

    # Active cell learning curve
    curve_cats: list[str] = []
    curve_series: list[dict] = []
    if active and active.get("curve"):
        curve_cats = [str(p["S"]) for p in active["curve"]]
        curve_series = [
            {
                "name": f"D={active['width']} η₀={active['eta0']:g}",
                "data": [p["roc"] for p in active["curve"]],
                "tone": "accent",
            }
        ]

    active_s = "—"
    if active:
        active_s = (
            f"D={active['width']} η₀={active['eta0']:g} "
            f"{active.get('phase','?')} step={active['step']}/{active['max_steps']}"
        )

    table = [
        [
            str(c["task"]),
            f"{c['eta0']:g}",
            c["status"],
            c.get("phase", ""),
            f"{c['step']}/{c['max_steps']}",
            None if c.get("best_roc") is None else round(float(c["best_roc"]), 5),
            None if c.get("last_roc") is None else round(float(c["last_roc"]), 5),
            "yes" if c.get("has_ckpt") else "no",
        ]
        for c in cells
    ]
    tones = [
        "success"
        if c["status"] == "done"
        else "info"
        if c["status"] == "running"
        else "danger"
        if c["status"] == "failed"
        else None
        for c in cells
    ]

    n_done = scrape.get("n_done", 0)
    n_run = scrape.get("n_run", 0)
    n_tot = scrape.get("n_tot", len(ETAS))
    runner = bool(scrape.get("runner"))

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
const HOST = {_js(LAMBDA_HOST)};
const RECIPE = {_js(
    "D=128 uniform pool · k★=6 R★=0.1 · B=64×8=512 · n_train=1M · "
    "10k flat @ η₀/√D → 10k cosine → 0.01×peak"
)};
const ACTIVE = {_js(active_s)};
const ENSURE = {_js(ensure)};
const N_DONE = {n_done};
const N_RUN = {n_run};
const N_TOT = {n_tot};
const RUNNER = {_js(runner)};
const FLAT = {FLAT_STEPS};
const COS = {COSINE_STEPS};
const MAX_STEPS = {MAX_STEPS};
const CATS: string[] = {_js(cats)};
const SERIES = {_js(series)};
const CURVE_CATS: string[] = {_js(curve_cats)};
const CURVE_SERIES = {_js(curve_series)};
const TABLE = {_js(table)};
const TONE = {_js(tones)};
const ERR = {_js(scrape.get("error") or "")};
// END_AUTO_DATA

function fmt(x: number | null | undefined, d = 4): string {{
  return x == null || Number.isNaN(Number(x)) ? "—" : Number(x).toFixed(d);
}}

export default function HxferUpoolD128FlatCosLive() {{
  const series = SERIES.map((s: any) => ({{
    name: s.name,
    data: (s.data as (number | null)[]).map((v) => (v == null ? Number.NaN : Number(v))),
    tone: s.tone,
  }}));
  const hasEta = SERIES.some((s: any) =>
    (s.data as (number | null)[]).some((v) => v != null)
  );
  const curveSeries = CURVE_SERIES.map((s: any) => ({{
    name: s.name,
    data: (s.data as number[]).map((v) => Number(v)),
    tone: s.tone,
  }}));
  const hasCurve = CURVE_CATS.length > 0;
  return (
    <Stack gap={{20}}>
      <Stack gap={{6}}>
        <H1>D=128 flat→cosine η₀ sweep</H1>
        <Text tone="secondary">{{RECIPE}}</Text>
        <Text tone="secondary">{{HOST}} · {{UPDATED}}</Text>
        <Row gap={{8}}>
          <Pill tone={{N_DONE === N_TOT ? "success" : "info"}}>
            {{N_DONE}}/{{N_TOT}} done
          </Pill>
          <Pill tone="info">{{N_RUN}} running</Pill>
          <Pill tone={{RUNNER ? "success" : "danger"}}>
            {{RUNNER ? "runner up" : "runner down"}}
          </Pill>
          <Pill tone="neutral">flat {{FLAT}} → cos {{COS}}</Pill>
        </Row>
        <Text>Active: {{ACTIVE}}</Text>
        {{ERR ? <Callout tone="danger" title="Scrape error">{{ERR}}</Callout> : null}}
        {{ENSURE.startsWith("restart") ? (
          <Callout tone="warning" title="Runner ensure">{{ENSURE}}</Callout>
        ) : null}}
      </Stack>

      <Grid columns={{4}} gap={{12}}>
        <Stat value={{`${{N_DONE}}/${{N_TOT}}`}} label="cells done" />
        <Stat value={{String(N_RUN)}} label="running" tone="info" />
        <Stat value={{String(MAX_STEPS)}} label="steps / cell" />
        <Stat value={{"128"}} label="width D" />
      </Grid>

      <Divider />
      <H2>Best / last val ROC vs η₀</H2>
      {{hasEta ? (
        <LineChart
          categories={{CATS}}
          series={{series}}
          height={{260}}
          fill
        />
      ) : (
        <Callout tone="info" title="Waiting on first vals">
          Cells report after ~500 steps / first validation.
        </Callout>
      )}}

      {{hasCurve ? (
        <>
          <Divider />
          <H2>Active cell — val ROC vs step</H2>
          <Text tone="secondary">
            Vertical transition at step {{FLAT}} (flat → cosine).
          </Text>
          <LineChart
            categories={{CURVE_CATS}}
            series={{curveSeries}}
            height={{240}}
          />
        </>
      ) : null}}

      <Divider />
      <H2>Cell status</H2>
      <Table
        headers={{["task", "η₀", "status", "phase", "step", "best ROC", "last ROC", "ckpt"]}}
        rows={{TABLE.map((r: any) => [
          r[0], r[1], r[2], r[3], r[4], fmt(r[5], 5), fmt(r[6], 5), r[7],
        ])}}
        rowTone={{TONE}}
      />
    </Stack>
  );
}}
"""


def main() -> None:
    scrape = _scrape()
    ensure = ensure_runner(scrape)
    if ensure.startswith("restarted"):
        scrape = _scrape()
    updated = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
    CANVAS.parent.mkdir(parents=True, exist_ok=True)
    CANVAS.write_text(render(scrape, updated, ensure))
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(
        json.dumps({"updated": updated, "ensure": ensure, **scrape}, indent=2) + "\n"
    )
    rows = []
    for c in scrape.get("cells") or []:
        rows.append(
            {
                "task": c.get("task"),
                "width": c.get("width"),
                "eta0": c.get("eta0"),
                "status": c.get("status"),
                "phase": c.get("phase"),
                "step": c.get("step"),
                "max_steps": c.get("max_steps"),
                "best_roc": c.get("best_roc"),
                "last_roc": c.get("last_roc"),
                "has_ckpt": c.get("has_ckpt"),
            }
        )
    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "width",
                "eta0",
                "status",
                "phase",
                "step",
                "max_steps",
                "best_roc",
                "last_roc",
                "has_ckpt",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(
        f"{updated} canvas done={scrape.get('n_done',0)}/{scrape.get('n_tot',5)} "
        f"run={scrape.get('n_run',0)} runner={bool(scrape.get('runner'))} "
        f"ensure={ensure} → {CANVAS.name}"
    )


if __name__ == "__main__":
    main()
