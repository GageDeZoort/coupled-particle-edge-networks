#!/usr/bin/env python3
"""Refresh live canvas for Lambda graph-construction panels (a)+(b).

  python scans/jets/update_graph_k_lambda_live_canvas.py

Panel (a): k @ R★=0.1 (complete)
Panel (b): R★ @ k★=6 (live)

Rewrites:
  ~/.cursor/projects/home-jdezoort/canvases/graph-k-R0p1-lambda-live.canvas.tsx
Also writes:
  paper/data/graph_k_R0p1_lambda_live.json
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
CANVAS = Path(
    "/home/jdezoort/.cursor/projects/home-jdezoort/canvases/graph-k-R0p1-lambda-live.canvas.tsx"
)
JSON_OUT = REPO / "paper" / "data" / "graph_k_R0p1_lambda_live.json"
ORG = REPO / "paper" / "data" / "graph_construction_lambda_v1lam"

LAMBDA_HOST = "ubuntu@193.122.155.21"
LAMBDA_CM = Path.home() / ".cache" / "cm-lambda-193"
KS = (1, 2, 3, 4, 5, 6, 7, 8, 10)
RADII = (0.05, 0.075, 0.1, 0.125, 0.15)
N_SHARD = 3
N_INIT = 2
N_PER = N_SHARD * N_INIT
MAX_STEPS = 1000
K_STAR = 6
ET = ZoneInfo("America/New_York")
POOL = "attentional"


def _js(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))


def _r_tag(r: float) -> str:
    return f"{r:g}".replace(".", "p")


def _lambda_ssh(cmd: str, *, timeout: int = 120) -> str:
    base = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12"]
    if LAMBDA_CM.is_socket() or LAMBDA_CM.exists():
        base += ["-S", str(LAMBDA_CM)]
    base.append(LAMBDA_HOST)
    try:
        proc = subprocess.run(
            base + [cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except Exception:
        return ""
    return proc.stdout if proc.returncode == 0 else (proc.stdout or "")


def _scrape_remote() -> dict:
    script = r"""
python3 - <<'PY'
import json, re, os, glob
from pathlib import Path

KS=[2,3,4,5,6,7,8,10]  # drop k=1 (degenerate 1-NN)
RADII=[0.05,0.075,0.1,0.125,0.15]
N_SHARD,N_INIT=3,2
MAX_STEPS=1000
K_TAG="graph-k-b64-g8-s1000-R0p1"
R_TAG_BASE="graph-R-b64-g8-s1000-k6"

def r_tag(r):
    return f"{r:g}".replace(".","p")

def parse_cell(log, runroot, knn_pat, tag_pat, shard, seed):
    text=log.read_text(errors="ignore") if log.exists() else ""
    pts=[]; step=0; best=None
    for m in re.finditer(r"\[val\].*?step=(\d+).*?val_roc_auc=([0-9.]+)", text):
        s,roc=int(m.group(1)), float(m.group(2))
        pts.append({"S":s,"roc":roc}); step=max(step,s); best=roc if best is None else max(best,roc)
    for m in re.finditer(r"global_step=(\d+)", text):
        step=max(step,int(m.group(1)))
    try:
        import pyarrow.parquet as pq
        pats=glob.glob(str(runroot/f"*{knn_pat}*{tag_pat}-shard{shard}-seed{seed}-v1lam.parquet"))
        if pats:
            df=pq.read_table(pats[0], columns=["global_step","val_roc_auc"]).to_pandas()
            if len(df):
                step=max(step,int(df["global_step"].max()))
                sub=df.dropna(subset=["val_roc_auc"])
                if len(sub):
                    by={int(r.global_step):float(r.val_roc_auc) for _,r in sub.iterrows()}
                    pts=[{"S":s,"roc":by[s]} for s in sorted(by)]
                    best=max(by.values())
    except Exception:
        pass
    return pts, step, best, text

# --- k panel ---
k_log=Path("/home/ubuntu/logs/graph_k")
k_root=Path("/home/ubuntu/runs/graph_k/jetclass_output/adam/sweep_lr")
k_master=(k_log/"master.log").read_text(errors="ignore") if (k_log/"master.log").exists() else ""
k_active=None
for line in k_master.splitlines():
    m=re.search(r"RUN task=(\d+) k=(\d+) shard=(\d+) seed=(\d+)", line)
    if m: k_active={"k":int(m.group(2)),"shard":int(m.group(3)),"seed":int(m.group(4))}
    m=re.search(r"OK task=(\d+) k=(\d+) shard=(\d+) seed=(\d+)", line)
    if m and k_active and k_active["k"]==int(m.group(2)) and k_active["shard"]==int(m.group(3)) and k_active["seed"]==int(m.group(4)):
        k_active=None

k_cells=[]
for k in KS:
  for shard in range(N_SHARD):
    for seed in range(N_INIT):
      log=k_log/f"k{k}_shard{shard}_seed{seed}_R0p1.out"
      pts,step,best,_=parse_cell(log,k_root,f"knn{k}-",K_TAG,shard,seed)
      is_act=bool(k_active and k_active["k"]==k and k_active["shard"]==shard and k_active["seed"]==seed)
      if is_act: status="running"
      elif step>=MAX_STEPS-5: status="done"
      elif step>0 or pts: status="running"
      else: status="pending"
      k_cells.append({"k":k,"shard":shard,"seed":seed,"status":status,"step":step,"max_steps":MAX_STEPS,"best_roc":best,"pts":pts})

# --- R panel ---
r_log=Path("/home/ubuntu/logs/graph_R")
r_root=Path("/home/ubuntu/runs/graph_R/jetclass_output/adam/sweep_lr")
r_master=(r_log/"master.log").read_text(errors="ignore") if (r_log/"master.log").exists() else ""
r_active=None
for line in r_master.splitlines():
    m=re.search(r"RUN task=(\d+) R=([0-9.]+) k=(\d+) shard=(\d+) seed=(\d+)", line)
    if m: r_active={"R":float(m.group(2)),"k":int(m.group(3)),"shard":int(m.group(4)),"seed":int(m.group(5))}
    m=re.search(r"OK task=(\d+) R=([0-9.]+) shard=(\d+) seed=(\d+)", line)
    if m and r_active and abs(r_active["R"]-float(m.group(2)))<1e-9 and r_active["shard"]==int(m.group(3)) and r_active["seed"]==int(m.group(4)):
        r_active=None
    m=re.search(r"FAIL task=(\d+) R=([0-9.]+) shard=(\d+) seed=(\d+)", line)
    if m and r_active and abs(r_active["R"]-float(m.group(2)))<1e-9 and r_active["shard"]==int(m.group(3)) and r_active["seed"]==int(m.group(4)):
        # keep failed marked below via step
        r_active=None

r_cells=[]
for R in RADII:
  rt=r_tag(R)
  for shard in range(N_SHARD):
    for seed in range(N_INIT):
      log=r_log/f"R{rt}_k6_shard{shard}_seed{seed}.out"
      tag=f"{R_TAG_BASE}-R{rt}"
      pts,step,best,text=parse_cell(log,r_root,"knn6-",tag,shard,seed)
      is_act=bool(r_active and abs(r_active["R"]-R)<1e-9 and r_active["shard"]==shard and r_active["seed"]==seed)
      failed=("FAIL" in r_master and f"R={R:g}" in r_master and f"shard={shard} seed={seed}" in r_master and step==0 and not is_act)
      # simpler fail: last event for this cell
      if is_act: status="running"
      elif step>=MAX_STEPS-5: status="done"
      elif "ValueError" in text and step==0: status="failed"
      elif step>0 or pts: status="running"
      else: status="pending"
      r_cells.append({"R":R,"k":6,"shard":shard,"seed":seed,"status":status,"step":step,"max_steps":MAX_STEPS,"best_roc":best,"pts":pts})

runner=bool(os.popen("pgrep -af 'run_graph_R_lambda8|test_cpen_jetclass.*graph-R|run_graph_k_lambda8' 2>/dev/null | head -3").read().strip())
print(json.dumps({"k_cells":k_cells,"r_cells":r_cells,"k_active":k_active,"r_active":r_active,"runner":runner}))
PY
"""
    out = _lambda_ssh(script, timeout=150)
    if not out.strip():
        return {"k_cells": [], "r_cells": [], "runner": False, "error": "ssh_empty"}
    try:
        line = [ln for ln in out.splitlines() if ln.strip().startswith("{")][-1]
        return json.loads(line)
    except Exception as e:
        return {"k_cells": [], "r_cells": [], "runner": False, "error": str(e)}


def _summarize_axis(cells: list[dict], key: str, grid: tuple) -> dict:
    by = {g: [] for g in grid}
    for c in cells:
        by.setdefault(c[key], []).append(c)
    means = []
    n_done = n_run = n_pd = n_fail = 0
    for g in grid:
        group = by.get(g, [])
        done = [
            c
            for c in group
            if c["status"] == "done"
            and c.get("best_roc") is not None
            and int(c.get("step") or 0) >= MAX_STEPS - 5
        ]
        run = [c for c in group if c["status"] == "running"]
        fail = [c for c in group if c["status"] == "failed"]
        pd_ = [c for c in group if c["status"] == "pending"]
        n_done += len(done)
        n_run += len(run)
        n_fail += len(fail)
        n_pd += len(pd_)
        rocs = [c["best_roc"] for c in done]
        means.append(
            {
                key: g,
                "n_done": len(done),
                "n_total": N_PER,
                "mean": statistics.mean(rocs) if rocs else None,
                "std": statistics.stdev(rocs) if len(rocs) > 1 else (0.0 if rocs else None),
            }
        )
    return {
        "n_done": n_done,
        "n_run": n_run,
        "n_pd": n_pd,
        "n_fail": n_fail,
        "n_total": len(grid) * N_PER,
        "means": means,
        "cells": cells,
    }


def ensure_runner(scrape: dict) -> str:
    if scrape.get("runner"):
        return "alive"
    r_cells = scrape.get("r_cells") or []
    incomplete = [
        c
        for c in r_cells
        if not (c["status"] == "done" and int(c.get("step") or 0) >= MAX_STEPS - 5)
    ]
    if not incomplete:
        return "r_all_done"
    # map to task index in RADII×6
    tasks = []
    for c in incomplete:
        r_idx = RADII.index(c["R"])
        tid = r_idx * N_PER + c["shard"] * N_INIT + c["seed"]
        tasks.append(tid)
    start = min(tasks)
    task_str = " ".join(str(t) for t in range(start, len(RADII) * N_PER))
    cmd = (
        "mkdir -p ~/logs/graph_R; "
        f"nohup env TASKS='{task_str}' FRESH=0 bash ~/cpen/scans/jets/run_graph_R_lambda8.sh "
        ">>~/logs/graph_R/master.log 2>&1 & echo RESTART_R_pid=$!"
    )
    out = _lambda_ssh(cmd, timeout=30)
    return "restarted_R:" + out.strip().replace("\n", " | ")[:200]


def render_canvas(k_sum: dict, r_sum: dict, scrape: dict, updated: str) -> str:
    k_means = k_sum["means"]
    r_means = r_sum["means"]
    k_cats = [str(m["k"]) for m in k_means]
    r_cats = [f"{m['R']:g}" for m in r_means]
    k_data = [m["mean"] for m in k_means]
    r_data = [m["mean"] for m in r_means]
    has_k = any(v is not None for v in k_data)
    has_r = any(v is not None for v in r_data)
    r_active = scrape.get("r_active")
    active = "—"
    if r_active:
        active = f"R={r_active['R']:g} shard={r_active['shard']} seed={r_active['seed']}"
    elif scrape.get("runner"):
        active = "runner alive"
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
const POOL = {_js(POOL)};
const HOST = {_js(LAMBDA_HOST)};
const RECIPE = {_js("D=256 L=4 H=8 η₀=0.25 B=64×8=512 · 1000 steps · attentional · part-interaction · 3×2")};
const RUNNER = {_js(bool(scrape.get("runner")))};
const ACTIVE = {_js(active)};
const K_DONE = {k_sum["n_done"]};
const K_TOTAL = {k_sum["n_total"]};
const R_DONE = {r_sum["n_done"]};
const R_RUN = {r_sum["n_run"]};
const R_PD = {r_sum["n_pd"]};
const R_FAIL = {r_sum["n_fail"]};
const R_TOTAL = {r_sum["n_total"]};
const K_CATS: string[] = {_js(k_cats)};
const K_MEAN: (number | null)[] = {_js(k_data)};
const K_ROWS = {_js(k_means)};
const R_CATS: string[] = {_js(r_cats)};
const R_MEAN: (number | null)[] = {_js(r_data)};
const R_ROWS = {_js(r_means)};
const R_CELLS = {_js(r_sum["cells"])};
const HAS_K = {_js(has_k)};
const HAS_R = {_js(has_r)};
// END_AUTO_DATA

function fmt(x: number | null | undefined, d = 4): string {{
  return x == null || Number.isNaN(Number(x)) ? "—" : Number(x).toFixed(d);
}}

export default function GraphConstructionLambdaLive() {{
  const kSeries = [{{ name: "mean best val ROC", data: K_MEAN.map((v) => (v == null ? Number.NaN : Number(v))), tone: "accent" as const }}];
  const rSeries = [{{ name: "mean best val ROC", data: R_MEAN.map((v) => (v == null ? Number.NaN : Number(v))), tone: "accent" as const }}];
  const rTable = R_CELLS.map((c: any) => [
    String(c.R), String(c.shard), String(c.seed), c.status,
    `${{c.step}}/${{c.max_steps}}`, fmt(c.best_roc),
  ]);
  const rTone = R_CELLS.map((c: any) =>
    c.status === "done" ? "success" : c.status === "running" ? "info" : c.status === "failed" ? "danger" : undefined
  );
  return (
    <Stack gap={{20}}>
      <Stack gap={{6}}>
        <H1>Graph construction — Lambda live</H1>
        <Text tone="secondary">{{RECIPE}}</Text>
        <Text tone="secondary">Pool: {{POOL}} · {{HOST}} · {{UPDATED}}</Text>
        <Row gap={{8}}>
          <Pill tone="success">a: {{K_DONE}}/{{K_TOTAL}} k done</Pill>
          <Pill tone={{R_DONE === R_TOTAL ? "success" : "info"}}>b: {{R_DONE}}/{{R_TOTAL}} R done</Pill>
          <Pill tone="info">{{R_RUN}} R running</Pill>
          {{R_FAIL > 0 ? <Pill tone="danger">{{R_FAIL}} R failed</Pill> : null}}
          <Pill tone={{RUNNER ? "success" : "danger"}}>{{RUNNER ? "runner up" : "runner down"}}</Pill>
        </Row>
        <Text>Active: {{ACTIVE}}</Text>
      </Stack>

      <Grid columns={{4}} gap={{12}}>
        <Stat value={{`${{K_DONE}}/${{K_TOTAL}}`}} label="panel (a) k cells" tone="success" />
        <Stat value={{`${{R_DONE}}/${{R_TOTAL}}`}} label="panel (b) R cells" />
        <Stat value={{String(R_RUN)}} label="R running" tone="info" />
        <Stat value={{POOL}} label="pooling" />
      </Grid>

      <Divider />
      <H2>Panel (a) — mean ROC vs k @ R★=0.1</H2>
      {{HAS_K ? (
        <LineChart categories={{K_CATS}} series={{kSeries}} height={{240}} fill />
      ) : (
        <Callout tone="info" title="No k cells yet">Waiting for panel (a).</Callout>
      )}}
      <Table
        headers={{["k", "done/6", "mean ROC", "std"]}}
        rows={{K_ROWS.map((m: any) => [String(m.k), `${{m.n_done}}/${{m.n_total}}`, fmt(m.mean), fmt(m.std)])}}
      />

      <Divider />
      <H2>Panel (b) — mean ROC vs R★ @ k★=6</H2>
      <Text tone="secondary">R★ ∈ {{0.05, 0.075, 0.1, 0.125, 0.15}} · filling live</Text>
      {{HAS_R ? (
        <LineChart categories={{R_CATS}} series={{rSeries}} height={{240}} fill />
      ) : (
        <Callout tone="info" title="R★ cells filling">First finished R★ cells will appear here.</Callout>
      )}}
      <Table
        headers={{["R★", "done/6", "mean ROC", "std"]}}
        rows={{R_ROWS.map((m: any) => [String(m.R), `${{m.n_done}}/${{m.n_total}}`, fmt(m.mean), fmt(m.std)])}}
      />

      <Divider />
      <H2>Panel (b) all cells</H2>
      <Table headers={{["R★", "shard", "seed", "status", "step", "best ROC"]}} rows={{rTable}} rowTone={{rTone}} />
    </Stack>
  );
}}
"""


def main() -> None:
    scrape = _scrape_remote()
    ensure_msg = ensure_runner(scrape)
    if ensure_msg.startswith("restarted"):
        scrape = _scrape_remote()
    k_cells = scrape.get("k_cells") or []
    r_cells = scrape.get("r_cells") or []
    k_sum = _summarize_axis(k_cells, "k", KS)
    r_sum = _summarize_axis(r_cells, "R", RADII)
    updated = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
    CANVAS.write_text(render_canvas(k_sum, r_sum, scrape, updated))
    payload = {
        "updated": updated,
        "ensure": ensure_msg,
        "pool": POOL,
        "panel_a": k_sum,
        "panel_b": r_sum,
        "runner": bool(scrape.get("runner")),
        "r_active": scrape.get("r_active"),
        "error": scrape.get("error"),
    }
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, indent=2) + "\n")
    # also snapshot into org tree
    snap = ORG / "k_R0p1" / "summaries" / "graph_construction_live.json"
    snap.parent.mkdir(parents=True, exist_ok=True)
    snap.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"{updated} canvas a={k_sum['n_done']}/{k_sum['n_total']} "
        f"b={r_sum['n_done']}/{r_sum['n_total']}(run={r_sum['n_run']}) "
        f"runner={bool(scrape.get('runner'))} ensure={ensure_msg} → {CANVAS.name}"
    )


if __name__ == "__main__":
    main()
