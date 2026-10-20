#!/usr/bin/env python3
"""Live canvas: attn vs uniform Adam const-LR η₀×D @ 10k (Lambda).

Rewrites:
  ~/.cursor/projects/home-jdezoort/canvases/hxfer-apool-eta-constlr-live.canvas.tsx
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
    "hxfer-apool-eta-constlr-live.canvas.tsx"
)
JSON_OUT = REPO / "paper" / "data" / "hxfer_apool_eta_constlr_live.json"
CSV_OUT = REPO / "paper" / "data" / "hxfer_pool_eta_constlr_live_status.csv"

LAMBDA_HOST = "ubuntu@193.122.155.21"
LAMBDA_CM = Path.home() / ".cache" / "cm-lambda-193"
ETAS = (0.01, 0.05, 0.1, 0.18, 0.25, 0.5, 1.0)
MAX_STEPS = 10000
ET = ZoneInfo("America/New_York")

# Della TopTagging FT from hxfer upool D=512 η₀=0.1
FT_JOB_NAME = "tt-ft-D512-upool"
FT_ROOT = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune")
FT_INIT = (
    FT_ROOT
    / "init_ckpts"
    / "hxfer-upool-D512-eta0p1-v2stream-last.ckpt"
)
FT_MAX_STEPS = 5000
FT_RUN_TAG = "ft-L4D512-upool-hxfer0p1-gputest-b128-g4-s5000-k6-R0p1"

ARMS = {
    "attn": {
        "widths": (512, 384, 256),
        "tag": "hxfer-apool-constlr-b64-g8-s10000-dh32-k6-R0p1-v2stream",
        "logdir": "/home/ubuntu/logs/hxfer_apool_eta",
        "runroot": "/home/ubuntu/runs/hxfer_apool_eta/jetclass_output/adam/sweep_lr",
        "runner": "run_hxfer_apool_eta_constlr_lambda8",
        "label": "attn",
    },
    "uniform": {
        "widths": (512, 384),
        "tag": "hxfer-upool-constlr-b64-g8-s10000-dh32-k6-R0p1-v2stream",
        "logdir": "/home/ubuntu/logs/hxfer_upool_eta",
        "runroot": "/home/ubuntu/runs/hxfer_upool_eta/jetclass_output/adam/sweep_lr",
        "runner": "run_hxfer_upool_eta_constlr_lambda8",
        "label": "uniform",
    },
}


def _js(x) -> str:
    return json.dumps(x)


def _eta_tag(eta: float) -> str:
    return f"{eta:g}".replace(".", "p")


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
import json, re, glob
from pathlib import Path

ETAS=[0.01,0.05,0.1,0.18,0.25,0.5,1.0]
MAX_STEPS=10000
ARMS={
 "attn":{"widths":[512,384,256],"tag":"hxfer-apool-constlr-b64-g8-s10000-dh32-k6-R0p1-v2stream",
         "logdir":str(Path.home()/"logs"/"hxfer_apool_eta"),
         "runroot":str(Path.home()/"runs"/"hxfer_apool_eta"/"jetclass_output"/"adam"/"sweep_lr"),
         "runner":"run_hxfer_apool_eta_constlr_lambda8"},
 "uniform":{"widths":[512,384],"tag":"hxfer-upool-constlr-b64-g8-s10000-dh32-k6-R0p1-v2stream",
         "logdir":str(Path.home()/"logs"/"hxfer_upool_eta"),
         "runroot":str(Path.home()/"runs"/"hxfer_upool_eta"/"jetclass_output"/"adam"/"sweep_lr"),
         "runner":"run_hxfer_upool_eta_constlr_lambda8"},
}

def eta_tag(e):
    return f"{e:g}".replace(".","p")

def scrape_arm(name, cfg):
    import subprocess
    runner=False
    try:
        out=subprocess.check_output(["pgrep","-af",cfg["runner"]], text=True)
        runner=cfg["runner"] in out
    except Exception:
        pass
    LOGDIR=Path(cfg["logdir"]); RUNROOT=Path(cfg["runroot"]); TAG=cfg["tag"]
    cells=[]; active=None
    for wi,W in enumerate(cfg["widths"]):
      for ei,eta in enumerate(ETAS):
        task=wi*len(ETAS)+ei
        et=eta_tag(eta)
        log=LOGDIR/f"D{W}_eta{et}_R0p1.out"
        text=log.read_text(errors="ignore") if log.exists() else ""
        is_v2=(TAG in text) or ("n_jets=5,120,000" in text) or ("n_train = 5120000" in text)
        # upool logs may say uniform-output-pool
        if name=="uniform" and ("uniform-output-pool" in text or "uniform_output_pool" in text or TAG in text):
            is_v2=True
        step=0; best=None; pts=[]
        pq_paths=sorted(glob.glob(str(RUNROOT/f"*_{W}_{et}_*{TAG}.parquet")))
        has_pq=bool(pq_paths)
        if is_v2:
            for m in re.finditer(r"\[val\].*?step=(\d+).*?val_roc_auc=([0-9.]+)", text):
                s,roc=int(m.group(1)), float(m.group(2))
                pts.append({"S":s,"roc":roc}); step=max(step,s)
                best=roc if best is None else max(best,roc)
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
                        pts=[{"S":int(r.global_step),"roc":float(r.val_roc_auc)} for r in sub.itertuples()]
                is_v2=True
            except Exception:
                pass
        # Attn D=256 remaining cells are stopped — keep finished ones only.
        status="pending"
        if not is_v2 and not has_pq:
            status="pending"
        elif step>=MAX_STEPS:
            status="done"
        elif step>0 or (is_v2 and any(s in text for s in (
            "Initializing distributed","GPU available: True","[attn-gamma]",
            "global_step=","n_jets=5,120,000","[train] single-task"))):
            status="running"
        if is_v2 and "Traceback" in text and status!="done" and step<=0:
            status="failed"
        cell={"arm":name,"task":task,"width":W,"eta0":eta,"status":status,
              "step":step,"max_steps":MAX_STEPS,"best_roc":best,"curve":pts[-40:]}
        cells.append(cell)
        if status=="running" and active is None:
            active=cell
    n_tot=len(cfg["widths"])*len(ETAS)
    return {"runner":runner,"cells":cells,"active":active,
            "n_done":sum(1 for c in cells if c["status"]=="done"),
            "n_run":sum(1 for c in cells if c["status"]=="running"),
            "n_fail":sum(1 for c in cells if c["status"]=="failed"),
            "n_tot":n_tot}

out={k: scrape_arm(k,v) for k,v in ARMS.items()}
print(json.dumps(out))
PY
"""
    try:
        out = _lambda_ssh(remote, timeout=150)
        return json.loads(out.strip().splitlines()[-1])
    except Exception as e:
        return {"attn": {"cells": [], "error": str(e)}, "uniform": {"cells": [], "error": str(e)}}


def scrape_toptag_ft() -> dict:
    """Local Della scrape: TopTagging FT from hxfer upool D=512 η₀=0.1."""
    import re

    job_id = None
    state = "unknown"
    node = ""
    elapsed = ""
    try:
        q = subprocess.run(
            [
                "squeue",
                "-u",
                "jdezoort",
                "-n",
                FT_JOB_NAME,
                "-h",
                "-o",
                "%i %T %M %R",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        line = (q.stdout or "").strip().splitlines()
        if line:
            parts = line[0].split(None, 3)
            job_id = parts[0]
            state = parts[1] if len(parts) > 1 else "unknown"
            elapsed = parts[2] if len(parts) > 2 else ""
            node = parts[3] if len(parts) > 3 else ""
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}

    log = None
    if job_id:
        cand = FT_ROOT / "logs" / f"tt_ft_{job_id}.out"
        if cand.exists():
            log = cand
    if log is None:
        logs = sorted(FT_ROOT.glob("logs/tt_ft_*.out"), key=lambda p: p.stat().st_mtime)
        # Prefer logs that mention this run tag
        for p in reversed(logs):
            try:
                head = p.read_text(errors="ignore")[:4000]
            except Exception:
                continue
            if FT_RUN_TAG in head or "hxfer-upool-D512-eta0p1" in head:
                log = p
                if job_id is None:
                    m = re.search(r"tt_ft_(\d+)\.out$", p.name)
                    job_id = m.group(1) if m else job_id
                break

    step = 0
    best_roc = None
    last_val = None
    text = ""
    if log and log.exists():
        try:
            # Tail only — FT logs can be large
            raw = log.read_bytes()
            text = raw[-400_000:].decode("utf-8", errors="ignore")
        except Exception:
            text = ""
        for m in re.finditer(r"global_step=(\d+)", text):
            step = max(step, int(m.group(1)))
        for m in re.finditer(r"\[val\].*?step=(\d+).*?val_roc_auc=([0-9.]+)", text):
            s, roc = int(m.group(1)), float(m.group(2))
            step = max(step, s)
            best_roc = roc if best_roc is None else max(best_roc, roc)
            last_val = roc
        for m in re.finditer(r"val_roc_auc=([0-9.]+)", text):
            roc = float(m.group(1))
            best_roc = roc if best_roc is None else max(best_roc, roc)

    # Parquet fallback
    pq_dir = FT_ROOT / "toptagging_output"
    if pq_dir.is_dir():
        for pq in pq_dir.rglob(f"*{FT_RUN_TAG}*.parquet"):
            try:
                import pyarrow.parquet as pqmod

                df = pqmod.read_table(
                    pq, columns=["global_step", "val_roc_auc"]
                ).to_pandas()
                if len(df):
                    step = max(step, int(df["global_step"].max()))
                    sub = df.dropna(subset=["val_roc_auc"])
                    if len(sub):
                        best_roc = float(sub["val_roc_auc"].max())
                        last_val = float(
                            sub.sort_values("global_step").iloc[-1]["val_roc_auc"]
                        )
            except Exception:
                pass

    status = "pending"
    if state in ("RUNNING", "COMPLETING"):
        status = "running"
    elif state in ("PENDING", "CONFIGURING"):
        status = "pending"
    elif step >= FT_MAX_STEPS:
        status = "done"
    elif job_id is None and step > 0:
        status = "done" if step >= FT_MAX_STEPS else "stopped"
    elif "Traceback" in text and step < FT_MAX_STEPS:
        status = "failed"
    elif state in ("COMPLETED",):
        status = "done" if step >= FT_MAX_STEPS else "stopped"
    elif state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
        status = "failed" if "Traceback" in text or state == "FAILED" else "stopped"

    return {
        "job_id": job_id,
        "slurm_state": state,
        "node": node,
        "elapsed": elapsed,
        "status": status,
        "step": step,
        "max_steps": FT_MAX_STEPS,
        "best_roc": best_roc,
        "last_val_roc": last_val,
        "init_from": str(FT_INIT),
        "run_tag": FT_RUN_TAG,
        "recipe": "D=512 η₀=0.1 upool → TopTag live k=6 R=0.1 · B=128×4 · 5k · gpu-test",
        "log": str(log) if log else "",
    }


def ensure_upool_runner(scrape: dict) -> str:
    uni = scrape.get("uniform") or {}
    if uni.get("runner"):
        return "ok"
    # Only ensure uniform runner (attn arm stopped by design).
    cmd = (
        "mkdir -p ~/logs/hxfer_upool_eta && "
        "nohup env FRESH=0 bash ~/cpen/scans/jets/run_hxfer_upool_eta_constlr_lambda8.sh "
        ">~/logs/hxfer_upool_eta/master.log 2>&1 & echo started:$!"
    )
    try:
        out = _lambda_ssh(cmd, timeout=30)
        return "restarted:" + out.strip().replace("\n", " ")[:120]
    except Exception as e:
        return f"restart_failed:{e}"


def _series(cells, widths):
    cats = [f"{e:g}" for e in ETAS]
    tones = ["accent", "info", "success"]
    series = []
    for i, W in enumerate(widths):
        data = []
        for eta in ETAS:
            cell = next(
                (c for c in cells if c["width"] == W and abs(c["eta0"] - eta) < 1e-12),
                None,
            )
            # Only plot finished (≥95% steps) or mid with a value
            v = None
            if cell and cell.get("best_roc") is not None:
                if cell["status"] == "done" or cell["step"] >= int(0.95 * MAX_STEPS):
                    v = cell["best_roc"]
                elif cell["status"] == "running" and cell["step"] >= 2000:
                    v = cell["best_roc"]
            data.append(v)
        series.append({"name": f"D={W}", "data": data, "tone": tones[i % len(tones)]})
    return cats, series


def render(scrape: dict, updated: str, ensure: str, ft: dict | None = None) -> str:
    attn = scrape.get("attn") or {}
    uni = scrape.get("uniform") or {}
    ft = ft or {}
    a_cells = attn.get("cells") or []
    u_cells = uni.get("cells") or []
    a_cats, a_series = _series(a_cells, ARMS["attn"]["widths"])
    u_cats, u_series = _series(u_cells, ARMS["uniform"]["widths"])
    active = uni.get("active") or attn.get("active")
    active_s = "—"
    if active:
        active_s = (
            f"{active.get('arm','?')} D={active['width']} η₀={active['eta0']:g} "
            f"step={active['step']}"
        )
    ft_status = ft.get("status") or "—"
    ft_pill_tone = (
        "success"
        if ft_status == "done"
        else "info"
        if ft_status == "running"
        else "warning"
        if ft_status in ("pending", "stopped")
        else "deleted"
        if ft_status in ("failed", "error")
        else "neutral"
    )
    ft_stat_tone = (
        "success"
        if ft_status == "done"
        else "info"
        if ft_status == "running"
        else "warning"
        if ft_status in ("pending", "stopped")
        else "danger"
        if ft_status in ("failed", "error")
        else "info"
    )
    ft_step_s = f"{ft.get('step', 0)}/{ft.get('max_steps', FT_MAX_STEPS)}"
    ft_roc = ft.get("best_roc")
    ft_job = ft.get("job_id") or "—"
    ft_node = ft.get("node") or "—"
    ft_elapsed = ft.get("elapsed") or "—"
    ft_recipe = ft.get("recipe") or ""
    ft_active = (
        f"TopTag FT job={ft_job} {ft_status} step={ft.get('step', 0)} "
        f"on {ft_node} ({ft_elapsed})"
        if ft
        else "—"
    )

    def table_rows(cells):
        return [
            [
                c.get("arm", ""),
                str(c["task"]),
                str(c["width"]),
                f"{c['eta0']:g}",
                c["status"],
                f"{c['step']}/{c['max_steps']}",
                None if c.get("best_roc") is None else round(float(c["best_roc"]), 5),
            ]
            for c in cells
        ]

    def tones(cells):
        return [
            "success"
            if c["status"] == "done"
            else "info"
            if c["status"] == "running"
            else "danger"
            if c["status"] == "failed"
            else None
            for c in cells
        ]

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
    "k★=6 R★=0.1 · B=64×8=512 · Adam const-LR · 10k single-pass · η₀ grid · attn vs uniform"
)};
const ACTIVE = {_js(active_s)};
const ENSURE = {_js(ensure)};
const A_DONE = {attn.get("n_done", 0)};
const A_TOT = {attn.get("n_tot", 15)};
const A_RUN = {attn.get("n_run", 0)};
const U_DONE = {uni.get("n_done", 0)};
const U_TOT = {uni.get("n_tot", 10)};
const U_RUN = {uni.get("n_run", 0)};
const U_RUNNER = {_js(bool(uni.get("runner")))};
const A_CATS: string[] = {_js(a_cats)};
const A_SERIES = {_js(a_series)};
const U_CATS: string[] = {_js(u_cats)};
const U_SERIES = {_js(u_series)};
const A_TABLE = {_js(table_rows(a_cells))};
const U_TABLE = {_js(table_rows(u_cells))};
const A_TONE = {_js(tones(a_cells))};
const U_TONE = {_js(tones(u_cells))};
const ERR = {_js((attn.get("error") or "") + (uni.get("error") or ""))};
const FT_STATUS = {_js(ft_status)};
const FT_PILL_TONE = {_js(ft_pill_tone)};
const FT_STAT_TONE = {_js(ft_stat_tone)};
const FT_STEP = {_js(ft_step_s)};
const FT_ROC = {_js(None if ft_roc is None else round(float(ft_roc), 5))};
const FT_JOB = {_js(ft_job)};
const FT_NODE = {_js(ft_node)};
const FT_ELAPSED = {_js(ft_elapsed)};
const FT_RECIPE = {_js(ft_recipe)};
const FT_ACTIVE = {_js(ft_active)};
const FT_INIT_PATH = {_js(ft.get("init_from") or str(FT_INIT))};
// END_AUTO_DATA

function fmt(x: number | null | undefined, d = 4): string {{
  return x == null || Number.isNaN(Number(x)) ? "—" : Number(x).toFixed(d);
}}

export default function HxferPoolEtaConstlrLive() {{
  const aSeries = A_SERIES.map((s: any) => ({{
    name: s.name,
    data: (s.data as (number | null)[]).map((v) => (v == null ? Number.NaN : Number(v))),
    tone: s.tone,
  }}));
  const uSeries = U_SERIES.map((s: any) => ({{
    name: s.name,
    data: (s.data as (number | null)[]).map((v) => (v == null ? Number.NaN : Number(v))),
    tone: s.tone,
  }}));
  const hasA = A_SERIES.some((s: any) => (s.data as (number | null)[]).some((v) => v != null));
  const hasU = U_SERIES.some((s: any) => (s.data as (number | null)[]).some((v) => v != null));
  return (
    <Stack gap={{20}}>
      <Stack gap={{6}}>
        <H1>Hparam transfer — attn vs uniform @ 10k</H1>
        <Text tone="secondary">{{RECIPE}}</Text>
        <Text tone="secondary">{{HOST}} · {{UPDATED}}</Text>
        <Callout tone="info" title="Campaign switch">
          Attentional arm stopped after D=512/384 (+ partial D=256). Now filling
          **uniform** pool for D=512→384 to compare η₀ transfer.
        </Callout>
        <Row gap={{8}}>
          <Pill tone="success">attn {{A_DONE}}/{{A_TOT}}</Pill>
          <Pill tone={{U_DONE === U_TOT ? "success" : "info"}}>uniform {{U_DONE}}/{{U_TOT}}</Pill>
          <Pill tone="info">{{U_RUN}} uni running</Pill>
          <Pill tone={{U_RUNNER ? "success" : "danger"}}>{{U_RUNNER ? "upool runner up" : "upool runner down"}}</Pill>
          <Pill tone={{FT_PILL_TONE as any}}>TopTag FT {{FT_STATUS}}</Pill>
        </Row>
        <Text>Active: {{ACTIVE}}</Text>
        <Text>TopTag FT: {{FT_ACTIVE}}</Text>
        {{ERR ? <Callout tone="danger" title="Scrape error">{{ERR}}</Callout> : null}}
      </Stack>

      <Grid columns={{4}} gap={{12}}>
        <Stat value={{`${{A_DONE}}/${{A_TOT}}`}} label="attn done" tone="success" />
        <Stat value={{`${{U_DONE}}/${{U_TOT}}`}} label="uniform done" />
        <Stat value={{String(U_RUN)}} label="uniform running" tone="info" />
        <Stat value={{"512"}} label="B_eff" />
      </Grid>

      <Divider />
      <H2>TopTagging fine-tune (Della gpu-test)</H2>
      <Text tone="secondary">{{FT_RECIPE}}</Text>
      <Grid columns={{4}} gap={{12}}>
        <Stat value={{FT_JOB}} label="slurm job" />
        <Stat value={{FT_STEP}} label="FT step" tone={{FT_STAT_TONE as any}} />
        <Stat value={{fmt(FT_ROC, 5)}} label="best val ROC" />
        <Stat value={{FT_ELAPSED}} label="elapsed" />
      </Grid>
      <Table
        headers={{["job", "status", "node", "step", "best ROC", "init"]}}
        rows={{[[
          FT_JOB,
          FT_STATUS,
          FT_NODE,
          FT_STEP,
          fmt(FT_ROC, 5),
          "hxfer-upool D512 η₀=0.1 last.ckpt",
        ]]}}
        rowTone={{[FT_STAT_TONE === "danger" ? "danger" : FT_STAT_TONE === "success" ? "success" : "info"]}}
      />
      <Text tone="secondary">init: {{FT_INIT_PATH}}</Text>

      <Divider />
      <H2>Attentional pool — best val ROC vs η₀</H2>
      {{hasA ? (
        <LineChart categories={{A_CATS}} series={{aSeries}} height={{260}} fill />
      ) : (
        <Callout tone="info" title="No attn cells">Waiting.</Callout>
      )}}
      <Table
        headers={{["arm", "task", "D", "η₀", "status", "step", "best ROC"]}}
        rows={{A_TABLE.map((r: any) => [r[0], r[1], r[2], r[3], r[4], r[5], fmt(r[6], 5)])}}
        rowTone={{A_TONE}}
      />

      <Divider />
      <H2>Uniform pool — best val ROC vs η₀ (live)</H2>
      {{hasU ? (
        <LineChart categories={{U_CATS}} series={{uSeries}} height={{260}} fill />
      ) : (
        <Callout tone="info" title="Uniform filling">D=512 η₀=0.05 launching first.</Callout>
      )}}
      <Table
        headers={{["arm", "task", "D", "η₀", "status", "step", "best ROC"]}}
        rows={{U_TABLE.map((r: any) => [r[0], r[1], r[2], r[3], r[4], r[5], fmt(r[6], 5)])}}
        rowTone={{U_TONE}}
      />
    </Stack>
  );
}}
"""


def main() -> None:
    scrape = _scrape()
    ensure = ensure_upool_runner(scrape)
    if ensure.startswith("restarted"):
        scrape = _scrape()
    ft = scrape_toptag_ft()
    updated = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
    CANVAS.parent.mkdir(parents=True, exist_ok=True)
    CANVAS.write_text(render(scrape, updated, ensure, ft))
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(
        json.dumps({"updated": updated, "ensure": ensure, "toptag_ft": ft, **scrape}, indent=2)
        + "\n"
    )
    rows = []
    for arm in ("attn", "uniform"):
        for c in (scrape.get(arm) or {}).get("cells") or []:
            rows.append(
                {
                    "arm": arm,
                    "task": c.get("task"),
                    "width": c.get("width"),
                    "eta0": c.get("eta0"),
                    "status": c.get("status"),
                    "step": c.get("step"),
                    "max_steps": c.get("max_steps"),
                    "best_roc": c.get("best_roc"),
                }
            )
    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["arm", "task", "width", "eta0", "status", "step", "max_steps", "best_roc"],
        )
        w.writeheader()
        w.writerows(rows)
    a = scrape.get("attn") or {}
    u = scrape.get("uniform") or {}
    print(
        f"{updated} canvas attn={a.get('n_done',0)}/{a.get('n_tot',15)} "
        f"uni={u.get('n_done',0)}/{u.get('n_tot',10)}(run={u.get('n_run',0)}) "
        f"runner={bool(u.get('runner'))} ensure={ensure} "
        f"ft={ft.get('status')}@{ft.get('step',0)} job={ft.get('job_id')} → {CANVAS.name}"
    )


if __name__ == "__main__":
    main()
