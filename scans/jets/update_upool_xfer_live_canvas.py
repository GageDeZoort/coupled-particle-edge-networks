#!/usr/bin/env python3
"""Refresh live canvas for μP transfer validation (uniform + attentional).

  python scans/jets/update_upool_xfer_live_canvas.py

Uniform: Della D=256 + D=512 on gputest (a100≤59m).
Attentional: Lambda 8×A100 (``ubuntu@193.122.155.21``).

Rewrites:
  ~/.cursor/projects/home-jdezoort/canvases/upool-xfer-validate-live.canvas.tsx
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law")
LOG_DIR = ROOT / "logs"
CANVAS = Path(
    "/home/jdezoort/.cursor/projects/home-jdezoort/canvases/upool-xfer-validate-live.canvas.tsx"
)
CSV_OUT = REPO / "paper" / "data" / "upool_xfer_validate.csv"
JSON_OUT = REPO / "paper" / "data" / "upool_xfer_validate_live.json"

# Uniform: D=256 job array 0–4; D=512 job array 5–9 (same gputest recipe).
JOB_D256 = "14882958"
JOB_D512 = "14890706"
JOB_LABEL = f"{JOB_D256}+{JOB_D512}"
TAG_UPOOL = "upool-xfer-b96-g4-s1000-dh32-k10-R0p1"
TAG_APOOL = "apool-xfer-b48-g8-s1000-dh32-k10-R0p1"
WIDTHS = (256, 512)
ETAS = (0.05, 0.1, 0.25, 0.5, 1.0)
MAX_STEPS = 1000
ET = ZoneInfo("America/New_York")
_JOB_NAMES = ("jc-upool-xfer",)

LAMBDA_HOST = "ubuntu@193.122.155.21"
LAMBDA_CM = Path.home() / ".cache" / "cm-lambda-193"
LAMBDA_LOGDIR = "/home/ubuntu/logs/apool_xfer"
LAMBDA_RUNROOT = "/home/ubuntu/runs/apool_xfer"


def _eta_tag(eta: float) -> str:
    return f"{eta:g}".replace(".", "p")


def _task_id(width: int, eta: float) -> int:
    return WIDTHS.index(width) * len(ETAS) + ETAS.index(eta)


def _slurm_meta(width: int, eta: float) -> tuple[str, int, str]:
    tid = _task_id(width, eta)
    if width == 256:
        return JOB_D256, tid, f"jc_upool_xfer_{JOB_D256}_{tid}"
    return JOB_D512, tid, f"jc_upool_xfer_{JOB_D512}_{tid}"


def _squeue_map() -> dict[int, str]:
    out: dict[int, str] = {}
    try:
        proc = subprocess.run(
            [
                "squeue",
                "-u",
                "jdezoort",
                "-o",
                "%i %t %j",
                "-h",
                "-n",
                ",".join(_JOB_NAMES),
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return out
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        jid, st, name = parts[0], parts[1], parts[2]
        if "_" not in jid:
            continue
        try:
            arr_tid = int(jid.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if name == "jc-upool-xfer":
            out[arr_tid] = st
    return out


def _sacct_for_job(job: str, *, logical_offset: int = 0) -> dict[int, str]:
    out: dict[int, str] = {}
    try:
        proc = subprocess.run(
            [
                "sacct",
                "-j",
                job,
                "--format=JobID,State",
                "-n",
                "-P",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:
        return out
    for line in proc.stdout.splitlines():
        if "|" not in line:
            continue
        jid, state = line.split("|", 1)
        if "_" not in jid or ".batch" in jid or ".extern" in jid:
            continue
        st = state.strip().split()[0]
        if st not in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
            continue
        try:
            arr_tid = int(jid.rsplit("_", 1)[-1])
        except ValueError:
            continue
        tid = logical_offset + arr_tid
        prev = out.get(tid)
        if prev == "FAILED" and st == "COMPLETED":
            continue
        out[tid] = st
    return out


def _sacct_terminal() -> dict[int, str]:
    out = {
        tid: st
        for tid, st in _sacct_for_job(JOB_D256, logical_offset=0).items()
        if tid < len(ETAS)
    }
    # D512 array tasks are already 5–9 on the gputest recipe.
    out.update(
        {
            tid: st
            for tid, st in _sacct_for_job(JOB_D512, logical_offset=0).items()
            if tid >= len(ETAS)
        }
    )
    return out


def _log_path(width: int, eta: float) -> Path:
    _, _, stem = _slurm_meta(width, eta)
    return LOG_DIR / f"{stem}.out"


def _parse_val_pts(text: str) -> tuple[list[dict], int]:
    pts: list[dict] = []
    step = 0
    for m in re.finditer(
        r"(?:\[val\].*?step=|global_step[=:\s]+)(\d+).*?val_roc_auc[=:\s]+([0-9.]+)",
        text,
    ):
        s, roc = int(m.group(1)), float(m.group(2))
        pts.append({"S": s, "roc": roc})
        step = max(step, s)
    for m in re.finditer(r"(?:global_step|step)=(\d+)", text):
        step = max(step, int(m.group(1)))
    return pts, step


def _parse_log(width: int, eta: float) -> dict:
    p = _log_path(width, eta)
    if not p.is_file():
        return {"pts": [], "step": 0, "log": p.name if p.exists() else ""}
    pts, step = _parse_val_pts(p.read_text(errors="ignore"))
    return {"pts": pts, "step": step, "log": p.name}


def _from_parquet(width: int, eta: float, tag: str, root: Path = ROOT) -> list[dict]:
    et = _eta_tag(eta)
    pts: list[dict] = []
    for p in sorted(root.glob(f"*_{width}_{et}_*{tag}*.parquet")):
        try:
            df = pd.read_parquet(p, columns=["global_step", "val_roc_auc"])
        except Exception:
            continue
        sub = df.dropna(subset=["val_roc_auc"]).sort_values("global_step")
        for _, row in sub.iterrows():
            pts.append({"S": int(row["global_step"]), "roc": float(row["val_roc_auc"])})
    by_s: dict[int, float] = {}
    for pt in pts:
        by_s[pt["S"]] = pt["roc"]
    return [{"S": s, "roc": by_s[s]} for s in sorted(by_s)]


def collect_uniform_cells() -> list[dict]:
    sq = _squeue_map()
    terminal = _sacct_terminal()
    cells: list[dict] = []
    for w in WIDTHS:
        for eta in ETAS:
            tid = _task_id(w, eta)
            pq_pts = _from_parquet(w, eta, TAG_UPOOL)
            log_info = _parse_log(w, eta)
            pts = pq_pts if pq_pts else log_info["pts"]
            step = max([0, *[p["S"] for p in pts], log_info["step"]])
            if tid in sq:
                st_raw = sq[tid]
                status = {
                    "R": "running",
                    "PD": "pending",
                    "CG": "running",
                }.get(st_raw, "running" if st_raw == "R" else "pending")
            elif tid in terminal:
                st = terminal[tid]
                if st == "COMPLETED" or step >= MAX_STEPS:
                    status = "done"
                elif st == "CANCELLED" and step < MAX_STEPS and not pts:
                    status = "pending"
                else:
                    status = "failed"
            elif step >= MAX_STEPS:
                status = "done"
            elif pts:
                status = "running" if step < MAX_STEPS else "done"
            else:
                status = "pending"
            best = max((p["roc"] for p in pts), default=None)
            last = pts[-1]["roc"] if pts else None
            cells.append(
                {
                    "task": tid,
                    "width": w,
                    "eta0": eta,
                    "status": status,
                    "step": step,
                    "max_steps": MAX_STEPS,
                    "best_roc": best,
                    "last_roc": last,
                    "pts": pts,
                    "log": log_info["log"],
                    "partition": "gputest",
                    "pool": "uniform",
                }
            )
    return cells


def _lambda_ssh(cmd: str, *, timeout: int = 60) -> str:
    """Run remote command via ControlMaster if available."""
    base = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
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


def collect_attn_cells() -> list[dict]:
    """Scrape Lambda attentional-pool sequential sweep."""
    master = _lambda_ssh(f"cat {LAMBDA_LOGDIR}/master.log 2>/dev/null || true")
    running_proc = _lambda_ssh(
        "pgrep -af 'test_cpen_jetclass.*apool-xfer' 2>/dev/null | head -1 || true"
    ).strip()

    # Active (D, eta) from latest RUN line not yet OK/FAIL.
    active: tuple[int, float] | None = None
    for line in master.splitlines():
        m = re.search(r"RUN task=\d+ D=(\d+) H=\d+ eta=([0-9.]+)", line)
        if m:
            active = (int(m.group(1)), float(m.group(2)))
        if re.search(r" OK D=(\d+) eta=([0-9.]+)", line) or re.search(
            r" FAIL D=(\d+) eta=([0-9.]+)", line
        ):
            # completed that cell; clear active if it matches
            m2 = re.search(r"D=(\d+) eta=([0-9.]+)", line)
            if m2 and active and active == (int(m2.group(1)), float(m2.group(2))):
                active = None

    done_ok: set[tuple[int, float]] = set()
    failed: set[tuple[int, float]] = set()
    for line in master.splitlines():
        m = re.search(r"OK D=(\d+) eta=([0-9.]+)", line)
        if m:
            done_ok.add((int(m.group(1)), float(m.group(2))))
        m = re.search(r"FAIL D=(\d+) eta=([0-9.]+)", line)
        if m:
            failed.add((int(m.group(1)), float(m.group(2))))

    # Bulk-fetch all cell logs in one SSH.
    names = " ".join(
        f"D{w}_eta{_eta_tag(e)}_R0p1.out" for w in WIDTHS for e in ETAS
    )
    bundled = _lambda_ssh(
        f"cd {LAMBDA_LOGDIR} 2>/dev/null && for f in {names}; do "
        f'echo "=====FILE:$f====="; cat "$f" 2>/dev/null || true; done'
    )
    by_file: dict[str, str] = {}
    cur = None
    buf: list[str] = []
    for line in bundled.splitlines():
        if line.startswith("=====FILE:") and line.endswith("====="):
            if cur is not None:
                by_file[cur] = "\n".join(buf)
            cur = line[len("=====FILE:") : -len("=====")]
            buf = []
        else:
            buf.append(line)
    if cur is not None:
        by_file[cur] = "\n".join(buf)

    cells: list[dict] = []
    for w in WIDTHS:
        for eta in ETAS:
            tid = _task_id(w, eta)
            fname = f"D{w}_eta{_eta_tag(eta)}_R0p1.out"
            text = by_file.get(fname, "")
            pts, step = _parse_val_pts(text) if text else ([], 0)
            key = (w, float(eta))
            # normalize eta key match
            key_match = any(
                kw == w and abs(ke - float(eta)) < 1e-9 for kw, ke in done_ok
            )
            fail_match = any(
                kw == w and abs(ke - float(eta)) < 1e-9 for kw, ke in failed
            )
            is_active = (
                active is not None
                and active[0] == w
                and abs(active[1] - float(eta)) < 1e-9
                and bool(running_proc)
            )
            if key_match or step >= MAX_STEPS:
                status = "done"
            elif fail_match and step < MAX_STEPS:
                status = "failed"
            elif is_active or (pts and step < MAX_STEPS and active and active[0] == w):
                status = "running"
            elif pts and step < MAX_STEPS:
                status = "running"
            elif text and "Traceback" in text and step < MAX_STEPS:
                status = "failed"
            else:
                status = "pending"
            # Prefer active flag for the current cell
            if is_active:
                status = "running"
            best = max((p["roc"] for p in pts), default=None)
            last = pts[-1]["roc"] if pts else None
            cells.append(
                {
                    "task": tid,
                    "width": w,
                    "eta0": eta,
                    "status": status,
                    "step": step,
                    "max_steps": MAX_STEPS,
                    "best_roc": best,
                    "last_roc": last,
                    "pts": pts,
                    "log": fname if text else "",
                    "partition": "lambda",
                    "pool": "attn",
                }
            )
    return cells


def _js(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))


def _panel_stats(cells: list[dict]) -> dict:
    n_done = sum(1 for c in cells if c["status"] == "done")
    n_run = sum(1 for c in cells if c["status"] == "running")
    n_pd = sum(1 for c in cells if c["status"] == "pending")
    n_fail = sum(1 for c in cells if c["status"] == "failed")
    active = next((c for c in cells if c["status"] == "running"), None)
    active_s = (
        f"D{active['width']} η={active['eta0']} step={active['step']}"
        if active
        else "—"
    )
    best_by_w: dict[int, list[float | None]] = {w: [] for w in WIDTHS}
    for w in WIDTHS:
        for eta in ETAS:
            c = next(
                (
                    x
                    for x in cells
                    if x["width"] == w and abs(float(x["eta0"]) - float(eta)) < 1e-9
                ),
                None,
            )
            best_by_w[w].append(
                c["best_roc"] if c and c["best_roc"] is not None else None
            )
    step_set: set[int] = set()
    for c in cells:
        for p in c["pts"]:
            step_set.add(int(p["S"]))
    steps = sorted(step_set)
    if len(steps) > 12:
        idx = sorted(
            {0, len(steps) - 1, *[round(i * (len(steps) - 1) / 11) for i in range(12)]}
        )
        steps = [steps[i] for i in idx]
    curve_series = []
    for c in cells:
        if not c["pts"]:
            continue
        by_s = {int(p["S"]): float(p["roc"]) for p in c["pts"]}
        data = [by_s.get(s) for s in steps]
        if all(v is None for v in data):
            continue
        curve_series.append({"name": f"D{c['width']} η={c['eta0']}", "data": data})
    return {
        "n_done": n_done,
        "n_run": n_run,
        "n_pd": n_pd,
        "n_fail": n_fail,
        "active": active_s,
        "best_256": best_by_w[256],
        "best_512": best_by_w[512],
        "steps": steps,
        "curves": curve_series,
        "cells": cells,
    }


def render_canvas(uniform: dict, attn: dict, updated: str) -> str:
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
const MAX_STEPS = {MAX_STEPS};
const ETA_CATS = {_js([str(e) for e in ETAS])};
const U_JOB = {_js(JOB_LABEL)};
const U_TAG = {_js(TAG_UPOOL)};
const U_DONE = {uniform["n_done"]};
const U_RUN = {uniform["n_run"]};
const U_PD = {uniform["n_pd"]};
const U_FAIL = {uniform["n_fail"]};
const U_ACTIVE = {_js(uniform["active"])};
const U_BEST_D256 = {_js(uniform["best_256"])};
const U_BEST_D512 = {_js(uniform["best_512"])};
const U_STEP_CATS: string[] = {_js([str(s) for s in uniform["steps"]])};
const U_CURVES: {{ name: string; data: (number | null)[] }}[] = {_js(uniform["curves"])};
const U_CELLS = {_js(uniform["cells"])};
const A_TAG = {_js(TAG_APOOL)};
const A_HOST = {_js(LAMBDA_HOST)};
const A_DONE = {attn["n_done"]};
const A_RUN = {attn["n_run"]};
const A_PD = {attn["n_pd"]};
const A_FAIL = {attn["n_fail"]};
const A_ACTIVE = {_js(attn["active"])};
const A_BEST_D256 = {_js(attn["best_256"])};
const A_BEST_D512 = {_js(attn["best_512"])};
const A_STEP_CATS: string[] = {_js([str(s) for s in attn["steps"]])};
const A_CURVES: {{ name: string; data: (number | null)[] }}[] = {_js(attn["curves"])};
const A_CELLS = {_js(attn["cells"])};
// END_AUTO_DATA

function fmt(x: number | null | undefined, d = 4): string {{
  return x == null || Number.isNaN(Number(x)) ? "—" : Number(x).toFixed(d);
}}

function numOrNan(x: number | null | undefined): number {{
  return x == null || Number.isNaN(Number(x)) ? Number.NaN : Number(x);
}}

function Panel(props: {{
  title: string;
  subtitle: string;
  recipe: string;
  nDone: number;
  nRun: number;
  nPd: number;
  nFail: number;
  active: string;
  best256: (number | null)[];
  best512: (number | null)[];
  stepCats: string[];
  curves: {{ name: string; data: (number | null)[] }}[];
  cells: any[];
}}) {{
  const bestSeries = [
    {{ name: "D=256", data: props.best256.map(numOrNan), tone: "info" as const }},
    {{ name: "D=512", data: props.best512.map(numOrNan), tone: "success" as const }},
  ];
  const hasBest = bestSeries.some((s) => s.data.some((v) => Number.isFinite(v)));
  const hasCurves = props.curves.length > 0 && props.stepCats.length > 0;
  const tableRows = props.cells.map((c: any) => [
    String(c.task),
    String(c.width),
    String(c.eta0),
    c.partition ?? "—",
    c.status,
    `${{c.step}}/${{c.max_steps}}`,
    fmt(c.best_roc),
    fmt(c.last_roc),
  ]);
  const rowTone = props.cells.map((c: any) =>
    c.status === "done"
      ? "success"
      : c.status === "running"
        ? "info"
        : c.status === "failed"
          ? "danger"
          : undefined
  );
  return (
    <Stack gap={{16}}>
      <Stack gap={{6}}>
        <H2>{{props.title}}</H2>
        <Text tone="secondary">{{props.subtitle}}</Text>
        <Row gap={{8}}>
          <Pill tone="success">{{props.nDone}}/10 done</Pill>
          <Pill tone="info">{{props.nRun}} running</Pill>
          <Pill>{{props.nPd}} pending</Pill>
          {{props.nFail > 0 ? <Pill tone="danger">{{props.nFail}} failed</Pill> : null}}
          <Pill>active: {{props.active}}</Pill>
        </Row>
      </Stack>
      <Grid columns={{4}} gap={{12}}>
        <Stat value={{`${{props.nDone}}/10`}} label="cells done" />
        <Stat value={{String(props.nRun)}} label="running" />
        <Stat value={{String(props.nPd)}} label="pending" />
        <Stat
          value={{props.nFail > 0 ? String(props.nFail) : props.active}}
          label={{props.nFail > 0 ? "failed" : "active cell"}}
        />
      </Grid>
      <Callout tone="neutral" title="Recipe">{{props.recipe}}</Callout>
      <Text tone="secondary" size="small">
        Best val ROC vs η₀ · peak within ≤{{MAX_STEPS}} steps
      </Text>
      {{hasBest ? (
        <LineChart
          categories={{ETA_CATS}}
          series={{bestSeries}}
          height={{260}}
          beginAtZero={{false}}
        />
      ) : (
        <Text tone="secondary">Waiting for first validation points…</Text>
      )}}
      {{hasCurves ? (
        <Stack gap={{8}}>
          <Text tone="secondary" size="small">Val ROC vs step</Text>
          <LineChart
            categories={{props.stepCats}}
            series={{props.curves.map((s) => ({{
              name: s.name,
              data: s.data.map(numOrNan),
            }}))}}
            height={{280}}
            beginAtZero={{false}}
          />
        </Stack>
      ) : null}}
      <Table
        headers={{["task", "D", "η₀", "partition", "status", "step", "best ROC", "last ROC"]}}
        rows={{tableRows}}
        rowTone={{rowTone}}
        striped
      />
    </Stack>
  );
}}

export default function TransferValidateLive() {{
  return (
    <Stack gap={{24}}>
      <Stack gap={{6}}>
        <H1>μP transfer validation</H1>
        <Text tone="secondary">
          Debug scale check — CompleteP + Adam η=η₀/√D. Uniform (Della) vs attentional
          (Lambda) output pool. Not a paper figure.
        </Text>
        <Text tone="tertiary" size="small">Updated {{UPDATED}}</Text>
      </Stack>

      <Panel
        title="Uniform output pool"
        subtitle={{`Jobs ${{U_JOB}} · tag ${{U_TAG}} · D256+D512=gputest`}}
        recipe={{`L=4 · d_h=32 · k=10 · R★=0.1 · B=96×4→384 · Adam · ${{MAX_STEPS}} steps · D∈{{256,512}} × η₀`}}
        nDone={{U_DONE}}
        nRun={{U_RUN}}
        nPd={{U_PD}}
        nFail={{U_FAIL}}
        active={{U_ACTIVE}}
        best256={{U_BEST_D256}}
        best512={{U_BEST_D512}}
        stepCats={{U_STEP_CATS}}
        curves={{U_CURVES}}
        cells={{U_CELLS}}
      />

      <Divider />

      <Panel
        title="Attentional output pool"
        subtitle={{`Host ${{A_HOST}} · tag ${{A_TAG}} · 8×A100 sequential`}}
        recipe={{`L=4 · d_h=32 · k=10 · R★=0.1 · B=48×8→384 · Adam · ${{MAX_STEPS}} steps · class-token attn pool`}}
        nDone={{A_DONE}}
        nRun={{A_RUN}}
        nPd={{A_PD}}
        nFail={{A_FAIL}}
        active={{A_ACTIVE}}
        best256={{A_BEST_D256}}
        best512={{A_BEST_D512}}
        stepCats={{A_STEP_CATS}}
        curves={{A_CURVES}}
        cells={{A_CELLS}}
      />
    </Stack>
  );
}}
"""


def main() -> None:
    uniform_cells = collect_uniform_cells()
    attn_cells = collect_attn_cells()
    uniform = _panel_stats(uniform_cells)
    attn = _panel_stats(attn_cells)
    updated = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in uniform_cells + attn_cells:
        rows.append(
            {
                "pool": c["pool"],
                "task": c["task"],
                "width": c["width"],
                "eta_0": c["eta0"],
                "partition": c["partition"],
                "status": c["status"],
                "step": c["step"],
                "val_roc_auc": c["best_roc"],
                "last_roc": c["last_roc"],
                "source": c["log"],
            }
        )
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
    JSON_OUT.write_text(
        json.dumps(
            {"updated": updated, "uniform": uniform_cells, "attn": attn_cells},
            indent=2,
        )
    )
    CANVAS.parent.mkdir(parents=True, exist_ok=True)
    CANVAS.write_text(render_canvas(uniform, attn, updated))
    print(
        f"{updated} xfer canvas "
        f"uniform={uniform['n_done']}/10(run={uniform['n_run']}) "
        f"attn={attn['n_done']}/10(run={attn['n_run']}) "
        f"→ {CANVAS.name}"
    )


if __name__ == "__main__":
    main()
