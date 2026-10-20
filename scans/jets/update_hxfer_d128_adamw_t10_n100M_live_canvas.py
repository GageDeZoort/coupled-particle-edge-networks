#!/usr/bin/env python3
"""Live canvas: AdamW t=10 n_train=100M const LR 40k — attn L4/D128 + L6/D256 only."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
CANVAS = Path(
    "/home/jdezoort/.cursor/projects/home-jdezoort/canvases/"
    "hxfer-d128-adamw-t10-n100M-live.canvas.tsx"
)
JSON_OUT = REPO / "paper" / "data" / "hxfer_d128_adamw_t10_n100M_live.json"
ROOT = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law")
SWEEP = ROOT / "jetclass_output" / "adamw" / "sweep_lr"
LOGS = ROOT / "logs"
ETAS = (0.05, 0.1, 0.25)
MAX_STEPS = 40000
ET = ZoneInfo("America/New_York")

CELLS = (
    {
        "depth": 4,
        "width": 128,
        "tag": "hxfer-L4-d128-ap-aw-t10-n100M-s40000-k6-R0p1",
        "partition": "gputest",
        "log_prefix": "jc-awL4d128ap",
        "job_prefix": "jc-awL4d128ap",
        "label": "L4 / D128",
    },
    {
        "depth": 6,
        "width": 256,
        "tag": "hxfer-L6-d256-ap-aw-t10-n100M-s40000-k6-R0p1",
        "partition": "ailab",
        "log_prefix": "jc-awL6d256ap",
        "job_prefix": "jc-awL6d256ap",
        "label": "L6 / D256",
    },
    {
        "depth": 8,
        "width": 512,
        "tag": "hxfer-L8-d512-ap-aw-t10-n100M-s40000-k6-R0p1",
        "partition": "ailab",
        "log_prefix": "jc-awL8d512ap",
        "job_prefix": "jc-awL8d512ap",
        "label": "L8 / D512",
    },
)


def _js(x) -> str:
    return json.dumps(x)


def eta_tag(e: float) -> str:
    return f"{e:g}".replace(".", "p")


def _val_curve_from_log(text: str) -> list[dict]:
    pts: dict[int, float] = {}
    for m in re.finditer(r"val_roc_auc=([0-9.eE+-]+)", text):
        head = text[max(0, m.start() - 400) : m.start()]
        sm = list(re.finditer(r"step=(\d+)", head))
        if sm:
            pts[int(sm[-1].group(1))] = float(m.group(1))
    return [{"S": s, "roc": r} for s, r in sorted(pts.items())]


def scrape_cell(wcfg: dict, eta: float, task: int) -> dict:
    import pyarrow.parquet as pq

    depth, width, tag = wcfg["depth"], wcfg["width"], wcfg["tag"]
    et = eta_tag(eta)
    step = 0
    best = last = None
    pts: list[dict] = []
    streaming = None
    n_train = None
    graph_mode = None
    status = "pending"
    log_text = ""

    logs = sorted(LOGS.glob(f"{wcfg['log_prefix']}_e{et}_*.out"))
    if logs:
        log_text = logs[-1].read_text(errors="ignore")[-200_000:]
        if "mode=ram-subset" in log_text:
            graph_mode = "ram-subset"
        elif "mode=stream" in log_text or "STREAM" in log_text:
            graph_mode = "stream"
        for m in re.finditer(r"global_step=(\d+)", log_text):
            step = max(step, int(m.group(1)))
        pts = _val_curve_from_log(log_text)
        if pts:
            step = max(step, pts[-1]["S"])
            best = max(p["roc"] for p in pts)
            last = pts[-1]["roc"]

    pqs = sorted(SWEEP.glob(f"*_{depth}_{width}_{et}_*{tag}*.parquet"))
    if pqs:
        try:
            df = pq.read_table(
                pqs[-1],
                columns=["global_step", "val_roc_auc", "n_train", "jetclass_streaming"],
            ).to_pandas()
            if len(df):
                step = max(step, int(df["global_step"].max()))
                if "n_train" in df.columns:
                    n_train = int(df["n_train"].iloc[0])
                if "jetclass_streaming" in df.columns:
                    streaming = bool(df["jetclass_streaming"].iloc[0])
                sub = df.dropna(subset=["val_roc_auc"]).sort_values("global_step")
                if len(sub):
                    best = float(sub["val_roc_auc"].max())
                    last = float(sub.iloc[-1]["val_roc_auc"])
                    pts = [
                        {"S": int(r.global_step), "roc": float(r.val_roc_auc)}
                        for r in sub.itertuples()
                    ]
        except Exception:
            pass

    name = f"{wcfg['job_prefix']}-e{et}"
    queued = False
    try:
        q = subprocess.run(
            ["squeue", "-u", "jdezoort", "-n", name, "-h"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        queued = bool(q.stdout.strip())
    except Exception:
        pass

    if step >= MAX_STEPS:
        status = "done"
    elif graph_mode == "ram-subset" or (n_train is not None and n_train < 10_000_000):
        status = "bad-data"
    elif queued or step > 0:
        status = "running"
    elif "Traceback" in log_text or "Error" in log_text[-2000:]:
        status = "failed"
    elif log_text.strip():
        status = "init"

    return {
        "task": task,
        "label": wcfg["label"],
        "depth": depth,
        "width": width,
        "partition": wcfg["partition"],
        "eta0": eta,
        "status": status,
        "step": step,
        "max_steps": MAX_STEPS,
        "best_roc": best,
        "last_roc": last,
        "n_train": n_train,
        "streaming": streaming,
        "graph_mode": graph_mode,
        "curve": pts[-100:],
    }


def scrape() -> dict:
    cells = []
    ti = 0
    for wcfg in CELLS:
        etas = (0.1,) if wcfg["width"] == 512 else ETAS
        for eta in etas:
            cells.append(scrape_cell(wcfg, eta, ti))
            ti += 1
    return {
        "cells": cells,
        "n_done": sum(1 for c in cells if c["status"] == "done"),
        "n_run": sum(1 for c in cells if c["status"] == "running"),
        "n_fail": sum(1 for c in cells if c["status"] in ("failed", "bad-data")),
        "n_tot": len(cells),
        "max_steps": MAX_STEPS,
    }


def _curves(cells: list[dict], depth: int, width: int) -> tuple[list[str], list[dict]]:
    tones = ["accent", "info", "warning"]
    raw = []
    for i, eta in enumerate(ETAS):
        cell = next(
            (
                c
                for c in cells
                if c["depth"] == depth
                and c["width"] == width
                and abs(c["eta0"] - eta) < 1e-12
            ),
            None,
        )
        if not cell or not cell.get("curve"):
            continue
        raw.append(
            {
                "name": f"η₀={eta:g}",
                "tone": tones[i % 3],
                "steps": [int(p["S"]) for p in cell["curve"]],
                "data": [float(p["roc"]) for p in cell["curve"]],
            }
        )
    if not raw:
        return [], []
    all_s = sorted({s for ser in raw for s in ser["steps"]})
    cats = [str(s) for s in all_s]
    series = []
    for ser in raw:
        m = dict(zip(ser["steps"], ser["data"]))
        held = float(ser["data"][0])
        filled = []
        for s in all_s:
            if s in m:
                held = float(m[s])
            filled.append(held)
        series.append({"name": ser["name"], "tone": ser["tone"], "data": filled})
    return cats, series


def render(data: dict, updated: str) -> str:
    cells = data.get("cells") or []
    table = [
        [
            c["label"],
            f"{c['eta0']:g}",
            c["status"],
            f"{c['step']}/{c['max_steps']}",
            None if c.get("last_roc") is None else round(float(c["last_roc"]), 5),
            None if c.get("best_roc") is None else round(float(c["best_roc"]), 5),
        ]
        for c in cells
    ]
    tones = [
        "success"
        if c["status"] == "done"
        else "info"
        if c["status"] == "running"
        else "danger"
        if c["status"] in ("failed", "bad-data")
        else None
        for c in cells
    ]
    c128, s128 = _curves(cells, 4, 128)
    c256, s256 = _curves(cells, 6, 256)
    c512, s512 = _curves(cells, 8, 512)

    best128 = max(
        (c["best_roc"] for c in cells if c["width"] == 128 and c.get("best_roc") is not None),
        default=None,
    )
    best256 = max(
        (c["best_roc"] for c in cells if c["width"] == 256 and c.get("best_roc") is not None),
        default=None,
    )
    best512 = max(
        (c["best_roc"] for c in cells if c["width"] == 512 and c.get("best_roc") is not None),
        default=None,
    )

    return f"""import {{
  Grid,
  H1,
  LineChart,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
}} from "cursor/canvas";

const UPDATED = {_js(updated)};
const N_DONE = {data.get("n_done", 0)};
const N_RUN = {data.get("n_run", 0)};
const N_TOT = {data.get("n_tot", 6)};
const BEST128 = {_js(None if best128 is None else round(float(best128), 5))};
const BEST256 = {_js(None if best256 is None else round(float(best256), 5))};
const BEST512 = {_js(None if best512 is None else round(float(best512), 5))};
const C128: string[] = {_js(c128)};
const S128 = {_js(s128)};
const C256: string[] = {_js(c256)};
const S256 = {_js(s256)};
const C512: string[] = {_js(c512)};
const S512 = {_js(s512)};
const TABLE = {_js(table)};
const TONE = {_js(tones)};

export default function Attn100MLive() {{
  const s128 = S128.map((s: any) => ({{
    name: s.name,
    data: (s.data as number[]).map((v) => Number(v)),
    tone: s.tone,
  }}));
  const s256 = S256.map((s: any) => ({{
    name: s.name,
    data: (s.data as number[]).map((v) => Number(v)),
    tone: s.tone,
  }}));
  const s512 = S512.map((s: any) => ({{
    name: s.name,
    data: (s.data as number[]).map((v) => Number(v)),
    tone: s.tone,
  }}));
  return (
    <Stack gap={{20}}>
      <Stack gap={{6}}>
        <H1>Attn pool · 100M stream · const LR</H1>
        <Text tone="secondary">
          L4/D128 · L6/D256 · L8/D512 (η₀=0.1) · AdamW t=10 · 40k · B=128×4
        </Text>
        <Text tone="secondary">{{UPDATED}}</Text>
        <Row gap={{8}}>
          <Pill tone={{N_DONE === N_TOT ? "success" : "info"}}>
            {{N_DONE}}/{{N_TOT}} done
          </Pill>
          <Pill tone="info">{{N_RUN}} running</Pill>
        </Row>
      </Stack>
      <Grid columns={{4}} gap={{12}}>
        <Stat value={{String(N_RUN)}} label="Running" />
        <Stat
          value={{BEST128 == null ? "—" : String(BEST128)}}
          label="Best L4/D128"
        />
        <Stat
          value={{BEST256 == null ? "—" : String(BEST256)}}
          label="Best L6/D256"
        />
        <Stat
          value={{BEST512 == null ? "—" : String(BEST512)}}
          label="Best L8/D512"
        />
      </Grid>
      {{C128.length > 0 ? (
        <Stack gap={{6}}>
          <Text weight="semibold">L=4 D=128 — val ROC vs step</Text>
          <LineChart
            categories={{C128}}
            series={{s128}}
            height={{220}}
            beginAtZero={{false}}
          />
        </Stack>
      ) : null}}
      {{C256.length > 0 ? (
        <Stack gap={{6}}>
          <Text weight="semibold">L=6 D=256 — val ROC vs step</Text>
          <LineChart
            categories={{C256}}
            series={{s256}}
            height={{220}}
            beginAtZero={{false}}
          />
        </Stack>
      ) : null}}
      {{C512.length > 0 ? (
        <Stack gap={{6}}>
          <Text weight="semibold">L=8 D=512 η₀=0.1 — val ROC vs step</Text>
          <LineChart
            categories={{C512}}
            series={{s512}}
            height={{220}}
            beginAtZero={{false}}
          />
        </Stack>
      ) : null}}
      <Table
        headers={{["model", "η₀", "status", "step", "last ROC", "best ROC"]}}
        rows={{TABLE}}
        rowTone={{TONE}}
      />
    </Stack>
  );
}}
"""


def main() -> None:
    data = scrape()
    updated = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
    CANVAS.write_text(render(data, updated))
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps({"updated": updated, **data}, indent=2))
    print(f"wrote {CANVAS}")
    print(f"wrote {JSON_OUT} done={data['n_done']} run={data['n_run']}")


if __name__ == "__main__":
    main()
