#!/usr/bin/env python3
"""Regenerate canvases/edge-bias-m11-eta-D256-k7-eb1-live.canvas.tsx (arm E)."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

ETAS = (0.05, 0.1, 0.25, 0.5)
MAX_STEPS = 40000
# Prefer Della gputest tag; also merge ailab if present.
RUN_TAGS = ("eb1E-gt", "eb1E-ailab")
JOB_NAMES = ("jc-eb1-gt", "jc-eb1-ailab")
LOG_GLOBS = ("jc_eb1_gt_*.out", "jc_eb1_ailab_*.out")
# Finals from the prior 40k untyped edge sweep (canvas baselines).
BASE_FALLBACK = {
    0.05: 0.9550268650054932,
    0.1: 0.9557061195373535,
    0.25: 0.9554095268249512,
    0.5: 0.9538306593894958,
}
SWEEP = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "jetclass_output/adam/sweep_lr"
)
LOGDIR = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs")
OUT = Path(
    "/home/jdezoort/.cursor/projects/home-jdezoort/canvases/"
    "edge-bias-m11-eta-D256-k7-eb1-live.canvas.tsx"
)


def js(obj) -> str:
    return json.dumps(obj)


def eta_tag(e: float) -> str:
    return f"{e:g}".replace(".", "p")


def parse_log(text: str) -> tuple[int, float | None]:
    step = 0
    for x in re.findall(r"global_step=(\d+)", text):
        step = max(step, int(x))
    for x in re.findall(r"\[train\] step=(\d+)", text):
        step = max(step, int(x))
    auc = None
    am = re.findall(r"val_roc_auc=([0-9.eE+-]+)", text)
    if am:
        try:
            auc = float(am[-1])
        except ValueError:
            pass
    return step, auc


def baseline_auc() -> dict[float, float]:
    # Matched 40k untyped finals are fixed; skip expensive SWEEP globs each refresh.
    return dict(BASE_FALLBACK)


def _tail_text(path: Path, max_bytes: int = 250_000) -> str:
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="ignore")
    except OSError:
        return ""


def collect() -> dict[float, dict]:
    out = {e: {"step": 0, "auc": None} for e in ETAS}
    for run_tag in RUN_TAGS:
        pat = re.compile(
            rf"capen-llama-att_4_256_(?P<eta>[0-9p]+)_.*{re.escape(run_tag)}"
        )
        for p in SWEEP.glob(f"*{run_tag}*"):
            name = p.name.replace(".parquet", "")
            m = pat.search(name)
            if not m:
                continue
            et = m.group("eta")
            eta = next((e for e in ETAS if eta_tag(e) == et), None)
            if eta is None:
                continue
            step, auc = 0, None
            pq = Path(str(p) + ".parquet") if p.is_dir() else p
            if pq.suffix == ".parquet" and pq.is_file():
                try:
                    import pyarrow.parquet as pqmod

                    df = pqmod.read_table(
                        pq, columns=["global_step", "val_roc_auc"]
                    ).to_pandas()
                    if "global_step" in df:
                        step = int(df["global_step"].dropna().max())
                    if "val_roc_auc" in df:
                        sub = df.dropna(subset=["val_roc_auc"])
                        if len(sub):
                            auc = float(sub["val_roc_auc"].iloc[-1])
                except Exception:
                    pass
            if step > out[eta]["step"]:
                out[eta]["step"] = step
            if auc is not None and (
                out[eta]["auc"] is None or step >= out[eta]["step"]
            ):
                out[eta]["auc"] = auc
    for glob_pat in LOG_GLOBS:
        # Newest logs first; skip ancient tails.
        logs = sorted(
            LOGDIR.glob(glob_pat),
            key=lambda q: q.stat().st_mtime,
            reverse=True,
        )[:40]
        for log in logs:
            text = _tail_text(log)
            if "eb1E" not in text and "edge-bias-m11" not in text:
                continue
            # Prefer explicit eta0= (array index maps changed when 0.05 was added).
            eta = None
            m_eta = re.search(r"eta0=([0-9.]+)", text)
            if m_eta:
                try:
                    cand = float(m_eta.group(1))
                except ValueError:
                    cand = None
                if cand in ETAS:
                    eta = cand
            if eta is None:
                continue
            st, auc = parse_log(text)
            if st > out[eta]["step"]:
                out[eta]["step"] = st
            if auc is not None:
                out[eta]["auc"] = auc
    return out


def squeue() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for job_name in JOB_NAMES:
        try:
            raw = subprocess.check_output(
                [
                    "squeue",
                    "-u",
                    "jdezoort",
                    "-n",
                    job_name,
                    "-o",
                    "%i %T %R",
                    "-h",
                ],
                text=True,
                timeout=30,
            )
        except Exception:
            continue
        for line in raw.splitlines():
            parts = line.split(None, 2)
            if len(parts) >= 2:
                rows.append(
                    (parts[0], parts[1], parts[2] if len(parts) > 2 else "")
                )
    return rows


def main() -> None:
    merged = collect()
    base = baseline_auc()
    jobs = squeue()
    n_run = sum(1 for _, s, _ in jobs if s == "RUNNING")
    n_pend = sum(1 for _, s, _ in jobs if s == "PENDING")
    n_done = sum(1 for e in ETAS if merged[e]["step"] >= MAX_STEPS - 5)
    updated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    steps = [merged[e]["step"] for e in ETAS]
    aucs = [merged[e]["auc"] for e in ETAS]
    fracs = [round(100.0 * s / MAX_STEPS, 1) for s in steps]
    base_auc = [base[e] for e in ETAS]
    cats = [str(e) for e in ETAS]

    table = []
    tones = []
    for eta in ETAS:
        st = merged[eta]["step"]
        a = merged[eta]["auc"]
        if st >= MAX_STEPS - 5:
            status, tone = "done", "success"
        elif n_run and st > 0:
            status, tone = "running", "info"
        elif n_pend and st == 0:
            status, tone = "pending", "neutral"
        elif st > 0:
            status, tone = "paused", "warning"
        else:
            status, tone = "idle", "neutral"
        table.append(
            [
                str(eta),
                status,
                f"{st}/{MAX_STEPS}",
                f"{100*st/MAX_STEPS:.0f}%",
                f"{a:.4f}" if a is not None else "—",
                f"{base[eta]:.4f}",
            ]
        )
        tones.append(tone)

    best = None
    for eta in ETAS:
        a = merged[eta]["auc"]
        if a is None:
            continue
        if best is None or a > best["auc"]:
            best = {"eta": eta, "auc": a, "step": merged[eta]["step"]}

    if n_run:
        callout = (
            f"{n_run} RUNNING. Best eb-dyn AUC: "
            + (
                f"{best['auc']:.4f} @ η₀={best['eta']} step {best['step']}"
                if best
                else "waiting first val"
            )
            + f". Matched baseline peak {max(base_auc):.4f}."
        )
        callout_title, callout_tone = "Training live", "success"
    elif n_pend:
        callout = f"{n_pend} pending (QOS / typed -s1i still holding slots)."
        callout_title, callout_tone = "Queued", "warning"
    else:
        callout = "No eb1 jobs in queue."
        callout_title, callout_tone = "Idle", "neutral"

    queue = [[jid, st, reason] for jid, st, reason in jobs] or [
        ["—", "—", "none in queue"]
    ]

    src = f"""import {{
  BarChart,
  Callout,
  Divider,
  Grid,
  H1,
  H2,
  Stack,
  Stat,
  Table,
  Text,
}} from "cursor/canvas";

const UPDATED = {js(updated)};
const MAX_STEPS = {MAX_STEPS};
const RECIPE = {js(
        f"L4/D256/H8 · typed §1 + edge-bias-m11 dynamic (arm E) · "
        f"k★=7 R★=0.1 · {MAX_STEPS // 1000}k warmup_flat_cosine · "
        f"eb1E-gt (Della gputest B=64×4)"
    )};
const ETA_CATS = {js(cats)};
const FRAC = {js(fracs)};
const EB_AUC = {js(aucs)};
const BASE_AUC = {js(base_auc)};
const TABLE = {js(table)};
const TONE = {js(tones)};
const QUEUE = {js(queue)};
const N_DONE = {n_done};
const N_RUN = {n_run};
const N_PEND = {n_pend};
const N_TOT = {len(ETAS)};
const BEST = {js(best)};
const CALLOUT_TONE = {js(callout_tone)};
const CALLOUT_TITLE = {js(callout_title)};
const CALLOUT_BODY = {js(callout)};

export default function EdgeBiasM11Live() {{
  return (
    <Stack gap={{16}}>
      <H1>Edge-bias M11 · arm E live</H1>
      <Text tone="secondary">{{RECIPE}}</Text>
      <Text tone="secondary">updated {{UPDATED}}</Text>
      <Callout tone={{CALLOUT_TONE}} title={{CALLOUT_TITLE}}>
        {{CALLOUT_BODY}}
      </Callout>
      <Grid columns={{4}} gap={{12}}>
        <Stat value={{`${{N_DONE}}/${{N_TOT}}`}} label="done" />
        <Stat value={{String(N_RUN)}} label="running" />
        <Stat value={{String(N_PEND)}} label="pending" />
        <Stat
          value={{BEST == null ? "—" : BEST.auc.toFixed(4)}}
          label={{
            BEST == null
              ? "best AUC"
              : `Best · η₀=${{BEST.eta}} @ ${{BEST.step}}`
          }}
        />
      </Grid>
      <H2>{{`Progress (% of ${{MAX_STEPS / 1000}}k)`}}</H2>
      <BarChart
        categories={{ETA_CATS}}
        series={{[{{ name: "% steps", data: FRAC }}]}}
        height={{180}}
      />
      <H2>Val ROC vs η₀ — eb-dyn vs baseline</H2>
      <BarChart
        categories={{ETA_CATS}}
        series={{[
          {{
            name: "baseline (done)",
            data: BASE_AUC.map((v) => (v == null ? Number.NaN : Number(v))),
          }},
          {{
            name: "eb-dyn live",
            data: EB_AUC.map((v) => (v == null ? Number.NaN : Number(v))),
          }},
        ]}}
        height={{220}}
      />
      <Divider />
      <H2>Per-η₀</H2>
      <Table
        headers={{["η₀", "status", "steps", "%", "eb-dyn AUC", "baseline AUC"]}}
        rows={{TABLE}}
        rowTone={{TONE}}
      />
      <H2>Queue</H2>
      <Table headers={{["job", "state", "reason"]}} rows={{QUEUE}} />
    </Stack>
  );
}}
"""
    OUT.write_text(src)
    print(f"wrote {OUT}")
    print(
        f"done={n_done}/{len(ETAS)} run={n_run} pend={n_pend} best={best}"
    )


if __name__ == "__main__":
    main()
