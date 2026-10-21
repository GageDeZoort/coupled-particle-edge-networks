#!/usr/bin/env python3
"""Regenerate canvases/typed-edge-eta-D256-k7-s40k-live.canvas.tsx."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

ETAS = (0.01, 0.05, 0.1, 0.25, 0.5)
MAX_STEPS = 40000

# Parallel submission paths (whichever advances wins per η).
PATHS = (
    {
        "key": "gputest",
        "label": "Della gputest B=64×4",
        "run_tag": "typed-edge-b64-g4-s40000-dh32-k7-R0p1-s1i",
        "job_name": "jc-typed-k7",
        "log_glob": "jc_typed_k7_s40k_gt_*.out",
    },
    {
        "key": "ailab",
        "label": "ailab qos=gpu-test B=256×1",
        "run_tag": "typed-edge-b256-g1-s40000-dh32-k7-R0p1-s1i-ailab",
        "job_name": "jc-typed-ailab",
        "log_glob": "jc_typed_k7_ailab_*.out",
    },
    {
        "key": "a100",
        "label": "gpu-short 1×A100 B=256×1",
        "run_tag": "typed-edge-b256-g1-s40000-dh32-k7-R0p1-s1i",
        "job_name": "jc-typed-a100",
        "log_glob": "jc_typed_k7_a100_*.out",
    },
)

BASE_TAG = "edge-b64-g4-s40000-dh32-k7-R0p1"
SWEEP = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "jetclass_output/adam/sweep_lr"
)
LOGDIR = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs")
OUT = Path(
    "/home/jdezoort/.cursor/projects/home-jdezoort/canvases/"
    "typed-edge-eta-D256-k7-s40k-live.canvas.tsx"
)


def run(cmd: list[str], timeout: int = 40) -> str:
    try:
        return subprocess.check_output(cmd, text=True, timeout=timeout)
    except Exception:
        return ""


def js(obj) -> str:
    return json.dumps(obj)


def eta_tag(e: float) -> str:
    return f"{e:g}".replace(".", "p")


def parse_log(text: str) -> tuple[int, float | None, float | None]:
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
    loss = None
    lm = re.findall(r"val_loss=([0-9.eE+-]+)", text)
    if lm:
        try:
            loss = float(lm[-1])
        except ValueError:
            pass
    return step, auc, loss


def parquet_auc(run_dir: Path) -> tuple[int, float | None]:
    pq = Path(str(run_dir) + ".parquet")
    if not pq.is_file():
        return 0, None
    try:
        import pyarrow.parquet as pqmod

        cols = ["global_step", "val_roc_auc"]
        try:
            df = pqmod.read_table(pq, columns=cols).to_pandas()
        except Exception:
            df = pqmod.read_table(pq).to_pandas()
        step = int(df["global_step"].dropna().max()) if "global_step" in df else 0
        auc = None
        if "val_roc_auc" in df:
            sub = df.dropna(subset=["val_roc_auc"])
            if len(sub):
                auc = float(sub["val_roc_auc"].iloc[-1])
        return step, auc
    except Exception:
        return 0, None


def collect_path(path: dict) -> dict[float, dict]:
    out: dict[float, dict] = {}
    tag = path["run_tag"]
    for eta in ETAS:
        et = eta_tag(eta)
        row = {
            "step": 0,
            "auc": None,
            "loss": None,
            "status": "idle",
            "source": None,
        }
        # Exact tag match (avoid matching …-ailab when looking for non-ailab).
        cands = [
            p
            for p in SWEEP.glob(f"capen-llama-att_4_256_{et}_*")
            if tag in p.name
            and (tag.endswith("-ailab") or "-ailab" not in p.name)
        ]
        for d in cands:
            run_dir = d if d.is_dir() else SWEEP / d.name.replace(".parquet", "")
            pq_st, pq_auc = parquet_auc(run_dir if run_dir.is_dir() else d)
            if d.suffix == ".parquet":
                pq_st2, pq_auc2 = parquet_auc(SWEEP / d.name.replace(".parquet", ""))
                pq_st = max(pq_st, pq_st2)
                pq_auc = pq_auc if pq_auc is not None else pq_auc2
            if pq_st >= row["step"]:
                row["step"] = pq_st
                if pq_auc is not None:
                    row["auc"] = pq_auc
                row["source"] = "parquet"
        out[eta] = row

    # Overlay log scrape (often has fresher val AUC / step than parquet).
    # Require this path's run_tag so debunked pre-v2 logs cannot leak in.
    for log in sorted(LOGDIR.glob(path["log_glob"]), key=lambda p: p.stat().st_mtime):
        text = log.read_text(errors="ignore")[-400_000:]
        m_tag = re.search(r"run_tag=(\S+)", text)
        if m_tag is None or m_tag.group(1) != tag:
            continue
        m_eta = re.search(r"eta0=([0-9.]+)", text)
        if not m_eta:
            continue
        try:
            eta = float(m_eta.group(1))
        except ValueError:
            continue
        if eta not in out:
            continue
        step, auc, loss = parse_log(text)
        if step >= out[eta]["step"]:
            out[eta]["step"] = step
            out[eta]["source"] = log.name
        if auc is not None:
            out[eta]["auc"] = auc
        if loss is not None:
            out[eta]["loss"] = loss
    return out


def squeue_map() -> dict[str, list[str]]:
    """job_name → list of 'eta_or_task:STATE' strings."""
    text = run(
        [
            "squeue",
            "-u",
            "jdezoort",
            "-h",
            "-o",
            "%j %i %T %R",
        ]
    )
    by_name: dict[str, list[str]] = {}
    for line in text.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 3:
            continue
        name, jid, state = parts[0], parts[1], parts[2]
        reason = parts[3] if len(parts) > 3 else ""
        by_name.setdefault(name, []).append(f"{jid}:{state}:{reason[:40]}")
    return by_name


def baseline_auc() -> dict[float, float]:
    out: dict[float, float] = {}
    for eta in ETAS:
        et = eta_tag(eta)
        matches = sorted(SWEEP.glob(f"capen-llama-att_4_256_{et}_*{BASE_TAG}.parquet"))
        if not matches:
            continue
        try:
            import pyarrow.parquet as pqmod

            df = pqmod.read_table(matches[-1], columns=["val_roc_auc"]).to_pandas()
            out[eta] = float(df["val_roc_auc"].dropna().max())
        except Exception:
            pass
    return out


def best_across(paths_data: dict[str, dict[float, dict]], eta: float) -> dict:
    best = {"step": 0, "auc": None, "path": None, "status": "idle"}
    for key, rows in paths_data.items():
        r = rows.get(eta, {})
        st = int(r.get("step") or 0)
        if st > best["step"] or (
            st == best["step"] and r.get("auc") is not None and best["auc"] is None
        ):
            best = {
                "step": st,
                "auc": r.get("auc"),
                "path": key,
                "status": r.get("status", "idle"),
            }
    return best


def main() -> None:
    sq = squeue_map()
    paths_data = {p["key"]: collect_path(p) for p in PATHS}
    # Annotate status from squeue.
    for p in PATHS:
        jobs = sq.get(p["job_name"], [])
        n_run = sum(1 for j in jobs if ":RUNNING:" in j or j.split(":")[1] == "RUNNING")
        n_pend = sum(1 for j in jobs if ":PENDING:" in j or j.split(":")[1] == "PENDING")
        for eta, row in paths_data[p["key"]].items():
            if row["step"] >= MAX_STEPS - 5:
                row["status"] = "done"
            elif n_run:
                row["status"] = "running"
            elif n_pend:
                row["status"] = "pending"
            elif row["step"] > 0:
                row["status"] = "paused"
            else:
                row["status"] = "idle"

    base = baseline_auc()
    merged = {eta: best_across(paths_data, eta) for eta in ETAS}

    n_done = sum(1 for eta in ETAS if merged[eta]["step"] >= MAX_STEPS - 5)
    n_run = sum(
        1
        for p in PATHS
        for j in sq.get(p["job_name"], [])
        if j.split(":")[1] == "RUNNING"
    )
    n_pend = sum(
        1
        for p in PATHS
        for j in sq.get(p["job_name"], [])
        if j.split(":")[1] == "PENDING"
    )

    typed_auc = [merged[e]["auc"] for e in ETAS]
    typed_steps = [merged[e]["step"] for e in ETAS]
    typed_frac = [round(100.0 * s / MAX_STEPS, 1) for s in typed_steps]
    base_auc = [base.get(e) for e in ETAS]
    eta_cats = [str(e) for e in ETAS]

    table = []
    tones = []
    for eta in ETAS:
        m = merged[eta]
        st = m["step"]
        frac = 100.0 * st / MAX_STEPS
        if st >= MAX_STEPS - 5:
            status, tone = "done", "success"
        elif st > 0 and any(
            j.split(":")[1] == "RUNNING"
            for p in PATHS
            for j in sq.get(p["job_name"], [])
        ):
            status, tone = "running", "info"
        elif st > 0:
            status, tone = "paused", "warning"
        elif n_pend:
            status, tone = "pending", "neutral"
        else:
            status, tone = "idle", "neutral"
        # Per-path steps for detail
        detail = " · ".join(
            f"{p['key']}={paths_data[p['key']][eta]['step']}"
            for p in PATHS
            if paths_data[p["key"]][eta]["step"] or paths_data[p["key"]][eta]["auc"]
        ) or "—"
        table.append(
            [
                str(eta),
                status,
                f"{st}/{MAX_STEPS}",
                f"{frac:.0f}%",
                f"{m['auc']:.4f}" if m["auc"] is not None else "—",
                f"{base.get(eta):.4f}" if eta in base else "—",
                m["path"] or "—",
                detail,
            ]
        )
        tones.append(tone)

    best_typed = None
    for eta in ETAS:
        a = merged[eta]["auc"]
        if a is None:
            continue
        if best_typed is None or a > best_typed["auc"]:
            best_typed = {"eta": eta, "auc": a, "step": merged[eta]["step"]}

    queue_rows = []
    for p in PATHS:
        jobs = sq.get(p["job_name"], [])
        if not jobs:
            queue_rows.append([p["label"], p["job_name"], "—", "none in queue"])
            continue
        for j in jobs:
            jid, state, *rest = j.split(":", 2)
            reason = rest[0] if rest else ""
            queue_rows.append([p["label"], jid, state, reason])

    updated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    recipe = (
        "L4/D256/H8 · typed-edge §1 (knn:M12; hyper:M12+M22; particles:M11+M21 all edges) · "
        "k★=7 R★=0.1 noloop · ailab/a100/gputest · target 40k · tag -s1i"
    )

    # Callout
    if n_run:
        callout_tone = "success"
        callout_title = "Training live"
        callout_body = (
            f"{n_run} job(s) RUNNING across queues. Best typed val AUC so far: "
            + (
                f"{best_typed['auc']:.4f} @ η₀={best_typed['eta']} step {best_typed['step']}"
                if best_typed
                else "waiting first val checkpoint"
            )
            + f". Baseline peak remains {max(base.values()):.4f} @ η₀=0.1."
        )
    elif n_pend:
        callout_tone = "warning"
        callout_title = "Queued — not running yet"
        callout_body = (
            f"{n_pend} pending. Della gputest / ailab gpu-test / gpu-short are all "
            "armed; first slot that starts advances its own run_tag."
        )
    else:
        callout_tone = "info"
        callout_title = "Idle"
        callout_body = "No typed jobs in queue."

    src = f'''import {{
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

const UPDATED = {js(updated)};
const RECIPE = {js(recipe)};
const ETA_CATS = {js(eta_cats)};
const TYPED_STEPS = {js(typed_steps)};
const TYPED_FRAC = {js(typed_frac)};
const TYPED_AUC = {js(typed_auc)};
const BASE_AUC = {js(base_auc)};
const TABLE = {js(table)};
const TONE = {js(tones)};
const QUEUE = {js(queue_rows)};
const N_DONE = {n_done};
const N_RUN = {n_run};
const N_PEND = {n_pend};
const N_TOT = {len(ETAS)};
const BEST = {js(best_typed)};
const BASE_BEST = {{ eta: 0.1, auc: 0.955706, step: 40000 }};

export default function TypedEdgeEtaD256K7Live() {{
  const aucSeries = [
    {{
      name: "baseline val ROC (done)",
      data: BASE_AUC.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "neutral" as const,
    }},
    {{
      name: "typed val ROC (best path)",
      data: TYPED_AUC.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "info" as const,
    }},
  ];
  const progSeries = [
    {{
      name: "% of 40k steps",
      data: TYPED_FRAC.map((v) => Number(v)),
      tone: "info" as const,
    }},
  ];

  return (
    <Stack gap={{20}}>
      <Stack gap={{6}}>
        <H1>Typed-edge η₀ — L4/D256 @ k★=7</H1>
        <Text tone="secondary">{{RECIPE}}</Text>
        <Text tone="secondary">updated {{UPDATED}}</Text>
        <Row gap={{8}} wrap>
          <Pill tone={{N_DONE === N_TOT ? "success" : "info"}}>
            {{N_DONE}}/{{N_TOT}} done
          </Pill>
          <Pill tone="info">{{N_RUN}} running</Pill>
          <Pill tone="warning">{{N_PEND}} pending</Pill>
        </Row>
      </Stack>

      <Callout tone="{callout_tone}" title={js(callout_title)}>
        {{{js(callout_body)}}}
      </Callout>

      <Grid columns={{4}} gap={{12}}>
        <Stat
          value={{`${{N_DONE}}/${{N_TOT}}`}}
          label="η cells done"
          tone={{N_DONE === N_TOT ? "success" : "info"}}
        />
        <Stat value={{String(N_RUN)}} label="jobs running now" />
        <Stat value={{String(N_PEND)}} label="jobs pending" />
        <Stat
          value={{BEST == null ? "—" : BEST.auc.toFixed(4)}}
          label={{
            BEST == null
              ? "Best typed val AUC"
              : `Best typed · η₀=${{BEST.eta}} @ ${{BEST.step}}`
          }}
        />
      </Grid>

      <Stack gap={{8}}>
        <H2>Progress toward 40k (%)</H2>
        <LineChart
          categories={{ETA_CATS}}
          series={{progSeries as any}}
          height={{200}}
          beginAtZero
        />
        <Text tone="secondary" size="small">
          X = η₀ · Y = percent of 40k · best across gputest / ailab / a100 paths
        </Text>
      </Stack>

      <Stack gap={{8}}>
        <H2>Val ROC vs η₀ — typed vs baseline</H2>
        <LineChart
          categories={{ETA_CATS}}
          series={{aucSeries as any}}
          height={{240}}
          beginAtZero={{false}}
        />
        <Text tone="secondary" size="small">
          Baseline tag {{`{BASE_TAG}`}} @ 40k · typed points appear after first val ckpt
        </Text>
      </Stack>

      <Divider />

      <Stack gap={{8}}>
        <H2>Per-η (best path)</H2>
        <Table
          headers={{[
            "η₀",
            "status",
            "steps",
            "%",
            "typed AUC",
            "baseline AUC",
            "best path",
            "path steps",
          ]}}
          rows={{TABLE}}
          rowTone={{TONE as any}}
        />
      </Stack>

      <Stack gap={{8}}>
        <H2>Queue snapshot</H2>
        <Table
          headers={{["path", "job", "state", "reason"]}}
          rows={{QUEUE}}
        />
      </Stack>

      <Stack gap={{8}}>
        <H2>Architecture under test</H2>
        <Table
          headers={{["Piece", "Baseline", "Typed"]}}
          rows={{[
            [
              "Encoders",
              "Shared Linear(7→D), kNN padded; √(D/7)",
              "Star √(D/7) + kNN √(D/4); no pad",
            ],
            [
              "M12 (→ edges)",
              "All edges",
              "All live edges (hyper + knn)",
            ],
            [
              "M21 (→ particles)",
              "All edges → particles",
              "All edges → particles (hypers + knn)",
            ],
            [
              "M22",
              "Degree>2; else f₂₂=LN(hₑ)",
              "Hyper↔hyper only; f₂₂=0 on knn/tiny-star",
            ],
            [
              "Edge residual",
              "1/(√2 L) all edges",
              "Hyper 1/(√2 L); non-hyper 1/L",
            ],
          ]}}
        />
      </Stack>
    </Stack>
  );
}}
'''
    OUT.write_text(src)
    print(f"wrote {OUT}")
    print(
        f"done={n_done}/{len(ETAS)} run_jobs={n_run} pend_jobs={n_pend} "
        f"best={best_typed}"
    )


if __name__ == "__main__":
    main()
