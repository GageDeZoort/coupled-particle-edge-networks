#!/usr/bin/env python3
"""Regenerate canvases/adamw-tepoch-D256-s40k-live.canvas.tsx.

Also records uniqueness + noloop confirmations for the edge η / graph-construction
campaigns, and overlays the Adam η₀=0.1 baseline from the L4/D256 40k scan.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

T_EPOCHS = (0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0)
MAX_STEPS = 40000
BATCH = 256
N_GPUS = 1
D_HEAD = 32
KNN_K = 7
R_TAG = "0p1"
ETA0 = 0.1
JOB_NAME = "jc-aw-tepoch"

ADAM_SWEEP = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "jetclass_output/adam/sweep_lr"
)
ADAMW_SWEEP = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/"
    "jetclass_output/adamw/sweep_lr"
)
LOGDIR = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs")
OUT = Path(
    "/home/jdezoort/.cursor/projects/home-jdezoort/canvases/"
    "adamw-tepoch-D256-s40k-live.canvas.tsx"
)
WATCH_LOG = Path.home() / ".cache/jc_aw_tepoch_watchdogs/nohup_aw_tepoch.out"
ADAM_EDGE_TAG = "edge-b64-g4-s40000-dh32-k7-R0p1"


def run(cmd: list[str], timeout: int = 40) -> str:
    try:
        return subprocess.check_output(cmd, text=True, timeout=timeout)
    except Exception:
        return ""


def js(obj) -> str:
    return json.dumps(obj)


def t_tag(t: float) -> str:
    return f"{t:g}".replace(".", "p")


def run_tag_for(t: float) -> str:
    return (
        f"aw-tepoch-t{t_tag(t)}-n100M-b{BATCH}-g{N_GPUS}-s{MAX_STEPS}"
        f"-dh{D_HEAD}-k{KNN_K}-R{R_TAG}"
    )


def parse_log(text: str) -> tuple[int | None, float | None, float | None]:
    step = None
    train_steps = [int(x) for x in re.findall(r"\[train\] step=(\d+)", text)]
    glob_steps = [int(x) for x in re.findall(r"global_step=(\d+)", text)]
    if train_steps:
        step = train_steps[-1]
    if glob_steps:
        step = max(step or 0, glob_steps[-1])
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


def parquet_metrics(sweep: Path, glob_pat: str) -> tuple[int, float | None, float | None]:
    best_step = 0
    auc = loss = None
    for p in sweep.glob(glob_pat):
        pqpath = p if p.suffix == ".parquet" else p.parent / f"{p.name}.parquet"
        if not pqpath.is_file():
            continue
        try:
            import pyarrow.parquet as pq

            df = pq.read_table(
                pqpath, columns=["global_step", "val_roc_auc", "val_loss"]
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
    return best_step, auc, loss


def adam_baseline() -> dict:
    step, auc, loss = parquet_metrics(
        ADAM_SWEEP, f"capen-llama-att_4_256_0p1_*{ADAM_EDGE_TAG}*"
    )
    return {
        "optimizer": "adam",
        "eta0": 0.1,
        "t_epoch": 1.0,
        "B_global": 256,
        "step": step,
        "auc": auc,
        "loss": loss,
        "tag": ADAM_EDGE_TAG,
        "n_train": 10_240_000,
        "noloop": True,
        "unique": True,
    }


def confirm_campaigns() -> dict:
    """Spot-check uniqueness + noloop on finished edge / graph-construction dirs."""
    edge_dirs = list(ADAM_SWEEP.glob(f"*{ADAM_EDGE_TAG}"))
    edge_dirs = [p for p in edge_dirs if p.is_dir()]
    noloop_n = sum(1 for p in edge_dirs if "noloop" in p.name)
    knnselt_n = len(list(ADAM_SWEEP.glob("*knnselt*")))
    panel_a = len(list(ADAM_SWEEP.glob("*graph-k-noloop*v2cpNL*")))
    panel_b = len(list(ADAM_SWEEP.glob("*graph-R-noloop*v2cpNL*")))
    # Sample one edge log for live_knn_include_self
    include_self = None
    n_train_logged = None
    epochs_logged = None
    for logp in sorted(LOGDIR.glob("jc_edge_k7_s40k_gt_*.out"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
        text = logp.read_text(errors="ignore")[:8000]
        if "live_knn_include_self" in text:
            m = re.search(r"live_knn_include_self\s*=\s*(True|False)", text)
            if m:
                include_self = m.group(1) == "True"
            m2 = re.search(r"n_train\s*=\s*(\d+)", text)
            if m2:
                n_train_logged = int(m2.group(1))
            m3 = re.search(r"epochs\s*=\s*(\d+)", text)
            if m3:
                epochs_logged = int(m3.group(1))
            break
    return {
        "edge_run_dirs": len(edge_dirs),
        "edge_noloop_dirs": noloop_n,
        "knnselt_dirs": knnselt_n,
        "panel_a_noloop_artifacts": panel_a,
        "panel_b_noloop_artifacts": panel_b,
        "live_knn_include_self": include_self,
        "n_train_logged": n_train_logged,
        "epochs_logged": epochs_logged,
        "presentations_per_run": 40_000 * 256,
    }


def latest_log_for_t(t: float) -> Path | None:
    needle = f"t_epoch={t:g}"
    alts = {needle, f"t_epoch={float(t)}", f"t_epoch={t}"}
    hits: list[Path] = []
    for p in LOGDIR.glob("jc_aw_tepoch_s40k_*.out"):
        try:
            head = p.read_text(errors="ignore")[:8000]
        except Exception:
            continue
        # Prefer restarted n100M wave; skip wiped D=10.24M leftovers.
        if "n100M" not in head and "n_train=100000000" not in head:
            continue
        if any(a in head for a in alts):
            hits.append(p)
    if not hits:
        return None
    return max(hits, key=lambda p: p.stat().st_mtime)


def verify_adamw_n100m() -> dict:
    """Confirm running/recent AdamW logs use prescribed n_train=100M."""
    n_train = None
    lambda_0 = None
    wd = None
    t_epoch = None
    for p in sorted(LOGDIR.glob("jc_aw_tepoch_s40k_*.out"), key=lambda x: x.stat().st_mtime, reverse=True)[:12]:
        text = p.read_text(errors="ignore")
        if "n_train=100000000" not in text and "ntr100M" not in text:
            continue
        m = re.search(r"n_train\s*=\s*(\d+)", text)
        if m:
            n_train = int(m.group(1))
        m2 = re.search(r"lambda_0\s*=\s*([0-9.eE+-]+)", text)
        if m2:
            lambda_0 = float(m2.group(1))
        m3 = re.search(r"weight_decay\s*=\s*([0-9.eE+-]+)", text)
        if m3:
            wd = float(m3.group(1))
        m4 = re.search(r"t_epoch\s*=\s*([0-9.eE+-]+)", text)
        if m4:
            t_epoch = float(m4.group(1))
        if n_train == 100_000_000:
            break
    ok = n_train == 100_000_000
    return {
        "ok": ok,
        "n_train": n_train,
        "lambda_0": lambda_0,
        "weight_decay": wd,
        "t_epoch": t_epoch,
    }


def main() -> None:
    baseline = adam_baseline()
    confirm = confirm_campaigns()
    n100 = verify_adamw_n100m()

    queue: dict[int, dict] = {}
    live_jobs: list[list[str]] = []
    live_tones: list[str] = []
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
        tid = int(m.group(1) or m.group(2)) if m else None
        te = f"{T_EPOCHS[tid]:g}" if tid is not None and 0 <= tid < len(T_EPOCHS) else "—"
        live_jobs.append([jid, te, state, elapsed, part or "—", reason or "—"])
        live_tones.append("info" if "run" in state else "neutral")
        if tid is None:
            continue
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
            r"submitted task=(\d+) t_epoch=([0-9.]+) job=(\d+)",
            WATCH_LOG.read_text(errors="ignore"),
        ):
            submitted[int(m.group(1))] = m.group(3)

    cells = []
    for task, te in enumerate(T_EPOCHS):
        tag = run_tag_for(te)
        step_pq, auc_pq, loss_pq = parquet_metrics(
            ADAMW_SWEEP, f"capen-llama-att_4_256_0p1_*{tag}*"
        )
        step = step_pq or None
        auc, loss = auc_pq, loss_pq
        logp = latest_log_for_t(te)
        if logp is not None:
            s, a, lo = parse_log(logp.read_text(errors="ignore")[-200_000:])
            if s is not None:
                step = max(step or 0, s)
            if a is not None:
                auc = a
            if lo is not None:
                loss = lo

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

        delta = None
        if auc is not None and baseline["auc"] is not None:
            delta = auc - float(baseline["auc"])

        cells.append(
            {
                "task": task,
                "t_epoch": te,
                "status": status,
                "step": step or 0,
                "auc": auc,
                "loss": loss,
                "delta": delta,
                "jid": (q or {}).get("jid") or submitted.get(task, "—"),
                "tag": tag,
            }
        )

    n_done = sum(1 for c in cells if c["status"] == "done")
    n_run = sum(1 for c in cells if c["status"] == "running")
    n_pend = sum(1 for c in cells if c["status"] == "pending")
    n_paused = sum(1 for c in cells if c["status"] == "paused")
    n_tot = len(T_EPOCHS)

    best = None
    for c in cells:
        if c["auc"] is not None:
            if best is None or c["auc"] > best["auc"]:
                best = {"auc": c["auc"], "t_epoch": c["t_epoch"], "step": c["step"]}

    cats = [f"{t:g}" for t in T_EPOCHS]
    auc_series = [c["auc"] if c["auc"] is not None else None for c in cells]
    step_frac = [100.0 * c["step"] / MAX_STEPS for c in cells]
    # Baseline as constant series for overlay (same length)
    base_series = [
        float(baseline["auc"]) if baseline["auc"] is not None else None for _ in T_EPOCHS
    ]

    table = []
    tones = []
    for c in cells:
        table.append(
            [
                f"{c['t_epoch']:g}",
                c["status"],
                f"{c['step']}/{MAX_STEPS}",
                f"{100.0 * c['step'] / MAX_STEPS:.0f}%",
                "—" if c["loss"] is None else f"{c['loss']:.4f}",
                "—" if c["auc"] is None else f"{c['auc']:.4f}",
                "—" if c["delta"] is None else f"{c['delta']:+.4f}",
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
    base_auc_s = (
        f"{baseline['auc']:.4f}" if baseline["auc"] is not None else "—"
    )
    noloop_ok = (
        confirm["live_knn_include_self"] is False
        and confirm["knnselt_dirs"] == 0
        and confirm["edge_noloop_dirs"] == confirm["edge_run_dirs"]
        and confirm["edge_run_dirs"] > 0
    )
    unique_ok = (
        confirm["n_train_logged"] == 10_240_000
        and confirm["epochs_logged"] == 1
    )

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
  "L4/D256/H8 · η₀=0.1 · AdamW + scheduler=none · B=256×1 · 40k · n_train=100M (prescribed D for t_epoch λ; 10.24M unique presentations) · noloop star kin7 k★=7 R★=0.1";
const N_DONE: number = {n_done};
const N_RUN: number = {n_run};
const N_PEND: number = {n_pend};
const N_PAUSED: number = {n_paused};
const N_TOT: number = {n_tot};
const BEST = {js(best)};
const BASE_AUC = {js(baseline["auc"])};
const BASE_STEP = {js(baseline["step"])};
const T_CATS: string[] = {js(cats)};
const AUC_SERIES = {js(auc_series)};
const BASE_SERIES = {js(base_series)};
const STEP_FRAC = {js(step_frac)};
const TABLE = {js(table)};
const TONE = {js(tones)};
const CONFIRM = {js(confirm)};
const N100 = {js(n100)};
const NOLOOP_OK = {js(noloop_ok)};
const UNIQUE_OK = {js(unique_ok)};
const LIVE_JOBS = {js(live_jobs)};
const LIVE_TONE = {js(live_tones)};
const HAS_LIVE = {js(len(live_jobs) > 0)};

export default function AdamwTepochD256S40kLive() {{
  const hasAuc = AUC_SERIES.some((v) => v != null) || BASE_AUC != null;
  const aucSeries = [
    {{
      name: "AdamW val ROC",
      data: AUC_SERIES.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "accent" as const,
    }},
    {{
      name: `Adam η₀=0.1 baseline (${{BASE_AUC == null ? "—" : BASE_AUC.toFixed(4)}} @ ${{BASE_STEP}})`,
      data: BASE_SERIES.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "neutral" as const,
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
        <H1>AdamW t_epoch sweep vs Adam — L4/D256</H1>
        <Text tone="secondary">{{RECIPE}}</Text>
        <Text tone="secondary">
          a100 2h rolling watchdog (≤4) · jobs {{JOBS}} · updated {{UPDATED}}
        </Text>
        <Row gap={{8}} wrap>
          <Pill tone={{N_DONE === N_TOT ? "success" : "info"}}>
            {{N_DONE}}/{{N_TOT}} done
          </Pill>
          <Pill tone="info">{{N_RUN}} running</Pill>
          <Pill tone="warning">{{N_PAUSED}} between slots</Pill>
          <Pill tone="neutral">{{N_PEND}} pending</Pill>
          <Pill tone={{N100.ok ? "success" : "warning"}}>
            D=100M {{N100.ok ? "confirmed" : "pending"}}
          </Pill>
          <Pill tone={{UNIQUE_OK ? "success" : "warning"}}>
            unique jets {{UNIQUE_OK ? "confirmed" : "check"}}
          </Pill>
          <Pill tone={{NOLOOP_OK ? "success" : "warning"}}>
            noloop {{NOLOOP_OK ? "confirmed" : "check"}}
          </Pill>
        </Row>
      </Stack>

      <Callout tone={{N100.ok ? "success" : "info"}} title="AdamW t_epoch uses n_train=100M">
        Prescribed D for λ = √width·B/(t_epoch·η₀·n_train) is the full JetClass
        train set (100M). Stream presentations = 40k × 256 = 10.24M ≪ 100M ⇒
        unique, no wrap. Prior D=10.24M wave was cancelled and wiped.
        Logged n_train={{N100.n_train ?? "—"}} · λ₀={{N100.lambda_0 ?? "—"}} ·
        WD={{N100.weight_decay ?? "—"}} · t_epoch={{N100.t_epoch ?? "—"}}.
      </Callout>

      {{HAS_LIVE ? (
        <Stack gap={{8}}>
          <H2>Live Slurm jobs (jc-aw-tepoch)</H2>
          <Table
            headers={{["job", "t_epoch", "state", "time", "partition", "reason"]}}
            rows={{LIVE_JOBS}}
            rowTone={{LIVE_TONE as any}}
          />
        </Stack>
      ) : null}}

      <Callout tone={{UNIQUE_OK && NOLOOP_OK ? "success" : "info"}} title="Campaign checks (edge η + graph construction)">
        Edge Adam scan: n_train = 40k × 256 = 10.24M, epochs=1, stream single-pass.
        Logged n_train={{CONFIRM.n_train_logged ?? "—"}} epochs={{CONFIRM.epochs_logged ?? "—"}}.
        Self-loops off: live_knn_include_self={js(confirm["live_knn_include_self"])} ·
        edge dirs noloop={{CONFIRM.edge_noloop_dirs}}/{{CONFIRM.edge_run_dirs}} ·
        knnselt dirs={{CONFIRM.knnselt_dirs}} · panel (a) noloop artifacts={{CONFIRM.panel_a_noloop_artifacts}} ·
        panel (b) noloop artifacts={{CONFIRM.panel_b_noloop_artifacts}}.
      </Callout>

      <Grid columns={{4}} gap={{12}}>
        <Stat
          value={{`${{N_DONE}}/${{N_TOT}}`}}
          label="t_epoch cells done"
          tone={{N_DONE === N_TOT ? "success" : "info"}}
        />
        <Stat
          value={{BASE_AUC == null ? "—" : BASE_AUC.toFixed(4)}}
          label={{`Adam baseline · η₀=0.1 @ step ${{BASE_STEP ?? "—"}}`}}
        />
        <Stat
          value={{BEST == null ? "—" : BEST.auc.toFixed(4)}}
          label={{
            BEST == null
              ? "Best AdamW AUC so far"
              : `Best AdamW · t_epoch=${{BEST.t_epoch}} @ ${{BEST.step}}`
          }}
        />
        <Stat value={{String(N_RUN + N_PAUSED + N_PEND)}} label="still open" />
      </Grid>

      {{N_DONE < N_TOT ? (
        <Callout tone="info" title="Rolling on a100 (2h slots)">
          Each slot resumes last.ckpt toward 40k. t_epoch ∈
          {"{0.1, 0.5, 1, 5, 10, 25, 50}"}. Constant LR (scheduler=none), AdamW
          WD from t_epoch with n_train=100M. Same OP as the Adam η scan.
        </Callout>
      ) : (
        <Callout tone="success" title="Sweep complete">
          All seven t_epoch cells reached 40k steps.
        </Callout>
      )}}

      <Stack gap={{6}}>
        <H2>Progress toward 40k (%)</H2>
        <LineChart
          categories={{T_CATS}}
          series={{progSeries as any}}
          height={{200}}
          beginAtZero
        />
        <Text tone="secondary" size="small">
          X = t_epoch · Y = percent of 40k · Source: adamw parquet + Slurm logs
        </Text>
      </Stack>

      {{hasAuc ? (
        <Stack gap={{6}}>
          <H2>Val ROC vs t_epoch (AdamW) with Adam baseline</H2>
          <LineChart
            categories={{T_CATS}}
            series={{aucSeries as any}}
            height={{240}}
            beginAtZero={{false}}
          />
          <Text tone="secondary" size="small">
            Horizontal reference = Adam η₀=0.1 @ B_global=256, 40k
            (tag {ADAM_EDGE_TAG}), AUC {base_auc_s}. Δ column = AdamW − Adam.
          </Text>
        </Stack>
      ) : null}}

      <Stack gap={{8}}>
        <H2>Cells</H2>
        <Table
          headers={{["t_epoch", "status", "step", "%", "val loss", "val ROC", "Δ vs Adam", "job"]}}
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
        f"pend={n_pend} best={best} adam_base={baseline['auc']} "
        f"unique_ok={unique_ok} noloop_ok={noloop_ok} n100m_ok={n100['ok']} "
        f"live_jobs={len(live_jobs)}"
    )


if __name__ == "__main__":
    main()
