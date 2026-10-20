#!/usr/bin/env python3
"""Probe Lambda edge η×D + hparam-xfer arms; refresh live canvas + CSVs.

Covers:
  - Panel (c) edge η×D (Arm 0) on Lambda + D=128 gpu-test fill
  - Arm 1 / 1b: t-sweep @ 12k with 5k flat → cosine (PAUSED after D=512 t=10)
  - Arm 2: const-LR 10k AdamW t∈{1,5,10} + Adam @ D=256 on Lambda (live)
  - Arm 2b: same recipe @ D=384 on Della ailab B=128×4 (live)
  - Arm 3: Adam η₀×D fill-in @ D∈{256,384} × η₀∈{0.05,0.1,0.5,1} on ailab (live)

Run once, or loop:
  SSH_AUTH_SOCK=... python scans/jets/update_edge_eta_live_canvas.py
  nohup bash scans/jets/watch_edge_eta_live.sh &
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path("/home/jdezoort/coupled-particle-edge-networks")
CANVAS = Path(
    "/home/jdezoort/.cursor/projects/home-jdezoort/canvases/edge-eta-lambda-live.canvas.tsx"
)
CSV = REPO / "paper" / "data" / "edge_eta_live_status.csv"
HXFER_CSV = REPO / "paper" / "data" / "hparam_xfer_t_live_status.csv"
CONSTLR_CSV = REPO / "paper" / "data" / "hparam_xfer_constlr_live_status.csv"
JSON_OUT = Path.home() / ".cache" / "edge_eta_monitor" / "status.json"
FIG_MD = REPO / "paper" / "figures" / "fig_graph_construction.md"

HOST = os.environ.get("LAMBDA_HOST", "ubuntu@163.192.125.147")
IDENTITY = os.path.expanduser("~/.ssh/id_ed25519")
MAX_STEPS = 4000
WIDTHS = [256, 384, 512]
ETAS = [0.05, 0.1, 0.25, 0.5, 1.0]
TAG = "edge-b96-g4-s4000-dh32-k12-R0p1"

HXFER_MAX_STEPS = 12000
# Arm 1 / 1b (paused): t-sweep with cosine.
HXFER_T_EPOCHS = [1.0, 5.0, 10.0, 20.0]
HXFER_WIDTHS = [256, 512]
HXFER_TAG = "hxfer-t-b64-g8-s12000-k12-R0p1"
HXFER_ETA0 = 0.25
HXFER_WIDTH = 512

# Arm 2 (live): constant LR, no cosine — Lambda D=256.
CONSTLR_MAX_STEPS = 10000
CONSTLR_WIDTH = 256
CONSTLR_ETA0 = 0.25
CONSTLR_TAG = "hxfer-constlr-b64-g8-s10000-k12-R0p1"
# (optimizer, t_epoch or None for Adam)
CONSTLR_CELLS = [("adamw", 1.0), ("adamw", 5.0), ("adamw", 10.0), ("adam", None)]

# Arm 2b (live): same recipe @ D=384 on Della ailab (B=128×4).
DELLA_ROOT = Path(
    "/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law"
)
DELLA_LOGS = DELLA_ROOT / "logs"
CONSTLR_D384_WIDTH = 384
CONSTLR_D384_TAG = "hxfer-constlr-b128-g4-s10000-k12-R0p1"
# task index in jc_hxfer_c384_ailab_%A_%a.out
CONSTLR_D384_CELLS = [
    (0, "adam", None),
    (1, "adamw", 1.0),
    (2, "adamw", 5.0),
    (3, "adamw", 10.0),
]
CONSTLR_D384_CSV = REPO / "paper" / "data" / "hparam_xfer_constlr_d384_live_status.csv"

# Arm 3 (live): Adam η₀×D fill-in on Della ailab (panel a LR transfer).
# task = 4*d_idx + e_idx; D∈{256,384}, η₀∈{0.05,0.1,0.5,1.0}
# task 8: D=256 η₀=0.25 (Arm 2 Adam skipped; D=384@0.25 from Arm 2b)
ADAM_ETA_JOB_NAME = "jc-hxfer-aeta"
ADAM_ETA_LOG_GLOB = "jc_hxfer_adam_eta_ailab_*_*.out"
ADAM_ETA_TAG = "hxfer-adam-eta-b128-g4-s10000-k12-R0p1"
ADAM_ETA_WIDTHS = [256, 384]
ADAM_ETA_ETAS = [0.05, 0.1, 0.5, 1.0]
ADAM_ETA_CELLS = [
    (4 * di + ei, w, e)
    for di, w in enumerate(ADAM_ETA_WIDTHS)
    for ei, e in enumerate(ADAM_ETA_ETAS)
] + [(8, 256, 0.25)]
ADAM_ETA_CSV = REPO / "paper" / "data" / "hparam_xfer_adam_eta_live_status.csv"

# Panel (c) extension: Edge CAPEN η₀ grids on Della gpu-test.
# D=128 wiped 2026-10-01 (held out for later re-queue); D=384 live.
EDGE_GT_TAG = "edge-b96-g4-s4000-dh32-k12-R0p1"
EDGE_GT_MAX_STEPS = 4000
EDGE_GT_ETAS = [0.05, 0.1, 0.25, 0.5, 1.0]
# (width, job_name, log_prefix) — log files: {prefix}_{jobid}_{task}.out
EDGE_GT_SWEEPS = [
    (384, "jc-edge384", "jc_edge384_gt"),
]
EDGE_GT_CELLS = [
    (w, i, e)
    for w, _, _ in EDGE_GT_SWEEPS
    for i, e in enumerate(EDGE_GT_ETAS)
]
EDGE_GT_CSV = REPO / "paper" / "data" / "edge_eta_gputest_live_status.csv"
# Back-compat aliases (D=128-only names used in older notes).
EDGE_D128_JOB_NAME = "jc-edge128"
EDGE_D128_TAG = EDGE_GT_TAG
EDGE_D128_MAX_STEPS = EDGE_GT_MAX_STEPS
EDGE_D128_CSV = EDGE_GT_CSV

REMOTE_PY = r"""
import re, json
from pathlib import Path
try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

LOGDIR = Path.home() / "logs/edge_eta"
ROOT = Path.home() / "runs/edge_eta/jetclass_output/adam/sweep_lr"
MASTER = LOGDIR / "master.log"
WIDTHS = [256, 384, 512]
ETAS = [0.05, 0.1, 0.25, 0.5, 1.0]
TAG = "edge-b96-g4-s4000-dh32-k12-R0p1"
MAX_STEPS = 4000

HXFER_LOG = Path.home() / "logs/hparam_xfer"
HXFER_ROOT = Path.home() / "runs/hparam_xfer/jetclass_output/adamw/sweep_lr"
HXFER_MAX = 12000
HXFER_TAG = "hxfer-t-b64-g8-s12000-k12-R0p1"
HXFER_ETA = 0.25
HXFER_WIDTHS = [256, 512]
HXFER_TS = [1.0, 5.0, 10.0, 20.0]

def eta_tag(e):
    return f"{e:g}".replace(".", "p")

def t_tag(t):
    return f"{t:g}".replace(".", "p")

master_tail = ""
for mf in (MASTER, LOGDIR / "master_bankB.log"):
    if mf.is_file():
        lines = mf.read_text(errors="ignore").splitlines()
        master_tail += ("\n" if master_tail else "") + "\n".join(lines[-8:])

hxfer_master = ""
for mf in (
    HXFER_LOG / "master_t5.log",
    HXFER_LOG / "master_adaptive.log",
    HXFER_LOG / "master_t1.log",
    HXFER_LOG / "master_d512.log",
    HXFER_LOG / "master_constlr_d256.log",
):
    if mf.is_file():
        lines = mf.read_text(errors="ignore").splitlines()
        hxfer_master += ("\n" if hxfer_master else "") + "\n".join(lines[-30:])

paused_t_sweep = (HXFER_LOG / "PAUSED_D512_T_SWEEP").is_file()
paused_note = ""
if paused_t_sweep:
    paused_note = (HXFER_LOG / "PAUSED_D512_T_SWEEP").read_text(errors="ignore").strip()

try:
    import subprocess
    r = subprocess.run(
        ["pgrep", "-af", "run_edge_eta_lambda8|run_hparam_xfer|test_cpen_jetclass"],
        capture_output=True, text=True,
    )
    running = bool(r.stdout.strip())
    procs = [ln for ln in r.stdout.splitlines() if "pgrep" not in ln][:60]
except Exception:
    running = False
    procs = []

cells = []
for wi, W in enumerate(WIDTHS):
    for ei, E in enumerate(ETAS):
        task = wi * 5 + ei
        et = eta_tag(E)
        out = LOGDIR / f"D{W}_eta{et}_R0p1.out"
        status = "pending"
        step = 0
        roc = None
        acc = None
        loss = None
        last = ""
        err = ""
        if out.is_file():
            text = out.read_text(errors="ignore")
            for m in re.finditer(r"\[train\] step=(\d+)", text):
                step = max(step, int(m.group(1)))
            for m in re.finditer(r"global_step=(\d+)", text):
                step = max(step, int(m.group(1)))
            for m in re.finditer(r"\[val\].*?step=(\d+).*?val_roc_auc=([0-9.]+)", text):
                step = max(step, int(m.group(1)))
                roc = float(m.group(2))
            for m in re.finditer(r"val_roc_auc=([0-9.]+)", text):
                roc = float(m.group(1))
            for m in re.finditer(r"val_acc=([0-9.]+)", text):
                acc = float(m.group(1))
            for m in re.finditer(r"train_loss=([0-9.]+)", text):
                loss = float(m.group(1))
            lines = [ln for ln in text.splitlines() if ln.strip()]
            last = lines[-1][:160] if lines else ""
            # Ignore benign NCCL_* deprecation warnings; only real crashes count.
            if "Traceback" in text or "NCCL error" in text or "NCCL_ERROR" in text:
                err = "error"
            status = "running" if step < MAX_STEPS else "done"
            if "FAIL" in master_tail and f"D={W} eta={E}" in master_tail:
                status = "failed"
            # Don't mark done cells failed for deprecation noise.
            if err == "error" and step < MAX_STEPS and "Traceback" in text:
                status = "failed"
        if pq is not None and ROOT.is_dir():
            for p in ROOT.glob(f"*_{W}_{et}_*{TAG}.parquet"):
                try:
                    df = pq.read_table(
                        p, columns=["global_step", "val_roc_auc", "val_acc"]
                    ).to_pandas()
                    if len(df):
                        step = max(step, int(df["global_step"].max()))
                        sub = df.dropna(subset=["val_roc_auc"])
                        if len(sub):
                            row = sub.loc[sub["global_step"].idxmax()]
                            roc = float(row["val_roc_auc"])
                            if "val_acc" in row and row["val_acc"] == row["val_acc"]:
                                acc = float(row["val_acc"])
                        if step >= MAX_STEPS:
                            status = "done"
                except Exception:
                    pass
        if status == "pending" and any(
            (f"--width {W}" in p)
            and (f"--etas {E}" in p or f"--etas {E:g}" in p)
            and "edge_eta" in p
            for p in procs
        ):
            status = "running"
        cells.append(dict(
            task=task, width=W, eta0=E, status=status, step=step,
            max_steps=MAX_STEPS, val_roc_auc=roc, val_acc=acc, train_loss=loss,
            last=last, err=err,
        ))

active_tasks = {int(m.group(1)) for m in re.finditer(r"RUN task=(\d+) ", master_tail)}
for p in procs:
    if "edge_eta" not in p and "edge-b96" not in p:
        continue
    wm = re.search(r"--width (\d+)", p)
    em = re.search(r"--etas ([0-9.]+)", p)
    if not (wm and em):
        continue
    W, E = int(wm.group(1)), float(em.group(1))
    if W in WIDTHS and any(abs(E - e) < 1e-12 for e in ETAS):
        e_exact = next(e for e in ETAS if abs(E - e) < 1e-12)
        active_tasks.add(WIDTHS.index(W) * 5 + ETAS.index(e_exact))
for c in cells:
    if c["task"] in active_tasks and c["status"] not in ("done", "failed"):
        c["status"] = "running"
    elif (
        c["status"] == "running"
        and c["task"] not in active_tasks
        and c["step"] == 0
        and c["step"] < MAX_STEPS
    ):
        c["status"] = "pending"

# --- hparam-xfer t-sweep (D=256 + D=512) ---
hxfer_cells = []
ti = 0
for W in HXFER_WIDTHS:
    for T in HXFER_TS:
        tt = t_tag(T)
        et = eta_tag(HXFER_ETA)
        out = HXFER_LOG / f"D{W}_eta{et}_t{tt}_R0p1.out"
        status = "pending"
        step = 0
        roc = None
        acc = None
        loss = None
        last = ""
        err = ""
        wd = None
        if out.is_file():
            text = out.read_text(errors="ignore")
            for m in re.finditer(r"global_step=(\d+)", text):
                step = max(step, int(m.group(1)))
            for m in re.finditer(r"\[train\] step=(\d+)", text):
                step = max(step, int(m.group(1)))
            for m in re.finditer(r"val_roc_auc=([0-9.]+)", text):
                roc = float(m.group(1))
            for m in re.finditer(r"val_acc=([0-9.]+)", text):
                acc = float(m.group(1))
            for m in re.finditer(r"train_loss=([0-9.]+)", text):
                loss = float(m.group(1))
            for m in re.finditer(r"weight_decay = ([0-9.eE+-]+)", text):
                wd = float(m.group(1))
            lines = [ln for ln in text.splitlines() if ln.strip()]
            last = lines[-1][:160] if lines else ""
            real_fail = (
                "Traceback" in text
                or "NCCL error" in text
                or "NCCL_ERROR" in text
                or re.search(r"nccl(?:Internal|Unhandled|System)Error", text, re.I) is not None
            )
            if real_fail and step < HXFER_MAX:
                err = "error"
                status = "failed"
            elif step >= HXFER_MAX:
                status = "done"
            elif step > 0 or "Initializing distributed" in text or "[attn-gamma]" in text:
                status = "running"
        if pq is not None and HXFER_ROOT.is_dir():
            # Require `-t{tt}-` (trailing dash) so t=1 does not match t=10 paths.
            for p in HXFER_ROOT.glob(
                f"*_{W}_{et}_t{tt}_*{HXFER_TAG}-t{tt}-*.parquet"
            ):
                if f"_{W}_{et}_t{tt}_" not in p.name:
                    continue
                if f"-t{tt}-" not in p.name:
                    continue
                try:
                    cols = ["global_step", "val_roc_auc", "val_acc"]
                    df = pq.read_table(p, columns=cols).to_pandas()
                    if len(df):
                        step = max(step, int(df["global_step"].max()))
                        sub = df.dropna(subset=["val_roc_auc"])
                        if len(sub):
                            row = sub.loc[sub["global_step"].idxmax()]
                            roc = float(row["val_roc_auc"])
                            if "val_acc" in row and row["val_acc"] == row["val_acc"]:
                                acc = float(row["val_acc"])
                        if step >= HXFER_MAX:
                            status = "done"
                except Exception:
                    pass
        t_argv = f"--t-epoch {int(T) if T == int(T) else T}"
        t_argv_g = f"--t-epoch {T:g}"
        # Require both width and an exact t-epoch token (avoid "--t-epoch 1" ⊂ "--t-epoch 10").
        def _has_t(p):
            return bool(re.search(rf"(?:^|\s){re.escape(t_argv)}(?:\s|$)", p)) or bool(
                re.search(rf"(?:^|\s){re.escape(t_argv_g)}(?:\s|$)", p)
            )
        live = any(
            _has_t(p) and "hparam_xfer" in p and f"--width {W}" in p for p in procs
        )
        # Live RUN marker: only trust master_d512 for W=512, older masters for W=256.
        if not live and status == "pending":
            width_master = ""
            if W == 512 and (HXFER_LOG / "master_d512.log").is_file():
                width_master = (HXFER_LOG / "master_d512.log").read_text(errors="ignore")
            elif W == 256:
                for mf in (
                    HXFER_LOG / "master_t5.log",
                    HXFER_LOG / "master_adaptive.log",
                    HXFER_LOG / "master_t1.log",
                ):
                    if mf.is_file():
                        width_master += mf.read_text(errors="ignore")
            run_hits = re.findall(r"RUN t=([0-9.]+)", width_master)
            if (
                run_hits
                and abs(float(run_hits[-1]) - T) < 1e-12
                and f"D={W}" in width_master
            ):
                live = True
        if live and status not in ("done", "failed"):
            status = "running"
        # Arm 1b intentionally paused at t=20.
        if (
            paused_t_sweep
            and W == 512
            and abs(T - 20.0) < 1e-12
            and status not in ("done",)
            and step < HXFER_MAX
        ):
            status = "paused"
        hxfer_cells.append(dict(
            task=ti, t_epoch=T, width=W, eta0=HXFER_ETA, status=status,
            step=step, max_steps=HXFER_MAX, val_roc_auc=roc, val_acc=acc,
            train_loss=loss, weight_decay=wd, last=last, err=err,
        ))
        ti += 1

# --- Arm 2: const-LR AdamW t + Adam @ D=256 ---
CONSTLR_MAX = 10000
CONSTLR_W = 256
CONSTLR_ETA = 0.25
CONSTLR_TAG = "hxfer-constlr-b64-g8-s10000-k12-R0p1"
CONSTLR_ROOT_ADAMW = Path.home() / "runs/hparam_xfer/jetclass_output/adamw/sweep_lr"
CONSTLR_ROOT_ADAM = Path.home() / "runs/hparam_xfer/jetclass_output/adam/sweep_lr"
constlr_cells = []
constlr_spec = [("adamw", 1.0), ("adamw", 5.0), ("adamw", 10.0), ("adam", None)]
ci = 0
for OPT, T in constlr_spec:
    et = eta_tag(CONSTLR_ETA)
    if OPT == "adamw":
        tt = t_tag(T)
        out = HXFER_LOG / f"D{CONSTLR_W}_eta{et}_constlr_adamw_t{tt}_R0p1.out"
        tag_pat = f"{CONSTLR_TAG}-adamw-t{tt}"
        root = CONSTLR_ROOT_ADAMW
    else:
        tt = None
        out = HXFER_LOG / f"D{CONSTLR_W}_eta{et}_constlr_adam_R0p1.out"
        tag_pat = f"{CONSTLR_TAG}-adam"
        root = CONSTLR_ROOT_ADAM
    status = "pending"
    step = 0
    roc = None
    acc = None
    loss = None
    last = ""
    err = ""
    wd = None
    if out.is_file():
        text = out.read_text(errors="ignore")
        for m in re.finditer(r"global_step=(\d+)", text):
            step = max(step, int(m.group(1)))
        for m in re.finditer(r"\[train\] step=(\d+)", text):
            step = max(step, int(m.group(1)))
        for m in re.finditer(r"val_roc_auc=([0-9.]+)", text):
            roc = float(m.group(1))
        for m in re.finditer(r"val_acc=([0-9.]+)", text):
            acc = float(m.group(1))
        for m in re.finditer(r"train_loss=([0-9.]+)", text):
            loss = float(m.group(1))
        for m in re.finditer(r"weight_decay = ([0-9.eE+-]+)", text):
            wd = float(m.group(1))
        if OPT == "adam":
            wd = 0.0
        lines = [ln for ln in text.splitlines() if ln.strip()]
        last = lines[-1][:160] if lines else ""
        real_fail = (
            "Traceback" in text
            or "NCCL error" in text
            or "NCCL_ERROR" in text
            or re.search(r"nccl(?:Internal|Unhandled|System)Error", text, re.I) is not None
        )
        if real_fail and step < CONSTLR_MAX:
            err = "error"
            status = "failed"
        elif step >= CONSTLR_MAX:
            status = "done"
        elif step > 0 or "Initializing distributed" in text or "[attn-gamma]" in text:
            status = "running"
    if pq is not None and root.is_dir():
        for p in root.glob(f"*_{CONSTLR_W}_{et}_*{tag_pat}*.parquet"):
            try:
                cols = ["global_step", "val_roc_auc", "val_acc"]
                df = pq.read_table(p, columns=cols).to_pandas()
                if len(df):
                    step = max(step, int(df["global_step"].max()))
                    sub = df.dropna(subset=["val_roc_auc"])
                    if len(sub):
                        row = sub.loc[sub["global_step"].idxmax()]
                        roc = float(row["val_roc_auc"])
                        if "val_acc" in row and row["val_acc"] == row["val_acc"]:
                            acc = float(row["val_acc"])
                    if step >= CONSTLR_MAX:
                        status = "done"
            except Exception:
                pass
    # Live process detection
    live = False
    for p in procs:
        if f"--width {CONSTLR_W}" not in p:
            continue
        if not re.search(r"--scheduler\s+none", p):
            continue
        if OPT == "adamw":
            if not re.search(r"--optimizer\s+adamw", p):
                continue
            t_argv = f"--t-epoch {int(T) if T == int(T) else T}"
            if re.search(rf"(?:^|\s){re.escape(t_argv)}(?:\s|$)", p):
                live = True
                break
        else:
            if re.search(r"--optimizer\s+adam(?:\s|$)", p) and not re.search(
                r"--optimizer\s+adamw", p
            ):
                live = True
                break
    # Master RUN line (sequential queue)
    if not live and status == "pending":
        mf = HXFER_LOG / "master_constlr_d256.log"
        if mf.is_file():
            mt = mf.read_text(errors="ignore")
            if OPT == "adamw":
                key_run = f"RUN adamw t={T:g}"
                key_done = f"DONE adamw t={T:g}"
            else:
                key_run = "RUN adam "
                key_done = "DONE adam "
            last_run = mt.rfind(key_run)
            last_done = mt.rfind(key_done)
            if last_run >= 0 and last_run > last_done:
                live = True
    if live and status not in ("done", "failed"):
        status = "running"
    constlr_cells.append(dict(
        task=ci, optimizer=OPT, t_epoch=T, width=CONSTLR_W, eta0=CONSTLR_ETA,
        status=status, step=step, max_steps=CONSTLR_MAX, val_roc_auc=roc,
        val_acc=acc, train_loss=loss, weight_decay=wd, last=last, err=err,
    ))
    ci += 1

paused_arm2 = (HXFER_LOG / "PAUSED_ARM2_D256_CONSTLR").is_file()
paused_arm2_note = ""
if paused_arm2:
    paused_arm2_note = (HXFER_LOG / "PAUSED_ARM2_D256_CONSTLR").read_text(errors="ignore").strip()
    for c in constlr_cells:
        if c["status"] == "done":
            continue
        # Mid-run kill (t=10) or never-started Adam → paused
        if c["status"] in ("running", "pending"):
            c["status"] = "paused"

# --- Arm 2c: const-LR Adam @ D=1024 (Lambda 8×A100) ---
CONSTLR_D1024_W = 1024
constlr_d1024 = None
out1024 = HXFER_LOG / f"D{CONSTLR_D1024_W}_eta{eta_tag(CONSTLR_ETA)}_constlr_adam_R0p1.out"
tag1024 = f"{CONSTLR_TAG}-adam"
status = "pending"
step = 0
roc = None
acc = None
loss = None
last = ""
err = ""
wd = 0.0
if out1024.is_file():
    text = out1024.read_text(errors="ignore")
    for m in re.finditer(r"global_step=(\d+)", text):
        step = max(step, int(m.group(1)))
    for m in re.finditer(r"\[train\] step=(\d+)", text):
        step = max(step, int(m.group(1)))
    for m in re.finditer(r"val_roc_auc=([0-9.]+)", text):
        roc = float(m.group(1))
    for m in re.finditer(r"val_acc=([0-9.]+)", text):
        acc = float(m.group(1))
    for m in re.finditer(r"train_loss=([0-9.]+)", text):
        loss = float(m.group(1))
    lines = [ln for ln in text.splitlines() if ln.strip()]
    last = lines[-1][:160] if lines else ""
    real_fail = (
        "Traceback" in text
        or "NCCL error" in text
        or "NCCL_ERROR" in text
        or "CUDA out of memory" in text
        or re.search(r"nccl(?:Internal|Unhandled|System)Error", text, re.I) is not None
    )
    if real_fail and step < CONSTLR_MAX:
        err = "error"
        status = "failed"
    elif step >= CONSTLR_MAX:
        status = "done"
    elif step > 0 or "Initializing distributed" in text or "[attn-gamma]" in text:
        status = "running"
if pq is not None:
    root = CONSTLR_ROOT_ADAM
    if root.is_dir():
        for p in root.glob(f"*_{CONSTLR_D1024_W}_{eta_tag(CONSTLR_ETA)}_*{tag1024}*.parquet"):
            try:
                cols = ["global_step", "val_roc_auc", "val_acc"]
                df = pq.read_table(p, columns=cols).to_pandas()
                if len(df):
                    step = max(step, int(df["global_step"].max()))
                    sub = df.dropna(subset=["val_roc_auc"])
                    if len(sub):
                        row = sub.loc[sub["global_step"].idxmax()]
                        roc = float(row["val_roc_auc"])
                        if "val_acc" in row and row["val_acc"] == row["val_acc"]:
                            acc = float(row["val_acc"])
                    if step >= CONSTLR_MAX:
                        status = "done"
            except Exception:
                pass
live1024 = False
for p in procs:
    if f"--width {CONSTLR_D1024_W}" not in p:
        continue
    if re.search(r"--scheduler\s+none", p) and re.search(r"--optimizer\s+adam(?:\s|$)", p) and not re.search(r"--optimizer\s+adamw", p):
        live1024 = True
        break
if not live1024 and status == "pending":
    mf = HXFER_LOG / "master_constlr_d1024.log"
    if mf.is_file():
        mt = mf.read_text(errors="ignore")
        if mt.rfind("RUN adam ") >= 0 and mt.rfind("RUN adam ") > mt.rfind("DONE adam "):
            live1024 = True
if live1024 and status not in ("done", "failed"):
    status = "running"
constlr_d1024 = dict(
    task=0, optimizer="adam", t_epoch=None, width=CONSTLR_D1024_W, eta0=CONSTLR_ETA,
    status=status, step=step, max_steps=CONSTLR_MAX, val_roc_auc=roc,
    val_acc=acc, train_loss=loss, weight_decay=wd, last=last, err=err,
)

print(json.dumps({
    "running_proc": running,
    "master_tail": master_tail,
    "hxfer_master": hxfer_master,
    "paused_t_sweep": paused_t_sweep,
    "paused_note": paused_note,
    "paused_arm2": paused_arm2,
    "paused_arm2_note": paused_arm2_note,
    "cells": cells,
    "hxfer_cells": hxfer_cells,
    "constlr_cells": constlr_cells,
    "constlr_d1024": constlr_d1024,
}))
"""


def ensure_agent_sock() -> None:
    if os.environ.get("SSH_AUTH_SOCK"):
        try:
            r = subprocess.run(
                ["ssh-add", "-l"], capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                return
        except Exception:
            pass
    candidates: list[Path] = []
    for p in sorted(
        Path("/tmp").glob("ssh-*/agent.*"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    ):
        candidates.append(p)
    for p in sorted(
        Path("/tmp").glob("ssh-agent-*.sock"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    ):
        candidates.append(p)
    for s in candidates:
        env = os.environ.copy()
        env["SSH_AUTH_SOCK"] = str(s)
        try:
            r = subprocess.run(
                ["ssh-add", "-l"], env=env, capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                os.environ["SSH_AUTH_SOCK"] = str(s)
                return
        except Exception:
            pass


def ssh_probe() -> dict:
    ensure_agent_sock()
    remote_script = "python3 <<'PY'\n" + REMOTE_PY + "\nPY\n"
    cmd = [
        "ssh",
        "-F",
        "/dev/null",
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        f"IdentityFile={IDENTITY}",
        "-o",
        "ConnectTimeout=20",
        HOST,
        "bash",
        "-s",
    ]
    r = subprocess.run(
        cmd, input=remote_script, capture_output=True, text=True, timeout=90
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or r.stdout.strip() or f"ssh rc={r.returncode}")
    lines = [ln for ln in r.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"no JSON from remote: {r.stdout[-800:]}\nerr={r.stderr[-400:]}")
    return json.loads(lines[-1])


def _eta_tag(eta0: float) -> str:
    s = f"{eta0:g}".replace(".", "p")
    return s


def _t_tag(t: float) -> str:
    return f"{t:g}".replace(".", "p")


def _squeue_ailab_states(
    jobid: str | None, job_name: str = "jc-hxfer-c384"
) -> dict[int, str]:
    """Map array task index → RUNNING|PENDING|… for a named ailab array."""
    out: dict[int, str] = {}
    try:
        r = subprocess.run(
            [
                "squeue",
                "-u",
                os.environ.get("USER", "jdezoort"),
                "-n",
                job_name,
                "-h",
                "-o",
                "%i %T",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception:
        return out
    if r.returncode != 0:
        return out
    for ln in r.stdout.splitlines():
        parts = ln.split()
        if len(parts) < 2:
            continue
        jid, state = parts[0], parts[1]
        if "_" not in jid:
            continue
        base, task_s = jid.rsplit("_", 1)
        if jobid and base != jobid:
            continue
        try:
            out[int(task_s)] = state
        except ValueError:
            continue
    return out


def _latest_ailab_jobid(
    job_name: str = "jc-hxfer-c384",
    log_glob: str = "jc_hxfer_c384_ailab_*_*.out",
    log_re: str = r"jc_hxfer_c384_ailab_(\d+)_(\d+)\.out$",
) -> str | None:
    """Return array parent job id (e.g. 14824394) for a named ailab array."""
    try:
        r = subprocess.run(
            [
                "squeue",
                "-u",
                os.environ.get("USER", "jdezoort"),
                "-n",
                job_name,
                "-h",
                "-o",
                "%i",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        if r.returncode == 0:
            bases: set[str] = set()
            for ln in r.stdout.splitlines():
                jid = ln.strip()
                if "_" in jid:
                    bases.add(jid.rsplit("_", 1)[0])
            if bases:
                return sorted(bases)[-1]
    except Exception:
        pass
    if not DELLA_LOGS.is_dir():
        return None
    best: tuple[float, str] | None = None
    for p in DELLA_LOGS.glob(log_glob):
        m = re.match(log_re, p.name)
        if not m:
            continue
        mtime = p.stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, m.group(1))
    return best[1] if best else None


def _parse_train_log(text: str, max_steps: int) -> dict:
    step = 0
    roc = None
    acc = None
    loss = None
    wd = None
    for m in re.finditer(r"global_step=(\d+)", text):
        step = max(step, int(m.group(1)))
    for m in re.finditer(r"\[train\] step=(\d+)", text):
        step = max(step, int(m.group(1)))
    for m in re.finditer(r"val_roc_auc=([0-9.]+)", text):
        roc = float(m.group(1))
    for m in re.finditer(r"val_acc=([0-9.]+)", text):
        acc = float(m.group(1))
    for m in re.finditer(r"train_loss=([0-9.]+)", text):
        loss = float(m.group(1))
    for m in re.finditer(r"weight_decay = ([0-9.eE+-]+)", text):
        wd = float(m.group(1))
    lines = [ln for ln in text.splitlines() if ln.strip()]
    last = lines[-1][:160] if lines else ""
    real_fail = (
        "Traceback" in text
        or "NCCL error" in text
        or "NCCL_ERROR" in text
        or re.search(r"nccl(?:Internal|Unhandled|System)Error", text, re.I) is not None
    )
    status = "pending"
    err = ""
    if real_fail and step < max_steps:
        err = "error"
        status = "failed"
    elif step >= max_steps:
        status = "done"
    elif step > 0 or "Initializing distributed" in text or "[attn-gamma]" in text:
        status = "running"
    return dict(
        step=step,
        val_roc_auc=roc,
        val_acc=acc,
        train_loss=loss,
        weight_decay=wd,
        last=last,
        err=err,
        status=status,
    )


def probe_della_constlr_d384() -> list[dict]:
    """Local probe of Della ailab Arm 2b (D=384 const-LR array)."""
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        pq = None  # type: ignore

    jobid = _latest_ailab_jobid()
    sq = _squeue_ailab_states(jobid)
    et = _eta_tag(CONSTLR_ETA0)
    cells: list[dict] = []
    for task, opt, t in CONSTLR_D384_CELLS:
        status = "pending"
        step = 0
        roc = None
        acc = None
        loss = None
        last = ""
        err = ""
        wd = 0.0 if opt == "adam" else None
        if opt == "adamw":
            tt = _t_tag(float(t))
            tag_pat = f"{CONSTLR_D384_TAG}-adamw-t{tt}"
            root = DELLA_ROOT / "jetclass_output" / "adamw" / "sweep_lr"
        else:
            tag_pat = f"{CONSTLR_D384_TAG}-adam"
            root = DELLA_ROOT / "jetclass_output" / "adam" / "sweep_lr"

        out = None
        if jobid:
            cand = DELLA_LOGS / f"jc_hxfer_c384_ailab_{jobid}_{task}.out"
            if cand.is_file():
                out = cand
        if out is None and DELLA_LOGS.is_dir():
            matches = sorted(
                DELLA_LOGS.glob(f"jc_hxfer_c384_ailab_*_{task}.out"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if matches:
                out = matches[0]

        if out is not None and out.is_file():
            text = out.read_text(errors="ignore")
            for m in re.finditer(r"global_step=(\d+)", text):
                step = max(step, int(m.group(1)))
            for m in re.finditer(r"\[train\] step=(\d+)", text):
                step = max(step, int(m.group(1)))
            for m in re.finditer(r"val_roc_auc=([0-9.]+)", text):
                roc = float(m.group(1))
            for m in re.finditer(r"val_acc=([0-9.]+)", text):
                acc = float(m.group(1))
            for m in re.finditer(r"train_loss=([0-9.]+)", text):
                loss = float(m.group(1))
            for m in re.finditer(r"weight_decay = ([0-9.eE+-]+)", text):
                wd = float(m.group(1))
            if opt == "adam":
                wd = 0.0
            lines = [ln for ln in text.splitlines() if ln.strip()]
            last = lines[-1][:160] if lines else ""
            real_fail = (
                "Traceback" in text
                or "NCCL error" in text
                or "NCCL_ERROR" in text
                or re.search(r"nccl(?:Internal|Unhandled|System)Error", text, re.I)
                is not None
            )
            if real_fail and step < CONSTLR_MAX_STEPS:
                err = "error"
                status = "failed"
            elif step >= CONSTLR_MAX_STEPS:
                status = "done"
            elif step > 0 or "Initializing distributed" in text or "[attn-gamma]" in text:
                status = "running"

        if pq is not None and root.is_dir():
            for p in root.glob(f"*_{CONSTLR_D384_WIDTH}_{et}_*{tag_pat}*.parquet"):
                try:
                    cols = ["global_step", "val_roc_auc", "val_acc"]
                    df = pq.read_table(p, columns=cols).to_pandas()
                    if len(df):
                        step = max(step, int(df["global_step"].max()))
                        sub = df.dropna(subset=["val_roc_auc"])
                        if len(sub):
                            row = sub.loc[sub["global_step"].idxmax()]
                            roc = float(row["val_roc_auc"])
                            if "val_acc" in row and row["val_acc"] == row["val_acc"]:
                                acc = float(row["val_acc"])
                        if step >= CONSTLR_MAX_STEPS:
                            status = "done"
                except Exception:
                    pass

        sq_state = sq.get(task)
        if sq_state == "RUNNING" and status not in ("done", "failed"):
            status = "running"
        elif sq_state == "PENDING" and status == "pending":
            status = "pending"
        elif sq_state is None and status == "running" and step >= CONSTLR_MAX_STEPS:
            status = "done"

        cells.append(
            dict(
                task=task,
                optimizer=opt,
                t_epoch=t,
                width=CONSTLR_D384_WIDTH,
                eta0=CONSTLR_ETA0,
                status=status,
                step=step,
                max_steps=CONSTLR_MAX_STEPS,
                val_roc_auc=roc,
                val_acc=acc,
                train_loss=loss,
                weight_decay=wd,
                last=last,
                err=err,
                jobid=jobid,
                slurm_state=sq_state,
            )
        )
    return cells


def probe_della_adam_eta() -> list[dict]:
    """Local probe of Della ailab Arm 3 (Adam η₀×D fill-in)."""
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        pq = None  # type: ignore

    jobid = _latest_ailab_jobid(
        ADAM_ETA_JOB_NAME,
        ADAM_ETA_LOG_GLOB,
        r"jc_hxfer_adam_eta_ailab_(\d+)_(\d+)\.out$",
    )
    # Aggregate squeue across all jc-hxfer-aeta arrays (main 0–7 + fill-in 8).
    sq = _squeue_ailab_states(None, ADAM_ETA_JOB_NAME)
    root = DELLA_ROOT / "jetclass_output" / "adam" / "sweep_lr"
    cells: list[dict] = []
    for task, width, eta0 in ADAM_ETA_CELLS:
        et = _eta_tag(eta0)
        status = "pending"
        step = 0
        roc = None
        acc = None
        loss = None
        last = ""
        err = ""
        wd = 0.0

        out = None
        if jobid:
            cand = DELLA_LOGS / f"jc_hxfer_adam_eta_ailab_{jobid}_{task}.out"
            if cand.is_file():
                out = cand
        if out is None and DELLA_LOGS.is_dir():
            matches = sorted(
                DELLA_LOGS.glob(f"jc_hxfer_adam_eta_ailab_*_{task}.out"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if matches:
                out = matches[0]

        if out is not None and out.is_file():
            parsed = _parse_train_log(out.read_text(errors="ignore"), CONSTLR_MAX_STEPS)
            step = parsed["step"]
            roc = parsed["val_roc_auc"]
            acc = parsed["val_acc"]
            loss = parsed["train_loss"]
            if parsed["weight_decay"] is not None:
                wd = parsed["weight_decay"]
            last = parsed["last"]
            err = parsed["err"]
            status = parsed["status"]

        if pq is not None and root.is_dir():
            for p in root.glob(f"*_{width}_{et}_*{ADAM_ETA_TAG}*.parquet"):
                try:
                    cols = ["global_step", "val_roc_auc", "val_acc"]
                    df = pq.read_table(p, columns=cols).to_pandas()
                    if len(df):
                        step = max(step, int(df["global_step"].max()))
                        sub = df.dropna(subset=["val_roc_auc"])
                        if len(sub):
                            row = sub.loc[sub["global_step"].idxmax()]
                            roc = float(row["val_roc_auc"])
                            if "val_acc" in row and row["val_acc"] == row["val_acc"]:
                                acc = float(row["val_acc"])
                        if step >= CONSTLR_MAX_STEPS:
                            status = "done"
                except Exception:
                    pass

        sq_state = sq.get(task)
        if sq_state == "RUNNING" and status not in ("done", "failed"):
            status = "running"
        elif sq_state == "PENDING" and status == "pending":
            status = "pending"
        elif sq_state is None and status == "running" and step >= CONSTLR_MAX_STEPS:
            status = "done"

        cells.append(
            dict(
                task=task,
                optimizer="adam",
                t_epoch=None,
                width=width,
                eta0=eta0,
                status=status,
                step=step,
                max_steps=CONSTLR_MAX_STEPS,
                val_roc_auc=roc,
                val_acc=acc,
                train_loss=loss,
                weight_decay=wd,
                last=last,
                err=err,
                jobid=jobid,
                slurm_state=sq_state,
            )
        )
    return cells


def probe_della_edge_gt() -> list[dict]:
    """Local probe of Della gpu-test panel-(c) edge η₀ fills (D=128, D=384)."""
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception:
        pq = None  # type: ignore

    root = DELLA_ROOT / "jetclass_output" / "adam" / "sweep_lr"
    cells: list[dict] = []
    for width, job_name, log_prefix in EDGE_GT_SWEEPS:
        jobid = _latest_ailab_jobid(
            job_name,
            f"{log_prefix}_*_*.out",
            rf"{re.escape(log_prefix)}_(\d+)_(\d+)\.out$",
        )
        sq = _squeue_ailab_states(None, job_name)
        for task, eta0 in enumerate(EDGE_GT_ETAS):
            et = _eta_tag(eta0)
            status = "pending"
            step = 0
            roc = None
            acc = None
            loss = None
            last = ""
            err = ""

            out = None
            if jobid:
                cand = DELLA_LOGS / f"{log_prefix}_{jobid}_{task}.out"
                if cand.is_file():
                    out = cand
            if out is None and DELLA_LOGS.is_dir():
                matches = sorted(
                    DELLA_LOGS.glob(f"{log_prefix}_*_{task}.out"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if matches:
                    out = matches[0]

            if out is not None and out.is_file():
                parsed = _parse_train_log(
                    out.read_text(errors="ignore"), EDGE_GT_MAX_STEPS
                )
                step = parsed["step"]
                roc = parsed["val_roc_auc"]
                acc = parsed["val_acc"]
                loss = parsed["train_loss"]
                last = parsed["last"]
                err = parsed["err"]
                status = parsed["status"]

            if pq is not None and root.is_dir():
                for p in root.glob(f"*_{width}_{et}_*{EDGE_GT_TAG}*.parquet"):
                    try:
                        cols = ["global_step", "val_roc_auc", "val_acc"]
                        df = pq.read_table(p, columns=cols).to_pandas()
                        if len(df):
                            step = max(step, int(df["global_step"].max()))
                            sub = df.dropna(subset=["val_roc_auc"])
                            if len(sub):
                                row = sub.loc[sub["global_step"].idxmax()]
                                roc = float(row["val_roc_auc"])
                                if "val_acc" in row and row["val_acc"] == row["val_acc"]:
                                    acc = float(row["val_acc"])
                            if step >= EDGE_GT_MAX_STEPS:
                                status = "done"
                    except Exception:
                        pass

            sq_state = sq.get(task)
            if sq_state == "RUNNING" and status not in ("done", "failed"):
                status = "running"
            elif sq_state == "PENDING" and status == "pending":
                status = "pending"
            elif (
                sq_state is None
                and status == "running"
                and step >= EDGE_GT_MAX_STEPS
            ):
                status = "done"

            cells.append(
                dict(
                    task=task,
                    width=width,
                    eta0=eta0,
                    status=status,
                    step=step,
                    max_steps=EDGE_GT_MAX_STEPS,
                    val_roc_auc=roc,
                    val_acc=acc,
                    train_loss=loss,
                    last=last,
                    err=err,
                    jobid=jobid,
                    slurm_state=sq_state,
                )
            )
    return cells


def probe_della_edge_d128() -> list[dict]:
    """Back-compat: D=128 subset of gpu-test edge fills."""
    return [c for c in probe_della_edge_gt() if c["width"] == 128]


def write_csv(path: Path, cells: list[dict], cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(cols)]
    for c in cells:
        lines.append(
            ",".join(str(c.get(k) if c.get(k) is not None else "") for k in cols)
        )
    path.write_text("\n".join(lines) + "\n")


def patch_fig_md(
    cells: list[dict],
    hxfer: list[dict],
    constlr: list[dict],
    constlr_d384: list[dict],
    adam_eta: list[dict],
    constlr_d1024: dict | None,
    edge_gt: list[dict],
    updated: str,
) -> None:
    if not FIG_MD.is_file():
        return
    done = [c for c in cells if c["status"] == "done"]
    rows = [
        "| D | η₀ | status | step | val ROC |",
        "|---:|---:|---|---:|---:|",
    ]
    for c in cells:
        roc = f"{c['val_roc_auc']:.4f}" if c["val_roc_auc"] is not None else "—"
        rows.append(
            f"| {c['width']} | {c['eta0']} | {c['status']} | {c['step']}/{c['max_steps']} | {roc} |"
        )
    hx_rows = [
        "| D | t | status | step | val ROC | WD |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for c in hxfer:
        roc = f"{c['val_roc_auc']:.4f}" if c["val_roc_auc"] is not None else "—"
        wd = f"{c['weight_decay']:.6g}" if c.get("weight_decay") is not None else "—"
        hx_rows.append(
            f"| {c['width']} | {c['t_epoch']:g} | {c['status']} | {c['step']}/{c['max_steps']} | {roc} | {wd} |"
        )
    cl_rows = [
        "| opt | t | status | step | val ROC | WD |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for c in constlr:
        roc = f"{c['val_roc_auc']:.4f}" if c["val_roc_auc"] is not None else "—"
        wd = f"{c['weight_decay']:.6g}" if c.get("weight_decay") is not None else "—"
        t_s = "—" if c.get("t_epoch") is None else f"{c['t_epoch']:g}"
        cl_rows.append(
            f"| {c['optimizer']} | {t_s} | {c['status']} | {c['step']}/{c['max_steps']} | {roc} | {wd} |"
        )
    cl384_rows = [
        "| opt | t | status | step | val ROC | WD |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for c in constlr_d384:
        roc = f"{c['val_roc_auc']:.4f}" if c["val_roc_auc"] is not None else "—"
        wd = f"{c['weight_decay']:.6g}" if c.get("weight_decay") is not None else "—"
        t_s = "—" if c.get("t_epoch") is None else f"{c['t_epoch']:g}"
        cl384_rows.append(
            f"| {c['optimizer']} | {t_s} | {c['status']} | {c['step']}/{c['max_steps']} | {roc} | {wd} |"
        )
    ae_rows = [
        "| D | η₀ | status | step | val ROC |",
        "|---:|---:|---|---:|---:|",
    ]
    for c in adam_eta:
        roc = f"{c['val_roc_auc']:.4f}" if c["val_roc_auc"] is not None else "—"
        ae_rows.append(
            f"| {c['width']} | {c['eta0']} | {c['status']} | {c['step']}/{c['max_steps']} | {roc} |"
        )
    e_gt_rows = [
        "| D | η₀ | status | step | val ROC |",
        "|---:|---:|---|---:|---:|",
    ]
    for c in edge_gt:
        roc = f"{c['val_roc_auc']:.4f}" if c["val_roc_auc"] is not None else "—"
        e_gt_rows.append(
            f"| {c['width']} | {c['eta0']} | {c['status']} | {c['step']}/{c['max_steps']} | {roc} |"
        )
    cl_done = sum(1 for c in constlr if c["status"] == "done")
    cl384_done = sum(1 for c in constlr_d384 if c["status"] == "done")
    ae_done = sum(1 for c in adam_eta if c["status"] == "done")
    e_gt_done = sum(1 for c in edge_gt if c["status"] == "done")
    d1024_line = "—"
    if constlr_d1024:
        roc = (
            f"{constlr_d1024['val_roc_auc']:.4f}"
            if constlr_d1024.get("val_roc_auc") is not None
            else "—"
        )
        d1024_line = (
            f"{constlr_d1024['status']} step {constlr_d1024['step']}/"
            f"{constlr_d1024['max_steps']} ROC {roc}"
        )
    block = (
        f"\n<!-- EDGE_ETA_LIVE_TABLE -->\n"
        f"_Live edge η×D grid (updated {updated}):_\n\n"
        + "\n".join(rows)
        + f"\n\nDone: **{len(done)}/15**. Pull: `scans/jets/pull_edge_eta_lambda.sh`.\n\n"
        f"_Hparam-xfer Arm 1/1b t-sweep (η₀=0.25, D∈{{256,512}}, 12k cutoff) — **PAUSED**; "
        f"do not claim WD transfer (see `fig_hparam_transfer_proof.md`):_\n\n"
        + "\n".join(hx_rows)
        + f"\n\n_Arm 2 const-LR @ D=256 (10k, `scheduler=none`) — live:_ "
        f"**{cl_done}/{len(constlr) or 4}** done\n\n"
        + "\n".join(cl_rows)
        + f"\n\n_Arm 2c Adam const-LR @ D=1024 Lambda — {d1024_line}_\n\n"
        + f"\n\n_Arm 2b const-LR @ D=384 ailab (B=128×4, 10k) — live:_ "
        f"**{cl384_done}/{len(constlr_d384) or 4}** done\n\n"
        + "\n".join(cl384_rows)
        + f"\n\n_Arm 3 Adam η₀×D ailab fill-in (10k, panel a) — live:_ "
        f"**{ae_done}/{len(adam_eta) or 8}** done\n\n"
        + "\n".join(ae_rows)
        + f"\n\n_Panel (c) D=384 edge η₀ gpu-test (B=96×4, 4k, k=12 R=0.1; D=128 wiped) — live:_ "
        f"**{e_gt_done}/{len(edge_gt) or 5}** done\n\n"
        + "\n".join(e_gt_rows)
        + "\n"
        f"<!-- /EDGE_ETA_LIVE_TABLE -->\n"
    )
    text = FIG_MD.read_text()
    if "<!-- EDGE_ETA_LIVE_TABLE -->" in text:
        text = re.sub(
            r"<!-- EDGE_ETA_LIVE_TABLE -->.*?<!-- /EDGE_ETA_LIVE_TABLE -->\n?",
            block,
            text,
            flags=re.S,
        )
    else:
        if "## Changelog" in text:
            text = text.replace("## Changelog", block + "\n## Changelog")
        else:
            text += "\n" + block
    text = re.sub(
        r"\| Edge @ k★=12, R★=0\.1 \|.*?\|",
        f"| Edge @ k★=12, R★=0.1 | **{len(done)}/15 done** on Lambda "
        f"(live table below; canvas: edge-eta-lambda-live) |",
        text,
        count=1,
    )
    FIG_MD.write_text(text)


def _render_canvas(payload: dict, updated: str) -> str:
    cells = payload["cells"]
    hxfer = payload.get("hxfer_cells") or []
    constlr = payload.get("constlr_cells") or []
    constlr_d384 = payload.get("constlr_d384_cells") or []
    adam_eta = payload.get("adam_eta_cells") or []
    constlr_d1024 = payload.get("constlr_d1024")
    edge_gt = payload.get("edge_gt_cells") or payload.get("edge_d128_cells") or []
    n_done = sum(1 for c in cells if c["status"] == "done")
    n_run = sum(1 for c in cells if c["status"] == "running")
    n_fail = sum(1 for c in cells if c["status"] == "failed")
    n_pend = sum(1 for c in cells if c["status"] == "pending")
    active = next((c for c in cells if c["status"] == "running"), None)
    hx_done = sum(1 for c in hxfer if c["status"] == "done")
    hx_run = sum(1 for c in hxfer if c["status"] == "running")
    hx_paused = sum(1 for c in hxfer if c["status"] == "paused")
    hx_active = next((c for c in hxfer if c["status"] == "running"), None)
    cl_done = sum(1 for c in constlr if c["status"] == "done")
    cl_run = sum(1 for c in constlr if c["status"] == "running")
    cl_paused = sum(1 for c in constlr if c["status"] == "paused")
    cl_active = next((c for c in constlr if c["status"] == "running"), None)
    cl384_done = sum(1 for c in constlr_d384 if c["status"] == "done")
    cl384_run = sum(1 for c in constlr_d384 if c["status"] == "running")
    cl384_pend = sum(1 for c in constlr_d384 if c["status"] == "pending")
    cl384_active = next((c for c in constlr_d384 if c["status"] == "running"), None)
    ae_done = sum(1 for c in adam_eta if c["status"] == "done")
    ae_run = sum(1 for c in adam_eta if c["status"] == "running")
    ae_pend = sum(1 for c in adam_eta if c["status"] == "pending")
    ae_active = next((c for c in adam_eta if c["status"] == "running"), None)
    e_gt_done = sum(1 for c in edge_gt if c["status"] == "done")
    e_gt_run = sum(1 for c in edge_gt if c["status"] == "running")
    e_gt_pend = sum(1 for c in edge_gt if c["status"] == "pending")
    e_gt_active = next((c for c in edge_gt if c["status"] == "running"), None)
    paused = bool(payload.get("paused_t_sweep"))
    paused_arm2 = bool(payload.get("paused_arm2"))
    d1024_run = 1 if constlr_d1024 and constlr_d1024.get("status") == "running" else 0
    data = "\n".join(
        [
            f'const UPDATED = "{updated}";',
            f"const MAX_STEPS = {MAX_STEPS};",
            f"const TAG = {json.dumps(TAG)};",
            f"const N_DONE = {n_done};",
            f"const N_RUN = {n_run};",
            f"const N_FAIL = {n_fail};",
            f"const N_PEND = {n_pend};",
            f"const ACTIVE = {json.dumps(active)};",
            f"const CELLS: Cell[] = {json.dumps(cells, indent=2)};",
            f"const HXFER_MAX_STEPS = {HXFER_MAX_STEPS};",
            f"const HXFER_TAG = {json.dumps(HXFER_TAG)};",
            f"const HXFER_N_DONE = {hx_done};",
            f"const HXFER_N_RUN = {hx_run};",
            f"const HXFER_N_PAUSED = {hx_paused};",
            f"const HXFER_N_TOTAL = {len(hxfer)};",
            f"const HXFER_ACTIVE = {json.dumps(hx_active)};",
            f"const HXFER_CELLS: HxferCell[] = {json.dumps(hxfer, indent=2)};",
            f"const HXFER_PAUSED = {json.dumps(paused)};",
            f"const CONSTLR_MAX_STEPS = {CONSTLR_MAX_STEPS};",
            f"const CONSTLR_TAG = {json.dumps(CONSTLR_TAG)};",
            f"const CONSTLR_N_DONE = {cl_done};",
            f"const CONSTLR_N_RUN = {cl_run};",
            f"const CONSTLR_N_PAUSED = {cl_paused};",
            f"const CONSTLR_N_TOTAL = {len(constlr)};",
            f"const CONSTLR_ACTIVE = {json.dumps(cl_active)};",
            f"const CONSTLR_CELLS: ConstlrCell[] = {json.dumps(constlr, indent=2)};",
            f"const CONSTLR_ARM2_PAUSED = {json.dumps(paused_arm2)};",
            f"const CONSTLR_D384_TAG = {json.dumps(CONSTLR_D384_TAG)};",
            f"const CONSTLR_D384_N_DONE = {cl384_done};",
            f"const CONSTLR_D384_N_RUN = {cl384_run};",
            f"const CONSTLR_D384_N_PEND = {cl384_pend};",
            f"const CONSTLR_D384_N_TOTAL = {len(constlr_d384)};",
            f"const CONSTLR_D384_ACTIVE = {json.dumps(cl384_active)};",
            f"const CONSTLR_D384_CELLS: ConstlrCell[] = {json.dumps(constlr_d384, indent=2)};",
            f"const ADAM_ETA_TAG = {json.dumps(ADAM_ETA_TAG)};",
            f"const ADAM_ETA_N_DONE = {ae_done};",
            f"const ADAM_ETA_N_RUN = {ae_run};",
            f"const ADAM_ETA_N_PEND = {ae_pend};",
            f"const ADAM_ETA_N_TOTAL = {len(adam_eta)};",
            f"const ADAM_ETA_ACTIVE = {json.dumps(ae_active)};",
            f"const ADAM_ETA_CELLS: Cell[] = {json.dumps(adam_eta, indent=2)};",
            f"const CONSTLR_D1024 = {json.dumps(constlr_d1024)};",
            f"const CONSTLR_D1024_N_RUN = {d1024_run};",
            f"const EDGE_GT_TAG = {json.dumps(EDGE_GT_TAG)};",
            f"const EDGE_GT_MAX_STEPS = {EDGE_GT_MAX_STEPS};",
            f"const EDGE_GT_N_DONE = {e_gt_done};",
            f"const EDGE_GT_N_RUN = {e_gt_run};",
            f"const EDGE_GT_N_PEND = {e_gt_pend};",
            f"const EDGE_GT_N_TOTAL = {len(edge_gt)};",
            f"const EDGE_GT_ACTIVE = {json.dumps(e_gt_active)};",
            f"const EDGE_GT_CELLS: Cell[] = {json.dumps(edge_gt, indent=2)};",
        ]
    )
    return (
        '''import {
  Callout,
  Divider,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useHostTheme,
} from "cursor/canvas";

type Cell = {
  task: number;
  width: number;
  eta0: number;
  status: string;
  step: number;
  max_steps: number;
  val_roc_auc: number | null;
  val_acc: number | null;
  train_loss: number | null;
  last: string;
  err: string;
};

type HxferCell = {
  task: number;
  t_epoch: number;
  width: number;
  eta0: number;
  status: string;
  step: number;
  max_steps: number;
  val_roc_auc: number | null;
  val_acc: number | null;
  train_loss: number | null;
  weight_decay: number | null;
  last: string;
  err: string;
};

type ConstlrCell = {
  task: number;
  optimizer: string;
  t_epoch: number | null;
  width: number;
  eta0: number;
  status: string;
  step: number;
  max_steps: number;
  val_roc_auc: number | null;
  val_acc: number | null;
  train_loss: number | null;
  weight_decay: number | null;
  last: string;
  err: string;
};

// BEGIN_AUTO_DATA
'''
        + data
        + """
// END_AUTO_DATA

function tone(status: string) {
  if (status === "done") return "success" as const;
  if (status === "running") return "info" as const;
  if (status === "failed") return "danger" as const;
  if (status === "paused") return "warning" as const;
  return "neutral" as const;
}

function fmt(x: number | null | undefined, digits = 4) {
  if (x == null || Number.isNaN(x)) return "—";
  return x.toFixed(digits);
}

function fmtWd(x: number | null | undefined) {
  if (x == null || Number.isNaN(x)) return "—";
  return x.toExponential(3);
}

function optLabel(c: ConstlrCell) {
  if (c.optimizer === "adam") return "Adam (WD=0)";
  return `AdamW t=${c.t_epoch}`;
}

export default function EdgeEtaLambdaLive() {
  const theme = useHostTheme();
  const done = CELLS.filter((c) => c.status === "done");
  const best = done
    .filter((c) => c.val_roc_auc != null)
    .slice()
    .sort((a, b) => (b.val_roc_auc ?? 0) - (a.val_roc_auc ?? 0))[0];

  return (
    <Stack gap={16} style={{ padding: 16 }}>
      <Stack gap={4}>
        <H1>Lambda + ailab live: const-LR WD check + Adam η₀×D</H1>
        <Text tone="secondary" size="small">
          Arm 2/2b/2c const-LR · Arm 3 Adam η₀×D · Arm 1/1b paused · Panel (c)
          incl. D=384 gpu-test · updated {UPDATED}
        </Text>
      </Stack>

      <Grid columns={4} gap={12}>
        <Stat
          value={
            CONSTLR_ACTIVE
              ? `${CONSTLR_ACTIVE.step}/${CONSTLR_ACTIVE.max_steps}`
              : CONSTLR_N_DONE >= CONSTLR_N_TOTAL && CONSTLR_N_TOTAL > 0
                ? `${CONSTLR_MAX_STEPS}/${CONSTLR_MAX_STEPS}`
                : "—"
          }
          label={
            CONSTLR_ACTIVE
              ? `D256 step (${optLabel(CONSTLR_ACTIVE)})`
              : "D256 const-LR step"
          }
          tone="info"
        />
        <Stat
          value={
            CONSTLR_D384_ACTIVE
              ? `${CONSTLR_D384_ACTIVE.step}/${CONSTLR_D384_ACTIVE.max_steps}`
              : CONSTLR_D384_N_DONE >= CONSTLR_D384_N_TOTAL &&
                  CONSTLR_D384_N_TOTAL > 0
                ? `${CONSTLR_MAX_STEPS}/${CONSTLR_MAX_STEPS}`
                : CONSTLR_D384_N_PEND > 0
                  ? `pend ${CONSTLR_D384_N_PEND}`
                  : "—"
          }
          label={
            CONSTLR_D384_ACTIVE
              ? `D384 step (${optLabel(CONSTLR_D384_ACTIVE)})`
              : "D384 ailab step"
          }
          tone="info"
        />
        <Stat
          value={`${CONSTLR_N_DONE + CONSTLR_D384_N_DONE}/${(CONSTLR_N_TOTAL || 4) + (CONSTLR_D384_N_TOTAL || 4)}`}
          label="const-LR done"
        />
        <Stat
          value={String(
            N_RUN +
              HXFER_N_RUN +
              CONSTLR_N_RUN +
              CONSTLR_D384_N_RUN +
              ADAM_ETA_N_RUN +
              CONSTLR_D1024_N_RUN +
              EDGE_GT_N_RUN
          )}
          label="jobs running"
          tone="info"
        />
      </Grid>

      {CONSTLR_ARM2_PAUSED ? (
        <Callout tone="warning" title="Arm 2 D=256 paused">
          Lambda compute reallocated to D=1024 Adam. AdamW t=1/5 done; t=10
          paused mid-run; Adam D=256 skipped.
        </Callout>
      ) : CONSTLR_ACTIVE ? (
        <Callout tone="info" title="Active Arm 2 (Lambda D=256)">
          {optLabel(CONSTLR_ACTIVE)} · D={CONSTLR_ACTIVE.width} · η₀=
          {CONSTLR_ACTIVE.eta0} · step {CONSTLR_ACTIVE.step}/
          {CONSTLR_ACTIVE.max_steps}
          {CONSTLR_ACTIVE.val_roc_auc != null
            ? ` · val ROC ${fmt(CONSTLR_ACTIVE.val_roc_auc)}`
            : ""}
          {CONSTLR_ACTIVE.weight_decay != null
            ? ` · WD ${fmtWd(CONSTLR_ACTIVE.weight_decay)}`
            : ""}
        </Callout>
      ) : CONSTLR_N_DONE >= CONSTLR_N_TOTAL && CONSTLR_N_TOTAL > 0 ? (
        <Callout tone="success" title="Arm 2 complete">
          All D=256 const-LR cells finished at {CONSTLR_MAX_STEPS} steps.
        </Callout>
      ) : (
        <Callout tone="warning" title="Arm 2 starting">
          Waiting for const-LR AdamW t∈{"{1,5,10}"} + Adam @ D=256.
        </Callout>
      )}

      {CONSTLR_D1024 ? (
        <Callout
          tone={
            CONSTLR_D1024.status === "done"
              ? "success"
              : CONSTLR_D1024.status === "failed"
                ? "danger"
                : "info"
          }
          title="Arm 2c — D=1024 Adam (Lambda)"
        >
          Adam WD=0 · D={CONSTLR_D1024.width} · η₀={CONSTLR_D1024.eta0} · step{" "}
          {CONSTLR_D1024.step}/{CONSTLR_D1024.max_steps}
          {CONSTLR_D1024.val_roc_auc != null
            ? ` · val ROC ${fmt(CONSTLR_D1024.val_roc_auc)}`
            : ""}{" "}
          · B=64×8
        </Callout>
      ) : null}

      {CONSTLR_D384_ACTIVE ? (
        <Callout tone="info" title="Active Arm 2b (ailab D=384)">
          {optLabel(CONSTLR_D384_ACTIVE)} · D={CONSTLR_D384_ACTIVE.width} · η₀=
          {CONSTLR_D384_ACTIVE.eta0} · step {CONSTLR_D384_ACTIVE.step}/
          {CONSTLR_D384_ACTIVE.max_steps}
          {CONSTLR_D384_ACTIVE.val_roc_auc != null
            ? ` · val ROC ${fmt(CONSTLR_D384_ACTIVE.val_roc_auc)}`
            : ""}
          {CONSTLR_D384_ACTIVE.weight_decay != null
            ? ` · WD ${fmtWd(CONSTLR_D384_ACTIVE.weight_decay)}`
            : ""}
        </Callout>
      ) : CONSTLR_D384_N_DONE >= CONSTLR_D384_N_TOTAL &&
        CONSTLR_D384_N_TOTAL > 0 ? (
        <Callout tone="success" title="Arm 2b complete">
          All D=384 ailab cells finished at {CONSTLR_MAX_STEPS} steps.
        </Callout>
      ) : CONSTLR_D384_N_TOTAL > 0 ? (
        <Callout tone="warning" title="Arm 2b queued / starting">
          ailab array · {CONSTLR_D384_N_RUN} running · {CONSTLR_D384_N_PEND}{" "}
          pending · Adam + AdamW t∈{"{1,5,10}"} @ D=384
        </Callout>
      ) : null}

      {HXFER_PAUSED ? (
        <Callout tone="warning" title="Arm 1b paused — do not claim WD transfer">
          D=512 best-t flipped to t=1 (D=256 preferred t=10). Cosine may anneal
          too early. t=20 killed ~step 1400. Focus is Arm 2 const-LR.
        </Callout>
      ) : null}

      <Divider />

      <Stack gap={8}>
        <H2>Arm 2 — const LR @ D=256 Lambda (scheduler=none, 10k){CONSTLR_ARM2_PAUSED ? " — paused" : ""}</H2>
        <Text tone="secondary" size="small">
          Tag {CONSTLR_TAG} · η₀=0.25 · AdamW t∈{"{1,5,10}"} then Adam (WD=0) ·
          isolates WD help independent of cosine
        </Text>
        <Table
          headers={["opt", "t", "status", "step", "val ROC", "val acc", "WD"]}
          columnAlign={[
            "left",
            "right",
            "left",
            "right",
            "right",
            "right",
            "right",
          ]}
          rows={CONSTLR_CELLS.map((c) => [
            c.optimizer,
            c.t_epoch == null ? "—" : String(c.t_epoch),
            c.status,
            c.step ? `${c.step}/${c.max_steps}` : "—",
            fmt(c.val_roc_auc),
            fmt(c.val_acc),
            c.optimizer === "adam" ? "0" : fmtWd(c.weight_decay),
          ])}
          rowTone={CONSTLR_CELLS.map((c) => tone(c.status))}
        />
        <Row gap={8} style={{ flexWrap: "wrap" }}>
          {CONSTLR_CELLS.map((c) => (
            <Pill key={c.task} tone={tone(c.status)} size="small">
              {`${optLabel(c)} ${c.status}`}
              {c.val_roc_auc != null ? ` ${fmt(c.val_roc_auc)}` : ""}
            </Pill>
          ))}
        </Row>
      </Stack>

      <Divider />

      <Stack gap={8}>
        <H2>Arm 2c — Adam const LR @ D=1024 Lambda (10k)</H2>
        <Text tone="secondary" size="small">
          Tag {CONSTLR_TAG}-adam · η₀=0.25 · WD=0 · B=64×8 · H=32 (d_h=32) ·
          replaces Arm 2 on Lambda
        </Text>
        <Table
          headers={["opt", "D", "η₀", "status", "step", "val ROC", "val acc"]}
          columnAlign={[
            "left",
            "right",
            "right",
            "left",
            "right",
            "right",
            "right",
          ]}
          rows={
            CONSTLR_D1024
              ? [
                  [
                    "adam",
                    String(CONSTLR_D1024.width),
                    String(CONSTLR_D1024.eta0),
                    CONSTLR_D1024.status,
                    CONSTLR_D1024.step
                      ? `${CONSTLR_D1024.step}/${CONSTLR_D1024.max_steps}`
                      : "—",
                    fmt(CONSTLR_D1024.val_roc_auc),
                    fmt(CONSTLR_D1024.val_acc),
                  ],
                ]
              : []
          }
          rowTone={CONSTLR_D1024 ? [tone(CONSTLR_D1024.status)] : []}
        />
      </Stack>

      <Divider />

      <Stack gap={8}>
        <H2>Arm 2b — const LR @ D=384 ailab (B=128×4, 10k)</H2>
        <Text tone="secondary" size="small">
          Tag {CONSTLR_D384_TAG} · η₀=0.25 · same recipe as Arm 2 · packing
          vs Lambda 64×8 ignored
        </Text>
        <Table
          headers={["opt", "t", "status", "step", "val ROC", "val acc", "WD"]}
          columnAlign={[
            "left",
            "right",
            "left",
            "right",
            "right",
            "right",
            "right",
          ]}
          rows={CONSTLR_D384_CELLS.map((c) => [
            c.optimizer,
            c.t_epoch == null ? "—" : String(c.t_epoch),
            c.status,
            c.step ? `${c.step}/${c.max_steps}` : "—",
            fmt(c.val_roc_auc),
            fmt(c.val_acc),
            c.optimizer === "adam" ? "0" : fmtWd(c.weight_decay),
          ])}
          rowTone={CONSTLR_D384_CELLS.map((c) => tone(c.status))}
        />
        <Row gap={8} style={{ flexWrap: "wrap" }}>
          {CONSTLR_D384_CELLS.map((c) => (
            <Pill key={`d384-${c.task}`} tone={tone(c.status)} size="small">
              {`${optLabel(c)} ${c.status}`}
              {c.val_roc_auc != null ? ` ${fmt(c.val_roc_auc)}` : ""}
            </Pill>
          ))}
        </Row>
      </Stack>

      <Divider />

      <Stack gap={8}>
        <H2>Arm 3 — Adam η₀×D ailab (panel a fill-in, 10k)</H2>
        <Text tone="secondary" size="small">
          Tag {ADAM_ETA_TAG} · Adam WD=0 · scheduler=none · D∈{"{256,384}"} ·
          η₀∈{"{0.05,0.1,0.25,0.5,1}"} · B=128×4 · D384 η=0.25 from Arm 2b
        </Text>
        <Table
          headers={["D", "η₀", "status", "step", "val ROC", "val acc"]}
          columnAlign={[
            "right",
            "right",
            "left",
            "right",
            "right",
            "right",
          ]}
          rows={ADAM_ETA_CELLS.map((c) => [
            String(c.width),
            String(c.eta0),
            c.status,
            c.step ? `${c.step}/${c.max_steps}` : "—",
            fmt(c.val_roc_auc),
            fmt(c.val_acc),
          ])}
          rowTone={ADAM_ETA_CELLS.map((c) => tone(c.status))}
        />
        <Row gap={8} style={{ flexWrap: "wrap" }}>
          {ADAM_ETA_CELLS.map((c) => (
            <Pill key={`ae-${c.task}`} tone={tone(c.status)} size="small">
              {`D${c.width} η=${c.eta0} ${c.status}`}
              {c.val_roc_auc != null ? ` ${fmt(c.val_roc_auc)}` : ""}
            </Pill>
          ))}
        </Row>
      </Stack>

      <Divider />

      <Stack gap={8}>
        <H2>Arm 1/1b — t-sweep (5k flat → cosine, 12k) — paused</H2>
        <Text tone="secondary" size="small">
          Tag {HXFER_TAG} · D=256 done · D=512 t=1/5/10 done, t=20 paused ·
          cutoff {HXFER_MAX_STEPS}
          {HXFER_N_PAUSED ? ` · ${HXFER_N_PAUSED} paused` : ""}
        </Text>
        <Table
          headers={["D", "t", "status", "step", "val ROC", "val acc", "WD"]}
          columnAlign={[
            "right",
            "right",
            "left",
            "right",
            "right",
            "right",
            "right",
          ]}
          rows={HXFER_CELLS.map((c) => [
            String(c.width),
            String(c.t_epoch),
            c.status,
            c.step ? `${c.step}/${c.max_steps}` : "—",
            fmt(c.val_roc_auc),
            fmt(c.val_acc),
            fmtWd(c.weight_decay),
          ])}
          rowTone={HXFER_CELLS.map((c) => tone(c.status))}
        />
      </Stack>

      <Divider />

      <Stack gap={8}>
        <H2>Edge η₀×D (panel c) — Lambda D∈{"{256,384,512}"}</H2>
        <Text tone="secondary" size="small">
          CAPEN @ k★=12, R★=0.1 · B≈96×4 · {MAX_STEPS} Adam steps · tag {TAG}
          {best
            ? ` · best ROC ${fmt(best.val_roc_auc)} (D=${best.width}, η=${best.eta0})`
            : ""}
        </Text>
        <Table
          headers={["task", "D", "η₀", "status", "step", "val ROC"]}
          columnAlign={["right", "right", "right", "left", "right", "right"]}
          rows={CELLS.map((c) => [
            String(c.task),
            String(c.width),
            String(c.eta0),
            c.status,
            c.step ? `${c.step}/${c.max_steps}` : "—",
            fmt(c.val_roc_auc),
          ])}
          rowTone={CELLS.map((c) => tone(c.status))}
        />
      </Stack>

      <Divider />

      <Stack gap={8}>
        <H2>Panel (c) fill — D=384 edge η₀ (gpu-test)</H2>
        <Text tone="secondary" size="small">
          Tag {EDGE_GT_TAG} · B=96×4 · {EDGE_GT_MAX_STEPS} Adam · k=12 R=0.1
          · η₀∈{"{0.05,0.1,0.25,0.5,1}"} · done {EDGE_GT_N_DONE}/
          {EDGE_GT_N_TOTAL || 5}
          {EDGE_GT_N_RUN ? ` · running ${EDGE_GT_N_RUN}` : ""}
          {EDGE_GT_N_PEND ? ` · pending ${EDGE_GT_N_PEND}` : ""}
          · D=128 wiped (re-queue later)
        </Text>
        {EDGE_GT_ACTIVE ? (
          <Callout tone="info" title="Active gpu-test edge cell">
            D={EDGE_GT_ACTIVE.width} · η₀={EDGE_GT_ACTIVE.eta0} · step{" "}
            {EDGE_GT_ACTIVE.step}/{EDGE_GT_ACTIVE.max_steps}
            {EDGE_GT_ACTIVE.val_roc_auc != null
              ? ` · val ROC ${fmt(EDGE_GT_ACTIVE.val_roc_auc)}`
              : ""}
          </Callout>
        ) : null}
        <Table
          headers={["D", "η₀", "status", "step", "val ROC", "val acc"]}
          columnAlign={["right", "right", "left", "right", "right", "right"]}
          rows={EDGE_GT_CELLS.map((c) => [
            String(c.width),
            String(c.eta0),
            c.status,
            c.step ? `${c.step}/${c.max_steps}` : "—",
            fmt(c.val_roc_auc),
            fmt(c.val_acc),
          ])}
          rowTone={EDGE_GT_CELLS.map((c) => tone(c.status))}
        />
        <Row gap={8} style={{ flexWrap: "wrap" }}>
          {EDGE_GT_CELLS.map((c) => (
            <Pill
              key={`egt-${c.width}-${c.task}`}
              tone={tone(c.status)}
              size="small"
            >
              {`D${c.width} η=${c.eta0} ${c.status}`}
              {c.val_roc_auc != null ? ` ${fmt(c.val_roc_auc)}` : ""}
            </Pill>
          ))}
        </Row>
      </Stack>

      <Text tone="secondary" size="small" style={{ color: theme.text.secondary }}>
        Source: Lambda ~/logs/edge_eta + ~/logs/hparam_xfer · Della
        scratch/.../logs/jc_hxfer_* + jc_edge384_gt_* ·
        poll ~30s · canvas edge-eta-lambda-live
      </Text>
    </Stack>
  );
}
"""
    )

def main() -> None:
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    et = ZoneInfo("America/New_York")
    updated = datetime.now(et).strftime("%Y-%m-%d %H:%M:%S ET")
    stale_note = ""
    try:
        payload = ssh_probe()
    except Exception as e:
        print(f"PROBE_FAIL {e}")
        if JSON_OUT.is_file():
            try:
                payload = json.loads(JSON_OUT.read_text())
                stale_note = f"SSH probe failed ({e}); showing last good snapshot"
                updated = f"{payload.get('updated', updated)} (STALE — {updated})"
            except Exception:
                raise SystemExit(1)
        else:
            raise SystemExit(1)
    cells = payload["cells"]
    hxfer = payload.get("hxfer_cells") or []
    constlr = payload.get("constlr_cells") or []
    try:
        constlr_d384 = probe_della_constlr_d384()
    except Exception as e:
        print(f"DELLA_D384_PROBE_FAIL {e}")
        constlr_d384 = payload.get("constlr_d384_cells") or []
    payload["constlr_d384_cells"] = constlr_d384
    try:
        adam_eta = probe_della_adam_eta()
    except Exception as e:
        print(f"DELLA_ADAM_ETA_PROBE_FAIL {e}")
        adam_eta = payload.get("adam_eta_cells") or []
    payload["adam_eta_cells"] = adam_eta
    try:
        edge_gt = probe_della_edge_gt()
    except Exception as e:
        print(f"DELLA_EDGE_GT_PROBE_FAIL {e}")
        edge_gt = (
            payload.get("edge_gt_cells") or payload.get("edge_d128_cells") or []
        )
    payload["edge_gt_cells"] = edge_gt
    payload["edge_d128_cells"] = [c for c in edge_gt if c.get("width") == 128]
    write_csv(
        CSV,
        cells,
        [
            "task",
            "width",
            "eta0",
            "status",
            "step",
            "max_steps",
            "val_roc_auc",
            "val_acc",
            "train_loss",
        ],
    )
    write_csv(
        HXFER_CSV,
        hxfer,
        [
            "task",
            "t_epoch",
            "width",
            "eta0",
            "status",
            "step",
            "max_steps",
            "val_roc_auc",
            "val_acc",
            "train_loss",
            "weight_decay",
        ],
    )
    write_csv(
        CONSTLR_CSV,
        constlr,
        [
            "task",
            "optimizer",
            "t_epoch",
            "width",
            "eta0",
            "status",
            "step",
            "max_steps",
            "val_roc_auc",
            "val_acc",
            "train_loss",
            "weight_decay",
        ],
    )
    write_csv(
        CONSTLR_D384_CSV,
        constlr_d384,
        [
            "task",
            "optimizer",
            "t_epoch",
            "width",
            "eta0",
            "status",
            "step",
            "max_steps",
            "val_roc_auc",
            "val_acc",
            "train_loss",
            "weight_decay",
        ],
    )
    write_csv(
        ADAM_ETA_CSV,
        adam_eta,
        [
            "task",
            "width",
            "eta0",
            "status",
            "step",
            "max_steps",
            "val_roc_auc",
            "val_acc",
            "train_loss",
        ],
    )
    write_csv(
        EDGE_GT_CSV,
        edge_gt,
        [
            "task",
            "width",
            "eta0",
            "status",
            "step",
            "max_steps",
            "val_roc_auc",
            "val_acc",
            "train_loss",
        ],
    )
    JSON_OUT.write_text(json.dumps({"updated": updated, **payload}, indent=2))
    CANVAS.parent.mkdir(parents=True, exist_ok=True)
    CANVAS.write_text(_render_canvas(payload, updated))
    patch_fig_md(
        cells,
        hxfer,
        constlr,
        constlr_d384,
        adam_eta,
        payload.get("constlr_d1024"),
        edge_gt,
        updated,
    )
    n_done = sum(1 for c in cells if c["status"] == "done")
    n_run = sum(1 for c in cells if c["status"] == "running")
    hx_done = sum(1 for c in hxfer if c["status"] == "done")
    hx_run = sum(1 for c in hxfer if c["status"] == "running")
    cl_done = sum(1 for c in constlr if c["status"] == "done")
    cl_run = sum(1 for c in constlr if c["status"] == "running")
    cl384_done = sum(1 for c in constlr_d384 if c["status"] == "done")
    cl384_run = sum(1 for c in constlr_d384 if c["status"] == "running")
    ae_done = sum(1 for c in adam_eta if c["status"] == "done")
    ae_run = sum(1 for c in adam_eta if c["status"] == "running")
    e_gt_done = sum(1 for c in edge_gt if c["status"] == "done")
    e_gt_run = sum(1 for c in edge_gt if c["status"] == "running")
    active = next((c for c in cells if c["status"] == "running"), None)
    hx_active = next((c for c in hxfer if c["status"] == "running"), None)
    cl_active = next((c for c in constlr if c["status"] == "running"), None)
    cl384_active = next((c for c in constlr_d384 if c["status"] == "running"), None)
    ae_active = next((c for c in adam_eta if c["status"] == "running"), None)
    e_gt_active = next((c for c in edge_gt if c["status"] == "running"), None)
    active_s = "—"
    if active:
        active_s = f"D{active['width']} η={active['eta0']} step={active['step']}"
    hx_s = "—"
    if hx_active:
        hx_s = f"D{hx_active['width']} t={hx_active['t_epoch']} step={hx_active['step']}"
    cl_s = "—"
    if cl_active:
        t = cl_active.get("t_epoch")
        cl_s = f"{cl_active['optimizer']} t={t} step={cl_active['step']}"
    cl384_s = "—"
    if cl384_active:
        t = cl384_active.get("t_epoch")
        cl384_s = f"{cl384_active['optimizer']} t={t} step={cl384_active['step']}"
    ae_s = "—"
    if ae_active:
        ae_s = f"D{ae_active['width']} η={ae_active['eta0']} step={ae_active['step']}"
    e_gt_s = "—"
    if e_gt_active:
        e_gt_s = (
            f"D{e_gt_active['width']} η={e_gt_active['eta0']} "
            f"step={e_gt_active['step']}"
        )
    print(
        f"{updated} edge={n_done}/15 run={n_run} active={active_s} "
        f"hxfer={hx_done}/{len(hxfer)} run={hx_run} active={hx_s} "
        f"constlr={cl_done}/{len(constlr)} run={cl_run} active={cl_s} "
        f"d384={cl384_done}/{len(constlr_d384)} run={cl384_run} active={cl384_s} "
        f"adam_eta={ae_done}/{len(adam_eta)} run={ae_run} active={ae_s} "
        f"edge_gt={e_gt_done}/{len(edge_gt)} run={e_gt_run} active={e_gt_s} "
        f"canvas={CANVAS.name}"
    )


if __name__ == "__main__":
    main()
