#!/usr/bin/env python3
"""Regenerate canvases/toptag-ft-L4cos-live.canvas.tsx for JetClass→TopTagging FT."""
from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

FT_ROOT = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/toptagging_finetune")
SWEEP = FT_ROOT / "toptagging_output" / "adam" / "sweep_lr"
SWEEP_AW = FT_ROOT / "toptagging_output" / "adamw" / "sweep_lr"
LOG_DIR = FT_ROOT / "logs"
INIT = FT_ROOT / "init_ckpts" / "L4D256-cosine-s781250-frozen.ckpt"
# Active sweep: drop η₀>0.1 (full FT underperformed there early).
ETAS = [0.01, 0.025, 0.05, 0.075, 0.1]
# AdamW + cosine probe (t_epoch=25, floor=0.01× peak LR).
AW_ETAS = [0.01, 0.025, 0.05]
AW_T_EPOCH = 25
AW_MIN_FRAC = 0.01
# Stronger-WD probe: AdamW t_epoch=2 · η₀=0.05 · B=256×4 (ailab/gputest).
AW_T2_ETA = 0.05
AW_T2_EPOCH = 2
AW_T2_BATCH = 256
AW_T2_GPUS = 4
EPOCHS = 10
OUT = Path(
    "/home/jdezoort/.cursor/projects/home-jdezoort/canvases/"
    "toptag-ft-L4cos-live.canvas.tsx"
)


def js(obj) -> str:
    return json.dumps(obj)


def run(cmd: list[str], timeout: int = 20) -> str:
    try:
        return subprocess.check_output(cmd, text=True, timeout=timeout)
    except Exception:
        return ""


def eta_tag(eta: float) -> str:
    return f"{eta:g}".replace(".", "p")


def squeue_ft() -> list[dict]:
    text = run(
        ["squeue", "-u", "jdezoort", "-h", "-o", "%i %j %T %M %R"],
        timeout=20,
    )
    out = []
    for line in text.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 4:
            continue
        name = parts[1]
        if "tt-ft" not in name:
            continue
        out.append(
            {
                "id": parts[0],
                "name": name,
                "state": parts[2],
                "time": parts[3],
                "reason": parts[4] if len(parts) > 4 else "",
            }
        )
    return out


def aw_run_tag(eta: float) -> str:
    et = eta_tag(eta)
    t_tag = f"{AW_T_EPOCH:g}".replace(".", "p")
    min_tag = f"{AW_MIN_FRAC:g}".replace(".", "p")
    # Must stay short — full auto path + long tag hit NAME_MAX=255.
    return f"aw-t{t_tag}-c{min_tag}-e{et}-b256-ep{EPOCHS}"


def aw_t2_run_tag(batch: int | None = None) -> str:
    et = eta_tag(AW_T2_ETA)
    t_tag = f"{AW_T2_EPOCH:g}".replace(".", "p")
    min_tag = f"{AW_MIN_FRAC:g}".replace(".", "p")
    b = AW_T2_BATCH if batch is None else batch
    return f"aw-t{t_tag}-c{min_tag}-e{et}-b{b}x{AW_T2_GPUS}-ep{EPOCHS}"


def parse_run(mode: str, eta: float) -> dict:
    tag_bit = "head-apool" if mode == "head" else "full-apool"
    freezebb = mode == "head"
    et = eta_tag(eta)
    run_tag = f"ft-L4D256-{tag_bit}-cos781k-eta{et}-b256-g1-ep{EPOCHS}-k7-R0p1"
    info: dict = {
        "mode": mode,
        "eta": eta,
        "run_tag": run_tag,
        "epoch": -1,
        "step": -1,
        "best_auc": None,
        "last_auc": None,
        "last_acc": None,
        "last_rej50": None,
        "last_rej30": None,
        "done": False,
        "curve_epochs": [],
        "curve_auc": [],
        "trainable_frac": None,
    }
    sweep = SWEEP
    if mode == "adamw":
        run_tag = aw_run_tag(eta)
        info["run_tag"] = run_tag
        info["mode"] = "adamw"
        sweep = SWEEP_AW
        freezebb = False
    elif mode == "adamw_t2":
        # Prefer B=256×4; fall back to B=128×4 if OOM flag remapped the run.
        sweep = SWEEP_AW
        freezebb = False
        info["mode"] = "adamw_t2"
        cands = []
        if sweep.is_dir():
            for b in (AW_T2_BATCH, 128):
                tag = aw_t2_run_tag(b)
                hits = [
                    p
                    for p in sweep.iterdir()
                    if p.is_dir() and tag in p.name
                ]
                if hits:
                    cands = hits
                    run_tag = tag
                    info["run_tag"] = tag
                    break
        if not cands:
            info["run_tag"] = aw_t2_run_tag()
            return info
        d = max(cands, key=lambda p: p.stat().st_mtime)
        # Fall through to metrics parse below with d set — skip generic cand search.
        if (d / "FT_EPOCHS_DONE").is_file():
            info["done"] = True
        for met in d.rglob("metrics.csv"):
            try:
                with met.open() as f:
                    rows = list(csv.DictReader(f))
            except Exception:
                continue
            for row in rows:
                ep_raw = row.get("epoch") or row.get("Epoch")
                auc_raw = row.get("val_roc_auc") or row.get("val_auc")
                if ep_raw in (None, ""):
                    continue
                try:
                    ep = int(float(ep_raw))
                except Exception:
                    continue
                info["epoch"] = max(info["epoch"], ep)
                if auc_raw not in (None, ""):
                    try:
                        auc = float(auc_raw)
                    except Exception:
                        continue
                    info["curve_epochs"].append(ep)
                    info["curve_auc"].append(auc)
                    info["last_auc"] = auc
                    if info["best_auc"] is None or auc > info["best_auc"]:
                        info["best_auc"] = auc
                if row.get("val_acc") not in (None, ""):
                    try:
                        info["last_acc"] = float(row["val_acc"])
                    except Exception:
                        pass
        if info["epoch"] >= EPOCHS - 1:
            info["done"] = True
        return info
    if not sweep.is_dir():
        return info
    cands = [
        p
        for p in sweep.iterdir()
        if p.is_dir() and run_tag in p.name and (("freezebb" in p.name) == freezebb)
    ]
    if not cands:
        return info
    d = max(cands, key=lambda p: p.stat().st_mtime)
    if (d / "FT_EPOCHS_DONE").is_file():
        info["done"] = True
    for met in d.rglob("metrics.csv"):
        try:
            with met.open() as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        for row in rows:
            ep_raw = row.get("epoch") or row.get("Epoch")
            auc_raw = row.get("val_roc_auc") or row.get("val_auc")
            if ep_raw in (None, ""):
                continue
            try:
                ep = int(float(ep_raw))
            except Exception:
                continue
            info["epoch"] = max(info["epoch"], ep)
            if auc_raw not in (None, ""):
                try:
                    auc = float(auc_raw)
                except Exception:
                    continue
                info["curve_epochs"].append(ep)
                info["curve_auc"].append(auc)
                info["last_auc"] = auc
                if info["best_auc"] is None or auc > info["best_auc"]:
                    info["best_auc"] = auc
            if row.get("val_acc") not in (None, ""):
                try:
                    info["last_acc"] = float(row["val_acc"])
                except Exception:
                    pass
    if info["epoch"] >= EPOCHS - 1:
        info["done"] = True
    return info


def scrape_logs_for_eta(eta: float, *, mode: str) -> dict:
    out: dict = {
        "epoch": -1,
        "step": -1,
        "auc": None,  # last epoch AUC
        "best_auc": None,  # max AUC across epochs in this log
        "acc": None,
        "rej50": None,
        "rej30": None,
        "trainable_frac": None,
        "curve_epochs": [],
        "curve_auc": [],
    }
    if mode == "adamw_t2":
        pattern = "tt_ft_aw_t2_*.out"
    elif mode == "adamw":
        pattern = "tt_ft_L4cos_aw_*.out"
    elif mode == "full":
        pattern = "tt_ft_L4cos_full_*.out"
    else:
        pattern = "tt_ft_L4cos_*.out"
    logs = sorted(LOG_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    # Head glob also matches full_* — drop those for the frozen arm.
    if mode == "head":
        logs = [p for p in logs if "_full_" not in p.name and "_aw_" not in p.name]
    et = eta_tag(eta)
    for log in reversed(logs[-50:]):
        try:
            text = log.read_text(errors="ignore")
        except Exception:
            continue
        if f"η₀={eta}" not in text and f"eta{et}" not in text:
            continue
        # Ignore legacy B=64×4 gputest slices (single-GPU FT arms are B=256×1).
        if mode in ("head", "full"):
            if "B=256×1" not in text and "B=256x1" not in text and "-b256-g1-" not in text:
                continue
        if mode == "head" and "FULL FT" in text:
            continue
        if mode == "full" and "freeze-backbone" in text and "FULL FT" not in text:
            continue
        if mode == "adamw" and "AdamW+cosine" not in text and "aw-t25" not in text:
            continue
        if mode == "adamw_t2" and "aw-t2" not in text and "t=2" not in text:
            continue
        m = re.search(r"trainable ([0-9,]+)/([0-9,]+) \(([0-9.]+)%\)", text)
        if m:
            out["trainable_frac"] = float(m.group(3)) / 100.0
        steps = [int(x) for x in re.findall(r"global_step=(\d+)", text)]
        if steps:
            out["step"] = max(steps)
        # Prefer joint epoch lines (include bg rejection). Track peak ≠ last.
        by_ep: dict[int, tuple] = {}
        for m in re.finditer(
            r"epoch=(\d+)\s+step=(\d+)\s+train_loss=[0-9.]+\s+train_acc=[0-9.]+\s+"
            r"val_loss=[0-9.]+\s+val_acc=([0-9.]+)\s+val_roc_auc=([0-9.]+)\s+"
            r"val_bg_rejection=([0-9.eE+-]+|inf)\s+val_bg_rejection_0p3=([0-9.eE+-]+|inf)",
            text,
        ):
            ep, st = int(m.group(1)), int(m.group(2))
            row = (
                st,
                float(m.group(3)),
                float(m.group(4)),
                None if m.group(5) == "inf" else float(m.group(5)),
                None if m.group(6) == "inf" else float(m.group(6)),
            )
            if ep not in by_ep or st >= by_ep[ep][0]:
                by_ep[ep] = row
        if not by_ep:
            for m in re.finditer(
                r"\[val\] epoch=(\d+) step=(\d+) val_loss=[0-9.]+ val_acc=([0-9.]+) "
                r"val_roc_auc=([0-9.]+)",
                text,
            ):
                ep, st = int(m.group(1)), int(m.group(2))
                row = (st, float(m.group(3)), float(m.group(4)), None, None)
                if ep not in by_ep or st >= by_ep[ep][0]:
                    by_ep[ep] = row
        if by_ep:
            for ep in sorted(by_ep):
                _st, acc, auc, r50, r30 = by_ep[ep]
                out["curve_epochs"].append(ep)
                out["curve_auc"].append(auc)
            last_ep = max(by_ep)
            out["epoch"] = last_ep
            out["acc"] = by_ep[last_ep][1]
            out["auc"] = by_ep[last_ep][2]
            out["rej50"] = by_ep[last_ep][3]
            out["rej30"] = by_ep[last_ep][4]
            best_ep = max(by_ep, key=lambda e: by_ep[e][2])
            out["best_auc"] = by_ep[best_ep][2]
        break
    return out


def enrich(info: dict) -> dict:
    logi = scrape_logs_for_eta(info["eta"], mode=info["mode"])
    if logi.get("step", -1) >= 0:
        info["step"] = max(info.get("step", -1), logi["step"])
    if logi.get("rej50") is not None:
        info["last_rej50"] = logi["rej50"]
    if logi.get("rej30") is not None:
        info["last_rej30"] = logi["rej30"]
    # Prefer log-derived peak when metrics.csv is missing / incomplete.
    if logi.get("best_auc") is not None:
        if info["best_auc"] is None or logi["best_auc"] > info["best_auc"]:
            info["best_auc"] = logi["best_auc"]
    if logi.get("curve_auc"):
        # Merge curves; keep higher AUC per epoch.
        merged: dict[int, float] = dict(
            zip(info.get("curve_epochs") or [], info.get("curve_auc") or [])
        )
        for ep, auc in zip(logi["curve_epochs"], logi["curve_auc"]):
            if ep not in merged or auc > merged[ep]:
                merged[ep] = auc
        info["curve_epochs"] = sorted(merged)
        info["curve_auc"] = [merged[e] for e in info["curve_epochs"]]
        if info["best_auc"] is None and info["curve_auc"]:
            info["best_auc"] = max(info["curve_auc"])
    if logi.get("epoch", -1) >= 0:
        info["epoch"] = max(info.get("epoch", -1), logi["epoch"])
    if logi.get("auc") is not None:
        info["last_auc"] = logi["auc"]
    if logi.get("acc") is not None:
        info["last_acc"] = logi["acc"]
    if logi.get("trainable_frac") is not None:
        info["trainable_frac"] = logi["trainable_frac"]
    return info


def fmt_rej(v) -> str:
    if v is None:
        return "—"
    if v == float("inf"):
        return "inf"
    return f"{v:.0f}"


def fmt_progress(info: dict) -> str:
    ep = max(info.get("epoch", -1), 0)
    base = f"{ep}/{EPOCHS}"
    step = info.get("step", -1)
    if step is not None and step >= 0 and not info.get("done"):
        return f"{base} · s{step}"
    return base


def fmt_auc(v) -> str:
    return "—" if v is None else f"{v:.4f}"


def fmt_acc(v) -> str:
    return "—" if v is None else f"{v:.4f}"


def eta_queue_state(eta: float, mode: str, jobs: list[dict]) -> str | None:
    """Map squeue job names → running/pending for this η₀ cell."""
    et = eta_tag(eta)
    if mode == "adamw_t2":
        names = {"tt-ft-aw-t2-e0p05", "tt-ft-aw-t2"}
    elif mode == "adamw":
        names = {f"tt-ft-L4cos-aw-e{et}", "tt-ft-L4cos-aw"}
    elif mode == "full":
        names = {f"tt-ft-L4cos-full-e{et}", "tt-ft-L4cos-full"}
    else:
        names = {f"tt-ft-L4cos-e{et}"}
        # Legacy single-name job was always η₀=0.01.
        if abs(eta - 0.01) < 1e-12:
            names.add("tt-ft-L4cos")
    matched = [j for j in jobs if j["name"] in names]
    if any(j["state"] == "RUNNING" for j in matched):
        return "running"
    if any(j["state"] == "PENDING" for j in matched):
        return "pending"
    return None


def status_label(info: dict, qstate: str | None) -> str:
    if info["done"]:
        return "done"
    if qstate == "running" or info["epoch"] >= 0:
        return "active"
    if qstate == "pending":
        return "pending"
    return "queued"


def main() -> None:
    jobs = squeue_ft()
    head_rows = []
    full_rows = []
    aw_rows = []
    head_best_etas: list[str] = []
    head_best_aucs: list[float | None] = []
    active = None

    for eta in ETAS:
        h = enrich(parse_run("head", eta))
        f = enrich(parse_run("full", eta))
        hq = eta_queue_state(eta, "head", jobs)
        fq = eta_queue_state(eta, "full", jobs)
        head_rows.append(
            [
                f"{eta:g}",
                "frozen head",
                status_label(h, hq),
                fmt_progress(h),
                fmt_auc(h["best_auc"]),
                fmt_auc(h["last_auc"]),
                fmt_acc(h.get("last_acc")),
            ]
        )
        full_rows.append(
            [
                f"{eta:g}",
                "full FT",
                status_label(f, fq),
                fmt_progress(f),
                fmt_auc(f["best_auc"]),
                fmt_auc(f["last_auc"]),
                fmt_rej(f.get("last_rej50")),
                fmt_rej(f.get("last_rej30")),
            ]
        )
        head_best_etas.append(f"{eta:g}")
        head_best_aucs.append(h["best_auc"])
        if h["curve_auc"] and (active is None or not active.get("done")):
            if active is None or (not h["done"] and active.get("done")):
                active = h
            elif not h["done"] and h["eta"] <= (active["eta"] if active else 99):
                active = h

    for eta in AW_ETAS:
        a = enrich(parse_run("adamw", eta))
        aq = eta_queue_state(eta, "adamw", jobs)
        aw_rows.append(
            [
                f"{eta:g}",
                "AdamW+cos",
                status_label(a, aq),
                fmt_progress(a),
                fmt_auc(a["best_auc"]),
                fmt_auc(a["last_auc"]),
                fmt_rej(a.get("last_rej50")),
                fmt_rej(a.get("last_rej30")),
            ]
        )

    t2 = enrich(parse_run("adamw_t2", AW_T2_ETA))
    t2q = eta_queue_state(AW_T2_ETA, "adamw_t2", jobs)
    aw_t2_rows = [
        [
            f"{AW_T2_ETA:g}",
            f"AdamW t={AW_T2_EPOCH:g} · B={AW_T2_BATCH}×{AW_T2_GPUS}",
            status_label(t2, t2q),
            fmt_progress(t2),
            fmt_auc(t2["best_auc"]),
            fmt_auc(t2["last_auc"]),
            fmt_rej(t2.get("last_rej50")),
            fmt_rej(t2.get("last_rej30")),
        ]
    ]

    if active is None:
        active = enrich(parse_run("head", 0.01))

    h01 = enrich(parse_run("head", 0.01))
    n_head_done = sum(1 for r in head_rows if r[2] == "done")
    n_full_done = sum(1 for r in full_rows if r[2] == "done")
    n_aw_done = sum(1 for r in aw_rows if r[2] == "done")
    n_aw_t2_done = 1 if aw_t2_rows[0][2] == "done" else 0
    running_jobs = [j for j in jobs if j["state"] == "RUNNING"]
    running = (
        ", ".join(f"{j['name']} · {j['id']}" for j in running_jobs)
        if running_jobs
        else None
    )
    pending_n = sum(1 for j in jobs if j["state"] == "PENDING")

    init_ok = INIT.is_file()
    init_mode = oct(INIT.stat().st_mode & 0o777) if init_ok else "missing"
    updated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")

    curve_cats = [str(e) for e in active.get("curve_epochs") or []]
    curve_auc = active.get("curve_auc") or []
    active_eta = f"{active['eta']:g}" if active else "—"
    has_bar = any(v is not None for v in head_best_aucs)

    # Prefer test-eval SUMMARY if present.
    test_summary = "—"
    test_json = FT_ROOT / "manifests" / "ft_L4D256_full_eta0p025_best_test.json"
    if test_json.is_file():
        try:
            tj = json.loads(test_json.read_text())
            m = tj.get("metrics") or {}
            test_summary = (
                f"AUC={m.get('test_roc_auc', float('nan')):.4f} · "
                f"Rej50={m.get('test_bg_rejection', float('nan')):.0f} · "
                f"Rej30={m.get('test_bg_rejection_0p3', float('nan')):.0f}"
            )
        except Exception:
            pass

    canvas = f'''import {{
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
const INIT_OK = {js(init_ok)};
const INIT_MODE = {js(init_mode)};
const H01_AUC = {js(h01.get("last_auc"))};
const H01_BEST = {js(h01.get("best_auc"))};
const H01_EP = {max(h01["epoch"], 0)};
const H01_ACC = {js(h01.get("last_acc"))};
const TRAINABLE = {js(h01.get("trainable_frac"))};
const HEAD_DONE = {n_head_done};
const FULL_DONE = {n_full_done};
const AW_DONE = {n_aw_done};
const AW_T2_DONE = {n_aw_t2_done};
const N_ETAS = {len(ETAS)};
const N_AW = {len(AW_ETAS)};
const RUNNING = {js(running)};
const PENDING = {pending_n};
const HEAD_TABLE = {js(head_rows)};
const FULL_TABLE = {js(full_rows)};
const AW_TABLE = {js(aw_rows)};
const AW_T2_TABLE = {js(aw_t2_rows)};
const TEST_SUMMARY = {js(test_summary)};
const CURVE_CATS: string[] = {js(curve_cats)};
const CURVE_AUC: (number | null)[] = {js(curve_auc)};
const BAR_ETAS: string[] = {js(head_best_etas)};
const BAR_AUC: (number | null)[] = {js(head_best_aucs)};
const ACTIVE_ETA = {js(active_eta)};
const HAS_CURVE = CURVE_AUC.length > 0;
const HAS_BAR = {js(has_bar)};

export default function ToptagFtL4cosLive() {{
  return (
    <Stack gap={{20}}>
      <Stack gap={{6}}>
        <H1>TopTagging fine-tune — L4 cosine init</H1>
        <Text tone="secondary">
          Adam const vs AdamW+cosine probe · attentional head · updated {{UPDATED}}
        </Text>
        <Row gap={{8}} wrap>
          <Pill tone={{INIT_OK ? "success" : "warning"}}>
            init {{INIT_OK ? `read-only (${{INIT_MODE}})` : "missing"}}
          </Pill>
          <Pill tone="neutral">Adam const (rolling)</Pill>
          <Pill tone="info">AdamW t=25 + cos→0.01×</Pill>
          <Pill tone="neutral">B=256×1 · 10 ep · k=7 R=0.1</Pill>
        </Row>
      </Stack>

      <Callout tone="info">
        Adam const-LR full FT (η₀=0.025) hit test AUC 0.9873 / Rej₅₀%=554 / Rej₃₀%=2499
        vs ParT-f.t. 0.9877 / 691 / 2766. New probe: AdamW with t_epoch=25 WD +
        epoch cosine to 0.01× peak LR over 10 epochs, η₀ ∈ {{0.01, 0.025, 0.05}}.
      </Callout>

      <Grid columns={{4}} gap={{12}}>
        <Stat
          value={{H01_AUC == null ? "—" : H01_AUC.toFixed(4)}}
          label={{`frozen η₀=0.01 val AUC (ep ${{H01_EP}})`}}
          tone="accent"
        />
        <Stat value={{TEST_SUMMARY}} label="Adam η₀=0.025 test (400k)" />
        <Stat value={{`${{FULL_DONE}}/${{N_ETAS}}`}} label="Adam full cells done" />
        <Stat value={{`${{AW_DONE}}/${{N_AW}}`}} label="AdamW+cos cells done" />
      </Grid>

      <H2>Protocol</H2>
      <Table
        headers={{["knob", "value"]}}
        rows={{[
          ["Init", "L4/D256 JetClass cosine last.ckpt @ 781250 (pretrain val AUC 0.9642)"],
          ["Protect", "immutable copy under toptagging_finetune/init_ckpts/ (chmod a-w)"],
          ["Head", "fresh attentional readout (skip_readout on load)"],
          ["Adam arm", "const LR · η₀ ∈ 0.01…0.1 · full FT"],
          ["AdamW probe", "AdamW t_epoch=25 · cosine → 0.01× peak · η₀ ∈ 0.01, 0.025, 0.05 · B=256×1"],
          ["AdamW t=2", "AdamW t_epoch=2 · η₀=0.05 · B=256×4 · ailab gpu-test (stronger WD)"],
          ["Graph", "live star R=0.1 + kNN k=7 · kin7 · part-interaction"],
          ["Budget", "10 epochs · checkpoint on val_roc_auc"],
          ["Metrics", "val AUC + Rej₅₀%/Rej₃₀% · 400k test for selected best.ckpt"],
        ]}}
      />

      <H2>Full fine-tune — Adam const LR</H2>
      <Text tone="secondary">
        Rej₅₀% / Rej₃₀% on 20k val are discrete (n_bg/n_fp); trust 400k test numbers.
        Test η₀=0.025: {{TEST_SUMMARY}}.
      </Text>
      <Table
        headers={{["η₀", "mode", "status", "epoch", "best AUC", "last AUC", "Rej₅₀%", "Rej₃₀%"]}}
        rows={{FULL_TABLE}}
      />

      <H2>Probe — AdamW t_epoch=25 + cosine → 0.01×</H2>
      <Table
        headers={{["η₀", "mode", "status", "epoch", "best AUC", "last AUC", "Rej₅₀%", "Rej₃₀%"]}}
        rows={{AW_TABLE}}
      />

      <H2>Probe — AdamW t_epoch=2 (stronger WD) · η₀=0.05 · B=256×4</H2>
      <Text tone="secondary">
        λ ≈ 50× the t=25 / B=256×1 arm. Queued on ailab gpu-test; falls back to
        B=128×4 on OOM.
      </Text>
      <Table
        headers={{["η₀", "mode", "status", "epoch", "best AUC", "last AUC", "Rej₅₀%", "Rej₃₀%"]}}
        rows={{AW_T2_TABLE}}
      />

      <H2>Frozen head — η₀ grid (cancelled)</H2>
      <Table
        headers={{["η₀", "mode", "status", "epoch", "best AUC", "last AUC", "last acc"]}}
        rows={{HEAD_TABLE}}
      />

      {{HAS_CURVE ? (
        <Stack gap={{8}}>
          <H2>{{`Val ROC AUC vs epoch (frozen η₀=${{ACTIVE_ETA}})`}}</H2>
          <LineChart
            categories={{CURVE_CATS}}
            series={{[
              {{
                name: `frozen η₀=${{ACTIVE_ETA}} val ROC`,
                data: CURVE_AUC.map((v) => (v == null ? Number.NaN : Number(v))),
                tone: "accent",
              }},
            ]}}
            height={{220}}
          />
        </Stack>
      ) : null}}

      {{HAS_BAR ? (
        <Stack gap={{8}}>
          <H2>Best val ROC by η₀ (frozen head, so far)</H2>
          <LineChart
            categories={{BAR_ETAS}}
            series={{[
              {{
                name: "best val ROC",
                data: BAR_AUC.map((v) => (v == null ? Number.NaN : Number(v))),
                tone: "info",
              }},
            ]}}
            height={{200}}
          />
        </Stack>
      ) : null}}

      <Callout tone="neutral">
        Queue: {{RUNNING ?? "no FT job running"}} · pending={{PENDING}} ·
        Adam {{FULL_DONE}}/{{N_ETAS}} · AdamW-t25 {{AW_DONE}}/{{N_AW}} ·
        AdamW-t2 {{AW_T2_DONE}}/1.
      </Callout>
    </Stack>
  );
}}
'''
    OUT.write_text(canvas)
    print(
        f"wrote {OUT} head_done={n_head_done}/{len(ETAS)} "
        f"aw_done={n_aw_done}/{len(AW_ETAS)} "
        f"h01_auc={h01.get('last_auc')} ep={h01['epoch']}"
    )


if __name__ == "__main__":
    main()
