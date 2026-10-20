#!/usr/bin/env python3
"""Regenerate canvases/jetclass-pretrain-live.canvas.tsx for L4 epoch + L8 giant."""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

PRETRAIN = Path("/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_pretrain")
SWEEP = PRETRAIN / "jetclass_output/adam/sweep_lr"
L4_LOGDIR = PRETRAIN / "logs"
L4_TAG = "pretrain-b64-g4-s390625-dh32-k7-R0p1"
L4_COS_TAG = "pretrain-cos-b64-g4-s781250"
L4_MAX = 390625
L4_COS_MAX = 781250
L4_SEED_STEP = 40000
L8_MIRROR = PRETRAIN / "lambda_L8D1024"
L8_MAX = 160000  # epoch-2 @ B_eff=128 (const η₀; epoch-1 finished at 80k)
OUT = Path(
    "/home/jdezoort/.cursor/projects/home-jdezoort/canvases/"
    "jetclass-pretrain-live.canvas.tsx"
)
LAMBDA_HOST = os.environ.get("LAMBDA_HOST", "ubuntu@168.110.21.94")


def js(obj) -> str:
    return json.dumps(obj)


def run(cmd: list[str], timeout: int = 45) -> str:
    try:
        return subprocess.check_output(cmd, text=True, timeout=timeout)
    except Exception:
        return ""


_SQUEUE_CACHE: dict | None = None


def squeue_jobs_by_name() -> dict[str, list[dict]]:
    """One squeue call for all jc-pretrain-* jobs (avoids N hung/slow queries)."""
    global _SQUEUE_CACHE
    if _SQUEUE_CACHE is not None:
        return _SQUEUE_CACHE
    text = run(
        [
            "squeue",
            "-u",
            "jdezoort",
            "-h",
            "-o",
            "%i %j %T %M %R %P",
        ],
        timeout=20,
    )
    by: dict[str, list[dict]] = {}
    for line in text.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 4:
            continue
        jid, name, state, tim = parts[0], parts[1], parts[2], parts[3]
        rest = parts[4] if len(parts) == 5 else (parts[5] if len(parts) >= 6 else "")
        part = parts[4] if len(parts) >= 6 else "?"
        reason = rest
        by.setdefault(name, []).append(
            {
                "id": jid,
                "state": state,
                "time": tim,
                "reason": reason,
                "partition": part,
            }
        )
    _SQUEUE_CACHE = by
    return by


def parquet_series(pq: Path) -> dict:
    out = {
        "step": 0,
        "auc": None,
        "loss": None,
        "acc": None,
        "train_loss": None,
        "curve_steps": [],
        "curve_auc": [],
        "curve_loss": [],
    }
    if not pq.is_file():
        return out
    try:
        import pyarrow.parquet as pqmod

        df = pqmod.read_table(pq).to_pandas()
    except Exception:
        return out
    if "global_step" not in df.columns or df.empty:
        return out
    df = df.sort_values("global_step")
    out["step"] = int(df["global_step"].max())
    for col, key in (
        ("val_roc_auc", "auc"),
        ("val_loss", "loss"),
        ("val_acc", "acc"),
        ("train_loss", "train_loss"),
    ):
        if col in df.columns:
            s = df[col].dropna()
            if len(s):
                out[key] = float(s.iloc[-1])
    if "val_roc_auc" in df.columns:
        sub = df.dropna(subset=["val_roc_auc"])
        # downsample for canvas size
        if len(sub) > 120:
            idx = list(range(0, len(sub), max(1, len(sub) // 120)))
            if idx[-1] != len(sub) - 1:
                idx.append(len(sub) - 1)
            sub = sub.iloc[idx]
        out["curve_steps"] = [int(x) for x in sub["global_step"].tolist()]
        out["curve_auc"] = [float(x) for x in sub["val_roc_auc"].tolist()]
        if "val_loss" in sub.columns:
            out["curve_loss"] = [
                float(x) if x == x else None for x in sub["val_loss"].tolist()
            ]
    return out


def ckpt_step(run_dir: Path) -> int:
    """Best-effort step from sidecar / zip meta — never torch.load (L8 ckpt ~4GB)."""
    best = 0
    if not run_dir.is_dir():
        return 0
    for ckpt in run_dir.glob("last*.ckpt"):
        best = max(best, _ckpt_global_step_fast(ckpt))
    return best


def _ckpt_global_step_fast(path: Path) -> int:
    """Read global_step without materializing tensors."""
    import pickle
    import zipfile

    # Sidecar written by some trainers
    for side in (path.with_suffix(path.suffix + ".json"), path.with_suffix(".json")):
        if side.is_file():
            try:
                data = json.loads(side.read_text())
                if "global_step" in data:
                    return int(data["global_step"])
            except Exception:
                pass
    try:
        with zipfile.ZipFile(path) as zf:
            # Lightning / torch zip: peek pickle bytes for ASCII 'global_step' nearby int
            for name in zf.namelist():
                if not name.endswith("data.pkl") and name not in {"data.pkl", "archive/data.pkl"}:
                    continue
                raw = zf.read(name)
                # Prefer a tiny regex over full unpickle (storages would fail anyway).
                # Pickle encodes ints nearby; fall through to Unpickler with stub storages.
                class _Stub:
                    def __init__(self, *a, **k):
                        pass

                class _U(pickle.Unpickler):
                    def persistent_load(self, pid):
                        return _Stub()

                    def find_class(self, module, name):
                        if module.startswith("torch"):
                            return _Stub
                        return super().find_class(module, name)

                try:
                    obj = _U(io_bytes := __import__("io").BytesIO(raw)).load()
                    if isinstance(obj, dict) and "global_step" in obj:
                        return int(obj["global_step"] or 0)
                    # Lightning wraps as {"state_dict":..., "callbacks":..., "epoch":...}
                    if isinstance(obj, dict):
                        for k in ("global_step", "global_step_count"):
                            if k in obj:
                                return int(obj[k] or 0)
                        loops = obj.get("loops") or {}
                        if isinstance(loops, dict):
                            fit = loops.get("fit_loop") or {}
                            if isinstance(fit, dict) and "epoch_loop" in fit:
                                el = fit["epoch_loop"]
                                if isinstance(el, dict) and "batch_progress" in el:
                                    bp = el["batch_progress"]
                                    if isinstance(bp, dict) and "total" in bp:
                                        tot = bp["total"]
                                        if isinstance(tot, dict) and "completed" in tot:
                                            return int(tot["completed"] or 0)
                except Exception:
                    # Byte scan: global_step followed by pickle INT protocol
                    m = re.search(rb"global_step.{0,40}", raw)
                    if m:
                        chunk = raw[m.start() : m.start() + 64]
                        nums = re.findall(rb"[I\x8a\x8b](.{1,8})", chunk)
                        # Last resort: decode any ascii digits in the window
                        digs = re.findall(rb"(\d{3,8})", chunk)
                        if digs:
                            return int(digs[0])
    except (zipfile.BadZipFile, Exception):
        pass
    return 0


def find_l4() -> tuple[Path | None, Path | None]:
    """Prefer the live run dir (``s390p625k`` from format_count), skip superseded seeds."""
    dirs = [p for p in SWEEP.glob(f"*4_256_0p1*{L4_TAG}") if p.is_dir()]
    # Drop wrong-named seed dir (…s390625… without p) if the live one exists.
    live = [p for p in dirs if "s390p625k" in p.name and not (p / "SUPERSEDED.txt").exists()]
    if live:
        dirs = live
    else:
        dirs = [p for p in dirs if not (p / "SUPERSEDED.txt").exists()]
    if not dirs:
        return None, None
    # Newest mtime wins if multiple
    d = max(dirs, key=lambda p: p.stat().st_mtime)
    pq = SWEEP / f"{d.name}.parquet"
    return d, pq if pq.is_file() else None


def _parse_train_log_metrics(text: str) -> tuple[int, float | None, float | None, float | None]:
    steps = [int(x) for x in re.findall(r"global_step=(\d+)", text)]
    steps += [int(x) for x in re.findall(r"\[train\] step=(\d+)", text)]
    step = max(steps) if steps else 0
    auc = loss = train = None
    am = re.findall(r"val_roc_auc=([0-9.eE+-]+)", text)
    if am:
        try:
            auc = float(am[-1])
        except ValueError:
            pass
    lm = re.findall(r"val_loss=([0-9.eE+-]+)", text)
    if lm:
        try:
            loss = float(lm[-1])
        except ValueError:
            pass
    tm = re.findall(r"train_loss=([0-9.eE+-]+)", text)
    if tm:
        try:
            train = float(tm[-1])
        except ValueError:
            pass
    return step, auc, loss, train


def log_step_l4() -> tuple[int, float | None, float | None, float | None]:
    """Fallback: parse the active (or newest live) Slurm .out for step / metrics."""
    # Prefer the currently queued/running job's log so a cancelled scratch run
    # cannot poison the canvas with a lower global_step.
    job_ids = [j["id"].split("_")[0] for j in squeue_jobs_by_name().get("jc-pretrain-L4", [])]
    logs: list[Path] = []
    for jid in job_ids:
        p = L4_LOGDIR / f"jc_pretrain_L4_gt_{jid}.out"
        if p.is_file():
            logs.append(p)
    if not logs:
        # Fall back to newest log that actually used the live s390p625k run dir.
        for p in sorted(
            L4_LOGDIR.glob("jc_pretrain_L4_gt_*.out"),
            key=lambda q: q.stat().st_mtime,
            reverse=True,
        ):
            head = p.read_text(errors="ignore")[:8000]
            if "s390p625k" in head:
                logs = [p]
                break
    if not logs:
        return 0, None, None, None
    text = logs[0].read_text(errors="ignore")[-200_000:]
    return _parse_train_log_metrics(text)


def find_l4_cos() -> tuple[Path | None, Path | None]:
    dirs = [
        p
        for p in SWEEP.glob(f"*4_256_0p1*{L4_COS_TAG}*")
        if p.is_dir() and not (p / "SUPERSEDED.txt").exists()
    ]
    if not dirs:
        return None, None
    d = max(dirs, key=lambda p: p.stat().st_mtime)
    pq = SWEEP / f"{d.name}.parquet"
    return d, pq if pq.is_file() else None


def log_step_l4cos() -> tuple[int, float | None, float | None, float | None, list[tuple[int, float]]]:
    """Cosine epoch-2 metrics + dense train_loss series from L4cos logs."""
    job_ids = [j["id"].split("_")[0] for j in squeue_jobs_by_name().get("jc-pretrain-L4cos", [])]
    logs: list[Path] = []
    for jid in job_ids:
        p = L4_LOGDIR / f"jc_pretrain_L4cos_gt_{jid}.out"
        if p.is_file():
            logs.append(p)
    if not logs:
        logs = sorted(
            L4_LOGDIR.glob("jc_pretrain_L4cos_gt_*.out"),
            key=lambda q: q.stat().st_mtime,
            reverse=True,
        )[:5]
    best_step, auc, loss, train = 0, None, None, None
    tl_map: dict[int, float] = {}
    for p in logs:
        text = p.read_text(errors="ignore")
        step, a, lo, tr = _parse_train_log_metrics(text[-200_000:])
        if step >= best_step:
            best_step = step
            if a is not None:
                auc = a
            if lo is not None:
                loss = lo
            if tr is not None:
                train = tr
        for m in re.finditer(
            r"\[train\] step=(\d+)\s+epoch=\d+\s+train_loss=([0-9.eE+-]+)", text
        ):
            tl_map[int(m.group(1))] = float(m.group(2))
    return best_step, auc, loss, train, sorted(tl_map.items())


def slurm_l4cos() -> dict:
    jobs = squeue_jobs_by_name().get("jc-pretrain-L4cos", [])
    running = sum(1 for j in jobs if j["state"] == "RUNNING")
    pending = sum(1 for j in jobs if j["state"] == "PENDING")
    return {"jobs": jobs, "running": running, "pending": pending}


def della_l8_status() -> dict:
    """L8 continue on Della gputest (preferred over Lambda once mirrored)."""
    info = {
        "step": 0,
        "train_loss": None,
        "status": "idle",
        "tl_steps": [],
        "tl_vals": [],
        "note": "",
        "gpus": "—",
    }
    tags = (
        "pretrain-L8D1024-b32-g4-s160000",
        "pretrain-L8D1024-b32-g4-s80000",
        "pretrain-L8D1024-b32-g4-s40000",
    )
    seed = L8_MIRROR / "checkpoints" / "last.ckpt"
    if seed.is_file():
        info["step"] = max(info["step"], _ckpt_global_step_fast(seed))
    for tag in tags:
        for p in SWEEP.glob(f"*8_1024_0p1*{tag}*"):
            if not p.is_dir() or (p / "SUPERSEDED.txt").exists():
                continue
            info["step"] = max(info["step"], ckpt_step(p))
    logs = sorted(L4_LOGDIR.glob("jc_pretrain_L8_gt_*.out"), key=lambda q: q.stat().st_mtime)
    if logs:
        # Tail only — full L8 logs can be large
        text = logs[-1].read_text(errors="ignore")[-400_000:]
        steps = [int(x) for x in re.findall(r"global_step=(\d+)", text)]
        steps += [int(x) for x in re.findall(r"\[train\] step=(\d+)", text)]
        if steps:
            info["step"] = max(info["step"], max(steps))
        tl_map = {}
        for m in re.finditer(
            r"\[train\] step=(\d+)\s+epoch=\d+\s+train_loss=([0-9.eE+-]+)", text
        ):
            tl_map[int(m.group(1))] = float(m.group(2))
        if tl_map:
            items = sorted(tl_map.items())
            info["train_loss"] = items[-1][1]
            if len(items) > 150:
                idx = list(range(0, len(items), max(1, len(items) // 150)))
                idx.append(len(items) - 1)
                items = [items[i] for i in sorted(set(idx))]
            info["tl_steps"] = [s for s, _ in items]
            info["tl_vals"] = [v for _, v in items]
    states = [j["state"] for j in squeue_jobs_by_name().get("jc-pretrain-L8", [])]
    if any(s == "RUNNING" for s in states):
        info["status"] = "training"
        info["gpus"] = "Della gputest 4×A100"
    elif any(s == "PENDING" for s in states):
        info["status"] = "pending"
        info["gpus"] = "Della gputest (queued)"
    elif info["step"] >= L8_MAX:
        info["status"] = "done"
    elif info["step"] > 0:
        info["status"] = "paused"
    info["note"] = "Epoch-2 → 160k · B=32×4 · seeded from finished 80k epoch"
    info["reachable"] = True
    info["ready"] = True
    info["train_files"] = 210
    # One-shot JetClass val at end of epoch-1 (80k), if present.
    val_json = PRETRAIN / "manifests" / "L8D1024_val_at_80k.json"
    if val_json.is_file():
        try:
            v = json.loads(val_json.read_text())
            mets = (v.get("metrics") or [{}])[0]
            info["auc"] = float(mets.get("val_roc_auc")) if mets.get("val_roc_auc") is not None else None
            info["val_step"] = int(v.get("global_step") or 0)
        except Exception:
            info["auc"] = None
    else:
        info["auc"] = None
    return info


def local_l8_staging() -> dict:
    """Read Della-side stage log so canvas updates without SSH."""
    info = {
        "files": 0,
        "min_per_class": 0,
        "oks": 0,
        "last": "",
    }
    log = Path("/tmp/lambda_jetclass_stage.log")
    if not log.is_file():
        return info
    text = log.read_text(errors="ignore")
    oks = re.findall(r" OK train/([A-Za-z0-9_]+\.root)", text)
    info["oks"] = len(oks)
    info["last"] = oks[-1] if oks else ""
    # Approximate unique completes from OK lines (SKIP+OK overcount; use remote status if present)
    status = L8_MIRROR / "status.json"
    if status.is_file():
        try:
            data = json.loads(status.read_text())
            info["files"] = int(data.get("train_files") or 0)
            info["min_per_class"] = int(data.get("min_per_class") or 0)
        except Exception:
            pass
    return info


def slurm_l4() -> dict:
    jobs = squeue_jobs_by_name().get("jc-pretrain-L4", [])
    running = sum(1 for j in jobs if j["state"] == "RUNNING")
    pending = sum(1 for j in jobs if j["state"] == "PENDING")
    return {"jobs": jobs, "running": running, "pending": pending}


def sync_l8_mirror() -> dict:
    """Best-effort pull of L8 parquet / status from Lambda into the pretrain home."""
    info = {
        "reachable": False,
        "step": 0,
        "auc": None,
        "loss": None,
        "train_loss": None,
        "status": "unreachable",
        "train_files": 0,
        "ready": False,
        "gpus": "—",
        "curve_steps": [],
        "curve_auc": [],
        "tl_steps": [],
        "tl_vals": [],
        "note": "",
    }
    L8_MIRROR.mkdir(parents=True, exist_ok=True)
    agent = os.environ.get("SSH_AUTH_SOCK", "")
    env = os.environ.copy()
    # Probe
    probe = run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=12",
            "-o",
            "ControlMaster=no",
            LAMBDA_HOST,
            "bash -s",
        ],
        timeout=60,
    )
    # Use a richer remote script via stdin
    remote = r"""
python3 - <<'PY'
from pathlib import Path
from collections import Counter
import json
raw = Path("/home/ubuntu/data/jetclass/train_100M")
files = list(raw.glob("*.root")) if raw.is_dir() else []
by = Counter(p.name.rsplit("_", 1)[0] for p in files)
ready = len(by) >= 10 and min(by.values(), default=0) >= 21
root = Path.home() / "runs/giant_L8D1024/jetclass_output/adam/sweep_lr"
pqs = sorted(root.glob("*.parquet")) if root.is_dir() else []
step = 0
auc = loss = train = None
curve_s, curve_a = [], []
tl_s, tl_v = [], []
if pqs:
    try:
        import pyarrow.parquet as pq
        df = pq.read_table(pqs[-1]).to_pandas().sort_values("global_step")
        step = int(df["global_step"].max()) if len(df) else 0
        if "val_roc_auc" in df.columns:
            s = df["val_roc_auc"].dropna()
            if len(s):
                auc = float(s.iloc[-1])
            sub = df.dropna(subset=["val_roc_auc"])
            if len(sub) > 80:
                idx = list(range(0, len(sub), max(1, len(sub)//80)))
                idx.append(len(sub)-1)
                sub = sub.iloc[sorted(set(idx))]
            curve_s = [int(x) for x in sub["global_step"]]
            curve_a = [float(x) for x in sub["val_roc_auc"]]
        if "val_loss" in df.columns:
            s = df["val_loss"].dropna()
            if len(s):
                loss = float(s.iloc[-1])
        if "train_loss" in df.columns:
            sub = df.dropna(subset=["train_loss"])
            if len(sub):
                train = float(sub["train_loss"].iloc[-1])
                if len(sub) > 150:
                    idx = list(range(0, len(sub), max(1, len(sub)//150)))
                    idx.append(len(sub)-1)
                    sub = sub.iloc[sorted(set(idx))]
                tl_s = [int(x) for x in sub["global_step"]]
                tl_v = [float(x) for x in sub["train_loss"]]
    except Exception as e:
        pass
# Log fallback: parquet may lag until first val/ckpt flush.
import re
log_path = Path.home() / "logs/giant_L8D1024/L8_D1024_eta0p1_R0p1.out"
if log_path.is_file():
    try:
        text = log_path.read_text(errors="ignore")
        log_steps = [int(x) for x in re.findall(r"global_step=(\d+)", text)]
        if log_steps:
            step = max(step, max(log_steps))
        # Dense train_loss from custom logger: [train] step=N ... train_loss=X
        tl_map = {}
        for m in re.finditer(
            r"\[train\] step=(\d+)\s+epoch=\d+\s+train_loss=([0-9.eE+-]+)", text
        ):
            tl_map[int(m.group(1))] = float(m.group(2))
        if tl_map:
            items = sorted(tl_map.items())
            if train is None:
                train = items[-1][1]
            # Prefer log curve when denser / parquet empty
            if len(items) >= len(tl_s):
                if len(items) > 150:
                    idx = list(range(0, len(items), max(1, len(items)//150)))
                    idx.append(len(items)-1)
                    items = [items[i] for i in sorted(set(idx))]
                tl_s = [s for s, _ in items]
                tl_v = [v for _, v in items]
    except Exception:
        pass
# training process?
import subprocess
# Avoid pgrep matching this remote script's own argv (bash -s heredoc).
def _alive(substr: str) -> bool:
    import os
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode().replace("\0", " ")
        except Exception:
            continue
        if substr in cmd and "bash -s" not in cmd:
            return True
    return False
training = _alive("test_cpen_jetclass.py") or _alive("run_giant_L8_D1024_eta0p1_lambda8.sh")
watching = _alive("watch_and_launch_giant_L8D1024.sh")
if training:
    status = "training"
elif not ready:
    status = "staging"
elif watching:
    status = "ready"
else:
    status = "idle"
gpus = subprocess.getoutput(
    "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr '\\n' ',' | sed 's/,$//'"
) or "—"
print(json.dumps({
  "train_files": len(files),
  "min_per_class": (min(by.values()) if by else 0),
  "ready": ready,
  "step": step,
  "auc": auc,
  "loss": loss,
  "train_loss": train,
  "curve_steps": curve_s,
  "curve_auc": curve_a,
  "tl_steps": tl_s,
  "tl_vals": tl_v,
  "status": status,
  "gpus": gpus,
  "training": training,
  "watching": watching,
}))
PY
"""
    try:
        out = subprocess.check_output(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                "-o",
                "ControlMaster=no",
                LAMBDA_HOST,
                "bash -s",
            ],
            input=remote,
            text=True,
            timeout=90,
        )
    except Exception as e:
        info["note"] = f"ssh failed: {e}"
        # Fall back to mirror parquet if present
        for pq in sorted(L8_MIRROR.glob("*.parquet")):
            m = parquet_series(pq)
            info.update(
                {
                    "step": m["step"],
                    "auc": m["auc"],
                    "loss": m["loss"],
                    "train_loss": m["train_loss"],
                    "curve_steps": m["curve_steps"],
                    "curve_auc": m["curve_auc"],
                    "status": "mirror",
                }
            )
            break
        return info

    # last JSON line
    data = None
    for line in reversed(out.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                break
            except Exception:
                continue
    if not data:
        info["note"] = "no json from lambda"
        return info
    info["reachable"] = True
    info["train_files"] = int(data.get("train_files") or 0)
    info["ready"] = bool(data.get("ready"))
    info["step"] = int(data.get("step") or 0)
    info["auc"] = data.get("auc")
    info["loss"] = data.get("loss")
    info["train_loss"] = data.get("train_loss")
    info["curve_steps"] = data.get("curve_steps") or []
    info["curve_auc"] = data.get("curve_auc") or []
    info["tl_steps"] = data.get("tl_steps") or []
    info["tl_vals"] = data.get("tl_vals") or []
    info["status"] = str(data.get("status") or "unknown")
    info["gpus"] = str(data.get("gpus") or "—")
    info["note"] = f"min_per_class={data.get('min_per_class', 0)}"
    # Persist a tiny status snapshot for offline canvas refreshes
    (L8_MIRROR / "status.json").write_text(json.dumps(data, indent=2) + "\n")
    return info


def downsample_xy(steps: list, vals: list, n: int = 100) -> tuple[list, list]:
    if len(steps) <= n:
        return steps, vals
    idx = list(range(0, len(steps), max(1, len(steps) // n)))
    if idx[-1] != len(steps) - 1:
        idx.append(len(steps) - 1)
    return [steps[i] for i in idx], [vals[i] for i in idx]


def main() -> None:
    global _SQUEUE_CACHE
    _SQUEUE_CACHE = None
    l4_dir, l4_pq = find_l4()
    l4 = parquet_series(l4_pq) if l4_pq else {
        "step": 0,
        "auc": None,
        "loss": None,
        "acc": None,
        "train_loss": None,
        "curve_steps": [],
        "curve_auc": [],
        "curve_loss": [],
    }
    if l4_dir:
        l4["step"] = max(l4["step"], ckpt_step(l4_dir))
    # Slurm log is the freshest signal while parquet/ckpt flush lags a slot.
    log_step, log_auc, log_loss, log_train = log_step_l4()
    if log_step > l4["step"]:
        l4["step"] = log_step
        if log_auc is not None:
            l4["auc"] = log_auc
        if log_loss is not None:
            l4["loss"] = log_loss
        if log_train is not None:
            l4["train_loss"] = log_train
    slurm = slurm_l4()
    l4_state = (
        "done"
        if l4["step"] >= L4_MAX
        else ("running" if slurm["running"] else ("pending" if slurm["pending"] else "paused"))
    )
    l4_pct = 100.0 * l4["step"] / L4_MAX
    l4_epoch_frac = l4["step"] / L4_MAX

    # L4 cosine epoch-2 (max_steps=781250); share run dir with const fingerprint.
    l4c_dir, l4c_pq = find_l4_cos()
    l4c = parquet_series(l4c_pq) if l4c_pq else {
        "step": 0,
        "auc": None,
        "loss": None,
        "acc": None,
        "train_loss": None,
        "curve_steps": [],
        "curve_auc": [],
        "curve_loss": [],
    }
    if l4c_dir:
        l4c["step"] = max(l4c["step"], ckpt_step(l4c_dir))
    # Const ckpt is the resume baseline when cosine has not advanced yet.
    if l4_dir and l4["step"] > l4c["step"]:
        l4c["step"] = max(l4c["step"], min(l4["step"], L4_MAX))
        if l4c["auc"] is None:
            l4c["auc"] = l4["auc"]
        if l4c["loss"] is None:
            l4c["loss"] = l4["loss"]
        if l4c["train_loss"] is None:
            l4c["train_loss"] = l4["train_loss"]
    cos_step, cos_auc, cos_loss, cos_train, cos_tl = log_step_l4cos()
    if cos_step > l4c["step"]:
        l4c["step"] = cos_step
        if cos_auc is not None:
            l4c["auc"] = cos_auc
        if cos_loss is not None:
            l4c["loss"] = cos_loss
        if cos_train is not None:
            l4c["train_loss"] = cos_train
    slurm_cos = slurm_l4cos()
    l4c_state = (
        "done"
        if l4c["step"] >= L4_COS_MAX
        else (
            "running"
            if slurm_cos["running"]
            else ("pending" if slurm_cos["pending"] else ("paused" if l4c["step"] > L4_MAX else "queued"))
        )
    )
    # If const epoch finished and cosine is waiting, surface pending over "queued".
    if l4_state == "done" and l4c_state == "queued" and slurm_cos["pending"]:
        l4c_state = "pending"
    elif l4_state == "done" and l4c_state == "queued" and not slurm_cos["jobs"]:
        l4c_state = "paused"
    l4c_pct = 100.0 * l4c["step"] / L4_COS_MAX
    l4c_epoch_frac = l4c["step"] / L4_MAX  # epoch units relative to 1-epoch length
    cos_job_ids = ",".join(j["id"] for j in slurm_cos["jobs"]) or "—"

    # Prefer Della continue; skip slow Lambda SSH while Della job is active.
    della = della_l8_status()
    if della["status"] in {"training", "pending", "done"} or della["step"] > 0:
        l8 = {
            "reachable": True,
            "step": della["step"],
            "auc": della.get("auc"),
            "loss": None,
            "train_loss": della.get("train_loss"),
            "status": della["status"],
            "train_files": 210,
            "ready": True,
            "gpus": della.get("gpus", "—"),
            "curve_steps": [],
            "curve_auc": [],
            "tl_steps": della.get("tl_steps") or [],
            "tl_vals": della.get("tl_vals") or [],
            "note": della.get("note", ""),
        }
    else:
        l8 = sync_l8_mirror()
        local = local_l8_staging()
        if not l8["reachable"] and local["files"]:
            l8["train_files"] = local["files"]
            l8["status"] = (
                "staging" if local["min_per_class"] < 21 else l8.get("status", "staging")
            )
            l8["note"] = (
                f"local stage log · min_per_class={local['min_per_class']} last={local['last']}"
            )
        elif l8["reachable"] and local["files"] > l8["train_files"]:
            l8["train_files"] = local["files"]
        if della["step"] > l8.get("step", 0):
            l8["step"] = della["step"]
            l8["status"] = della["status"]
            l8["train_loss"] = della.get("train_loss")
            l8["tl_steps"] = della.get("tl_steps") or []
            l8["tl_vals"] = della.get("tl_vals") or []
            l8["note"] = della.get("note", l8.get("note", ""))
            l8["reachable"] = True
            l8["gpus"] = della.get("gpus", l8.get("gpus", "—"))
    # Always surface staging file count even before training
    if l8["status"] in {"staging", "ready", "idle"} and not l8["step"]:
        if not l8.get("note"):
            l8["note"] = f"min_per_class≈{local.get('min_per_class', '?')}"
    l8_pct = 100.0 * l8["step"] / L8_MAX if L8_MAX else 0.0

    # Curves — use step as category labels (stringified) for LineChart
    c_steps, c_auc = downsample_xy(l4["curve_steps"], l4["curve_auc"], 100)
    cats = [str(s) for s in c_steps]
    auc_series = c_auc

    l8_cats = [str(s) for s in l8["curve_steps"]]
    l8_auc = l8["curve_auc"]

    updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    job_ids = ",".join(j["id"] for j in slurm["jobs"]) or "—"

    # Progress table
    table = [
        [
            "L4/D256 const",
            "Della gputest",
            l4_state,
            f"{l4['step']}/{L4_MAX}",
            f"{l4_pct:.1f}%",
            f"{l4_epoch_frac:.3f} ep",
            f"{l4['auc']:.4f}" if l4["auc"] is not None else "—",
            f"{l4['train_loss']:.4f}" if l4["train_loss"] is not None else "—",
            job_ids,
        ],
        [
            "L4/D256 cosine",
            "Della gputest",
            l4c_state,
            f"{l4c['step']}/{L4_COS_MAX}",
            f"{l4c_pct:.1f}%",
            f"{l4c_epoch_frac:.3f} ep",
            f"{l4c['auc']:.4f}" if l4c["auc"] is not None else "—",
            f"{l4c['train_loss']:.4f}" if l4c["train_loss"] is not None else "—",
            cos_job_ids,
        ],
        [
            "L8/D1024",
            "Della gputest",
            l8["status"],
            f"{l8['step']}/{L8_MAX}",
            f"{l8_pct:.1f}%",
            f"{l8['train_files']}/210 files",
            f"{l8['auc']:.4f}" if l8["auc"] is not None else "—",
            f"{l8['train_loss']:.4f}" if l8["train_loss"] is not None else "—",
            l8.get("gpus", "—"),
        ],
    ]

    has_l4_curve = len(cats) >= 1
    has_l8_curve = len(l8_cats) >= 1
    # Prefer denser train_loss history for L4 when val AUC is sparse
    l4_tl_map: dict[int, float] = {}
    for pq_path in (l4_pq, l4c_pq):
        if not pq_path or not pq_path.is_file():
            continue
        try:
            import pyarrow.parquet as pqmod
            df = pqmod.read_table(pq_path).to_pandas().sort_values("global_step")
            if "train_loss" in df.columns:
                sub = df.dropna(subset=["train_loss"])
                for _, row in sub.iterrows():
                    l4_tl_map[int(row["global_step"])] = float(row["train_loss"])
        except Exception:
            pass
    for s, v in cos_tl:
        l4_tl_map[int(s)] = float(v)
    l4_tl_steps = sorted(l4_tl_map)
    l4_tl_raw = [l4_tl_map[s] for s in l4_tl_steps]
    l4_tl_steps, l4_tl_raw = downsample_xy(l4_tl_steps, l4_tl_raw, n=150)
    l4_tl_cats = [str(int(x)) for x in l4_tl_steps]
    l4_tl_vals = [float(x) for x in l4_tl_raw]
    has_l4_train = len(l4_tl_cats) >= 2

    # Cosine-only train loss (epoch-2 region) for a dedicated chart
    l4c_tl_map: dict[int, float] = {
        int(s): float(v) for s, v in cos_tl if int(s) >= L4_MAX - 1000
    }
    if l4c_pq and l4c_pq.is_file():
        try:
            import pyarrow.parquet as pqmod
            df = pqmod.read_table(l4c_pq).to_pandas().sort_values("global_step")
            if "train_loss" in df.columns:
                sub = df.dropna(subset=["train_loss"])
                sub = sub[sub["global_step"] >= L4_MAX - 1000]
                for _, row in sub.iterrows():
                    l4c_tl_map[int(row["global_step"])] = float(row["train_loss"])
        except Exception:
            pass
    l4c_tl_steps = sorted(l4c_tl_map)
    l4c_tl_raw = [l4c_tl_map[s] for s in l4c_tl_steps]
    l4c_tl_steps, l4c_tl_raw = downsample_xy(l4c_tl_steps, l4c_tl_raw, n=100)
    l4c_tl_cats = [str(int(x)) for x in l4c_tl_steps]
    l4c_tl_vals = [float(x) for x in l4c_tl_raw]
    has_l4c_train = len(l4c_tl_cats) >= 2

    l8_tl_steps = list(l8.get("tl_steps") or [])
    l8_tl_vals = list(l8.get("tl_vals") or [])
    if len(l8_tl_steps) != len(l8_tl_vals):
        n = min(len(l8_tl_steps), len(l8_tl_vals))
        l8_tl_steps, l8_tl_vals = l8_tl_steps[:n], l8_tl_vals[:n]
    l8_tl_steps, l8_tl_vals = downsample_xy(l8_tl_steps, l8_tl_vals, n=150)
    l8_tl_cats = [str(int(x)) for x in l8_tl_steps]
    has_l8_train = len(l8_tl_cats) >= 2

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
const L4 = {{
  state: {js(l4_state)},
  step: {l4["step"]},
  max: {L4_MAX},
  seed: {L4_SEED_STEP},
  pct: {l4_pct:.4f},
  auc: {js(l4["auc"])},
  loss: {js(l4["loss"])},
  train_loss: {js(l4["train_loss"])},
  jobs: {js(job_ids)},
}};
const L4C = {{
  state: {js(l4c_state)},
  step: {l4c["step"]},
  max: {L4_COS_MAX},
  pct: {l4c_pct:.4f},
  auc: {js(l4c["auc"])},
  loss: {js(l4c["loss"])},
  train_loss: {js(l4c["train_loss"])},
  jobs: {js(cos_job_ids)},
}};
const L8 = {{
  state: {js(l8["status"])},
  step: {l8["step"]},
  max: {L8_MAX},
  pct: {l8_pct:.4f},
  auc: {js(l8["auc"])},
  loss: {js(l8["loss"])},
  train_loss: {js(l8["train_loss"])},
  files: {l8["train_files"]},
  ready: {js(l8["ready"])},
  gpus: {js(l8.get("gpus", "—"))},
  note: {js(l8.get("note", ""))},
  reachable: {js(l8["reachable"])},
}};
const TABLE = {js(table)};
const L4_CATS: string[] = {js(cats)};
const L4_AUC: (number | null)[] = {js(auc_series)};
const L4_TL_CATS: string[] = {js(l4_tl_cats)};
const L4_TL: (number | null)[] = {js(l4_tl_vals)};
const L4C_TL_CATS: string[] = {js(l4c_tl_cats)};
const L4C_TL: (number | null)[] = {js(l4c_tl_vals)};
const L8_CATS: string[] = {js(l8_cats)};
const L8_AUC: (number | null)[] = {js(l8_auc)};
const L8_TL_CATS: string[] = {js(l8_tl_cats)};
const L8_TL: (number | null)[] = {js(l8_tl_vals)};
const HAS_L4_CURVE = {js(has_l4_curve)};
const HAS_L4_TRAIN = {js(has_l4_train)};
const HAS_L4C_TRAIN = {js(has_l4c_train)};
const HAS_L8_CURVE = {js(has_l8_curve)};
const HAS_L8_TRAIN = {js(has_l8_train)};

export default function JetclassPretrainLive() {{
  const l4Series = [
    {{
      name: "L4/D256 val ROC",
      data: L4_AUC.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "accent" as const,
    }},
  ];
  const l4TrainSeries = [
    {{
      name: "L4/D256 train loss",
      data: L4_TL.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "warning" as const,
    }},
  ];
  const l4cTrainSeries = [
    {{
      name: "L4 cosine train loss",
      data: L4C_TL.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "accent" as const,
    }},
  ];
  const l8TrainSeries = [
    {{
      name: "L8/D1024 train loss",
      data: L8_TL.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "info" as const,
    }},
  ];
  const l8Series = [
    {{
      name: "L8/D1024 val ROC",
      data: L8_AUC.map((v) => (v == null ? Number.NaN : Number(v))),
      tone: "info" as const,
    }},
  ];

  return (
    <Stack gap={{20}}>
      <Stack gap={{6}}>
        <H1>JetClass pretrain — L4 epoch + L8 giant</H1>
        <Text tone="secondary">
          Home: jetclass_pretrain · L4 const → 390625 then cosine → 781250
          (η→0.01×, frozen for FT) · L8 epoch-2 → 160k @ B_eff=128 on Della · updated {{UPDATED}}
        </Text>
        <Row gap={{8}} wrap>
          <Pill tone={{L4.state === "done" ? "success" : L4.state === "running" ? "info" : "warning"}}>
            L4 const {{L4.state}}
          </Pill>
          <Pill tone={{L4C.state === "done" ? "success" : L4C.state === "running" ? "info" : "warning"}}>
            L4 cosine {{L4C.state}}
          </Pill>
          <Pill tone={{L8.state === "training" ? "info" : L8.state === "staging" ? "warning" : "neutral"}}>
            L8 {{L8.state}}
          </Pill>
          <Pill tone="neutral">k★=7 R★=0.1 noloop</Pill>
        </Row>
      </Stack>

      <Grid columns={{4}} gap={{12}}>
        <Stat
          value={{`${{L4.step.toLocaleString()}}`}}
          label={{`L4 const / ${{L4.max.toLocaleString()}}`}}
          tone={{L4.state === "done" ? "success" : "info"}}
        />
        <Stat
          value={{`${{L4C.step.toLocaleString()}}`}}
          label={{`L4 cosine / ${{L4C.max.toLocaleString()}}`}}
          tone={{L4C.state === "running" ? "info" : L4C.state === "done" ? "success" : "warning"}}
        />
        <Stat
          value={{L4C.auc == null ? (L4.auc == null ? "—" : L4.auc.toFixed(4)) : L4C.auc.toFixed(4)}}
          label="L4 latest val ROC"
        />
        <Stat
          value={{`${{L8.step.toLocaleString()}}`}}
          label={{`L8 step / ${{L8.max.toLocaleString()}}`}}
          tone={{L8.state === "training" ? "info" : "neutral"}}
        />
      </Grid>

      <Callout
        tone={{L4.state === "running" ? "info" : "neutral"}}
        title="L4/D256 const epoch"
      >
        Seeded from edge η₀=0.1 @ step {{L4.seed}}. Target {{L4.max.toLocaleString()}} steps
        = one unique pass over 100M jets (B=64×4). Jobs: {{L4.jobs}}. Progress {{L4.pct.toFixed(1)}}%.
      </Callout>

      <Callout
        tone={{L4C.state === "running" ? "info" : L4C.state === "pending" ? "warning" : "neutral"}}
        title="L4/D256 cosine epoch-2"
      >
        Resume from const ckpt · max_steps={{L4C.max.toLocaleString()}} · cosine over
        390625 steps to 0.01× peak. Jobs: {{L4C.jobs}}. Progress {{L4C.pct.toFixed(1)}}%
        · train_loss {{L4C.train_loss == null ? "—" : L4C.train_loss.toFixed(4)}}.
      </Callout>

      <Callout
        tone={{L8.reachable ? (L8.ready ? "info" : "warning") : "danger"}}
        title="L8/D1024 Della continue"
      >
        {{L8.reachable
          ? `${{L8.note || "Della gputest"}}. GPUs: ${{L8.gpus}}.`
          : `Unreachable — ${{L8.note || "check job/logs"}}.`}}
        Epoch-2 const LR → 160k (epoch-1 done @80k, JetClass val AUC 0.9567).
      </Callout>

      <Stack gap={{6}}>
        <H2>Run status</H2>
        <Table
          headers={{[
            "model",
            "where",
            "state",
            "step",
            "%",
            "detail",
            "val ROC",
            "train loss",
            "job / GPUs",
          ]}}
          rows={{TABLE}}
        />
      </Stack>

      {{HAS_L4_TRAIN ? (
        <Stack gap={{6}}>
          <H2>L4/D256 train loss vs step</H2>
          <Text tone="secondary">
            Source: const + cosine parquet/logs · includes epoch-2 when available
          </Text>
          <LineChart
            categories={{L4_TL_CATS}}
            series={{l4TrainSeries as any}}
            height={{260}}
            beginAtZero={{false}}
          />
        </Stack>
      ) : null}}

      {{HAS_L4C_TRAIN ? (
        <Stack gap={{6}}>
          <H2>L4 cosine epoch-2 train loss</H2>
          <Text tone="secondary">
            Steps near/after {{L4.max.toLocaleString()}} · jc-pretrain-L4cos logs + cosine parquet
          </Text>
          <LineChart
            categories={{L4C_TL_CATS}}
            series={{l4cTrainSeries as any}}
            height={{220}}
            beginAtZero={{false}}
          />
        </Stack>
      ) : null}}

      {{HAS_L8_TRAIN ? (
        <Stack gap={{6}}>
          <H2>L8/D1024 train loss vs step</H2>
          <Text tone="secondary">
            Source: Della train log · logged every 50 steps · target {{L8.max.toLocaleString()}} steps
          </Text>
          <LineChart
            categories={{L8_TL_CATS}}
            series={{l8TrainSeries as any}}
            height={{260}}
            beginAtZero={{false}}
          />
        </Stack>
      ) : null}}

      {{HAS_L4_CURVE ? (
        <Stack gap={{6}}>
          <H2>L4/D256 val ROC vs step</H2>
          <Text tone="secondary">
            Source: jetclass_pretrain parquet · val every 1000 steps after resume
          </Text>
          <LineChart
            categories={{L4_CATS}}
            series={{l4Series as any}}
            height={{220}}
          />
        </Stack>
      ) : null}}

      {{HAS_L8_CURVE ? (
        <Stack gap={{6}}>
          <H2>L8/D1024 val ROC vs step</H2>
          <Text tone="secondary">Source: Lambda giant parquet (mirrored)</Text>
          <LineChart
            categories={{L8_CATS}}
            series={{l8Series as any}}
            height={{260}}
          />
        </Stack>
      ) : null}}
    </Stack>
  );
}}
'''
    OUT.write_text(canvas)
    print(
        f"wrote {OUT} L4={l4_state}@{l4['step']} "
        f"L4cos={l4c_state}@{l4c['step']} L8={l8['status']}@{l8['step']}"
    )


if __name__ == "__main__":
    main()
