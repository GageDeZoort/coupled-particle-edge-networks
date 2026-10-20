#!/bin/bash
# Wait until 21 files/class (unique 40k @ B_eff=512) are on disk, then launch.
# Stream plan_shards is fixed at dataset init → need the full 40k slice before start.
#
#   nohup bash scans/jets/watch_and_launch_giant_L8D1024.sh > ~/logs/giant_L8D1024/watch.out 2>&1 &
set -euo pipefail

RAW="${JETCLASS_RAW:-/home/ubuntu/data/jetclass}"
NEED_PER_CLASS="${NEED_PER_CLASS:-21}"
NEED_CLASSES="${NEED_CLASSES:-10}"
REPO="${HOME}/cpen"
LOGDIR="${HOME}/logs/giant_L8D1024"
mkdir -p "$LOGDIR"

echo "$(date -Iseconds) watcher start need ${NEED_PER_CLASS}/class × ${NEED_CLASSES} classes raw=$RAW (unique 40k train-only)"

while true; do
  set +e
  out=$(python3 - <<PY
from pathlib import Path
from collections import Counter
raw = Path("${RAW}") / "train_100M"
files = list(raw.glob("*.root")) if raw.is_dir() else []
by = Counter(p.name.rsplit("_", 1)[0] for p in files)
need_n = int("${NEED_PER_CLASS}")
wanted = {
    "HToBB","HToCC","HToGG","HToWW2Q1L","HToWW4Q",
    "TTBar","TTBarLep","WToQQ","ZJetsToNuNu","ZToQQ",
}
ok = all(by.get(c, 0) >= need_n for c in wanted)
print(f"train={len(files)} per_class={dict(sorted(by.items()))} ready={ok}")
raise SystemExit(0 if ok else 1)
PY
)
  rc=$?
  set -e
  echo "$(date -Iseconds) $out"
  if (( rc == 0 )); then
    echo "$(date -Iseconds) unique-40k balanced slice ready — launching"
    break
  fi
  sleep 60
done

if pgrep -f 'run_giant_L8_D1024_eta0p1_lambda8.sh|test_cpen_jetclass.py' >/dev/null; then
  echo "$(date -Iseconds) already running — exit"
  exit 0
fi

cd "$REPO"
nohup bash scans/jets/run_giant_L8_D1024_eta0p1_lambda8.sh \
  >"$LOGDIR/nohup_launch.out" 2>&1 &
echo "$(date -Iseconds) launched pid=$!"
