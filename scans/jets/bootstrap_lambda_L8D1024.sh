#!/bin/bash
# Stage code + unique JetClass ROOTs onto Lambda 168.110.21.94, then launch
# L=8/D=1024 d_h=64 η₀=0.1 @ B=64×8 → 40k single-pass unique jets.
#
# Usage (from Della, with SSH agent that can reach the box):
#   export SSH_AUTH_SOCK=...   # if needed
#   bash scans/jets/bootstrap_lambda_L8D1024.sh
#
# Env overrides:
#   LAMBDA_HOST=ubuntu@168.110.21.94
#   FILES_PER_CLASS=22          # ≥21 needed for 20.48M unique
#   SKIP_DATA=0|1  SKIP_CODE=0|1  SKIP_DEPS=0|1  SKIP_LAUNCH=0|1
#
set -euo pipefail

HOST="${LAMBDA_HOST:-ubuntu@168.110.21.94}"
REPO_LOCAL="${REPO_LOCAL:-$HOME/coupled-particle-edge-networks}"
JET_LOCAL="${JET_LOCAL:-/projects/j/jdezoort/jetclass}"
FILES_PER_CLASS="${FILES_PER_CLASS:-22}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 "$HOST")
RSYNC_SSH="ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30"

SKIP_DATA="${SKIP_DATA:-0}"
SKIP_CODE="${SKIP_CODE:-0}"
SKIP_DEPS="${SKIP_DEPS:-0}"
SKIP_LAUNCH="${SKIP_LAUNCH:-0}"

CLASSES=(HToBB HToCC HToGG HToWW2Q1L HToWW4Q TTBar TTBarLep WToQQ ZJetsToNuNu ZToQQ)
RAW_REMOTE="/lambda/nfs/jetclass"

echo "$(date -Iseconds) bootstrap → $HOST files/class=$FILES_PER_CLASS"

"${SSH[@]}" "mkdir -p $RAW_REMOTE/train_100M $RAW_REMOTE/val_5M ~/cpen ~/runs ~/logs ~/logs/giant_L8D1024"

if [[ "$SKIP_CODE" != "1" ]]; then
  echo "$(date -Iseconds) rsync code → ~/cpen"
  rsync -az --delete \
    --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'jetclass_output' --exclude 'runs' --exclude '.venv' \
    --exclude 'paper/figures/*.png' --exclude 'paper/figures/*.pdf' \
    -e "$RSYNC_SSH" \
    "$REPO_LOCAL/" "$HOST:~/cpen/"
fi

if [[ "$SKIP_DEPS" != "1" ]]; then
  echo "$(date -Iseconds) install python deps (user site)"
  "${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
python3 -m pip install --user -q --upgrade pip
# torch already present on image; install the rest
python3 -m pip install --user -q \
  "lightning>=2.0" "torch-geometric>=2.4" "pandas>=2.0" "pyarrow>=14.0" \
  "numpy>=1.24" "scikit-learn>=1.3" "tables>=3.8" \
  uproot awkward vector einops tqdm rich pyyaml fsspec packaging
# optional: jetnet can be heavy/fail; not required for JetClass stream path
python3 - <<'PY'
import torch, lightning, pyarrow, uproot
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
print("lightning", lightning.__version__, "pyarrow", pyarrow.__version__, "uproot", uproot.__version__)
PY
REMOTE
fi

if [[ "$SKIP_DATA" != "1" ]]; then
  echo "$(date -Iseconds) stage train: ${FILES_PER_CLASS}/class (=$((FILES_PER_CLASS*10)) files, ≥20.48M unique)"
  LIST=$(mktemp)
  for c in "${CLASSES[@]}"; do
    for ((i=0; i<FILES_PER_CLASS; i++)); do
      idx=$(printf '%03d' "$i")
      f="${c}_${idx}.root"
      if [[ ! -f "$JET_LOCAL/train_100M/$f" ]]; then
        echo "missing $JET_LOCAL/train_100M/$f" >&2
        exit 1
      fi
      echo "$f" >> "$LIST"
    done
  done
  echo "$(date -Iseconds) rsync $(wc -l < "$LIST") train ROOT files"
  rsync -a --info=progress2 --partial --files-from="$LIST" \
    -e "$RSYNC_SSH" \
    "$JET_LOCAL/train_100M/" "$HOST:$RAW_REMOTE/train_100M/"
  rm -f "$LIST"

  echo "$(date -Iseconds) stage val_5M (all)"
  rsync -a --info=progress2 --partial \
    -e "$RSYNC_SSH" \
    "$JET_LOCAL/val_5M/" "$HOST:$RAW_REMOTE/val_5M/"

  "${SSH[@]}" "echo TRAIN=\$(ls $RAW_REMOTE/train_100M/*.root 2>/dev/null | wc -l); echo VAL=\$(ls $RAW_REMOTE/val_5M/*.root 2>/dev/null | wc -l); du -sh $RAW_REMOTE/train_100M $RAW_REMOTE/val_5M"
fi

if [[ "$SKIP_LAUNCH" != "1" ]]; then
  echo "$(date -Iseconds) launch giant L8/D1024"
  "${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
mkdir -p ~/logs/giant_L8D1024
cd ~/cpen
nohup bash scans/jets/run_giant_L8_D1024_eta0p1_lambda8.sh \
  > ~/logs/giant_L8D1024/nohup_launch.out 2>&1 &
echo "pid=$!"
sleep 2
tail -30 ~/logs/giant_L8D1024/nohup_launch.out || true
ps -u ubuntu -o pid,etime,cmd | grep -E 'run_giant_L8|test_cpen_jetclass' | grep -v grep || true
REMOTE
fi

echo "$(date -Iseconds) bootstrap done"
