# Coupled Particle-Edge Networks (CPEN)

Research codebase for **CAPEN** (coupled particle–edge attention networks) on
jet tagging, with hyperparameter-transfer / scaling-law experiments on
**JetClass**. Secondary apps (stellar streams, PascalVOC-SP, MNIST) live under
`scans/` but are not the active focus documented here.

This README covers the **tools we currently use** on Della and how to run them.
Older one-off Slurm scripts under `scans/jets/` that are not listed below are
legacy; prefer the recipes here.

**Public front-door draft:** [`README.public.md`](README.public.md) ·
**Org / public-tree checklist:** [`docs/PUBLIC_TREE_CHECKLIST.md`](docs/PUBLIC_TREE_CHECKLIST.md)

---

## Install (Della)

```bash
module load anaconda3/2023.9
conda activate mamba-env
cd /home/jdezoort/coupled-particle-edge-networks
pip install -e ".[dev]"   # once per env update
```

Training entrypoint imports live under `src/cpen/` and `scans/common/`.
All JetClass Slurm scripts `cd` into `scans/jets/` and call
`test_cpen_jetclass.py`.

---

## Layout (what matters)

```
src/cpen/
  models/capen_llama_att.py   Production model: capen-llama-att
  graphs/                     Incidence / kNN / star-R primitives
  training/                   Lightning module, schedulers, parquet logging
  apps/jets/                  JetClass + TopTagging data paths

scans/
  common/sweep_common.py      Shared CLI + run-dir naming + sweep driver
  jets/
    test_cpen_jetclass.py     JetClass training entrypoint
    GRAPH_CONSTRUCTION_RECIPE.md
    scaling_law_L4_graph_k_sweep_gputest.slurm   # plot-1 k sweep (active)
    scaling_law_L4_D256_wfc_gputest.slurm        # AdamW WFC (paused)
    L4_D256_wfc_gputest_watchdog.sh
    start_L4_D256_wfc_gputest_watchdogs.sh
    graph_k_live_summary.sh

paper/
  data/collect_jetclass_paper_metrics.py
  style/plot_jetclass_paper.py
  figures/                    Active paper figures
  archive/                    Frozen outdated graph-construction plots/CSVs
```

---

## Paths on Della

| Role | Path |
|------|------|
| Repo | `/home/jdezoort/coupled-particle-edge-networks` |
| JetClass ROOT (stream) | `/projects/j/jdezoort/jetclass` |
| Dataset / cache root | `/scratch/gpfs/BHANIN/jdezoort/datasets/jetclass` |
| Run root | `/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law` |
| Logs | `…/jetclass_scaling_law/logs/` |
| Adam runs | `…/jetclass_output/adam/sweep_lr/` |
| AdamW runs | `…/jetclass_output/adamw/sweep_lr/` |
| Archived graph ablations | `…/archive/graph_construction_2026-09-28/` |

Run directories are named from architecture + graph recipe + `--run-tag`.
Re-submitting the **same** name resumes from `last.ckpt`.

---

## Core training tool: `test_cpen_jetclass.py`

```bash
cd scans/jets
python test_cpen_jetclass.py --help
```

CLI is assembled in `scans/common/sweep_common.py`. Flags we use every day:

| Flag | Typical value | Meaning |
|------|---------------|---------|
| `--model` | `capen-llama-att` | Production CAPEN-Llama + edge attention |
| `--operators` | `incidence` | Incidence-based message passing |
| `--jetclass-stream` | on | Build graphs live from ROOT (no prebuilt cache) |
| `--jetclass-raw-root` | `/projects/j/jdezoort/jetclass` | ROOT location |
| `--jetclass-features` | `kin7` | 7 kinematic particle features |
| `--jetclass-edge-features` | `part-interaction` | ParT pairwise edge features |
| `--live-knn-k` | `2…10` | Directed ΔR-kNN 2-edges |
| `--star-radius` | e.g. `0.125` | Star-\(R_\star\) hyperedges |
| `--no-star-hyperedges` | optional | kNN only (still pass `--star-radius` for feature scale) |
| `--all-edge-m22` | **off** | Default = M22 on **hyperedges only**; M12/M21 on all edges |
| `--optimizer` | `adam` or `adamw` | |
| `--etas` / `--eta0` | `0.25` | Base LR \(\eta_0\) (μP: \(\eta=\eta_0/\sqrt{D}\)) |
| `--t-epoch` | `1` (Adam) or WFC length | Schedule length for AdamW cosine |
| `--scheduler` | (WFC only) | `warmup_flat_cosine` via Slurm script |
| `--max-steps` | e.g. `2000` | Hard step budget |
| `--n-train` | `1000000` | Stream cap / unique-pass size |
| `--batch-size` | `128` | Per-GPU batch |
| `--n-gpus` | `4` | DDP world size → \(B_\mathrm{global}=B\times n_\mathrm{gpus}\) |
| `--depth` / `--width` / `--heads` | `4` / `256` / `8` | \(L\), \(D\), \(H\) (\(d_\mathrm{head}=D/H\)) |
| `--run-tag` | recipe string | Disambiguates run dirs |
| `--root` | scratch run root | Parent of `jetclass_output/…` |
| `--val-check-interval` | `500` | Val every N optimizer steps |
| `--checkpoint-every-n-steps` | `500` | Write `last.ckpt` / `best.ckpt` |
| `--no-torch-compile` | on | Stable on Della |

**Attention convention (default `capen-llama-att`):**

- **M12 / M21** — all live edges (kNN 2-edges + star hyperedges)
- **M22** — hyperedges only (deg > 2), unless `--all-edge-m22`

Metrics land in a sidecar parquet next to the run directory and in Slurm `.out`
as `[val] … val_roc_auc=…`.

---

## Matched compute budget (gpu-test)

For graph-construction comparisons we **fix batch size and step count**:

| Knob | Value |
|------|-------|
| Partition | Della `gputest` (request `a100` + `--time=0:59:00`; do **not** set `--partition=gputest`) |
| GPUs | 4×A100 |
| \(B\) per GPU | 128 → **\(B_\mathrm{global}=512\)** |
| Steps | **2000** (~1.024M jet-updates) |
| Model | \(L{=}4\), \(D{=}256\), \(H{=}8\), \(\eta_0{=}0.25\), Adam |
| Data | kin7 + part-interaction, \(p{=}80\), live stream |

Exclude bad nodes (as in current Slurm): `--exclude=della-l02g6,della-l03g11`.

Do **not** mix results from different \(B\) or step budgets on the same plot
axes. Older mismatched ablations are under
`paper/archive/graph_construction_2026-09-28/` and
`…/archive/graph_construction_2026-09-28/` on scratch.

---

## Graph construction plots

Detailed plan: [`scans/jets/GRAPH_CONSTRUCTION_RECIPE.md`](scans/jets/GRAPH_CONSTRUCTION_RECIPE.md).

### Plot 1 — kNN \(k\) at fixed \(R_\star=0.125\)

```bash
sbatch scans/jets/scaling_law_L4_graph_k_sweep_gputest.slurm
# array 0–4 → k = 2, 4, 6, 8, 10
```

Monitor:

```bash
squeue -u $USER -n jc-graph-k
ARRAY=<jobid> bash scans/jets/graph_k_live_summary.sh
# logs: …/logs/jc_graph_k_<ARRAY>_<task>.{out,err}
```

Typical speed (live graph, 4×A100): ~210–350 ms/step depending on \(k\) →
full 2k steps in ~7–12 min per arm. QOS often allows only a few gputest jobs
at once; the rest stay `PENDING (QOSMaxJobsPerUserLimit)`.

### Plot 2 — \(R_\star\) at best \(k\) (after plot 1)

Planned \(R_\star \in \{0.05, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225\}\),
same budget and M22 hyperedge-only rule. Script TBD once \(k^\star\) is chosen.

### Plot 3 — LR × width vs M11-only

At best \((k, R_\star)\): \(\eta_0 \in \{0.05, 0.1, 0.25, 0.5, 1\}\) for
**edges** vs **M11-only**, at \((D{=}256, H{=}8)\) and \((D{=}384, H{=}12)\).
Script TBD.

---

## AdamW warmup → flat → cosine (WFC)

Long JetClass unique-pass runs with AdamW + `warmup_flat_cosine`
(10k warmup + 20k flat + cosine). Currently **paused** so graph construction
can be fixed first; tooling remains:

| Script | Role |
|--------|------|
| `scaling_law_L4_D256_wfc_gputest.slurm` | One 59 min chunk (`T_EPOCH`, `ETA0`, `BATCH=128`, 4 GPUs) |
| `L4_D256_wfc_gputest_watchdog.sh` | Resubmits until `TARGET` steps |
| `start_L4_D256_wfc_gputest_watchdogs.sh` | Starts watchdogs for active \(t\) arms |

```bash
# Active arms only: t ∈ {5, 10, 25}, TARGET=150000
bash scans/jets/start_L4_D256_wfc_gputest_watchdogs.sh

# Stop one arm
touch ~/.cache/jc_L4D256_wfc_watchdogs/STOP_t5
# Stop all WFC watchdogs: kill their PIDs (see start script) + STOP_t{5,10,25}
```

Retired short warmups \(t \in \{0.1, 0.25, 0.5, 1, 2\}\) are refused by both the
watchdog and the Slurm script. Watchdog logs:
`~/.cache/jc_L4D256_wfc_watchdogs/watchdog_t*.log`.

**Never** `pkill -f L4_D256_wfc_gputest_watchdog.sh` from a shell whose command
line contains that string (it matches itself). Kill by PID instead.

---

## Paper figures

```bash
conda activate mamba-env
cd /home/jdezoort/coupled-particle-edge-networks
python paper/data/collect_jetclass_paper_metrics.py   # rebuild CSVs from parquets/logs
python paper/style/plot_jetclass_paper.py             # write paper/figures/*
```

Style helpers: `paper/style/nature_plots.py` (`use_nature`, `save_paper_figure`).
See [`paper/README.md`](paper/README.md).

**Active figure today:** `fig_jetclass_adam_transfer_45k` (η₀ transfer @ 45k).
Graph-construction figures/CSVs from the mismatched-B campaign are under
`paper/archive/graph_construction_2026-09-28/` — regenerate after the new
matched sweeps finish; update `collect_jetclass_paper_metrics.py` to point at
the `graph-k-…` run tags.

---

## Watchdogs (pattern)

Long campaigns on gputest (~59 min) use a login-node bash loop:

1. Read latest `global_step` from the run parquet / ckpt.
2. If below `TARGET` and no job is queued, `sbatch` the Slurm script.
3. Sleep (`SLEEP_SEC`, often 900), repeat.
4. Exit if `STOP_t*` / `STOP_ALL` appears under `~/.cache/jc_*_watchdogs/`.

Start helpers: `scans/jets/start_L4_*_watchdogs.sh`. Prefer those over
hand-rolled `nohup` so STOP sentinels and PIDs stay consistent.

---

## Day-to-day ops

```bash
# Queue (JetClass jobs are named jc-*)
squeue -u $USER -o '%.14i %.28j %.10P %.8T %.10M %R'

# Cancel all JetClass jobs (leave other projects alone)
squeue -u $USER -h -o '%i %j' | awk '/ jc-/ {print $1}' | xargs -r scancel

# Tail a running graph-k task
tail -f /scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/jc_graph_k_<ARRAY>_<task>.out

# Latest val lines
rg '\[val\]' …/jc_graph_k_<ARRAY>_*.out
```

Resume after preemption: re-`sbatch` the **same** script / env so the run-dir
name matches; Lightning loads `last.ckpt`.

---

## Smoke test (optional, local/CPU)

```bash
python scans/jets/test_cpen_toptagging.py \
  --mode sweep_lr --etas 0.3 \
  --depth 4 --width 64 --epochs 2 \
  --batch-size 32 --n-gpus 0 --root ./runs
```

JetClass live-stream needs GPUs + ROOT access; use the Slurm recipes above.

---

## Other apps (not active focus)

| App | Entrypoint |
|-----|------------|
| TopTagging | `scans/jets/test_cpen_toptagging.py`, `FINETUNE.md` |
| Stellar streams | `scans/streams/test_cpen_stream.py` |
| PascalVOC-SP | `scans/pascal/` |
| MNIST | `scans/mnist/README.md` |

---

## Design rules we enforce

1. **Matched budgets** for any plot that compares graph recipes: same
   \(B_\mathrm{global}\), same `max_steps`, same \(\eta_0\) / \(L\) / \(D\) / \(H\)
   unless the axis *is* that knob.
2. **One knob at a time** for graph construction (first \(k\), then \(R_\star\),
   then LR / M11).
3. **Default M22 = hyperedge-only**; do not pass `--all-edge-m22` unless the
   experiment is explicitly “full M22”.
4. **Archive, don’t overwrite** when a campaign is invalidated (batch confound,
   wrong recipe): move runs + figures into dated `archive/` folders and update
   the recipe doc.
