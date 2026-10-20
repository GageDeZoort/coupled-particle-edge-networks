# Graph construction panel (b) — CompleteP v2 @ k★=3

**Status:** **complete** — job **15295488**, **30/30** · peak **R★=0.1** (0.9427 ± 0.0014)  
**Note:** ran at self-included k=3. R★ peak likely robust; \(k\) label for OP
deferred to noloop panel (a′).  
**Do not mix** archived Lambda `v1lam` panel-(b) parquets into this campaign.

Parent: `paper/figures/fig_graph_construction.md`  
Panel (a) self-incl.: `fig_graph_construction_k_R0p1_completep_v2.md` · noloop: `fig_graph_construction_k_R0p1_noloop_completep_v2.md`

---

## Specs (locked; matched to panel a v2cp)

| Item | Value |
|---|---|
| Model | `capen-llama-att` — attentional pool + class unembed (CompleteP) |
| Edges | part-interaction + incidence; star hyperedges on |
| Geometry | \(k^\star=3\); \(R^\star\in\{0.05,0.075,0.1,0.125,0.15\}\) |
| Width / depth / heads | D=256, L=4, H=8 |
| Optimizer | Adam, η₀ = **0.1**, t_epoch = 1 |
| Host | Della **a100** (1:05:00 → regular queue, not 59m gputest) |
| GPUs | **1×A100** / job |
| Batch | B_gpu=**128** → B_global=**128** |
| Steps | **2500** |
| Data | JetClass stream kin7; n_train = **320k** / cell; n_val = 25k; train-skip by shard |
| Cells | 3 shards × 2 seeds = **6 / R** → **30** |
| Tag | `v2cp` (`graph-R-b128-g1-s2500-k3-shard{S}-seed{E}-v2cp`) |
| Slurm | `scans/jets/scaling_law_L4_graph_R_1a100_v2cp.slurm` |
| Organized data | `paper/data/graph_construction_completep_v2/R_k3/` |
| Scratch org | `/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/graph_construction_completep_v2/R_k3/` |

---

## Grid

| R★ | shards × seeds | cells |
|---:|---:|---:|
| 0.05, 0.075, 0.1, 0.125, 0.15 | 3 × 2 | **30** |

Array index: `r_idx = task // 6`, `shard = (task%6)//2`, `seed = (task%6)%2`.  
Concurrency: `#SBATCH --array=0-29%8`.

---

## Changelog

| Date | Note |
|---|---|
| 2026-10-09 | Opened CompleteP v2 panel (b) at **k★=3** (panel-a winner; vs v1lam k★=6). |
| 2026-10-09 | Submitted **30-cell** R★ sweep @ k★=3: job **15295488** (`jc-gR-v2cp`, array 0–29%8). Live canvas: `graph-R-completep-v2-live.canvas.tsx`. |
| 2026-10-12 | **30/30 done.** Peak at R★=0.1 (0.9427). Means: R=0.05→0.9406, 0.075→0.9421, 0.1→0.9427, 0.125→0.9424, 0.15→0.9414. |
| 2026-10-12 | Tentative OP (2, 0.1) after L8 depth check; **OP freeze held** after kNN self-loop discovery + noloop re-sweep **15419558**. |
