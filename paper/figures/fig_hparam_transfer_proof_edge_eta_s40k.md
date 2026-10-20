# Hyperparameter transfer proof — Adam η₀ @ 40k (frozen OP)

**Living experimental log.** Latest panel-(a) / LR-transfer evidence at the
**frozen graph-construction operating point**. Supersedes older k★=12 / k★=6
10k arms for “where we are now” before shipping a giant model.

## Goal

Show that peak LR `η = η₀ / √D` yields a shared η₀ optimum across model size
under Adam + const LR, with **no train-data repeats** over the 40k-step budget.

| Item | Value |
|---|---|
| Claim | Adam η₀ transfer @ 40k · frozen OP (k★=7, R★=0.1, noloop) |
| Status | **L4/D256 9/9 done**; **L6/D512** prior arm; **L3/D192 rolling** (η₀∈{0.01,0.05,0.1,0.25,0.5}) |
| Models | **L3/D192/H6**, **L4/D256/H8**, **L6/D512/H16** (`d_h=32`) |
| Host | Della **gputest** (a100 ≤59m) via rolling watchdogs |
| Canvases | `edge-eta-L3D192-k7-s40k-live` · `edge-eta-D256-k7-s20k-live` · `edge-eta-L6D512-k7-s40k-live` |
| Plot | `scans/jets/plot_hxfer_edge_eta_s40k.py` |
| Figure | `paper/figures/fig_hparam_transfer_proof_edge_eta_s40k.{png,pdf,eps}` |
| CSV | `paper/data/hxfer_edge_eta_s40k.csv` |

## Job specs (locked)

| Item | Value |
|---|---|
| Model | `capen-llama-att` (M22 hyperedges; all-seed star kin7) |
| Graph | **ΔR-kNN noloop** + star · **(k★, R★) = (7, 0.1)** |
| Optimizer | **Adam**, `--scheduler` default / const peak LR `η = η₀/√D` |
| Batch | B_gpu=**64** × **4** → **B_eff=256** |
| Steps | **40 000** (Lightning resumes `last.ckpt`) |
| Epochs | **1** |
| n_train | **10 240 000** = steps × B_eff (single-pass unique) |
| n_val / n_test | 25 000 / 0 |
| Seed | 0 |

### Campaign A — L4/D256

| Item | Value |
|---|---|
| η₀ grid | `{0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0}` |
| Tag | `edge-b64-g4-s40000-dh32-k7-R0p1` |
| Slurm / watchdog | `scaling_law_L4_edge_eta_D256_k7_gputest_s20k.slurm` · `edge_eta_D256_k7_gputest_watchdog.sh` |
| Living log | `fig_graph_construction_edge_eta_k7_s20k.md` |
| Status | **9/9 @ 40k** · best **η₀=0.1**, val ROC **0.9557** |

### Campaign B — L6/D512

| Item | Value |
|---|---|
| η₀ grid | `{0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75}` (no 1.0) |
| Tag | `edge-L6D512-b64-g4-s40000-dh32-k7-R0p1` |
| Slurm / watchdog | `scaling_law_L6_edge_eta_D512_k7_gputest_s40k.slurm` · `edge_eta_L6D512_k7_gputest_watchdog.sh` |
| Living log | `fig_graph_construction_edge_eta_L6D512_s40k.md` |
| Status | prior arm (see living log) |

### Campaign C — L3/D192 (new)

| Item | Value |
|---|---|
| η₀ grid (first pass) | `{0.01, 0.05, 0.1, 0.25, 0.5}` |
| Tag | `edge-L3D192-b64-g4-s40000-dh32-k7-R0p1` |
| Slurm / watchdog | `scaling_law_L3_edge_eta_D192_k7_gputest_s40k.slurm` · `edge_eta_L3D192_k7_gputest_watchdog.sh` |
| Living log | `fig_graph_construction_edge_eta_L3D192_s40k.md` |
| Canvas | `edge-eta-L3D192-k7-s40k-live` |
| Status | **rolling** on gputest (shares pool with L8 pretrain) |

## Snapshot (2026-10-19)

| Model | Best η₀ so far | Val ROC | Step |
|---|---:|---:|---:|
| L3/D192 | — (just launched) | — | 0 |
| L4/D256 | **0.1** | **0.95571** | 40k |
| L6/D512 | **0.05** (among finished @ prior snapshot) | **0.95616** | 40k |

Open markers on the figure = mid-run (<95% of 40k). Re-plot when L3 cells finish.

**Note.** Depth and width move together (L3/D192 → L4/D256 → L6/D512). Treat as
size-transfer evidence at fixed OP, not a pure fixed-L width μP collapse.
All arms share **B_eff=256** and the frozen (k★,R★)=(7,0.1) OP.

## Paths

| Kind | Location |
|---|---|
| Della SWEEP | `…/jetclass_output/adam/sweep_lr/*edge-b64-g4-s40000*` · `*edge-L6D512*s40000*` · `*edge-L3D192*s40000*` |
| Logs | `…/logs/jc_edge_k7_s40k_gt_*.out` · `jc_edge_L6D512_s40k_gt_*.out` · `jc_edge_L3D192_s40k_gt_*.out` |
| Graph OP freeze | `fig_graph_construction.md` · CompletenessP v2 noloop |

## Related (not this figure)

| Campaign | Role |
|---|---|
| AdamW t_epoch @ L4/D256 η₀=0.1, n_train=100M | WD / schedule probe — `fig_adamw_tepoch_L4D256_s40k.md` |
| upool/apool 10k @ k★=6 | Prior Nature LR-transfer figs (still valid archive) |

## Changelog

| Date | Note |
|---|---|
| 2026-10-14 | Organized L4/D256 + L6/D512 40k Adam η sweeps into transfer-proof figure + this log. Marked as **latest** before giant-model ship. |
| 2026-10-14 | Replot: L4 9/9 (best η₀=0.1 @ 0.9557); L6 6/8 (best η₀=0.075 @ 0.9564; 0.5/0.75 still early). |
| 2026-10-19 | **Campaign C:** L3/D192/H6 (`d_h=32`) Adam η₀ @40k on gputest — first grid `{0.01,0.05,0.1,0.25,0.5}`, B=64×4 matched to L4/D256. Canvas `edge-eta-L3D192-k7-s40k-live`. |
