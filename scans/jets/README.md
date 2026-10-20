# JetClass scan scripts — active paper workflows

Only scripts needed to **reproduce or extend current paper figures** live here.
Everything else is under [`archive/`](archive/) (gitignored reference copies;
not deleted).

## Active figures

| Figure | Plot / data | Launch / collect |
|---|---|---|
| `fig_stability_attn` / `_uniform` | `plot_stability_nature.py` · `paper/data/stability_one_step*.csv` | one-step μP probe below |
| `fig_graph_construction` (Nature 3-panel) | `plot_graph_construction_nature.py` | panels (a)/(b)/(c) below |
| `fig_hparam_transfer_proof` (WIP) | experimental log in `paper/figures/` | Arm 1/2 runners below |
| `fig_hparam_transfer_proof_edge_eta_s40k` (**latest**) | `plot_hxfer_edge_eta_s40k.py` | Adam η₀ @40k · L4/D256 + L6/D512 · frozen k★=7 |
| `fig_jetclass_adam_transfer_45k` | `paper/style/plot_jetclass_paper.py` | L4 D×η₀ @45k gputest set |

Shared training entrypoint: **`test_cpen_jetclass.py`** (uses `scans/common/sweep_common.py`).

Recipe notes: [`GRAPH_CONSTRUCTION_RECIPE.md`](GRAPH_CONSTRUCTION_RECIPE.md).
Stability log: [`paper/figures/fig_stability.md`](../../paper/figures/fig_stability.md).

---

## Stability — one-step Adam μP proof (`fig_stability_attn` / `fig_stability_uniform`)

```bash
sbatch scans/jets/mup_one_step_transfer_probe.slurm
python scans/jets/plot_stability_nature.py
python scans/jets/plot_stability_nature.py --all-cells   # SI across (k,R★)
```

Data: `paper/data/stability_one_step.csv`. Job logs: `paper/data/stability_logs/`.

---

## Panel (a)/(b) — graph geometry (gpu-test, B=96×4, tag `v6gt`/`v6gt2`)

```bash
bash scans/jets/start_graph_k_gputest_watchdogs.sh   # k @ R★=0.125
bash scans/jets/start_graph_R_gputest_watchdogs.sh   # R★ @ k★=12
python scans/jets/stream_knn_edge_counts.py          # twin-axis edge counts
python scans/jets/stream_star_support_counts.py      # twin-axis star support
python scans/jets/plot_graph_construction_nature.py
```

Slurms: `scaling_law_L4_graph_{k,k16,R}_shard_gputest.slurm`.

## Panel (c) — M11 vs edge η₀×D

```bash
bash scans/jets/start_m11_eta_gputest_watchdogs.sh   # M11 on Della gpu-test
# Edge on Lambda 8×A100 (4-GPU banks):
bash scans/jets/launch_edge_eta_lambda_bank.sh 0,1,2,3 29510 0
bash scans/jets/pull_edge_eta_lambda.sh
bash scans/jets/watch_edge_eta_live.sh               # live canvas
```

## Hparam transfer (AdamW `t` / const-LR)

```bash
# Arm 2 (const LR, D=256) — Lambda 8×A100
bash scans/jets/run_hparam_xfer_constlr_lambda8.sh
# Arm 2b (const LR, D=384) — Della ailab 4×H200, B=128×4
sbatch scans/jets/scaling_law_L4_hxfer_constlr_D384_ailab.slurm
```

## Adam η₀ transfer @ 45k (older active fig)

`scaling_law_L4_D{256,384,512,768}_*_gputest.slurm` + matching `L4_D*_gputest_watchdog.sh`
/ `start_L4_*_watchdogs.sh`. Metrics: `paper/data/collect_jetclass_paper_metrics.py`.

---

## Archive

| Path | Contents |
|---|---|
| `archive/graph_construction_jobs_2026-09-29/` | Pre-v6gt MIG / v5a80 / live-canvas probes |
| `archive/non_paper_2026-10-01/` | Everything else moved on 2026-10-01 (see that folder’s README) |

Restore a script: `cp scans/jets/archive/non_paper_2026-10-01/<category>/<file> scans/jets/`
