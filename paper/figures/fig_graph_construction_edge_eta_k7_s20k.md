# Panel (c) edge η₀ — L4/D256 @ (k★=7, R★=0.1), 40k steps

**Living experimental log.** Real HP sweep for the frozen graph OP.

Also the **L4/D256 arm of the latest hyperparameter-transfer proof** (Adam η₀ @
40k) — see `fig_hparam_transfer_proof_edge_eta_s40k.md`.

## Recipe

| Knob | Value |
|---|---|
| Model | `capen-llama-att` (M22 = hyperedges only) |
| Graph | all-seed star kin7 + ΔR-kNN **noloop** |
| L / D / H | 4 / 256 / 8 (`d_h=32`) |
| (k★, R★) | **(7, 0.1)** |
| η₀ grid | `{0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0}` |
| B × GPUs | **64 × 4** → B_global=256 (gpu-test via a100 ≤59m) |
| Steps | **40 000** (Lightning resumes `last.ckpt`) |
| n_train | **10 240 000** (= steps × B_global; single-pass unique jets) |
| Tag | `edge-b64-g4-s40000-dh32-k7-R0p1` |
| Seed | 0 |

## Jobs

| Artifact | Path |
|---|---|
| Slurm | `scans/jets/scaling_law_L4_edge_eta_D256_k7_gputest_s20k.slurm` |
| Watchdog | `scans/jets/edge_eta_D256_k7_gputest_watchdog.sh` |
| Start | `bash scans/jets/start_edge_eta_D256_k7_gputest_watchdogs.sh` |
| Stop | `touch ~/.cache/jc_edge_k7_watchdogs/STOP` |
| Live canvas | `~/.cursor/projects/home-jdezoort/canvases/edge-eta-D256-k7-s20k-live.canvas.tsx` |
| Canvas refresh | `python scans/jets/refresh_edge_eta_D256_k7_s20k_canvas.py` |

## Status

| Date | Note |
|---|---|
| 2026-10-13 | Submitted rolling gputest watchdog (MaxJobsPU≈3). |
| 2026-10-13 | Extended to 40k + η∈{0.5,0.75,1.0}; migrated run dirs to `ntr10240k` / `s40000`; unique single-pass. |
| 2026-10-14 | **9/9 done @40k**; best η₀=0.1 ROC **0.9557**. Folded into transfer-proof fig `fig_hparam_transfer_proof_edge_eta_s40k`. |
