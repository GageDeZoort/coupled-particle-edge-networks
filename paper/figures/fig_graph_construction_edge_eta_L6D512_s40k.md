# Panel (c) edge η₀ — L6/D512 @ (k★=7, R★=0.1), 40k steps

**Living experimental log.** Same frozen graph OP as L4/D256, larger model.

Also the **L6/D512 arm of the latest hyperparameter-transfer proof** (Adam η₀ @
40k) — see `fig_hparam_transfer_proof_edge_eta_s40k.md`.

## Recipe

| Knob | Value |
|---|---|
| Model | `capen-llama-att` (M22 = hyperedges only) |
| Graph | all-seed star kin7 + ΔR-kNN **noloop** |
| L / D / H | 6 / 512 / 16 (`d_h=32`) |
| (k★, R★) | **(7, 0.1)** |
| η₀ grid | `{0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75}` |
| B × GPUs | **64 × 4** → B_global=256 (gpu-test via a100 ≤59m) |
| Steps | **40 000** (Lightning resumes `last.ckpt`) |
| n_train | **10 240 000** (= steps × B_global; single-pass unique jets) |
| Tag | `edge-L6D512-b64-g4-s40000-dh32-k7-R0p1` |
| Seed | 0 |

## Jobs

| Artifact | Path |
|---|---|
| Slurm | `scans/jets/scaling_law_L6_edge_eta_D512_k7_gputest_s40k.slurm` |
| Watchdog | `scans/jets/edge_eta_L6D512_k7_gputest_watchdog.sh` |
| Start | `bash scans/jets/start_edge_eta_L6D512_k7_gputest_watchdogs.sh` |
| Stop | `touch ~/.cache/jc_edge_L6D512_watchdogs/STOP` |
| Live canvas | `~/.cursor/projects/home-jdezoort/canvases/edge-eta-L6D512-k7-s40k-live.canvas.tsx` |

## Status

| Date | Note |
|---|---|
| 2026-10-14 | Launched L6/D512 η sweep (8 cells, no 1.0); shares gputest MaxJobsPU≈3 with L4/D256. |
| 2026-10-14 | Folded into transfer-proof fig `fig_hparam_transfer_proof_edge_eta_s40k` (latest before giant-model ship). |
