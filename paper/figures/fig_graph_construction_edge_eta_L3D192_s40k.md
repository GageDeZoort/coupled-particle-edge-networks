# Panel (c) edge η₀ — L3/D192 @ (k★=7, R★=0.1), 40k steps

**Living experimental log.** Smaller-model arm for the hyperparameter-transfer
proof at the frozen graph OP. Matched batch / steps / OP to L4/D256.

Also the **L3/D192 arm** of `fig_hparam_transfer_proof_edge_eta_s40k.md`.

## Recipe

| Knob | Value |
|---|---|
| Model | `capen-llama-att` (M22 = hyperedges only) |
| Graph | all-seed star kin7 + ΔR-kNN **noloop** |
| L / D / H | **3 / 192 / 6** (`d_h=32`) |
| (k★, R★) | **(7, 0.1)** |
| η₀ grid (first pass) | `{0.01, 0.05, 0.1, 0.25, 0.5}` |
| B × GPUs | **64 × 4** → B_global=256 (gpu-test via a100 ≤59m) |
| Steps | **40 000** (Lightning resumes `last.ckpt`) |
| n_train | **10 240 000** (= steps × B_global; single-pass unique jets) |
| Tag | `edge-L3D192-b64-g4-s40000-dh32-k7-R0p1` |
| Seed | 0 |

## Jobs

| Artifact | Path |
|---|---|
| Slurm | `scans/jets/scaling_law_L3_edge_eta_D192_k7_gputest_s40k.slurm` |
| Watchdog | `scans/jets/edge_eta_L3D192_k7_gputest_watchdog.sh` |
| Start | `bash scans/jets/start_edge_eta_L3D192_k7_gputest_watchdogs.sh` |
| Stop | `touch ~/.cache/jc_edge_L3D192_watchdogs/STOP` |
| Live canvas | `~/.cursor/projects/home-jdezoort/canvases/edge-eta-L3D192-k7-s40k-live.canvas.tsx` |
| Canvas refresh | `python scans/jets/refresh_edge_eta_L3D192_k7_s40k_canvas.py` |
| Canvas watch | `bash scans/jets/watch_edge_eta_L3D192_canvas.sh` |

## Status

| Date | Note |
|---|---|
| 2026-10-19 | Launched rolling gputest watchdog (shares pool with L8 pretrain). First η₀ grid `{0.01, 0.05, 0.1, 0.25, 0.5}` @ 40k, B=64×4 matched to L4/D256 transfer arm. |
