# AdamW t_epoch sweep — L4/D256 η₀=0.1, 40k steps

Compare AdamW + constant LR + t_epoch schedule vs panel-(c) Adam η₀=0.1.

## Recipe

| Knob | Value |
|---|---|
| Model | `capen-llama-att` L4/D256/H8 |
| Graph | all-seed star kin7 + ΔR-kNN **noloop** (k★=7, R★=0.1) |
| η₀ | **0.1** (fixed) |
| Optimizer | **AdamW** · `--scheduler none` (constant LR) |
| t_epoch | `{0.1, 0.5, 1, 5, 10, 25, 50}` |
| B × GPUs | **256 × 1** (a100, 2h slots) |
| Steps | **40 000** (watchdog resumes `last.ckpt`) |
| n_train | **100 000 000** (prescribed D for t_epoch λ; 10.24M unique presentations) |
| Tag | `aw-tepoch-t{{T}}-n100M-b256-g1-s40000-dh32-k7-R0p1` |
| Adam baseline | `edge-b64-g4-s40000-dh32-k7-R0p1` η₀=0.1 @ B_global=256 |

## Jobs

| Artifact | Path |
|---|---|
| Slurm | `scans/jets/scaling_law_L4_adamw_tepoch_D256_1a100_s40k.slurm` |
| Watchdog | `scans/jets/adamw_tepoch_D256_1a100_watchdog.sh` |
| Start | `bash scans/jets/start_adamw_tepoch_D256_1a100_watchdogs.sh` |
| Stop | `touch ~/.cache/jc_aw_tepoch_watchdogs/STOP` |
| Live canvas | `adamw-tepoch-D256-s40k-live.canvas.tsx` |

## Status

| Date | Note |
|---|---|
| 2026-10-14 | Launched a100 2h rolling watchdog, MAX_A100=4. |
| 2026-10-14 | **Restarted**: cancelled D=10.24M WD wave; now n_train=100M prescribed. |
