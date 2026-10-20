# JetClass pretrain — L6/D512 Adam η₀=0.1 (flat epoch-1)

Matched to L4/D256 protocol (`fig_pretrain_L4D256_epoch_cosine.md`).

| Field | Value |
| --- | --- |
| Model | `capen-llama-att` L6/D512/H16 (`d_h=32`) |
| Graph | noloop star kin7 · k★=7 · R★=0.1 |
| Optimizer | Adam · η₀=**0.1** · `--scheduler none` |
| Epoch 1 | `max_steps=390625` · B=64×4=256 · n_train=1e8 |
| Seed | Edge η₀=0.1 @ step 40k (`edge-L6D512-b64-g4-s40000…`) |

## Artifacts

| Artifact | Path |
| --- | --- |
| Slurm / watchdog | `scaling_law_L6_pretrain_epoch_gputest.slurm` · `pretrain_L6_epoch_gputest_watchdog.sh` |
| Start | `bash scans/jets/start_pretrain_L6_epoch_gputest_watchdogs.sh` |
| Stop | `touch ~/.cache/jc_pretrain_L6_watchdogs/STOP` |
| Home | `/scratch/.../jetclass_pretrain/` |

## Status

| Date | Note |
| --- | --- |
| 2026-10-19 | Launched flat-LR epoch-1 on Della gputest (seed 40k → 390625). |
