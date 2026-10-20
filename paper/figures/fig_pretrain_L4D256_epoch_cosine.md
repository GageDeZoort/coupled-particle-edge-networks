# JetClass pretrain — L4/D256 Adam η₀=0.1 (const epoch → cosine epoch)

| Field | Value |
| --- | --- |
| Model | `capen-llama-att` L4/D256/H8 |
| Graph | noloop star kin7 · k★=7 · R★=0.1 |
| Optimizer | Adam · η₀=**0.1** (peak = η₀/√D = 0.00625) |
| Epoch 1 | `--scheduler none` · `max_steps=390625` · B=64×4=256 · n_train=1e8 |
| Epoch 2 | `--scheduler warmup_flat_cosine` · warmup=0 · flat=0 · `cosine_steps=390625` · **`lr_decay_min_frac=0.01`** (floor = 1% of peak) · `max_steps=781250` |
| Seed | Unique-40k edge η₀=0.1 @ step 40k → const epoch → cosine |

## Artifacts

| Artifact | Path |
| --- | --- |
| Const slurm / watchdog | `scaling_law_L4_pretrain_epoch_gputest.slurm` · `pretrain_L4_epoch_gputest_watchdog.sh` |
| Cosine slurm / watchdog | `scaling_law_L4_pretrain_cosine_epoch_gputest.slurm` · `pretrain_L4_cosine_epoch_gputest_watchdog.sh` |
| Start cosine | `bash scans/jets/start_pretrain_L4_cosine_epoch_gputest_watchdogs.sh` |
| Stop cosine | `touch ~/.cache/jc_pretrain_L4cos_watchdogs/STOP` |
| Home | `/scratch/.../jetclass_pretrain/` |

## Status

| Date | Note |
| --- | --- |
| 2026-10-14 | Const-LR epoch launched on gputest (resume @40k → 390625). |
| 2026-10-15 | Val ROC soft-plateau ~0.962; finish const epoch then cosine epoch-2 to 0.01× LR. |
| 2026-10-15 21:36 | Const epoch **done** (`done pretrain L4`, last log ~390600). Cosine watchdog submitted `jc-pretrain-L4cos` → 781250 with min_frac=0.01. |
| 2026-10-16 | Bug: many gputest slots ran ~3–4 min then exited — `epochs=1` made Lightning stop at `max_epochs=1` after finishing epoch-0 (~390627). `last.ckpt` never passed 390003 (no 1k-step flush before exit). **Fixed: `EPOCHS=2`**, pending requeue. |
