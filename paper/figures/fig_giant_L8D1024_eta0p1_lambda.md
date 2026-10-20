# Giant CAPEN — L=8 / D=1024 / d_h=32 @ η₀=0.1

| Field | Value |
| --- | --- |
| Model | `capen-llama-att`, L=8, D=1024, **d_h=32 → H=32** |
| OP | k★=7, R★=0.1, noloop star kin7 |
| Optimizer | Adam, `--scheduler none`, **η₀=0.1** (peak LR = η₀/√D) |
| Lambda host | `ubuntu@168.110.21.94` (8×A100-40GB) |
| Lambda batch | B_gpu=**32** × 8 → **B_eff=256** (B=64 OOMed on 40GB) |
| Lambda data | 21 ROOT/class (210 files); `n_train=10_240_000`; `max_steps=40_000` |
| Val | **none** on Lambda (`n_val=0`) |
| Della continue | gputest 4×A100, B=32×4 → **B_eff=128**, same `max_steps=40_000` |
| Della home | `/scratch/.../jetclass_pretrain/` (`lambda_L8D1024/` mirror + `jetclass_output/...`) |

## Artifacts

| Artifact | Path |
| --- | --- |
| Lambda runner | `scans/jets/run_giant_L8_D1024_eta0p1_lambda8.sh` |
| Della slurm | `scans/jets/scaling_law_L8_pretrain_continue_gputest.slurm` |
| Della watchdog | `scans/jets/pretrain_L8_continue_gputest_watchdog.sh` |
| Start | `bash scans/jets/start_pretrain_L8_continue_gputest_watchdogs.sh` |
| Stop | `touch ~/.cache/jc_pretrain_L8_watchdogs/STOP` |
| Mirror ckpt | `jetclass_pretrain/lambda_L8D1024/checkpoints/last.ckpt` |
| Seed manifest | `jetclass_pretrain/manifests/L8D1024_della_continue_seed.json` |

## Status

| Time | Event |
| --- | --- |
| 2026-10-14 12:46 | Replan: full-epoch dh32; staging began on Lambda. |
| 2026-10-14 23:08 | Launched B=32×8 unique-40k after corrupt-ROOT + OOM@B64 repairs. |
| 2026-10-15 18:38 | **Paused** Lambda at `global_step≈17050` (ckpt `last.ckpt` @ **17000**). GPUs cleared. |
| 2026-10-15 14:40 | Mirrored ckpt → `jetclass_pretrain/lambda_L8D1024/`; Della gputest continue + watchdog armed. |
| 2026-10-15 21:31 | First Della slot resumed @17001 then crashed (`expandable_segments` CUDA allocator assert). Relaunch with `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` only. |
