# JetClass → TopTagging fine-tune

## Layout

| Stage | Root |
|-------|------|
| JetClass pretrain | `…/jetclass_scaling_law/jetclass_output/…` |
| TopTagging fine-tune | `…/toptagging_finetune/toptagging_output/…` |
| TopTagging from scratch | `…/toptagging_output/…` (unchanged) |

FT runs are tagged `ft`, `initfrom`, `live`, `partint`, `knn{K}`, `p{P}`, `pstep{N}`.

## Recipe (match pretrain)

- `--toptagging-live --star-radius 0.2 --live-knn-k 6`
- `--jetclass-edge-features part-interaction`
- `--num-particles` equal to the pretrain run (currently **80** on the L4 scaling arms)
- kin7 body (TopLandscape HDF5 → ParT kin features automatically)

## Launch

```bash
INIT_FROM=/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/jetclass_output/adam/sweep_lr/<run>/last.ckpt \
ETA0=0.5 WIDTH=384 HEAD=12 NUM_PARTICLES=80 \
sbatch scans/jets/finetune_toptagging_from_jetclass.slurm
```

Omit `ETA0` to default the fine-tune learning-rate scale to the pretrain `eta_0` stored in the checkpoint. Sweep `ETA0` later to find the FT optimum.

## Semantics

- `--init-from` copies **weights only** (no optimizer / `global_step` from pretrain).
- Class-token / unembed / decoder heads are **not** copied (10-way → 2-way).
- If an FT `last.ckpt` already exists in the FT run dir, Lightning resumes **that** FT run.
- `--root` under a JetClass pretrain tree is rejected.
