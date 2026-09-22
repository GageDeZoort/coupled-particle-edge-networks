# Coupled Particle-Edge Networks (CPEN)

Research codebase for hyperparameter transfer with CPEN / CAPEN on multiple
physics applications: **jet tagging**, **stellar streams**, and PascalVOC-SP.

## Install

```bash
pip install -e ".[dev]"
```

## Layout

```
src/cpen/
  models/           Shared CPEN / CAPEN / WIRE / encoders
  graphs/           Shared incidence, knn/ΔR primitives, undirected helpers
  training/         Base Lightning module, metrics, sweeps, logging
  apps/
    jets/           TopTagging + JetClass data, caches, lit modules
    streams/        Stellar-stream data, caches, lit modules
    pascal/         PascalVOC-SP data, caches, lit modules
  utils/            Compatibility shims → graphs / training / apps.*
  datamodules/      Compatibility shims → apps.* / training
  lit_models/       Compatibility shims → apps.* / training

scans/
  jets/             Jet build + train entrypoints + Slurm
  streams/          Stream train entrypoints + Slurm
  pascal/           Pascal build + train entrypoints + Slurm
  common/           Shared sweep CLI (sweep_common), profiling
  testing/          Thin shims for old paths (prefer scans/{jets,streams,…})

notebooks/
  jets/  streams/  pascal/  shared/
```

Prefer new import paths, e.g. `cpen.apps.streams.stream_datamodule`. Old
`cpen.utils.*` / `cpen.datamodules.*` / `cpen.lit_models.*` imports still work
via shims.

## Particle-only baseline (T1→T1)

The default model is `particle-only`, implementing the particle-centric residual network with CAPEN-Llama muP scalings (`σ = D^{-1/2}`):

- Encoder: `W^(0) ~ N(0, σ²)`, forward `× 1/(σ √n_0) = √(D/n_0)`
- Residual: `W ~ N(0, 1)`, forward `× (1/L)(1/√(n_ops D))`
- Decoder: `W^(L+1) ~ N(0, σ²)`, forward `× 1/(D σ) = 1/√D` (graph: after alpha-pool; node: per particle)
- Adam LR: `η = η_0 / √D` via `model.get_lr()`

```bash
python scans/jets/test_cpen_toptagging.py \
  --model particle-only \
  --mode sweep_lr \
  --operators identity,ones \
  --normalization energy-weights \
  --etas 0.3 \
  --depth 4 --width 128 \
  --epochs 50 \
  --root ./runs
```

## Quick smoke test (CPU, synthetic data)

```bash
python scans/jets/test_cpen_toptagging.py \
  --mode sweep_lr \
  --etas 0.3 \
  --depth 4 --width 64 \
  --epochs 2 \
  --batch-size 32 \
  --n-gpus 0 \
  --root ./runs
```

## Stellar streams

```bash
python scans/streams/test_cpen_stream.py --help
# Slurm: sbatch scans/streams/test_stream.slurm
```

## Hierarchical jet graphs (build)

```bash
bash scans/jets/submit_hier_graphs_cputest.sh
```

## Sweep modes

| Mode | Fixed | Swept |
|------|-------|-------|
| `sweep_lr` | depth, width | `--etas` |
| `sweep_size` | `--eta0` | `--sizes "d,w;d,w"` |
| `sweep_t_epoch` | `--eta0` (AdamW) | `--t-epochs` |

## SLURM

See `scans/jets/*.slurm`, `scans/streams/*.slurm`, and `scans/common/test.slurm`.
Relaunch the same command after timeout; training resumes from `last.ckpt`.
