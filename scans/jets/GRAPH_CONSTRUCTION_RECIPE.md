# Graph construction plots — matched recipe (2026-09-29, v5)

Matched budget on Della **1×A100-80GB** (`gpu-short`, `constraint=gpu80`):

| knob | value |
|------|-------|
| hardware | **1×A100-80GB** / job, B=**384** |
| steps | **1000** (~384k jet-updates = 1 pass over a shard) |
| train shard | **384 000** jets; 3 disjoint shards via `--train-skip-jets` |
| inits | **2** seeds per shard → **6** runs per \(k\) |
| packing | **one Slurm job per \(k\)** (array 0–6); 6 cells sequential |
| model | `capen-llama-att`, L=4, D=256, H=8 |
| optimizer | Adam, η₀=0.25, no schedule |
| data | JetClass stream, kin7 + part-interaction, p=80, n_val=25k |
| attention | M12/M21 all edges; **M22 hyperedge-only** |
| note | B=512 OOMs at \(k=14\) on 80GB; B=384 is the working batch |

## Plot 1 — \(k\) at \(R_\star=0.125\) (shard × init)

```bash
sbatch scans/jets/scaling_law_L4_graph_k_shard_a10080.slurm
# array 0–6: one k each; each job runs shard∈{0,1,2} × seed∈{0,1}
```

Shards use `skip = shard * 384000`. Run tag suffix `-v5a80`.

After completion, aggregate mean ± std over the 6 (shard, seed) cells per \(k\)
from epoch-end / last val (~step 1000); treat as the matched “1k-step” result.

Superseded submitters / MIG probes / older 1M ablate grids live under
`scans/jets/archive/graph_construction_jobs_2026-09-29/`.


## Plot 2 — \(R_\star\) at \(k^\star\) (same 3×2 design)

TBD after Plot 1 picks \(k^\star\). Same 1k / 512k shard protocol.

## Plot 3 — LR × width vs M11-only

Paused until Plots 1–2 freeze a working \((k,R_\star)\). Resume with
`bash scans/jets/resume_plot3.sh` only after that choice is locked.
