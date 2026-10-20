# Graph construction plots — matched recipe (2026-09-29, v6gt)

Matched budget on Della **gpu-test** (`4×A100`, qos via `time≤59m`):

| knob | value |
|------|-------|
| hardware | **4×A100** / job, B_gpu=**96** → B_global=**384** |
| steps | **1000** (~384k jet-updates = 1 pass over a shard) |
| train shard | **384 000** jets; 3 disjoint shards via `--train-skip-jets` |
| inits | **2** seeds per shard → **6** runs per \(k\) |
| packing | **one Slurm task per cell** (array 0–41); pool watchdog ≤3 concurrent |
| model | `capen-llama-att`, L=4, D=256, H=8 |
| optimizer | Adam, η₀=0.25, no schedule |
| data | JetClass stream, kin7 + part-interaction, p=80, n_val=25k |
| attention | M12/M21 all edges; **M22 hyperedge-only** |
| run tag | `-v6gt` (supersedes `-v5a80` 1×A100-80GB) |

## Plot 1 — \(k\) at \(R_\star=0.125\) (shard × init)

```bash
bash scans/jets/start_graph_k_gputest_watchdogs.sh
# keeps ≤3 jc-gk-gt tasks in flight until all 42 cells hit ~1000 steps
```

Shards use `skip = shard * 384000`. Nature panel (a) refreshes via
`plot_graph_construction_nature.py` as cells finish.

Twin-axis edge counts (mean ± s.d. across the same 3 shards, k through 16)::

```bash
python scans/jets/stream_knn_edge_counts.py   # writes paper/data/graph_geometry_knn.csv
```

Superseded submitters / MIG probes / older 1M ablate grids / v5a80 / exploratory
Lambda R★ jobs live under `scans/jets/archive/` (see `scans/jets/README.md`).
Active launchers stay in `scans/jets/`.


## Plot 2 — \(R_\star\) at \(k^\star{=}12\) (same 3×2 design)

```bash
bash scans/jets/start_graph_R_gputest_watchdogs.sh
# pool ≤3 concurrent on gputest until all 30 cells hit ~1000 steps
# 4×A100, B=96→384, tag -v6gt, k★=12
```

One-shot (may hit MaxSubmit / MaxJobs): `bash scans/jets/submit_graph_R_gputest.sh`.
Do **not** use `submit_graph_R_a10080.sh` for the paper grid — Priority backlog.
Nature panel (b) collects `*graph-R*v6gt*` (prefers over v6a80).

## Plot 3 — LR × width vs M11-only

M11 η₀×D @4k on gpu-test is complete (`m11-b96-g4-s4000-dh32`). Edge series TBD.
