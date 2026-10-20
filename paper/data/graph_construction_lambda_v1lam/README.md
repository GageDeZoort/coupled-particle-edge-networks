# Graph construction — Lambda v1lam campaign

Matched recipe: D=256, L=4, H=8, η₀=0.25, B=64×8=512, 1000 steps,
attentional pool, part-interaction edges, 3 shards × 2 seeds.

## Panel (a) — k @ R★=0.1
- Grid: k ∈ {2,3,4,5,6,7,8,10} on the live figure (k=1 trained but dropped)
- Trained cells: 54 (incl. k=1) under SWEEP / org tree
- Summary CSV: `k_R0p1/summaries/graph_k_val_from_trainlog.csv`
- Incl-k1 CSV: `k_R0p1/summaries/graph_k_val_from_trainlog_incl_k1.csv`
- Artifact manifest: `k_R0p1/summaries/artifact_manifest.csv`
- Parquets / ckpt symlinks: `k_R0p1/{parquets,checkpoints}/`
- Scratch org: `/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/graph_construction_lambda_v1lam/k_R0p1/`
- Tag: `v1lam` (`graph-k-b64-g8-s1000-R0p1-…-v1lam`)

## Panel (b) — R★ @ k★=6
- Grid: R★ ∈ {0.05, 0.075, 0.1, 0.125, 0.15} — **30/30 done**
- Summary CSV: `R_k6/summaries/graph_R_val_from_trainlog.csv`
- Artifact manifest: `R_k6/summaries/artifact_manifest.csv`
- Parquets / ckpt symlinks: `R_k6/{parquets,checkpoints}/`
- Scratch org: `/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/graph_construction_lambda_v1lam/R_k6/`
- Tag: `graph-R-b64-g8-s1000-k6-R{R}-…-v1lam`

## Pull
`scans/jets/pull_graph_construction_v1lam_full.sh`

## Live canvas
`~/.cursor/projects/home-jdezoort/canvases/graph-k-R0p1-lambda-live.canvas.tsx`

## Logs
`paper/figures/fig_graph_construction_k_R0p1.md`
