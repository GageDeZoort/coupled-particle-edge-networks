# Depth/width check — noloop k-sweep @ L=8 D=1024, R★=0.1

**Status:** **submitted** — job **15440018** (`jc-gk-nl-L8`, array 0–53%3)  
**Question:** does noloop panel-(a′) **k★=7** hold at matched CompleteP scale L8/D1024?

Parent: `paper/figures/fig_graph_construction.md`  
L4 noloop: `fig_graph_construction_k_R0p1_noloop_completep_v2.md`  
Prior self-incl. L8/D256: `fig_graph_construction_k_R0p1_L8_completep_v2.md`

---

## Specs (fair vs L4/D256 noloop)

| Item | Value |
|---|---|
| Model | CompleteP `capen-llama-att` |
| L / D / H | **8 / 1024 / 32** (\(d_h=32\)) |
| Geometry | \(R^\star=0.1\); \(k\in\{1,2,3,4,5,6,7,8,10\}\) **noloop** |
| Jets / cell | **320 000** |
| Batch / steps | B_gpu=**32** × **4** → B_global=**128** × **2500** |
| Optimizer | Adam, η₀ = **0.1** |
| Host | Della **a100** 4×A100 |
| Cells | 3×2 × 9 k = **54** |
| Tag | `v2cpNL8` |
| Slurm | `scans/jets/scaling_law_L8_graph_k_noloop_4a100_v2cp.slurm` |
| Job | **15440018** |
| Canvas | `graph-k-noloop-L8D1024-completep-v2-live.canvas.tsx` |
| Concurrency | `#SBATCH --array=0-53%3` |

Array: same as L4 noloop (`k_idx = task // 6`, …).

---

## Changelog

| Date | Note |
|---|---|
| 2026-10-12 | Submitted **15440018** after L4 noloop established k★≈7. Fair recipe matches L8 k=1 self control (`v2cpK1b`). |
