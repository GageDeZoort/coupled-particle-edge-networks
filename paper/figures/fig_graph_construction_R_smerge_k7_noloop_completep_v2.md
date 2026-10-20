# Star overlap-merge R★ @ k=7 noloop — CompleteP v2

**CLOSED (2026-10-13).** Cancelled mid-flight; all-seed star+kin7 frozen instead.
Partial @500 steps (~0.9215) did not beat all-seed (~0.9255–0.927). Artifacts
under `scans/jets/archive/hyperedge_ablations_2026-10-13/`.

~~Ablation of star hyperedge *structure* vs the canonical all-seed stars
(panel b′ mean ~0.9391).~~

## Recipe

| Knob | Value |
|---|---|
| Model | `capen-llama-att` (M22 = **hyperedges only**, default) |
| L / D / H | 4 / 256 / 8 |
| η₀ / opt | 0.1 / Adam |
| B × steps | 128 × 2500 (≈320k jets / shard) |
| kNN | k=7 **noloop** part-interaction |
| Stars | seed if **pT > 1 GeV**; support = ΔR < R★; **transitive merge** if overlap **> 3** |
| Star features | center↔centroid **part-interaction** (same as original stars) |
| R★ grid | `{0.05, 0.075, 0.1, 0.125, 0.15}` × 3 shards × 2 seeds = **30 cells** |
| Tag | `gsmergeNL` (+ auto `smerge` / `pt1p0` / `ov3` / `noloop`) |

**Question:** can sparse pT-seeded + overlap-merged stars beat (or match) the
original dense all-seed star hypergraph? If not, freeze the original scheme.

**Role in the project:** last structural ablation before **pinning graph
construction** and scaling the full CompleteP model. Canvas:
`graph-R-smerge-k7-completep-v2-live.canvas.tsx`. Org checklist:
`docs/PUBLIC_TREE_CHECKLIST.md`.

## Jobs

| Job | Partition | Status |
|---|---|---|
| **15496550** | a100 | **cancelled** (Priority stall) |
| **15496930** | gputest ≤59m array **0–24%3** | **cancelled** |
| watchdog | fill tasks **25–29** | **stopped** |

Slurm: `scans/jets/scaling_law_L4_graph_R_smerge_k7_noloop_1a100_v2cp.slurm`  
Canvas: `canvases/graph-R-smerge-k7-completep-v2-live.canvas.tsx`  
Refresh: `scans/jets/refresh_graph_R_smerge_v2cp_canvas.py`

## Baseline (do not mix)

| Arm | Mean val ROC | Note |
|---|---:|---|
| kNN + all-seed stars (part-int) @ R★≈0.1 | **0.9391** | panel (b′) ref |
| kNN-only | 0.9376 | stars help ~+0.0015 |
| kNN-only + full M22 | 0.9389 | expensive substitute |
| star kin7 + pT>1 (no merge) | ~0.9385 | featurization ≈ inert |

## Results

| R★ | n_done | mean ROC | std | Δ vs 0.9391 |
|---:|---:|---:|---:|---:|
| 0.05 | — | — | — | — |
| 0.075 | — | — | — | — |
| 0.1 | — | — | — | — |
| 0.125 | — | — | — | — |
| 0.15 | — | — | — | — |

## Chronology

| When | Note |
|---|---|
| 2026-10-13 | Implemented `build_star_overlap_merge_plus_knn_graph`; submitted **15496550** (a100), Priority-stalled.
| 2026-10-13 | Cancelled 15496550; gputest wave1 **15496930** (0–24%3); watchdog fills 25–29 (MaxSubmit=25). |
