# Ablation — R★ sweep star-only (no kNN) — CompleteP v2

**Status:** **complete** — job **15407319** (`jc-gR-star`), **30/30**  
**Question:** does the ΔR-kNN path help, or is star-\(R^\star\) enough?

Parent: `paper/figures/fig_graph_construction.md`  
Compare to: panel (b) star+k=3 · `fig_graph_construction_R_k3_completep_v2.md`

---

## Specs (matched to panel b, minus kNN)

| Item | Value |
|---|---|
| Model | CompleteP `capen-llama-att` |
| Geometry | **star-\(R^\star\) only** (no `--live-knn-k`) |
| \(R^\star\) | \{0.05, 0.075, 0.1, 0.125, 0.15\} |
| L / D / H | 4 / 256 / 8 |
| Adam η₀ | 0.1 |
| Host | Della 1×A100 @ 1:05 |
| Batch / steps | B=128 × 2500 → **320k jets/cell** |
| Cells | 3×2 × 5 R = **30** |
| Tag | `v2cpStar` |
| Slurm | `scans/jets/scaling_law_L4_graph_R_staronly_1a100_v2cp.slurm` |
| Canvas | `graph-R-staronly-completep-v2-live.canvas.tsx` |

---

## Results (30/30 @ 2500)

| R★ | star-only mean | ± std | star+k=3 (panel b) | Δ |
|---:|---:|---:|---:|---:|
| 0.05 | 0.9323 | 0.0016 | 0.9406 | **−0.008** |
| 0.075 | 0.9324 | 0.0010 | 0.9421 | **−0.010** |
| 0.1 | **0.9330** | 0.0013 | **0.9427** | **−0.010** |
| 0.125 | 0.9328 | 0.0019 | 0.9424 | **−0.010** |
| 0.15 | 0.9325 | 0.0018 | 0.9414 | **−0.009** |

Almost flat in R★; **~1 ROC point below** star+kNN everywhere → kNN block
helps, but later k=1 self-only showed most of that help is **not** “many
true neighbors” (see k=1 control + noloop re-sweep).

---

## Changelog

| Date | Note |
|---|---|
| 2026-10-12 | Opened after tentative OP (k★,R★)=(2, 0.1). |
| 2026-10-12 | Submitted **30-cell** job **15407319**. |
| 2026-10-12 | **30/30 done.** Peak-ish at R=0.1 (0.9330); Δ≈−0.01 vs star+k=3. |
