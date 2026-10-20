# Graph construction panel (a) — CompleteP v2 @ R★=0.1

**Status:** **complete** — 48/48 · best mean **k★ = 3** (ROC 0.9427 ± 0.0014)  
**Superseded for operating-point freeze:** self was **included** in ΔR-kNN
(CLI \(k\) = self + \(k-1\) true neighbors). Canonical re-sweep:
`fig_graph_construction_k_R0p1_noloop_completep_v2.md` (job **15419558**).

Parent: `paper/figures/fig_graph_construction.md` · see §kNN self-loops there.

---

## Why a new log

Post–CompleteP / restored attentional-pool + unembed geometry, L4/D256 and
L8/D1024 μP checks track closely and beat the older questionably-scaled edge
recipes at matched η₀. Panel (a) (val ROC vs \(k\) at fixed \(R^\star=0.1\))
needed a clean re-run under the locked CompleteP parameterization.

---

## Specs (locked) — **self-included kNN**

| Item | Value |
|---|---|
| Model | `capen-llama-att` — attentional pool + class unembed (CompleteP) |
| Edges | part-interaction + incidence; star hyperedges on |
| Geometry | \(R^\star=0.1\); \(k\in\{2,3,4,5,6,7,8,10\}\) (no k=1) |
| kNN self | **included** (legacy; ΔR=0 self always in top-\(k\)) |
| Width / depth / heads | D=256, L=4, H=8 |
| Optimizer | Adam, η₀ = **0.1**, t_epoch = 1 |
| Host | Della **a100** (1:05:00) |
| GPUs | **1×A100** / job |
| Batch | B_gpu=**128** → B_global=**128** |
| Steps | **2500** |
| Data | JetClass stream kin7; n_train = **320k** / cell; n_val = 25k |
| Cells | 3×2 × 8 k = **48** |
| Tag | `v2cp` |
| Slurm | `scans/jets/scaling_law_L4_graph_k_1a100_v2cp.slurm` |
| Job | **15281965** |
| Organized data | `paper/data/graph_construction_completep_v2/k_R0p1/` |

---

## Results (self-included)

| k | mean val ROC | std | n |
|---:|---:|---:|---:|
| 2 | 0.9419 | 0.0021 | 6 |
| **3** | **0.9427** | 0.0014 | 6 |
| 4 | 0.9412 | 0.0022 | 6 |
| 5 | 0.9396 | 0.0026 | 6 |
| 6 | 0.9391 | 0.0017 | 6 |
| 7 | 0.9392 | 0.0014 | 6 |
| 8 | 0.9391 | 0.0014 | 6 |
| 10 | 0.9389 | 0.0018 | 6 |

Reinterpretation after self-loop discovery: peak at CLI k=3 ≈ **2 true neighbors**.

---

## Changelog

| Date | Note |
|---|---|
| 2026-10-09 | Opened CompleteP v2 log; cancelled B-ladder probe. |
| 2026-10-09 | Submitted **48-cell** k-sweep: B=128, η₀=0.1, 2500 steps (`jc-gk-v2cp`). |
| 2026-10-09 | **48/48 COMPLETED.** Peak **k=3** (0.9427±0.0014); k=2 within 0.001. Job **15281965**. |
| 2026-10-09 | Live canvas: `graph-k-completep-v2-live.canvas.tsx`. |
| 2026-10-12 | **Superseded for OP.** kNN self-inclusion documented; noloop re-sweep is canonical. Keep this log for overlay / historical comparison only. |
