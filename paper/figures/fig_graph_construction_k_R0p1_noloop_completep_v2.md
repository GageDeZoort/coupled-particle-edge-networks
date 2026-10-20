# Graph construction panel (a′) — k-sweep **without kNN self-loops**

**Status:** **running** — a100 **15419558** (`jc-gk-noloop`, backlog) + **gputest greedy fill** (`jc-gk-nl-gt`, ≤3)  
**Canonical panel (a) for CompleteP v2 going forward.**

Parent: `paper/figures/fig_graph_construction.md` (§kNN self-loops)  
Prior self-included: `fig_graph_construction_k_R0p1_completep_v2.md`

---

## Motivation

1. Legacy `build_delta_r_knn_edges` put **self at ΔR=0** in every top-\(k\), so
   CLI \(k\) = self + \(k-1\) true neighbors.
2. Self-loop part-interaction features hit floors \(10^{-3}\) → standardized
   **degeneracy spikes**; particle self-attention already lives in **M11**.
3. Control: k=1 self-only @ R★=0.1 → L4 mean **0.9410** ≈ prior k=2, **not**
   star-only (~0.933) → most of the “low \(k\) wins” signal was self slots /
   capacity / preprocess, not local geometry.
4. Therefore re-sweep with **true neighbors only** (`include_self=False`,
   run-dir tag `noloop`).

---

## Specs

| Item | Value |
|---|---|
| Model | CompleteP `capen-llama-att` |
| Geometry | \(R^\star=0.1\); \(k\in\{1,2,3,4,5,6,7,8,10\}\) |
| kNN self | **excluded** (default); **k=1 = one true neighbor** |
| L / D / H | 4 / 256 / 8 |
| Adam η₀ | 0.1 |
| Host | Della 1×A100 @ 1:05 |
| Batch / steps | B=128 × 2500 → **320k jets/cell** |
| Cells | 3×2 × 9 k = **54** |
| Tag | `v2cpNL` (+ `noloop` in run dir) |
| Slurm | `scaling_law_L4_graph_k_noloop_1a100_v2cp.slurm` (a100) · `…_gputest_v2cp.slurm` (≤59m) |
| Job | a100 **15419558**; gputest via watchdog (`jc-gk-nl-gt`) |
| Watchdog | `start_graph_k_noloop_gputest_watchdogs.sh` (MaxJobsPU≈3) |
| Canvas | `graph-k-noloop-completep-v2-live.canvas.tsx` |
| Overlay | prior self-included panel-(a) means |

Array: `k_idx = task // 6`, `shard = (task%6)//2`, `seed = (task%6)%2`.

---

## Code / docs checkpoints

| Piece | Change |
|---|---|
| `src/cpen/graphs/graph_star.py` | `include_self=False` default on `build_delta_r_knn_edges` / `build_star_plus_knn_graph` |
| `base_lit_cpen.py` | `live_knn_include_self` (default False) |
| `sweep_common.py` | `--live-knn-include-self`; tags `noloop` / `knnselt` |
| `attention_temperature.py` | threaded `live_knn_include_self` into γ / materialize |
| `EDGE_FEATURES_AND_FORWARD.md` | documents noloop default |

---

## Changelog

| Date | Note |
|---|---|
| 2026-10-12 | Default kNN excludes self; opened this log. |
| 2026-10-12 | Submitted **54-cell** noloop k-sweep job **15419558** (`jc-gk-noloop`). Hold \(k^\star\) freeze until complete. |
| 2026-10-12 | **gputest greedy fill:** same recipe @ 0:59 (`jc-gk-nl-gt`); watchdog ≤3 parents. First wave tasks 0–2 → **15420596–98**; cancelled matching a100 indices. Prior 1×A100 CompleteP cells ~10–28 min → fits gputest. |
| 2026-10-12 | Near-complete: peak **k★=7** (0.9391 ± 0.0017); plateau 6–8. Opened panel (b′) R★ scan @ k=7 noloop. |
