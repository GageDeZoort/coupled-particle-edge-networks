# Graph construction panel (b′) — R★-sweep @ k★=7 **noloop** — CompleteP v2

**Status:** **submitted** — a100 backlog + gputest greedy fill (`jc-gR-nl-gt`)  
**Canonical panel (b) for CompleteP v2** once complete (follows noloop panel a′ k★=7).

Parent: `paper/figures/fig_graph_construction.md`  
Panel (a′): `fig_graph_construction_k_R0p1_noloop_completep_v2.md`  
Prior self-incl. panel (b) @ k=3: `fig_graph_construction_R_k3_completep_v2.md`

---

## Specs

| Item | Value |
|---|---|
| Model | CompleteP `capen-llama-att` |
| Geometry | \(k^\star=\mathbf{7}\) noloop; \(R^\star\in\{0.06,0.08,0.1,0.12,0.14\}\) |
| L / D / H | 4 / 256 / 8 |
| Adam η₀ | 0.1 |
| Host | Della 1×A100 @ 1:05 + gputest ≤59m fill |
| Batch / steps | B=128 × 2500 → **320k jets/cell** |
| Cells | 3×2 × 5 R = **30** |
| Tag | `v2cpNL` (`graph-R-noloop-…-k7-…`) |
| Slurm | `scaling_law_L4_graph_R_noloop_k7_1a100_v2cp.slurm` · `…_gputest_v2cp.slurm` |
| Watchdog | `start_graph_R_noloop_gputest_watchdogs.sh` |
| Canvas | `graph-R-noloop-k7-completep-v2-live.canvas.tsx` |

Array: `r_idx = task // 6`, `shard = (task%6)//2`, `seed = (task%6)%2`.

---

## Changelog

| Date | Note |
|---|---|
| 2026-10-12 | Opened after noloop panel (a′) established **k★=7** (mean 0.9391). R grid tightened around 0.1. |
