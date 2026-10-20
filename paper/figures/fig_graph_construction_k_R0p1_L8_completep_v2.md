# Graph construction depth check — k-sweep @ L=8 D=256, R★=0.1

**Status:** **complete** — job **15296385** (`jc-gk-L8v2`), **48/48**  
**Question:** is panel-(a) **k★=3** an artifact of only L=4 message-passing layers?

**Caveat (2026-10-12):** this sweep used **self-included** kNN (same as L4
panel a). Reinterpret CLI \(k\) as self+\(k-1\) neighbors. Noloop L8 follow-up
not yet scheduled; L4 noloop re-sweep is **15419558**.

Parent: `paper/figures/fig_graph_construction.md`  
L4 reference: `fig_graph_construction_k_R0p1_completep_v2.md` (self-included)

---

## Specs (matched jets to L4/v2cp)

| Item | Value |
|---|---|
| Model | `capen-llama-att` — CompleteP |
| Depth / width / heads | **L=8**, D=256, H=8 |
| Geometry | \(R^\star=0.1\); \(k\in\{2,3,4,5,6,7,8,10\}\) |
| kNN self | **included** (legacy) |
| Jets / cell | **320 000** |
| Batch / steps | B=128 × 2500 (probe **15296336**: B∈{32…128} all OK) |
| Optimizer | Adam, η₀ = **0.1** |
| Host | Della **a100** 1×A100 |
| Cells | 3×2 × 8 k = **48** |
| Tag | `v2cpL8` |
| Slurm | `scans/jets/scaling_law_L8_graph_k_1a100_v2cp.slurm` |
| Canvas | `graph-k-L8-completep-v2-live.canvas.tsx` |

---

## Results (48/48 @ 2500)

| k | L8 mean | ± std | Δ vs L4 self-incl. |
|---:|---:|---:|---:|
| **2** | **0.9420** | 0.0019 | +0.0001 |
| 3 | 0.9402 | 0.0039 | −0.0025 |
| 4 | 0.9406 | 0.0015 | −0.0006 |
| 5 | 0.9405 | 0.0016 | +0.0009 |
| 6 | 0.9400 | 0.0010 | +0.0009 |
| 7 | 0.9395 | 0.0012 | +0.0003 |
| 8 | 0.9399 | 0.0010 | +0.0008 |
| 10 | 0.9401 | 0.0010 | +0.0012 |

**k★=2** at L=8 (still low-\(k\); not “deeper wants denser kNN”). Combined with
k=1 self-only ≈ 0.941, low-\(k\) preference under self-included geometry is
likely **not** a depth artifact alone.

---

## Changelog

| Date | Note |
|---|---|
| 2026-10-09 | Opened; batch probe **15296336** → B=128. |
| 2026-10-09 | Submitted **48-cell** job **15296385**. |
| 2026-10-12 | **48/48 done.** Peak k=2 (0.9420). Noted self-included caveat; OP hold for noloop. |
