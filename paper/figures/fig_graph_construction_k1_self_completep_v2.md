# Control — k=1 (self-loops only) @ R★=0.1 — CompleteP v2

**Status:** L4 **6/6 done**; L8 fair recipe **pending**  
**Question:** with self included in ΔR-kNN, does **k=1** (self only) beat
star-only or approach k=2?

Parent: `paper/figures/fig_graph_construction.md` (§kNN self-loops)

---

## Specs

| Arch | Recipe | Jets/cell | Cells | Job | Tag |
|---|---|---|---|---|---|
| L=4 D=256 H=8 | B=128×1, 2500 steps, η₀=0.1, 1×A100 | 320k | 3×2=6 | **15414480** | `v2cpK1` |
| L=8 D=1024 H=32 | B=32×4, **2500** steps, η₀=0.1, 4×A100 | 320k | 3×2=6 | **15418124** | `v2cpK1b` |

Slurm: `scaling_law_L4_graph_k1_self_1a100_v2cp.slurm`,
`scaling_law_L8_graph_k1_self_4a100_v2cp.slurm`  
Canvas: `graph-k1-self-completep-v2-live.canvas.tsx`

**Fairness:** first L8 attempt used B=64×4 × 1250 steps (fewer updates than
L4) — **cancelled** (job 15414481). Resubmitted matched **steps+jets**.

---

## Self-loop edge features

Same ParT interaction 4-vector on \((p,p)\):

- \(\Delta=0\), \(k_T=0\) → floored at \(10^{-3}\) → \(\ln= -6.91\)
- \(z=1/2\) → \(\ln z \approx -0.69\)
- \(m^2=4m_p^2\) (≈0 for massless PF) → floored

Then affine preprocess \((\mu,s)\) → large standardized spike. Incidence
deg‑1; still enters M12/M21. **Not** “no edge.”

---

## Results

### References (L4 @ R★=0.1)
| Setup | mean val ROC |
|---|---:|
| star-only | 0.9330 |
| self-included k=2 | 0.9419 |
| self-included k=3 | 0.9427 |

### L4/D256 k=1 self-only — job **15414480** (**6/6**)

| shard | seed | val ROC @ 2500 |
|---:|---:|---:|
| 0 | 0 | 0.9406 |
| 0 | 1 | 0.9395 |
| 1 | 0 | 0.9428 |
| 1 | 1 | 0.9426 |
| 2 | 0 | 0.9402 |
| 2 | 1 | 0.9403 |
| **mean** | | **0.9410 ± 0.0014** |

**Interpretation:** k=1 self-only ≈ prior k=2, **≪** gap to star-only.
Gain from “turning on kNN” is largely self-slots / spike tokens / preprocess,
not one true ΔR neighbor. Motivates noloop re-sweep.

### L8/D1024 k=1 — job **15418124**
Pending (Priority) under fair B=32×4 × 2500 recipe.

---

## Changelog

| Date | Note |
|---|---|
| 2026-10-12 | Opened k=1 self-loop control at L4/D256 and L8/D1024. |
| 2026-10-12 | Submitted L4 **15414480** and L8 **15414481** (short recipe). |
| 2026-10-12 | Cancelled short L8; resubmitted **15418124** as `v2cpK1b` (matched 2500 steps). |
| 2026-10-12 | L4 **6/6 done**, mean **0.9410 ± 0.0014**. Documented spike features; triggered noloop default + panel (a′) re-sweep. |
