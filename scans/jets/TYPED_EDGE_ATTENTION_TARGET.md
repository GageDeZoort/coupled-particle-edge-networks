# Typed edge attention — target semantics

**Status:** §1 implemented in `--typed-edge-streams` (2026-10-20). New runs use
`run_tag` suffix `-s1`. Do **not** resume archived v1/v2 ckpts.

Archive (debunked v1/v2 exploration):

```
/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/archive/typed_edge_exploration_2026-10-20/
  sweep_lr/   # run dirs (v1 + v2 tags)
  logs/       # jc_typed_k7_*.{out,err}
  README.txt
```

---

## 1. Goal (authoritative, stream-centric)

Who is **updated** by which interactions:

| Stream | Updated by |
|--------|------------|
| **kNN edges** (and other non-hyper edges) | particle→edge messages only |
| **Hyperedges** (`deg > 2`) | particle→edge **and** hyperedge↔hyperedge messages |
| **Particles** | particle↔particle **and** edge→particle messages from **all** edges (kNN + hyperedges) |

In words:

1. **kNN edges** receive information from their endpoint particles; they do **not**
   attend to other edges.
2. **Hyperedges** receive information from member particles **and** from other
   hyperedges (edge–edge attention among `deg > 2` only).
3. **Particles** attend to each other (M11) and to **every** live edge token
   (kNN and hyperedges) via the bipartite edge→particle map.

`f₂₂ = 0` on non-hyperedge slots (no LN self-copy). Dual encoders
(star kin7 √(D/7), kNN part-int √(D/4), no pad).

---

## 2. Index convention — **read this before coding**

This repo’s CAPEN-Llama-att residual is (`EDGE_FEATURES_AND_FORWARD.md`):

\[
H_X \leftarrow H_X + \alpha(f_{11}+f_{21}),\qquad
H_E \leftarrow H_E + \alpha(f_{22}+f_{12}).
\]

| Repo `relation` | Q from | K/V from | Updates |
|-----------------|--------|----------|---------|
| `11` | particles | particles | particles |
| `21` | particles | edges | **particles** |
| `12` | edges | particles | **edges** |
| `22` | edges | edges | **edges** |

Writeup / discussion language sometimes swaps the bipartite labels (calling the
edge←particle map “M21” and the particle←edge map “M12”). **When in doubt,
trust the stream-centric table in §1 and this repo table — not the digit order
in casual chat.**

### Goal ↔ repo relations

| Goal (§1) | Repo implementation |
|-----------|---------------------|
| kNN / tiny-star edges updated by particle→edge only | `f_12` on those rows; `f_22 = 0` |
| Hyperedges updated by particle→edge + hyper↔hyper | `f_12` on hyper rows; `f_22` among hypers only |
| Particles updated by M11 + all edges | `f_11` + `f_21` with **unmasked** knn **and** hyper columns in `m_21` |

Residual scale (locked):

- Edge row with **two** live branches (`f_12` and `f_22`): \(\alpha = 1/(\sqrt{2}\,L)\) (hyperedges).
- Edge row with **one** live branch (`f_12` only): \(\alpha = 1/L\) (kNN / tiny stars).
- Particles keep the usual two-branch \(\alpha = 1/(\sqrt{2}\,L)\) on `(f_11+f_21)`.

---

## 3. What the paused exploration got wrong

| Attempt | Mistake |
|---------|---------|
| v1 (`typed-edge-…` without `-v2`) | Treated **all star slots** as the full-interaction set; zeroed **M12** on knn; gave knn its own knn↔knn M22. |
| v2 (`…-v2`) | Switched full set to `deg>2` hyperedges (good for M22), but still **zeroed M12 on knn** and kept knn↔knn M22 — **not** the §1 goal. |
| **s1 (current)** | Matches §1: knn get M12 + `f22=0`; hypers get M12+M22; particles see all edges in M21. |

---

## 4. Implementation checklist

- [x] kNN rows: apply `f_12`; force `f_22 = 0` (no self-copy, no knn↔knn block).
- [x] Hyper rows (`deg > 2`): apply `f_12` and hyper↔hyper `f_22`.
- [x] Tiny stars (`deg ≤ 2`): same as knn for routing (`f_12` yes; `f_22 = 0`).
- [x] `f_21`: **no** type mask — knn and hyperedges both keys/values for particles.
- [x] Dual encoders + no pad.
- [x] New `run_tag` suffix `-s1`; do not resume archive ckpts.
- [x] Unit tests: `tests/test_typed_edge_streams.py`.
- [x] CLI help quotes §1.

---

## 5. Related code

- `src/cpen/models/capen_llama_att.py` — `typed_edge_streams` forward (§1).
- `src/cpen/graphs/graph_star.py` — `typed_features=True` star(7)+knn(4).
- `scans/common/sweep_common.py` — `--typed-edge-streams`.
- `scans/jets/EDGE_FEATURES_AND_FORWARD.md` — baseline (non-typed) forward.
- `tests/test_typed_edge_streams.py` — §1 routing tests.

Baseline (untouched) remains: M12/M21 on all edges; M22 on hypers only with
identity `f_22` on 2-edges (the self-copy Claude flagged). §1 is the typed
replacement for that identity path.
