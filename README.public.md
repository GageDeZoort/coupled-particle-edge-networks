# CAPEN — Coupled Particle–Edge Networks

**CompleteP-scaled** particle–edge attention for jet tagging, with a graph recipe
that mixes local ΔR-kNN edges and meso-scale star hyperedges.

This page is the **public front door**. Cluster ops and living experiment logs
live in the lab tree (`README.md`, `scans/`, `paper/figures/fig_*.md`).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](#)
[![License](https://img.shields.io/badge/license-TBD-lightgrey.svg)](#)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) <!-- wire after notebooks/public/ lands -->

---

## Why this exists

Deep jet taggers are sensitive to **how you connect constituents** and to
**learning-rate / width scaling**. CAPEN couples:

1. **Particle self-attention (M11)**
2. **Particle ↔ edge messages (M12 / M21)** on kNN 2-edges and star hyperedges
3. **Edge ↔ edge attention (M22)** on hyperedges only (by default)

…under a **CompleteP / μP** parameterization so $\eta_0$ can transfer across
width and depth.

---

## Graph construction (operating point)

Pinned after JetClass CompleteP-v2 ablations (see paper living logs):

| Piece | Default story |
| --- | --- |
| 2-edges | ΔR-kNN, **no self-loops** (`noloop`) — self lives in M11 |
| Edge features | ParT **part-interaction** |
| Hyperedges | Star-$R^\star$ supports (freeze pending final structural ablation) |
| M22 | **Hyperedges only** |
| Probe OP | $k^\star{=}7$, $R^\star{\approx}0.1$, L4/D256 |

> **Status:** structural freeze in progress (sparse pT-seeded overlap-merge vs
> all-seed stars). Do not treat $R^\star$ as final until that gate closes.

---

## Install

```bash
pip install -e .
# needs: torch, torch-geometric, (optional) lightning for training entrypoints
```

```python
from cpen.models.capen_llama_att import CAPENLlamaAtt  # name may vary by export
from cpen.graphs.graph_star import build_star_plus_knn_graph
```

---

## Quick demo (Colab)

*Coming soon:* `notebooks/public/capen_graph_construction_colab.ipynb`

1. Build kNN vs kNN+star incidence on a small jet batch  
2. Train a few dozen steps on a GPU  
3. Optional: same $\eta_0$ at two widths (μP smoke test)

---

## Repository map

```
src/cpen/          Models, graph builders, training utilities
tests/             Unit / smoke tests
docs/              Public docs + organization checklist
notebooks/public/  Colab-facing demos (lab notebooks stay elsewhere)
paper/figures/     Final figures for the manuscript
```

Lab-only (not part of the public story): Slurm arrays, watchdogs, job-id
files, Della absolute paths, living `fig_*.md` campaign logs.

Organization checklist: [`docs/PUBLIC_TREE_CHECKLIST.md`](docs/PUBLIC_TREE_CHECKLIST.md)

---

## Citation

```bibtex
@inproceedings{capen_completep,
  title  = {CompleteP CAPEN for Jet Tagging},
  author = {TBD},
  year   = {2026},
  note   = {Update on camera-ready}
}
```

---

## Related

- Hyperparameter transfer for residual GNNs (LoG companion): see `hp-transfer-gnns`
- CompleteP scale table: `paper/COMPLETEP_SCALES.md`
