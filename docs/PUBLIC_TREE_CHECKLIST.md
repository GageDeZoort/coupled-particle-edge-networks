# Public tree checklist & organization todo

Living checklist for making **coupled-particle-edge-networks** publicly
presentable without breaking Della ops. Update checkboxes and notes as we go.

**Companion draft README (stranger-facing):** [`../README.public.md`](../README.public.md)  
**Lab README (Della ops):** [`../README.md`](../README.md)

---

## North star

> Public visitors see: what CAPEN is, how graphs are built, one runnable demo,
> one paper figure, a citation.  
> Lab operators keep: Slurm, watchdogs, living experiment logs, scratch paths.

**Near-term science gate (do first):** pin the **graph construction OP**, then
scale the full model. Do not expand public polish that depends on an unfrozen
recipe.

| Gate | Status | Notes |
|---|---|---|
| Graph construction OP freeze | ✅ frozen | all-seed star + kin7 + noloop kNN; **(k★,R★)=(7,0.1)** |
| Full-model scale-up | ⏳ next | L↑ / D↑ / data↑ under CompleteP; panel (c) at OP |
| Public repo cut | ⏳ after freeze narrative | |

---

## Keep in public tree

| Path | Why |
|---|---|
| `src/cpen/` | Product surface: models, graphs, training, apps |
| `tests/` | Smoke + unit tests (trim Della-only later) |
| `pyproject.toml` / `setup.*` / `requirements.txt` | Install |
| `LICENSE` | **add** — missing today |
| `README.md` | Replace with public front door when ready; keep Della doc as `docs/della.md` |
| `paper/figures/*.png,pdf` + captions | Selected final figures only |
| `paper/COMPLETEP_SCALES.md` | Method reference |
| `docs/` | Public docs + this checklist |
| One Colab / notebook under `notebooks/public/` | Demo (create) |

---

## Keep private / lab-only (move or don’t publish)

| Path | Action | Status |
|---|---|---|
| `scans/jets/*.slurm` (dozens) | Stay in lab tree; archive stale → `scans/jets/archive/` | ⬜ |
| `scans/jets/*watchdog*` / `start_*` | Lab only | ⬜ |
| `scans/jets/*jobid*.txt` | gitignore or lab-only | ⬜ |
| `scans/jets/refresh_*_canvas.py` | Lab only (canvases live under `~/.cursor/.../canvases/`) | ⬜ |
| `paper/figures/fig_*.md` living logs | Lab; optional publish *summaries* later | ⬜ |
| `paper/archive/` | Already gitignored — keep | ✅ |
| `notebooks/streams/` (~231M) | Out of public; gitignore artifacts | ⬜ |
| `paper_scratch` symlink | Never publish | ✅ (symlink to scratch) |
| Della absolute paths in README | Move to `docs/della.md` | ⬜ |

---

## Public tree target (sketch)

```
coupled-particle-edge-networks/          # or a thin public fork
  README.md                              # = today’s README.public.md
  LICENSE
  pyproject.toml
  src/cpen/
  tests/                                 # CPU smoke subset
  docs/
    PUBLIC_TREE_CHECKLIST.md             # this file
    della.md                             # cluster ops (optional in public)
    architecture.md                      # mermaid CAPEN / M11–M22
  notebooks/
    public/
      capen_graph_construction_colab.ipynb
  paper/
    figures/                             # final assets only
    COMPLETEP_SCALES.md
```

---

## Todo — organization (low → high effort)

### A. Housekeeping (do anytime)

- [ ] Add `LICENSE` (MIT or Apache-2.0)
- [ ] Expand `.gitignore`: `*jobid*.txt`, `notebooks/**/*.parquet`, `.cache/`
- [ ] Move stale slurm/watchdogs into `scans/jets/archive/` with a short README
- [ ] Split README: public front door vs `docs/della.md`
- [ ] Add `docs/architecture.md` (one mermaid: particles ↔ edges ↔ M11/12/21/22)

### B. Graph-construction pin (science — current focus)

- [x] Cancel smerge; freeze **all-seed star + kin7 + noloop kNN**
- [x] Write freeze line into `paper/figures/fig_graph_construction.md`
- [x] Tag canonical OP: `(k★=7, R★=0.1, noloop, M22=hyper, star=kin7 all-seed)`
- [x] Archive superseded ablations (smerge, dbscan, knn-only, fullm22)
- [ ] Re-queue panel (c) η₀×D at (7, 0.1); fill fig_graph_construction (c)

### C. Demo / flash (after freeze)

- [ ] Colab: build kNN vs kNN+stars on a tiny jet batch + 50-step train
- [ ] Optional second cell: μP η₀ transfer D=64→256 (toy or CiteSeer sibling)
- [ ] Export 1–2 public CSVs of scale / graph-construction curves
- [ ] HuggingFace model card + one L4/D256 checkpoint (optional)

### D. Public cut

- [ ] Create public remote / orphan branch with keep-list only
- [ ] CI: `pytest tests/ -q` CPU smoke on PR
- [ ] Citation BibTeX + link to paper when ready

---

## Chronology

| Date | Note |
|---|---|
| 2026-10-20 | Lab progress committed on `experiments/jetclass-pretrain-ft-2026-10` (package + scans + living logs). `main` left at last public-ish tip. **Do not delete `scans/common/` from main** — tests import it. Public cut = orphan/filter-repo later, after moving shared harness into `src/cpen` or keeping a thin `scans/common` on both. |
| 2026-10-13 | Checklist created. Public README drafted as `README.public.md`. Smerge ablation running to pin graph construction before scale-up. |
