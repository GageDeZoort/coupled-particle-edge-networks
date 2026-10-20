# Archived JetClass scan scripts (reference only)

Scripts here are **not deleted** — they are superseded, exploratory, or
orthogonal to the current paper-figure workflows. The active set lives one
level up in `scans/jets/` (see `../README.md`).

This tree is **gitignored** (`scans/jets/archive/` in repo `.gitignore`) so it
does not clutter `git status`, but remains on disk for archaeology.

## Bundles

### `graph_construction_jobs_2026-09-29/`

Pre-`v6gt` Plot-1 iteration: MIG probes, older gpu-test submitters, live k-sweep
canvas updaters, 1M/MIG ablate grids.

### `non_paper_2026-10-01/`

Bulk tidy on 2026-10-01. Categories:

| Subdir | What |
|---|---|
| `graph_construction_superseded/` | a10080/MIG R|k submitters, Lambda R0p075/R0p125 exploratories, old `plot_graph_{k,R}_*.py`, `eval_graph_k_ckpt*` |
| `plot3_early_hxfer/` | Early Plot-3 M11-vs-edge, Della WFC hparam-xfer, adaptive t=1 interceptor |
| `width_scaling_exploratory/` | D384 H6 / D1024 / a100short / corr-embed / 40k-width chains (not the frozen 45k transfer fig) |
| `knn6_eta_transfer/` | Pre-operating-point η sweeps at kNN=6 |
| `offline_build_graphs/` | Offline `build_*graphs*` JetClass/TopTagging graph dumps |
| `toptagging_finetune/` | TopTagging fine-tune entrypoints + `FINETUNE.md` |
| `logs_joblists/` | Stale `*.out`, `jc_*_jobs.txt`, wishlist notes |
| `misc_probes/` | GPU-mem probes and leftover one-offs |

## Restore

```bash
cp scans/jets/archive/non_paper_2026-10-01/<category>/<script> scans/jets/
```

Do **not** point paper docs at paths under `archive/` unless documenting history.
