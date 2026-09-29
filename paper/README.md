# Paper figures (Nature Machine Intelligence standard)

## Style contract

Every plot marked **paper** must:

1. Call `paper.style.nature_plots.use_nature()` (scienceplots `science` + `nature` + `no-latex`).
2. Save via `save_paper_figure(fig, stem)` → `paper/figures/{stem}.png|pdf|eps` at 600 dpi.
3. Prefer editing text in Illustrator (`pdf.fonttype = 42`).

## Regenerate JetClass figures

```bash
conda activate mamba-env
cd coupled-particle-edge-networks
python paper/data/collect_jetclass_paper_metrics.py
python paper/style/plot_jetclass_paper.py
```

## Current figures

| Stem | Status | Content |
|------|--------|---------|
| `fig_jetclass_adam_transfer_45k` | **active** | Adam η₀ transfer @ 45k: train loss + val ROC |
| `fig_jetclass_ablation_1m` | archived | Old graph-construction 1×3 (mismatched B) |
| `fig_jetclass_graph_geometry` | archived | Geometry + old ROC-vs-\(R_\star\) |

Archived copies: `paper/archive/graph_construction_2026-09-28/`.

New graph-construction figures will be rebuilt from the matched A100-80GB
recipe in `scans/jets/GRAPH_CONSTRUCTION_RECIPE.md` (Plot 1 = \(k\) sweep at
\(R_\star{=}0.125\), B=384, 1000 steps, 3 shards × 2 inits). Superseded
submitters live under `scans/jets/archive/` (gitignored).

## Geometry notebooks (still valid structurally)

TopTagging geometry CSVs can be rebuilt from
`notebooks/jets/knn_graph_study.ipynb` and
`notebooks/jets/radius_hypergraph_study.ipynb` (independent of training B).
