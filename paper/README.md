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
# or iterate in notebooks/paper_jetclass_nature_figures.ipynb
```

## Current figures

| Stem | Content |
|------|---------|
| `fig_jetclass_ablation_1m` | Graph construction: edges vs M11; kNN $k$; star radius $R_\star$ |
| `fig_jetclass_graph_geometry` | Graph geometry vs $k$ / $R_\star$ + JetClass ROC plateau vs $R_\star$ |
| `fig_jetclass_adam_transfer_45k` | Adam η₀ transfer @ 45k: train loss + val ROC (D/H labels) |

Geometry CSVs (`graph_geometry_knn.csv`, `graph_geometry_star.csv`) come from the
TopTagging scans in `notebooks/jets/knn_graph_study.ipynb` and
`notebooks/jets/radius_hypergraph_study.ipynb` (\(k{=}6\) on the linear branch
from \(k{=}1\)).

## Known data gaps

- **D=384 background rejection @ 45k** was not logged (`heavy_metrics` off). ROC/acc are present.
- Ablation panel **b** uses a matched 2-epoch B=256 budget; panel **a** is 1 epoch @ B=512.

## Why D=384 has no background rejection at 45k

`val_bg_rejection` (ε_sig=0.5) and `val_bg_rejection_0p3` (ε_sig=0.3) are computed
together in the same heavy-metrics path. Mid-run `[val]` log lines intentionally omit
both (see `ParquetLoggerCallback._VAL_PRINT_KEYS`). D=256 jobs that **ended** at 45k
persisted rejection via the final parquet / end-of-run summary. D=384 jobs continued to
100k with **no parquet logger output**, so rejection at 45k was never written—neither
at 0.5 nor at 0.3. Fix: short offline re-eval from a ~45k ckpt (or re-run with parquet).
