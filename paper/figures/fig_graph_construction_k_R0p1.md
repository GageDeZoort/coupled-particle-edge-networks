# Graph construction panel (a)/(b) — Lambda \(R^\star=0.1\) campaign

**Status:** panel (a) **complete**; panel (b) **running** on Lambda.

Parent: `paper/figures/fig_graph_construction.md`  
Canvas: [`graph-k-R0p1-lambda-live.canvas.tsx`](/home/jdezoort/.cursor/projects/home-jdezoort/canvases/graph-k-R0p1-lambda-live.canvas.tsx)  
Organized data: `paper/data/graph_construction_lambda_v1lam/`

---

## Pooling (confirmed)

**Attentional** class-token output pool — **not** uniform.
(`uniform_output_pool = False`; runners never pass `--uniform-output-pool`.)

---

## Edge features (confirmed healthy)

`part-interaction` + incidence + star. Finite γ / live-edge estimates for all
finished \(k\). For \(R^\star\notin\{0.1,0.125,0.15,0.2\}\), γ-path now falls
back to **live robust** part-int preprocess (`graph_star.py`, 2026-10-03).

---

## Matched recipe

| Knob | Value |
|---|---|
| Host | Lambda `193.122.155.21` (8×A100) |
| Model | `capen-llama-att`, attentional pool |
| \(D,L,H\) | 256 / 4 / 8 |
| Opt | Adam \(\eta_0=0.25\) |
| Batch | \(64\times8=512\) |
| Steps | 1000 |
| Replicates | 3 shards × 2 seeds |
| Tag | `v1lam` |

---

## Panel (a) — \(k\) @ \(R^\star=0.1\) — **DONE 48/48 (k=1 dropped)**

| \(k\) | n | mean best ROC | std |
|---|---|---|---|
| 1 | 6 | 0.9397 | 0.0009 |
| 2 | 6 | 0.9376 | 0.0026 |
| 3 | 6 | 0.9382 | 0.0015 |
| 4 | 6 | 0.9399 | 0.0019 |
| 5 | 6 | 0.9389 | 0.0030 |
| 6 | 6 | 0.9398 | 0.0010 |
| 7 | 6 | 0.9398 | 0.0006 |
| 8 | 6 | 0.9400 | 0.0008 |
| 10 | 6 | 0.9398 | 0.0015 |

Skipped \(k\in\{12,14,16\}\). Plateau from \(k\approx4\); operating \(k^\star=6\).

**Local organization**
```
paper/data/graph_construction_lambda_v1lam/
  README.md
  k_R0p1/summaries/graph_k_val_from_trainlog.csv
  k_R0p1/summaries/graph_k_means_by_k.csv
  k_R0p1/parquets/   # 54 symlinks → scratch SWEEP
```
Ckpts/run dirs remain in scratch SWEEP `*graph-k-b64-g8-s1000-R0p1*v1lam/`.

---

## Panel (b) — \(R^\star\) @ \(k^\star=6\) — **RUNNING**

| Knob | Value |
|---|---|
| Fixed \(k^\star\) | **6** |
| \(R^\star\) | \{0.05, 0.075, 0.1, 0.125, 0.15\} |
| Cells | 30 |
| Runner | `scans/jets/run_graph_R_lambda8.sh` |
| Runs / logs | `~/runs/graph_R` · `~/logs/graph_R` |
| Pull | `scans/jets/pull_graph_R_lambda.sh` |

First attempt failed on \(R=0.05\) (missing part-int table). Fixed + restarted
2026-10-03 ~11:36 ET. Canvas tracks live progress.

---

## Launch / refresh

```bash
# R★ resume / keep-alive (also done by canvas updater ensure_runner):
nohup bash scans/jets/run_graph_R_lambda8.sh >>~/logs/graph_R/master.log 2>&1 &

python scans/jets/update_graph_k_lambda_live_canvas.py
bash scans/jets/pull_graph_k_lambda.sh
bash scans/jets/pull_graph_R_lambda.sh
python scans/jets/plot_graph_construction_nature.py
```

Daemon: `/tmp/graph_construction_lambda_daemon.sh` → 3m canvas refresh + R keep-alive.

---

## Changelog

| Date | Note |
|---|---|
| 2026-10-02 | Spec locked; k∈{2…16} then refined to odds fill + skip 12+. |
| 2026-10-02 | Data shortfall fixed (2M jets); attentional + part-int confirmed. |
| 2026-10-03 | Panel (a) **54/54**; organized under `graph_construction_lambda_v1lam/`. |
| 2026-10-03 | Panel (b) preprocess fix; R★ sweep restarted; dual-panel canvas. |
| 2026-10-05 | **Panel (c) M11 redo** after encoder-scaling fix: η₀∈{0.01,0.05,0.1,0.25,0.5,1} × D∈{256,384,512}, 2500 steps, B=96×4=384, tag `m11-b96-g4-s2500-dh32`. D=256/384 → gpu-test (`jc-m11-gt`); D=512 → ailab (`jc-m11-a512`). Live canvas `m11-eta-panel-c-live`. Edge arm still held until M11 transfer is sharp. |

## 2026-10-03 — panel (c) archived
Old η₀×D edge/M11 cells @ (k★,R★)=(12,0.1) moved to
`paper/archive/graph_construction_panel_c_20261003_k12_R0p1/`. Live figure
panel (c) is empty pending redo at **k★=6**. See `fig_graph_construction.md`.

## 2026-10-03 — drop k=1 from panel (a)
k=1 removed from the live figure / `K_GRID` (1-NN is degenerate as a multi-neighbor
graph). Trained cells retained in
`paper/data/graph_construction_lambda_v1lam/k_R0p1/summaries/graph_k_val_from_trainlog_incl_k1.csv`.
Live panel (a) is **48/48** on \(k\in\{2,3,4,5,6,7,8,10\}\).

## 2026-10-03 — panel (b) complete; figure redrawn
Panel (b) **30/30** at \(k^\star=6\), \(R^\star\in\{0.05,0.075,0.1,0.125,0.15\}\)
(v1lam). Peak mean ROC at \(R^\star=0.1\) (≈0.9398). Live figure refreshed;
panel (c) still archived.

## 2026-10-03 — full artifact pull (a)+(b) from Lambda

Pulled run dirs, parquets, `best.ckpt`/`last.ckpt`, and stdout logs.

| Panel | Cells | Della SWEEP | Organized tree | Manifest |
|---|---:|---|---|---|
| (a) k @ R★=0.1 | 54 | `…/adam/sweep_lr/*graph-k-b64-g8-s1000-R0p1*v1lam*` | `/scratch/…/graph_construction_lambda_v1lam/k_R0p1/{runs,parquets,checkpoints,logs}` | `paper/data/graph_construction_lambda_v1lam/k_R0p1/summaries/artifact_manifest.csv` |
| (b) R★ @ k★=6 | 30 | `…/adam/sweep_lr/*graph-R-b64-g8-s1000-k6*v1lam*` | `/scratch/…/graph_construction_lambda_v1lam/R_k6/{runs,parquets,checkpoints,logs}` | `paper/data/graph_construction_lambda_v1lam/R_k6/summaries/artifact_manifest.csv` |

Paper-side symlinks: `paper/data/graph_construction_lambda_v1lam/{k_R0p1,R_k6}/{parquets,checkpoints}/`.
Pull script: `scans/jets/pull_graph_construction_v1lam_full.sh`.
Counts verified: **54** k best/last, **30** R best/last (108 / 60 ckpt links).
