# Hyperparameter transfer proof v2 — uniform-pool Adam const-LR η₀×D @ 10k

**Living experimental log.** Nature figure claim for μP LR transfer under
**uniform** output pooling at the locked graph-construction OP.

## Goal

Show that peak LR `η = η₀ / √D` collapses val ROC vs η₀ across widths under
**uniform** pooling, with **no train-data repeats** over the 10k-step budget.

| Item | Value |
|---|---|
| Claim | Adam const-LR η₀×D transfer @ 10k (uniform pool, single-pass data) |
| Status | **10/10 core grid done**; **+4 filling** η₀∈{0.01, 0.18} × D∈{512, 384} |
| Host | `ubuntu@193.122.155.21` (8×A100-40GB) |
| Canvas | `hxfer-apool-eta-constlr-live` (dual attn vs uniform) |
| Runner | `scans/jets/run_hxfer_upool_eta_constlr_lambda8.sh` |
| Pull | `scans/jets/pull_hxfer_upool_eta_constlr_lambda.sh` |
| Plot | `scans/jets/plot_hxfer_upool_eta_constlr.py` |

## Job specs (locked; twin of attn arm)

| Item | Value |
|---|---|
| Model | `capen-llama-att` + **`--uniform-output-pool`** |
| Edges | part-interaction + incidence; star hyperedges on |
| Geometry | **k★ = 6**, **R★ = 0.1** |
| Depth / head dim | L=4, **d_h=32** → H=D/32 |
| Widths | **512 → 384** (twin of attn; D=256 not run for uniform) |
| η₀ grid | {0.05, 0.1, 0.25, 0.5, 1.0} + **extras** {0.01, 0.18} |
| Optimizer | **Adam**, `--scheduler none` (constant peak LR) |
| Peak LR | `η = η₀ / √D` |
| Batch | B_gpu=**64** × **8** → **B_eff=512** |
| Steps | **10 000** |
| Epochs | **1** |
| n_train | **5 120 000** (single pass) |
| Mode | stream |
| n_val / n_test | 25 000 / 0 |
| Seed | 0 |
| Tag | `hxfer-upool-constlr-b64-g8-s10000-dh32-k6-R0p1-v2stream` |
| RAW | `/lambda/nfs/test-jetclass-filesystem` |

## Result (2026-10-05)

| Width | Best η₀ | Val ROC @ 10k |
|---:|---:|---:|
| 384 | **0.1** | 0.95560 |
| 512 | **0.1** | 0.95589 |

Cross-width |ΔROC| ≤ ~6×10⁻⁴ for η₀ ≤ 0.5; η₀=1 is a cliff (sharper on D=384).
Both widths share the same peak η₀ → **μP LR transfer under uniform pooling**.

## Nature figure

| Artifact | Path |
|---|---|
| Figure | `paper/figures/fig_hparam_transfer_proof_upool10k.{png,pdf,eps}` |
| Caption | `paper/figures/fig_hparam_transfer_proof_upool10k_caption.txt` |
| CSV | `paper/data/hxfer_upool_eta_constlr.csv` |
| Org + manifests | `paper/data/hxfer_constlr_v2stream/` (see README) |

## Paths (Lambda → Della)

| Kind | Location |
|---|---|
| Lambda runs | `~/runs/hxfer_upool_eta/jetclass_output/adam/sweep_lr/` |
| Lambda logs | `~/logs/hxfer_upool_eta/` |
| Della SWEEP (canonical) | `/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/jetclass_output/adam/sweep_lr/*hxfer-upool*v2stream*` |
| Della org (symlinks) | `/scratch/.../hxfer_constlr_v2stream/upool/{runs,parquets,checkpoints,logs}` |
| Paper logs / summaries | `paper/data/hxfer_constlr_v2stream/upool/` |

**Pull inventory (2026-10-05):** 10 parquets, 10×(best+last).ckpt, ~7.4 G runs + 721 K logs.

## Changelog

| Date | Note |
|---|---|
| 2026-10-04 | Launched twin of attn arm after stopping D=256 attn; uniform D=512→384. |
| 2026-10-05 | All 10 cells complete; Nature figure drawn; full Lambda→Della pull + org. |
| 2026-10-05 | Extra cells launched: η₀∈{0.01, 0.18} × D∈{512, 384} (same v2stream tag; seq on 8×A100). Master log `~/logs/hxfer_upool_eta/master_extra_eta0p01_0p18.log`. |
| 2026-10-05 | TopTagging FT (gpu-test): init from D=512 η₀=0.1 `last.ckpt` → `…/toptagging_finetune/init_ckpts/hxfer-upool-D512-eta0p1-v2stream-last.ckpt`; submit `scans/jets/submit_toptagging_ft_D512_upool_gputest.sh` (matched k★=6 R★=0.1 + uniform pool, 5k steps @ B=128×4). |
| 2026-10-05 | Pulled/refreshed upool fig: **12** done cells @10k (incl. latest extras). Peak η₀=0.1 both widths. |
| 2026-10-05 | Added intermediate-step copies from Lambda logs: `fig_hparam_transfer_proof_upool{2,4,6,8}k` (+ CSVs `hxfer_upool_eta_constlr_{2,4,6,8}k.csv`; curve source `hxfer_upool_eta_val_curve_from_logs.csv`). |
