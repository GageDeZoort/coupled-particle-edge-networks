# Hyperparameter transfer proof v2 — attentional Adam const-LR η₀×D @ 10k

**Living experimental log.** Attn-pool arm of the v2stream hxfer campaign.
**Primary Nature figure is the uniform twin** — see
`fig_hparam_transfer_proof_upool10k.md`.

## Goal

Show that peak LR `η = η₀ / √D` collapses val ROC vs η₀ across widths under
**attentional** pooling at the locked graph-construction operating point, with
**no train-data repeats** over the 10k-step budget.

| Item | Value |
|---|---|
| Claim | Adam const-LR η₀×D transfer @ 10k steps (attn pool, single-pass data) |
| Status | **frozen** at **12/15** (D=256 η₀∈{0.25,0.5,1.0} stopped; soft transfer) |
| Uniform twin | **complete 10/10** — sharper shared peak η₀=0.1 (Nature fig) |
| Host | `ubuntu@193.122.155.21` (8×A100-40GB) |
| Canvas | `hxfer-apool-eta-constlr-live` |
| Runner | `scans/jets/run_hxfer_apool_eta_constlr_lambda8.sh` |
| Pull | `scans/jets/pull_hxfer_apool_eta_constlr_lambda.sh` |
| Logs pull | `scans/jets/pull_hxfer_logs_lambda.sh` |

## Job specs (locked)

| Item | Value |
|---|---|
| Model | `capen-llama-att` — **attentional** pool (no `--uniform-output-pool`) |
| Edges | part-interaction + incidence; star hyperedges on |
| Geometry | **k★ = 6**, **R★ = 0.1** |
| Depth / head dim | L=4, **d_h=32** → H=D/32 |
| Widths (order) | **512 → 384 → 256** (largest first) |
| η₀ grid | {0.05, 0.1, 0.25, 0.5, 1.0} |
| Optimizer | **Adam**, `--scheduler none` (constant peak LR) |
| Peak LR | `η = η₀ / √D` |
| Batch | B_gpu=**64** × **8** → **B_eff=512** |
| Steps | **10 000** (`--max-steps 10000`) |
| Epochs | **1** (must stay 1 so `max_epochs` cannot bind early) |
| n_train | **5 120 000** = `max_steps × B_eff` (single pass, no repeats) |
| Mode | stream (`n_train > RAM_SUBSET_LIMIT=2e6`) |
| n_val / n_test | 25 000 / 0 |
| Seed | 0 |
| Val interval | every 500 steps |
| Ckpt | every 1000 steps; monitor `val_loss` |
| Tag | `hxfer-apool-constlr-b64-g8-s10000-dh32-k6-R0p1-v2stream` |
| RAW | `/lambda/nfs/test-jetclass-filesystem` (staged from Della `/projects/j/jdezoort/jetclass`) |
| Train files staged | **8 / class × 10 classes = 80** ROOT files (≈8.0M jets @ 100k/file) |
| Val files staged | **50** (`val_5M`, full) |

### Task map (sequential on 8×A100)

| TASK | D | η₀ |
|---:|---:|---:|
| 0–4 | 512 | 0.05, 0.1, 0.25, 0.5, 1.0 |
| 5–9 | 384 | 0.05, 0.1, 0.25, 0.5, 1.0 |
| 10–14 | 256 | 0.05, 0.1, 0.25, 0.5, 1.0 |

**Completed:** tasks 0–11 (D=512 all five; D=384 all five; D=256 η₀=0.05, 0.1).
**Stopped:** D=256 η₀∈{0.25, 0.5, 1.0} — transfer under attn was soft/flat;
GPUs reallocated to uniform twin.

## Paths

| Kind | Location |
|---|---|
| Lambda runs | `~/runs/hxfer_apool_eta/jetclass_output/adam/sweep_lr/` |
| Lambda logs | `~/logs/hxfer_apool_eta/` (`D{D}_eta{η}_R0p1.out`, `master.log`) |
| Bad 1M-repeat archive | `~/runs|logs/hxfer_apool_eta/archive_repeat1M_20261003/` |
| Della SWEEP (canonical) | `/scratch/.../jetclass_output/adam/sweep_lr/*hxfer-apool-constlr*v2stream*` |
| Della org (symlinks) | `/scratch/.../hxfer_constlr_v2stream/apool/{runs,parquets,checkpoints,logs}` |
| Paper logs / summaries | `paper/data/hxfer_constlr_v2stream/apool/` |
| Live JSON / CSV | `paper/data/hxfer_apool_eta_constlr_live.{json,csv}` |
| Metrics CSV | `paper/data/hxfer_apool_eta_constlr.csv` |

**Pull inventory (2026-10-05):** 12 parquets @10k, 13 run dirs (extra = mid-stop
D=256 η₀=0.25 with ckpts only), ~8.5–9.0 G + 1.2 M logs.

## Bug (2026-10-03) — why jobs stopped ~3908

1. Lambda only had a **tiny** JetClass subset (2 files/class).
2. Runner used `n_train=1M` → **ram-subset (repeatable)**.
3. Runner set `epochs=2` with ~1953 steps/epoch → Lightning hit
   **`max_epochs=2` at step ≈3908**, ignoring `max_steps=10000`.

Fix: stage ≥5.12M unique jets, set `n_train=5_120_000`, `epochs=1`, new
`v2stream` tag, width order 512→384→256.

## Preliminary figure (attn)

| Artifact | Path |
|---|---|
| Figure | `paper/figures/fig_hparam_transfer_proof_apool10k.{png,pdf,eps}` |
| CSV | `paper/data/hxfer_apool_eta_constlr.csv` |
| Plot script | `scans/jets/plot_hxfer_apool_eta_constlr.py` |

12/15 cells at 10k; transfer softer than uniform (flat η₀ response).

## Uniform twin (Nature)

| Artifact | Path |
|---|---|
| Log | `paper/figures/fig_hparam_transfer_proof_upool10k.md` |
| Figure | `paper/figures/fig_hparam_transfer_proof_upool10k.{png,pdf,eps}` |
| CSV | `paper/data/hxfer_upool_eta_constlr.csv` |
| Org | `paper/data/hxfer_constlr_v2stream/README.md` |

## Changelog

| Date | Note |
|---|---|
| 2026-10-03 | Spec locked; first launch. |
| 2026-10-03 | Path fix → `test-jetclass-filesystem`; n_train briefly 1M (repeat). |
| 2026-10-03 | Diagnosed early stop = `max_epochs=2` × 1953 steps/epoch. |
| 2026-10-03 | **v2stream restart:** rsync 80 train + 50 val ROOT files; n_train=5.12M; epochs=1; D order 512→384→256; archive repeat runs. |
| 2026-10-04 | Preliminary attn figure (12/15); D=256 remaining cells stopped; uniform twin launched. |
| 2026-10-05 | Uniform complete; Nature fig; full Lambda→Della pull (attn+uniform runs/ckpts/logs) + org under `hxfer_constlr_v2stream/`. |
| 2026-10-05 | Refreshed apool fig from Della SWEEP (pull no new cells). |
