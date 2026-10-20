# Hyperparameter transfer proof — experimental log

**Goal.** Show that (a) learning rate and (b) weight decay transfer across
width at fixed geometry, using CAPEN with the locked JetClass operating point.

| Plot panel | Claim | Status |
|---|---|---|
| (a) LR transfer | Peak LR `η = η₀ / √D` collapses val ROC vs η₀ across model size | **latest: 40k Adam @ frozen OP** — see `fig_hparam_transfer_proof_edge_eta_s40k.md` |
| (b) WD transfer | AdamW WD from EMA timescale `t` collapses val ROC vs `t` across widths | **do not claim yet** — AdamW t_epoch @ L4/D256 not beating Adam (see `fig_adamw_tepoch_L4D256_s40k.md`) |

**Standing ops rule:** whenever I think Slurm is unreachable, it is not.

---

## Decision (2026-10-01): pause width `t`-sweep; test WD ∥ schedule

Arm 1b (D=512, 5k flat → cosine, cutoff 12k) does **not** support claiming
WD / `t`-transfer yet:

| Observation | Detail |
|---|---|
| Optimal `t` flips with width | D=256 best at **t=10** (ROC 0.9572); D=512 best at **t=1** (ROC 0.9570) |
| Cosine may confound WD | At D=512, t=5/10 barely move (or drop) after the 5k flat; t=1 keeps climbing through the cosine window |
| Hypothesis | We are annealing too early; any “transfer” claim at 12k mixes schedule × WD |

**Schedule target (locked in).** Do **not** start cosine until val ROC is near
the long-run plateau ≈ **0.96**. Historical feel (same η₀=0.25, CAPEN):

| Phase | val ROC | Steps (rough) |
|---|---:|---|
| Arm 1 anneal start (old) | ~0.95 | 5k — **too early** |
| Mid-climb | ~0.954–0.958 | ~10k–25k |
| Near plateau | **~0.96** | typically **≳30k–45k** on prior D256 gputest (k=6/R★=0.2); confirm on k★=12/R★=0.1 via Arm 2 const-LR |

So: long constant-LR (or flat) until ~0.96, **then** cosine — not 5k flat +
early anneal.

**Action taken**

1. **Paused** D=512 t=20 mid-run (~step 1400). Marker: Lambda / Della
   `logs/PAUSED_D512_T_SWEEP` (`PAUSED_T20`).
2. **Pulled** all Arm 1 / 1b artifacts to Della `hparam_xfer_lambda/`.
3. **Next experiment (Arm 2):** constant LR (`--scheduler none`), **10k steps**,
   D=256 / d_h=32 / η₀=0.25 — AdamW at t∈{1,5,10} **plus** Adam-only (WD=0).
   Isolates whether AdamW / `t`-WD helps independent of cosine. (10k is still
   mid-climb; use it for WD comparison, not as the anneal trigger.)

Do **not** claim panel (b) until Arm 2 (and a longer flat to ~0.96) clarifies
the confound.

---

## Locked geometry / model

| Item | Value |
|---|---|
| Model | `capen-llama-att` (part-interaction edges; M12/M21 + hyperedge M22) |
| Features | kin7, 80 particles |
| Geometry | **k★=12, R★=0.1** (from `fig_graph_construction`) |
| Depth / heads | L=4, **d_h=32** → H=D/32 |
| Optimizer (Arms 1–1b) | **AdamW** |
| Peak LR | `η = η₀ / √D` with **η₀ = 0.25** |
| Weight decay | `λ = √D · B_eff / (t · η₀ · n_train)` via `--t-epoch t` |
| Host | Lambda **8×A100-80GB** (`ubuntu@163.192.125.147`) |

### `t_epoch` ↔ λ (reference)

Prescribed `n_train = 100M`, `B_eff = 512`:

```
λ = √D · B_eff / (t · η₀ · n_train)
λη = B_eff / (t · n_train)     # width-invariant at fixed t
```

| D | t | λ (logged) |
|---:|---:|---:|
| 256 | 1 | 3.2768×10⁻⁴ |
| 256 | 5 | 6.5536×10⁻⁵ |
| 256 | 10 | 3.2768×10⁻⁵ |
| 256 | 20 | 1.6384×10⁻⁵ |
| 512 | 1 | 4.6341×10⁻⁴ |
| 512 | 5 | 9.2682×10⁻⁵ |
| 512 | 10 | 4.6341×10⁻⁵ |
| 512 | 20 | 2.3170×10⁻⁵ |

Wiring check: logged `weight_decay` matches the formula (not a scale bug).

---

## Training procedure — Arm 1 / 1b (plot cutoff = 12k steps)

| Quantity | Value |
|---|---|
| One epoch | `EPOCH = ceil(1e8 / 512) = **195 313**` (prescribed 100M) |
| Warmup | **0** |
| Constant (peak) LR | **5 000** |
| Cosine **schedule** length | `EPOCH − 5k = **190 313**` |
| Cosine **run** | **7 000** (steps 5k–12k) |
| **`max_steps` (cutoff)** | **12 000** |
| `n_train` (prescribed D) | **100 000 000** |
| Effective batch | **B_eff = 512** = 64 × 8 |
| Tag base | `hxfer-t-b64-g8-s12000-k12-R0p1` |
| Scheduler | `warmup_flat_cosine` |

---

## Arm 1 — `t` sweep at η₀=0.25, D=256 (done)

| `t` | Status | val ROC @5k | val ROC @12k | WD | Notes |
|---:|---|---:|---:|---:|---|
| 1 | **done** | 0.9475 | **0.955525** | 3.2768×10⁻⁴ | still climbing in cosine |
| 5 | **done** | 0.9532 | **0.957053** | 6.5536×10⁻⁵ | |
| 10 | **done** | 0.9532 | **0.957215** | 3.2768×10⁻⁵ | **best** of the four |
| 20 | **done** | 0.9502 | **0.951090** | 1.6384×10⁻⁵ | worse than t=5/10 |

### Pulled artifacts (Della)

Root: `/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/hparam_xfer_lambda/`

| Kind | Path |
|---|---|
| CSV | `paper/data/hparam_xfer_t_d256.csv` |
| Logs | `hparam_xfer_lambda/logs/D256_eta0p25_t{1,5,10,20}_R0p1.out` |
| Runs | `hparam_xfer_lambda/runs/capen-llama-att_4_256_0p25_t{T}_…-s12k-…-t{T}-w0f5kc190313min0p01/` |

**Checkpoints (local copies):**

| t | best.ckpt | last.ckpt |
|---:|---|---|
| 1 | present | present |
| 5 | present | present |
| 10 | **missing on Lambda** (dir deleted; metrics from log only) | — |
| 20 | present | present |

Full absolute paths are in `paper/data/hparam_xfer_t_d256.csv`.

---

## Arm 1b — D=512 (d_h=32 → H=16), same recipe — **PAUSED**

| Item | Value |
|---|---|
| Width | **D=512**, H=16 |
| `t` order | **1 → 5 → 10 → 20** (`ADAPTIVE_AFTER_T10=0`) |
| Master log | `hparam_xfer_lambda/logs/master_d512.log` |
| Pause marker | `hparam_xfer_lambda/logs/PAUSED_D512_T_SWEEP` (`PAUSED_T20`, 2026-10-01T12:55:29Z) |
| CSV | `paper/data/hparam_xfer_t_d512.csv` |
| Wall clock | ~2.9 h / cell (~820 ms/step); ~2.3× slower than D=256 (compute-bound) |

| `t` | Status | val ROC @5k | val ROC @12k | WD | Notes |
|---:|---|---:|---:|---:|---|
| 1 | **done** | 0.9509 | **0.956963** | 4.6341×10⁻⁴ | **best** at D=512; still climbing in cosine |
| 5 | **done** | 0.9534 | **0.953458** | 9.2682×10⁻⁵ | flat through cosine window |
| 10 | **done** | 0.9537 | **0.953302** | 4.6341×10⁻⁵ | slight **drop** after 5k |
| 20 | **paused @1400** | — | — | 2.3170×10⁻⁵ | killed intentionally; last val @1k ROC=0.9416; ckpts kept |

### Pulled artifacts (Della)

| Kind | Path |
|---|---|
| Logs | `logs/D512_eta0p25_t{1,5,10,20}_R0p1.out`, `master_d512.log`, `PAUSED_D512_T_SWEEP` |
| Runs t=1,5,10 | dirs + `.parquet` + `best.ckpt`/`last.ckpt` |
| Run t=20 | dir + `best.ckpt`/`last.ckpt` (**no** parquet; incomplete) |

### Cross-width snapshot @12k (do **not** treat as transfer proof)

| t | D=256 ROC | D=512 ROC | Δ (512−256) |
|---:|---:|---:|---:|
| 1 | 0.9555 | **0.9570** | +0.0014 |
| 5 | **0.9571** | 0.9535 | −0.0036 |
| 10 | **0.9572** | 0.9533 | −0.0039 |
| 20 | 0.9511 | — (paused) | — |

Best-`t` mismatch + cosine plateau at D=512 t≥5 → WD panel on hold.

---

## Arm 2 — constant LR, D=256: AdamW `t` × Adam (**PAUSED**)

**Question.** Does AdamW / `t`-epoch WD help at D=256 **independently of cosine**?
Suspect: 5k flat → cosine@12k anneals too early.

| Item | Value |
|---|---|
| Width / heads | **D=256**, H=8 (d_h=32) |
| Peak LR | η₀=0.25 → `η = 0.25/√256` (held **constant**) |
| Scheduler | **`none`** (no cosine / no warmup) |
| `max_steps` | **10 000** |
| Geometry / batch | same as Arm 1 (k★=12, R★=0.1, B_eff=512, n_train=100M prescribed) |
| Queue | AdamW t∈{**1, 5, 10**} then **Adam** (WD=0) η₀=0.25 |
| Tag base | `hxfer-constlr-b64-g8-s10000-k12-R0p1` |
| Runner | `scans/jets/run_hparam_xfer_constlr_lambda8.sh` |
| Logs (Lambda) | `~/logs/hparam_xfer/D256_eta0p25_constlr_{adamw_t*,adam}_R0p1.out` |

| Cell | Status | Notes |
|---|---|---|
| AdamW t=1 | **done** @10k ROC **0.9527** | |
| AdamW t=5 | **done** @10k ROC **0.9525** | |
| AdamW t=10 | **paused** @step **6700** ROC **0.9538** | killed 2026-10-01 to free Lambda |
| Adam (no WD) | **skipped** | never started |

Pause marker: `~/logs/hparam_xfer/PAUSED_ARM2_D256_CONSTLR`.

---

## Arm 2c — Adam const LR @ **D=1024** on Lambda (**running**)

**Question.** How does Adam (WD=0) behave at large width under the same μP
LR / batch / geometry?

| Item | Value |
|---|---|
| Host | Lambda **8×A100-80GB** |
| Width / heads | **D=1024**, H=32 (d_h=32) |
| Optimizer | **Adam** (WD=0) |
| Peak LR | η₀=0.25 → constant `η = 0.25/√1024` |
| Scheduler | **`none`** |
| `max_steps` | **10 000** |
| Batch | **B_eff = 512** = 64 × 8 |
| Geometry | k★=12, R★=0.1 |
| Tag | `hxfer-constlr-b64-g8-s10000-k12-R0p1-adam` |
| Log | `~/logs/hparam_xfer/D1024_eta0p25_constlr_adam_R0p1.out` |
| Master | `~/logs/hparam_xfer/master_constlr_d1024.log` |
| Port | `MASTER_PORT=29640` |

Launched after Arm 2 pause. Fits ~48–52 GiB/GPU at start.

---

## Arm 2b — const LR, D=384 on Della **ailab** (compare to Lambda D=256)

| Item | Value |
|---|---|
| Host | Della **ailab** (H200), account `bhanin` |
| Width / heads | **D=384**, H=12 (d_h=32) |
| Peak LR | η₀=0.25 → constant `η = 0.25/√384` |
| Scheduler | **`none`** |
| `max_steps` | **10 000** (~3 h / cell expected) |
| Batch | **B_eff = 512** = 128 × 4 |
| Geometry | k★=12, R★=0.1; n_train=100M prescribed for WD |
| Queue | Adam (WD=0) + AdamW t∈{1,5,10} |
| Tag base | `hxfer-constlr-b128-g4-s10000-k12-R0p1` |
| Slurm | `scans/jets/scaling_law_L4_hxfer_constlr_D384_ailab.slurm` |
| Logs | `…/logs/jc_hxfer_c384_ailab_%A_%a.{out,err}` |

Direct comparison to Lambda Arm 2 (D=256, same B_eff / schedule / t grid).

---

## Arm 3 — Adam η₀×D fill-in on Della **ailab** (panel a)

**Question.** With Adam (WD=0) + const LR, does `η = η₀/√D` transfer across
width at fixed geometry? Fill missing η₀ points around Arm 2/2b’s η₀=0.25.

| Item | Value |
|---|---|
| Host | Della **ailab** (H200), account `bhanin` |
| Optimizer | **Adam** (WD=0) |
| Scheduler | **`none`** (constant LR) |
| `max_steps` | **10 000** |
| Batch | **B_eff = 512** = 128 × 4 |
| Geometry | k★=12, R★=0.1; L=4, d_h=32 |
| Grid | D ∈ {**256, 384**} × η₀ ∈ {**0.05, 0.1, 0.25, 0.5, 1.0**} |
| η₀=0.25 | D=384 from Arm 2b Adam; **D=256 via Arm 3 task 8** (Arm 2 Adam was skipped) |
| Tag base | `hxfer-adam-eta-b128-g4-s10000-k12-R0p1` |
| Root | `/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law` |
| Slurm | `scans/jets/scaling_law_L4_hxfer_adam_eta_ailab.slurm` (`--array=8-8` for D256 η=0.25) |
| Logs | `…/logs/jc_hxfer_adam_eta_ailab_%A_%a.{out,err}` |
| Live CSV | `paper/data/hparam_xfer_adam_eta_live_status.csv` |
| Canvas | `edge-eta-lambda-live` · Arm 3 section |

Array map (`task = 4·d_idx + e_idx`; task **8** = D256 η₀=0.25):

| task | D | η₀ |
|---:|---:|---:|
| 0 | 256 | 0.05 |
| 1 | 256 | 0.1 |
| 2 | 256 | 0.5 |
| 3 | 256 | 1.0 |
| 4 | 384 | 0.05 |
| 5 | 384 | 0.1 |
| 6 | 384 | 0.5 |
| 7 | 384 | 1.0 |
| **8** | **256** | **0.25** |

Primary metric: **val ROC @10k** (fixed horizon). Also track late-window max as a
robustness check — do not promote max to primary for transfer claims.

---

## Later arms (blocked on Arm 2 / 2b / 3)

1. If const-LR shows clear AdamW/`t` benefit: revisit schedule (longer flat /
   later cosine) before re-running D=512 `t`-sweep.
2. D=384 / D=512 WD transfer only after schedule × WD is clean **and** AdamW
   beats Adam.
3. Panel (a) LR transfer: Arm 3 fills η₀ grid under Adam; claim only if curves
   collapse across D.
4. Aggregate → Nature-style 2-panel figure — **only if transfer holds**.

---

## Ops

| Item | Path |
|---|---|
| Arm 1 runner | `scans/jets/run_hparam_xfer_t_lambda8.sh` |
| Arm 2 runner | `scans/jets/run_hparam_xfer_constlr_lambda8.sh` |
| Arm 2b slurm | `scans/jets/scaling_law_L4_hxfer_constlr_D384_ailab.slurm` |
| Arm 3 slurm | `scans/jets/scaling_law_L4_hxfer_adam_eta_ailab.slurm` |
| Logs (Lambda) | `~/logs/hparam_xfer/` |
| Runs (Lambda) | `~/runs/hparam_xfer/jetclass_output/{adamw,adam}/sweep_lr/` |
| Della root | `…/jetclass_scaling_law/` (`jetclass_output/`, `logs/`) |
| Della mirror (Arm 1) | `…/jetclass_scaling_law/hparam_xfer_lambda/{runs,logs}/` |
| Live CSV | `paper/data/hparam_xfer_t_live_status.csv` |
| Arm 1 CSV | `paper/data/hparam_xfer_t_d256.csv` |
| Arm 1b CSV | `paper/data/hparam_xfer_t_d512.csv` |
| Arm 3 CSV | `paper/data/hparam_xfer_adam_eta_live_status.csv` |

### Launch (on Lambda)

```bash
# Arm 1 (D=256) — complete:
T_EPOCHS="5 10 20" bash scans/jets/run_hparam_xfer_t_lambda8.sh
# Arm 1b (D=512) — PAUSED after t=1,5,10 done; t=20 killed @~1400:
# WIDTH=512 HEAD=16 ADAPTIVE_AFTER_T10=0 T_EPOCHS="1 5 10 20" \
#   MASTER_PORT=29620 bash scans/jets/run_hparam_xfer_t_lambda8.sh

# Arm 2 (const LR, D=256) — PAUSED after t=1,5 done; t=10 killed @6700:
# MASTER_PORT=29630 bash scans/jets/run_hparam_xfer_constlr_lambda8.sh

# Arm 2c (Adam const LR, D=1024) — current:
WIDTH=1024 HEAD=32 ONLY=adam MASTER_PORT=29640 \
  bash scans/jets/run_hparam_xfer_constlr_lambda8.sh
```

Anti-stall defaults: `NCCL_P2P_DISABLE=1`, `NCCL_IB_DISABLE=1`, `NUM_WORKERS=2`, `FRESH=1`.

---

## Changelog

| Date | Note |
|---|---|
| 2026-09-30 | Created. Locked k★=12, R★=0.1, η₀=0.25, B_eff=512 via 64×8. |
| 2026-09-30 | **Recipe v3:** 5k flat → cosine; cutoff 12k. Order t=5,10,20 at D=256. |
| 2026-09-30 | **Bug:** WD used 20M available jets → λ 5× too large. Fix: prescribe `n_train=100M`. |
| 2026-09-30 | Standing ops rule: Slurm “unreachable” is usually a client glitch. |
| 2026-10-01 | D=256 t∈{1,5,10,20} complete. Best t=10 ROC=0.9572. Pulled runs+ckpts (t=1,5,20) + all logs to Della `hparam_xfer_lambda/`; CSV `paper/data/hparam_xfer_t_d256.csv`. **t=10 run dir missing on Lambda** (log retained). |
| 2026-10-01 | Launched D=512 H=16 t=1→5→10→20. Canvas shows D×t; fixed t=1/t=10 path-prefix confusion. |
| 2026-10-01 | D=512 t=1/5/10 **done**; best **t=1** (0.9570). t=5/10 plateau or drop after 5k flat. **Paused** t=20 @~1400. Pulled all to Della; CSV `hparam_xfer_t_d512.csv`. **Do not claim WD transfer.** Start Arm 2: const-LR 10k AdamW t∈{1,5,10} + Adam @ D=256. |
| 2026-10-01 | Schedule target: flat until val ROC ≈ **0.96**, then cosine. Arm 2b: D=384 const-LR 10k on **ailab** (Adam + AdamW t∈{1,5,10}, B=128×4). |
| 2026-10-01 | **Arm 3:** Adam η₀×D fill-in on ailab — D∈{256,384} × η₀∈{0.05,0.1,0.5,1.0}, const-LR 10k, B=128×4. Panel (a) started; η₀=0.25 from Arm 2/2b. Canvas + `hparam_xfer_adam_eta_live_status.csv`. |
| 2026-10-01 | **Paused Arm 2** D=256 (t=1/5 done; t=10 killed @6700 ROC 0.9538; Adam skipped). **Arm 2c:** D=1024 Adam const-LR 10k on Lambda 8×A100, B=64×8, η₀=0.25. Canvas + proof log updated. |
| 2026-10-01 | Arm 3 fill-in: submitted task **8** (D=256 η₀=0.25 Adam) — was missing after Arm 2 pause; D=384 η₀=0.25 already from Arm 2b. |
| 2026-10-14 | **Latest panel (a):** organized L4/D256 + L6/D512 Adam η₀ @40k (frozen k★=7 noloop) into `fig_hparam_transfer_proof_edge_eta_s40k`. AdamW t_epoch @ L4/D256 not beating Adam → WD claim still held. |
| 2026-10-19 | **L3/D192 arm** added to transfer-proof (η₀∈{0.01,0.05,0.1,0.25,0.5} @40k, B=64×4 matched). Live canvas `edge-eta-L3D192-k7-s40k-live`; log `fig_graph_construction_edge_eta_L3D192_s40k.md`. |
| 2026-10-19 | TopTagging full FT (L4/D256 from cosine pretrain): frozen-head cancelled; full η₀≤0.1 rolling. Val AUC ~0.9871 vs ParT-f.t. 0.9877. **Full test eval** of η₀=0.025 `best.ckpt` submitted (`eval_toptag_ft_test_gputest.slurm`) for AUC + Rej₅₀%/Rej₃₀%. Next FT recipe to try: **AdamW + cosine** (ParT-style dual LR also on the table). |
| 2026-10-19 | **Test η₀=0.025 (Adam const):** AUC **0.9873** / Rej₅₀%=**554** / Rej₃₀%=**2499** on 400k (ParT-f.t. 0.9877 / 691 / 2766). Val Rej=1985 was n_bg/5 discreteness, not a bug. **New probe:** AdamW `t_epoch=25` + epoch cosine → 0.01× peak over 10 ep, η₀∈{0.01,0.025,0.05} (`finetune_toptagging_L4cos_full_adamw_cos_gputest.slurm`). |

---

## 2026-10-14 — latest: Adam η₀ @ 40k, frozen OP (k★=7)

**Current** LR-transfer evidence before shipping a giant model:

- Spec / living log: **`fig_hparam_transfer_proof_edge_eta_s40k.md`**
- Figure: `fig_hparam_transfer_proof_edge_eta_s40k.{png,pdf,eps}`
- Geometry: **k★=7, R★=0.1, noloop**, B=64×4=256, **40k** Adam, single-pass 10.24M (L3/D192 + L4/D256 + L6/D512)
- Models: **L4/D256** (9/9 done; peak η₀=0.1, ROC 0.9557) · **L6/D512** (rolling)
- Canvases: `edge-eta-D256-k7-s20k-live` · `edge-eta-L6D512-k7-s40k-live`
- Plot: `scans/jets/plot_hxfer_edge_eta_s40k.py`

Prior 10k Nature arms (k★=6 upool/apool) remain as archive; do not treat as the
active OP for the next scale-up.

---

## 2026-10-03 — v2 attentional Adam const-LR η₀×D @ 10k

New transfer-proof campaign (supersedes Arm-3 scraps for the *new* figure version):

- Spec / living log: **`fig_hparam_transfer_proof_apool10k.md`**
- Geometry: **k★=6, R★=0.1**, attentional pool, B=64×8=512, 10k const-LR Adam
- Grid: D∈{256,384,512} × η₀∈{0.05,0.1,0.25,0.5,1.0} (15 cells) on Lambda
- Canvas: `hxfer-apool-eta-constlr-live`
