# Uniform-pool μP transfer validation (debug) — experimental log

**Status:** scale-check only — **not** a paper figure. Pins that
CompleteP + Adam \(\eta=\eta_0/\sqrt{D}\) + **uniform output pool** transfers
across width on a short JetClass run. Attentional-pool twin on Lambda.

Related: one-step stability `paper/figures/fig_stability.md` /
`COMPLETEP_SCALES.md`.

---

## Recipe

| Knob | Value |
|---|---|
| Model | `capen-llama-att` + `--uniform-output-pool` |
| \(L\), \(d_h\) | 4, 32 → \(H=D/32\) |
| Geometry | **\(k=10\)**, \(R^\star=0.1\) (note: stability paper uses \(k^\star=12\)) |
| Batch | \(B_\mathrm{gpu}=96\times 4\) → \(B_\mathrm{eff}=384\) |
| Opt | Adam, \(\eta=\eta_0/\sqrt{D}\), default \(t\)-epoch schedule |
| Horizon | **1000** steps |
| Grid | \(D\in\{256,512\}\) × \(\eta_0\in\{0.05,0.1,0.25,0.5,1.0\}\) |
| Partition | **gputest** (≤59m via `a100`) — both \(D=256\) and \(D=512\) |

**Slurm:** `scans/jets/scaling_law_L4_upool_xfer_validate_gputest.slurm`  
Array `0–9`: `width_idx=task//5`, `eta_idx=task%5`.  
**Tag:** `upool-xfer-b96-g4-s1000-dh32-k10-R0p1`

| Width | Job | Array tasks |
|---|---|---|
| \(D=256\) | **`14882958`** | `0–4` **DONE** |
| \(D=512\) | **`14890706`** | `5–9` (ailab twin `14885124` cancelled; moved to gputest) |

**Debug plot**
```bash
python scans/jets/plot_upool_xfer_validate.py
# → paper/data/upool_xfer_validate.csv
# → paper/figures/fig_upool_xfer_validate_debug.{png,pdf}
```

**Pass intuition.** Peak \(\eta_0\) (and ROC vs \(\eta_0\) shape) should roughly
align for \(D=256\) and \(D=512\). Misaligned optima → scaling bug or schedule /
batch confound — dig before trusting paper stability claims for training.

---

## Live canvas

Open beside chat: [`upool-xfer-validate-live.canvas.tsx`](/home/jdezoort/.cursor/projects/home-jdezoort/canvases/upool-xfer-validate-live.canvas.tsx)

Shows **both** panels: Della uniform-pool + Lambda attentional-pool.

Refresh (manual):
```bash
python scans/jets/update_upool_xfer_live_canvas.py
```
Agent loop refreshes every **3 minutes** while this scan is active.

---

## Results (fill as cells finish)

Snapshot **2026-10-02 ~15:26 ET** (best val ROC @ ≤1000 steps).

| \(D\) | \(\eta_0\) | best val ROC | step | notes |
|---|---|---|---|---|
| 256 | 0.05 | 0.9307 | 1000 | gputest `14882958_0` |
| 256 | 0.1 | 0.9330 | 1000 | |
| 256 | 0.25 | **0.9357** | 1000 | D256 peak |
| 256 | 0.5 | 0.9351 | 1000 | |
| 256 | 1.0 | 0.9335 | 1000 | |
| 512 | 0.05 | | 50+ | **running** `14890706_5` |
| 512 | 0.1 | | | pending gputest |
| 512 | 0.25 | | | pending gputest |
| 512 | 0.5 | | | pending gputest |
| 512 | 1.0 | | | pending gputest |

---

## Attentional-pool twin (Lambda)

Same grid / horizon as above, but **class-token attentional pool** (no
`--uniform-output-pool`) on Lambda `193.122.155.21` (8×A100-40GB).

| Knob | Value |
|---|---|
| Model | `capen-llama-att` (default attentional pool) |
| Batch | \(B_\mathrm{gpu}=48\times 8\) → \(B_\mathrm{eff}=384\) |
| Geometry | \(k=10\), \(R^\star=0.1\), \(L=4\), \(d_h=32\) |
| Horizon | **1000** steps |
| Grid | \(D\in\{256,512\}\) × \(\eta_0\in\{0.05,0.1,0.25,0.5,1.0\}\) |
| Host | Lambda `ubuntu@193.122.155.21` |
| Runner | `scans/jets/run_apool_xfer_validate_lambda8.sh` |
| Tag | `apool-xfer-b48-g8-s1000-dh32-k10-R0p1` |
| Runs | `~/runs/apool_xfer` · logs `~/logs/apool_xfer` |

### Attn results (fill as cells finish)

| \(D\) | \(\eta_0\) | best val ROC | step | notes |
|---|---|---|---|---|
| 256 | 0.05 | 0.9365 | 1000 | |
| 256 | 0.1 | 0.9363 | 1000 | |
| 256 | 0.25 | **0.9374** | 1000 | D256 peak; beats uniform |
| 256 | 0.5 | 0.9362 | 1000 | |
| 256 | 1.0 | 0.9343 | 1000 | |
| 512 | 0.05 | 0.9369 | 1000 | |
| 512 | 0.1 | **0.9370** | 1000 | |
| 512 | 0.25 | ~0.929 | 400 | **running** |
| 512 | 0.5 | | | pending |
| 512 | 1.0 | | | pending |

**Early read.** At 1000 steps, attn D256 peak (**0.937**) > uniform D256 peak (**0.936**);
D512 η₀=0.05/0.1 already match D256 peak — η₀ transfer looks healthy so far.

---

## Changelog

| Date | Note |
|---|---|
| 2026-10-02 | Created; attn paper array cancelled to focus on uniform. |
| 2026-10-02 | Uniform stability D1536/D2048 still finishing (`14877878_7/8`). |
| 2026-10-02 | Attn-pool xfer twin launched on Lambda 8×A100 (`193.122.155.21`), B=48×8. |
| 2026-10-02 | Uniform D256 (**`14882958`**) all 5 η₀ **done**; peak η₀=0.25 → 0.9357. |
| 2026-10-02 | Uniform D512 moved off ailab (`14885124` cancelled, QOS/Priority stall) → gputest **`14890706_[5-9]`**. |
| 2026-10-02 | Attn D256 all done (peak 0.9374 @ η₀=0.25); D512 η₀=0.05/0.1 done (~0.937); η₀=0.25 running. |
| 2026-10-02 | Live canvas + updater track both pools (`update_upool_xfer_live_canvas.py`). |
