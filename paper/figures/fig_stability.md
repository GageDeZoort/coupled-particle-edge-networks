# Feature stability (one-step Adam μP) — experimental log

**Canonical scales:** `paper/COMPLETEP_SCALES.md` (ask before any scale change).

**Live focus (paper):**
- Uniform gputest **`14877878`**: D≤1024 complete; D1536/2048 **TIMEOUT** (partial shards).
- Attentional: D∈{64…**1024**} **complete** (48/48); D1536/2048 incomplete on 59m gputest → refill **`14890943_[7-8]`** on regular **a100/gpu @ 1:05:00** (1 GPU).
- Transfer-validation (debug): `fig_stability_transfer_validation.md` (uniform Della gputest + attn Lambda).

**Archives (do not cite for paper figures)**
| Path | What |
|---|---|
| `paper/archive/stability_20261002_encoder_N01_bug/` | encoder \(N(0,1)\) bug (job 14861246 + older plots) |
| `paper/archive/stability_20261002_exploratory_fast/` | post-fix fast checks 14874323 / 14874934 + old live figs |
| `paper/data/archive_stability/` | scale=1 / pre-xe / pre-unembed CSV mistakes |

---

## Paper recipe (locked)

| Knob | Value |
|---|---|
| Depth \(L\) | **4** (fixed; width claim only) |
| Widths \(D\) | \{64, 128, 256, 384, 512, 768, 1024, 1536, 2048\} |
| Head dim | **\(d_h=32\)** → \(H=D/32\) |
| Trials / point | **10** model seeds × **10** data shards |
| \(\eta_0\) | \{0.01, 0.05, 0.1, 0.25, 0.5, 1.0\} |
| Geometry | \(k^\star=12\), \(R^\star=0.1\) |
| Batch | synthetic \(B=8\), \(N=128\) particles |
| Pools | **uniform** + **attn** (separate arrays) |
| Partition | **gpu-test** for \(D\le1024\) (`--time=0:59:00`); **exception:** \(D\in\{1536,2048\}\) may use **1×A100 @ 1:05:00** (gputest times out) |
| Array | one width per task (`--array=0-8`) |

**Claim.** CompleteP + Adam \(\eta=\eta_0/\sqrt{D}\): \(\Delta\rho\) **flat vs \(D\)**
at encode / last residual / **output**, fixed \(\eta_0\). Absolute readout \(\rho\)
may shrink with \(D\) (unembed \(1/\sqrt{D}\)).

\[
\rho = \frac{\|Z\|_F^2}{N_{\mathrm{live}}\, d},\qquad
\Delta\rho = \frac{\|\Delta Z\|_F^2}{N_{\mathrm{live}}\, d}
\]

| Column | Stage | Measured |
|---|---|---|
| (a) | \(\ell=0\) | \(X^{(0)}, E^{(0)}\) after encoders |
| (b) | \(\ell=L\) | \(X^{(L)}, E^{(L)}\) after last residual |
| (c) | \(\ell=L+1\) | readout (uniform decode / attn class-token pool) |

Combo rows: \(X\rho\), \(X\Delta\rho\), \(E\rho\), \(E\Delta\rho\). Color = \(\eta_0\).

**Figure stems (paper only, live)**
| Stem | Pool |
|---|---|
| `fig_stability_paper_uniform{,_rho,_combo}` | uniform |
| `fig_stability_paper_attn{,_rho,_combo}` | attentional |

**Complete shard sizes** (agg rows; used by `--complete-only`):
uniform = \(6\times6=36\); attn = \(8\times6=48\) (adds `output_fused`, `output_logits`).

**Combo layout (width now; depth rows later)**
| Row | Content | Encoding |
|---|---|---|
| 1 | pre-step \(\rho\) vs \(D\) | color = stream (\(X\) blue / \(E\) vermillion) |
| 2 | one-step \(\Delta\rho\) vs \(D\) | color = stream; **linestyle** = \(\eta_0\in\{0.25,0.5,1\}\) |
| 3–4 | *(later)* depth \(\rho\) / \(\Delta\rho\) vs \(L\) | same encodings |

Columns: encode / last residual / output. No error bars (SEM invisible at scale).
Do not half-split cells.

---

## Commands

```bash
# Submit (gputest arrays)
sbatch scans/jets/mup_one_step_transfer_probe_uniform_paper.slurm
sbatch scans/jets/mup_one_step_transfer_probe_attn_paper.slurm
# Attn refill incomplete/missing widths only (gputest):
sbatch --array=4-8 scans/jets/mup_one_step_transfer_probe_attn_paper.slurm
# Large-D attn only (regular A100, 1 GPU, 65m — bypasses 59m gputest timeout):
sbatch --array=7-8 --time=1:05:00 scans/jets/mup_one_step_transfer_probe_attn_paper.slurm

# After array tasks complete
python scans/jets/merge_stability_paper.py --pool uniform --complete-only
python scans/jets/merge_stability_paper.py --pool attn --complete-only
```

Slurm: `scans/jets/mup_one_step_transfer_probe_{uniform,attn}_paper.slurm`  
Merge: `scans/jets/merge_stability_paper.py`  
Probe: `scans/jets/mup_one_step_transfer_probe.py --d-head 32`

Per-width shards (live while running):
`paper/data/stability_one_step_{uniform,attn}_paper_D{W}.csv`  
(+ `_trials.csv`). Merged canon: `…_paper.csv` (after merge).

---

## Jobs — paper (canonical)

| Job | Pool | Recipe | Status (2026-10-02 ~15:25 ET) |
|---|---|---|---|
| **`14877878_[0-8]`** | **uniform** | L=4, \(d_h=32\), 10×10, gputest | D≤1024 **COMPLETE**; D1536/2048 **TIMEOUT** (partial) |
| **`14877879_[0-8]`** | **attn** | same | **CANCELLED** mid-run; left complete D64–384 + partial D512 |
| **`14886895_[4-8]`** | **attn** | refill D∈{512…2048} gputest | D512/768/1024 **COMPLETE**; D1536 **TIMEOUT** (40/48); D2048 **CANCELLED** mid-run (16/48) |
| **`14890943_[7-8]`** | **attn** | D1536+2048 @ **1:05:00** (1×A100 / `gpu`) | **PENDING** — fill remaining stability widths |
| `14874984` / `14874985` | — | 12h a100 | **CANCELLED** (superseded; use short 65m exception above) |

Array index → width: `0→64, 1→128, 2→256, 3→384, 4→512, 5→768, 6→1024, 7→1536, 8→2048`.

**Logs:** `/scratch/gpfs/BHANIN/jdezoort/cpen_runs/jetclass_scaling_law/logs/mup_one_step_{upool,attn}_paper_%A_%a.{out,err}`

**Attn shard checklist**
| \(D\) | Status |
|---|---|
| 64, 128, 256, 384 | complete (48/48) from `14877879` |
| 512, 768, 1024 | complete (48/48) from `14886895` |
| 1536 | incomplete (40/48) after gputest TIMEOUT — **re-running** `14890943_7` @ 1:05 |
| 2048 | incomplete (16/48); gputest task cancelled — **re-running** `14890943_8` @ 1:05 |

---

## Jobs — archived / superseded

| Job | Mode | Why not paper | Where |
|---|---|---|---|
| `14861246` | uniform fast | encoder \(N(0,1)\) → output Δρ∝D | `archive/…_encoder_N01_bug/` |
| `14874323` | uniform fast | joint \(L{\propto}D\); 10×1; exploratory post-fix ✓ | `archive/…_exploratory_fast/` |
| `14874934` | attn fast | same + legacy H=8 | `archive/…_exploratory_fast/` |
| `14860001/2`, `14860480/1`, … | various | scale=1 / missing pyc / cancelled | see archives + changelog |

---

## Observations

### 2026-10-02 — attn paper ladder through D1024; large-D on 65m A100

Gputest refill **`14886895`** completed D512/768/1024 (48/48 each). D1536
timed out at 59m with 40/48; D2048 was still incomplete (~16/48) near the
wall. Cancelled the dying D2048 gputest task and submitted
**`14890943_[7-8]`** with `--time=1:05:00` (lands on regular `gpu`/A100,
1 GPU) to finish the width ladder.

Live attn paper figs refreshed with `--complete-only` → D∈{64…**1024**}:
`fig_stability_paper_attn{,_rho,_combo}.*`. Re-merge once 1536/2048 finish.

### 2026-10-02 — attn paper ladder re-enabled (gputest)

Transfer validation shows attentional pool beating uniform at short horizon;
resume width-stability attn array on **gputest** (not ailab). Reuse complete
D64–384 shards; resubmit D512+ as **`14886895_[4-8]`**.

### 2026-10-02 — uniform paper ladder

Merged complete D∈{64…**1024**} → `fig_stability_paper_uniform{,_rho,_combo}.*`.
D1536/2048 timed out at 59m with partial shards — need longer walltime or
split trials if we want those points.

**Plot corrections (same day):** combo **2×3** — ρ color=stream, **autoscale** y;
Δρ color=stream, linestyle η₀ (solid=1, long dash=0.5, short dash=0.25);
**no error bars**. Encode \(\rho_E{\sim}5\) vs \(\rho_X{\sim}1\): feature second-moment
(not a width blow-up); leave scales alone.

### Prior (archived) — uniform fast post-fix **14874323**

Confirmed encoder fix: output \(\Delta\rho\) flat vs \(D\) (log–log slope
\(\sim 0\) for `output_e`; mild for `output_x`). Justified proceeding to
paper recipe; **not** cited as paper figure.

### Prior (archived) — encoder bug **14861246**

Output \(\Delta\rho\sim D\); fixed init \(W\sim\mathcal N(0,\sigma^2)\) in
`capen_bytecode.py`. See `COMPLETEP_SCALES.md` + encoder-bug archive.

---

## Changelog

| Date | Note |
|---|---|
| 2026-10-02 | Encoder \(N(0,1)\) bug diagnosed (14861246); fix + COMPLETEP_SCALES.md. |
| 2026-10-02 | Post-fix fast uniform **14874323** (Δρ flat); attn fast **14874934**. |
| 2026-10-02 | Paper recipe locked: L=4, \(d_h=32\), D∈{64…2048}, 10×10, gputest arrays. |
| 2026-10-02 | Long a100 **14874984/85** cancelled → gputest **14877878/79**. |
| 2026-10-02 | Archived exploratory figs/CSVs; live focus = paper shards + this log. |
| 2026-10-02 | Combo → **2×3**: ρ (color=stream) + Δρ (color=stream, linestyle=η₀). |
| 2026-10-02 | Draft uniform combo with complete D∈{64…**1024**}. |
| 2026-10-02 | Attn paper **14877879** cancelled to pin uniform; D64–384 shards retained. |
| 2026-10-02 | Transfer canvas: uniform Della + attn Lambda (`upool-xfer-validate-live`). |
| 2026-10-02 | Attn refill **`14886895_[4-8]`** on gputest; draft attn figs D≤384; merge complete-rows pool-aware. |
| 2026-10-02 | Uniform D1536/2048 **TIMEOUT** on 59m gputest (partial). |
| 2026-10-02 | Attn D512–1024 **complete**; D1536 TIMEOUT (40/48); D2048 partial → **`14890943_[7-8]`** @ 1:05 A100. |
| 2026-10-02 | Merged attn paper figs through D1024 (`fig_stability_paper_attn*`). |
