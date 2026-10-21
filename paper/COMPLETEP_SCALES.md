# CompleteP / μP scalings (authoritative)

**Status:** canonical. Do not change init variances, forward multipliers, or
Adam LR mappings without **explicit user approval**, then update this file,
then **re-run the one-step stability probes**
(`scans/jets/mup_one_step_transfer_probe*.slurm`) and check
`paper/figures/fig_stability_{attn,uniform}*` before trusting any training run.

**Process for agents / collaborators**

1. Read this file before touching `*_scale`, weight `nn.init`, or Adam LR.
2. **Ask the user** before changing any scaling listed here.
3. After an approved change: update this file in the same commit/PR.
4. Verify with one-step Adam \(\rho\) / \(\Delta\rho\) plots (encode, last residual,
   output) across the width ladder. **Output \(\Delta\rho\) must be width-flat**
   at fixed \(\eta_0\); that is the transfer claim.

**Primary code mirrors**

| Piece | Where |
|---|---|
| Encoder / decoder \(\sigma\)-param | `ParticleOnlyNetwork`; CAPEN `capen_bytecode.py` |
| SupportAttention \(\alpha_A=1\) | `SupportAttention` in `capen_bytecode.py`; `OutputAttentionPool` |
| Residual \(\alpha_A^{\mathrm{res}}\), \(\alpha_M\) | CAPEN / CAPEN-Llama `__init__` |
| Adam \(\eta=\eta_0/\sqrt{D}\) | `cpen.models.base.adam_lr` |
| Probe | `scans/jets/mup_one_step_transfer_probe.py` |

---

## Global

\[
\sigma \;=\; D^{-1/2}.
\]

Width \(D\) is the residual stream width (`width` / `model.D`).
Depth \(L\) is the number of residual blocks (`depth`).

All Linear / embedding weights below are **bias-free** unless noted.

---

## 1. Encoder (input embedding)

**Confirmed.**

\[
W^{(0)} \sim \mathcal N(0,\sigma^2),\qquad
H^{(0)} = W^{(0)} x \cdot \frac{1}{\sigma\sqrt{n_0}}
         = W^{(0)} x \cdot \sqrt{\frac{D}{n_0}}.
\]

| Stream | \(n_0\) (JetClass / paper default) | Forward scale |
|---|---|---|
| Particles \(X\) | \(n_0 = n_{\mathrm{features}}\) (kin7 → 7) | \(\sqrt{D/7}\) |
| Edges \(E\) (shared / untyped) | \(m_0 = n_{\mathrm{edge\_features}}\) (padded part-int → often 7) | \(\sqrt{D/m_0}\) |
| Star edges (typed) | \(m_0^{\star}=7\) (kin7 aggregate) | \(\sqrt{D/7}\) |
| kNN edges (typed) | \(m_0^{\mathrm{knn}}=4\) (part-interaction) | \(\sqrt{D/4}\) |

Typed (`--typed-edge-streams`): dual CompleteP encoders `encoder_e_star` /
`encoder_e_knn` with the same \(W\sim\mathcal N(0,\sigma^2)\) rule; the unused
shared `encoder_e` is **deleted** so it is not counted in \(P\).

**Do not** initialize encoder weights \(\sim\mathcal N(0,1)\). That makes
\(\rho\propto D\) and **breaks width-stable output \(\Delta\rho\)** (seen
2026-10-02; typed-edge regression 2026-10-21 when `init_std` defaulted to 1).

---

## 2. SupportAttention (\(\alpha_A = 1\))

Used for residual relations \(11,12,21,22\) and for class-token pools.

\[
W_{Q,K,V,O} \sim \mathcal N(0,1),\qquad
d_{\mathrm{head}} = D/H,
\]

\[
Q,K,V \leftarrow W_{\cdot}h\cdot\frac{1}{\sqrt{D}},\qquad
\mathrm{logits} \leftarrow \frac{QK^\top}{d_{\mathrm{head}}}
  \;\bigl(/\gamma_{rs}\;\text{when applicable}\bigr),
\]

\[
\mathrm{out} \leftarrow W_O(\mathrm{attn}\,V)\cdot\frac{1}{\sqrt{D}}.
\]

Optional WIRE/RoPE rotates \(Q,K\) only (never \(V\)).

### Degree temperatures \(\gamma_{rs}\) (`attention_normalization=gamma`)

\(\gamma_{rs}\) is the mean support row-degree for relation \(rs\). Use it **only**
on sparse incidence-like supports:

| Relation | Support | \(\gamma\)? |
|---|---|---|
| \(11\) | all-to-all particles | **no** (\(\gamma_{11}=1\); degree \(\approx N_{\mathrm{tok}}\) would uniformize) |
| \(12\), \(21\) | incidence \(S\) / \(S^\top\) | **yes** (degree normalization) |
| \(22\) | all-to-all among hyperedges | **no** (\(\gamma_{22}=1\)) |

Code: `CAPEN._gamma_for` + `sweep_common._clamp_all_to_all_gammas`.

---

## 3. Residual branch multipliers

After LN (no affine):

\[
H_X \leftarrow H_X + \alpha_A^{\mathrm{res}}(f_{11}+f_{21}) + \alpha_M\,\mathrm{MLP}_X(H_X),
\]
\[
H_E \leftarrow H_E + \alpha_A^{\mathrm{res}}(f_{22}+f_{12}) + \alpha_M\,\mathrm{MLP}_E(H_E),
\]

\[
\alpha_A^{\mathrm{res}} = \frac{1}{\sqrt{2}\,L},\qquad
\alpha_M = \frac{1}{L}.
\]

(Two attn branches into each stream → \(\sqrt{2}\) in the attn residual scale.)

---

## 4. MLP / SwiGLU (inside a block)

Weights \(\sim\mathcal N(0,1)\). Expansion \(4D\).

\[
\text{gate/up (or }W_1\text{) scale} = \frac{1}{\sqrt{D}},\qquad
\text{down (or }W_2\text{) scale} = \frac{1}{\sqrt{4D}}.
\]

CAPEN: GELU MLP. CAPEN-Llama / att: SwiGLU
\(\mathrm{down}(\mathrm{silu}(\mathrm{gate})\odot\mathrm{up})\).

---

## 5. Output / unembed (CompleteP readout) — critical

**Confirmed.**

\[
W_U \sim \mathcal N(0,\sigma^2),\qquad
\sigma_{\mathrm{readout}} = \frac{1}{D\sigma} = \frac{1}{\sqrt{D}}.
\]

### Uniform pool (`uniform_output_pool=True`)

\[
z_X = W_X^{\mathrm{dec}} H_X^{(L)}\cdot\sigma_{\mathrm{readout}},\quad
z_E = W_E^{\mathrm{dec}} H_E^{(L)}\cdot\sigma_{\mathrm{readout}},
\]

then mask-mean over tokens and fuse \(\frac{1}{\sqrt{2}}(\bar z_X+\bar z_E)\).

### Attentional pool (`OutputAttentionPool`, default)

**Non-negotiable CompleteP head.** Classification is always

\[
\mathrm{logit}_a
=
\frac{1}{D\sigma}\, X_{\mathrm{out},a}\cdot w_a,
\qquad
w_a\sim\mathcal N(0,\sigma^2),\quad
\sigma=D^{-1/2}.
\]

Absolute logits may shrink with \(D\); **one-step output feature updates
\(\Delta\rho\) must stay \(O(1)\) in width** (and trunk residuals \(O(1)\) in
depth via \(\alpha_A^{\mathrm{res}}\)). That constraint fixes the head — it is
not optional when the pool becomes attentional.

**What attention is allowed to change.** Replacing mean-pool by class-token
MHSA only redesigns how \(X_{\mathrm{out}}\) is *built*. The pool must emit
**Θ(1) features** (same role as a hidden state), then the head above applies.
Do **not** fold the head into \(W_O\!:D\to 1\) under \(\alpha_A=1\) init
\(W\sim\mathcal N(0,1)\): that is a different map from
\(W\sim\mathcal N(0,\sigma^2)\) × \(1/(D\sigma)\) and empirically breaks
width-flat output \(\Delta\rho\).

Exact stage (\(H=D/d_h\), typically \(d_h=32\); same heads as the trunk):

\[
\begin{aligned}
P_{\bullet}
&=
\mathrm{MHSA}_{\alpha_A=1}\!\bigl(q,\,H_{\bullet}^{(L)}\bigr)
&&
(\bullet\in\{X,E\},\; W_O\!:D\to D,\;\times\tfrac{1}{\sqrt{D}}),\\[0.4em]
X_{\mathrm{out}}
&=
\frac{1}{\sqrt{2}}\bigl(P_X+P_E\bigr)
&&
\text{(two-stream fuse \emph{only}),}\\[0.4em]
\mathrm{logit}_a
&=
\frac{1}{D\sigma}\, X_{\mathrm{out},a}\cdot w_a.
\end{aligned}
\]

BaselineTransformer (`node_only`) drops \(P_E\) and the \(\frac{1}{\sqrt{2}}\)
fuse. Never replace \(\sigma_{\mathrm{readout}}=1/(D\sigma)\) by \(1\). Do not
use \(H=1\) in the pool (attention logits would scale as \(1/D\)).

---

## 6. Optimizer

\[
\eta = \frac{\eta_0}{\sqrt{D}}
\quad\text{for all trainable params (Adam / AdamW base LR).}
\]

Optional `corr` multiplies **only** first-layer encoder param-group LRs
(`encoder_x`, `encoder_e`, and typed `encoder_e_star` / `encoder_e_knn`);
it does **not** change `adam_lr` itself.

AdamW weight decay (when used): \(\lambda = \lambda_0\sqrt{D}\) so
\(\lambda\eta\) is width-invariant.

Probe Adam \(\varepsilon = 10^{-14}\) (`cpen.training.stability.ADAM_EPS`).

---

## 7. Stability check (required after scaling or trunk changes)

```bash
# Paper-quality width ladder (gputest arrays only; L=4, d_h=32, 10×10)
sbatch scans/jets/mup_one_step_transfer_probe_uniform_paper.slurm
sbatch scans/jets/mup_one_step_transfer_probe_attn_paper.slurm
python scans/jets/merge_stability_paper.py --pool uniform
python scans/jets/merge_stability_paper.py --pool attn
```

Log + job table: `paper/figures/fig_stability.md`. Exploratory fast probes are
archived under `paper/archive/stability_20261002_*`.

**Pass criteria (at fixed \(\eta_0\), width ladder)**

| Stage | \(\rho\) | \(\Delta\rho\) |
|---|---|---|
| Encode \(\ell=0\) | \(O(1)\) (not \(\propto D\)) | flat vs \(D\) |
| Last residual \(\ell=L\) | \(O(1)\) | flat vs \(D\) |
| Output \(\ell=L+1\) | may shrink with \(D\) | **flat vs \(D\)** |

Log: `paper/figures/fig_stability.md`.

---

## 8. Algebraic identities (avoid “double scale” confusion)

\[
\frac{1}{\sigma\sqrt{n_0}} = \sqrt{\frac{D}{n_0}},\qquad
\frac{1}{D\sigma} = \frac{1}{\sqrt{D}}.
\]

\(\sigma\)-param decoder \(W\sim\mathcal N(0,\sigma^2)\) with multiplier \(1/(D\sigma)\)
is equivalent to \(U\sim\mathcal N(0,1)\) with multiplier \(1/D\).

---

## Changelog

| Date | Note |
|---|---|
| 2026-10-21 | Typed dual encoders: same \(N(0,\sigma^2)\) with \(\sigma=D^{-1/2}\) (not `init_std=1`); drop dead shared `encoder_e`; `corr` covers `encoder_e_{star,knn}`; typed skips \(\gamma_{22}\) in-model. |
| 2026-10-07 | Pool: multi-head concat \(\frac{1}{D\sigma}\sum_h W_{O,h}v_h\) (match trunk \(H\)); skip \(\gamma\) on all-to-all \(M_{11}/M_{22}\); keep \(\gamma_{12,21}\). BaselineTransformer residual \(1/L\) (no \(\sqrt{2}\)). |
| 2026-10-02 | File created after encoder \(N(0,1)\) bug broke uniform output \(\Delta\rho\). Encoder confirmed as \(N(0,\sigma^2)\) × \(1/(\sigma\sqrt{n_0})\). |
