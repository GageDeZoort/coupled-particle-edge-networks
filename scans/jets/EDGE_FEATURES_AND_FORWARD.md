# Edge features, \(k\), \(R^\star\), and the CAPEN-Llama-att forward pass

Authoritative for re-running graph-construction / transfer plots after the
2026-10-01 preprocess quarantine. Code: `src/cpen/graphs/graph_star.py`,
`src/cpen/training/base_lit_cpen.py`, `src/cpen/models/capen_llama_att.py`,
`scans/common/sweep_common.py::resolve_part_interaction_preprocess`.

---

## 1. Graph layout as a function of \(k\) and \(R^\star\)

Production JetClass stream uses **star-\(R^\star\) + directed \(\Delta R\)-kNN**:

| component | slots (edge axis) | incidence degree | role in attention |
|-----------|-------------------|------------------|-------------------|
| star hyperedges | \(N\) (one per particle slot) | \(\ge 1\) (self); often \(>2\) | M12/M21; **M22 only if deg \(>2\)** |
| \(\Delta R\)-kNN 2-edges | \(N\cdot k\) | 1 (self-only pad) or 2 | M12/M21; **identity \(f_{22}\)** |

Concatenation order: `[star (N) | knn (N·k)]`
(`build_star_plus_knn_graph`).

**Star support (\(R^\star\))** — in \((\eta,\phi)\):

\[
S_{c j}^{\mathrm{star}} = \mathbf{1}[\Delta R_{\eta\phi}(c,j) < R^\star]
  \;\lor\; (c=j \land \text{live}),
\]
masked by live particles. Inactive centers → zero incidence row and zero
edge features.

**kNN (\(k\))** — directed nearest \(k\) in \(\Delta R_{\eta\phi}\).
**Default (2026-10-12+):** self is **excluded** (`include_self=False`);
particle self-attention already lives in M11. Legacy
`--live-knn-include-self` puts self at \(\Delta R{=}0\) (floored
part-interaction spikes; CLI \(k{=}1\) was then self-only). Each live slot
is a 2-edge \((i\to j)\). Run dirs tag `noloop` (default) or `knnselt`.

**How \(k,R^\star\) change the stream**

- Larger \(R^\star\) → larger star supports → more hyperedges with deg \(>2\)
  (more M22 mass) and different centroid four-vectors → different raw
  interaction logs.
- Larger \(k\) → more 2-edge slots (\(N\cdot k\)); star block unchanged.
  Feature *distribution* shifts because kNN edges dominate count and their
  \(\Delta\) is typically smaller than star-to-centroid \(\Delta\).
- Edge count (live): \(\sim N_{\mathrm{live}} + N_{\mathrm{live}}\cdot k\)
  (pads zeroed).

`--no-star-hyperedges` + `--live-knn-k` → kNN-only. `--star-radius` alone →
star-only.

---

## 2. Raw edge features (Particle Transformer interaction set)

Mode `part-interaction` (paper default). For a pair of four-vectors
\((p,q)\) (center vs support centroid for star; source vs neighbor for kNN):

\[
\begin{aligned}
\Delta &= \Delta R_{y\phi}(p,q)
  &&\text{(true rapidity, not }\eta\text{)}\\
k_T &= \min(p_T^p,p_T^q)\,\Delta\\
z &= \min(p_T^p,p_T^q)\,/\,(p_T^p+p_T^q)\\
m^2 &= (p+q)^2
\end{aligned}
\]

Stored as
\(\bigl(\ln\Delta,\;\ln k_T,\;\ln z,\;\ln m^2\bigr)\) with floors
\((10^{-3},10^{-3},10^{-3},10^{-3})\) so degenerate hyperedges
(centroid \(=\) center when no neighbors in \(R^\star\)) do not hit
\(\ln\varepsilon\) spikes of order \(-18\).

**Star centroid.** For `part-interaction`, default weight is **\(p_T\)**
(boost/rotation covariant). Energy weighting is reserved for legacy
`logdot-dp`.

**Invariance.** All four logs are invariant under rotations about the beam
and boosts along it when \(q\) is a \(p_T\)-weighted composite.

Alternative mode `logdot-dp`: Minkowski dot + lab \(\Delta p\) vs energy-weighted
centroid — not used for paper panels.

---

## 3. Normalization → \(\Theta(1)\) (the \(k,R^\star\)-dependent part)

Raw logs are **not** \(\Theta(1)\). Standardization is affine per channel:

\[
\tilde e_a = (e_a - \mu_a)\, s_a,\qquad a\in\{\Delta,k_T,z,m^2\}.
\]

Constants \((\mu_a,s_a)\) come from **robust** stats on the **same** live
edge stream that training will see:

\[
\mu_a = \mathrm{median}(e_a),\qquad
s_a = 1.35 / \mathrm{IQR}(e_a).
\]

### Production path (default, 2026-10-01+)

`resolve_part_interaction_preprocess` (`--estimate-part-interaction-preprocess`,
default **on**):

1. Build the first \(\sim\)512 train jets with `standardize=False` using the
   **exact** construction (`star+kNN` / star / knn-only, same \(R^\star,k\),
   same centroid weight).
2. Collect live edge rows (`|e|>0`).
3. `robust_part_interaction_preprocess` → four \((\mu,s)\) pairs.
4. Pass into Lit as `live_part_interaction_preprocess`; every GPU batch in
   `on_after_batch_transfer` applies those constants.

So **\(\mu,s\) adapt to \(k\) and \(R^\star\)** (and to star vs knn mix).
They are estimated once per run and held fixed for that run.

### Static table (legacy / `--no-estimate-…`)

`PART_INTERACTION_PREPROCESS` keyed by **exact** \(R\) only
(no nearest-key fallback):

| \(R\) | geometry assumed when measured |
|------|--------------------------------|
| 0.10, 0.125 | star + kNN \(k{=}12\) |
| 0.15, 0.20 | **star-only** |

Wrong table (e.g. star-only \(R{=}0.15\) constants on a star+kNN \(R{=}0.1\)
stream) silently mis-scales \(\ln m^2\) etc. and **breaks edge-encoder μP**.
That is why the experimental CSVs from before live estimation are quarantined.

### What is *not* done

- No per-edge L2 row norm in the current graph builders (older comments in
  `part_kin.py` are stale relative to `graph_star.py`).
- Attention \(\gamma_{rs}\) is separate: mean support degrees estimated from
  incidence; it does **not** depend on feature values.

---

## 4. How edges enter the forward pass

### Encode

Particle inputs: kin7 affine features \(x\in\mathbb{R}^{N\times 7}\).
Edges: standardized \(\tilde e\in\mathbb{R}^{M\times 4}\),
\(M = N + Nk\) for star+kNN.

\[
H_X^{(0)} = W_X x \cdot \sigma_X,\quad
H_E^{(0)} = W_E \tilde e \cdot \sigma_E
\]

with CompleteP encoder scales
\(\sigma_X=\sqrt{D/7}\), \(\sigma_E=\sqrt{D/4}\)
(\(W\sim\mathcal N(0,\sigma^2)\) with \(\sigma=D^{-1/2}\), bias-free;
equivalently forward multiplier \(1/(\sigma\sqrt{n_0})\)).
**Authoritative list:** [`paper/COMPLETEP_SCALES.md`](../COMPLETEP_SCALES.md).
Do not change scalings without updating that file and re-running stability probes.

### Residual block \(\ell=0,\ldots,L-1\)

LayerNorm (no affine) → four SupportAttentions on supports from incidence:

| relation | Q from | K/V from | support |
|----------|--------|----------|---------|
| 11 | \(X\) | \(X\) | line-graph of particles (or all-to-all if configured) |
| 21 | \(X\) | \(E\) | \(S^\top\) |
| 12 | \(E\) | \(X\) | \(S\) |
| 22 | \(E\) | \(E\) | hyperedges only (deg \(>2\)); 2-edges keep identity \(f_{22}\) |

Logits use \(\alpha_A=1\) scales (\(1/d_{\mathrm{head}}\)), multiplied by
buffers `attn_gamma_rs` (estimated mean degrees). Then:

\[
\begin{aligned}
H_X &\leftarrow H_X + \alpha_A^{\mathrm{res}}(f_{11}+f_{21})
  + \alpha_M\,\mathrm{SwiGLU}_X(H_X)\\
H_E &\leftarrow H_E + \alpha_A^{\mathrm{res}}(f_{22}+f_{12})
  + \alpha_M\,\mathrm{SwiGLU}_E(H_E)
\end{aligned}
\]

with \(\alpha_A^{\mathrm{res}}=1/(\sqrt{2}\,L)\), \(\alpha_M=1/L\).

**Dependence on \(k,R^\star\)** here is only through:

1. which tokens exist and their features \(\tilde e(k,R^\star)\);
2. incidence → supports → \(\gamma_{rs}(k,R^\star)\) and M22 hyperedge set.

### Readout

- **Attentional** (`OutputAttentionPool`, default): class tokens pool \(X^{(L)}\)
  and \(E^{(L)}\) independently (\(\alpha_A=1\)), fuse
  fuse \(X_{\mathrm{out}}=\tfrac{1}{\sqrt{2}}(P_X+P_E)\), then CompleteP
  \(\mathrm{logit}_a=\tfrac{1}{D\sigma}X_{\mathrm{out},a}\cdot w_a\)
  (\(w_a\sim\mathcal N(0,\sigma^2)\), \(\sigma=D^{-1/2}\))
  (scale \(1/\sqrt{D}\): logits shrink, updates stay \(O(1)\)).
  Pool temperature is **none** — softmax over \(O(10^2{-}10^3)\) edge slots
  with no extra \(\gamma\).
- **Uniform** (`--uniform-output-pool`): mask-mean of `decoder_x/e` outputs,
  same \(\tfrac{1}{\sqrt{2}}\) fuse. Linear decoders
  \(z = W_{\mathrm{dec}} H\cdot\sigma_{\mathrm{readout}}\).

### Optimizer μP

Adam base LR \(\eta=\eta_0/\sqrt{D}\) for all params; optional `corr` multiplies
**only** `encoder_x` / `encoder_e`.

---

## 5. Checklist before re-running \(k\) / \(R^\star\) plots

1. Live preprocess **on** (default); do not reuse old parquet / paper CSVs.
2. Same construction for estimate and train (star+kNN, same \(k,R^\star\)).
3. Confirm logged `part_interaction_preprocess` in run metadata matches the cell.
4. γ estimated from the same incidence (already default).
5. Prefer one-step Adam μP probe
   (`scans/jets/mup_one_step_transfer_probe.py`) across
   \((L,D)\in\{(2,128),(4,256),(6,512),(8,1024)\}\) × \(\{k,R^\star\}\) ×
   ± attentional pool **before** long graph-construction sweeps.
