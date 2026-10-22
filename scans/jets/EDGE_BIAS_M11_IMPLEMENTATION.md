# Edge-biased particle attention (M11 bias) — implementation note and sweep

**Status:** design note for the edge→particle coupling study (checklist q06 / b11), 2026-10-21.
**Applies to:** `capen-llama-att --typed-edge-streams` (`-s1i`), JetClass, kin7, star R★=0.1 + kNN k=7.
**Do not change** any existing scaling in `paper/COMPLETEP_SCALES.md`; this adds one new branch
with its own rule (add a §2b to that file in the same commit, approved by Gage).

---

## 1. What changes, in one sentence

Each kNN edge's **current** state adds a learned, per-head scalar to the particle↔particle (M11)
attention logit of its particle pair, so edges influence *which* particles attend to each other
(routing), as the pairwise interaction matrix does in ParT — but here the interaction is learned
and evolves with the edge stream.

## 2. Notation (matches the code)

- Particles: `h_x ∈ R^{B×N×D}`, N = 80 slots, live mask `m_x`.
- Typed edge stream: `h_e = [h_star (N slots) | h_knn (N·k slots)] ∈ R^{B×(N+Nk)×D}`.
  kNN slot for (source particle i, rank r) lives at edge index `N + i·k + r`, r < k_eff = min(k, N−1).
- Per layer ℓ: `x_ln = LN(h_x)`, `e_ln = LN(h_e)` (parameter-free LN, already computed in forward).
- Heads H = D / d_h with d_h = 32; σ = D^{−1/2}.
- Current M11 logit (head h, query i, key j), γ₁₁ = 1:
  ```
  s^h_ij = (q^h_i · k^h_j) / d_h,   q = W_Q x_ln / √D,  k = W_K x_ln / √D,  W ~ N(0,1)
  ```

## 3. The new branch

For every layer ℓ, a bias head `W_b^(ℓ) ∈ R^{H×D}`, bias-free:

```
W_b^(ℓ) ~ N(0, σ²),  σ = D^{-1/2}                                   (init)
u^(ℓ)_{i,r,h} = (1/(D σ)) · W_b^(ℓ)[h] · e_ln^(ℓ)[N + i·k + r]       (= (1/√D) · w_h · z)
b^(ℓ),h_{ij}  = u^(ℓ)_{i,r,h}  if j = dst(i, r) and slot (i, r) is live
             = 0              otherwise (non-kNN pairs, self, padding)
s^h_ij       ← s^h_ij + b^(ℓ),h_{ij}                                (added after the 1/d_h scale)
```

- **Directed:** edge i→j biases query i attending to key j only. If j also chose i, that is a separate
  kNN slot and biases query j → key i. (Optional later flag `--edge-bias-symmetric` adds each edge's
  bias to both (i, j) and (j, i); not in v1.)
- **Stars:** not used in the bias in v1. They still reach particles through M21 when M21 is on.
- **Temperature:** the bias is not divided by any γ (γ₁₁ = 1 anyway).
- **Masking:** apply the bias before the support mask; masked entries stay −∞.
- **Dynamic vs static:**
  - `dynamic`: `e_ln^(ℓ)` is the edge state entering layer ℓ (the same `e_ln` used by M12/M22 at ℓ).
  - `static`: every layer uses `e_ln^(0) = LN(h_e^(0))`, the encoder output. Each layer keeps its own
    `W_b^(ℓ)`. Gradients still flow into the kNN encoder. This is the ParT-like control.

### Why this transfers across width (put in Methods as a remark)
The bias head is a readout from a D-dim state to a scalar, so it takes the readout rule (same as the
class head, Table 1 readout row).
- Init: `w_h·z` is a sum of D terms with variance 1/D → Θ(1); × 1/√D → bias Θ(D^{-1/2}). The model
  starts essentially equal to the no-bias model.
- One Adam step (sign-update heuristic, LR η = η₀/√D): δw ~ η₀/√D per entry, coherent with z, so
  δb ≈ (1/√D)·(η₀/√D)·D = Θ(η₀). Width-independent, like the q·k/d_h term (d_h fixed).
- **Do not** implement as a plain `nn.Linear(D, H)` with default init and no multiplier: then
  δb = Θ(η₀ √D) and the optimal η₀ drifts with width.
- The bias lives in logits, not the residual stream → no 1/L factor. H grows with D, one bias row per
  head (same many-heads limit as the trunk).
- Optimizer: `W_b` in the main param group (η₀/√D); **not** an encoder param (`corr` does not apply);
  same weight-decay rule as other weights when AdamW is used.

## 4. Residual scales per arm (n_src rule)

Each stream's attention residual multiplier is `1/(√n_src · L)`, n_src = number of learned attention
branches writing that token.

| Arm | Particle attention residual | Edge residual |
|---|---|---|
| M21 on (A, C, E) | `1/(√2 L)` on f11 + f21 | unchanged (hyper `1/(√2L)`, other `1/L`) |
| M21 off (B, D, F) | `1/L` on f11 only | unchanged |

The bias does not add a branch (it modifies f11), so it does not change n_src.
MLP residual stays `1/L` everywhere.

## 5. Code changes

1. **Graph builder** (`src/cpen/graphs/graph_star.py`)
   - `build_delta_r_knn_edges(..., return_index=False)`: when True also return
     `knn_dst: LongTensor (B, N, k)` (= `knn_idx`, right-padded to k with 0) and
     `knn_valid: BoolTensor (B, N, k)` (= `finite`, False for r ≥ k_eff).
   - `build_star_kin7_plus_knn_graph(..., typed_features=True, return_knn_index=False)` passes these
     through; `BaseLitCPEN.on_after_batch_transfer` stores them as `batch["knn_dst"]`,
     `batch["knn_valid"]` and `_model_kwargs` forwards them.
2. **Attention** (`capen_bytecode.py::SupportAttention.forward`)
   - New optional `attn_bias: Tensor | None` of shape (B, H, T, S), added right after
     `logits = einsum(...) * _logit_scale` (and after the temperature division). If/when this path
     moves to `F.scaled_dot_product_attention`, pass `bias.masked_fill(~support, -inf)` as a float
     mask and handle empty rows as already planned.
   - The probe second pass must add the same bias so pathology stats reflect the real logits.
3. **Model** (`capen_llama_att.py`)
   - Ctor args: `edge_bias_m11: Literal["none","dynamic","static"] = "none"`, `m21: bool = True`.
     Only valid with `typed_edge_streams=True`.
   - If `edge_bias_m11 != "none"`: `self.bias_11 = nn.ModuleList([nn.Linear(D, H, bias=False) for _ in range(L)])`,
     init `normal_(std=1/√D)`, forward multiplier `self._bias_scale = 1/(D·σ) = 1/√D`.
   - Build the bias once per layer, vectorized, no host syncs:
     ```python
     z = e_ln[:, N:, :] if dynamic else e_ln0[:, N:, :]          # (B, N*k, D)
     u = self.bias_11[l](z) * self._bias_scale                   # (B, N*k, H)
     u = u.view(B, N, k, H).permute(0, 3, 1, 2)                  # (B, H, N, k)
     u = u * knn_valid.unsqueeze(1)                              # zero invalid slots
     idx = knn_dst.unsqueeze(1).expand(B, H, N, k)
     bias = u.new_zeros(B, H, N, N).scatter_add_(-1, idx, u)     # (B, H, N, N)
     ```
     Invalid slots point at index 0 with value 0, so `scatter_add_` is safe. (B=64, H=8, N=80:
     ~3.3M elements per layer.)
   - Pass `attn_bias=bias` to `_run_attn(... relation="11")`.
   - If `m21=False`: skip f21 (return zeros), use `self._attn_residual_scale_single` (1/L) for the
     particle attention residual; keep `attn_21` parameters out of the model entirely (do not build
     them) so P and DDP stay clean.
   - `static`: keep `e_ln0 = LN(h_e^(0))` from right after the encoders.
4. **Degree temperature for arm C:** new CLI `--gamma21-mode {mean,one}`; `one` sets γ₂₁ = 1 in the
   model (after estimation/clamping), leaving γ₁₂ unchanged.
5. **CLI / tags** (`scans/common/sweep_common.py`): `--edge-bias-m11 {none,dynamic,static}`,
   `--no-m21`, `--gamma21-mode`. Run-tag tokens `eb-dyn`, `eb-sta`, `nom21`, `g21one`; schema suffix
   `-eb1` on all coupling-study runs (including arm A reruns) so they share one comparable set.
6. **COMPLETEP_SCALES.md:** add "§2b Edge→M11 logit bias: W_b ~ N(0,σ²), ×1/(Dσ), main LR group, no
   1/L" and a changelog line.

## 6. Tests (CPU)

- **Zero-head equivalence:** with `W_b = 0`, logits/outputs equal the no-bias model (same seed, same
  batch), for both dynamic and static.
- **Placement:** bias nonzero only at (i, dst(i,r)) for live slots; zero on diagonal, non-kNN pairs and
  padded rows/cols; directed (i→j set does not imply j→i).
- **Static uses layer 0:** perturbing `h_e` after layer 0 changes dynamic bias but not static bias.
- **Init scale:** RMS of bias over live pairs ≈ c·D^{-1/2} at D ∈ {64, 256} (ratio ≈ 2 within 20%).
- **M21 off:** no `attn_21` params; particle residual scale = 1/L.
- **Probe second pass** includes the bias (logit stats match forward).

## 7. Stability probe (before any long run)

Add channel `bias11` (RMS of bias on live kNN pairs, per layer) to
`scans/jets/mup_one_step_transfer_probe.py`. Run the attentional-pool probe at **fixed L = 4**,
D ∈ {128, 256, 512}, η₀ ∈ {0.05, 0.25}, real JetClass batches, arms D and F.
Pass: `bias11` Δ and output-logit Δ flat in D at fixed η₀ (same criterion as the class head).

## 8. Coupling sweep (b11)

**Fixed:** typed `-s1i` + `-eb1`, L = 4, D = 256, H = 8, d_h = 32, k = 7, R★ = 0.1, kin7, P = 80,
Adam ε = 1e-14, B_global = 256, one shared `data_seed`, n_val ≥ 200k (fixed slice).
**Budget:** 30k steps, `warmup_flat_cosine` (1k warmup, flat to 24k, cosine to 30k, min_frac 0.01)
— ~7.7M jets; same for every arm.

| Arm | Flags | Question |
|---|---|---|
| A | (none) | current baseline |
| B | `--no-m21` | does M21 carry anything? |
| C | `--gamma21-mode one` | is mean-degree flattening the problem? |
| D | `--no-m21 --edge-bias-m11 dynamic` | routing instead of messaging |
| E | `--edge-bias-m11 dynamic` | routing + messaging |
| F | `--no-m21 --edge-bias-m11 static` | does the bias need evolving edges? (ParT-like) |
| P0 | `--model baseline-transformer` | particle-only anchor |
| ParT | official ParT code, kin7, same jets/val slice | external anchor (b08) |

**Grid:** η₀ ∈ {0.1, 0.25, 0.5} for every arm with seed 0 (arms may shift the optimum); then seed 1
at each arm's best η₀. 6 arms × 3 + 6 = 24 CAPEN runs, + P0 (3 + 1), + ParT.

**Record:** final validation loss (after decay), accuracy, macro AUC, class-vs-QCD rejection at
ε_S = 0.5 (needs checklist c02), training FLOPs and parameter count, wall-clock per step; probe stats
at the end (M11 entropy with and without bias, `bias11` RMS, M21 entropy where present).

**Decision rule:** pick the arm with the lowest validation loss if it beats A by more than twice the
seed-to-seed difference; on a tie prefer the simpler arm (fewer branches). Then freeze it and run the
width check: D ∈ {128, 512} (fixed L = 4), η₀ ∈ {0.05, 0.1, 0.25, 0.5}, same budget, to confirm the
optimum η₀ does not move.

**Reading the result:**
- B ≈ A → M21 is not doing useful work; drop or replace it.
- D < A and D < F → evolving routing helps: strongest version of the paper's claim.
- D ≈ F < A → routing helps but evolving edges do not (report honestly; reframe §2.4).
- C < A → the mean-degree temperature was hurting; revisit q03 for M12 too.
