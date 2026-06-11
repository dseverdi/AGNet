# PO-AGP Adjustment Plan

## Tier 0 — Baseline & Guardrails (1 run)
- Freeze a reference config and random seed.
- Log consistently each epoch:
  - `loss`
  - `best_reward_mean`
  - `coverage_greedy`, `coverage_stoch`
  - `|S|/n` (greedy + stochastic)
  - `|S|/opt`
  - `mean|Δlogπ|`
- Keep early stopping enabled:
  - `early_stop_patience = 4`
  - `early_stop_min_delta = 0.0005`

## Tier 1 — Low-cost Hyperparameter Sweep (4–8 runs)
- Keep all settings fixed except below knobs.
- Sweep alpha (`α`) with current `exponential` preference loss:
  - `α ∈ {0.03, 0.1, 0.2, 0.5}`
- Sweep rollout count (`K`):
  - `K ∈ {8, 12}` (or `16` if budget allows)
- Selection rule:
  - choose top 2 settings by lowest `|S|/opt`
  - require matched quality: `coverage_stoch >= 0.999`

## Tier 2 — Objective Shaping (2–4 runs)
- Add margin enhancement term (`γ`) in pairwise objective.
- Test margin values:
  - `γ ∈ {0.00, 0.01, 0.03}`
- Add/adjust length-control coefficient in preference comparison.
- Test values:
  - `{1.0, 1.5}`
- Target outcome:
  - reduce greedy/stochastic `|S|/n` gap by >= 15%

## Tier 3 — Training Regime Upgrade (2 runs)
- Scale up training data and steps:
  - `train_size >= 1000`
  - `epochs = 40–80` (early stop still enabled)
- Enable short LS fine-tuning only for best Tier-2 setup:
  - about 5% of main training epochs

## Tier 4 — Decision Gate
Promote a configuration only if all are true:
- `coverage_greedy >= 0.995`
- `coverage_stoch >= 0.999`
- `|S|/opt` improves by >= 10% vs baseline
- runtime per epoch remains acceptable

If criteria are not met:
- run an A/B test with `ranking_mode = false` to verify whether ranking reward is the limiting factor.

## Tier 5 — Architecture Upgrade (Transformer Encoder)  *— tried and reverted (Mar 2026)*

> **Status:** Implemented in commit `62002b2` (5 Mar 2026), removed in commit `99dc2bd` (12 Mar 2026). The `configs/po_agp_transformer.json` referenced below was never checked in. The pivot away from this direction toward the learned-editor approach is documented in "Current Direction: Learned Editor" at the end of this file.

Motivation: The paper tests PO exclusively on Transformer-based architectures (AM, POMO,
Sym-NCO, Pointerformer). Our current LSTM Pointer Network (330K params, 1 layer, 1 glimpse)
does not match any model in the paper. Structural weakness: the LSTM hidden state cannot
maintain a spatial coverage map over n=128 mean vertices, causing the observed bimodal trap.

**New architecture (`model_type: "transformer"`):**
- Input projection: `Linear(2 → 128)`
- Encoder: 3 × [MHA(8 heads) + LayerNorm + FFN(128→512→128) + LayerNorm]
  - matches AM (Kool et al., 2019) exactly
- Decoder: context = `Linear([graph_mean ‖ last_selected], 128)` → 1× MHA glimpse → dot-product pointer with tanh(10) clip
- ~1.8M params (6× the LSTM model)

**CLI / config keys added:**
- `model_type`: `"lstm"` (default, backward-compat) or `"transformer"`
- `n_heads`: attention heads (default 8)
- `n_enc_layers`: encoder depth (default 3)
- `ff_dim`: FFN inner dimension (default 512)

**Starting config:** `configs/po_agp_transformer.json` — matches AM paper settings.

## PO Paper Observations (Pan et al., ICML 2025 — arXiv:2505.08735)

### Model Architecture
All tested architectures are Transformer encoder-decoder models (AM, POMO, Sym-NCO,
Pointerformer, MatNet). LSTM Pointer Networks are NOT used. Smallest tested: AM (3 layers,
128 dim, 8 heads, 512 FFN). Our LSTM is strictly more limited.

### Training Speed
- PO converges **1.5×–2.5× faster** than REINFORCE on POMO/Sym-NCO
- POMO + PO reaches RF-level in 40–60% of RF's training epochs
- Training lengths: 2000 epochs for TSP-100 (POMO), 4000 for CVRP-100

### Alpha (α) Calibration (Appendix E.2)
Grid searched over `{0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0}`.
- Models **with** built-in exploration (POMO multi-start) → **lower α** (0.01–0.03)
- Models **without** built-in exploration → **higher α** (0.05–0.1)
- POMO/TSP: α=0.05 (BT) — matches our current setting
- FFSP/large-scale: α=1.0–2.0 with Exponential model
- **Our plain pointer net has no built-in exploration → lean toward α=0.1**

### Preference Model Selection (Appendix F.4)  Critical for our bimodal trap problem:
- **Bradley-Terry (BT):** conservative; gradient vanishes when `σ(Δlogπ) ≈ 1` (saturated pairs)
  → works well for **small-scale or easy** problems
- **Exponential:** `f(x) = exp(x)`; stronger gradient, doesn't saturate
  → recommended for **harder/larger** problems to escape local optima
- **Impact on us:** During bimodal reward splits (covered vs uncovered), BT pairs quickly
  saturate and produce zero gradient — this is the primary cause of our "dead run" episodes.
  Switch to `preference_loss: "exponential"` for the Transformer run.

### Adapting to New Problems (Appendix E.2)
Two extensions recommended for novel COP applications:

1. **Length-control normalization** (for variable-length outputs):
   `f(α · [logπ(τ₁)/|τ₁| − logπ(τ₂)/|τ₂|])`
   Already done in our implementation via length-normalized log-probs.

2. **Margin enhancement** (for limited-capacity models):
   `f(α · [logπ(τ₁) − logπ(τ₂)] − γ)` where γ is a margin parameter.
   Explicitly for models that struggle to separate similar solutions.
   → **Add `margin` config key** for Tier 2 sweeps.

### Local Search Fine-tuning (§3.4)
After policy convergence, standard training reliably stops improving (RL and PO both).
LS fine-tuning: form pairs `(τ, LS(τ))` where `LS(τ)` is locally improved solution.
PO handles off-policy LS naturally (no importance sampling needed, unlike REINFORCE).
POMO+PO+FT: 0.07% → **0.03% gap** on TSP-100.
- For AGP: 3-opt–style guard swapping (remove a guard, add 2 nearby covering ones) is the
  natural LS move. Implement as `skip_finetune: false` in a later run.

### Advantage Separation (§4.2, Figure 3)
- PO clearly separates positive (good trajectory) vs negative (bad trajectory) advantages
- REINFORCE collapses to near-zero advantages when rewards are similar
- PO shows much broader advantage distribution → more stable gradient directions
- This is exactly why PO is beneficial for our coverage-saturated regime

## Execution Notes
- Keep one variable family changing per tier to isolate effect.
- Save run metadata (`config`, checkpoint path, date, git branch) with each experiment result.
- Compare only runs evaluated on the same validation subset size (`epoch_eval_k`).

## Implemented Now (Tier 0 + Tier 1)
- Tier 0 baseline config:
  - `configs/tier0/po_agp_tier0_baseline.json`
- Tier 1 sweep definition:
  - `configs/tier1/po_agp_tier1_sweep.json`
- Tier 1 generator/runner:
  - `tools/run_po_tier1_sweep.py`

### Commands
- Run Tier 0 baseline:
  - `python po_agp.py --config configs/tier0/po_agp_tier0_baseline.json --verbose`
- Generate Tier 1 configs only:
  - `python tools/run_po_tier1_sweep.py`
- Generate and run Tier 1 sweep sequentially:
  - `python tools/run_po_tier1_sweep.py --run`

---

## Current Direction: Learned Editor (active)

The PO fine-tuning approaches in Tiers 2–4 consistently failed to absorb LS into the policy
(every variant slid along the same size/coverage trade-off curve). Quantitative scoping
attributed the failure to a structural set-vs-sequence mismatch: the autoregressive
LSTM hidden state cannot represent set-symmetric pruning decisions cleanly. Tier 5's
Transformer-encoder pivot (5 Mar 2026) was tried and reverted (12 Mar 2026).

Pivot: rather than re-architecting the policy, **add a small set-equivariant editor module
on top of the frozen `lstm_bt` pointer**. Pointer produces a seed; editor refines it via
short sequences of REMOVE / SWAP / STOP decisions imitating LS.

### Phase 1 — Oracle editor (DONE)

Editor reads 6-dim per-vertex features at inference: `(x, y, in_S, vis_frac, marg_cov,
redundancy)`. The last three come from the `disc_vis` matrix per state — i.e. a geometric
oracle is queried each edit step.

Files: `edit_head.py`, `train_editor.py`, `eval_editor.py`, `tools/build_ls_trajectories.py`.

**Results on full validation (1224 polygons), frozen `lstm_bt` pointer:**

|                                  | cov   | \|S\|/n | \|S\|/OPT | recovery median |
|---                               |---    |---      |---        |---              |
| pointer alone                    | 0.970 | 0.214   | 1.36      | (seed)          |
| pointer + LS (target)            | 0.970 | 0.158   | 0.92      | 1.0 (gold)      |
| pointer + editor (oracle, no DAgger) | 0.972 | 0.187   | 1.05  | **0.881**       |
| pointer + editor (oracle, DAgger ep-21) | 0.940 | 0.166 | 0.92 | **0.955**     |

Headline: editor recovers **88 % of LS reward gain on a typical polygon, beats LS on 25 %**.
With DAgger it climbs to **95 %**, at the cost of dropping cov below baseline on the tail.

### Phase 2 — Geo-free pilot (DONE — under-powered negative)

Editor reads only `(x, y, in_S)` at inference — no geometric oracle. `disc_vis` is still
allowed during training (LS labels, DAgger queries, coverage gate) but never read at
deployment.

Pilot run (`configs/editor_train_dagger_geo_free.json`, 80 k params, 1 attention layer):

- recovery_median **0.000** at 5 epochs.
- `remove_top1 = 0.06`, `swap_in_top1 = 0.007` (essentially random).
- `gate_stopped = 0.9` — coverage gate did 90 % of the halting; editor itself was blind.

This is a *fair pilot* of the simplest geo-free setup, **not** a fair test of the
hypothesis "visibility can be learned implicitly from LS action labels." The pilot was
underpowered along three axes simultaneously: too few parameters for 2D geometric
reasoning, no polygon topology in the input, no signal pushing latents toward encoding
visibility.

### Phase 3 — Scaled geo-free editor (ACTIVE)

Goal: actually test the implicit-visibility hypothesis. **Three additive upgrades**, all
keeping the pointer frozen:

1. **Bigger editor.** `hidden = 128`, `n_attn_layers = 3`, `heads = 8`. About 500 k – 1 M
   parameters. Enough capacity for multi-hop reasoning over polygon vertices.
2. **Polygon topology as input.** Add per-vertex features that describe the polygon's
   shape (not its visibility):
   - `pos_norm = v / n` — cyclic position in the boundary.
   - `(dx_prev, dy_prev)`, `(dx_next, dy_next)` — vectors to adjacent boundary vertices.
   New `D_in = 8`. Critically, none of these require `disc_vis`; they are derivable from
   the polygon's `(x, y)` list plus its cyclic order, which is part of the input by
   definition.
3. **Auxiliary visibility-prediction loss (training-only).** Add a small head on the
   editor's per-vertex latents that predicts `(vis_frac, marg_cov, redundancy)`. The
   targets come from `disc_vis` during training. Multi-task loss:
   `L = w_action·L_action + w_stop·L_stop + w_aux·MSE(latent_proj, vis_targets)`.
   At inference, the auxiliary head is unused — but the latents it shaped during
   training are still queried by the action head. This is the operationalisation of
   "encode visibility in latent space."

**Files to modify (all additive):**
- `edit_head.py` — extended `compute_vertex_features_geo_free` (D_in=8), `EditHead`
  gains `aux_visibility` flag + `aux_head` module, `edit_loss` gains `aux_target` arg.
- `train_editor.py` — compute aux targets per batch (uses `disc_vis`, allowed during
  training), pass through `edit_loss`. New CLI flags: `--aux-visibility`,
  `--aux-weight`, plus `--editor-hidden 128 --editor-attn-layers 3 --editor-heads 8`
  for the model scale-up.
- `eval_editor.py` — no behaviour change at inference (aux head is dropped by `predict`);
  feature dim derives from saved checkpoint args.
- `configs/editor_train_dagger_geo_free_v2.json` — new config.

**Decision criterion:**

| recovery_median on full val | verdict |
|---|---|
| ≥ 0.75 | **success** — paper claim holds: visibility learned implicitly, no geometric oracle at inference |
| 0.40 – 0.75 | partial — useful as an ablation; the oracle editor remains the main contribution |
| < 0.40 | strong negative — implicit visibility learning at this data scale is intractable; pivot to "one-time static visibility features at deployment" |

### Phase 4 — Eval + ablation (NEXT)

Two side-by-side full-val runs to produce the paper table:
- pointer + editor (oracle, 6-dim) — existing best
- pointer + editor (geo-free v2, 8-dim + aux loss) — Phase 3 output

### Verification

1. `ast.parse` on `edit_head.py`, `train_editor.py`, `eval_editor.py`.
2. Smoke train (50 polygons, 1 epoch) confirms the upgraded model initialises and
   forward/backward pass on the new feature dim.
3. Smoke eval with `--no-ls-reference` confirms `[disc_vis]` is not loaded at inference
   for the geo-free-v2 checkpoint.
4. Full training run logs auxiliary loss separately; rollout banner reports
   `recovery_med` per epoch.
