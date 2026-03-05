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

## Tier 5 — Architecture Upgrade (Transformer Encoder)

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
