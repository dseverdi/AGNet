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
