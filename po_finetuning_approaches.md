# PO Fine-Tuning for Art Gallery Problem — Full History of Approaches

## Problem Statement

We have a trained LSTM Pointer Network that solves the Art Gallery Problem (AGP): given a simple polygon with n vertices, select a subset S of vertices as guards to maximize coverage of the polygon interior, using as few guards as possible.

The model outputs a variable-length sequence of vertex indices autoregressively: $v_1, v_2, \ldots, v_k, \text{EOS}$. At each step, it attends over all candidate vertices and selects one, then masks it. The EOS token terminates the sequence.

**Goal:** Fine-tune the pre-trained policy using Preference Optimization (PO) with Local Search (LS) to reduce |S|/OPT (guard count relative to optimal) while maintaining high coverage, WITHOUT using any geometric processing at inference time (only (x,y) coordinates in, guard indices out).

## Architecture

- **Model:** LSTM Pointer Network (Vinyals et al., 2015 style)
  - Encoder: unidirectional LSTM (embedding=128, hidden=128, 1 layer)
  - Decoder: LSTM with attention (1 glimpse), pointer mechanism with tanh(10) clip
  - Input: (x,y) coordinates of polygon vertices
  - Output: autoregressive sequence of guard vertex indices + EOS
  - Parameters: 330,880
  - The encoder is NOT bidirectional — vertex i only has context from v_1...v_i
- **Training:** REINFORCE with baseline, then PO fine-tuning
- **Reward:** `coverage - λ·(|S|/n) - π·max(0, τ-coverage)` where τ=0.99, λ=1.0, π=3.0
- **Data:** 8,867 training polygons, 1,224 validation. Polygons have ~128 vertices on average. Variable complexity. Cannot generate unlimited data (each polygon requires CGAL visibility precomputation).

## Reference Paper

Pan et al., "Preference Optimization for Combinatorial Optimization Problems," ICML 2025 (arXiv:2505.08735).

Key points from the paper:
- **§3.4 Eq.9:** `f(α·[log πθ(LS(τ)|x) − log πθ(τ|x)])` — gradients through BOTH on-policy and LS log-probs
- **No reference model** in fine-tuning (unlike standard DPO/RLHF)
- **Exponential preference** (App F.4): `f(x) = x` recommended for hard problems (no saturation unlike Bradley-Terry sigmoid)
- **Length normalization** (§4.2): `log π(τ)/|τ|` for variable-length outputs
- **All paper experiments use Transformer encoder-decoder models** (AM, POMO, Sym-NCO, Pointerformer). No LSTM tested.
- Fine-tuning epochs ≈ 5% of total training epochs
- PO absorbs off-policy LS solutions naturally without importance sampling

## Baseline Performance

| Metric | Pre-FT model (greedy) | LS post-processing (K=30) |
|--------|----------------------|--------------------------|
| Coverage | 0.974 | 0.989 |
| |S|/n | 0.187 | 0.148 |
| |S|/OPT | 1.182 | 0.928 |
| Inference | ~0.2s, no geometry | Several seconds, needs visibility matrix |

The LS post-processing target (cov=0.989, |S|/OPT=0.928) represents what we want the model to learn to do WITHOUT geometry at inference.

## Local Search Implementation

The LS (`local_search_improve_disc`) operates on the discretized visibility matrix (precomputed, 500 sample points):
- **REMOVE:** Try removing each guard, keep the removal that improves reward most (maintains coverage while reducing |S|)
- **SWAP:** Try replacing each guard with each non-guard vertex, keep best improvement
- **ADD:** Try adding each non-guard vertex
- Greedy first-improvement, iterates up to `max_iter` rounds
- Uses `monotone_coverage=True` during FT (coverage can only stay same or increase within LS)

When LS is used during PO fine-tuning:
1. Sample K rollouts on-policy → τ₁...τ_K
2. Run LS on each rollout → LS(τ₁)...LS(τ_K)
3. Teacher-force LS solutions through the model → log π(LS(τ_k))
4. Pool 2K solutions (K on-policy + K LS-improved), build preference matrix
5. Compute PO loss through both on-policy and LS log-probs

**Teacher-forcing detail:** LS solutions are re-ordered to match the original rollout ordering: `ls_ordered = [g for g in rollout_order if g in ls_sol_set]`. So if LS only removes guards, the teacher sequence is a strict subsequence of the rollout. Swaps introduce new vertices the rollout never produced.

## Key Structural Limitation: LSTM Hidden-State Cascade

The LSTM decoder has hidden state $h_t = f(h_{t-1}, x_{a_{t-1}})$ — each step depends on the previous step's action embedding. When teacher-forcing a sequence that differs from what the model would naturally produce:
- Step k forces action $a_k^{LS}$ instead of $a_k^{model}$
- This changes $h_{k+1}$ which cascades to all subsequent steps
- By step k+3, the hidden state may be in a region never seen during training
- Log-probs computed from these corrupted states are unreliable

**Transformers don't have this problem** because self-attention computes each position independently from the full context. This is the fundamental reason the paper's approach works on Transformers but struggles on our LSTM.

## Comparison with Other Methods in Our Codebase

| Method | N | Coverage | |S|/n | |S|/OPT | Time | Notes |
|--------|---|----------|-------|---------|------|-------|
| **PO-AGP + LS** (K=30) | 1224 | 0.989 | 0.148 | **0.928** | ~few s | Best overall, uses geometry |
| **Greedy Prune** (exact oracle) | 1224 | 0.999 | 0.166 | 1.087 | 59s | Near-perfect but very slow |
| **PO-AGP baseline** (no FT) | 1224 | 0.970 | 0.194 | 1.213 | 0.2s | Pre-FT model, exact eval |
| **SS-AGP Prune** (v2, greedy) | 1224 | 0.991 | 0.131 | 0.821 | — | v2, sampled oracle (different eval) |
| **POMO-AGP** | 200 | 0.987 | 0.334 | 2.336 | 1.3s | Overguards 2× |
| **RL-AGP** (v3) | 100 | 0.151 | 1.000 | 6.539 | 0.1s | Degenerate (broken) |
| **Q-Learning Prune** | 1224 | 1.000 | 1.000 | 6.926 | 40s | Degenerate (no pruning) |

---

## Fine-Tuning Approaches: Full History

### Approach 1: SFT (Supervised Fine-Tuning on LS Solutions)

**Idea:** Directly train the model to reproduce LS solutions via cross-entropy loss (standard teacher forcing).

**Result:** loss=7.23, coverage dropped, early stop triggered immediately.

**Why it failed:** Teacher-forcing cascade. The LSTM hidden state diverges catastrophically when forced to emit the full LS sequence. Every step's error compounds through the recurrent state. The NLL loss is absolute (not relative), so the model receives large gradients from corrupted hidden states.

---

### Approach 2: LS Distillation v1/v2/v3

**Idea:** Various distillation schemes — per-rollout LS targets, different KL formulations, weighted losses.

**Result:** All collapsed. Per-rollout targets didn't help.

**Why it failed:** Same hidden-state cascade as SFT. Absolute NLL on LS sequences = pure teacher-forcing regardless of how it's wrapped. The relative quality of the target doesn't matter when the hidden state is corrupted.

---

### Approach 3: Offline DPO on Cached Pairs

**Idea:** Pre-compute (rollout, LS(rollout)) pairs offline. Train with DPO loss on cached pairs.

**Result:** Didn't reduce guards.

**Why it failed:**
- Pair acceptance required cov_teacher ≥ 0.99 → rejected ~50% of pairs
- Guard-reducing pairs were de-weighted to 0.176 in the loss
- Best model was tracked by reward (not |S|/OPT) → saved wrong checkpoint
- Stale pairs: model changes but pairs don't update → off-policy drift

---

### Approach 4: LEXRF (REINFORCE with Lexicographic Rank Advantages)

**Idea:** Use REINFORCE but with lexicographic ranking of K rollouts — feasible solutions (cov≥τ) ranked by fewer guards, infeasible ranked by higher coverage.

**Result:** Not fully tested. Minimal signal observed.

**Why it failed:** Converged policy → K=8 rollouts are near-identical → near-zero advantages. Low entropy from the pre-trained model means all rollouts look the same, so there's no preference signal.

---

### Approach 5: LexPO v1 (On-Policy PO, LS as Evaluator Only)

**Idea:** Use PO with K on-policy rollouts only. LS evaluates quality but doesn't provide teacher sequences.

**Result:** Minimal signal.

**Why it failed:** Without LS solutions in the pool, all K rollouts are near-identical (same issue as LEXRF). No diverse "better" solutions to learn from.

---

### Approach 6: LexPO v2 (Paper §3.4 Faithful)

**Config:** Pool of 2K solutions (K on-policy + K LS-improved). Bradley-Terry loss. Teacher-force LS with gradients. Reference model KL regularization. LS budget=3.

**Result:** Epoch 1: |S|/OPT 1.182 → 1.087 (−8%!). Epoch 2: COLLAPSED (loss 1.4 → 8.0).

**Why it failed:** Gradients through teacher-forced LS log-probs → hidden-state cascade caused gradient explosion. The un-detached LS log-probs had enormous gradients flowing through corrupted hidden states.

**Key observation:** Epoch 1 showed the signal IS there — 8% guard reduction in one epoch. The problem is stability, not signal quality.

---

### Approach 7: LexPO v2 + Detached LS Log-Probs

**Change from v6:** `ls_log_probs.detach()` — gradients only flow through on-policy rollouts, not through teacher-forced LS sequences.

**Result:** Epochs 1-3: |S|/OPT 1.182 → 1.096 (−7.3%). Epoch 5: collapsed.

**Why it failed:** Coverage floor too tight (0.97, computed from LS-inflated per-instance best). LS solutions had cov~0.99 → per-instance floor ≈ 0.97 → most on-policy rollouts (cov~0.95) classified as "infeasible" → model pushed coverage UP instead of reducing guards.

---

### Approach 8: LexPO v2 + Detached + Relaxed Floor (0.943)

**Changes from v7:**
- Per-instance floor computed from on-policy coverages only (not LS-inflated)
- Global floor = baseline - 0.03 = 0.943
- Per-instance margin of 0.03

**Result:** Epochs 1-3: 1.182 → 1.096 (−7.3%). Epoch 5: 1.210. Epoch 7: 2.637 (collapse).

**Why it failed:**
1. Best model tracked by REWARD → saved wrong checkpoint (reward favors coverage; epoch 5 had higher reward but worse |S|/OPT)
2. No LR decay → overshooting after 3 good epochs
3. Detaching LS log-probs removed the "learn to produce LS solutions" signal — only pushes bad rollouts DOWN, doesn't pull policy TOWARD LS

---

### Approach 9: Paper-Faithful with LSTM Workarounds (BEST RUN)

**Changes (all applied simultaneously):**
1. **Un-detached LS log-probs** — gradients through both terms per paper Eq.9
2. **Exponential preference** — `f(x) = x` instead of BT `f(x) = log σ(x)`. No saturation on separated pairs.
3. **Removed reference model** — no KL regularization, per paper §3.4
4. **Length normalization** — `log π(τ)/|τ|` per paper §4.2
5. **Aggressive grad_norm=0.1** — key LSTM stability measure against teacher-forcing cascade
6. **Cosine LR schedule** — LR=1e-5 decaying to 1e-6
7. **Best model tracked by |S|/OPT** (lower=better) with coverage floor gate
8. **Early stop** on |S|/OPT > 1.25× baseline
9. **LS budget=5**, sample_temp=1.5, epochs=50

**Config:**
```json
{
    "max_grad_norm": 0.1,
    "alpha": 0.1,
    "finetune_epochs": 50,
    "finetune_lr": 1e-5,
    "finetune_k": 8,
    "ft_loss_type": "lexpo",
    "ft_no_ref_model": true,
    "ft_rl_ls_budget": 5,
    "ft_sample_temp": 1.5,
    "ft_cap_coverage": "false"
}
```

**Result — Run A (cov_floor_global = baseline - 0.03):**
```
Baseline: cov=0.974, |S|/OPT=1.182
Epoch  1: cov=0.950, |S|/OPT=1.017  *best saved*
Epoch  3: cov=0.939, |S|/OPT=0.972
Epoch  9: cov=0.935, |S|/OPT=0.972
Epoch 11: cov=0.934, |S|/OPT=0.950
Epoch 12: cov=0.928, |S|/OPT=0.920  ← true best but NOT saved (cov < floor 0.944)
Epoch 13: cov=0.904, |S|/OPT=0.879  ← absolute best but NOT saved
Epoch 14: cov=0.978, |S|/OPT=2.798  ← collapse, early stop triggered
Restored epoch 1 (|S|/OPT=1.017)

Final eval: greedy cov=0.948, |S|/OPT=1.017
            eval best (K=8): |S|/OPT=1.07, cov=0.966
```

**Analysis:** First run to sustain improvement past epoch 3. Steady progress for 13 epochs. Collapse at epoch 14 (same pattern, just delayed). But best-model gating floor (0.944) was too tight — didn't save the truly best checkpoints (epochs 12-13 had cov < 0.944).

**Result — Run B (cov_floor_global relaxed to baseline - 0.07):**
```
Baseline: cov=0.975, |S|/OPT=1.182
Epoch  1: cov=0.954, |S|/OPT=1.015  *best*
Epoch  2: cov=0.948, |S|/OPT=0.988  *best*
Epoch  3: cov=0.941, |S|/OPT=0.974  *best*
Epoch  4: cov=0.942, |S|/OPT=0.970  *best*
Epoch  9: cov=0.935, |S|/OPT=0.947  *best*
Epoch 10: cov=0.920, |S|/OPT=0.893  *best* ← SAVED
Epoch 11: cov=0.954, |S|/OPT=1.651  ← collapse, early stop
Restored epoch 10 (|S|/OPT=0.893)

Final eval: greedy cov=0.916, |S|/OPT=0.893
            eval best (K=8): |S|/OPT=1.01, cov=0.943
```

**Analysis:** Best result yet. |S|/OPT=0.893 beats LS target (0.928). But coverage dropped to 0.916 (greedy) / 0.943 (eval best). The model learned to emit fewer guards but NOT to place them better — just stopping earlier. Coverage drops ~2.5% per 10% guard reduction, consistent with "same placement quality, fewer guards."

**Key pattern across all epochs:** coverage and guard count drop in lockstep. The model cannot simultaneously maintain coverage AND reduce guards because it lacks geometric reasoning about visibility.

---

### Approach 10: Minimal LS Edits (Budget=1)

**Hypothesis:** Budget=5 allows LS to drift far from the rollout → model can't isolate which specific change helps. Budget=1 = at most one remove or swap → minimal-edit preference pairs → cleaner learning signal.

**Additional change:** Tighter coverage floor (baseline - 0.01 = 0.963 lexpo, baseline - 0.02 = 0.953 global) to force guard reduction via better placement rather than early stopping.

**Result:**
```
Baseline: cov=0.973, |S|/OPT=1.182
Epoch 1: cov=0.945, |S|/OPT=0.987
Epoch 2: cov=0.944, |S|/OPT=0.972
Epoch 5: cov=0.955, |S|/OPT=1.059  *best saved* (only one above floor 0.963)
Epoch 6: cov=0.977, |S|/OPT=1.461  ← model pushed coverage up
Epochs 8-50: oscillated between cov=0.94-0.97, |S|/OPT=1.06-1.19
No further improvement after epoch 5.

Final eval: greedy cov=0.957, |S|/OPT=1.059
            eval best (K=8): |S|/OPT=1.23, cov=0.975
```

**Why it failed:** Two problems confounded:
1. **Budget=1 produces almost no signal.** Loss magnitudes ~0.01 vs ~0.15 with budget=5. Most rollouts get zero or one guard changed — preference pairs are near-ties.
2. **Tight floor (0.963) killed guard reduction.** Most on-policy rollouts have cov < 0.963 → ranked infeasible → model chased coverage instead of reducing guards.

---

### Approach 11 (PENDING): Budget=5 + Tight Floor

**Hypothesis:** Combine strong LS signal (budget=5) with tight coverage constraint (floor = baseline - 0.01). The model must maintain coverage (can't just stop early) while learning from LS which guards to swap.

**Status:** Config ready, not yet run.

---

## Summary of All Runs

| Run | LS Budget | Cov Floor | Preference | Grad Norm | Best Greedy |S|/OPT | Best Greedy Cov | Eval Best |S|/OPT | Eval Best Cov | Collapse? |
|-----|-----------|-----------|------------|-----------|-------------|---------|----------------|-----------|---------|
| 1. SFT | — | — | CE | default | FAILED | — | — | — | Immediate |
| 2. Distill v1-3 | — | — | KL variants | default | FAILED | — | — | — | Immediate |
| 3. Offline DPO | — | — | DPO | default | ~baseline | ~baseline | ~baseline | ~baseline | No improvement |
| 4. LEXRF | 0 | — | REINFORCE | default | ~baseline | ~baseline | — | — | No signal |
| 5. LexPO v1 | 0 | — | BT | default | ~baseline | ~baseline | — | — | No signal |
| 6. LexPO v2 | 3 | 0.97 | BT | 0.5 | 1.087 (e1) | ~0.96 | — | — | e2 explosion |
| 7. LexPO v2 + detach | 3 | 0.97 | BT | 0.5 | 1.096 (e3) | ~0.95 | — | — | e5 collapse |
| 8. v7 + relaxed floor | 3 | 0.943 | BT | 0.5 | 1.096 (e3) | ~0.95 | — | — | e7 collapse |
| **9A. Paper-faithful** | **5** | **0.944** | **Exp** | **0.1** | **1.017 (e1)** | **0.950** | **1.07** | **0.966** | e14 collapse |
| **9B. 9A + relaxed gate** | **5** | **0.905** | **Exp** | **0.1** | **0.893 (e10)** | **0.920** | **1.01** | **0.943** | e11 collapse |
| 10. Minimal LS | 1 | 0.963 | Exp | 0.1 | 1.059 (e5) | 0.955 | 1.23 | 0.975 | No collapse but no learning |

## Core Diagnosis

### What works
- PO + LS with Exponential preference, length normalization, no ref model
- Aggressive grad_norm=0.1 delays collapse from epoch 3 to epoch 10-14
- Cosine LR schedule prevents late-epoch overshooting
- Best-model tracking by |S|/OPT (not reward)
- Budget=5 LS gives strong signal; budget=1 is too weak

### What doesn't work
- The model cannot simultaneously reduce guards AND maintain coverage
- Every run shows the same pattern: coverage drops ~2.5% per 10% guard reduction
- This is because the model learns "emit fewer guards" (easy — just trigger EOS earlier) instead of "place guards more efficiently" (hard — requires understanding visibility geometry)
- The LSTM with 128-d hidden state and (x,y) inputs cannot learn visibility relationships
- LS uses an explicit visibility matrix to decide swaps; the model has no access to this information and cannot infer it from coordinates alone

### The fundamental limitation
The autoregressive LSTM decoder places guards sequentially. Once placed, a guard cannot be reconsidered. The model's hidden state at step k encodes "what I've done so far" in 128 floats — insufficient to represent which areas of a ~128-vertex polygon are covered.

LS achieves cov=0.989 + |S|/OPT=0.928 because it has the visibility matrix and can make targeted remove/swap decisions. Teacher-forcing LS sequences through the LSTM provides a training signal, but the hidden-state cascade corrupts the gradient for anything beyond simple guard removal.

### The question
Can PO fine-tuning teach the LSTM to internalize LS-quality placement from (x,y) coordinates alone? After 11 approaches, the evidence suggests:
- YES for guard reduction: the model reliably learns to use fewer guards
- NO for placement quality: coverage always drops proportionally to guard reduction
- The gap between |S|/OPT=0.893@cov=0.916 (our best) and |S|/OPT=0.928@cov=0.989 (LS) is exactly this placement quality gap

## Open Questions / Untried Ideas

1. **Budget=5 + tight floor (pending):** Will forcing cov≥0.96 feasibility while giving strong LS signal produce better-placed guards? Or will it simply reduce the learning signal (most rollouts infeasible → fewer useful pairs)?

2. **Richer input features:** Precompute per-vertex visibility area fraction and feed as 3rd input feature. Gives the model geometric info it currently cannot infer. But requires visibility computation at inference — conflicts with "no geometry at inference" constraint. Could be used only during training as auxiliary info.

3. **Bidirectional encoder:** Current encoder is unidirectional — vertex i only sees v_1...v_i. Polygon vertices are cyclic, so a bidirectional encoder gives full context at each position. Small architectural change, may improve representation quality.

4. **Transformer encoder-decoder:** Planned as Tier 5. Eliminates hidden-state cascade (no teacher-forcing corruption). Global attention over all candidates at each decode step. But 1.8M params with 8,867 training instances → overfitting risk. Paper's problems (TSP, CVRP) have unlimited synthetic data.

5. **Curriculum from LS:** Start with remove-only LS (strict subsequences, no cascade), then gradually introduce swaps as the model stabilizes.

6. **Coverage-aware reward shaping during FT:** Instead of binary feasible/infeasible, use a continuous coverage bonus that strongly penalizes cov < 0.96 in the preference scoring.

7. **LS as post-processing (pragmatic path):** Accept the LSTM ceiling. Use the FT model as a better starting point for LS at inference. Evaluate FT checkpoint with --local-search to measure: does FT + LS beat baseline + LS?
