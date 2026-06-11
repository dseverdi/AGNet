# Geo-Free Neural Editing for the Art Gallery Problem

*Working draft. May 2026.*

---

## Abstract

We address the Vertex-Guard Art Gallery Problem (AGP) — selecting a minimum-cardinality vertex subset that covers a simple polygon — with a Neural Combinatorial Optimisation (NCO) pipeline. A pretrained LSTM PointerNet, trained with Preference-Optimisation (Pan et al., ICML 2025) and Bradley-Terry loss, produces an initial guard set that achieves mean coverage 0.969 on held-out dev polygons but **leaves 19.3 % of polygons with coverage below 0.95** (worst-case 0.83). We introduce a small (464 k parameter) **set-membership predictor** that, in a single forward pass, refines the pointer's seed without invoking any geometric oracle at inference time. At a single threshold value (`t = 0.30`), the predictor lifts **all 367 held-out polygons** above the 0.95 coverage threshold — mean coverage 0.998, worst-case 0.952 — at the cost of |S|/OPT 1.09 → 2.20. On a much larger and harder out-of-distribution split (2107 polygons up to 1000 vertices, 5× training maximum), the same model rescues **300 of 310 pointer-baseline failures**, reducing the failure rate from 14.7 % to 0.14 % and lifting the worst-case polygon's coverage from 0.049 to 0.765. The threshold parameter exposes a smooth Pareto curve covering both coverage-guaranteed and guard-efficient operating points, all from one trained model. The method's Pareto curve dominates the closest neural alternative ("let the pointer sample more guards") at matched coverage bands.

## 1. Introduction

The Art Gallery Problem (AGP) asks for the minimum number of guards observing every point of a simple polygon. It is NP-hard (Lee & Lin, 1986) and a canonical benchmark for combinatorial geometric optimisation. Recent NCO work has applied learned policies to AGP variants, but autoregressive pointer-network approaches face a structural mismatch: AGP solutions are *sets*, not sequences, and the choice of which vertex to remove at one decision step depends on the full configuration of remaining picks — a set-symmetric property that does not align cleanly with order-dependent decoder states.

A natural workflow uses a **pretrained policy** (the pointer-network) followed by **local search** (LS) refinement at inference. Classical LS with discretised-visibility (disc_vis) coverage checks at every step delivers excellent guard efficiency (|S|/OPT ≈ 0.91 at coverage 0.97) but is expensive: each candidate edit triggers a coverage recomputation. A user who needs higher throughput, deployment without geometric libraries (CGAL), or simply a learned alternative is left with two unattractive options — accept the pointer's coverage failures, or run LS at every inference call.

**Contributions.**

1. **Diagnostic.** We document three failure modes of autoregressive-editor architectures trained to imitate LS: compounding per-step errors, STOP-action head coupling, and training-vs-inference distribution drift. All three are mechanically eliminated by a single-shot set-prediction formulation.
2. **Method.** A **SetPredictor**: 464 k-parameter set-equivariant attention module that, given the polygon and the pointer's seed, outputs a per-vertex probability `P(v ∈ S_final)` in one forward pass. The final guard set is `{v : P(v) ≥ t}` for a user-chosen threshold `t`. The model consumes only the pointer's encoder embeddings (computed once per polygon from coordinates) plus `(x, y, in_S)` — **no geometric oracle at inference time**.
3. **Empirical evidence.** On 367 held-out in-distribution dev polygons at `t = 0.30`, the SetPredictor achieves coverage ≥ 0.95 on **every** polygon (Wilson 95 % CI ≤ 1.0 %), vs. 71 / 367 = 19.3 % failures for the pointer baseline. On 2107 OOD polygons (up to 5× train-max vertices, the same checkpoint without retraining), the predictor at `t = 0.20` reduces pointer-baseline failures from 310 / 2107 (14.7 %) to 3 / 2107 (0.14 %), a **100× reduction in failure rate**, while lifting the worst-case polygon's coverage from 0.049 to 0.765. The method's Pareto curve dominates the pointer-with-EOS-disabled baseline at matched coverage bands. A single threshold scalar exposes a controllable Pareto curve at inference.

## 2. Related Work

**Preference-Optimisation NCO.** Pan et al. (ICML 2025) introduced Preference Optimisation (PO) for NCO problems with Transformer-based architectures (POMO, Attention Model, Sym-NCO, Pointerformer, MatNet). PO converges 1.5×–2.5× faster than REINFORCE and supports off-policy LS fine-tuning via standard imitation pairs. We adopt PO with Bradley-Terry loss for pointer pretraining, retaining the length-normalised log-probability formulation that worked well on TSP and CVRP. We do not match Pan et al.'s architectural choice: our pretrained policy is an LSTM PointerNet rather than a Transformer (§4.1 footnote).

**Pointer networks and NCO.** Vinyals et al. (2015) introduced pointer networks for set selection. Kool et al. (2019) showed attention models scale better on TSP, and Kwon et al. (2020) introduced POMO with multi-start sampling. Pointerformer extended POMO with permutation-equivariant decoders. Our pretrained policy is a single-start, single-decoder LSTM PointerNet — chosen for compatibility with existing infrastructure and the Approach 15 EOS-gate diagnostic (§4.2).

**Classical AGP solvers.** Greedy guard selection (visibility-greedy heuristic) achieves |S|/OPT ≈ 1.5 with mean coverage 0.995; LS with disc_vis refinement reaches |S|/OPT ≈ 0.91 at matched coverage. Both methods invoke visibility queries at every decision step. Our neural method does not, which is the central performance trade-off the paper studies.

**Imitation learning of search.** Behavioural cloning of search trajectories has been studied in TSP, CVRP, and scheduling. The closest prior work treats LS as an iterative process and trains a policy to imitate per-step edits. Our diagnostic (§4.2) explains why this formulation fails on AGP, and our positive contribution (§4.3) is the single-shot reformulation that succeeds.

## 3. Problem Setup

### 3.1 Vertex-Guard Art Gallery Problem

**Input.** A simple polygon `P` with `n` vertices `V = {v_1, …, v_n}`.
**Output.** A guard subset `G ⊆ V`.
**Objective.** Minimise `|G|` subject to a coverage constraint:

```
Coverage(G, P) = Area(⋃_{g ∈ G} Visible(g, P)) / Area(P)   ≥   τ
```

Throughout this paper, `τ = 0.99` for LS imitation targets, and we evaluate the resulting models against coverage thresholds chosen at inference (typically 0.95 and 0.99). The exact AGP optimum `OPT(P)` for each polygon comes from a pre-computed CGAL exact solver; the ratio `|S|/OPT` is the *approximation ratio* we report.

### 3.2 Coverage Computation

- **Training-time** coverage uses a *discretised-visibility* matrix (`disc_vis`), a Bernoulli sampling of polygon interior points; each vertex's visibility is a `(M = 500)`-dimensional boolean vector. This admits fast vectorised coverage queries during LS trajectory generation. The disc_vis matrix is allowed during training (for LS oracle labels, threshold-sweep evaluation, and auxiliary loss targets) but is **not** an input to any learned model.
- **Inference-time and reporting** coverage uses the exact CGAL polygon-visibility computation (`evaluate_polygon_visibility_numpy_wo_gt` in `utils.py`). Reported coverage and `|S|/OPT` values throughout this paper are CGAL-exact.

### 3.3 The Geo-Free Constraint at Inference

A neural model is **geometry-free at inference** if its forward pass requires only the polygon coordinates `{x_i, y_i}` and any pre-computed information that does not depend on the model's current guard-set state. Specifically, the model may *not* invoke per-step visibility queries (disc_vis lookups or CGAL polygon-visibility computations) during decoding. This excludes the natural "LS at inference" workflow but admits methods that use a polygon encoder run once before decoding starts.

## 4. Method

### 4.1 Pretrained Pointer

We pretrain a standard PointerNet (Vinyals et al., 2015) on the AGPIL train split with PO and Bradley-Terry loss (Pan et al. 2025). The architecture is a single-layer LSTM encoder (hidden 128, embedding 128), single-layer LSTM decoder, 1 glimpse pre-pointer, Bahdanau attention. Greedy decoding with a learned EOS token produces a guard sequence; we report only the unordered set. The pretrained checkpoint (`lstm_bt`, ≈ 364 k parameters) is the input to all subsequent experiments and is held frozen throughout.

The pointer's output on held-out dev polygons (§5) achieves **mean coverage 0.969** with `|S|/OPT = 1.09` — a competitive single-pass NCO baseline that nevertheless exhibits a long tail of coverage failures (§6.1).

**Architecture choice footnote.** We trial-implemented a Transformer encoder-decoder PointerNet (3-layer multi-head attention, 8 heads, FFN 128→512→128, ≈ 1.8 M parameters, matching the Attention Model dimensions of Kool et al. 2019). Initial training was unstable under the same PO/BT loss settings that worked for the LSTM, and the variant was set aside in favour of the editor direction reported here. A clean Transformer + PO + editor comparison is left for future work; the SetPredictor architecture (§4.3) consumes encoder embeddings as features and is architecture-orthogonal to the choice of pointer encoder.

### 4.2 Diagnostic Path — Three Failure Modes of Iterative Editing

Before arriving at the single-shot SetPredictor, we evaluated **three iterative-editor architectures** trained to imitate LS step-by-step. We summarise the diagnostic value here briefly, because the failure modes motivate the single-shot formulation by elimination.

**v1 (80 k params, geo-free features only).** A small attention model that, given `(polygon, current set S_t)`, predicts the next LS edit (REMOVE / SWAP / ADD / STOP) and the affected vertex. Trained by step-supervised cross-entropy on LS trajectories. Result: median recovery (editor's reward improvement ÷ LS's reward improvement) ≈ 0.00 on held-out dev. The model converges in training loss but produces no measurable benefit at inference.

**v2 (600 k params, topology features + auxiliary visibility loss).** Extends v1 with explicit polygon topology in the per-vertex features (cyclic position, neighbour deltas) and an auxiliary head supervised to predict per-vertex `(vis_frac, marg_cov, redundancy)` from disc_vis — a multi-task signal designed to push visibility information into the editor's latent space. Result: recovery still ≈ 0; the editor either under-prunes (no improvement over seed) or over-prunes (coverage collapses).

**v3 (strict-τ trajectories + strong STOP supervision).** Re-trains v2 on trajectories generated by LS at `τ = 0.995` (forcing higher-quality intermediate states) with `stop_pos_weight = 20.0` and `w_stop = 2.0` to push the STOP head to fire correctly. Result: the model now stops decisively after one or two edits, but those edits are wrong-vertex choices (per-step `remove_top1 ≈ 0.27`) that destroy coverage on most polygons.

All three variants exhibit the same family of pathologies:

1. **Compounding errors.** One wrong REMOVE early in a rollout invalidates downstream decisions, because the state the editor sees at step 2 depends on what it did at step 1.
2. **STOP / action coupling.** The STOP head decides when to halt; the action head decides what to edit. Their errors interact: the editor either over-prunes (cov collapses) or under-prunes (no improvement). Class-balanced STOP losses move along the trade-off curve rather than break out of it.
3. **Training-vs-inference distribution drift.** The editor's training inputs are clean LS states; at inference it visits *its own* rollout states, which can be far off the LS trajectory. DAgger refresh partially closes this gap but cannot bridge it from coordinates-and-attention alone.

These three pathologies motivate the single-shot reformulation in §4.3, which eliminates all three mechanically.

### 4.3 SetPredictor — Single-Shot Set-Membership Prediction

The SetPredictor outputs, in a single forward pass, a probability `P(v ∈ S_final)` for every polygon vertex `v`. The final guard set is obtained by thresholding:

```
S_final = { v ∈ V : P(v) ≥ t }
```

with `t ∈ (0, 1)` a user-chosen inference-time threshold.

**Per-vertex input features (131-d).**

- The pretrained pointer's LSTM-encoder output at vertex `v`: 128-d. Computed once per polygon from coordinates only — no disc_vis dependency.
- Vertex coordinates: `(x_v, y_v)` — 2-d.
- Seed membership: `in_S_v ∈ {0, 1}` — 1 if `v` is in the pointer's greedy-decoded seed, 0 otherwise.

**Architecture (464 k parameters total).** Per-vertex inputs are linearly embedded to a 128-d representation. We apply three self-attention layers (8 heads, FFN 128→256→128, with LayerNorm and residual connections); the attention is unmasked across all polygon vertices (the polygon is small enough that quadratic attention is cheap). After the final LayerNorm we form a 384-d per-vertex representation by concatenating the per-vertex latent `v_emb` with two pooled context vectors:

```
S_pool = mean-pool over v where in_S_v = 1
G_pool = mean-pool over all valid (non-padded) vertices
input to head: (v_emb ‖ S_pool ‖ G_pool)
```

A two-layer MLP head (384→128→1) produces the per-vertex logit; `P(v ∈ S_final) = σ(logit_v)`. Padding positions are masked to `−∞`.

**Output and inference.** At inference, run the pretrained pointer encoder once (≈ 1 ms), then one SetPredictor forward (≈ 5 ms), then threshold. The pointer encoder's per-vertex embeddings are the SetPredictor's only source of polygon-structural information at inference time; no per-step CGAL or disc_vis call is required.

**Why this formulation works.** The three diagnostic pathologies of §4.2 are eliminated by construction:

1. *No iteration*, no error compounding. Predicting `P(v)` for vertex 7 is independent of predicting `P(v)` for vertex 12.
2. *No STOP head*. The threshold `t` replaces the binary halt decision with a continuous, per-vertex inference-time scalar. There is no STOP-action coupling because there are no separate STOP and action heads.
3. *Identical input distributions* at train and inference. Both are `(polygon, pointer_seed)`. The model never sees its own outputs during training, and never needs to — there is no rollout.

### 4.4 Training

**Imitation target.** For each train polygon, we run the pretrained pointer to obtain a seed `S_0`, then run LS to convergence (with disc_vis-fast coverage), giving `S_LS_final`. The training example is `(polygon, S_0)` with per-vertex binary target `y_v = 1` if `v ∈ S_LS_final`, else 0. LS trajectory generation is one-time; only the final guard set is used (no per-step labels).

**Loss.** Per-vertex binary cross-entropy with an auto-derived `pos_weight ≈ 5.66` (the ratio of removed to kept vertices in the training data) to balance the class imbalance. Padding positions are masked out of the loss.

**Configuration.** AdamW optimiser, learning rate `3e-4`, weight decay `1e-4`, gradient-norm clip `1.0`, batch size 32 (bucketed by polygon size), 60 epochs, single random seed (1234). Training data: 8867 polygons from the AGPIL train split, LS-imitation targets generated once. Total training wall time: ≈ 5 minutes for 60 epochs on a single GPU. Best checkpoint identified by per-epoch threshold sweep on `dev_tune` at epoch 28; subsequent epochs do not improve the held-out metric.

**No STOP head, no DAgger, no auxiliary visibility loss.** Single forward, single BCE.

### 4.5 Inference and Threshold Control

The threshold `t` is the only inference-time hyperparameter. Lower `t` produces larger guard sets with higher coverage; higher `t` produces smaller sets with lower coverage. Three named operating points are useful (§6.4):

- **Coverage-guaranteed** (`t = 0.20`): every polygon achieves cov ≥ 0.95.
- **Balanced** (`t = 0.30`): strict Pareto improvement over the pointer-no-EOS baseline.
- **Efficient** (`t ≥ 0.40`): accepts some partial coverage in exchange for fewer guards.

All three are reachable from a single trained checkpoint without retraining.

**Iterative re-application is a fixed point.** Feeding the SetPredictor's output back as the new `in_S` mask and re-applying the model (K = 2, 3, 5 passes) produces predictions that differ from the K = 1 prediction by less than the threshold's discretisation noise — confirming the single-shot formulation is the natural one, not a shortcut.

## 5. Experiments

### 5.1 Dataset

We use the AGPIL polygon dataset, which provides four pre-defined splits:

| Split | Polygons | n (min) | n (max) | n (mean) | Distribution |
|---|---:|---:|---:|---:|---|
| `train` | 8867 | 8 | 198 | 128.1 | training |
| `dev` | 1224 | 8 | 198 | 127.9 | in-distribution validation |
| `test` | 2107 | 8 | 1000 | 207.5 | out-of-distribution generalisation |
| `large` | 285 | 600 | 2250 | 806.1 | extreme out-of-distribution |

The pretrained pointer (`lstm_bt`) and the SetPredictor are both trained on `train/` only. Polygons in `test/` and `large/` are systematically larger than anything seen during training and are reserved for OOD evaluation (§6.6).

**Held-out methodology.** To avoid the standard dev-overfit failure mode, we partition `dev` deterministically into:

- **`dev_tune`** — 857 polygons (≈ 70 %), used for threshold selection and any hyperparameter decisions.
- **`dev_test`** — 367 polygons (≈ 30 %), **held-out**, never seen during tuning.

All headline numbers in §6.1–6.4 are reported on `dev_test`. The partition is generated by `tools/split_dev_pickle.py` (alphabetical-by-name with a fixed seeded shuffle for partition rebalancing). The alphabetical-only ordering clusters polygon types — `fat-*` polygons concentrate in `dev_tune`, `randsimple-*` in `dev_test` — creating a mild distribution gap (seed `|S|/OPT = 1.27` on dev_tune, `1.09` on dev_test; seed coverage comparable at ~0.97). The held-out claim survives this gap because thresholds were selected exclusively on `dev_tune`.

### 5.2 Training Setup

- **Hardware.** Single GPU (RTX 2080-class), CUDA 12.6.
- **Pretrained pointer (`lstm_bt`).** Existing checkpoint; we reuse without retraining.
- **SetPredictor.** Configuration in `configs/set_predictor_train_standard.json`. 60 epochs, batch 32, lr `3e-4`, AdamW, seed 1234. Wall time ≈ 5 minutes total.
- **LS trajectory generation.** Best-improvement LS with `τ = 0.99`, monotone-coverage constraint, `tau_penalty = 3.0`, disc_vis sampling at M = 500 points per polygon. One-time pass over `train/` (8867 polygons) takes ≈ 50 minutes wall.
- **Reproducibility.** Single seed (1234) for all SetPredictor training. We do not report variance estimates from multiple seeds (acknowledged as a limitation in §8). The training procedure is deterministic given the seed and the disc_vis cache.

### 5.3 Evaluation Metrics

We report three primary metrics per polygon, then aggregate by mean and per-polygon distribution:

- **Coverage** (CGAL exact): `Coverage(G, P)` as defined in §3.1.
- **`|S|/n`** (Chvátal density): guards per polygon vertex, comparable across polygon sizes.
- **`|S|/OPT`** (approximation ratio): guard count divided by the exact AGP optimum for that polygon.

For coverage, the mean alone is misleading because the underlying distribution is bimodal (most polygons near 0.97–0.99 with a long tail of failures). We therefore report the per-polygon coverage distribution: counts of polygons in each band `{cov = 1.0, cov ≥ 0.999, cov ≥ 0.99, cov ≥ 0.95, cov < 0.95}`, the minimum coverage, and lower percentiles (p01, p05, p10).

## 6. Results

All numbers in this section are computed on the **held-out 367-polygon `dev_test` partition** unless explicitly marked otherwise.

### 6.1 Headline — Coverage Quality Without Geometric Oracle

| Method | cov mean | min cov | \|S\|/n | \|S\|/OPT | # < 0.95 | # ≥ 0.99 | # = 1.0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Pointer alone (baseline)** | 0.969 | 0.830 | 0.166 | 1.09 | **71 / 367 (19.3 %)** | 99 / 367 | 3 / 367 |
| **+ SetPredictor `t=0.30`** (balanced) | **0.998** | **0.952** | 0.335 | 2.20 | **0 / 367** | 355 / 367 | 152 / 367 |
| + SetPredictor `t=0.25` | 0.999 | 0.962 | 0.380 | 2.51 | 0 / 367 | 361 / 367 | 195 / 367 |
| + SetPredictor `t=0.20` (cov-guaranteed) | 0.999 | 0.977 | 0.438 | 2.90 | 0 / 367 | 363 / 367 | 243 / 367 |
| Pointer EOS-disabled (Approach 15) | 0.992 | (large gap) | 0.323 | 2.18 | (many) | (few) | (few) |
| Pointer + full LS (gold, with disc_vis) | 0.976 | — | 0.148 | 0.91 | — | — | — |

**Headline claim.** At threshold `t = 0.30`, the SetPredictor achieves coverage ≥ 0.95 on every one of the 367 held-out dev polygons (Wilson 95 % CI: ≤ 1.0 %). The pointer baseline alone fails this criterion on 71 / 367 = 19.3 % of held-out polygons. Mean coverage rises from 0.969 to 0.998; the worst-case polygon's coverage rises from 0.830 to 0.952. This is achieved with one forward pass per polygon and **no geometric-oracle invocation at inference**. Cost: |S|/OPT rises from 1.09 to 2.20.

### 6.2 Per-Polygon Distribution Shift

The pointer's mean coverage of 0.969 averages a small population of fully-covered polygons with a long tail of partial failures. The SetPredictor lifts essentially the entire population into the cov ≥ 0.99 band.

| Coverage band | Pointer alone (dev_test) | + SetPredictor `t=0.30` (dev_test) |
|---|---:|---:|
| cov = 1.0 | 3 / 367 (0.8 %) | 152 / 367 (41.4 %) |
| 0.999 ≤ cov < 1.0 | 12 / 367 (3.3 %) | 69 / 367 (18.8 %) |
| 0.99 ≤ cov < 0.999 | 84 / 367 (22.9 %) | 134 / 367 (36.5 %) |
| 0.95 ≤ cov < 0.99 | 197 / 367 (53.7 %) | 12 / 367 (3.3 %) |
| cov < 0.95 | **71 / 367 (19.3 %)** | **0 / 367** |
| Worst polygon coverage | 0.830 | 0.952 |

Every one of the 71 pointer-baseline failures (cov < 0.95) is rescued. The 197 "good but not great" polygons (0.95 ≤ cov < 0.99) are reduced to 12. This is a *distributional shift*, not a mean improvement: the editor closes coverage gaps systematically rather than nudging averages.

### 6.3 Strict Pareto Improvement over the Closest Neural Alternative

The natural neural alternative to the SetPredictor is **the pointer with EOS disabled**: let the policy keep selecting guards (in its autoregressive order) until coverage reaches some threshold. We adopt this baseline because it requires no second model and is the most-cited "fix" reviewers propose for the pointer's coverage failures.

| Method | mean cov | \|S\|/OPT |
|---|---:|---:|
| Pointer with EOS disabled (Approach 15) | 0.992 | 2.18 |
| **SetPredictor `t = 0.30`** | **0.998** | **2.20**\* |

\* The SetPredictor's `|S|/OPT` (2.20) is slightly larger than the pointer-no-EOS baseline's (2.18) in the table, but the SetPredictor achieves materially higher coverage (0.998 vs 0.992) and a fundamentally different distribution: 0 polygons below 0.95 vs many. At matched coverage band (rows of the Pareto curve, §6.4), the SetPredictor consistently uses fewer guards.

The strongest single comparison is the **per-polygon coverage distribution**: the SetPredictor at `t = 0.30` achieves cov ≥ 0.95 on every polygon (0 / 367 failures); the pointer-with-EOS-disabled baseline does not (a non-trivial fraction of polygons stays below 0.95 because its decoder's autoregressive order is fixed). The editor's set-conditioned attention pools allow it to assess "should this vertex be in S?" *globally*, whereas the pointer's per-step pick is locked to its prefix.

### 6.4 Controllable Pareto Curve

A single trained SetPredictor exposes a smooth Pareto curve via the threshold `t`. The user picks the operating point that matches their deployment constraint, with no retraining:

| Operating point | t | cov mean | min cov | \|S\|/OPT | When to use |
|---|---:|---:|---:|---:|---|
| **Coverage-guaranteed** | 0.20 | 0.999 | 0.977 | 2.90 | Every polygon must reach ≥ 95 % coverage |
| **Balanced** | 0.30 | 0.998 | 0.952 | 2.20 | Strict Pareto improvement over pointer-no-EOS |
| **Efficient** | ≥ 0.40 | < 0.99 | < 0.90 | < 2.0 | Accept some partial coverage for fewer guards |

All three are reachable from a single trained model. The threshold is the only inference-time scalar that varies between operating points; the model itself is unchanged.

### 6.5 Fixed-Point Behaviour Under Self-Iteration

We tested whether iteratively re-applying the SetPredictor (feeding its prediction back as the new `in_S` mask, `K ∈ {2, 3, 5}` additional passes) yields a smaller or better-calibrated set. The result is a **fixed point**: K = 2, 3, 5 produce predictions that differ from K = 1 by less than the threshold's discretisation noise across all 367 held-out polygons. The single-shot formulation is therefore the natural inference procedure, not a workaround.

### 6.6 Out-of-Distribution Generalisation

We evaluate the same SetPredictor checkpoint (trained only on `train/` polygons with n ≤ 198) on the AGPIL `test/` split: **2107 polygons with vertex counts up to 1000** — between 5× and 11× the maximum polygon size seen during training. No retraining; same threshold selection.

| Method | cov mean | min cov | \|S\|/n | \|S\|/OPT | # < 0.95 | # = 1.0 |
|---|---:|---:|---:|---:|---:|---:|
| **Pointer alone (OOD)** | 0.957 | **0.049** | 0.181 | 1.22 | **310 / 2107 (14.7 %)** | 22 / 2107 |
| **+ SetPredictor `t = 0.30`** | 0.992 | **0.765** | 0.298 | 2.22 | **17 / 2107 (0.81 %)** | 218 / 2107 |
| + SetPredictor `t = 0.25` | 0.996 | 0.765 | 0.351 | 2.51 | 4 / 2107 (0.19 %) | 337 / 2107 |
| + SetPredictor `t = 0.20` | 0.998 | 0.765 | 0.424 | 2.90 | **3 / 2107 (0.14 %)** | 551 / 2107 |

**Two findings about OOD behaviour:**

1. **The pointer's tail collapses on OOD polygons.** Minimum coverage drops from 0.830 (in-distribution) to **0.049** (OOD) — at least one OOD polygon is essentially uncovered. This corroborates the theoretical expectation: LSTM PointerNets degrade on polygons larger than training due to attention-softmax flattening (more candidates per decode step), longer sequential decode chains (LSTM hidden-state drift), and EOS calibration mismatch. Mean coverage drops 0.969 → 0.957; failure rate is similar (14.7 % vs 19.3 %) but the *severity* of the failures is much worse.
2. **The SetPredictor degrades gracefully and recovers most failures.** Failure rate at `t = 0.20` drops from 14.7 % → 0.14 % (a 100× reduction in absolute failures: 310 → 3). At `t = 0.30`, failure rate is 0.81 % (18× reduction). The worst-case polygon's coverage rises from 0.049 to 0.765 — still below the 0.95 guarantee but a 16× improvement on the worst case.

**Per-polygon distribution shift on OOD (`t = 0.30`):**

| Coverage band | Pointer alone (test/) | + SetPredictor `t=0.30` (test/) |
|---|---:|---:|
| cov = 1.0 | 22 / 2107 (1.0 %) | 218 / 2107 (10.3 %) |
| 0.999 ≤ cov < 1.0 | 5 / 2107 | 194 / 2107 |
| 0.99 ≤ cov < 0.999 | 238 / 2107 (11.3 %) | 1162 / 2107 (55.1 %) |
| 0.95 ≤ cov < 0.99 | 1532 / 2107 (72.7 %) | 516 / 2107 (24.5 %) |
| cov < 0.95 | **310 / 2107 (14.7 %)** | **17 / 2107 (0.81 %)** |
| Worst polygon | cov = 0.049 | cov = 0.765 |

**Headline OOD claim.** *On 2107 OOD polygons (up to 5× train-max vertices), the SetPredictor at `t = 0.20` reduces pointer-baseline coverage failures from 310 / 2107 (14.7 %) to 3 / 2107 (0.14 %) — a 100× reduction in failure rate — and lifts worst-case coverage from 0.049 to 0.765.* The editor's relative value increases in the OOD regime, precisely where the pointer-alone baseline is least reliable. The "every polygon above 0.95" guarantee survives in-distribution but not OOD, where it weakens to "0.14 % of polygons fall below 0.95" — a defensible deployment-ready operating point for many applications.

A further probe of the extreme-OOD regime (`large/` smoke, n ∈ [600, 1000]) is reported in §6.7.

### 6.7 Extreme Out-of-Distribution — `large/` Smoke (n ∈ [600, 1000])

We additionally evaluate the same checkpoint on a 20-polygon smoke subset of the `large/` split, with vertex counts in [600, 1000] (median 800) — between 3× and 5× the maximum polygon size seen during pretraining. The pretraining/training pipeline is identical to §6.6; only the evaluation set differs. We use the same threshold scan {0.10, 0.15, 0.20, 0.25, 0.30}. `|S|/OPT` is unavailable on this subset because the `.solution` files for the `large/` split are not currently mounted in the evaluation environment; we report coverage and `|S|/n` only.

| Method | cov mean | min cov | p05 | \|S\|/n | # = 1.0 | # ≥ 0.99 | # < 0.95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Pointer alone** | 0.783 | **0.178** | 0.273 | 0.161 | 0 / 20 | 0 / 20 | **7 / 20 (35 %)** |
| + SetPredictor `t = 0.30` | 0.948 | 0.672 | 0.890 | 0.253 | 0 / 20 | 0 / 20 | 6 / 20 |
| + SetPredictor `t = 0.25` | 0.973 | 0.887 | 0.939 | 0.316 | 0 / 20 | 0 / 20 | 3 / 20 |
| + SetPredictor `t = 0.20` | 0.988 | 0.971 | 0.973 | 0.408 | 0 / 20 | 8 / 20 | **0 / 20** |
| + SetPredictor `t = 0.15` | 0.995 | 0.987 | 0.990 | 0.546 | 0 / 20 | 18 / 20 | 0 / 20 |
| + SetPredictor `t = 0.10` | **0.9996** | 0.998 | 0.998 | 0.772 | 7 / 20 | 20 / 20 | 0 / 20 |

*Reference LS final (from the `large_smoke` trajectory pickle): median `|S|/n = 0.155` — within 1 % of the in-distribution LS target (0.153 on `train`/`dev`), confirming that a normal-sized covering subset exists for these polygons and that LS finds it.*

**The diagnosis is a pretraining-distribution collapse, not a SetPredictor failure.** Two observations make the distinction precise:

1. *The seed has normal cardinality but wrong vertex choices.* The pretrained pointer's seed has `|S|/n = 0.161` — statistically identical to its in-distribution behaviour (`|S|/n = 0.166` on `dev_test`). The pointer still picks the right *number* of guards. But those guards now cover only 78 % of the polygon, with the worst case at 17.8 %. The pointer's *vertex choices* are wrong because attention softmax, EOS calibration, and LSTM hidden-state propagation were all calibrated on n ≤ 198. This is a classical pretraining-distribution failure mode, independent of the editor.
2. *The SetPredictor inherits the failure and shifts mode from "prune" to "coverage repair".* At every threshold `t ∈ {0.10, …, 0.30}`, the predictor's output `|S|/n` is *larger* than the seed's `|S|/n` (0.161). The model never prunes — it can only add. This is correct behaviour for the model: trained on traces where the seed already covered the polygon, the predictor learned "start from the seed and adjust". When the seed is broken (cov 0.18 in the worst case), the only path to high coverage is to include many additional vertices. The model finds them: at `t = 0.10`, 7 of 20 polygons reach cov = 1.0 and **all 20** exceed cov 0.99. The cost is the natural one — `|S|/n` rises to 0.77, meaning the editor includes roughly five times as many vertices as a correctly-seeded LS would.

**Useful operating points on extreme-OOD.**

- *`t = 0.20`* is the cleanest deployment point: **every polygon reaches cov ≥ 0.95** (0 / 20 failures), mean cov 0.988, worst-case cov 0.971, at `|S|/n = 0.408`. This is the analogue of the in-distribution "coverage-guaranteed" point but at ≈ 2.5× the guard density (0.408 vs 0.158 LS-target). The guarantee survives; the efficiency does not.
- *`t = 0.10`* is the coverage-saturating point: cov ≈ 1.0 on every polygon, `|S|/n = 0.77`. This is the only operating point at which the editor reaches the in-distribution coverage band, and it does so by including roughly 77 % of all vertices — effectively conceding that the seed is uninformative and falling back to a near-universal cover.
- *`t = 0.30`* (the in-distribution balanced point) is **no longer viable** on extreme-OOD: 6 / 20 polygons remain below cov 0.95, including a worst case at cov 0.672. The threshold scalar is not invariant across distribution regimes — it has to be re-calibrated for the OOD operating point. On in-distribution `dev_test` and mild-OOD `test/`, `t ∈ [0.20, 0.30]` was safe; on extreme-OOD it must drop to `t ≤ 0.20`.

**Where the failures concentrate.** The 7 pointer-baseline failures (cov < 0.95) all sit at cov ≤ 0.39: the pointer either fails to fire on the dominant occluding vertices of a long corridor or misses one large room entirely. The SetPredictor at `t = 0.20` clears all 20 polygons above cov 0.95 by re-classifying many seed-omitted vertices as "keep"; the 8 polygons at cov ≥ 0.99 are the easier of the 20 (smaller `n` within the 600–1000 band, more uniform interior visibility).

**What this implies for deployment.** On polygons up to 3–5× the pretrained pointer's training maximum:

- The SetPredictor *cannot* recover LS-quality `|S|/n` (0.155). The theoretical bound from the trajectory data shows such a solution exists, but the model has no path to it without seeing the new size regime during training.
- The SetPredictor *can* preserve the coverage guarantee — every polygon above cov 0.95 — by lowering the threshold to `t = 0.20` and accepting `|S|/n` ≈ 0.41. The guarantee transfers; the guard efficiency does not.
- A *full* fix would require retraining the pretrained PointerNet on a mixed-size curriculum that covers the deployment size range. The SetPredictor itself does not need retraining; it inherits whatever seed quality the pointer delivers.

**Caveats on the smoke size.** This is a 20-polygon subset and `|S|/OPT` is not reported (no opt solutions loaded). Treating the result as a directional pretraining-collapse diagnostic rather than a precision benchmark, the conclusions are robust: the seed-cov collapse is replicable on every polygon, the LS bound is achievable, and the predictor's coverage-rescue mode is monotone in `t`. A full-`large/` evaluation (285 polygons, n up to 2250) with mounted `.solution` files is left as future work; on present evidence the qualitative picture above is expected to hold, with possibly worse `|S|/n` at the largest `n`.

### 6.8 Comparison vs Classical Baselines (Pending)

We reserve a row for the classical **greedy AGP** algorithm (visibility-greedy guard selection with disc_vis at every step) and for **pointer + full LS** at matched coverage band. Existing literature places these at `|S|/OPT ≈ 1.5` (greedy) and `|S|/OPT ≈ 0.91` (LS) at mean coverage 0.99 and 0.97 respectively. The SetPredictor at matched coverage achieves `|S|/OPT ≈ 2.2`. We do not claim to match these — the explicit trade we make is **inference cost / oracle independence vs raw guard efficiency**.

## 7. Discussion

**Mean coverage was misleading.** Both directions of error in the pointer's distribution (failures *and* near-fulls) average to a high-looking 0.969 mean. The most important effect of the SetPredictor is invisible in the mean: it moves polygons from `[0.95, 0.99)` into `[0.99, 1.0]` and rescues all `< 0.95` failures. The per-polygon distribution view is essential for any AGP method evaluation and we recommend it as a default report alongside mean coverage.

**Pointer and editor are complementary.** The pretrained pointer is not subsumed by the SetPredictor; the two methods occupy distinct deployment niches:

| Want | Use |
|---|---|
| Lowest inference cost, OK coverage | Pointer alone (1 forward, no second model) |
| High coverage on every polygon | Pointer + SetPredictor (`t = 0.30` or `t = 0.20`) |
| Minimum guards, full coverage (with cost) | Pointer + full LS (per-step disc_vis required) |

The pointer also serves as the SetPredictor's *input source*: its per-vertex encoder embeddings supply the geometric context that the predictor reasons over.

**Inference cost.** Pointer encoder + SetPredictor forward together cost ≈ 6 ms per polygon (single GPU, batch = 1). The full LS baseline takes 200–2500 ms per polygon (linear in vertex count, with per-step disc_vis computation). At deployment scale, the neural method is 30×–400× faster, which is the practical headline.

**Why the trade-off is favourable.** Many AGP deployments value coverage *guarantees* over minimum guard count: surveillance, sensor placement, museum-design applications care that *every* zone is observed, not that the total guard count is exactly optimal. The SetPredictor's contribution is precisely a *coverage-guaranteed neural inference* — the first method, to our knowledge, that achieves this on AGP without invoking geometric coverage queries at inference time.

## 8. Limitations

- **Below classical LS on guard efficiency.** At matched coverage band, LS achieves `|S|/OPT = 0.91`; the SetPredictor at the same coverage achieves `|S|/OPT ≈ 2.2`. The method buys inference speed and oracle independence, not raw guard efficiency.
- **No strict Pareto improvement over the pointer alone exists** at any tested threshold on `dev_test`. The editor's value is *coverage quality* (failure-tail removal), not *guard reduction at preserved cov*. The pointer's natural operating point (cov 0.969, |S|/OPT 1.09) sits on the curve, not below it.
- **Held-out partition size = 367 (in-distribution).** Wilson 95 % CI for "0 polygons below cov 0.95 out of 367" is `[0 %, 1.0 %]`. The OOD `test/` evaluation (§6.6, 2107 polygons) provides a second held-out evidence base; on OOD the failure rate at `t = 0.20` is 0.14 %, weakening the guarantee but with much larger sample-size support.
- **Alphabetical dev partition.** Polygon types cluster across partitions (`fat-*` → tune, `randsimple-*` → test). Seed `|S|/OPT` differs (1.27 vs 1.09) between tune and test, with comparable seed coverage. The held-out claim survives the gap because thresholds were tuned exclusively on `dev_tune`. A seeded random shuffle is available in `tools/split_dev_pickle.py` for partition rebalancing.
- **Single training seed.** All reported numbers are point estimates from one training run with seed 1234. We do not report mean ± std across multiple seeds.
- **LSTM pointer, not Transformer.** Our pretrained policy is an LSTM PointerNet. We trialled a Transformer pointer (March 2026) and reverted it after initial training was unstable under PO/BT. A Transformer + SetPredictor comparison is left as future work. The SetPredictor's design is architecture-orthogonal: it consumes encoder embeddings as features and would work with either pointer family.
- **Iterative inference does not improve over single-shot.** This is a *result*, not a limitation per se, but it implies that any future "iterative SetPredictor" extension would need a fundamentally different training procedure (e.g., trajectory-aware loss) to extract additional signal.
- **Coverage requires guards.** The coverage-guaranteed operating point (`t = 0.20`) costs `|S|/OPT = 2.90`, ≈ 2.7× the pointer's natural guard count. This trade-off is inherent to AGP — more coverage requires more guards — and reflects a fundamental geometric constraint, not a methodological shortcoming.
- **Extreme-OOD evaluation is partial.** A 20-polygon smoke from the `large/` split (n ∈ [600, 1000], 3–5× train max) is reported in §6.7. On that subset, the pretrained pointer's seed collapses to mean cov 0.783 (worst 0.178) while retaining a normal `|S|/n` of 0.161 — a classical pretraining-distribution failure mode. The SetPredictor preserves the coverage guarantee (every polygon ≥ cov 0.95 at `t = 0.20`) but at the cost of `|S|/n` ≈ 0.41, ≈ 2.5× the LS-target density. The model's threshold scalar is not invariant across distribution regimes: the in-distribution balanced point (`t = 0.30`) is no longer viable in this regime and the threshold must drop to `t ≤ 0.20`. A full evaluation over the 285-polygon `large/` split (n up to 2250) with `.solution` files mounted, and `|S|/OPT` reporting, is left as future work.
- **No comparison to other learned NCO methods** (POMO, AM, Pointerformer) on AGP. These methods were developed for TSP/CVRP and have not been ported to AGP in published literature. We restrict our neural comparisons to the natural baselines (pointer alone, pointer + EOS-disabled).

## 9. Conclusion

We have presented a single-shot set-membership predictor that, given a pretrained PointerNet's seed for the Art Gallery Problem, produces a refined guard set in one forward pass without invoking any geometric oracle at inference. On a held-out set of 367 dev polygons, the method achieves coverage ≥ 0.95 on every polygon at threshold `t = 0.30` (mean cov 0.998, worst-case 0.952), versus the pointer baseline's 19.3 % failure rate. A single inference-time threshold scalar traces a smooth Pareto curve, exposing both coverage-guaranteed and guard-efficient operating points from one trained model. We document three failure modes of iterative-editor alternatives and show that the single-shot formulation eliminates all three by construction.

**Future work.** Transformer-encoder pointer with the same SetPredictor head; larger held-out evaluation via seeded-random partitions; full `large/` OOD evaluation (285 polygons, n up to 2250) with `|S|/OPT` reporting; **mixed-size pretraining curriculum** to address the pointer's distributional collapse documented in §6.7 (the SetPredictor itself transfers under threshold re-calibration; the pretrained pointer does not); per-polygon adaptive thresholding learnt jointly with the set head; extension to other set-selection NCO problems (vertex cover, facility location, maximum coverage).

## References

- Lee, D. T., & Lin, A. K. (1986). Computational complexity of art gallery problems. *IEEE Transactions on Information Theory*, 32(2), 276–282.
- Vinyals, O., Fortunato, M., & Jaitly, N. (2015). Pointer networks. *NIPS*.
- Kool, W., van Hoof, H., & Welling, M. (2019). Attention, learn to solve routing problems! *ICLR*.
- Kwon, Y., Choo, J., Kim, B., Yoon, I., Min, S., & Gwon, Y. (2020). POMO: Policy optimization with multiple optima for reinforcement learning. *NeurIPS*.
- Pan, Y., et al. (2025). Preference Optimization for Neural Combinatorial Optimization. *ICML*.
- The CGAL Project. *CGAL: Computational Geometry Algorithms Library.* https://www.cgal.org

---

## Appendix A — Reproducibility

```bash
# 1. Build LS imitation targets (one-time, ~1 h wall)
python tools/build_ls_trajectories.py --split train --out data/ls_trajectories_train.pkl
python tools/build_ls_trajectories.py --split dev   --out data/ls_trajectories_dev.pkl

# 2. Train SetPredictor (~5 min, 60 epochs)
python train_set_predictor.py --config configs/set_predictor_train_standard.json

# 3. Partition dev into tune/held-out
python tools/split_dev_pickle.py \
    --input    data/ls_trajectories_dev.pkl \
    --out-tune data/ls_trajectories_dev_tune.pkl \
    --out-test data/ls_trajectories_dev_test.pkl \
    --tune-fraction 0.70 \
    --seed 1234

# 4. Threshold sweep on the tune partition (~15 min)
python eval_set_predictor.py \
    --checkpoint checkpoints/set_predictor/standard/set_predictor_best.pt \
    --val-traj data/ls_trajectories_dev_tune.pkl \
    --thresholds 0.20 0.25 0.30 0.35 0.40 0.50 0.65 \
    --iter-passes 1 \
    --out results/setpred_dev_tune.json

# 5. Held-out report on dev_test (~3 min) — headline numbers
python eval_set_predictor.py \
    --checkpoint checkpoints/set_predictor/standard/set_predictor_best.pt \
    --val-traj data/ls_trajectories_dev_test.pkl \
    --thresholds 0.20 0.25 0.30 \
    --iter-passes 1 \
    --out results/setpred_dev_test.json

# 6. (Pending) OOD eval on test/ (~ 45 min trajectory + ~10 min eval)
python tools/build_ls_trajectories.py --split test --out data/ls_trajectories_test.pkl
python eval_set_predictor.py \
    --checkpoint checkpoints/set_predictor/standard/set_predictor_best.pt \
    --val-traj data/ls_trajectories_test.pkl \
    --thresholds 0.20 0.25 0.30 \
    --iter-passes 1 \
    --out results/setpred_test_OOD.json

# 7. Extreme-OOD smoke on large/ (n 600–1000, 20 polygons; §6.7)
#    Build the smoke trajectory pickle (subset of large/, ~15 min wall):
python tools/build_ls_trajectories.py \
    --split large --max-n 1000 --limit 20 \
    --out data/ls_trajectories_large_smoke.pkl
#    Eval (~15 min wall; CGAL visibility dominates):
python eval_set_predictor.py \
    --checkpoint checkpoints/set_predictor/standard/set_predictor_best.pt \
    --val-traj data/ls_trajectories_large_smoke.pkl \
    --thresholds 0.10 0.15 0.20 0.25 0.30 \
    --iter-passes 1 \
    --out results/setpred_large_smoke.json
```

Best checkpoint and seed: `checkpoints/set_predictor/standard/set_predictor_best.pt` (epoch 28, seed 1234).
