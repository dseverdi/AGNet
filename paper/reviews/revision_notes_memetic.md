# Response to reviewers — Memetic Computing submission 5a4a8644

Point-by-point map from each reviewer comment to the change made. Section/line
references are to the revised `paper.tex`. This document seeds the eventual
cover letter; it is not itself submitted.

## Editor

> conclusions are stated somewhat more strongly than the evidence supports

Addressed by (i) scoping every "what RL/NCO internalizes" claim to *this* policy
(abstract, C1, Discussion, Conclusion), (ii) rewriting the abstract's central
inference as "the better-supported of two readings" rather than a certainty, and
(iii) a probe-capacity ladder that separates representation content from probe
capacity (new experiment, below).

> recent references on probing in Neural Combinatorial Optimization should be considered

Added three references and a positioning paragraph in Related Work (§3):
- Zhang, Ma, Cao, Lau, *Probing Neural Combinatorial Optimization Models*, NeurIPS 2025 (Spotlight), arXiv:2510.22131.
- its extended OpenReview version (*Unveiling …*).
- Narad, Boussioux, Wagner, *Probing Neural TSP Representations for Prescriptive Decision Support*, arXiv:2602.07216, 2026.
We position our work as complementary (constrained geometric covering vs. routing;
RL preference-optimization policy; coverage-feasibility probed property; geo-free
inference constraint) and soften C1 accordingly.

## Reviewer 1 (Minor Revision)

**Typos** (all fixed): EOS token "(EOS)" (§4.1); "dev, whereas test…" semicolon
splice (§5.2); "is shown in Section 6.2" (§6.1); linear-classifier sentence
fragment rewritten (§6.2); "unseen during training" (§6.4).

**"Feasibility" vs. 0.95.** Terminology overhaul: "feasible/feasibility" is now
reserved for exact full coverage Cov = 1 (Eq. 2); the 0.95 criterion is renamed
the **0.95 coverage gate** (a near-feasibility reporting threshold), with an
explicit definitional paragraph in §5.3 and ~30 downstream phrasings updated
("gate-clearing rate", "below the gate", "coverage tail").

**Geo-free scope.** The abstract and intro now state explicitly that geo-free
constrains only inference; visibility is used freely during training (reward +
probe targets) and evaluation.

**Threshold selection.** §5.2 now states the rule: t = 0.20 is the
coverage-favoring end of the swept {0.20, 0.25, 0.30}, chosen on dev (it clears
the 0.95 gate on all 857 dev polygons); thresholds are fixed on dev before any
test/OOD use. §6.4's higher range t ∈ {0.5,…,0.8} is now justified in the main
text as the *stringent* near-boundary test (fewer vertices selected ⇒ iteration
most likely to move); the headline low-t case is a fortiori more stable
[+ direct low-t fixed-point run, if included].

**Policy runs / seeds (contradiction).** Reconciled: the released policy is a
**single** PO/BT run (seed 1234, 200 epochs, best-greedy checkpoint). The "best
of several runs" wording is corrected — what was compared during development were
different training *recipes* (REINFORCE baselines, alternative objectives), not
seed replicates. "Four seeds {1234,11,22,33}" always denotes SetPredictor *probe*
seeds; this is stated in §5.2 and Limitation 4.

**Table 1 protocol.** Caption now states the concrete differences: the REINFORCE
rows are 30-epoch runs scored *before* the train/eval-leakage deduplication that
produced the clean splits used elsewhere, on the combined dev+test pool; they
support only the objective choice (3.5–7× over-guarding, too large to be a
protocol artifact), not a cell-for-cell comparison.

## Reviewer 2

**1. Practical motivation + runtime AND memory comparison.** Rewrote the "Why
geo-free inference" intro paragraph around the amortization argument (visibility
cost paid once at training vs. per instance) with three concrete AGP-native
realms: design-space search, triage/warm-starting the exact ILP, interactive
tooling; framed as fast inner estimator + exact outer authority (so the 2–3×
over-guarding is a tolerable role, not a liability). New runtime+memory
subsection §6.5 (Table `tab:runtime`, `results/classical_timing.json`): visibility
precompute alone — the geometry the geo-free method avoids — is 0.12 s (n=200) to
1.6 s (n=2000), i.e. 19–34× the learned GPU pass (6–57 ms); full greedy ≈13.8 s at
n=200 and its O(n²) union step is native-crash-prone at large n. Memory: fixed
≈0.83 M params (~3.3 MB), independent of n, vs. per-instance visibility-polygon +
n×M matrix storage that grows with n.

**2. Novelty = combination/adaptation.** Acknowledged in Related Work and intro:
framed as an analysis-of-representations study adapting established components to
a constrained geometric covering problem, not a new architecture.

**3. Probing shows decodability, not full geometry.** Abstract/Discussion
softened to the "better-supported of two readings"; the capacity ladder + control
task make the representation-vs-probe-capacity distinction explicit.

**4. Guard-count overhead + feasibility definition.** The 2–3× overhead is
reported prominently and not disguised; the Cov=1 vs 0.95 conflation is fixed
(see R1).

**5. OOD is mostly size extrapolation / same ecosystem.** Acknowledged as a
limitation (single instance ecosystem; main OOD axis is size).

**6. Ablations compared at a common threshold.** New matched-**budget** table
(Table `tab:ablation_thresholds`) reports every condition across the whole
operating curve {0.20,0.25,0.30}, so conditions are compared at a matched guard
budget rather than a common threshold. The ranking is unchanged and sharper: at
|S|/n ≈ 0.31 the encoder-only probe nearly matches full; no-encoder cannot reach
that tail at any comparable budget; coords-only never leaves the flooding regime.

## Reviewer 3

**1. C1 overstated / NCO-probing framing.** C1 softened; NCO-probing references
and positioning added (see Editor).

**2. Practical challenge under-motivated.** New "Why geo-free inference"
paragraph (see R2.1).

**3. Probe expressiveness (central point).** New **probe-capacity ladder**
experiment (`paper/data/probe_ladder.json`, Table `tab:probe_ladder`): linear
probe → attention-free MLP (0 self-attention layers) → 1-attention-layer probe →
full 3-layer SetPredictor, all on one held-out protocol with per-vertex ROC-AUC.
Result: 0.837 (linear) → 0.909 (attention-free MLP) → 0.912 (1 layer) → 0.923
(full). The attention-free MLP already closes ~84% of the linear→full gap; the
two attention rungs add only 0.014. Direct answer to R3's question: the signal is
readily accessible through a modest nonlinear readout, NOT specifically requiring
the expressive Transformer — which strengthens "the encoder carries the geometry"
rather than undercutting it. Discussion now states this nuanced conclusion. Note: the paper never reported a SetPredictor ROC-AUC ≈ 0.98; that
figure was the reviewer's inference — the only prior AUC was the linear probe
(0.843). The ladder supplies the missing per-capacity AUCs.

**4. "What RL/NCO internalizes" scope.** All such claims scoped to the studied
PO/BT pointer policy (abstract, Discussion, Conclusion); a sentence in the
Conclusion states the single-policy limitation explicitly.

**5. Title.** Retitled to foreground the probing contribution:
*"Probing Reinforcement-Learned Representations for Geo-Free Guard Placement in
the Vertex-Guard Art Gallery Problem."*

## Reviewer 4

**Q1. NeurIPS 2025 probing paper (Zhang et al., arXiv:2510.22131).** Read the
full paper and rewrote the Related Work comparison accordingly. Their setup:
freeze AM/POMO/LEHD on TSP/CVRP/JSSP; four *linear* probing tasks (Euclidean
distance, myopia-avoidance, same-route, capacity-demand); CS-Probing localizes
which embedding *dimensions* carry each; constraints handled by decode-time
masking. Four contrasted axes: (i) problem — constrained geometric *covering*
with an un-maskable coverage constraint vs. routing (free/masked feasibility);
(ii) probed property — a *global* coverage decision (union of visibility regions)
vs. local/pairwise properties like inter-node distance; (iii) geo-free inference
constraint (no analogue in their work); (iv) probe capacity — they probe linearly
(sidesteps the confound but reads only linear signal), we use an expressive probe
+ capacity ladder. NOTE corrected from an earlier draft: their AM/POMO are *also*
RL-trained, so "RL vs. supervised" is NOT a valid distinguisher — the real
differences are problem/property/geo-free/capacity, not learning paradigm.

**Q2. Transformer trained directly from coordinates.** This experiment already
exists as the **coords-only** ablation: the identical 464K SetPredictor trained
directly on vertex coordinates against the same LS targets (encoder channels
masked to zero). Now made explicit in §6.2: it is not competitive, holding
coverage only by flooding guards (|S|/n ≥ 0.67, |S|/OPT ≈ 4.6) — so the RL
encoder, not the coordinates or the Transformer, buys the economical budget.

## New experiments added in revision

- Probe-capacity ladder (R3.3): `run_probe_ladder.sh`, `paper/scripts/eval_probe_ladder.py`, `paper/scripts/build_ladder_table.py` → `paper/data/probe_ladder.json`, `tables/tab_probe_ladder.tex`.
- Matched-budget ablation table (R2.6): `paper/scripts/build_matched_threshold_table.py` → `tables/tab_ablation_thresholds.tex`.
- Classical-pipeline runtime + memory (R2.1): `tools/time_classical_pipeline.py` → `results/classical_timing.json`, `tables/tab_runtime.tex`, §6.5.
- Low-t fixed-point sweep (R1 threshold-range question): NOT run. Answered by the
  in-text argument (§6.4) that the high-t range is the stringent near-boundary
  test and low-t is a fortiori more stable; a direct t∈{0.20,0.25,0.30} K-sweep
  could be added if a reviewer insists (eval-only, `eval_set_predictor.py
  --iter-passes-sweep`).

## Gotchas / provenance notes for future work
- The released full-probe config `configs/set_predictor_train_standard.json` is
  gone; ladder rungs use the hardcoded args in `run_probe_ladder.sh` (60 epochs,
  hidden 128, heads 8). Per-epoch dev eval (`--rollout-eval-k`) dominates runtime
  (~20 min/run with it, ~3 min without); ladder trained with it disabled and uses
  `set_predictor_final.pt`.
- `time_classical_pipeline.py` v1 crashed silently (skgeom/CGAL native segfault in
  the growing-union greedy at large n); v2 caps greedy at n≤250, times visibility
  precompute across the full grid, writes incrementally.
- SetPredictor per-vertex ROC-AUC (0.923) did NOT exist before this revision; R3's
  "≈0.98" was an inference — the only prior AUC was the linear probe's 0.843.
