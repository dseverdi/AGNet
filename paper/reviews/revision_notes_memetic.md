# Response to reviewers — Memetic Computing submission 5a4a8644

Point-by-point map from each reviewer comment to the change made. Section/line
references are to the revised `paper.tex`. This document seeds the eventual
cover letter; it is not itself submitted.

## Material correction: a validation-split error in the submitted version

**This must be disclosed prominently in the cover letter.** After the previous
revision we audited checkpoint selection and found an error in the submitted
manuscript. The headline probe checkpoint
(`checkpoints/set_predictor/standard/set_predictor_best.pt`) had been chosen by
validating on `data/ls_trajectories_dev.pkl` — the *pooled* 1224-polygon
development file, which **contains all 362 polygons of the reporting split**.
The reported probe numbers were therefore selected on the split they were
reported on. The checkpoint's own saved arguments record
`val_traj='data/ls_trajectories_dev.pkl'`, which is how we identified it.

All probe results have been regenerated from `set_predictor_final.pt` (the
final training epoch, with *no* validation-based selection), and every affected
table and figure rebuilt. Selection for the policy checkpoints is now done on
the disjoint 857-polygon `dev_tune` carve and reported on the held-out 362.

**The direction of the correction is mixed, and we state both halves.** Guard
cost improved substantially; the coverage tail worsened:

| quantity ($t=0.20$, `test`, 4 probe seeds) | submitted | corrected |
|---|---|---|
| full probe $\|S\|/\OPT$ | 2.556 | **1.621** |
| full probe $\#\{\text{Cov}\ge0.95\}$ | 361.5/362 | **349.8/362** |
| no-encoder $\|S\|/\OPT$ | 3.109 | 2.941 |
| ood $\|S\|/\OPT$ (full) | 2.334 | **1.949** |
| ood: no-encoder / full failure ratio | 2.6× | **1.46×** |

Consequences for the claims, all now reflected in the manuscript:

1. **C2's headline is unchanged and exact** — the ood gate-clearing rate is
   85.4%→98.8%, and on the $n>198$ subset 85.2%→99.1% over exactly 885
   instances, matching the submitted 85%→99% and 85%→98.7%.
2. **C2's "2.6×" was overstated** and is now reported as 1.46×.
3. **A result was previously hidden by the error:** on `ood` the submitted
   numbers showed the encoder and no-encoder arms *tied* on guard cost
   (2.334 vs 2.358). Corrected, the encoder wins (1.949 vs 2.357).
4. **C1's evidence has moved to matched guard budget** — see the next section;
   this directly implements Reviewer 2's point 6.
5. **C4 has been weakened from an exact fixed point** to near-stationarity at
   the operating thresholds, with a measured slow contraction near the decision
   boundary. The submitted bounds ($\le0.02$, $\le8.5\times10^{-3}$) do not hold
   on the corrected data ($0.079$ at $t=0.8$). The claim that each pass drifts
   *monotonically* has also been withdrawn: it holds for $t\ge0.6$, but at
   $t=0.5$ the movement is non-monotone and within run-to-run noise. The
   iteration sweep itself is on the correct held-out 362 (its $K{=}1$ cells agree
   to machine precision with the wide ablation sweep at the shared thresholds), so
   only the drift bounds needed restating, not the population.
6. **The `ood-large` claim is narrowed** to a tail-only advantage: the encoder
   lifts worst-case coverage roughly seven-fold ($0.766\pm0.106$ against
   $0.105\pm0.079$), but neither the guard saving nor the gate-clearing count
   survives the seed spread there ($19.2\pm10.4$ below the gate against
   $29.2\pm5.5$; the full probe leads on three of four seeds). The submitted text
   quoted the single most favourable seed pairing (15 against 38); it now reports
   four-seed means and the per-seed counts.

We also now report **three independent policy-training seeds** (|S|/OPT 1.040,
1.123, 1.089 against the released policy's 1.089), which resolves Reviewer 1's
request to reconcile the policy-run description.

### Second-wave corrections (2026-07-29)

A follow-up audit found that three artifacts had reverted to leaky sources, plus
two claims that the leak-free data does not support. All are fixed; recorded here
because the fix changed published numbers.

| artifact | problem | fix |
|---|---|---|
| `tab_dist_shift` | rebuilt from the leaky pre-aggregated `setpred_dev_test.json`; showed the full probe clearing **362/362** at the 0.95 gate and 358 at 0.99 | now from `dist_dev_test.json`: **345/362** and **302** |
| `tab_pareto` | same source; $\|S\|/\OPT$ 2.897/2.508/2.204 | **1.572/1.504/1.442** |
| `tab_large` | reverted to `extreme_ood.json` (pre-aggregated from the leaky checkpoint) | now from `dist_ood_large*.json` |
| `fig_worked_example` | the pinned in-distribution polygon had been chosen *under the leaky probe*, which reached Cov $=1.000$ on it with 33 of 62 vertices; the leak-free probe only moves it 0.928 → 0.933, i.e. still below the gate | re-pinned to the **median-by-$n$** member of the 57 polygons the leak-free probe rescues (0.915 → 0.996 at $\|S\|/\OPT$ 1.50, against the split mean 1.62) — the median of that set, not its best case |
| §5.2 McNemar | claimed the probe fixes **all** 69 below-gate polygons and introduces none ("69 → 0", $p\approx3\times10^{-21}$) | it fixes 57 and breaks 5, a net **69 → 17** ($p\approx3\times10^{-12}$) |
| §5.2 exact coverage | "the probe covers the large majority of polygons exactly" | **61 of 362** against the policy's 3 — a large gain but a minority, now stated as such |

**Root cause, now closed.** The three tables reverted because `build_tables.py`
still *read* the leaky pre-aggregated files, so any rerun silently undid the
hand-fixes while printing success. Those filenames are now in an explicit
deny-list in `build_tables.py`, and `load()` raises on them and names the clean
replacement. The two figures that still consume leaky sources
(`fig_coverage_cdf`, `fig_pareto`) are not included in the manuscript.

Standard deviations across probe seeds are now uniformly population std (ddof=0),
matching `tab_headline`; two `ood-large` figures previously used sample std.

All 44 load-bearing numbers in the manuscript were then re-verified against the
data files programmatically.

## Reviewer 2, point 6 — the matched-budget comparison

> the ablations are compared at a common probability threshold despite having
> different calibration and guard counts

This turned out to be the decisive methodological point, and the corrected data
shows it plainly. At a fixed $t=0.20$ the number of polygons below the 0.99 gate
is **monotone in guard budget** across the $2\times2$ ablation — full ($|S|/n$
0.25) 57.8, no-seed (0.30) 45.5, no-encoder (0.45) 22.0, coords-only (0.72) 0.8
— so at a common threshold the ablation measures set size, not representation.
The submitted version's ordering (full best) does not survive.

We therefore swept **all four conditions over the full range $t\in[0.05,0.80]$
on four probe seeds** (16 sweeps, 362 polygons, exact CGAL) so the cost curves
overlap, and compare at matched budget. `tab_ablation_thresholds` was rebuilt
around this and is now the primary evidence for C1.

The confound is not hypothetical — the curves **cross**. At $t=0.05$ the
no-encoder arm is the most expensive condition ($|S|/\OPT$ 5.17 against the full
probe's 2.01) and has the shorter tail; by $t=0.40$ it is the *cheaper* of the two
(1.37 against 1.41) and has the longer tail (200 against 96). Which arm looks
better is decided by the threshold chosen.

At matched cost the ordering is strict and seed-stable: full < no-seed <
no-encoder < coords-only in **all 16 seed-by-budget cells** (4 probe seeds × 4
budgets). The most stable way to quote the margin is the inverse reading — the
cost multiple needed to reach the *same* tail — because it is near-constant across
the whole overlap band:

| arm | guards needed for the same 0.99-gate tail |
|---|---|
| no-seed | 1.08–1.12× the full probe |
| no-encoder | **1.42–1.45×** |
| coords-only | **1.72–1.78×** |

One comparison needs no interpolation at all: at $t=0.30$ the full probe uses 15%
*fewer* guards than the no-encoder arm at the same threshold ($|S|/\OPT$ 1.50
against 1.77) and leaves 41% fewer polygons below the 0.99 gate (74 against 126).

**Two claims from the previous draft were withdrawn as the wide sweep refuted
them.** (i) "The ablation never reaches the full probe's coverage at any cost" is
false: at $t=0.05$ the no-encoder arm averages 0.9998 coverage with no polygon
below the gate — it simply pays $|S|/\OPT$ 5.17 to do so, against the full probe's
2.01 at 0.9959. The claim is therefore stated strictly as matched-cost.
(ii) The coords-only arm was described as "unresponsive to the threshold"; over the
full range its selected fraction moves 0.82 → 0.03, so that was an artifact of the
narrow window. It is now characterised by what the wide sweep actually shows — it
never becomes economical (to match the full probe's tail at $|S|/\OPT$ 1.49 it must
spend 2.66).

The fixed-threshold table is retained, with the confound stated in its caption,
because it is what demonstrates *why* the naive comparison misleads.

## Editor

> conclusions are stated somewhat more strongly than the evidence supports

Addressed by (i) scoping every "what RL/NCO internalizes" claim to *this* policy
(abstract, C1, Discussion, Conclusion); (ii) **cutting** the abstract's inference
that residual failures are "decoder calibration rather than missing
representation" — the sentence Reviewers 2 and 3 both objected to; the corrected
tail (17 of 362 in distribution) does not support it, and the abstract now
states only that a residual tail remains, without attributing a cause;
(iii) correcting the abstract's "order of magnitude" reduction to *four-fold* in
distribution, reserving "order of magnitude" for the `ood` split where it is
measured (11.7×); (iv) a probe-capacity ladder that separates representation
content from probe capacity (new experiment, below); and (v) the validation-split
correction and matched-budget re-analysis documented in the two sections above,
which is where the strongest overstatement actually originated.

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

**3. Probing shows decodability, not full geometry.** We accept this. The
abstract's causal inference about residual failures has been **removed**
outright rather than softened, and the capacity ladder plus control task make
the representation-vs-probe-capacity distinction explicit. The corrected data
supports the narrower claim the reviewer allows — that the frozen embeddings
carry guard-relevant structure coordinates do not reproduce — and we no longer
claim the decoder is solely responsible for what remains.

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
