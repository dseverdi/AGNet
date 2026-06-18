# Response to Reviewers

**Manuscript:** *Learning to Place Guards by Reinforcement: A Geo-Free Neural Policy for the Vertex-Guard Art Gallery Problem*

We thank the reviewer for the careful and constructive reading, and for recognizing the geo-free
protocol, the representation–decoder decomposition, and the tail-focused evaluation. The review
recommended *revise and resubmit* with eight concrete improvements (additional ablations, an
invariance test, stricter feasibility reporting, missing hyperparameters/costs, and fuller
discussion). We have addressed every point. Two of the nine questions led to **new experiments**
(the no-seed ablation and a decode-time search test); the rest are addressed with new analyses of
existing data or with text, and the small number we leave to future work are stated explicitly and
justified.

Below we answer each *Question for Authors* in order, then list the remaining changes. Section and
table numbers refer to the revised manuscript. New numbers are quoted directly; all are reproducible
from the released code and data.

---

## Q1. Vertex ordering, normalization, and robustness to re-indexing / rotation / mirroring

**Addressed — specification added (Section 5.2) and a new empirical robustness test (Limitations,
"Invariance and vertex ordering").**

- *Ordering / normalization.* Vertices are consumed by the LSTM in the source-library file order,
  with no canonical start vertex and no enforced orientation; coordinates are min–max normalized per
  polygon to $[0,1]^2$ before the encoder. The normalization makes the pipeline invariant to
  translation and axis-aligned scaling **by construction**; it is not, in principle, invariant to
  rotation or vertex re-indexing. This is now stated in Section 5.2.
- *Robustness test (new).* We re-ran the frozen policy **and** probe on transformed copies of the
  test split, re-decoding the seed each time. Under rotation ($45^\circ/90^\circ/180^\circ$) and
  mirroring the feasibility rate is **unchanged** ($362/362$ at the $0.95$ gate, mean coverage
  $\geq 0.998$): in aggregate the learned features tolerate these transforms. A cyclic re-indexing of
  the vertices is slightly more disruptive ($359/362$ feasible, with one small polygon dropping to
  coverage $0.79$), consistent with the order-dependence a recurrent encoder introduces. We report
  this honestly and note an equivariant encoder as the structural fix (see Q8).

## Q2. No-seed ablation (with and without the encoder)

**Addressed — new experiment; the 2×2 ablation is now Table 5 (Section 6.2).**

This is the reviewer's central concern — that the per-vertex seed indicator blurs attribution between
encoder knowledge and decoder decisions. We completed the full $2\times2$ design (encoder on/off ×
seed on/off), holding the supervised target, architecture, and parameter count fixed and masking the
absent input to zeros. On the test split at $t=0.20$ (four-seed mean, polygons below $0.99$ / $|S|/\OPT$):

| inputs | enc | seed | # below 0.99 | $|S|/\OPT$ |
|---|:--:|:--:|--:|--:|
| full | ✓ | ✓ | $11.0$ | $2.56$ |
| no-seed (encoder only) | ✓ | ✗ | $17.2$ | $2.42$ |
| no-encoder (seed only) | ✗ | ✓ | $29.0$ | $3.11$ |
| coords-only (neither) | ✗ | ✗ | $0.8$ | $4.64$ |

Masking the seed alone barely moves the probe (the encoder-only probe nearly matches the full probe
on both axes); masking the encoder widens the tail; with neither input the probe stays feasible only
by **flooding guards** (selecting $72\%$ of all vertices, $|S|/\OPT=4.64$). The seed thus contributes
little, whereas the encoder is what lets the probe place few guards and still cover — it is the
representation, not the decoder's seed, that carries the placement geometry. We have also woven this
result into the "distilling the oracle" rebuttal (Section 6.4) and the Discussion, where it now
adjudicates between the two readings (encoder-learned geometry vs. decoder mis-calibration) in the
encoder's favour.

## Q3. Missing PO/BT hyperparameters ($R$, $\alpha$, $\lambda$, $\rho$) and their sensitivity

**Addressed — values stated (Sections 4.1, 5.2); sensitivity discussed (Limitations).**

The released `lstm_bt` checkpoint was trained with $R=8$ rollouts per polygon, Bradley–Terry
temperature $\alpha=0.05$, reward weights $\lambda=1.0$ and $\rho=3.0$ at the gate $\tau=0.99$, for
$200$ epochs. These are now given at the reward equation (Eq. 3), the BT loss (Eq. 6), and the
training-setup paragraph (Section 5.2). On sensitivity: we did not sweep these scalars, as that
requires re-running PO/BT training; we note (Limitations) that the design turns on the *form* of the
reward — capping coverage at $\tau$ and linearly penalizing the deficit — rather than the precise
values, and leave a sweep to future work.

## Q4. Beam search / stochastic sampling at decode time

**Addressed — new experiment; decode-time search is now Table 9 (Section 6.4).**

We tested whether better decoding of the *same* policy closes the tail, geo-free. For each test
polygon we drew $K=32$ stochastic rollouts and selected among them by length-normalized
log-likelihood (no visibility oracle). The result is statistically indistinguishable from greedy
($294$ vs. $293$ of $362$ feasible, $266$ vs. $265$ below $0.99$, identical $|S|/\OPT\approx1.09$),
because the policy's maximum-likelihood decode already **is** the greedy seed — likelihood-ranked
search never prefers a more-covering set (and occasionally prefers a less-covering one). Allowing an
*oracle* to pick the highest-coverage sample of the $32$ (classical active search, no longer
geo-free) raises feasibility only to $333/362$ and still leaves $199$ below $0.99$, far short of the
probe ($362/362$ feasible, $\approx 4$ below $0.99$). The coverage tail is therefore **not** a
decoding artifact that more search removes; closing it requires reading the encoder. We did not
implement exhaustive beam search — the decoder loop is not exposed for per-step beam expansion, and
best-of-$K$ sampling is the stronger search in practice — and say so in Limitations.

## Q5. OPT ratios conditioned on Cov=1, and stricter feasibility reporting

**Addressed — (Section 6.2 and the Table 3 footnote).**

- *Strict feasibility.* We now emphasize the $\Covf{S}=1$ requirement that defines the problem: the
  probe covers $240$ of the $362$ polygons exactly, against the policy's $3$ (and the no-encoder
  ablation's $120$) — so the gain is not an artifact of reading feasibility at the $0.95$ gate. The
  per-threshold counts ($0.95/0.99/0.999/1$) are in Table 4.
- *Conditioned ratios.* For any guard set with $\Covf{S}=1$ we have $|S|\geq\OPT$ by definition, so
  conditioned on exactly-covered polygons the ratio is necessarily $\geq 1$; restricted to its
  $240/362$ exactly-covered polygons the probe averages $|S|/\OPT\approx3.2$. This also explains the
  local-search teacher's sub-unit ratio ($0.885$): it halts at its discrete-visibility gate and
  slightly undercovers, so it averages in polygons short of full coverage — not a loose $\OPT$.

## Q6. Larger $M$ / stratified sampling for discrete visibility

**Acknowledged as future work (Limitations), with the right fix identified.**

The training-time coverage estimate uses $M=500$ Monte-Carlo samples (the per-vertex visibility
polygons are exact). We did not sweep $M$; we note that the sample average has standard error
$O(1/\sqrt{M})$ and that stratified sampling near reflex vertices — where the uncovered slivers are
hardest to detect — would reduce target noise. We did not run this sweep because it touches the
training pipeline (reward and label generation); we judged it lower-value than the attribution and
decoding experiments above and leave it to future work.

## Q7. Computational cost

**Addressed — new "Computational cost" paragraph (Section 5.3).**

Training the probe is light: $\approx5$ s/epoch, $\approx5$ min for a 60-epoch run ($\approx20$ min
for the four-seed ensemble); the frozen policy is reused, not retrained. The exact per-vertex CGAL
visibility polygons and the $M=500$ discretized matrix are computed **once per polygon and cached**
($\sim$9.9k library polygons), so they are not repeated across rollouts, edits, or seeds. At inference
the method is a single geo-free forward pass: the full policy-encode-plus-probe pipeline runs in
$\approx25$ ms at $n=1000$ and $\approx57$ ms at $n=2000$ on a GPU ($\approx89$ and $\approx264$ ms on
CPU), the probe itself contributing $\approx10$ ms at $n=2000$. Exact CGAL coverage is used only for
evaluation and is the dominant offline cost.

## Q8. Permutation-/rotation-equivariant encoder

**Discussed (Limitations) and empirically motivated (Q1).**

Our invariance test (Q1) quantifies exactly the dependence an equivariant encoder would remove: the
recurrent, order-dependent encoder is robust to rotation/mirroring in aggregate but shows a small
re-indexing sensitivity. We note that a permutation- or rotation-equivariant encoder (e.g., attention
with relative positional encoding) would remove this dependence by construction, and identify it as a
natural next step. We did not build one for this revision: it is a new architecture and training run,
and our claim is about what a *given* RL-trained encoder represents, not about the optimal
architecture.

## Q9. Transfer beyond AGP (terrains, range-limited watchtowers)

**Discussed (Conclusion).**

We added that the encoder-probing decomposition is not specific to vertex guarding: it should transfer
to other geometric set-cover problems whose feasibility oracle is costly at inference — terrain
guarding, range-limited watchtower placement — where geo-free inference stays meaningful precisely
because the oracle one would otherwise consult is expensive. Whether a frozen encoder carries the
requisite structure there is an open question our method is designed to ask. We frame this as a
direction rather than a tested claim.

---

## Additional revisions prompted by the review

- **Submodular coverage / greedy guarantees.** Related Work now makes explicit that area coverage is
  monotone submodular, that the classical marginal-coverage rule is the greedy submodular-maximization
  heuristic with its diminishing-returns guarantee (Nemhauser, Wolsey & Fisher, 1978, now cited), and
  that a geo-free learner forgoes such guarantees by construction — the price of measuring what the
  representation has learned rather than what an oracle computes.
- **Greedy + local-search pipeline.** The Discussion now notes that a greedy-then-local-search
  pipeline is the natural stronger classical reference, and that the *LS on policy seed* row of
  Table 3 (which reaches $362/362$ feasibility) already bounds what such refinement achieves on this
  data. We are explicit that the learner is not built to contest classical methods on cardinality.
- **OOD disclosure.** The full OOD evaluations are present at the same rigor as the in-distribution
  test: `ood` ($2081$ polygons, Table 7) and `ood-large` ($285$ polygons, $n$ up to $2250$, Table 8),
  plus the distributional figure. (The review noted the excerpt did not include these.)

## What we deliberately leave to future work (and why)

Consistent with the paper's scope — a *methodological* study of what an RL-trained encoder represents,
not a new state-of-the-art solver — we leave four items to future work, each stated in Limitations:
(i) a sweep of $M$ / stratified visibility sampling (Q6); (ii) a permutation/rotation-equivariant
encoder (Q8); (iii) exhaustive beam search (Q4; best-of-$K$ sampling is the stronger geo-free search
we did run); and (iv) averaging over multiple *policy* seeds — our four-seed variance is on the probe,
while the probed PO/BT encoder is a single run. None of these affects the validity of the reported
measurements; they would extend generality.

We believe the revision substantially strengthens the manuscript and addresses the review in full, and
we thank the reviewer again for feedback that directly improved the attribution and robustness claims.
