# Cover letter — resubmission

> **DRAFT for the author to review before sending.** Written for resubmission to
> Memetic Computing, which held the prior submission and whose reviews are in
> hand. If the target is a different venue, cut Section 4 (the point-by-point,
> which references reviewer numbering) and keep Sections 1–3 — the disclosure in
> Section 2 should travel with the paper either way, because a reader comparing
> against the earlier version will otherwise see changed numbers and assume the
> worst.
>
> Editorial decisions left to the author: whether to name the prior submission ID
> (recommended — it is discoverable and volunteering it reads better), and how
> much of Section 3 to keep, since it concedes ground.
>
> Resolved 2026-07-30: the policy-seed replication is now complete and in the
> manuscript (new results subsection plus a table), so the claim in Section 4 is
> supported. Every number in this letter has been checked against the data files.

Dear Professor Zhu,

We are resubmitting "Learning to Place Guards by Reinforcement: A Geo-Free Neural
Policy for the Vertex-Guard Art Gallery Problem" (previously submission
5a4a8644-bc30-4439-a722-a7947efa0d3a), which you rejected on two grounds: that
"some conclusions are stated somewhat more strongly than the evidence supports,"
and that recent references on probing in Neural Combinatorial Optimization should
be considered.

We have addressed both. The second was straightforward. The first turned out to be
more serious than the reviewers knew, and most of the work since has gone into it.

## 1. The reference gap

The probing-in-NCO literature is now discussed and positioned against, including
Zhang et al. on probing neural combinatorial optimization, alongside the
representation-probing methodology we adopt (Alain and Bengio; Hewitt and Liang;
Belinkov). Reviewer 3's central methodological objection — that an expressive
probe may compute the geometry itself rather than read it — is answered on its own
terms by a new capacity ladder (linear readout, attention-free MLP, one attention
layer, full probe), reported as held-out per-vertex ROC-AUC.

## 2. The overclaiming, and a validation-split error we found ourselves

Auditing our own checkpoint selection, we discovered that the headline probe
checkpoint had been selected by validating on the *pooled* development file, which
contains every polygon of the reporting split. The reported probe numbers were
therefore selected on the split they were reported on. The checkpoint's own saved
arguments record this, which is how we identified it. No reviewer raised it.

Every probe result has been regenerated from the final training epoch, with no
validation-based selection, and every affected table and figure rebuilt. Policy
checkpoint selection now uses a disjoint development carve and reports on the
held-out split.

**The correction cuts both ways, and we state both halves.** Guard cost improved
substantially; the coverage tail worsened:

| quantity (t = 0.20, test split, four probe seeds) | submitted | corrected |
|---|---|---|
| full probe \|S\|/OPT | 2.556 | **1.621** |
| full probe #{Cov ≥ 0.95} | 361.5/362 | **349.8/362** |
| ood \|S\|/OPT (full probe) | 2.334 | **1.949** |
| ood: no-encoder / full failure ratio | 2.6× | **1.46×** |

Consequences, all now reflected in the manuscript:

- The main claim **survives and is better evidenced**. It no longer rests on a
  fixed-threshold comparison but on a matched-guard-budget one over a wide
  threshold sweep, where the ordering full < no-seed < no-encoder < coordinates-only
  is strict in all sixteen seed-by-budget cells. The margin is quoted in the form
  that is stable across the sweep: to leave the same coverage tail, the
  no-encoder ablation must spend 1.42–1.45× the full probe's guards and
  coordinates alone 1.72–1.78×. We then **repeated this on three further policies
  trained from scratch**, where it holds in all eleven reachable
  policy-by-budget cells (1.30–1.81× across all four policies), so the result is
  a property of the method rather than of one training run.
- The out-of-distribution headline is **unchanged and exact** (85% → 99% clearing
  the gate; 85% → 99% on the larger-than-training subset alone).
- Several statements were **withdrawn or narrowed**: the "2.6×" figure above; an
  abstract inference that residual failures were decoder calibration rather than
  missing representation (objected to by two reviewers, and not supported by the
  corrected tail); "an order of magnitude" in distribution, which is four-fold;
  and the claim that iterative application of the probe changes nothing, which is
  near-stationary at the operating thresholds but contracts measurably near the
  decision boundary.

## 3. Two findings that make the paper claim less than before

We report these because they are what the corrected data shows, and because both
were previously masked.

**A third finding runs the other way, and we have narrowed a claim because of it.**
The manuscript previously said the encoder holds "the geometry that coverage
requires." Geometry is invariant to rigid motions and this representation is not,
so we now say what the measurement supports: the encoder holds guard-relevant
structure *as presented* in the orientation and vertex ordering it was trained on.
That is a weaker claim than before, and we think it is the correct one.

**The pipeline is not invariant to rotation, reflection, or vertex re-indexing.**
The submitted version reported these transforms as harmless. That measurement was
invalid: it had been run through a code path that silently supplied the probe with
an empty policy seed, and the over-guarding of the leaky checkpoint saturated the
coverage gate regardless, so the defect was invisible. Corrected, the gate-clearing
count falls from 345/362 under identity to 205 under reflection and 223 under a
180° rotation. We now report this as a genuine limitation on the generality of the
representation, and it makes the equivariant-encoder direction we suggest
load-bearing rather than speculative.

**On the most extreme out-of-distribution split, the encoder's advantage is
confined to the worst case.** It lifts worst-polygon coverage roughly seven-fold,
but the gate-clearing count no longer separates the two arms beyond the seed
spread. The submitted text quoted the single most favourable seed pairing; the
manuscript now reports four-seed means and the per-seed counts.

We also accept Reviewer 2's point 4 and Reviewer 4's related comment without
qualification: at two to three times the optimum guard count, this is not a
competitive AGP solver, and classical greedy is near-optimal on these instances.
The paper does not claim otherwise. Its contribution is the measurement
instrument — geo-free inference as a way to separate what a policy internalizes
from what an oracle supplies. The practical motivation that invited the comparison
has been reframed rather than removed: the manuscript still describes where
geo-free inference shifts the visibility cost, but now states plainly that we do
not rest the paper on that argument, that a geo-free pass returns a guard set and
not a coverage value, and that where an exact oracle is cheap a classical solver is
simply the better tool.

## 4. Point-by-point

Reviewer 1 (minor revision) asked us to reconcile the description of the policy
runs. We now disclose that several runs of the recipe were trained during
development and the released one selected on training performance, state that this
selection uses no held-out data, and report independently trained policy seeds so
the contrast can be seen across policies rather than asserted from one.

Reviewer 2's point 6 — that the ablations were compared at a common probability
threshold despite differing calibration and guard counts — proved to be the
decisive methodological point, and we have made it the backbone of the encoder
argument rather than a caveat. The corrected data demonstrates the confound
directly: the arms' cost curves cross, so at one threshold the no-encoder arm is
the most expensive condition with the shorter tail, and at another it is the
cheaper with the longer one. We retain the fixed-threshold table precisely because
it shows why the naive comparison misleads, with the confound stated in its
caption.

Reviewer 3's capacity objection is addressed by the ladder described in Section 1
above, and we have scoped every claim about what "reinforcement learning" learns
to what *this* policy learns.

Reviewer 4's request for a coordinate-trained model of comparable architecture is
answered by the coordinates-only condition: the same 464K-parameter Transformer,
same targets, same schedule, with the encoder channels masked. The manuscript now
says so explicitly. It never becomes economical.

A full record of every number that moved, with sources, is in
`paper/reviews/revision_notes_memetic.md` in the accompanying materials.

We are grateful for the reviews. They identified a real weakness, and pursuing it
led us to a concrete error of our own and to a stronger version of the central
result.

Yours sincerely,

Domagoj Ševerdija (on behalf of the authors)
