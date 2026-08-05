"""An optimized, behavior-equivalent alternative to greedy_agp.py's
greedy_guard_selection_fast, for a fair "how fast could a well-engineered
exact greedy be" timing comparison.

greedy_guard_selection_fast (greedy_agp.py) is the reference implementation
used for every guard-count/coverage number elsewhere in the paper; it is left
untouched. This module is a separate, timing-only path: same exact CGAL
visibility polygons, same exact polygon-set unions for every gain that is
actually computed, same coverage_threshold/max_guards semantics -- verified
to produce identical guard_idxs and coverages (see verify_agp_greedy_fast.py)
-- but computes far fewer full unions per round. Two changes, both provably
behavior-preserving:

1. Loop-invariant hoist: `covered`'s own area does not change while scanning
   candidates within a round, so it is computed once per round instead of
   once per candidate per round (a real, if modest, win on its own: ~4% at
   n=200 in isolation).

2. Lazy ("accelerated") greedy [Minoux, 1978]: coverage gain is monotone
   submodular, so a candidate's marginal gain can only shrink as more guards
   are added. We keep a max-heap of each candidate's most recently computed
   gain (an upper bound on its true current gain) and, each round, refresh
   only the heap's current top candidate; if its refreshed (exact) gain is
   still >= every other candidate's stale upper bound, it cannot be beaten
   and is accepted immediately -- otherwise it is re-pushed with its fresh
   value and the new top is tried. This selects exactly the same candidate
   greedy_guard_selection_fast's full scan would (same ties: Python's set
   iteration order for small non-negative ints is numeric in practice,
   matching the heap's (gain, idx) tie-break), while a full union is only
   ever computed for candidates actually being refreshed -- in later rounds,
   once most candidates' gains have collapsed near zero, this is far fewer
   than the full remaining candidate set.

We deliberately do NOT parallelize the candidate scan: greedy_agp.py already
caps its visibility-polygon threading at 2 workers "for safety", i.e. the
skgeom/CGAL bindings are not assumed thread-safe for arbitrary concurrent
polygon-set operations. Introducing threads into the (much larger volume of)
union/area calls here risks silent corruption rather than just being slow,
so this stays single-threaded like the rest of the selection loop it
replaces.

We also deliberately do NOT switch the per-candidate gain computation to a
difference/intersection-based shortcut (area(vis_poly \\ covered), valid by
inclusion-exclusion) even though skgeom.PolygonSet exposes `difference`:
that would change which library operations run and would need its own
numerical-equivalence verification. Out of scope for this pass; the two
changes above are pure call-count reductions around identical operations.
"""

from __future__ import annotations

import heapq

import skgeom

from greedy_agp import precompute_visibility_polygons


def greedy_guard_selection_lazy(points, max_guards=None, coverage_threshold=1.0, verbose=False, name=None):
    """Same selection as greedy_guard_selection_fast, fewer redundant unions."""
    vis_polys, poly = precompute_visibility_polygons(points, name)
    N = len(points)
    if max_guards is None:
        max_guards = N
    covered = skgeom.PolygonSet()
    guard_idxs = []
    coverages = []
    poly_area = abs(float(poly.area()))

    def covered_area(pset):
        return sum(
            abs(float(p.outer_boundary().area())) - sum(abs(float(h.area())) for h in p.holes)
            for p in pset.polygons
        )

    # Seed the heap with a +inf upper bound for every candidate with a valid
    # visibility polygon, so round 1 (correctly) refreshes everyone once --
    # the savings accumulate from round 2 onward, precisely where the
    # per-union cost is highest (covered is largest/most complex).
    heap = [(-float("inf"), idx) for idx in range(N) if vis_polys[idx] is not None]
    heapq.heapify(heap)

    while heap:
        old_area = covered_area(covered)  # loop-invariant: computed once per round
        best_idx = None
        best_gain = 0.0
        best_union = None
        while heap:
            _neg_stale_gain, idx = heapq.heappop(heap)
            vis_poly = vis_polys[idx]
            candidate_union = covered.union(skgeom.PolygonSet([vis_poly]))
            true_gain = covered_area(candidate_union) - old_area
            if not heap or true_gain >= -heap[0][0]:
                # true_gain is >= every remaining candidate's (stale, hence
                # an over-estimate of their true) gain: idx cannot be beaten.
                best_idx = idx
                best_gain = true_gain
                best_union = candidate_union
                break
            heapq.heappush(heap, (-true_gain, idx))

        if best_idx is None or best_gain <= 0.0:
            break
        guard_idxs.append(best_idx)
        covered = best_union
        coverage = covered_area(covered) / poly_area if poly_area > 0 else 0.0
        coverages.append(coverage)
        if verbose:
            print(f"Added guard {best_idx}, coverage: {coverage:.4f}")
        if coverage >= coverage_threshold or len(guard_idxs) >= max_guards:
            break

    del vis_polys, poly, covered
    import gc
    gc.collect()

    return guard_idxs, coverages
