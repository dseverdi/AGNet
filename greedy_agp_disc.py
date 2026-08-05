"""Greedy guard selection scored by discretized visibility.

Motivation: the classical greedy baseline everywhere else in this paper
(greedy_agp.py:greedy_guard_selection_fast) scores each candidate's marginal
gain with an exact skgeom PolygonSet union -- expensive, and not vectorizable
across candidates (one CGAL call per candidate, every round). That is the
right choice where greedy serves as the paper's exact, non-learned quality
anchor (Table tab:headline etc.). It is not the fastest a classical solver
could be: this module runs the identical additive-greedy algorithm scored
instead against the same (n, M) discretized visibility matrix used to build
PO/BT rewards and the LS oracle targets (see tools/build_ls_trajectories.py,
whose ADD block is the template for the scoring below) -- the same
approximation the paper already relies on for every training-time coverage
query, applied here to a from-scratch guard-set build instead of seed
refinement.

Because ranking is done under M=500-sample noise rather than exact area,
this greedy's guard count to reach a coverage target is, a priori, >= the
exact greedy's -- exact selection is the best this heuristic can do; this is
an approximation of it, not a free improvement. The final guard set is
therefore always re-scored with exact CGAL
(utils.evaluate_polygon_visibility_numpy_wo_gt) so a speed number is never
reported without the quality number beside it.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from utils import get_or_build_disc_vis, evaluate_polygon_visibility_numpy_wo_gt


def greedy_guard_selection_discvis(
    points: np.ndarray,
    name: str,
    max_guards: Optional[int] = None,
    coverage_threshold: float = 1.0,
    n_samples: int = 500,
) -> tuple[list[int], list[float], dict]:
    """From-scratch additive greedy; marginal gain scored in one vectorized
    NumPy call over all remaining candidates each round (no per-candidate
    CGAL call, unlike greedy_guard_selection_fast).

    Returns (guard_idxs, disc_cov_history, info). ``info`` carries disc_vis
    build time/validity/matrix size for timing and diagnostics.
    """
    t0 = time.perf_counter()
    disc = get_or_build_disc_vis(points, name, n_samples=n_samples)
    disc_vis_s = time.perf_counter() - t0

    n = int(points.shape[0])
    info: dict = {"disc_vis_s": disc_vis_s, "valid": bool(disc.get("valid")), "n": n}
    if not disc.get("valid"):
        return [], [], info

    vis = disc["vis_matrix"]          # (n, M) bool
    M = int(disc["n_samples"])
    info["n_samples"] = M
    info["matrix_bytes"] = int(vis.nbytes)

    if max_guards is None:
        max_guards = n

    covered = np.zeros(M, dtype=bool)
    available = np.ones(n, dtype=bool)
    guard_idxs: list[int] = []
    cov_history: list[float] = []
    cur_cov = 0.0

    while len(guard_idxs) < max_guards and available.any():
        cand_idx = np.nonzero(available)[0]
        new_covered = covered[None, :] | vis[cand_idx]           # (|cand|, M)
        new_cov = new_covered.sum(axis=1, dtype=np.float64) / M
        best_local = int(np.argmax(new_cov))
        best_gain = float(new_cov[best_local] - cur_cov)
        if best_gain <= 0.0:
            break
        best_idx = int(cand_idx[best_local])
        guard_idxs.append(best_idx)
        available[best_idx] = False
        covered = new_covered[best_local]
        cur_cov = float(new_cov[best_local])
        cov_history.append(cur_cov)
        if cur_cov >= coverage_threshold:
            break

    info["selection_rounds"] = len(cov_history)
    return guard_idxs, cov_history, info


def greedy_disc_full(points: np.ndarray, name: str,
                      coverage_threshold: float = 0.99,
                      n_samples: int = 500) -> dict:
    """End-to-end timed run: disc_vis build + vectorized selection, then a
    post-hoc EXACT coverage check of the final guard set -- the same
    train-cheap/evaluate-exact pattern the paper already uses for its probe.
    """
    t0 = time.perf_counter()
    guards, cov_hist, info = greedy_guard_selection_discvis(
        points, name, coverage_threshold=coverage_threshold, n_samples=n_samples,
    )
    total_s = time.perf_counter() - t0
    selection_s = total_s - info["disc_vis_s"]

    disc_cov = cov_hist[-1] if cov_hist else 0.0
    exact_cov = (evaluate_polygon_visibility_numpy_wo_gt(
        points, np.array(guards, dtype=np.int64), name,
    ) if guards else 0.0)

    return {
        "name": name,
        "n": int(points.shape[0]),
        "n_guards": len(guards),
        "disc_vis_s": info["disc_vis_s"],
        "selection_s": selection_s,
        "total_s": total_s,
        "disc_vis_coverage": disc_cov,
        "exact_coverage": exact_cov,
        "matrix_bytes": info.get("matrix_bytes", 0),
        "valid": info["valid"],
        "guards": guards,
    }


if __name__ == "__main__":
    import argparse
    import pickle

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", default="data/ls_trajectories_dev_test_clean.pkl")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--coverage-threshold", type=float, default=0.99)
    args = ap.parse_args()

    recs = pickle.load(open(args.traj, "rb"))["records"]
    rec = recs[args.index]
    pts = np.asarray(rec["points"], dtype=np.float64)
    result = greedy_disc_full(pts, rec["name"], coverage_threshold=args.coverage_threshold)
    for k, v in result.items():
        if k != "guards":
            print(f"  {k}: {v}")
    print(f"  n_guards: {len(result['guards'])}")
