"""Timing + quality harness for the discretized-visibility greedy
(greedy_agp_disc.py), matched to the same size grid as
tools/time_classical_pipeline.py (exact-CGAL greedy) and results/probe_timing.json
(the learned pipeline), so all three can sit in one table/figure.

Unlike the exact greedy, this one is affordable at every bucket (selection is
vectorized NumPy, not a per-candidate CGAL call), so we sample more polygons
per bucket to get a credible per-bucket quality distribution, not one anecdote.

For every run we record both the disc_vis-ESTIMATED coverage the algorithm
stopped on and the TRUE exact-CGAL coverage of the same guard set, plus guard
count -- speed is never reported without the quality number beside it (see
greedy_agp_disc.py's module docstring for why).

Output: results/discvis_greedy_timing.json

Usage:
  python -u tools/time_discvis_greedy.py
  python -u tools/time_discvis_greedy.py --per-bucket 8
"""

from __future__ import annotations

import argparse
import json
import pickle
import resource
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from greedy_agp_disc import greedy_disc_full

PICKLES = [
    REPO / "data/ls_trajectories_dev_test_clean.pkl",
    REPO / "data/ls_trajectories_test_clean.pkl",
    REPO / "data/ls_trajectories_large.pkl",
]
BUCKETS = [200, 500, 1000, 2000]   # matched to tools/time_classical_pipeline.py
TOL = 0.20


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_polys():
    by_n = {}
    for pk in PICKLES:
        if not pk.exists():
            continue
        for r in pickle.load(open(pk, "rb"))["records"]:
            if "points" in r:
                by_n.setdefault(int(r["n"]), r)
    return by_n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=8)
    ap.add_argument("--coverage-threshold", type=float, default=0.99)
    ap.add_argument("--out", type=str, default=str(REPO / "results/discvis_greedy_timing.json"))
    args = ap.parse_args()

    by_n = load_polys()
    rows = []

    for b in BUCKETS:
        cands = sorted((n for n in by_n if abs(n - b) <= TOL * b), key=lambda n: abs(n - b))
        cands = cands[:args.per_bucket]
        if not cands:
            print(f"[bucket {b}] no polygons", flush=True)
            continue

        per_poly = []
        for n in cands:
            rec = by_n[n]
            pts = np.asarray(rec["points"], dtype=np.float64)
            try:
                res = greedy_disc_full(pts, rec["name"], coverage_threshold=args.coverage_threshold)
            except Exception as e:
                print(f"[bucket {b}] {rec['name']} n={n}: FAILED ({e})", flush=True)
                continue
            gap = res["disc_vis_coverage"] - res["exact_coverage"]
            per_poly.append(res | {"gap": gap})
            print(f"[bucket {b}] {rec['name']} n={n}: disc_vis={res['disc_vis_s']*1000:.0f}ms "
                  f"select={res['selection_s']*1000:.1f}ms guards={len(res['guards'])} "
                  f"disc_cov={res['disc_vis_coverage']:.4f} exact_cov={res['exact_coverage']:.4f} "
                  f"gap={gap:+.4f}", flush=True)

        if not per_poly:
            continue

        def agg(key):
            vals = [r[key] for r in per_poly]
            return float(np.mean(vals)), float(np.std(vals))

        disc_vis_mean, disc_vis_std = agg("disc_vis_s")
        sel_mean, sel_std = agg("selection_s")
        guards_mean, guards_std = agg("n_guards")
        exact_cov_mean, exact_cov_std = agg("exact_coverage")
        disc_cov_mean, _ = agg("disc_vis_coverage")
        gap_mean, gap_std = agg("gap")

        rows.append({
            "bucket": b,
            "n_polys": len(per_poly),
            "n_mean": float(np.mean([r["n"] for r in per_poly])),
            "disc_vis_s_mean": disc_vis_mean, "disc_vis_s_std": disc_vis_std,
            "selection_s_mean": sel_mean, "selection_s_std": sel_std,
            "total_s_mean": disc_vis_mean + sel_mean,
            "n_guards_mean": guards_mean, "n_guards_std": guards_std,
            "disc_vis_coverage_mean": disc_cov_mean,
            "exact_coverage_mean": exact_cov_mean, "exact_coverage_std": exact_cov_std,
            "gap_mean": gap_mean, "gap_std": gap_std,
            "matrix_bytes": per_poly[0]["matrix_bytes"],
            "peak_rss_mb": _rss_mb(),
        })

    out = {
        "note": "disc_vis-scored greedy (greedy_agp_disc.py): from-scratch additive "
                "greedy scored on the same (n,500) discretized visibility matrix used "
                "for PO/BT reward and LS-target construction. disc_vis_s/selection_s "
                "are the timed pipeline; exact_coverage is a post-hoc CGAL check, not "
                "part of the timed pipeline. gap = disc_vis_coverage - exact_coverage.",
        "coverage_threshold": args.coverage_threshold,
        "buckets": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}", flush=True)
    print(f"{'n':>6}{'disc_vis(ms)':>14}{'select(ms)':>12}{'guards':>8}{'exact_cov':>11}{'gap':>9}", flush=True)
    for r in rows:
        print(f"{int(r['n_mean']):>6}{r['disc_vis_s_mean']*1000:>14.0f}"
              f"{r['selection_s_mean']*1000:>12.1f}{r['n_guards_mean']:>8.1f}"
              f"{r['exact_coverage_mean']:>11.4f}{r['gap_mean']:>9.4f}", flush=True)


if __name__ == "__main__":
    main()
