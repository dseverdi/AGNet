"""Is disc-vis greedy's coverage shortfall a matter of guards, or a hard ceiling?

The gap analysed by analyze_discvis_gap.py is measured at the point where the
algorithm stops, i.e. when its own estimate crosses the 0.99 gate. That leaves
the obvious question open: would a few more guards close it? It would not.

The estimate is the fraction of M=500 fixed sample points the guard set sees, so
once every point is covered no remaining vertex has positive marginal gain and
greedy_guard_selection_discvis breaks (best_gain <= 0) reporting an estimate of
exactly 1.0. That is a CEILING: past it the algorithm has no signal at all, and
adding guards cannot help because nothing distinguishes them.

We run each polygon with coverage_threshold=1.0 to drive it to that ceiling, then
score the returned set with exact CGAL. If the exact coverage at the ceiling is
below the 0.99 gate, then at that size the method cannot reach the gate at ANY
guard count -- which is what its runtime in tab_runtime has to be read against.

Output: results/discvis_ceiling.json

Usage:
  python -u tools/analyze_discvis_ceiling.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from greedy_agp_disc import greedy_guard_selection_discvis  # noqa: E402
from utils import evaluate_polygon_visibility_numpy_wo_gt as exact_cov  # noqa: E402

GATE = 0.99
PICKLES = [
    REPO / "data/ls_trajectories_dev_test_clean.pkl",
    REPO / "data/ls_trajectories_test_clean.pkl",
    REPO / "data/ls_trajectories_large.pkl",
]
# Two per size band, spanning small polygons (where 500 points are plenty) to the
# largest we have (where they are not).
BANDS = [(0, 200), (200, 600), (600, 1100), (1100, 1800), (1800, 10**9)]
PER_BAND = 3


def main() -> None:
    recs = []
    for p in PICKLES:
        if p.exists():
            recs += pickle.load(open(p, "rb"))["records"]

    chosen, seen = [], set()
    for lo, hi in BANDS:
        band = sorted((r for r in recs if lo <= r["points"].shape[0] < hi),
                      key=lambda r: r["points"].shape[0])
        step = max(1, len(band) // PER_BAND)
        for r in band[::step][:PER_BAND]:
            if r["name"] not in seen:
                seen.add(r["name"])
                chosen.append(r)

    rows = []
    for r in chosen:
        pts, name = r["points"], r["name"]
        n = int(pts.shape[0])
        # threshold 1.0 => never stop at the gate; run until no marginal gain.
        guards, hist, info = greedy_guard_selection_discvis(
            pts, name, coverage_threshold=1.0)
        if not guards or not info.get("valid"):
            continue
        e = float(exact_cov(pts, np.array(guards, dtype=np.int64), name))
        rows.append({"name": name, "n": n, "n_guards_at_ceiling": len(guards),
                     "disc_vis_at_ceiling": float(hist[-1]),
                     "exact_at_ceiling": e, "clears_gate": bool(e >= GATE)})
        print(f"  n={n:<6} guards={len(guards):<5} est={hist[-1]:.4f} "
              f"exact={e:.4f} {'clears' if e >= GATE else 'BELOW GATE'}")

    big = [x for x in rows if x["n"] >= 800]
    out = {
        "note": ("disc-vis greedy driven to its own ceiling (no gate stop). The "
                 "estimate saturates at 1.0 once all M=500 sample points are "
                 "covered; exact_at_ceiling is the true coverage there. Where "
                 "clears_gate is false, no guard count reaches the 0.99 gate."),
        "gate": GATE, "n_polygons": len(rows), "rows": rows,
        "n_ge_800": {
            "count": len(big),
            "n_clearing_gate": sum(1 for x in big if x["clears_gate"]),
            "exact_min": min((x["exact_at_ceiling"] for x in big), default=None),
            "exact_max": max((x["exact_at_ceiling"] for x in big), default=None),
        },
    }
    dst = REPO / "results" / "discvis_ceiling.json"
    dst.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {dst.relative_to(REPO)}")
    print(f"  n>=800: {out['n_ge_800']['n_clearing_gate']}/{len(big)} clear the gate; "
          f"exact at ceiling {out['n_ge_800']['exact_min']:.4f}"
          f"--{out['n_ge_800']['exact_max']:.4f}")


if __name__ == "__main__":
    main()
