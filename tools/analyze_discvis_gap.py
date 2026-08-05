"""Does the disc-vis greedy's coverage-estimation gap track guard count, or
polygon size n? A diverse, family-mixed sample lets the two be told apart
(they are correlated with each other, since bigger polygons need more
guards, but not perfectly so).

For each sampled polygon: gap = disc_vis_coverage - exact_coverage (both from
greedy_agp_disc.py's greedy_guard_selection_discvis). We report Spearman rank
correlations of gap with n and with guard count, plus partial correlations
(each controlling for the other) to separate a direct effect from a merely
confounded one.

Output: results/discvis_gap_correlation.json

Usage:
  python -u tools/analyze_discvis_gap.py
"""

from __future__ import annotations

import json
import pickle
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


def rank(a: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(a)).astype(float)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def partial(r_xy: float, r_xz: float, r_yz: float) -> float:
    return (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))


def main() -> None:
    pool, seen = [], set()
    for pk in PICKLES:
        if not pk.exists():
            continue
        for r in pickle.load(open(pk, "rb"))["records"]:
            if "points" not in r or r["name"] in seen:
                continue
            seen.add(r["name"])
            pool.append(r)

    # Stratify by (family prefix, coarse n-bin) so guard count and n aren't
    # perfectly confounded by construction.
    rng = np.random.default_rng(0)
    rng.shuffle(pool)
    by_family_n: dict = {}
    for r in pool:
        fam = r["name"].split("-")[0]
        key = (fam, int(r["n"]) // 100)
        by_family_n.setdefault(key, []).append(r)
    sample = [recs[0] for recs in by_family_n.values()][:40]
    print(f"sampled {len(sample)} polygons, "
          f"n range [{min(r['n'] for r in sample)}, {max(r['n'] for r in sample)}]")

    rows = []
    for i, rec in enumerate(sample):
        pts = np.asarray(rec["points"], dtype=np.float64)
        try:
            res = greedy_disc_full(pts, rec["name"], coverage_threshold=0.99)
        except Exception as e:
            print(f"  [{i}] {rec['name']} FAILED: {e}")
            continue
        gap = res["disc_vis_coverage"] - res["exact_coverage"]
        rows.append({"name": rec["name"], "n": res["n"],
                     "n_guards": len(res["guards"]), "gap": gap})
        print(f"  [{i+1}/{len(sample)}] n={res['n']:>5} guards={len(res['guards']):>3} gap={gap:+.4f}")

    ns = np.array([r["n"] for r in rows], float)
    gs = np.array([r["n_guards"] for r in rows], float)
    gaps = np.array([r["gap"] for r in rows], float)

    r_gap_n = spearman(gaps, ns)
    r_gap_g = spearman(gaps, gs)
    r_n_g = spearman(ns, gs)
    p_gap_g_given_n = partial(r_gap_g, r_gap_n, r_n_g)
    p_gap_n_given_g = partial(r_gap_n, r_gap_g, r_n_g)

    out = {
        "note": "Does the disc-vis greedy's coverage gap track guard count or n? "
                "Spearman + partial-Spearman correlations over a family-diverse sample.",
        "n_polygons": len(rows),
        "rows": rows,
        "spearman_gap_vs_n": r_gap_n,
        "spearman_gap_vs_guards": r_gap_g,
        "spearman_n_vs_guards": r_n_g,
        "partial_gap_guards_given_n": float(p_gap_g_given_n),
        "partial_gap_n_given_guards": float(p_gap_n_given_g),
    }
    out_path = REPO / "results/discvis_gap_correlation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")
    print(f"Spearman gap-n:        {r_gap_n:.3f}")
    print(f"Spearman gap-guards:   {r_gap_g:.3f}")
    print(f"Spearman n-guards:     {r_n_g:.3f}")
    print(f"Partial gap-guards|n:  {p_gap_g_given_n:.3f}")
    print(f"Partial gap-n|guards:  {p_gap_n_given_g:.3f}")


if __name__ == "__main__":
    main()
