"""End-to-end runtime + memory comparison: geo-free learned pipeline vs. the
classical visibility-based pipeline (reviewer R2's request).

The geo-free pipeline avoids all per-instance visibility computation at inference
(policy encode + probe forward = coordinate-only; results/probe_timing.json).
The classical pipeline must, per instance, build each vertex's exact visibility
polygon (CGAL) and then greedily select over the visibility-area unions.

We time, on real polygons across a size grid matched to probe_timing.json:
  vis_ms    : precompute_visibility_polygons --- the CGAL cost the geo-free
              method avoids. Stable in time and memory (n independent polygons).
  greedy_ms : greedy_guard_selection_fast --- vis + greedy over growing exact-area
              unions. Timed only up to --greedy-max-n: the growing-union step is
              O(n^2) in polygon-set ops and can exhaust skgeom/CGAL at large n
              (a silent native crash), which is itself evidence of the classical
              pipeline's per-instance cost. We report it where it is stable and
              note that it scales worse.

Robust: per-polygon try/except, unbuffered prints, incremental JSON writes, so a
native crash in one polygon cannot lose the whole run. Run UNCONTENDED.

Output: results/classical_timing.json

Usage:
  python -u tools/time_classical_pipeline.py
  python -u tools/time_classical_pipeline.py --per-bucket 2 --greedy-max-n 250
"""

from __future__ import annotations

import argparse
import json
import pickle
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from greedy_agp import precompute_visibility_polygons, greedy_guard_selection_fast

PICKLES = [
    REPO / "data/ls_trajectories_dev_test_clean.pkl",
    REPO / "data/ls_trajectories_test_clean.pkl",
    REPO / "data/ls_trajectories_large.pkl",
]
BUCKETS = [200, 500, 1000, 2000]   # matched to results/probe_timing.json
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
                by_n.setdefault(int(r["n"]), r)   # one representative per size
    return by_n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=2)
    ap.add_argument("--greedy-max-n", type=int, default=250,
                    help="Time full greedy only up to this n (growing-union step "
                         "is O(n^2) and native-crash-prone at large n).")
    ap.add_argument("--coverage-threshold", type=float, default=0.99)
    ap.add_argument("--out", type=str, default=str(REPO / "results/classical_timing.json"))
    args = ap.parse_args()

    by_n = load_polys()
    rows = []
    out = {"note": "classical visibility-based pipeline vs geo-free learned pipeline; "
                   f"greedy timed only up to n={args.greedy_max_n} (see module docstring)",
           "classical": rows}

    def flush_json():
        # learned side for context
        pt = REPO / "results/probe_timing.json"
        if pt.exists():
            learned = {}
            for r in json.load(open(pt))["rows"]:
                learned.setdefault(r["device"], {})[r["n"]] = r["total_ms"]
            out["learned_total_ms"] = learned
        Path(args.out).write_text(json.dumps(out, indent=2))

    for b in BUCKETS:
        cands = sorted((n for n in by_n if abs(n - b) <= TOL * b), key=lambda n: abs(n - b))
        cands = cands[:args.per_bucket]
        if not cands:
            print(f"[bucket {b}] no polygons", flush=True)
            continue
        vis_ms, greedy_ms, ns = [], [], []
        for n in cands:
            r = by_n[n]
            pts = np.asarray(r["points"], dtype=np.float64)
            try:
                m0 = _rss_mb(); t0 = time.perf_counter()
                vp, _poly = precompute_visibility_polygons(pts, r["name"])
                dt = (time.perf_counter() - t0) * 1e3
                peak = _rss_mb()
                del vp, _poly
                vis_ms.append(dt); ns.append(n)
                print(f"[bucket {b}] {r['name']} n={n}: vis={dt:.0f}ms peakRSS={peak:.0f}MB", flush=True)
            except Exception as e:
                print(f"[bucket {b}] {r['name']} n={n}: vis FAILED ({e})", flush=True)
                continue
            if n <= args.greedy_max_n:
                try:
                    t2 = time.perf_counter()
                    greedy_guard_selection_fast(pts, coverage_threshold=args.coverage_threshold, name=r["name"])
                    gt = (time.perf_counter() - t2) * 1e3
                    greedy_ms.append(gt)
                    print(f"[bucket {b}] {r['name']} n={n}: greedy_full={gt:.0f}ms", flush=True)
                except Exception as e:
                    print(f"[bucket {b}] {r['name']} n={n}: greedy FAILED ({e})", flush=True)
        if vis_ms:
            rows.append({
                "bucket": b, "n_mean": float(np.mean(ns)), "n_polys": len(vis_ms),
                "vis_ms_mean": float(np.mean(vis_ms)), "vis_ms_median": float(np.median(vis_ms)),
                "greedy_ms_mean": float(np.mean(greedy_ms)) if greedy_ms else None,
                "greedy_n_timed": len(greedy_ms),
                "peak_rss_mb": _rss_mb(),
            })
            flush_json()

    flush_json()
    print(f"\nwrote {args.out}", flush=True)
    learned = out.get("learned_total_ms", {})
    print(f"{'n':>6}{'vis (ms)':>12}{'greedy (ms)':>14}{'learned CPU':>14}{'learned CUDA':>14}", flush=True)
    for r in rows:
        n = int(round(r["n_mean"]))
        def near(dd):
            if not dd: return float("nan")
            k = min(dd, key=lambda k: abs(k - n)); return dd[k]
        g = f"{r['greedy_ms_mean']:.0f}" if r["greedy_ms_mean"] else "n/a(>cap)"
        print(f"{n:>6}{r['vis_ms_mean']:>12.0f}{g:>14}"
              f"{near(learned.get('cpu',{})):>14.1f}{near(learned.get('cuda',{})):>14.1f}", flush=True)


if __name__ == "__main__":
    main()
