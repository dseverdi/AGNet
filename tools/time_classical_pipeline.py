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
              unions. Timed only up to --greedy-max-n in-process: the growing-union
              step is O(n^2) in polygon-set ops and can exhaust skgeom/CGAL at large
              n (a silent native crash), which is itself evidence of the classical
              pipeline's per-instance cost. We report it where it is stable and
              note that it scales worse.

Beyond --greedy-max-n, pass --greedy-timeout-s to attempt it anyway: each such
polygon is timed in an isolated subprocess with a wall-clock timeout, so a hang
cannot run forever and a native crash cannot take down the rest of the run.
Results merge into any existing --out file by bucket, so re-running a subset of
--buckets never clobbers previously measured buckets.

Robust: per-polygon try/except, unbuffered prints, incremental JSON writes, so a
native crash in one polygon cannot lose the whole run. Run UNCONTENDED.

Output: results/classical_timing.json

Usage:
  python -u tools/time_classical_pipeline.py
  python -u tools/time_classical_pipeline.py --per-bucket 2 --greedy-max-n 250
  # attempt the large buckets too, isolated + time-boxed, without touching n=200:
  python -u tools/time_classical_pipeline.py --buckets 500,1000,2000 \\
      --greedy-max-n 2000 --greedy-timeout-s 1800 --per-bucket 1
"""

from __future__ import annotations

import argparse
import json
import pickle
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from greedy_agp import precompute_visibility_polygons, greedy_guard_selection_fast
from agp_greedy_fast import greedy_guard_selection_lazy

IMPLS = {
    "original": greedy_guard_selection_fast,
    "lazy": greedy_guard_selection_lazy,
}

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


def _worker_time_one(n: int, coverage_threshold: float, impl: str) -> None:
    """Child-process entry point: time greedy on a single polygon, print the
    result tagged so the parent can find it among any library stdout noise."""
    by_n = load_polys()
    if n not in by_n:
        print("WORKER_RESULT:" + json.dumps({"error": f"n={n} not in polygon pool"}))
        sys.exit(1)
    r = by_n[n]
    pts = np.asarray(r["points"], dtype=np.float64)
    fn = IMPLS[impl]
    t0 = time.perf_counter()
    guard_idxs, coverages = fn(pts, coverage_threshold=coverage_threshold, name=r["name"])
    dt = (time.perf_counter() - t0) * 1e3
    print("WORKER_RESULT:" + json.dumps({
        "greedy_ms": dt, "name": r["name"], "n": n,
        "n_guards": len(guard_idxs), "final_coverage": coverages[-1] if coverages else None,
    }))


def _time_greedy_subprocess(n: int, coverage_threshold: float, timeout_s: float, impl: str):
    """Time greedy on polygon n in an isolated subprocess. A hang is killed at
    timeout_s; a native crash only takes down the child. Returns (ms, status, meta)
    with ms=None on failure."""
    try:
        proc = subprocess.run(
            [sys.executable, "-u", str(Path(__file__).resolve()),
             "--worker-n", str(n), "--coverage-threshold", str(coverage_threshold),
             "--impl", impl],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None, f"timeout>{timeout_s:.0f}s", {}
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        detail = tail[-1] if tail else f"exit {proc.returncode}"
        return None, f"crashed ({detail[:160]})", {}
    for line in proc.stdout.splitlines():
        if line.startswith("WORKER_RESULT:"):
            data = json.loads(line[len("WORKER_RESULT:"):])
            if "greedy_ms" in data:
                meta = {"n_guards": data.get("n_guards"), "final_coverage": data.get("final_coverage")}
                return data["greedy_ms"], "ok", meta
            return None, f"error ({data.get('error', 'unknown')})", {}
    return None, "no result (unparseable subprocess output)", {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bucket", type=int, default=2)
    ap.add_argument("--greedy-max-n", type=int, default=250,
                    help="Time full greedy in-process only up to this n (growing-union "
                         "step is O(n^2) and native-crash-prone at large n).")
    ap.add_argument("--greedy-timeout-s", type=float, default=None,
                    help="If set, also attempt greedy above --greedy-max-n, in an "
                         "isolated subprocess killed after this many seconds.")
    ap.add_argument("--coverage-threshold", type=float, default=0.99)
    ap.add_argument("--buckets", type=str, default=None,
                     help="Comma-separated bucket list to (re)compute, e.g. "
                          "'500,1000,2000'. Default: all of " + str(BUCKETS) + ". "
                          "Buckets not listed are left untouched in --out.")
    ap.add_argument("--out", type=str, default=str(REPO / "results/classical_timing.json"))
    ap.add_argument("--impl", type=str, default="original", choices=sorted(IMPLS),
                     help="Which greedy implementation to time: 'original' "
                          "(greedy_agp.greedy_guard_selection_fast, the reference "
                          "implementation used for every guard-count/coverage number "
                          "elsewhere in the paper) or 'lazy' "
                          "(agp_greedy_fast.greedy_guard_selection_lazy, a "
                          "behavior-equivalent but much faster alternative -- see its "
                          "module docstring).")
    ap.add_argument("--worker-n", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker_n is not None:
        _worker_time_one(args.worker_n, args.coverage_threshold, args.impl)
        return

    selected_buckets = (
        [int(b) for b in args.buckets.split(",")] if args.buckets else list(BUCKETS)
    )

    out_path = Path(args.out)
    if out_path.exists():
        out = json.loads(out_path.read_text())
        rows_by_bucket = {row["bucket"]: row for row in out.get("classical", [])}
    else:
        out = {"note": "classical visibility-based pipeline vs geo-free learned pipeline; "
                       f"greedy timed in-process up to n={args.greedy_max_n}, beyond that "
                       "only with --greedy-timeout-s (see module docstring)"}
        rows_by_bucket = {}

    by_n = load_polys()

    def flush_json():
        out["classical"] = [rows_by_bucket[b] for b in sorted(rows_by_bucket)]
        pt = REPO / "results/probe_timing.json"
        if pt.exists():
            learned = {}
            for r in json.load(open(pt))["rows"]:
                learned.setdefault(r["device"], {})[r["n"]] = r["total_ms"]
            out["learned_total_ms"] = learned
        out_path.write_text(json.dumps(out, indent=2))

    for b in selected_buckets:
        cands = sorted((n for n in by_n if abs(n - b) <= TOL * b), key=lambda n: abs(n - b))
        cands = cands[:args.per_bucket]
        if not cands:
            print(f"[bucket {b}] no polygons", flush=True)
            continue
        vis_ms, greedy_ms, ns, greedy_status = [], [], [], []
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
                    fn = IMPLS[args.impl]
                    t2 = time.perf_counter()
                    guard_idxs, coverages = fn(pts, coverage_threshold=args.coverage_threshold, name=r["name"])
                    gt = (time.perf_counter() - t2) * 1e3
                    greedy_ms.append(gt); greedy_status.append("ok")
                    print(f"[bucket {b}] {r['name']} n={n}: greedy_full={gt:.0f}ms "
                          f"(impl={args.impl}, guards={len(guard_idxs)}, cov={coverages[-1]:.4f})", flush=True)
                except Exception as e:
                    greedy_status.append(f"crashed ({e})")
                    print(f"[bucket {b}] {r['name']} n={n}: greedy FAILED ({e})", flush=True)
            elif args.greedy_timeout_s is not None:
                t2 = time.perf_counter()
                gt, status, meta = _time_greedy_subprocess(n, args.coverage_threshold, args.greedy_timeout_s, args.impl)
                wall = time.perf_counter() - t2
                greedy_status.append(status)
                if gt is not None:
                    greedy_ms.append(gt)
                    print(f"[bucket {b}] {r['name']} n={n}: greedy_full={gt:.0f}ms "
                          f"(subprocess, impl={args.impl}, wall={wall:.0f}s, "
                          f"guards={meta.get('n_guards')}, cov={meta.get('final_coverage')})", flush=True)
                else:
                    print(f"[bucket {b}] {r['name']} n={n}: greedy {status} (wall={wall:.0f}s)", flush=True)
            # flush after every polygon, not just every bucket: a hang/crash on
            # the next polygon must not lose what this one already produced.
            if vis_ms:
                rows_by_bucket[b] = {
                    "bucket": b, "n_mean": float(np.mean(ns)), "n_polys": len(vis_ms),
                    "vis_ms_mean": float(np.mean(vis_ms)), "vis_ms_median": float(np.median(vis_ms)),
                    "greedy_ms_mean": float(np.mean(greedy_ms)) if greedy_ms else None,
                    "greedy_n_timed": len(greedy_ms),
                    "greedy_status": greedy_status,
                    "peak_rss_mb": _rss_mb(),
                }
                flush_json()

    print(f"\nwrote {args.out}", flush=True)
    learned = out.get("learned_total_ms", {})
    print(f"{'n':>6}{'vis (ms)':>12}{'greedy (ms)':>14}{'learned CPU':>14}{'learned CUDA':>14}", flush=True)
    for b in sorted(rows_by_bucket):
        r = rows_by_bucket[b]
        n = int(round(r["n_mean"]))
        def near(dd):
            if not dd: return float("nan")
            k = min(dd, key=lambda k: abs(k - n)); return dd[k]
        g = f"{r['greedy_ms_mean']:.0f}" if r["greedy_ms_mean"] else "n/a"
        print(f"{n:>6}{r['vis_ms_mean']:>12.0f}{g:>14}"
              f"{near(learned.get('cpu',{})):>14.1f}{near(learned.get('cuda',{})):>14.1f}", flush=True)


if __name__ == "__main__":
    main()
