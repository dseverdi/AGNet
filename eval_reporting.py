"""eval_reporting.py

Small helpers to standardize evaluation reports across methods.

Schema goals:
- Comparable across greedy prune, ss_agp_prune (policy), and qlearning_prune.
- Includes per-instance metrics and aggregated descriptive stats:
  min/max/percentiles/mean/std.

This module is intentionally dependency-light (numpy only).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np


PCTS_DEFAULT: Sequence[float] = (0, 5, 25, 50, 75, 95, 100)


def _to_float_list(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            out.append(fv)
    return out


def summarize_numeric(values: Iterable[Any], percentiles: Sequence[float] = PCTS_DEFAULT) -> Dict[str, Optional[float]]:
    vals = _to_float_list(values)
    if not vals:
        return {
            "count": 0,
            "mean": None,
            "stdev": None,
            "min": None,
            "max": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
        }

    arr = np.asarray(vals, dtype=np.float64)
    p = np.percentile(arr, list(percentiles))
    # percentiles expected to be (0,5,25,50,75,95,100)
    out = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "stdev": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p05": float(p[list(percentiles).index(5)]) if 5 in percentiles else None,
        "p25": float(p[list(percentiles).index(25)]) if 25 in percentiles else None,
        "median": float(p[list(percentiles).index(50)]) if 50 in percentiles else float(np.median(arr)),
        "p75": float(p[list(percentiles).index(75)]) if 75 in percentiles else None,
        "p95": float(p[list(percentiles).index(95)]) if 95 in percentiles else None,
    }
    return out


def summarize_seconds(values: Iterable[Any]) -> Dict[str, Optional[float]]:
    vals = _to_float_list(values)
    base = summarize_numeric(vals)
    base["total_s"] = float(np.sum(vals)) if vals else 0.0
    return base


def make_report(
    *,
    method: str,
    per_instance: List[Dict[str, Any]],
    args: Dict[str, Any],
    dataset: Dict[str, Any],
    oracle: Dict[str, Any],
    timing: Optional[Dict[str, Any]] = None,
    schema_version: int = 1,
) -> Dict[str, Any]:
    guards = [x.get("guards") for x in per_instance]
    guard_ratio = [x.get("guard_ratio") for x in per_instance]
    coverage = [x.get("coverage") for x in per_instance]
    approx_ratio = [x.get("approx_ratio") for x in per_instance]
    time_s = [x.get("time_s") for x in per_instance]

    report = {
        "schema_version": int(schema_version),
        "method": str(method),
        "dataset": dataset,
        "oracle": oracle,
        "summary": {
            "guards": summarize_numeric(guards),
            "guard_ratio": summarize_numeric(guard_ratio),
            "coverage": summarize_numeric(coverage),
            "approx_ratio": summarize_numeric(approx_ratio),
            "time_s": summarize_seconds(time_s),
        },
        "per_instance": per_instance,
        "args": args,
    }
    if timing is not None:
        report["timing"] = timing
    return report
