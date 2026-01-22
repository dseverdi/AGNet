"""compare_prune_methods.py

Loads standardized evaluation reports (see eval_reporting.py) and prints
side-by-side summaries for pruning baselines.

Intended usage:
- Run greedy_agp_prune.py -> produces results/greedy/greedy_prune_report.json (by default)
- Run ss_agp_prune.py     -> produces results/ss_agp_prune_*.json
- Run qlearning_prune.py  -> produces results/qlearning_prune_*.json

Then:
  python compare_prune_methods.py --reports <path1.json> <path2.json> <path3.json>

"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _fmt(x, nd=3):
    if x is None:
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _row(method: str, summary: Dict[str, Any]) -> List[str]:
    g = summary.get("guards", {})
    gr = summary.get("guard_ratio", {})
    cov = summary.get("coverage", {})
    ar = summary.get("approx_ratio", {})
    ts = summary.get("time_s", {})

    return [
        method,
        _fmt(g.get("mean"), 2),
        _fmt(g.get("stdev"), 2),
        _fmt(gr.get("mean"), 3),
        _fmt(cov.get("mean"), 3),
        _fmt(cov.get("p05"), 3),
        _fmt(cov.get("p95"), 3),
        _fmt(ar.get("mean"), 2),
        _fmt(ts.get("mean"), 3),
        _fmt(ts.get("p95"), 3),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare pruning method reports")
    ap.add_argument("--reports", nargs="+", required=True, help="Paths to standardized report JSON files")
    args = ap.parse_args()

    reports = [_load(p) for p in args.reports]

    headers = [
        "method",
        "|S| mean",
        "|S| stdev",
        "|S|/n mean",
        "cov mean",
        "cov p05",
        "cov p95",
        "|S|/opt mean",
        "time mean (s)",
        "time p95 (s)",
    ]

    rows = []
    for r in reports:
        rows.append(_row(r.get("method", "?"), r.get("summary", {})))

    # Simple console table
    widths = [max(len(str(h)), max((len(str(row[i])) for row in rows), default=0)) for i, h in enumerate(headers)]

    def print_row(cols):
        print(" | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cols)))

    print_row(headers)
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print_row(row)


if __name__ == "__main__":
    main()
