"""Which data files do the paper's tables and figures actually read?

Static grepping of the generators is unreliable: build_tables.py names the leaky
pre-aggregated files inside its LEAKY deny-list, so a text search reports them as
dependencies when in fact load() refuses to open them. This instead patches
builtins.open and records every path actually read, giving ground truth.

Only the generators that CONSUME data to produce a published artifact are run.
build_paper_data.py is excluded: it is the producer, needs a GPU and the dataset,
and would rewrite the very files we are auditing.

build_figures.py is invoked with only the five figures the manuscript includes;
fig_coverage_cdf and fig_pareto are deliberately left out because they read leaky
sources and are not in the paper.

Usage:  python tools/audit_paper_data_deps.py [--json OUT]
"""

from __future__ import annotations

import argparse
import builtins
import io
import json
import os
import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WATCH = ("paper/data", "results", "data")

# (script, argv) pairs. argv[0] is filled in per run.
CONSUMERS = [
    ("paper/scripts/build_tables.py", []),
    ("paper/scripts/build_matched_threshold_table.py", []),
    ("paper/scripts/build_operating_curve_table.py", []),
    ("paper/scripts/build_policy_seed_ood_table.py", []),
    ("paper/scripts/build_ablation_table.py", []),
    ("paper/scripts/build_ladder_table.py", []),
    ("paper/scripts/build_figures.py",
     ["fig_po_training", "fig_worked_example", "fig_embedding",
      "fig_distributions", "fig_mechanism"]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="results/paper_data_deps.json")
    args = ap.parse_args()

    reads: dict[str, set[str]] = {}
    current = {"who": "?"}
    real_open = builtins.open

    def watching_open(file, mode="r", *a, **kw):
        try:
            p = os.fspath(file)
        except TypeError:
            return real_open(file, mode, *a, **kw)
        if "r" in mode and "+" not in mode:
            try:
                rel = os.path.relpath(os.path.realpath(p), REPO)
            except ValueError:
                rel = p
            if rel.startswith(WATCH):
                reads.setdefault(current["who"], set()).add(rel)
        return real_open(file, mode, *a, **kw)

    # Patching builtins.open alone is NOT enough: pathlib.Path.read_text() goes
    # through io.open, which pathlib resolved at import time, so generators using
    # (DATA / name).read_text() -- build_tables, build_operating_curve_table,
    # build_policy_seed_ood_table -- would silently report zero reads. Patch the
    # Path methods too.
    import pathlib as _pl
    real_read_text = _pl.Path.read_text
    real_read_bytes = _pl.Path.read_bytes
    real_path_open = _pl.Path.open

    def _note(self):
        try:
            rel = os.path.relpath(os.path.realpath(str(self)), REPO)
        except ValueError:
            return
        if rel.startswith(WATCH):
            reads.setdefault(current["who"], set()).add(rel)

    def rt(self, *a, **kw):
        _note(self)
        return real_read_text(self, *a, **kw)

    def rb(self, *a, **kw):
        _note(self)
        return real_read_bytes(self, *a, **kw)

    def po(self, mode="r", *a, **kw):
        if "r" in mode and "+" not in mode:
            _note(self)
        return real_path_open(self, mode, *a, **kw)

    builtins.open = watching_open
    _pl.Path.read_text, _pl.Path.read_bytes, _pl.Path.open = rt, rb, po
    os.chdir(REPO)
    failures = []
    try:
        for script, extra in CONSUMERS:
            current["who"] = Path(script).name
            reads.setdefault(current["who"], set())
            saved_argv, saved_out = sys.argv[:], sys.stdout
            sys.argv = [script] + extra
            sys.stdout = io.StringIO()          # generators are chatty
            try:
                runpy.run_path(script, run_name="__main__")
            except SystemExit:
                pass
            except Exception as e:               # noqa: BLE001
                failures.append((script, f"{type(e).__name__}: {e}"))
            finally:
                sys.argv, sys.stdout = saved_argv, saved_out
    finally:
        builtins.open = real_open
        _pl.Path.read_text = real_read_text
        _pl.Path.read_bytes = real_read_bytes
        _pl.Path.open = real_path_open

    union = sorted({f for s in reads.values() for f in s})
    out = {
        "watched_prefixes": list(WATCH),
        "per_generator": {k: sorted(v) for k, v in sorted(reads.items())},
        "union": union,
        "failures": failures,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=1))

    for who, files in sorted(reads.items()):
        print(f"\n{who}  ({len(files)} reads)")
        for f in sorted(files):
            print(f"    {f}")
    if failures:
        print("\nFAILURES (dependency list is incomplete):")
        for s, e in failures:
            print(f"    {s}: {e}")
    print(f"\nunion: {len(union)} data files are load-bearing for the paper")
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
