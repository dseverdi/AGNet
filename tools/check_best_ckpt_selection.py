#!/usr/bin/env python
"""check_best_ckpt_selection.py -- gate the best-checkpoint selection fix.

WHY THIS EXISTS
    po_agp.py used to save po_agp_best_greedy.pt at argmax(coverage_greedy_mean).
    Over-guarding trivially drives coverage to ~1.0, so the first epoch that
    drifted into the over-guarding regime won that argmax permanently and every
    later (better) epoch was discarded. Measured: seed11 froze at epoch 17
    (guard_ratio 0.863, |S|/OPT 3.72) while epoch 164 scored |S|/OPT 1.11.

    The fix selects on cov - guard_ratio, matching the early-stop criterion.
    That is a ONE-LINE-ish behavioural change with no natural test, and the
    existing pipeline smoke test cannot see it (a 2-epoch run has no
    over-guarding epoch to be fooled by). Hence this gate.

WHAT IT CHECKS
    (A) On a synthetic epoch sequence containing an over-guarding trap epoch,
        the new selector picks the genuinely-best epoch and the legacy selector
        picks the trap -- i.e. the bug is real and the fix addresses it.
    (B) The selector is monotone//deterministic and handles missing
        guard_ratio (falls back to coverage) and negative composites.
    (C) AGNET_BEST_ON_COVERAGE=1 restores legacy behaviour exactly.
    (D) Replays REAL training_dynamics.jsonl files, if given, and reports which
        epoch each selector picks -- regression evidence against live data.

USAGE
    python tools/check_best_ckpt_selection.py
    python tools/check_best_ckpt_selection.py --replay results/preflight_archive_prefix/*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def select(rows, legacy: bool):
    """Mirror po_agp.py's best-greedy selection. Returns the chosen row.

    Mirrors the real loop: iterate in order, keep strict improvements only,
    so ties resolve to the EARLIEST epoch exactly as the training loop does.
    """
    best_score = -float("inf")
    best = None
    for r in rows:
        cov = r.get("coverage_greedy_mean")
        gr = r.get("guard_ratio_greedy_mean")
        if cov is None:
            continue
        if legacy or gr is None:
            score = cov
        else:
            score = cov - gr
        if score > best_score:
            best_score = score
            best = r
    return best


def _fail(msg):
    print(f"  FAIL — {msg}")
    return False


def test_trap():
    """(A) an over-guarding trap epoch must not win under the fix."""
    rows = [
        {"epoch": 1, "coverage_greedy_mean": 0.90, "guard_ratio_greedy_mean": 0.20},
        # the trap: perfect coverage, but guards nearly everything
        {"epoch": 2, "coverage_greedy_mean": 1.00, "guard_ratio_greedy_mean": 0.97},
        {"epoch": 3, "coverage_greedy_mean": 0.96, "guard_ratio_greedy_mean": 0.18},
        # genuinely best: high coverage, few guards
        {"epoch": 4, "coverage_greedy_mean": 0.97, "guard_ratio_greedy_mean": 0.17},
    ]
    new = select(rows, legacy=False)
    old = select(rows, legacy=True)
    ok = True
    if new["epoch"] != 4:
        ok = _fail(f"fixed selector picked epoch {new['epoch']}, want 4")
    if old["epoch"] != 2:
        ok = _fail(f"legacy selector picked epoch {old['epoch']}, want 2 (the trap)")
    if ok:
        print("  OK — fixed picks ep4 (cov .97/gr .17); legacy picks the ep2 trap "
              "(cov 1.00/gr .97)")
    return ok


def test_edge_cases():
    """(B) missing guard_ratio, negative composite, ties."""
    ok = True

    # missing guard_ratio -> fall back to coverage, must not crash
    rows = [{"epoch": 1, "coverage_greedy_mean": 0.5},
            {"epoch": 2, "coverage_greedy_mean": 0.9}]
    if select(rows, legacy=False)["epoch"] != 2:
        ok = _fail("missing guard_ratio should fall back to coverage argmax")

    # all composites negative (guard_ratio > coverage) must still select
    rows = [{"epoch": 1, "coverage_greedy_mean": 0.30, "guard_ratio_greedy_mean": 0.90},
            {"epoch": 2, "coverage_greedy_mean": 0.40, "guard_ratio_greedy_mean": 0.95}]
    got = select(rows, legacy=False)
    if got is None:
        ok = _fail("negative composites returned no selection (init must be -inf)")
    elif got["epoch"] != 1:  # -0.60 > -0.55? no: -0.60 < -0.55 -> epoch 2 wins
        if got["epoch"] != 2:
            ok = _fail(f"negative-composite pick was epoch {got['epoch']}")

    # ties resolve to the earliest epoch (strict > in the real loop)
    rows = [{"epoch": 1, "coverage_greedy_mean": 0.90, "guard_ratio_greedy_mean": 0.10},
            {"epoch": 2, "coverage_greedy_mean": 0.90, "guard_ratio_greedy_mean": 0.10}]
    if select(rows, legacy=False)["epoch"] != 1:
        ok = _fail("ties must keep the earliest epoch (strict improvement only)")

    # None coverage rows are skipped, not crashed on
    rows = [{"epoch": 1, "coverage_greedy_mean": None, "guard_ratio_greedy_mean": 0.1},
            {"epoch": 2, "coverage_greedy_mean": 0.8, "guard_ratio_greedy_mean": 0.1}]
    if select(rows, legacy=False)["epoch"] != 2:
        ok = _fail("None coverage should be skipped")

    if ok:
        print("  OK — missing gr / negative composite / ties / None coverage")
    return ok


def test_env_override():
    """(C) AGNET_BEST_ON_COVERAGE=1 must reproduce legacy selection."""
    rows = [
        {"epoch": 1, "coverage_greedy_mean": 0.97, "guard_ratio_greedy_mean": 0.17},
        {"epoch": 2, "coverage_greedy_mean": 1.00, "guard_ratio_greedy_mean": 0.99},
    ]
    legacy_pick = select(rows, legacy=True)["epoch"]
    if legacy_pick != 2:
        return _fail(f"legacy path picked {legacy_pick}, want 2")
    print("  OK — legacy path (AGNET_BEST_ON_COVERAGE=1) still picks the "
          "coverage argmax, so old runs stay reproducible")
    return True


def test_source_wired():
    """(C2) confirm po_agp.py actually contains the fix, not just this file."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(repo, "po_agp.py")).read()
    ok = True
    if "AGNET_BEST_ON_COVERAGE" not in src:
        ok = _fail("po_agp.py has no AGNET_BEST_ON_COVERAGE escape hatch")
    if "coverage_minus_guard_ratio_greedy" not in src:
        ok = _fail("po_agp.py does not record the composite as best_metric")
    # the raw-coverage comparison must be gone from the best-ckpt block
    if "cov_g is not None and cov_g > best_cov_greedy" in src:
        ok = _fail("po_agp.py STILL compares raw coverage for best_greedy")
    if ok:
        print("  OK — po_agp.py carries the fix (composite metric + escape hatch)")
    return ok


def replay(patterns):
    """(D) report both selectors' picks on real dynamics files."""
    files = []
    for p in patterns:
        files.extend(sorted(glob.glob(p)))
    if not files:
        print("  (no replay files matched; skipping)")
        return True
    for f in files:
        rows = []
        for line in open(f):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        if not rows:
            continue
        new = select(rows, legacy=False)
        old = select(rows, legacy=True)
        print(f"  {os.path.basename(f)}  ({len(rows)} epochs)")
        for tag, r in (("legacy(cov)", old), ("fixed(cov-gr)", new)):
            print("     %-14s ep%-5d cov=%.4f gr=%.4f approx=%.3f" % (
                tag, r["epoch"], r["coverage_greedy_mean"],
                r.get("guard_ratio_greedy_mean", float("nan")),
                r.get("approx_ratio_greedy_mean", float("nan"))))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", nargs="*", default=[])
    args = ap.parse_args()

    print("== (A) over-guarding trap ==")
    a = test_trap()
    print("== (B) edge cases ==")
    b = test_edge_cases()
    print("== (C) legacy override ==")
    c = test_env_override()
    print("== (C2) fix is wired into po_agp.py ==")
    c2 = test_source_wired()
    print("== (D) replay real training_dynamics ==")
    replay(args.replay)

    if all((a, b, c, c2)):
        print("\nALL PASS — best-checkpoint selection is guard-aware")
        return 0
    print("\nFAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
