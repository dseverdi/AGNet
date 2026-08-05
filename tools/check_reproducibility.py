#!/usr/bin/env python
"""check_reproducibility.py -- is policy training reproducible run-to-run?

WHY
    We have ONE archived seed-11 run that reached |S|/OPT 1.0627 (matching the
    released policy) and one rerun of the SAME seed that saturated at
    guard_ratio 1.0 and died. The suspected cause is that the good run used
    --epochs 200 while the bad one used 150, and lr_schedule=cosine sets
    T_max=epochs -- so the epoch budget rescales the entire LR curve and every
    epoch trains at a different learning rate.

    That is a THEORY. This script tests it against the archived trajectory.

WHAT "REPRODUCIBLE" MEANS HERE
    Strong: same seed + same config -> same per-epoch metrics (within fp noise).
            Training is deterministic; a good result can be re-obtained on
            demand. This is what the paper needs.
    Weak:   the rerun is also good, but follows a DIFFERENT trajectory.
            Training is stochastic beyond --seed; "good" was partly luck, and
            a single matching run does not establish reproducibility.
    Failed: the rerun is bad (saturates, or never escapes the high-guard
            region).

    Distinguishing strong from weak matters: only strong supports "we can
    reproduce the released policy", which is the claim under review.

USAGE
    python tools/check_reproducibility.py \
        --ref results/preflight_archive_prefix/seed11_training_dynamics_PRE-FIX.jsonl \
        --new checkpoints/v3/po_agp/lstm_bt_seed11/training_dynamics.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys

FIELDS = ("train_loss", "best_reward_mean", "coverage_greedy_mean",
          "guard_ratio_greedy_mean", "approx_ratio_greedy_mean")


def load(path):
    if not os.path.exists(path):
        return []
    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="archived reference run")
    ap.add_argument("--new", required=True, help="the rerun to check")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="max |delta| to call two epochs identical")
    ap.add_argument("--loose-tol", type=float, default=0.02,
                    help="max |delta| to call two epochs 'close'")
    args = ap.parse_args()

    ref = load(args.ref)
    new = load(args.new)
    if not ref:
        sys.exit(f"no reference rows in {args.ref}")
    if not new:
        print(f"[wait] rerun has no epochs yet ({args.new})")
        return 0

    n = min(len(ref), len(new))
    print(f"reference : {len(ref)} epochs   ({os.path.basename(args.ref)})")
    print(f"rerun     : {len(new)} epochs   ({os.path.basename(args.new)})")
    print(f"comparing : first {n} epochs\n")

    hdr = f"{'ep':>4}  {'gr ref':>8} {'gr new':>8} {'Δgr':>9}   {'appr ref':>9} {'appr new':>9}"
    print(hdr)
    print("-" * len(hdr))
    exact = close = 0
    first_div = None
    for i in range(n):
        r, w = ref[i], new[i]
        dg = w["guard_ratio_greedy_mean"] - r["guard_ratio_greedy_mean"]
        worst = max(abs(w[f] - r[f]) for f in FIELDS if f in r and f in w)
        if worst <= args.tol:
            exact += 1
        elif worst <= args.loose_tol:
            close += 1
        elif first_div is None:
            first_div = r["epoch"]
        if i < 30 or abs(dg) > args.loose_tol:
            print(f"{r['epoch']:>4}  {r['guard_ratio_greedy_mean']:8.4f} "
                  f"{w['guard_ratio_greedy_mean']:8.4f} {dg:+9.4f}   "
                  f"{r['approx_ratio_greedy_mean']:9.3f} "
                  f"{w['approx_ratio_greedy_mean']:9.3f}")

    print(f"\nepochs bit-identical (<= {args.tol}) : {exact}/{n}")
    print(f"epochs close        (<= {args.loose_tol}) : {close}/{n}")
    if first_div is not None:
        print(f"first real divergence at epoch      : {first_div}")

    # verdict
    print()
    if exact == n:
        print("VERDICT: STRONG reproducibility -- trajectories are identical.")
    elif exact + close == n:
        print("VERDICT: reproducible within numerical noise (no exact match, "
              "likely nondeterministic GPU kernels but same optimisation path).")
    else:
        gr_new = min(r["guard_ratio_greedy_mean"] for r in new)
        gr_ref_at_n = min(r["guard_ratio_greedy_mean"] for r in ref[:n])
        print("VERDICT: trajectories DIVERGE.")
        print(f"  lowest guard_ratio so far -- ref {gr_ref_at_n:.4f} vs "
              f"new {gr_new:.4f}")
        if gr_new > 0.95:
            print("  rerun looks SATURATED (gr ~ 1.0): the epoch-budget theory "
                  "does NOT explain the failure.")
        else:
            print("  rerun is on a different but not-yet-failed path: training "
                  "is stochastic beyond --seed, so a single good run does not "
                  "establish reproducibility.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
