#!/usr/bin/env python
"""carve_traj_by_reference.py — split a trajectory pickle to match canonical splits.

WHY
    `build_ls_trajectories.py --split dev` reads DATASET_PATH/dev, which holds
    1224 polygons: the POOLED dev+test set. The paper's splits are carved from
    it downstream --

        ls_trajectories_dev.pkl            1224   pooled
          -> ls_trajectories_dev_tune.pkl   857   tuning / checkpoint selection
          -> ls_trajectories_dev_test.pkl   367   held out
               -> ..._dev_test_clean.pkl    362   after train-leak dedup  <-- the
                                                   paper's `test` split

    So a pipeline that trains and evaluates on the raw 1224 is mixing the tuning
    and held-out partitions, and its numbers are not comparable to the paper's.

    When a NEW policy is trained (e.g. a fresh seed), its trajectories must be
    carved the same way. Re-running the 70/30 shuffle plus the dedup would risk
    drifting from the canonical partition, so instead we filter by polygon NAME
    against the existing canonical pickles. The polygon set is identical (same
    dev directory), only the trajectories differ, so name-matching reproduces
    the canonical splits exactly.

USAGE
    python tools/carve_traj_by_reference.py \
        --input  data/ls_trajectories_dev_pseed11.pkl \
        --ref-tune data/ls_trajectories_dev_tune.pkl \
        --ref-test data/ls_trajectories_dev_test_clean.pkl \
        --out-tune data/ls_trajectories_dev_tune_pseed11.pkl \
        --out-test data/ls_trajectories_dev_test_pseed11.pkl
"""
from __future__ import annotations

import argparse
import pickle
import sys


def load(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def names_of(d: dict) -> set:
    return {r.get("name") for r in d["records"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="pooled per-seed dev trajectories")
    p.add_argument("--ref-tune", default="data/ls_trajectories_dev_tune.pkl")
    p.add_argument("--ref-test", default="data/ls_trajectories_dev_test_clean.pkl")
    p.add_argument("--out-tune", required=True)
    p.add_argument("--out-test", required=True)
    p.add_argument("--no-strict", dest="strict", action="store_false", default=True,
                   help="tolerate reference polygons missing from --input "
                        "(only for smoke tests built with --n-samples)")
    args = p.parse_args()

    src = load(args.input)
    want_tune = names_of(load(args.ref_tune))
    want_test = names_of(load(args.ref_test))

    overlap = want_tune & want_test
    if overlap:
        sys.exit(f"ERROR: reference tune/test overlap on {len(overlap)} polygons "
                 f"-- refusing to produce a leaking split")

    by_name = {r.get("name"): r for r in src["records"]}
    got_tune = [by_name[n] for n in by_name.keys() & want_tune]
    got_test = [by_name[n] for n in by_name.keys() & want_test]

    missing_tune = want_tune - by_name.keys()
    missing_test = want_test - by_name.keys()
    if (missing_tune or missing_test) and args.strict:
        sys.exit(f"ERROR: {len(missing_tune)} tune and {len(missing_test)} test "
                 f"reference polygons absent from {args.input}. The input was "
                 f"probably built from a different split or a partial run.")

    for out, recs, tag, ref in ((args.out_tune, got_tune, "tune", want_tune),
                                (args.out_test, got_test, "test", want_test)):
        summary = dict(src.get("summary", {}))
        summary.update({
            "n_polygons": len(recs),
            "partition": tag,
            "carved_from": args.input,
            "reference": args.ref_tune if tag == "tune" else args.ref_test,
        })
        with open(out, "wb") as f:
            pickle.dump({"summary": summary, "records": recs}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        status = "OK" if len(recs) == len(ref) else f"WARN {len(recs)}/{len(ref)}"
        print(f"  {tag:5} -> {out}  ({len(recs)} records) [{status}]")


if __name__ == "__main__":
    main()
