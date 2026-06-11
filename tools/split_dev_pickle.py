"""Deterministic 70/30 carve of a trajectory pickle into tune/test partitions.

For paper-clean evaluation: hyperparameter tuning (threshold sweep, "best"
checkpoint selection) is done on the 70 % tune partition; final headline
numbers are reported on the 30 % held-out test partition. Same distribution
(both drawn from the same source pickle, e.g. dev).

Deterministic: records are sorted by polygon name to give a stable order,
then shuffled with a fixed numpy seed. Same input pickle + same seed →
same partitions every run. The shuffle prevents alphabetical sorting from
clustering related polygons (e.g. all `fat-*` in tune, all `randsimple-*`
in test) which would create a distribution mismatch.

Usage:
    python tools/split_dev_pickle.py \\
        --input data/ls_trajectories_dev.pkl \\
        --out-tune data/ls_trajectories_dev_tune.pkl \\
        --out-test data/ls_trajectories_dev_test.pkl \\
        --tune-fraction 0.70 \\
        --seed 1234
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True,
                   help="Source trajectory pickle (e.g. dev).")
    p.add_argument("--out-tune", required=True,
                   help="Destination pickle for tuning partition.")
    p.add_argument("--out-test", required=True,
                   help="Destination pickle for held-out partition.")
    p.add_argument("--tune-fraction", type=float, default=0.70,
                   help="Fraction of records used for tuning (rest for test). "
                        "Default 0.70.")
    p.add_argument("--seed", type=int, default=1234,
                   help="Seed for the shuffle. Same seed → same partitions.")
    args = p.parse_args()

    import numpy as np

    with open(args.input, "rb") as f:
        d = pickle.load(f)
    records = d["records"]
    summary = d.get("summary", {})

    # Deterministic shuffle: sort by name for stable input order, then
    # shuffle with fixed seed. This balances polygon types (e.g. `fat-*`,
    # `randsimple-*`) across the partitions, avoiding the distribution
    # mismatch that alphabetical-only ordering produces.
    records_sorted = sorted(records, key=lambda r: r["name"])
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(records_sorted))
    records_shuffled = [records_sorted[i] for i in perm]

    n = len(records_shuffled)
    cut = int(round(n * args.tune_fraction))

    tune = records_shuffled[:cut]
    test = records_shuffled[cut:]

    print(f"[split] input: {args.input}  ({n} records)")
    print(f"[split] tune fraction: {args.tune_fraction:.2f}  cut={cut}")
    print(f"[split] tune: {len(tune)} records  -> {args.out_tune}")
    print(f"[split] test: {len(test)} records  -> {args.out_test}")

    def _dump(path: str, recs: list[dict], partition: str) -> None:
        meta = dict(summary)
        meta.setdefault("partition", partition)
        meta["n_records_after_split"] = len(recs)
        meta["source_pickle"] = args.input
        meta["tune_fraction"] = args.tune_fraction
        meta["split_seed"] = args.seed
        meta["split_method"] = "name-sort + seeded-shuffle"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"summary": meta, "records": recs}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[split] saved {len(recs)} records -> {path}")

    _dump(args.out_tune, tune, "tune")
    _dump(args.out_test, test, "held_out_test")

    # Sanity: report polygon size distribution per partition.
    import numpy as np
    for label, recs in [("tune", tune), ("test", test)]:
        sizes = np.array([r["n"] for r in recs])
        print(f"  {label} sizes  min={sizes.min()}  max={sizes.max()}  "
              f"mean={sizes.mean():.1f}  median={int(np.median(sizes))}")


if __name__ == "__main__":
    main()
