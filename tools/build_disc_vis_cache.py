#!/usr/bin/env python
"""build_disc_vis_cache.py -- precompute the disc-vis cache for train+dev.

Purpose: the disc-vis matrix (exact CGAL visibility discretised to M sample
points) is the same for every seed and every phase of run_policy_seeds.sh, so
it only needs to be built ONCE, locally (where skgeom/CGAL is already
installed and verified), and shipped as a finished .pkl to a rented box. That
avoids needing CGAL to work there at all for this step, and avoids re-paying
the CPU cost per seed / per machine.

Loads the existing cache (if present) so already-computed polygons are reused,
fills in whatever train+dev polygons are still missing, and saves the result
back. Safe to re-run: it is a pure top-up, not a rebuild.

USAGE
  python tools/build_disc_vis_cache.py
  python tools/build_disc_vis_cache.py --workers 16 --n-samples 500
"""
from __future__ import annotations

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from po_agp import prepare_datasets                                    # noqa: E402
from utils import (                                                    # noqa: E402
    prewarm_disc_vis_cache, save_disc_vis_cache, load_disc_vis_cache,
    _cache_key, _DISC_VIS_CACHE,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/disc_vis_cache.pkl")
    p.add_argument("--n-samples", type=int, default=500,
                    help="Must match --disc-vis-samples used at train time "
                         "(the released recipe uses 500).")
    p.add_argument("--workers", type=int, default=0,
                    help="0 = auto (min(cpu_count, 16)).")
    args = p.parse_args()

    dp = os.getenv("DATASET_PATH")
    if not dp:
        sys.exit("DATASET_PATH must be set (see .env)")

    train_ds, val_ds = prepare_datasets(
        os.path.join(dp, "train"), os.path.join(dp, "dev"), normalize=True)
    total = len(train_ds) + len(val_ds)

    n0 = 0
    if os.path.exists(args.out):
        n0 = load_disc_vis_cache(args.out, verbose=True)
        print(f"[build] starting from {n0} cached entries")
    else:
        print(f"[build] no existing cache at {args.out}; building from scratch")

    # Cache is an in-memory OrderedDict capped by AGNET_DISC_VIS_CACHE_SIZE
    # (default 10000, LRU-evicted via popitem(last=False) -- oldest-TOUCHED
    # entry, not oldest-loaded -- as new entries are added; see utils.py).
    #
    # This bit us TWICE: the cap must exceed n0 (everything already loaded
    # from disk) PLUS the number of NEW entries this run will add, because
    # both sets coexist in memory simultaneously before the save. A cap of
    # max(n0, total)+500 is wrong -- it is smaller than n0 + (missing count)
    # whenever there is a large gap to fill, so the eviction removes
    # loaded-but-untouched (i.e. already-good) entries to make room for the
    # newly-built ones, which is the opposite of a top-up.
    train_names = {os.path.splitext(os.path.basename(f))[0]
                   for f in __import__("glob").glob(os.path.join(dp, "train", "*.pol"))}
    dev_names = {os.path.splitext(os.path.basename(f))[0]
                 for f in __import__("glob").glob(os.path.join(dp, "dev", "*.pol"))}
    from utils import _DISC_VIS_CACHE as _dvc
    have = {k.rsplit("|", 1)[0] for k in _dvc.keys()}
    to_build = len((train_names | dev_names) - have)
    os.environ["AGNET_DISC_VIS_CACHE_SIZE"] = str(n0 + to_build + 500)
    print(f"[build] cache cap set to {n0} (loaded) + {to_build} (to build) "
          f"+ 500 headroom = {n0 + to_build + 500}")

    t0 = time.time()
    for name, ds in (("train", train_ds), ("dev", val_ds)):
        print(f"[build] prewarming {name} ({len(ds)} polygons) ...")
        prewarm_disc_vis_cache(ds, n_samples=args.n_samples,
                               n_workers=(args.workers or None), verbose=True)

    save_disc_vis_cache(args.out, verbose=True)
    dt = time.time() - t0
    print(f"[build] done in {dt/60:.1f} min. {len(_DISC_VIS_CACHE)} entries "
          f"total -> {args.out}")

    # Verify: every train+dev polygon actually has a matching key.
    missing = 0
    for name_ds, ds in (("train", train_ds), ("dev", val_ds)):
        for i in range(len(ds)):
            data, _, name = ds[i]
            key = _cache_key(data.numpy(), name)
            if key not in _DISC_VIS_CACHE:
                missing += 1
    if missing:
        print(f"[build] WARNING: {missing} polygons still missing "
              f"(likely CGAL-invalid; they fall back to exact reward at "
              f"train time, which is correct but slower).")
    else:
        print("[build] verified: full train+dev coverage.")


if __name__ == "__main__":
    main()
