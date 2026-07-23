#!/usr/bin/env python
"""probe_max_batch.py — largest PO/BT batch that fits this GPU.

WHY
    po_agp.py batches through BucketBatchSampler, which chunks length-sorted
    indices into buckets and only then splits a bucket by batch_size. The
    historical bucket_size=10 therefore capped the EFFECTIVE batch at 10
    regardless of --batch-size.

    That cap is a memory constraint, not an oversight. Each PO step decodes
    batch*K sequences autoregressively and retains the graph across all decode
    steps, so peak memory grows steeply with the polygon size. Measured on a
    7.57 GiB RTX 2080 SUPER at the worst-case training polygon (n=198):

        batch  8  -> 3.82 GiB      batch 12 -> 5.70 GiB
        batch 10  -> 4.61 GiB      batch 16 -> OOM

    Raising the batch only pays off on a bigger card, and the right value is
    hardware-specific -- hence this probe. Run it on the target machine and set
    both PO_BATCH and AGNET_BUCKET_SIZE from the result.

    Note the speedup is sub-linear: decoding is latency-bound, so k times fewer
    steps does not mean k times faster. Expect a useful but smaller gain.

USAGE
    python tools/probe_max_batch.py
    python tools/probe_max_batch.py --rollouts 8 --headroom 0.85
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset import Dataset, collate_fn                              # noqa: E402
from po_agp import (                                                 # noqa: E402
    create_agp_model, get_lengths_from_dataset, prepare_datasets,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", type=int, default=8, help="K, as in training")
    p.add_argument("--headroom", type=float, default=0.85,
                   help="fraction of GPU memory to allow (leaves slack for "
                        "fragmentation and the eval pass)")
    p.add_argument("--candidates", nargs="+", type=int,
                   default=[8, 10, 12, 16, 24, 32, 48, 64, 96, 128])
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible")

    dp = os.getenv("DATASET_PATH")
    if not dp:
        sys.exit("DATASET_PATH must be set (see .env)")

    tr, _ = prepare_datasets(os.path.join(dp, "train"),
                             os.path.join(dp, "dev"), normalize=True)
    L = np.array(get_lengths_from_dataset(tr))
    # Worst case is the longest bucket: BucketBatchSampler groups by length, so
    # some batch WILL consist of the largest polygons.
    worst = Dataset([tr.samples[i] for i in np.argsort(-L)[: max(args.candidates)]])
    n_max = int(L.max())

    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    name = torch.cuda.get_device_name(0)
    print(f"  GPU            : {name}  ({total:.2f} GiB)")
    print(f"  worst-case n   : {n_max}   (K={args.rollouts})")
    print(f"  usable budget  : {total * args.headroom:.2f} GiB "
          f"({args.headroom:.0%} headroom)\n")

    model = create_agp_model(128, 128, 1, 10.0, True, 1.0).to("cuda")
    best, per_sample = None, None

    for B in sorted(args.candidates):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            bd, pm, ln, _ = collate_fn([worst[i] for i in range(B)])
            bd, pm = bd.to("cuda"), pm.to("cuda")
            lt = torch.as_tensor(ln, device="cuda")
            K = args.rollouts
            idxs, lp = model(bd.repeat_interleave(K, 0),
                             padding_mask=pm.repeat_interleave(K, 0),
                             lengths=lt.repeat_interleave(K),
                             deterministic=False)
            loss = (sum(l.sum() for l in lp)
                    if isinstance(lp, (list, tuple)) else lp.sum())
            loss.backward()
            peak = torch.cuda.max_memory_allocated() / 2**30
            ok = peak <= total * args.headroom
            print(f"  batch={B:4}  peak={peak:6.2f} GiB  "
                  f"{'ok' if ok else 'over budget'}")
            if ok:
                best, per_sample = B, peak / B
            del idxs, lp, loss
            model.zero_grad(set_to_none=True)
        except torch.OutOfMemoryError:
            print(f"  batch={B:4}  OOM")
            break
        except Exception as e:  # noqa: BLE001
            print(f"  batch={B:4}  failed: {type(e).__name__}: {e}")
            break

    print()
    if best is None:
        print("  No candidate fits. Try --headroom 0.95, or a smaller K.")
        return
    steps = 8000 // best
    print(f"  RECOMMENDED: PO_BATCH={best}  AGNET_BUCKET_SIZE={best * 8}")
    print(f"    -> {steps} steps/epoch on the 8000-polygon train subset")
    print(f"    -> ~{per_sample:.2f} GiB per sample at n={n_max}; a card with G GiB")
    print(f"       fits roughly batch = {args.headroom:.2f}*G/{per_sample:.2f} "
          f"= {args.headroom / per_sample:.1f}*G")
    print("\n  Set BOTH: bucket_size must be >= batch_size or the batch is capped")
    print("  at bucket_size and the setting silently does nothing.")


if __name__ == "__main__":
    main()
