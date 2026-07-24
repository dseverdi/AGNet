#!/usr/bin/env python
"""plot_training_dynamics.py — per-epoch PO/BT training curve from the live log.

The released policy's per-epoch curve was not preserved, so the paper's
training-dynamics figure was reconstructed from four checkpoints (Limitation 11).
po_agp.py now appends one JSON line per epoch to
<checkpoint_dir>/training_dynamics.jsonl; this reads it into a clean per-epoch
table and (optionally) a figure, turning that limitation into a real curve.

Dedup: on --resume-from the epochs after the resume point are replayed, so the
log can contain a given epoch more than once. We keep the LAST occurrence of
each epoch (the value from the run that actually continued past it).

USAGE
    python tools/plot_training_dynamics.py checkpoints/v3/po_agp/lstm_bt_seed11
    python tools/plot_training_dynamics.py <dir1> <dir2> ... --out paper/data/training_dynamics_seeds.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def load_dedup(path: str) -> list[dict]:
    """Read the JSONL, keep the last record per epoch, return sorted by epoch."""
    by_epoch: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from a hard kill
            ep = rec.get("epoch")
            if ep is not None:
                by_epoch[int(ep)] = rec
    return [by_epoch[e] for e in sorted(by_epoch)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+",
                   help="checkpoint dir(s) containing training_dynamics.jsonl")
    p.add_argument("--out", default=None, help="write combined JSON here")
    p.add_argument("--fig", default=None, help="write a PNG/PDF figure here")
    args = p.parse_args()

    series = {}
    for d in args.dirs:
        path = os.path.join(d, "training_dynamics.jsonl")
        if not os.path.exists(path):
            print(f"  [skip] no log at {path}", file=sys.stderr)
            continue
        recs = load_dedup(path)
        label = os.path.basename(os.path.normpath(d))
        series[label] = recs
        eps = [r["epoch"] for r in recs]
        cov = [r.get("coverage_greedy_mean") for r in recs if r.get("coverage_greedy_mean") is not None]
        print(f"  {label:32} {len(recs):4} epochs "
              f"(range {min(eps)}..{max(eps)})"
              + (f"  greedy cov {cov[0]:.3f} -> {cov[-1]:.3f}" if cov else ""))

    if not series:
        sys.exit("no dynamics logs found")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(series, f, indent=1)
        print(f"  wrote {args.out}")

    if args.fig:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            sys.exit("matplotlib not available; JSON written, skip --fig")
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        for label, recs in series.items():
            e = [r["epoch"] for r in recs]
            cov = [r.get("coverage_greedy_mean") for r in recs]
            opt = [r.get("approx_ratio_greedy_mean") for r in recs]
            a1.plot(e, cov, label=label, lw=1)
            a2.plot(e, opt, label=label, lw=1)
        a1.set(xlabel="epoch", ylabel="greedy mean coverage")
        a2.set(xlabel="epoch", ylabel="greedy |S|/OPT")
        a1.legend(fontsize=7)
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.fig) or ".", exist_ok=True)
        fig.savefig(args.fig, dpi=150)
        print(f"  wrote {args.fig}")


if __name__ == "__main__":
    main()
