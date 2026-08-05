"""Extract per-epoch PO/BT training curves from the policy-seed run logs.

WHY THIS EXISTS. The released policy (seed 1234) was trained before per-epoch
metric logging was in place, so its training curve could only ever be
reconstructed from the four surviving checkpoints (epochs 110/114/160/200) --
four late points, no early phase, on one seed. The three replication policies
(seeds 11/22/33, trained later by run_policy_seeds.sh) DID log per epoch, and
those logs survive in logs/policy_seed{11,22,33}.log. This script recovers them.

WHAT THE LOG LINES LOOK LIKE

    [epoch 164]  greedy cov=0.970 |S|/n=0.174  stoch cov=0.972 |S|/n=0.176  |S|/opt=1.11

WHAT THE METRICS ARE -- read this before using them. epoch_eval in po_agp.py
calls evaluate_po(model, small_train, args.agp_train_dir), where
small_train = train_ds.samples[:train_size] and eval_k = epoch_eval_k. With
configs/po_agp_transformer_bt.json (train_size=8000, epoch_eval_k=50) that is
the FIRST 50 POLYGONS OF THE TRAINING PREFIX, scored with disc-vis approximate
coverage (disc_vis_samples=500), greedy decode. So these curves are:

  * TRAINING data, not held out -- they say nothing about generalization;
  * APPROXIMATE coverage, which over-reads relative to exact CGAL;
  * NOT comparable cell-for-cell with any results table (all held-out + exact).

They are usable for the SHAPE of optimization (does the guard/coverage trade
converge, does a seed saturate) and nothing else. The paper's figure built on
them was cut for exactly this reason; only seed 22's saturation survives, in
sec:method-pointer prose.

The three runs stop at different epochs (164/41/98) rather than a common 200:
seed 11 early-stopped, and seeds 22/33 were cut short. Emitting the true
per-seed length is deliberate -- padding or truncating to a common axis would
misrepresent what was run.

Usage:
    python tools/extract_po_training_curves.py            # writes paper/data/
    python tools/extract_po_training_curves.py --dry-run  # print, write nothing
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
OUT = ROOT / "paper" / "data" / "po_training_curves.json"

SEEDS = (11, 22, 33)

# [epoch N]  greedy cov=X |S|/n=Y  stoch cov=Z |S|/n=W  |S|/opt=V
LINE = re.compile(
    r"\[epoch (?P<epoch>\d+)\]\s+"
    r"greedy cov=(?P<gcov>[\d.]+)\s+\|S\|/n=(?P<gchv>[\d.]+)\s+"
    r"stoch cov=(?P<scov>[\d.]+)\s+\|S\|/n=(?P<schv>[\d.]+)\s+"
    r"\|S\|/opt=(?P<sopt>[\d.]+)"
)


def parse_log(path: Path) -> dict:
    """Last occurrence of each epoch wins (a resumed run may log an epoch twice)."""
    rows: dict[int, dict] = {}
    with path.open(errors="ignore") as fh:
        for line in fh:
            m = LINE.search(line)
            if not m:
                continue
            g = m.groupdict()
            rows[int(g["epoch"])] = {
                "cov": float(g["gcov"]),
                "chv": float(g["gchv"]),
                "sopt": float(g["sopt"]),
                "stoch_cov": float(g["scov"]),
            }
    epochs = sorted(rows)
    if not epochs:
        raise SystemExit(f"[extract] no epoch lines matched in {path}")
    gaps = [b - a for a, b in zip(epochs, epochs[1:]) if b - a > 1]
    return {
        "epochs": epochs,
        "coverage_greedy": [rows[e]["cov"] for e in epochs],
        "guard_ratio_greedy": [rows[e]["chv"] for e in epochs],
        "size_over_opt_greedy": [rows[e]["sopt"] for e in epochs],
        "coverage_stoch": [rows[e]["stoch_cov"] for e in epochs],
        "n_epochs": len(epochs),
        "epoch_first": epochs[0],
        "epoch_last": epochs[-1],
        "missing_epochs": gaps,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = {
        "note": (
            "Per-epoch PO/BT training curves for the three replication policies "
            "(seeds 11/22/33), recovered from logs/policy_seed*.log by "
            "tools/extract_po_training_curves.py. Metrics are greedy-decode "
            "coverage, |S|/n and |S|/OPT on the first 50 polygons of the 8000 "
            "TRAINING prefix (epoch_eval_k=50, train_size=8000), scored with "
            "disc-vis approximate coverage. This is training data under an "
            "approximate metric: usable for the shape of optimization, NOT "
            "comparable cell-for-cell with the results tables, which are all "
            "held-out and exact-CGAL scored. The released "
            "policy (seed 1234) predates per-epoch logging and is absent here; "
            "its four surviving checkpoints live in po_agp_training.json. Runs "
            "end at different epochs (early stop / cut short), which is "
            "reported as-is."
        ),
        "eval_subsample": 50,
        "seeds": {},
    }
    for sd in SEEDS:
        p = LOGS / f"policy_seed{sd}.log"
        if not p.exists():
            print(f"[extract] WARNING missing {p}, skipping seed {sd}")
            continue
        d = parse_log(p)
        out["seeds"][str(sd)] = d
        print(f"[extract] seed{sd}: {d['n_epochs']} epochs "
              f"({d['epoch_first']}..{d['epoch_last']}), "
              f"gaps={len(d['missing_epochs'])}  "
              f"cov {d['coverage_greedy'][0]:.3f}->{d['coverage_greedy'][-1]:.3f}  "
              f"|S|/OPT {d['size_over_opt_greedy'][0]:.2f}->"
              f"{d['size_over_opt_greedy'][-1]:.2f}")

    if args.dry_run:
        print("[extract] --dry-run: nothing written")
        return
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"[extract] wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
