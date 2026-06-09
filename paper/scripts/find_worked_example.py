"""Score OOD test polygons as candidates for the worked-example figure (row 2).

The original selector (build_paper_data.build_worked_examples) picked by seed-
coverage gap alone and never bounded how many guards the PROBE adds, landing on
randsimple-300-4 (probe uses 229/300 vertices, |S|/OPT~6 -- a tail case that
misrepresents the method). This script computes, for OOD candidates, the full
set of metrics a good HONEST example needs:

  n, seed_cov, probe_cov@0.20, |S_seed|, |S_probe|, |OPT|,
  |S_probe|/OPT, |S_probe|/n

so we can pick an example that is genuinely OOD, legible, shows a real seed gap,
and recovers feasibility at a guard cost REPRESENTATIVE of the test-split mean
(|S|/OPT ~ 2.9 at t=0.20), not a 6x tail outlier.

Run: /home/dseverdi/.conda/envs/MLAG/bin/python paper/scripts/find_worked_example.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from po_agp import _read_opt_solution
from set_predictor import extract_pointer_embeddings
# reuse the exact same loaders / coverage as the figure-data builder
from build_paper_data import (
    _load_pointer, _load_setpredictor, _coverage_exact, TEST_TRAJ,
)

# Candidate window: genuinely (and for the figure, clearly) OOD -- n well above
# the n=198 training max -- but still legible in a 3-panel figure, with a
# visible-but-not-catastrophic seed gap.
N_LO, N_HI = 199, 400
SEED_COV_LO, SEED_COV_HI = 0.75, 0.95
THRESHOLD = 0.20


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pointer = _load_pointer(device)
    setpred = _load_setpredictor(device)

    recs = pickle.load(open(TEST_TRAJ, "rb"))["records"]
    DATASET_PATH = os.getenv("DATASET_PATH")
    sol_dir = os.path.join(DATASET_PATH, "test") if DATASET_PATH else None

    cands = [r for r in recs
             if N_LO <= r["n"] <= N_HI
             and SEED_COV_LO <= r["seed_cov"] <= SEED_COV_HI]
    print(f"scoring {len(cands)} candidates "
          f"(n in [{N_LO},{N_HI}], seed_cov in [{SEED_COV_LO},{SEED_COV_HI}])\n")

    rows = []
    for rec in cands:
        n = rec["n"]
        pts = torch.tensor(np.asarray(rec["points"], dtype=np.float32),
                           device=device).unsqueeze(0)
        lengths = torch.tensor([n], device=device)
        in_S = torch.zeros(1, n, dtype=torch.bool, device=device)
        for idx in rec["seed"]:
            if 0 <= idx < n:
                in_S[0, idx] = True
        pad = torch.zeros(1, n, dtype=torch.bool, device=device)
        with torch.no_grad():
            ptr_emb = extract_pointer_embeddings(pointer, pts, lengths)
            logits = setpred(ptr_emb, pts, in_S, pad)
            probs = torch.sigmoid(logits)
            keep = (probs >= THRESHOLD) & (~pad)
        probe_idxs = keep[0].nonzero(as_tuple=True)[0].cpu().tolist()
        pts_np = pts[0].cpu().numpy()
        probe_cov = _coverage_exact(pts_np, probe_idxs, rec["name"])

        opt_idxs = _read_opt_solution(sol_dir, rec["name"]) if sol_dir else None
        n_opt = len(opt_idxs) if opt_idxs else None

        rows.append({
            "name": rec["name"],
            "n": int(n),
            "seed_cov": float(rec["seed_cov"]),
            "probe_cov": float(probe_cov),
            "n_seed": int(len(rec["seed"])),
            "n_probe": int(len(probe_idxs)),
            "n_opt": n_opt,
            "probe_over_opt": (len(probe_idxs) / n_opt) if n_opt else None,
            "probe_over_n": len(probe_idxs) / n,
        })

    # Keep only the ones where the probe actually restores feasibility AND OPT exists.
    good = [r for r in rows
            if r["probe_cov"] >= 0.95 and r["n_opt"] and r["seed_cov"] < 0.95]
    # Rank: representative+economical probe first (low |S|/OPT), then legibility (small n).
    good.sort(key=lambda r: (r["probe_over_opt"], r["n"]))

    hdr = (f"{'name':<22}{'n':>5}{'seedCov':>9}{'probeCov':>9}"
           f"{'|seed|':>7}{'|probe|':>8}{'|opt|':>6}{'pr/opt':>8}{'pr/n':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in good:
        print(f"{r['name']:<22}{r['n']:>5}{r['seed_cov']:>9.3f}"
              f"{r['probe_cov']:>9.3f}{r['n_seed']:>7}{r['n_probe']:>8}"
              f"{r['n_opt']:>6}{r['probe_over_opt']:>8.2f}{r['probe_over_n']:>7.2f}")

    out = REPO_ROOT / "paper" / "data" / "worked_example_candidates.json"
    out.write_text(json.dumps({"candidates": good, "all_scored": rows}, indent=2))
    print(f"\nwrote {out} ({len(good)} feasible candidates of {len(rows)} scored)")


if __name__ == "__main__":
    main()
