"""Reconstruct a coarse PO/BT training curve from saved checkpoints.

Per-epoch metrics were not preserved during the original PO/BT training run.
However, four checkpoints were saved at named epochs (110, 114 = "best",
160, 200). We greedy-decode each checkpoint on a sample of dev_test polygons
and compute CGAL exact coverage and |S|/OPT. The resulting 4-point curve
shows the late-training trajectory of the policy.

Writes paper/data/po_agp_training.json with the schema build_figures.py
expects:

  {"epochs": [int, ...],
   "coverage_greedy_mean": [float, ...],
   "guard_ratio_greedy_mean": [float, ...],
   "size_over_opt_greedy_mean": [float, ...]}
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DATA = REPO_ROOT / "paper" / "data"

sys.path.insert(0, str(REPO_ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from po_agp import create_agp_model, _read_opt_solution
from utils import evaluate_polygon_visibility_numpy_wo_gt

CKPT_DIR = REPO_ROOT / "checkpoints/v3/po_agp/lstm_bt"
DEV_TEST_TRAJ = REPO_ROOT / "data/ls_trajectories_dev_test.pkl"
N_SAMPLE = 100

CHECKPOINTS = [
    ("po_agp_epoch110.pt", 110),
    ("po_agp_best_greedy.pt", 114),
    ("po_agp_epoch160.pt", 160),
    ("po_agp_final_epoch200.pt", 200),
]


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_pointer(ckpt_name: str, device: str):
    pointer = create_agp_model(128, 128, 1, 10.0, use_tanh=True, temperature=1.0)
    ckpt = torch.load(CKPT_DIR / ckpt_name, map_location=device, weights_only=False)
    psd = ckpt["model_state_dict"]
    pointer.load_state_dict(psd, strict=False)
    pointer.to(device).eval()
    for p in pointer.parameters():
        p.requires_grad_(False)
    return pointer


def _coverage_exact(pts_np: np.ndarray, guards: list, name: str) -> float:
    if not guards:
        return 0.0
    try:
        return float(evaluate_polygon_visibility_numpy_wo_gt(
            pts_np, np.array(guards, dtype=np.int64), name,
        ))
    except Exception:
        return 0.0


@torch.no_grad()
def eval_checkpoint(ckpt_name: str, records: list, sol_dir: str | None,
                    device: str) -> dict:
    print(f"  loading {ckpt_name}")
    pointer = _load_pointer(ckpt_name, device)
    covs, ratios, opts = [], [], []
    for rec in records:
        pts_np = np.asarray(rec["points"], dtype=np.float32)
        n = rec["n"]
        pts_t = torch.tensor(pts_np, device=device).unsqueeze(0)  # (1, n, 2)
        lt = torch.tensor([n], dtype=torch.long, device=device)
        # collate_fn convention: True for real vertices, False for padding.
        pad = torch.ones(1, n, dtype=torch.bool, device=device)
        try:
            det_idxs, _ = pointer(pts_t, padding_mask=pad, lengths=lt,
                                   deterministic=True)
        except Exception as exc:
            print(f"  warn: decode failed for {rec['name']}: {exc}")
            continue
        sol = [int(idx) for idx in det_idxs[0] if int(idx) < n]
        cov = _coverage_exact(pts_np, sol, rec["name"])
        covs.append(cov)
        ratios.append(len(sol) / max(1, n))
        opt_sol = _read_opt_solution(sol_dir, rec["name"]) if sol_dir else None
        if opt_sol and sol:
            opts.append(len(sol) / len(opt_sol))
    return {
        "coverage_greedy_mean": float(np.mean(covs)) if covs else None,
        "guard_ratio_greedy_mean": float(np.mean(ratios)) if ratios else None,
        "size_over_opt_greedy_mean": float(np.mean(opts)) if opts else None,
        "n_evaluated": len(covs),
    }


def main() -> None:
    device = _device()
    print(f"device: {device}")

    with open(DEV_TEST_TRAJ, "rb") as f:
        records = pickle.load(f)["records"]

    rng = np.random.default_rng(1234)
    idx = rng.choice(len(records), size=min(N_SAMPLE, len(records)), replace=False)
    sample = [records[i] for i in sorted(idx.tolist())]
    print(f"sampling {len(sample)} dev_test polygons (seed 1234)")

    DATASET_PATH = os.getenv("DATASET_PATH")
    dev_sol_dir = os.path.join(DATASET_PATH, "dev") if DATASET_PATH else None

    epochs, covs, ratios, opts = [], [], [], []
    t0 = time.perf_counter()
    for ckpt_name, ep in CHECKPOINTS:
        if not (CKPT_DIR / ckpt_name).exists():
            print(f"  skip {ckpt_name}: not found")
            continue
        m = eval_checkpoint(ckpt_name, sample, dev_sol_dir, device)
        if m["coverage_greedy_mean"] is None:
            print(f"  skip {ckpt_name}: no valid evaluations")
            continue
        opt_s = (f"{m['size_over_opt_greedy_mean']:.4f}"
                 if m['size_over_opt_greedy_mean'] is not None else "n/a")
        print(f"  epoch={ep}: cov={m['coverage_greedy_mean']:.4f}  "
              f"|S|/n={m['guard_ratio_greedy_mean']:.4f}  "
              f"|S|/OPT={opt_s}  "
              f"(n_evaluated={m['n_evaluated']})")
        epochs.append(ep)
        covs.append(m["coverage_greedy_mean"])
        ratios.append(m["guard_ratio_greedy_mean"])
        opts.append(m["size_over_opt_greedy_mean"])

    out = {
        "epochs": epochs,
        "coverage_greedy_mean": covs,
        "guard_ratio_greedy_mean": ratios,
        "size_over_opt_greedy_mean": opts,
        "note": (
            f"Reconstructed from {len(epochs)} late-training checkpoints "
            f"evaluated on {len(sample)} dev_test polygons. "
            f"Per-epoch logging was not preserved during the original "
            f"training run; this curve shows only the checkpointed epochs."
        ),
        "n_dev_sample": len(sample),
        "seed": 1234,
    }
    out_path = PAPER_DATA / "po_agp_training.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path} ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
