"""Standalone evaluation for the SetPredictor.

Loads a saved SetPredictor checkpoint and runs a threshold sweep on the
validation set, with optional iterative refinement (the model feeds its own
output back as the new ``in_S`` for K more passes).

Three operating modes:

  1. Single pass (default):
         python eval_set_predictor.py --checkpoint <path>

  2. Iterative refinement:
         python eval_set_predictor.py --checkpoint <path> --iter-passes 3

  3. Full sweep (thresholds × iterations) — produces a 2-D Pareto table:
         python eval_set_predictor.py --checkpoint <path> \\
             --iter-passes-sweep 1 2 3 5

The pointer encoder embeddings are computed once per polygon (no per-step
geometry cost). Iterating the SetPredictor on its own output is *not* the
same as the iterative editor — each pass is a confident full-set
classification, not a per-step incremental edit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
except ImportError:
    pass

from set_predictor import SetPredictor, extract_pointer_embeddings
from utils import evaluate_polygon_visibility_numpy_wo_gt
from po_agp import create_agp_model, _read_opt_solution
from train_set_predictor import SetPredDataset, make_batch


# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to set_predictor_*.pt checkpoint.")
    p.add_argument("--val-traj", type=str, required=True,
                   help="Trajectory pickle (records used for seed + polygon "
                        "points; trajectory itself is ignored).")
    p.add_argument("--pointer-checkpoint", type=str,
                   default="checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt")
    p.add_argument("--embedding-size", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--n-glimpses", type=int, default=1)
    p.add_argument("--tanh-exploration", type=float, default=10.0)

    p.add_argument("--thresholds", nargs="+", type=float,
                   default=[0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9],
                   help="Probability thresholds to evaluate.")
    p.add_argument("--iter-passes", type=int, default=1,
                   help="Apply the SetPredictor K times, feeding each "
                        "output back as the new in_S for the next pass. "
                        "K=1 is the standard single-shot (no iteration).")
    p.add_argument("--iter-passes-sweep", nargs="+", type=int,
                   default=None,
                   help="If set, sweep over multiple values of --iter-passes "
                        "and produce a 2-D table.")

    p.add_argument("--n-samples", type=int, default=-1,
                   help="Number of dev polygons (-1 = all).")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", type=str, default="results/set_predictor_eval.json")
    p.add_argument("--sol-dir", type=str, default=None,
                   help="Directory containing .solution files for |S|/OPT. "
                        "Overrides the default DATASET_PATH/dev path. "
                        "Pass the directory directly (e.g. DATASET_PATH/large).")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
#  Iterative inference
# ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_iterative_inference(model: SetPredictor, ptr_emb: torch.Tensor,
                            pts: torch.Tensor, in_S_init: torch.Tensor,
                            pad: torch.Tensor, threshold: float,
                            n_passes: int) -> torch.Tensor:
    """Apply SetPredictor n_passes times, feeding the previous prediction
    back as the new in_S mask.

    Pointer embeddings (ptr_emb) and polygon coords (pts) stay fixed across
    passes — only in_S changes. Returns the final (B, L) boolean mask of
    predicted set membership.
    """
    in_S = in_S_init
    for _ in range(n_passes):
        logits = model(ptr_emb, pts, in_S, pad)
        probs = torch.sigmoid(logits)
        in_S_new = probs >= threshold
        # Mask out padded positions so they never enter the set.
        in_S_new = in_S_new & (~pad)
        in_S = in_S_new
    return in_S


def _coverage_exact(pts_np: np.ndarray, guards: list[int], name: str) -> float:
    if not guards:
        return 0.0
    try:
        return float(evaluate_polygon_visibility_numpy_wo_gt(
            pts_np, np.array(guards, dtype=np.int64), name,
        ))
    except Exception:
        return 0.0


def sweep_thresholds_x_iters(model: SetPredictor, ds: SetPredDataset,
                             pointer, device, args,
                             thresholds: list[float],
                             iter_counts: list[int],
                             dev_sol_dir: str | None) -> dict:
    """For each (threshold, K) pair, run the model and average cov, |S|/n,
    |S|/OPT over args.n_samples polygons."""
    model.eval()
    pointer.eval()

    if args.n_samples > 0:
        indices = list(range(min(args.n_samples, len(ds))))
    else:
        indices = list(range(len(ds)))

    # Bucket by polygon size for batched forward.
    buckets = defaultdict(list)
    for i in indices:
        buckets[ds.records[i]["n"]].append(i)

    # Accumulators per (t, K).
    metrics: dict[tuple[float, int], dict] = {
        (t, K): {"cov": [], "chv": [], "opt": []}
        for t in thresholds for K in iter_counts
    }
    seed_acc = {"cov": [], "chv": [], "opt": []}

    n_done = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for n, ids in buckets.items():
            for start in range(0, len(ids), args.batch_size):
                chunk = ids[start: start + args.batch_size]
                ptr_emb, pts, in_S_init, pad, _, names, _ = make_batch(
                    ds, chunk, pointer, device,
                )

                pts_np_batch = [pts[b].cpu().numpy() for b in range(len(chunk))]
                # Seed metrics — same across (t, K).
                for b, idx in enumerate(chunk):
                    rec = ds.records[idx]
                    seed_idx = rec["seed"]
                    cov = _coverage_exact(pts_np_batch[b], seed_idx, names[b])
                    seed_acc["cov"].append(cov)
                    seed_acc["chv"].append(len(seed_idx) / max(1, n))
                    opt_sol = _read_opt_solution(dev_sol_dir, names[b]) if dev_sol_dir else None
                    if opt_sol:
                        seed_acc["opt"].append(len(seed_idx) / len(opt_sol))

                # For each threshold, run all K values (we can reuse the
                # iteration: K=3 includes K=1's and K=2's intermediate outputs).
                for t in thresholds:
                    in_S = in_S_init
                    max_K = max(iter_counts)
                    snapshots: dict[int, torch.Tensor] = {}
                    for k in range(1, max_K + 1):
                        logits = model(ptr_emb, pts, in_S, pad)
                        probs = torch.sigmoid(logits)
                        in_S = (probs >= t) & (~pad)
                        if k in iter_counts:
                            snapshots[k] = in_S.clone()

                    for K, mask in snapshots.items():
                        for b, idx in enumerate(chunk):
                            keep = mask[b].nonzero(as_tuple=True)[0].cpu().tolist()
                            cov = _coverage_exact(pts_np_batch[b], keep, names[b])
                            metrics[(t, K)]["cov"].append(cov)
                            metrics[(t, K)]["chv"].append(len(keep) / max(1, n))
                            opt_sol = _read_opt_solution(dev_sol_dir, names[b]) if dev_sol_dir else None
                            if opt_sol and keep:
                                metrics[(t, K)]["opt"].append(len(keep) / len(opt_sol))

                n_done += len(chunk)
        wall = time.perf_counter() - t0

    def _dist(cov_list: list[float]) -> dict:
        """Per-polygon coverage distribution stats.

        Reports: mean, min, percentile 1/5/10, and counts of polygons
        achieving various coverage thresholds. The 'not_covered_*' fields
        tell us how many polygons fail the criterion — directly answers
        'how many polygons aren't fully covered at this threshold?'.
        """
        if not cov_list:
            return {}
        arr = np.array(cov_list)
        n = len(arr)
        return {
            "cov_mean":    float(arr.mean()),
            "cov_min":     float(arr.min()),
            "cov_p01":     float(np.percentile(arr, 1)),
            "cov_p05":     float(np.percentile(arr, 5)),
            "cov_p10":     float(np.percentile(arr, 10)),
            "n_total":     n,
            "n_cov_eq_1":     int((arr >= 0.99995).sum()),    # essentially exact
            "n_cov_ge_0999":  int((arr >= 0.999).sum()),
            "n_cov_ge_099":   int((arr >= 0.99).sum()),
            "n_cov_ge_095":   int((arr >= 0.95).sum()),
            "not_covered_eq_1":    int((arr < 0.99995).sum()),
            "not_covered_ge_0999": int((arr < 0.999).sum()),
            "not_covered_ge_099":  int((arr < 0.99).sum()),
            "not_covered_ge_095":  int((arr < 0.95).sum()),
        }

    seed_summary = {
        "cov": float(np.mean(seed_acc["cov"])) if seed_acc["cov"] else None,
        "chv": float(np.mean(seed_acc["chv"])) if seed_acc["chv"] else None,
        "opt": float(np.mean(seed_acc["opt"])) if seed_acc["opt"] else None,
        "dist": _dist(seed_acc["cov"]),
    }
    cell_summary = {}
    for (t, K), m in metrics.items():
        cell_summary[f"t={t}|K={K}"] = {
            "cov": float(np.mean(m["cov"])) if m["cov"] else None,
            "chv": float(np.mean(m["chv"])) if m["chv"] else None,
            "opt": float(np.mean(m["opt"])) if m["opt"] else None,
            "n": len(m["cov"]),
            "dist": _dist(m["cov"]),
        }
    return {
        "seed": seed_summary,
        "cells": cell_summary,
        "n_polygons": n_done,
        "wall_s": wall,
    }


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval-setpred] device={device}")
    print(f"[eval-setpred] checkpoint={args.checkpoint}")

    # Resolve which K values to sweep.
    if args.iter_passes_sweep:
        iter_counts = sorted(set(args.iter_passes_sweep))
    else:
        iter_counts = [args.iter_passes]
    print(f"[eval-setpred] iter_passes = {iter_counts}")
    print(f"[eval-setpred] thresholds  = {args.thresholds}")

    # Load checkpoint to learn architecture.
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    H = saved_args.get("predictor_hidden", 128)
    L = saved_args.get("predictor_attn_layers", 3)
    HD = saved_args.get("predictor_heads", 8)
    H_ptr = saved_args.get("hidden_size", args.hidden_size)
    no_enc = saved_args.get("disable_ptr_emb", False)
    print(f"[eval-setpred] architecture: hidden={H} layers={L} heads={HD} ptr_emb_dim={H_ptr} disable_ptr_emb={no_enc}")

    model = SetPredictor(ptr_emb_dim=H_ptr, hidden=H, n_attn_layers=L, heads=HD,
                         disable_ptr_emb=no_enc).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[eval-setpred] params = {model.num_params():,}")

    # Pointer (frozen).
    pointer = create_agp_model(
        args.embedding_size, args.hidden_size, args.n_glimpses,
        args.tanh_exploration, use_tanh=True, temperature=1.0,
    )
    pkpath = args.pointer_checkpoint
    if not os.path.isabs(pkpath):
        pkpath = os.path.join(REPO_ROOT, pkpath)
    pckpt = torch.load(pkpath, map_location=device, weights_only=False)
    psd = pckpt["model_state_dict"] if isinstance(pckpt, dict) and "model_state_dict" in pckpt else pckpt
    pointer.load_state_dict(psd, strict=False)
    pointer.eval()
    for p in pointer.parameters():
        p.requires_grad_(False)

    # Dataset.
    ds = SetPredDataset(args.val_traj)

    # Solution directory for |S|/OPT.
    if args.sol_dir:
        dev_sol_dir = args.sol_dir
    else:
        DATASET_PATH = os.getenv("DATASET_PATH")
        dev_sol_dir = os.path.join(DATASET_PATH, "dev") if DATASET_PATH else None

    summary = sweep_thresholds_x_iters(
        model, ds, pointer, device, args,
        thresholds=args.thresholds,
        iter_counts=iter_counts,
        dev_sol_dir=dev_sol_dir,
    )

    # ── Pretty print: one table per K ───────────────────────────────
    print()
    print("=" * 70)
    print(f"  SetPredictor sweep  ({summary['n_polygons']} polygons, wall {summary['wall_s']:.0f}s)")
    print("=" * 70)
    seed = summary["seed"]
    opt_s = f"{seed['opt']:.4f}" if seed['opt'] is not None else "  —  "
    print(f"  seed  cov={seed['cov']:.4f}  |S|/n={seed['chv']:.4f}  "
          f"|S|/OPT={opt_s}")
    if seed.get("dist"):
        d = seed["dist"]
        print(f"  seed dist: min cov={d['cov_min']:.4f}  "
              f"p01={d['cov_p01']:.4f}  p05={d['cov_p05']:.4f}  "
              f"p10={d['cov_p10']:.4f}")
        print(f"  seed fully-covered: cov=1.0: {d['n_cov_eq_1']}/{d['n_total']}  "
              f"cov≥0.999: {d['n_cov_ge_0999']}/{d['n_total']}  "
              f"cov≥0.99: {d['n_cov_ge_099']}/{d['n_total']}  "
              f"cov≥0.95: {d['n_cov_ge_095']}/{d['n_total']}")

    for K in iter_counts:
        print(f"\n  K = {K} iterations:")
        print(f"    {'t':>6}  {'cov':>8}  {'|S|/n':>8}  {'|S|/OPT':>10}  "
              f"{'min':>8}  {'p05':>8}  {'#=1':>6}  {'#≥.999':>8}  {'#≥.99':>7}  {'#<.95':>7}")
        for t in args.thresholds:
            m = summary["cells"][f"t={t}|K={K}"]
            cov = m["cov"] if m["cov"] is not None else 0.0
            chv = m["chv"] if m["chv"] is not None else 0.0
            opt = m["opt"] if m["opt"] is not None else 0.0
            d = m.get("dist", {})
            print(f"    {t:>6.2f}  {cov:>8.4f}  {chv:>8.4f}  {opt:>10.4f}  "
                  f"{d.get('cov_min', 0):>8.4f}  {d.get('cov_p05', 0):>8.4f}  "
                  f"{d.get('n_cov_eq_1', 0):>6}  {d.get('n_cov_ge_0999', 0):>8}  "
                  f"{d.get('n_cov_ge_099', 0):>7}  {d.get('not_covered_ge_095', 0):>7}")

    # ── Best cov-preserving operating point ────────────────────────
    print("\n  Best cov-preserving operating points (cov ≥ seed AND |S|/n < seed):")
    seed_cov = seed["cov"] or 0.0
    seed_chv = seed["chv"] or 1.0
    cands = []
    for K in iter_counts:
        for t in args.thresholds:
            m = summary["cells"][f"t={t}|K={K}"]
            if m["cov"] is None or m["chv"] is None:
                continue
            if m["cov"] + 1e-4 >= seed_cov and m["chv"] < seed_chv:
                cands.append((t, K, m))
    cands.sort(key=lambda x: x[2]["chv"])
    if cands:
        for t, K, m in cands[:5]:
            print(f"    t={t:.2f}  K={K}  "
                  f"cov={m['cov']:.4f}  |S|/n={m['chv']:.4f}  "
                  f"|S|/OPT={m['opt']:.4f}  "
                  f"Δ|S|/n={seed_chv - m['chv']:+.4f}")
    else:
        print("    (none)")

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO_ROOT, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved -> {out_path}")


if __name__ == "__main__":
    main()
