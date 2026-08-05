"""Scoping experiment for Idea 2 (pointer + neural editor).

For every validation polygon:
  1. Load the lstm_bt baseline pointer.
  2. Greedy decode -> seed guard set S_0.
  3. Run local_search_improve_disc to convergence.
  4. Record number of edits (remove + add + swap), pre/post coverage,
     pre/post |S|, and OPT size when available.

Output: per-instance JSON + summary histogram. The histogram tells us
whether a small neural editor (1-3 step) is enough to imitate LS, or
whether LS typically takes many steps (in which case a single-step
editor is unrealistic).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
except ImportError:
    pass

from dataset import collate_fn
from po_agp import (
    create_agp_model,
    prepare_datasets,
    local_search_improve_disc,
    _read_opt_solution,
)
from utils import evaluate_polygon_visibility_numpy_wo_gt, get_or_build_disc_vis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt")
    p.add_argument("--val-dir", type=str, default=None,
                   help="Override validation dir; defaults to $DATASET_PATH/dev")
    p.add_argument("--n-samples", type=int, default=-1,
                   help="-1 = full validation set")
    p.add_argument("--embedding-size", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--n-glimpses", type=int, default=1)
    p.add_argument("--tanh-exploration", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=0.99)
    p.add_argument("--tau-penalty", type=float, default=3.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--disc-vis-samples", type=int, default=500)
    p.add_argument("--ls-max-iter", type=int, default=200,
                   help="Cap iterations so we get true convergence count")
    p.add_argument("--out", type=str,
                   default="results/scope_ls_edits.json")
    return p.parse_args()


def load_model(args, device):
    model = create_agp_model(
        args.embedding_size, args.hidden_size, args.n_glimpses,
        args.tanh_exploration, use_tanh=True, temperature=1.0,
    )
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [load] missing keys: {len(missing)}")
    if unexpected:
        print(f"  [load] unexpected keys: {len(unexpected)}")
    model.eval()
    return model


def main() -> None:
    args = parse_args()

    DATASET_PATH = os.getenv("DATASET_PATH")
    if DATASET_PATH is None:
        raise EnvironmentError("DATASET_PATH not set")
    val_dir = args.val_dir or os.path.join(DATASET_PATH, "dev")
    train_dir = os.path.join(DATASET_PATH, "train")  # only needed for prepare_datasets signature

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[scope] device={device}  checkpoint={args.checkpoint}")
    print(f"[scope] val_dir={val_dir}")

    _, val_ds = prepare_datasets(train_dir, val_dir, normalize=True)
    if args.n_samples > 0:
        val_ds.samples = val_ds.samples[: args.n_samples]
    print(f"[scope] using {len(val_ds)} validation polygons")

    model = load_model(args, device)
    loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    records: list[dict] = []
    t_total = time.perf_counter()

    with torch.no_grad():
        for i, (batch_data, pad_mask, lengths, names) in enumerate(loader):
            batch_data = batch_data.to(device)
            pad_mask = pad_mask.to(device)
            lens_t = torch.tensor(lengths, dtype=torch.long, device=device)
            n = int(lengths[0])
            name = names[0]
            pts = batch_data[0, :n].detach().cpu().numpy()

            # Prebuild disc_vis (so the cost isn't blamed on LS).
            get_or_build_disc_vis(pts, name, n_samples=args.disc_vis_samples)

            # Greedy decode (matches evaluate_po path with no_eos=False).
            t0 = time.perf_counter()
            det_idxs, _ = model(
                batch_data, padding_mask=pad_mask, lengths=lens_t,
                deterministic=True, no_eos=False,
                eos_cov_threshold=0.0,
            )
            seed_sol = [int(idx) for idx in det_idxs[0] if int(idx) < n]
            t_decode = time.perf_counter() - t0

            # Coverage of the seed.
            try:
                seed_cov = float(evaluate_polygon_visibility_numpy_wo_gt(
                    pts, np.array(seed_sol, dtype=np.int64), name,
                )) if seed_sol else 0.0
            except Exception:
                seed_cov = 0.0

            # Run LS to convergence.
            t0 = time.perf_counter()
            ls_guards, ls_r, ls_stats = local_search_improve_disc(
                pts, seed_sol, name, n,
                max_iter=args.ls_max_iter, enable_swap=True,
                enable_remove=True, enable_add=True,
                lam=args.lam, tau=args.tau, tau_penalty=args.tau_penalty,
                cap_at_tau=False, n_samples=args.disc_vis_samples,
                reward_fn_fallback=None, monotone_coverage=True,
            )
            t_ls = time.perf_counter() - t0

            try:
                ls_cov = float(evaluate_polygon_visibility_numpy_wo_gt(
                    pts, np.array(ls_guards, dtype=np.int64), name,
                )) if ls_guards else 0.0
            except Exception:
                ls_cov = 0.0

            opt_sol = _read_opt_solution(val_dir, name)
            opt_size = len(opt_sol) if opt_sol else None

            edits = ls_stats.get("n_remove", 0) + ls_stats.get("n_add", 0) + ls_stats.get("n_swap", 0)

            records.append({
                "name": name,
                "n": n,
                "seed_size": len(seed_sol),
                "seed_cov": seed_cov,
                "ls_size": len(ls_guards),
                "ls_cov": ls_cov,
                "n_remove": ls_stats.get("n_remove", 0),
                "n_add": ls_stats.get("n_add", 0),
                "n_swap": ls_stats.get("n_swap", 0),
                "n_edits": edits,
                "iterations": ls_stats.get("iterations", 0),
                "delta_size": len(ls_guards) - len(seed_sol),
                "delta_cov": ls_cov - seed_cov,
                "opt_size": opt_size,
                "t_decode_s": t_decode,
                "t_ls_s": t_ls,
            })

            if (i + 1) % 50 == 0:
                running = [r["n_edits"] for r in records]
                med = float(np.median(running))
                p90 = float(np.percentile(running, 90))
                print(f"  [{i+1}/{len(val_ds)}] median_edits={med:.0f}  p90_edits={p90:.0f}  "
                      f"last_name={name}  edits={edits}  Δ|S|={records[-1]['delta_size']}  Δcov={records[-1]['delta_cov']:+.3f}")

    # ── Summary ──────────────────────────────────────────────────────
    edits = np.array([r["n_edits"] for r in records], dtype=np.int64)
    iters = np.array([r["iterations"] for r in records], dtype=np.int64)
    seed_sz = np.array([r["seed_size"] for r in records], dtype=np.float64)
    ls_sz = np.array([r["ls_size"] for r in records], dtype=np.float64)
    seed_cov = np.array([r["seed_cov"] for r in records], dtype=np.float64)
    ls_cov = np.array([r["ls_cov"] for r in records], dtype=np.float64)

    edit_hist = Counter(int(e) for e in edits)
    summary = {
        "n_polygons": len(records),
        "edits_mean": float(edits.mean()),
        "edits_median": float(np.median(edits)),
        "edits_p25": float(np.percentile(edits, 25)),
        "edits_p75": float(np.percentile(edits, 75)),
        "edits_p90": float(np.percentile(edits, 90)),
        "edits_p99": float(np.percentile(edits, 99)),
        "edits_max": int(edits.max()) if len(edits) else 0,
        "iters_mean": float(iters.mean()),
        "remove_total": int(sum(r["n_remove"] for r in records)),
        "add_total": int(sum(r["n_add"] for r in records)),
        "swap_total": int(sum(r["n_swap"] for r in records)),
        "seed_size_mean": float(seed_sz.mean()),
        "ls_size_mean": float(ls_sz.mean()),
        "seed_cov_mean": float(seed_cov.mean()),
        "ls_cov_mean": float(ls_cov.mean()),
        "delta_size_mean": float((ls_sz - seed_sz).mean()),
        "edit_histogram": dict(sorted(edit_hist.items())),
        "frac_edits_le_1": float((edits <= 1).mean()),
        "frac_edits_le_2": float((edits <= 2).mean()),
        "frac_edits_le_3": float((edits <= 3).mean()),
        "frac_edits_le_5": float((edits <= 5).mean()),
        "checkpoint": args.checkpoint,
        "val_dir": val_dir,
        "ls_params": {
            "tau": args.tau, "lam": args.lam,
            "tau_penalty": args.tau_penalty,
            "disc_vis_samples": args.disc_vis_samples,
            "max_iter": args.ls_max_iter,
            "monotone_coverage": True,
        },
        "wall_time_s": time.perf_counter() - t_total,
    }

    print()
    print("=" * 60)
    print(f"  Scoping summary  ({len(records)} polygons)")
    print("=" * 60)
    print(f"  edits   mean={summary['edits_mean']:.2f}   median={summary['edits_median']:.0f}")
    print(f"          p25={summary['edits_p25']:.0f}   p75={summary['edits_p75']:.0f}   "
          f"p90={summary['edits_p90']:.0f}   p99={summary['edits_p99']:.0f}   max={summary['edits_max']}")
    print(f"  P[edits<=1]={summary['frac_edits_le_1']:.3f}  "
          f"P[edits<=2]={summary['frac_edits_le_2']:.3f}  "
          f"P[edits<=3]={summary['frac_edits_le_3']:.3f}  "
          f"P[edits<=5]={summary['frac_edits_le_5']:.3f}")
    print(f"  ops     remove={summary['remove_total']}  add={summary['add_total']}  swap={summary['swap_total']}")
    print(f"  size    seed={summary['seed_size_mean']:.2f} -> ls={summary['ls_size_mean']:.2f}  "
          f"(Δ={summary['delta_size_mean']:+.2f})")
    print(f"  cov     seed={summary['seed_cov_mean']:.3f} -> ls={summary['ls_cov_mean']:.3f}")
    print(f"  wall    {summary['wall_time_s']:.1f}s")
    print()
    top_bins = sorted(edit_hist.items())[:15]
    print("  histogram (first 15 bins):")
    for k, v in top_bins:
        bar = "#" * int(50 * v / max(edit_hist.values()))
        print(f"    {k:3d} edits : {v:5d} {bar}")

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO_ROOT, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"\n  saved -> {out_path}")


if __name__ == "__main__":
    main()
