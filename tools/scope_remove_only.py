"""REMOVE-only LS ablation.

Question: of the reward improvement that full LS (remove + swap + add)
makes over the pretrained-pointer seed, what fraction does REMOVE alone
recover?

For each validation polygon:
  seed   = pretrained-pointer greedy decode
  remove = local_search_improve_disc(seed, enable_swap=False, enable_add=False)
  full   = local_search_improve_disc(seed, all enabled)

Recovery fraction = (r_remove - r_seed) / (r_full - r_seed)

Decision rule:
  median recovery >= 0.80  -> single-head pruner is justified
  median recovery <  0.50  -> swap/add does real work; need 2 heads
  in between               -> ambiguous, look at distribution + cov/|S|
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

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
)
from utils import evaluate_polygon_visibility_numpy_wo_gt, get_or_build_disc_vis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt")
    p.add_argument("--val-dir", type=str, default=None)
    p.add_argument("--n-samples", type=int, default=-1)
    p.add_argument("--embedding-size", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--n-glimpses", type=int, default=1)
    p.add_argument("--tanh-exploration", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=0.99)
    p.add_argument("--tau-penalty", type=float, default=3.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--disc-vis-samples", type=int, default=500)
    p.add_argument("--ls-max-iter", type=int, default=200)
    p.add_argument("--out", type=str,
                   default="results/scope_remove_only.json")
    return p.parse_args()


def reward_scalar(cov: float, k: int, n: int, *, lam: float, tau: float,
                  tau_penalty: float) -> float:
    return cov - lam * k / max(1, n) - tau_penalty * max(0.0, tau - cov)


def coverage_exact(pts: np.ndarray, guards: list[int], name: str) -> float:
    if not guards:
        return 0.0
    try:
        return float(evaluate_polygon_visibility_numpy_wo_gt(
            pts, np.array(guards, dtype=np.int64), name,
        ))
    except Exception:
        return 0.0


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
    missing, _ = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [load] missing keys: {len(missing)}")
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    DATASET_PATH = os.getenv("DATASET_PATH")
    if DATASET_PATH is None:
        raise EnvironmentError("DATASET_PATH not set")
    val_dir = args.val_dir or os.path.join(DATASET_PATH, "dev")
    train_dir = os.path.join(DATASET_PATH, "train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[scope-remove] device={device}  checkpoint={args.checkpoint}")
    print(f"[scope-remove] val_dir={val_dir}")
    print(f"[scope-remove] reward params  lam={args.lam}  tau={args.tau}  tau_penalty={args.tau_penalty}")

    _, val_ds = prepare_datasets(train_dir, val_dir, normalize=True)
    if args.n_samples > 0:
        val_ds.samples = val_ds.samples[: args.n_samples]
    print(f"[scope-remove] using {len(val_ds)} validation polygons")

    model = load_model(args, device)
    loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    records: list[dict] = []
    t_total = time.perf_counter()

    def _ls(seed_sol, pts, name, n, *, enable_swap, enable_add, enable_remove):
        return local_search_improve_disc(
            pts, seed_sol, name, n,
            max_iter=args.ls_max_iter,
            enable_swap=enable_swap,
            enable_remove=enable_remove,
            enable_add=enable_add,
            lam=args.lam, tau=args.tau, tau_penalty=args.tau_penalty,
            cap_at_tau=False, n_samples=args.disc_vis_samples,
            reward_fn_fallback=None, monotone_coverage=True,
        )

    with torch.no_grad():
        for i, (batch_data, pad_mask, lengths, names) in enumerate(loader):
            batch_data = batch_data.to(device)
            pad_mask = pad_mask.to(device)
            lens_t = torch.tensor(lengths, dtype=torch.long, device=device)
            n = int(lengths[0])
            name = names[0]
            pts = batch_data[0, :n].detach().cpu().numpy()

            get_or_build_disc_vis(pts, name, n_samples=args.disc_vis_samples)

            det_idxs, _ = model(
                batch_data, padding_mask=pad_mask, lengths=lens_t,
                deterministic=True, no_eos=False,
                eos_cov_threshold=0.0,
            )
            seed_sol = [int(idx) for idx in det_idxs[0] if int(idx) < n]
            seed_cov = coverage_exact(pts, seed_sol, name)
            seed_r = reward_scalar(seed_cov, len(seed_sol), n,
                                   lam=args.lam, tau=args.tau,
                                   tau_penalty=args.tau_penalty)

            # REMOVE-only LS
            rem_sol, _, rem_stats = _ls(seed_sol, pts, name, n,
                                        enable_swap=False, enable_add=False,
                                        enable_remove=True)
            rem_cov = coverage_exact(pts, rem_sol, name)
            rem_r = reward_scalar(rem_cov, len(rem_sol), n,
                                  lam=args.lam, tau=args.tau,
                                  tau_penalty=args.tau_penalty)

            # FULL LS (re-run from same seed)
            full_sol, _, full_stats = _ls(seed_sol, pts, name, n,
                                          enable_swap=True, enable_add=True,
                                          enable_remove=True)
            full_cov = coverage_exact(pts, full_sol, name)
            full_r = reward_scalar(full_cov, len(full_sol), n,
                                   lam=args.lam, tau=args.tau,
                                   tau_penalty=args.tau_penalty)

            d_rem = rem_r - seed_r
            d_full = full_r - seed_r
            # Recovery fraction. Guard against tiny / negative full deltas:
            #   - if full_r ~ seed_r, full LS didn't improve -> recovery
            #     undefined; mark as None and exclude from aggregate.
            #   - if full_r < seed_r somehow (numerical), also exclude.
            if d_full > 1e-6:
                recovery = d_rem / d_full
            else:
                recovery = None

            records.append({
                "name": name,
                "n": n,
                "seed_size": len(seed_sol),
                "seed_cov": seed_cov,
                "seed_r": seed_r,
                "rem_size": len(rem_sol),
                "rem_cov": rem_cov,
                "rem_r": rem_r,
                "rem_n_remove": rem_stats.get("n_remove", 0),
                "full_size": len(full_sol),
                "full_cov": full_cov,
                "full_r": full_r,
                "full_n_remove": full_stats.get("n_remove", 0),
                "full_n_swap":   full_stats.get("n_swap", 0),
                "full_n_add":    full_stats.get("n_add", 0),
                "delta_r_remove": d_rem,
                "delta_r_full":   d_full,
                "recovery": recovery,
            })

            if (i + 1) % 50 == 0:
                recs = [r["recovery"] for r in records if r["recovery"] is not None]
                if recs:
                    arr = np.array(recs)
                    print(f"  [{i+1}/{len(val_ds)}] "
                          f"recovery median={np.median(arr):.3f}  mean={arr.mean():.3f}  "
                          f"P[>=0.8]={(arr>=0.8).mean():.3f}  "
                          f"valid={len(recs)}/{len(records)}")

    # ── Summary ──────────────────────────────────────────────────────
    valid = [r for r in records if r["recovery"] is not None]
    rec_arr = np.array([r["recovery"] for r in valid], dtype=np.float64)
    full_d = np.array([r["delta_r_full"] for r in records], dtype=np.float64)
    rem_d  = np.array([r["delta_r_remove"] for r in records], dtype=np.float64)
    seed_sz = np.array([r["seed_size"] for r in records], dtype=np.float64)
    rem_sz = np.array([r["rem_size"] for r in records], dtype=np.float64)
    full_sz = np.array([r["full_size"] for r in records], dtype=np.float64)
    seed_cov = np.array([r["seed_cov"] for r in records], dtype=np.float64)
    rem_cov = np.array([r["rem_cov"] for r in records], dtype=np.float64)
    full_cov = np.array([r["full_cov"] for r in records], dtype=np.float64)

    summary = {
        "n_polygons": len(records),
        "n_valid_recovery": len(valid),
        "n_full_no_improvement": len(records) - len(valid),
        "recovery_mean": float(rec_arr.mean()) if len(rec_arr) else None,
        "recovery_median": float(np.median(rec_arr)) if len(rec_arr) else None,
        "recovery_p25": float(np.percentile(rec_arr, 25)) if len(rec_arr) else None,
        "recovery_p75": float(np.percentile(rec_arr, 75)) if len(rec_arr) else None,
        "recovery_p10": float(np.percentile(rec_arr, 10)) if len(rec_arr) else None,
        "recovery_p90": float(np.percentile(rec_arr, 90)) if len(rec_arr) else None,
        "frac_recovery_ge_0.5": float((rec_arr >= 0.5).mean()) if len(rec_arr) else None,
        "frac_recovery_ge_0.8": float((rec_arr >= 0.8).mean()) if len(rec_arr) else None,
        "frac_recovery_ge_1.0": float((rec_arr >= 1.0).mean()) if len(rec_arr) else None,
        "delta_r_full_mean":   float(full_d.mean()),
        "delta_r_remove_mean": float(rem_d.mean()),
        "size_seed_mean":   float(seed_sz.mean()),
        "size_remove_mean": float(rem_sz.mean()),
        "size_full_mean":   float(full_sz.mean()),
        "cov_seed_mean":   float(seed_cov.mean()),
        "cov_remove_mean": float(rem_cov.mean()),
        "cov_full_mean":   float(full_cov.mean()),
        "checkpoint": args.checkpoint,
        "val_dir": val_dir,
        "ls_params": {"tau": args.tau, "lam": args.lam,
                      "tau_penalty": args.tau_penalty,
                      "disc_vis_samples": args.disc_vis_samples,
                      "max_iter": args.ls_max_iter,
                      "monotone_coverage": True},
        "wall_time_s": time.perf_counter() - t_total,
    }

    print()
    print("=" * 64)
    print(f"  REMOVE-only ablation  ({len(records)} polygons, {len(valid)} valid for recovery)")
    print("=" * 64)
    print(f"  recovery   mean={summary['recovery_mean']:.3f}   median={summary['recovery_median']:.3f}")
    print(f"             p10={summary['recovery_p10']:.3f}   p25={summary['recovery_p25']:.3f}   "
          f"p75={summary['recovery_p75']:.3f}   p90={summary['recovery_p90']:.3f}")
    print(f"  P[recovery >= 0.50] = {summary['frac_recovery_ge_0.5']:.3f}")
    print(f"  P[recovery >= 0.80] = {summary['frac_recovery_ge_0.8']:.3f}")
    print(f"  P[recovery >= 1.00] = {summary['frac_recovery_ge_1.0']:.3f}   (REMOVE alone matches/beats full LS)")
    print()
    print(f"  Δr   mean   remove={summary['delta_r_remove_mean']:+.4f}   full={summary['delta_r_full_mean']:+.4f}")
    print(f"  |S|  mean   seed={summary['size_seed_mean']:.2f} "
          f"-> remove={summary['size_remove_mean']:.2f}  "
          f"-> full={summary['size_full_mean']:.2f}")
    print(f"  cov  mean   seed={summary['cov_seed_mean']:.3f} "
          f"-> remove={summary['cov_remove_mean']:.3f}  "
          f"-> full={summary['cov_full_mean']:.3f}")
    print(f"  full LS made no improvement on {summary['n_full_no_improvement']} polygons")
    print(f"  wall  {summary['wall_time_s']:.1f}s")
    print()

    # ── Verdict line ─────────────────────────────────────────────────
    med = summary["recovery_median"] or 0.0
    if med >= 0.80:
        verdict = "GREEN  -> single-head pruner is justified; defer swap/add to v2"
    elif med < 0.50:
        verdict = "RED    -> swap/add does real work; need a 2-head architecture"
    else:
        verdict = "YELLOW -> ambiguous; inspect distribution + cov/|S| trade"
    print(f"  VERDICT: {verdict}")

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO_ROOT, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"\n  saved -> {out_path}")


if __name__ == "__main__":
    main()
