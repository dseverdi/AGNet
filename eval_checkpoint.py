#!/usr/bin/env python3
"""Evaluate a PO-AGP checkpoint with exact CGAL visibility on the val set."""

import argparse
import os
import sys
import json
import numpy as np
import torch

# Re-use the project infrastructure
from dataset import Dataset, agp_read_samples, collate_fn
from models import create_actor
from po_agp import evaluate_po, make_report, prepare_datasets, create_agp_model
from utils import prewarm_vis_cache

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "dataset", "AGPIL",
)


def main():
    p = argparse.ArgumentParser(description="Evaluate PO-AGP checkpoint")
    p.add_argument("checkpoint", type=str, help="Path to .pt checkpoint")
    p.add_argument("--val-dir", type=str,
                   default=os.path.join(DATASET_PATH, "dev"))
    p.add_argument("--sol-dir", type=str, default=None,
                   help="Solution dir (defaults to val-dir)")
    p.add_argument("--eval-k", type=int, default=-1,
                   help="Number of instances to evaluate (-1 = all)")
    p.add_argument("--K", type=int, default=8,
                   help="Stochastic rollouts")
    p.add_argument("--aug", type=int, default=1, choices=[1, 8])
    p.add_argument("--disc-vis-samples", type=int, default=0,
                   help="0 = exact CGAL visibility (default); >0 = fast disc proxy")
    p.add_argument("--local-search", action="store_true", default=False,
                   help="Refine best solution via LS post-processing")
    p.add_argument("--ls-max-iter", type=int, default=50,
                   help="LS max iterations per instance")
    p.add_argument("--verbose", "-v", action="store_true", default=False,
                   help="Print per-instance progress")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output", type=str, default=None,
                   help="Save JSON report to this path")
    args = p.parse_args()

    if args.sol_dir is None:
        args.sol_dir = args.val_dir

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Determine model params
    params = ckpt.get("checkpoint_params", ckpt.get("params", {}))
    ckpt_args = ckpt.get("args", {})

    embedding_size = params.get("embedding_size", ckpt_args.get("embedding_size", 128))
    hidden_size = params.get("hidden_size", ckpt_args.get("hidden_size", 128))
    n_glimpses = params.get("n_glimpses", ckpt_args.get("n_glimpses", 1))
    temperature = params.get("temperature", ckpt_args.get("temperature", 1.0))
    tanh_exploration = params.get("tanh_exploration",
                                  ckpt_args.get("tanh_exploration", 10.0))

    epoch = ckpt.get("epoch", "?")
    print(f"  epoch={epoch}  embed={embedding_size}")

    # Build model
    model = create_agp_model(
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        n_glimpses=n_glimpses,
        tanh_exploration=tanh_exploration,
        use_tanh=True,
        temperature=temperature,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load dataset
    _, dataset = prepare_datasets(
        os.path.join(DATASET_PATH, "train"), args.val_dir, normalize=True,
    )
    n_eval = len(dataset) if args.eval_k < 0 else min(args.eval_k, len(dataset))
    if n_eval < len(dataset):
        dataset = Dataset(dataset.samples[:n_eval])

    print(f"Evaluating {n_eval} instances | K={args.K} | aug={args.aug} | "
          f"disc_vis={'exact' if args.disc_vis_samples == 0 else args.disc_vis_samples}"
          f" | LS={'on' if args.local_search else 'off'}")

    # Pre-build CGAL visibility caches in parallel (exact mode only)
    if args.disc_vis_samples == 0:
        prewarm_vis_cache(dataset, verbose=True)

    # Reward fn for LS (if needed)
    tau = params.get("tau", 0.99)
    lam = params.get("reward_lambda", 1.0)

    from po_agp import po_reward_smooth
    def reward_fn(points, solution, name, length=None):
        return po_reward_smooth(
            points, solution, name, length=length,
            lam=lam, tau=tau, tau_penalty=3.0, cap_at_tau=False,
        )

    per_instance = evaluate_po(
        model, dataset, args.sol_dir,
        K=args.K,
        aug_factor=args.aug,
        reward_fn=reward_fn,
        eval_k=n_eval,
        local_search=args.local_search,
        ls_max_iter=args.ls_max_iter,
        no_eos=False,
        tau=tau,
        disc_vis_samples=args.disc_vis_samples,
    )

    for idx, rec in enumerate(per_instance):
        if args.verbose:
            ls_str = ""
            if rec.get("ls_pre_guards") is not None:
                ls_str = (f"  LS: {rec['ls_pre_guards']}g/{rec.get('ls_pre_cov',0):.3f}cov"
                          f" -> {rec['ls_post_guards']}g/{rec.get('ls_post_cov',0):.3f}cov")
            opt_str = f"  opt={rec.get('opt_size', '?')}" if rec.get('opt_size') else ""
            ar_str = f"  |S|/opt={rec.get('approx_ratio', 0):.2f}" if rec.get('approx_ratio') else ""
            print(f"  [{idx+1}/{n_eval}] {rec['name']}  n={rec['n']}  "
                  f"greedy={rec.get('guards_greedy','?')}g/{rec.get('coverage_greedy',0):.3f}cov  "
                  f"best={rec['guards']}g/{rec.get('coverage',0):.3f}cov"
                  f"{ls_str}{opt_str}{ar_str}")

    # ── Summarise ──────────────────────────────────────────────────
    def _stats(vals, fmt=".4f"):
        """Return 'mean ± std' string."""
        if not vals:
            return "N/A"
        a = np.array(vals, dtype=float)
        return f"{a.mean():{fmt}} ± {a.std():{fmt}}"

    def _pct(count, total):
        return f"{count}/{total} ({100*count/max(1,total):.0f}%)"

    has_ls = any(r.get("ls_pre_guards") is not None for r in per_instance)

    # Collect arrays
    covs_g   = [r["coverage_greedy"] for r in per_instance if r.get("coverage_greedy") is not None]
    gr_g     = [r["guard_ratio_greedy"] for r in per_instance]
    ar_g     = [r["guards_greedy"] / r["opt_size"] for r in per_instance if r.get("opt_size")]
    sizes_g  = [r["guards_greedy"] for r in per_instance]

    covs_f   = [r["coverage"] for r in per_instance if r.get("coverage") is not None]
    gr_f     = [r["guard_ratio"] for r in per_instance]
    ar_f     = [r["approx_ratio"] for r in per_instance if r.get("approx_ratio") is not None]
    sizes_f  = [r["guards"] for r in per_instance]

    feas_g   = sum(1 for c in covs_g if c >= tau)
    feas_f   = sum(1 for c in covs_f if c >= tau)

    # Pre-LS (stochastic best-of-K, before LS)
    if has_ls:
        covs_pre  = [r["ls_pre_cov"] for r in per_instance if r.get("ls_pre_cov") is not None]
        gr_pre    = [r["ls_pre_guards"] / max(1, r["n"]) for r in per_instance if r.get("ls_pre_guards") is not None]
        ar_pre    = [r["ls_pre_guards"] / r["opt_size"] for r in per_instance if r.get("ls_pre_guards") is not None and r.get("opt_size")]
        sizes_pre = [r["ls_pre_guards"] for r in per_instance if r.get("ls_pre_guards") is not None]
        feas_pre  = sum(1 for c in covs_pre if c >= tau)
        ls_removes = [r.get("ls_removes", 0) for r in per_instance if "ls_removes" in r]
        ls_adds    = [r.get("ls_adds", 0) for r in per_instance if "ls_adds" in r]
        ls_swaps   = [r.get("ls_swaps", 0) for r in per_instance if "ls_swaps" in r]
        ls_delta   = [r.get("ls_delta_guards", 0) for r in per_instance if "ls_delta_guards" in r]

    W = 62
    print(f"\n{'=' * W}")
    print(f"  Checkpoint : {args.checkpoint}  (epoch {epoch})")
    print(f"  Eval mode  : {'EXACT CGAL' if args.disc_vis_samples == 0 else f'disc_vis({args.disc_vis_samples})'}")
    print(f"  Instances  : {len(per_instance)}   K={args.K}  aug={args.aug}  τ={tau}")
    print(f"{'=' * W}")

    hdr = f"  {'':18s} {'|S|/opt':>14s} {'|S|/n':>14s} {'cov':>14s} {'cov≥τ':>8s}"
    sep = f"  {'─'*18} {'─'*14} {'─'*14} {'─'*14} {'─'*8}"
    print(hdr)
    print(sep)

    print(f"  {'Greedy':18s} {_stats(ar_g, '.3f'):>14s} {_stats(gr_g):>14s} {_stats(covs_g):>14s} {_pct(feas_g, len(covs_g)):>8s}")
    if has_ls:
        print(f"  {'Pre-LS best-of-K':18s} {_stats(ar_pre, '.3f'):>14s} {_stats(gr_pre):>14s} {_stats(covs_pre):>14s} {_pct(feas_pre, len(covs_pre)):>8s}")
    print(f"  {'Final (post-LS)' if has_ls else 'Best-of-K':18s} {_stats(ar_f, '.3f'):>14s} {_stats(gr_f):>14s} {_stats(covs_f):>14s} {_pct(feas_f, len(covs_f)):>8s}")

    print(sep)
    # Median & extremes for |S|/opt
    if ar_f:
        a = np.array(ar_f)
        print(f"  |S|/opt  median={np.median(a):.3f}  min={a.min():.3f}  max={a.max():.3f}  "
              f"<1.0={_pct(int((a < 1.0).sum()), len(a))}  <1.5={_pct(int((a < 1.5).sum()), len(a))}")

    if has_ls:
        print(f"\n  LS moves (mean):  removes={np.mean(ls_removes):.1f}  adds={np.mean(ls_adds):.1f}  "
              f"swaps={np.mean(ls_swaps):.1f}  Δguards={np.mean(ls_delta):+.1f}")

    # Δ summary: greedy → final
    if covs_g and covs_f:
        d_cov = np.mean(covs_f) - np.mean(covs_g)
        d_gr  = np.mean(gr_f) - np.mean(gr_g)
        d_ar  = (np.mean(ar_f) - np.mean(ar_g)) if ar_g and ar_f else 0
        sign = lambda v: f"+{v:.4f}" if v >= 0 else f"{v:.4f}"
        print(f"\n  Δ greedy→final:  Δcov={sign(d_cov)}  Δ|S|/n={sign(d_gr)}  Δ|S|/opt={sign(d_ar)}")

    print(f"{'=' * W}")

    # Optionally save JSON report
    if args.output:
        report = {
            "checkpoint": args.checkpoint,
            "epoch": epoch,
            "eval_mode": "exact" if args.disc_vis_samples == 0 else f"disc_vis_{args.disc_vis_samples}",
            "n_instances": len(per_instance),
            "K": args.K,
            "aug_factor": args.aug,
            "tau": tau,
            "greedy": {
                "coverage_mean": float(np.mean(covs_g)) if covs_g else None,
                "coverage_std": float(np.std(covs_g)) if covs_g else None,
                "guard_ratio_mean": float(np.mean(gr_g)) if gr_g else None,
                "approx_ratio_mean": float(np.mean(ar_g)) if ar_g else None,
                "feasible": feas_g,
            },
            "final": {
                "coverage_mean": float(np.mean(covs_f)) if covs_f else None,
                "coverage_std": float(np.std(covs_f)) if covs_f else None,
                "guard_ratio_mean": float(np.mean(gr_f)) if gr_f else None,
                "approx_ratio_mean": float(np.mean(ar_f)) if ar_f else None,
                "approx_ratio_median": float(np.median(ar_f)) if ar_f else None,
                "feasible": feas_f,
            },
            "per_instance": per_instance,
        }
        if has_ls:
            report["pre_ls"] = {
                "coverage_mean": float(np.mean(covs_pre)) if covs_pre else None,
                "guard_ratio_mean": float(np.mean(gr_pre)) if gr_pre else None,
                "approx_ratio_mean": float(np.mean(ar_pre)) if ar_pre else None,
                "feasible": feas_pre,
            }
            report["ls_moves"] = {
                "removes_mean": float(np.mean(ls_removes)),
                "adds_mean": float(np.mean(ls_adds)),
                "swaps_mean": float(np.mean(ls_swaps)),
                "delta_guards_mean": float(np.mean(ls_delta)),
            }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved -> {args.output}")


if __name__ == "__main__":
    main()
