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
          f"disc_vis={'exact' if args.disc_vis_samples == 0 else args.disc_vis_samples}")

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
        no_eos=False,
        tau=tau,
        disc_vis_samples=args.disc_vis_samples,
    )

    # Summarise
    covs_g = [x["coverage_greedy"] for x in per_instance if x.get("coverage_greedy") is not None]
    gr_g   = [x["guard_ratio_greedy"] for x in per_instance]
    covs_s = [x["coverage_stoch"] for x in per_instance if x.get("coverage_stoch") is not None]
    gr_s   = [x["guard_ratio_stoch"] for x in per_instance]
    rats   = [x["approx_ratio"] for x in per_instance if x.get("approx_ratio") is not None]
    sizes_g = [x["guards_greedy"] for x in per_instance]
    sizes_s = [x["guards_stoch"] for x in per_instance]
    sizes_best = [x["guards"] for x in per_instance]
    opt_sizes = [x["opt_size"] for x in per_instance if x.get("opt_size") is not None]

    print(f"\n{'='*60}")
    print(f"Checkpoint: {args.checkpoint}  (epoch {epoch})")
    print(f"Eval mode:  {'EXACT CGAL' if args.disc_vis_samples == 0 else f'disc_vis({args.disc_vis_samples})'}")
    print(f"Instances:  {len(per_instance)}")
    print(f"{'='*60}")
    if covs_g:
        print(f"  greedy | cov={np.mean(covs_g):.4f} | |S|/n={np.mean(gr_g):.4f} | |S|_mean={np.mean(sizes_g):.1f}")
    if covs_s:
        print(f"  stoch  | cov={np.mean(covs_s):.4f} | |S|/n={np.mean(gr_s):.4f} | |S|_mean={np.mean(sizes_s):.1f}")
    if sizes_best:
        print(f"  best   | |S|_mean={np.mean(sizes_best):.1f} | |S|/n={np.mean([x['guard_ratio'] for x in per_instance]):.4f}")
    if rats:
        print(f"  |S|/opt mean={np.mean(rats):.3f}  median={np.median(rats):.3f}  "
              f"max={np.max(rats):.3f}  <1.5={np.mean(np.array(rats)<1.5)*100:.1f}%")
    if opt_sizes:
        print(f"  opt |S| mean={np.mean(opt_sizes):.1f}")
    print(f"{'='*60}")

    # Optionally save JSON report
    if args.output:
        report = {
            "checkpoint": args.checkpoint,
            "epoch": epoch,
            "eval_mode": "exact" if args.disc_vis_samples == 0 else f"disc_vis_{args.disc_vis_samples}",
            "n_instances": len(per_instance),
            "K": args.K,
            "aug_factor": args.aug,
            "coverage_greedy_mean": float(np.mean(covs_g)) if covs_g else None,
            "coverage_stoch_mean": float(np.mean(covs_s)) if covs_s else None,
            "guard_ratio_greedy_mean": float(np.mean(gr_g)) if gr_g else None,
            "guard_ratio_stoch_mean": float(np.mean(gr_s)) if gr_s else None,
            "approx_ratio_mean": float(np.mean(rats)) if rats else None,
            "approx_ratio_median": float(np.median(rats)) if rats else None,
            "per_instance": per_instance,
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved -> {args.output}")


if __name__ == "__main__":
    main()
