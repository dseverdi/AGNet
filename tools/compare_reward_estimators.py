#!/usr/bin/env python
"""compare_reward_estimators.py — does the PO/BT reward change under exact CGAL?

WHY THIS RATHER THAN A TRAINING RUN
    The released policy was trained with --fast-reward (discretised visibility,
    M=500). The natural way to ask "would exact CGAL have trained a different
    policy?" is to retrain, which costs hours per condition.

    But PO/BT does not consume the reward's magnitude. It consumes the *pairwise
    ordering* of rollouts: for every pair with u(pi_i) > u(pi_j), a Bradley-Terry
    term pushes one up and the other down. So a reward estimator only changes
    training to the extent that it changes which rollout of a pair looks better.

    That is directly measurable on the frozen policy in minutes: draw the same
    K rollouts training would have drawn, score each under BOTH estimators, and
    count how often the pairwise ordering flips.

WHAT IT REPORTS
    1. Coverage bias      mean(disc - exact), overall and binned by n and |S|.
                          The sign matters: disc-vis over-estimating coverage is
                          the dangerous direction, because the reward gates at
                          tau and a policy rewarded for reaching a gate it has
                          not reached learns to stop early.
    2. Reward agreement   Spearman correlation of the two reward vectors.
    3. PAIRWISE ORDERING  over all C(K,2) rollout pairs per polygon, the fraction
       AGREEMENT          where sign(r_i - r_j) matches. THIS IS THE HEADLINE:
                          it is the fraction of BT gradient terms that would have
                          pointed the same way under exact CGAL.
    4. Gate disagreement  how often the two estimators straddle tau (one says the
                          rollout cleared the coverage gate, the other does not).
                          Threshold effects are outsized because the reward is
                          capped at tau and penalised linearly below it.

INTERPRETING IT
    High ordering agreement (say >95%) => the approximation is not steering
    training, disc-vis is validated, and no retraining is needed. Report the
    number and move on.
    Low agreement, or a large positive coverage bias that grows with |S| or n
    => the approximation IS shaping the policy, and the coverage tail may be
    partly a training-reward artefact rather than only decoder calibration.
    In that case run the short two-condition training comparison.

USAGE
    python tools/compare_reward_estimators.py                    # 200 polygons
    python tools/compare_reward_estimators.py --n-polygons 500 --out results/reward_estimator_agreement.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from dataset import Dataset, collate_fn                              # noqa: E402
from po_agp import create_agp_model, prepare_datasets                # noqa: E402
from utils import (                                                  # noqa: E402
    evaluate_polygon_visibility_numpy_wo_gt,
    get_or_build_disc_vis,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt")
    p.add_argument("--n-polygons", type=int, default=200,
                   help="polygons from the TRAIN split (the distribution the "
                        "reward actually acted on during training)")
    p.add_argument("--rollouts", type=int, default=8, help="K, as in training")
    # Reward hyperparameters -- must match the released policy (Section 5.2).
    p.add_argument("--tau", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--tau-penalty", type=float, default=3.0)
    p.add_argument("--cap-coverage", action="store_true", default=True)
    p.add_argument("--disc-vis-samples", type=int, default=500)
    p.add_argument("--embedding-size", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--n-glimpses", type=int, default=1)
    p.add_argument("--tanh-exploration", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=str,
                   default="results/reward_estimator_agreement.json")
    return p.parse_args()


def reward_from_coverage(cov: float, n_guards: int, n: int, *,
                         lam: float, tau: float, tau_penalty: float,
                         cap: bool) -> float:
    """The PO reward of Eq. (3), given a coverage value from either estimator.

    Mirrors po_reward_smooth / po_reward_smooth_disc exactly; the only thing
    that differs between the two is where `cov` came from.
    """
    if n_guards == 0:
        return float(-tau_penalty)
    guard_ratio = n_guards / max(1, n)
    effective = min(cov, tau) if cap else cov
    return float(effective - lam * guard_ratio
                 - tau_penalty * max(0.0, tau - cov))


def disc_coverage(pts: np.ndarray, guards: list[int], name: str,
                  n_samples: int) -> float | None:
    disc = get_or_build_disc_vis(pts, name, n_samples=n_samples)
    if not disc.get("valid"):
        return None
    vm, M = disc["vis_matrix"], disc["n_samples"]
    covered = np.zeros(M, dtype=np.bool_)
    for v in guards:
        if 0 <= v < vm.shape[0]:
            np.bitwise_or(covered, vm[v], out=covered)
    return float(covered.sum()) / M


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset_path = os.getenv("DATASET_PATH")
    if not dataset_path:
        sys.exit("DATASET_PATH must be set (see .env)")

    train_ds, _ = prepare_datasets(
        os.path.join(dataset_path, "train"),
        os.path.join(dataset_path, "dev"),
        normalize=True,
    )
    ds = Dataset(train_ds.samples[: args.n_polygons])
    print(f"[data] {len(ds)} train polygons")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_agp_model(args.embedding_size, args.hidden_size,
                             args.n_glimpses, args.tanh_exploration,
                             True, 1.0).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # strict=False matches how build_ls_trajectories.py and train_set_predictor.py
    # load this same checkpoint: the current model carries later-added heads
    # (marg_cov_*) that the March policy predates. They are inert here because
    # cov_inject_active requires vis_matrices_list, and we decode with it None.
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
    inert = {"marg_cov_inject_gain", "marg_cov_head", "marg_cov_inject_proj"}
    surprising = [k for k in missing
                  if not any(tag in k for tag in inert)]
    if surprising or unexpected:
        print(f"[warn] unexpected state-dict mismatch: missing={surprising} "
              f"unexpected={list(unexpected)}")
    model.eval()
    print(f"[model] {args.checkpoint} (epoch {ck.get('epoch')}) on {device}")

    loader = torch.utils.data.DataLoader(
        ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    rows, pair_agree, pair_total = [], 0, 0
    gate_disagree = gate_total = 0
    skipped = 0

    with torch.no_grad():
        for batch_data, pad_mask, lens, names in loader:
            name = names[0]
            n = int(lens[0])
            pts = batch_data[0, :n].cpu().numpy()
            bd = batch_data.to(device)
            pm = pad_mask.to(device) if pad_mask is not None else None
            lt = torch.as_tensor(lens, device=device)

            # Draw the same K stochastic rollouts training would have drawn.
            K = args.rollouts
            idxs, _ = model(
                bd.repeat_interleave(K, dim=0),
                padding_mask=pm.repeat_interleave(K, dim=0) if pm is not None else None,
                lengths=lt.repeat_interleave(K),
                deterministic=False, no_eos=False,
            )

            r_disc, r_exact = [], []
            for k in range(K):
                guards = sorted({int(v) for v in idxs[k] if int(v) < n})
                cd = disc_coverage(pts, guards, name, args.disc_vis_samples)
                if cd is None:
                    continue
                try:
                    ce = evaluate_polygon_visibility_numpy_wo_gt(
                        pts, np.array(guards, dtype=np.int64), name) if guards else 0.0
                except Exception:
                    continue

                kw = dict(lam=args.lam, tau=args.tau,
                          tau_penalty=args.tau_penalty, cap=args.cap_coverage)
                r_disc.append(reward_from_coverage(cd, len(guards), n, **kw))
                r_exact.append(reward_from_coverage(ce, len(guards), n, **kw))
                rows.append({"name": name, "n": n, "n_guards": len(guards),
                             "cov_disc": cd, "cov_exact": ce,
                             "gap": cd - ce})

                gate_total += 1
                if (cd >= args.tau) != (ce >= args.tau):
                    gate_disagree += 1

            # Pairwise ordering: the quantity BT actually consumes.
            for i in range(len(r_disc)):
                for j in range(i + 1, len(r_disc)):
                    d_i, d_j = r_disc[i], r_disc[j]
                    e_i, e_j = r_exact[i], r_exact[j]
                    if d_i == d_j or e_i == e_j:
                        continue
                    pair_total += 1
                    if (d_i > d_j) == (e_i > e_j):
                        pair_agree += 1

            if len(rows) and len(rows) % 200 == 0:
                print(f"  ...{len(rows)} rollouts scored")

    if not rows:
        sys.exit("no rollouts scored -- check DATASET_PATH and the checkpoint")

    gaps = np.array([r["gap"] for r in rows])
    ns = np.array([r["n"] for r in rows])
    gs = np.array([r["n_guards"] for r in rows])
    rd = np.array([r["cov_disc"] for r in rows])
    re_ = np.array([r["cov_exact"] for r in rows])

    def corr(a, b):
        try:
            from scipy.stats import spearmanr
            return float(spearmanr(a, b).correlation)
        except Exception:
            return float(np.corrcoef(a, b)[0, 1])

    summary = {
        "n_polygons": len(ds),
        "n_rollouts_scored": len(rows),
        "rollouts_per_polygon": args.rollouts,
        "reward": {"tau": args.tau, "lam": args.lam,
                   "tau_penalty": args.tau_penalty,
                   "disc_vis_samples": args.disc_vis_samples},
        "coverage_bias_disc_minus_exact": {
            "mean": float(gaps.mean()), "median": float(np.median(gaps)),
            "std": float(gaps.std()),
            "frac_disc_optimistic": float((gaps > 0).mean()),
            "p95_abs": float(np.percentile(np.abs(gaps), 95)),
        },
        "coverage_corr_spearman": corr(rd, re_),
        "gap_vs_n_spearman": corr(gaps, ns),
        "gap_vs_guardcount_spearman": corr(gaps, gs),
        "pairwise_ordering_agreement": (
            float(pair_agree / pair_total) if pair_total else None),
        "pairwise_comparisons": pair_total,
        "gate_disagreement_rate": (
            float(gate_disagree / gate_total) if gate_total else None),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=1)

    b = summary["coverage_bias_disc_minus_exact"]
    print("\n" + "=" * 62)
    print("  REWARD-ESTIMATOR AGREEMENT  (disc-vis vs exact CGAL)")
    print("=" * 62)
    print(f"  rollouts scored            {summary['n_rollouts_scored']}")
    print(f"  coverage bias (disc-exact) {b['mean']:+.4f} mean, {b['median']:+.4f} median")
    print(f"  disc-vis optimistic on     {b['frac_disc_optimistic']*100:.1f}% of rollouts")
    print(f"  coverage corr (Spearman)   {summary['coverage_corr_spearman']:.4f}")
    print(f"  gap vs n / vs guard count  {summary['gap_vs_n_spearman']:.3f} / "
          f"{summary['gap_vs_guardcount_spearman']:.3f}")
    print(f"  gate disagreement rate     {summary['gate_disagreement_rate']*100:.2f}%")
    print(f"  >> PAIRWISE ORDERING AGREEMENT: "
          f"{summary['pairwise_ordering_agreement']*100:.2f}%  "
          f"({summary['pairwise_comparisons']} pairs)")
    print("=" * 62)
    agree = summary["pairwise_ordering_agreement"]
    if agree is not None and agree >= 0.95:
        print("  => >=95% of BT gradient terms would point the same way under")
        print("     exact CGAL. The approximation is not steering training;")
        print("     disc-vis is validated and no retraining is needed.")
    else:
        print("  => a material share of BT terms would flip under exact CGAL.")
        print("     Worth running the two-condition short training comparison.")
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
