#!/usr/bin/env python3
"""pomo_agp.py — POMO-style NCO training for the Art Gallery Problem (AGP).

Implements the key ideas from POMO (Kwon et al., NeurIPS 2020) adapted
for the vertex-guard AGP, plus battle-tested features from rl_agp_v2:

  1. **Shared baseline** — For each polygon, K stochastic rollouts are
     produced.  The REINFORCE baseline is the mean reward across the K
     rollouts *for that instance*.  This is instance-specific, unbiased,
     and empirically very low-variance — far better than a global EMA.

  2. **All rollouts contribute to the gradient** — Unlike best-of-K
     (which back-propagates only through the single best trajectory),
     every rollout receives an advantage signal.  Good rollouts are
     reinforced; bad ones are suppressed.  This yields K× more gradient
     signal per training step.

  3. **Smooth reward** —
         r = coverage − λ · (|S| / n) − π · max(0, τ − coverage)
     Monotonically improving with solution quality.  Avoids the mode-
     switching trap of the hysteresis reward where crossing the coverage
     threshold can *decrease* reward and fool the shared baseline.

  4. **Augmented inference** — 8 dihedral-group augmentations of the
     polygon coordinates (reflections + 90° rotations, as in POMO) give
     8K candidate solutions per instance at test time — essentially free
     extra quality.

  Features ported from rl_agp_v2:
  - JSON config file (--config)
  - EOS logit bias (--eos-bias-init, --eos-bias-learnable)
  - Entropy bonus (--entropy-weight)
  - Per-trajectory log-prob normalization by decoded length
  - Dual eval: greedy + stochastic per epoch
  - Best-checkpoint saving (best greedy cov, best stochastic cov)
  - Early-stop on collapse detection
  - Optimizer state resume from checkpoint

Usage:
    python pomo_agp.py --pomo-k 8 --epochs 50 --batch-size 64 \\
        --reward-lambda 0.24 --aug-factor 8 --verbose

    python pomo_agp.py --config configs/pomo_agp_train.json --verbose

Reference:
    Kwon et al., "POMO: Policy Optimization with Multiple Optima for
    Reinforcement Learning", NeurIPS 2020, arXiv:2010.16011.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import argparse
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment]

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

# ── Local imports ──────────────────────────────────────────────────
from dataset import Dataset, agp_read_samples, collate_fn
from models import create_actor
from utils import evaluate_polygon_visibility_numpy_wo_gt
from eval_reporting import make_report


# ===================================================================
#  0.  Shared infrastructure (avoid cross-file version mismatches)
# ===================================================================

def _load_json_config(path: str) -> Dict:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object at the top level.")
    return data


def _coerce_value(value, default):
    if default is None:
        return value
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, str):
        return str(value)
    return value


def _get_explicit_args(parser: argparse.ArgumentParser, argv: List[str]) -> set:
    explicit = set()
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            if opt in argv:
                explicit.add(action.dest)
                break
            prefix = opt + "="
            if any(arg.startswith(prefix) for arg in argv):
                explicit.add(action.dest)
                break
    return explicit


def _apply_config_to_args(args, config: Dict, defaults: Dict,
                          keys: List[str], explicit_args: set) -> None:
    for key in keys:
        if key in config and key not in explicit_args and getattr(args, key) == defaults.get(key):
            setattr(args, key, _coerce_value(config[key], defaults.get(key)))


class BucketBatchSampler(Sampler):
    """Batch sampler that groups samples of similar length into the same batch."""
    def __init__(self, lengths, batch_size, shuffle=True, drop_last=False, bucket_size=10):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_size = bucket_size
        self.sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self.buckets = []
        for i in range(0, len(self.sorted_indices), bucket_size):
            self.buckets.append(self.sorted_indices[i:i + bucket_size])

    def __iter__(self):
        buckets = self.buckets.copy()
        if self.shuffle:
            np.random.shuffle(buckets)
        for bucket in buckets:
            if self.shuffle:
                np.random.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
                    yield batch

    def __len__(self):
        total = 0
        for bucket in self.buckets:
            n = len(bucket)
            total += n // self.batch_size
            if not self.drop_last and n % self.batch_size != 0:
                total += 1
        return total


def get_lengths_from_dataset(dataset):
    return [sample[0].shape[0] for sample in dataset]


def prepare_datasets(train_path, val_path, normalize=True):
    def get_pol_files(path):
        if os.path.isdir(path):
            return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pol')]
        elif os.path.isfile(path) and path.endswith('.pol'):
            return [path]
        else:
            raise ValueError(f"Provided path {path} is neither a .pol file nor a directory containing .pol files.")
    agp_train_paths = get_pol_files(train_path)
    agp_val_paths = get_pol_files(val_path)
    print(f"Found {len(agp_train_paths)} training and {len(agp_val_paths)} validation AGP .pol files.")
    train_samples = agp_read_samples(agp_train_paths, normalize=normalize)
    val_samples = agp_read_samples(agp_val_paths, normalize=normalize)
    return Dataset(train_samples), Dataset(val_samples)


def create_agp_model(embedding_size, hidden_size, n_glimpses, tanh_exploration,
                     use_tanh, reward_fn, temperature,
                     eos_logit_bias_init=None, eos_logit_bias_learnable=False):
    return create_actor(
        embedding_size, hidden_size, None, n_glimpses,
        tanh_exploration, use_tanh, "Bahdanau", reward_fn,
        temperature=temperature,
        eos_logit_bias_init=eos_logit_bias_init,
        eos_logit_bias_learnable=eos_logit_bias_learnable,
    )


def _read_opt_solution(sol_dir: str, name: str):
    base_name = os.path.splitext(os.path.basename(name))[0]
    opt_sol_path = os.path.join(sol_dir, f"{base_name}.solution")
    try:
        with open(opt_sol_path, 'r') as f:
            lines = f.read().splitlines()
            if len(lines) >= 2:
                return [int(x) for x in lines[1].split()]
    except Exception:
        return None
    return None


# ===================================================================
#  1.  Reward functions
# ===================================================================

def pomo_reward_smooth(
    points: np.ndarray,
    solution,
    name: str,
    length: int | None = None,
    *,
    lam: float = 0.2,
    tau: float = 0.99,
    tau_penalty: float = 5.0,
    cap_at_tau: bool = False,
) -> float:
    """Smooth, monotone reward for POMO training.

    Default mode:
        r = coverage − λ · (|S|/n) − π · max(0, τ − coverage)

    Capped mode (cap_at_tau=True):
        r = min(coverage, τ) − λ · (|S|/n) − π · max(0, τ − coverage)

    With cap_at_tau, once coverage ≥ τ the reward becomes
        r = τ − λ · (|S|/n)
    so extra coverage above the threshold yields *zero* marginal
    reward.  The only way to increase reward is to reduce |S|.
    This prevents the model from inflating the guard set to chase
    marginal coverage improvements.

    Args:
        points:      (N, 2) polygon vertices.
        solution:    Guard indices (list/array of ints).
        name:        Instance identifier (for the visibility cache).
        length:      Number of real vertices (if points is padded).
        lam:         Weight for the guard-sparsity penalty.
        tau:         Coverage feasibility threshold.
        tau_penalty: Linear penalty weight for coverage below tau.
        cap_at_tau:  If True, cap coverage reward at tau.

    Returns:
        Scalar reward (float).
    """
    n = length if length else len(points)
    sol = np.asarray(solution, dtype=np.int64)
    if len(sol) == 0:
        return float(-tau_penalty)  # empty guard set -> worst reward

    pts = points[:n] if length else points
    try:
        coverage = evaluate_polygon_visibility_numpy_wo_gt(pts, sol, name)
    except Exception:
        coverage = 0.0
    guard_ratio = len(sol) / max(1, n)

    effective_cov = min(coverage, tau) if cap_at_tau else coverage
    r = effective_cov - lam * guard_ratio
    deficit = max(0.0, tau - coverage)
    r -= tau_penalty * deficit
    return float(r)


def pomo_reward_smooth_meta(
    points: np.ndarray,
    solution,
    name: str,
    length: int | None = None,
    *,
    lam: float = 0.2,
    tau: float = 0.99,
    tau_penalty: float = 5.0,
    cap_at_tau: bool = False,
) -> tuple[float, dict]:
    """Same as pomo_reward_smooth but also returns a metadata dict."""
    n = length if length else len(points)
    sol = np.asarray(solution, dtype=np.int64)
    if len(sol) == 0:
        return float(-tau_penalty), {
            "coverage": 0.0,
            "num_guards": 0,
            "guard_ratio": 0.0,
            "reward": float(-tau_penalty),
            "regime": "empty",
        }
    pts = points[:n] if length else points
    try:
        coverage = evaluate_polygon_visibility_numpy_wo_gt(pts, sol, name)
    except Exception:
        coverage = 0.0
    guard_ratio = len(sol) / max(1, n)
    effective_cov = min(coverage, tau) if cap_at_tau else coverage
    r = effective_cov - lam * guard_ratio - tau_penalty * max(0.0, tau - coverage)
    regime = "feasible" if coverage >= tau else "infeasible"
    return float(r), {
        "coverage": float(coverage),
        "num_guards": len(sol),
        "guard_ratio": float(guard_ratio),
        "reward": float(r),
        "regime": regime,
    }


# ===================================================================
#  2.  Augmentation utilities  (dihedral group of the square)
# ===================================================================

N_AUGMENTATIONS = 8


def augment_xy(xy: torch.Tensor, aug_idx: int) -> torch.Tensor:
    """Apply one of 8 dihedral-group transforms to 2-D coordinates.

    Assumes coordinates are in [0, 1] (the case after our min-max
    normalization).  The 8 transforms are: identity, 3 rotations
    (90/180/270 deg) and 4 reflections -- exactly the symmetry group of
    the unit square, matching the POMO augmentations for TSP/CVRP.

    The vertex ordering is unchanged, so guard indices produced by the
    model on augmented coordinates are valid on the original polygon.
    """
    x, y = xy[..., 0], xy[..., 1]
    if   aug_idx == 0: ax, ay = x, y               # identity
    elif aug_idx == 1: ax, ay = 1 - x, y            # reflect x
    elif aug_idx == 2: ax, ay = x, 1 - y            # reflect y
    elif aug_idx == 3: ax, ay = 1 - x, 1 - y        # rotate 180
    elif aug_idx == 4: ax, ay = y, x                 # transpose
    elif aug_idx == 5: ax, ay = 1 - y, x             # rotate 90 CW
    elif aug_idx == 6: ax, ay = y, 1 - x             # rotate 270 CW
    elif aug_idx == 7: ax, ay = 1 - y, 1 - x         # anti-transpose
    else:
        raise ValueError(f"aug_idx must be 0-7, got {aug_idx}")
    return torch.stack([ax, ay], dim=-1)


# ===================================================================
#  3.  POMO rollouts (batched, length-normalized log-probs)
# ===================================================================

_budget_logged_once = True   # one-shot diagnostic flag

def _pomo_rollouts(
    model: torch.nn.Module,
    batch_data: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    batch_names: List[str],
    reward_fn: Callable,
    K: int,
    episode_log: bool = False,
    normalize_log_probs: bool = True,
    max_guard_ratio: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run K stochastic rollouts per instance.

    The batch is expanded K-fold along dim 0, processed in a *single*
    forward pass, then reshaped to (B, K).

    Log-prob normalization: divides each trajectory's sum-log-prob by its
    decoded length (number of selected guards).  This prevents long
    rollouts (many guards) from dominating the gradient over short ones,
    which is critical for variable-length AGP outputs.

    Returns
    -------
    rewards :       (B, K) float tensor -- non-differentiable.
    log_probs :     (B, K) float tensor -- length-normalized, for REINFORCE.
    log_probs_raw : (B, K) float tensor -- un-normalized sum log-probs,
                    for entropy bonus (so EOS confidence is not penalized).
    """
    B = batch_data.size(0)
    device = batch_data.device

    # -- expand the batch K times --
    exp_data    = batch_data.repeat_interleave(K, dim=0)   # (B*K, N, 2)
    exp_mask    = mask.repeat_interleave(K, dim=0)          # (B*K, N)
    exp_lengths = lengths.repeat_interleave(K)              # (B*K,)

    # -- single forward pass --
    # Compute per-sample guard budget from Art Gallery Theorem ratio.
    # Each polygon gets its own ⌊n_i * ratio⌋ budget.
    global _budget_logged_once
    if max_guard_ratio is not None:
        max_decode_steps = (exp_lengths.float() * max_guard_ratio).clamp(min=1).long()
        if _budget_logged_once:
            _budget_logged_once = False
            print(f"[budget] max_guard_ratio={max_guard_ratio}, "
                  f"budgets min={max_decode_steps.min().item()} "
                  f"max={max_decode_steps.max().item()} "
                  f"(lengths min={exp_lengths.min().item()} "
                  f"max={exp_lengths.max().item()})")
    else:
        max_decode_steps = None
        if _budget_logged_once:
            _budget_logged_once = False
            print("[budget] WARNING: max_guard_ratio is None — no budget enforced!")

    all_idxs, all_log_probs_raw = model(
        exp_data, padding_mask=exp_mask, lengths=exp_lengths,
        deterministic=False, max_decode_steps=max_decode_steps,
    )
    # all_log_probs_raw: (B*K,), all_idxs: list of B*K index lists

    # -- length-normalize log-probs for REINFORCE (v2 feature) --
    if normalize_log_probs:
        step_counts = torch.tensor(
            [max(1, len(idxs)) for idxs in all_idxs],
            dtype=torch.float32, device=device,
        )
        all_log_probs_norm = all_log_probs_raw / step_counts
    else:
        all_log_probs_norm = all_log_probs_raw

    # -- pre-extract per-instance numpy points --
    pts_cache: list[np.ndarray] = []
    for b in range(B):
        n = int(lengths[b].item())
        pts_cache.append(batch_data[b, :n].detach().cpu().numpy())

    # -- compute rewards --
    rewards_flat: list[float] = []
    for i in range(B * K):
        b_idx = i // K
        n = int(lengths[b_idx].item())
        sol = [idx for idx in all_idxs[i] if idx < n]
        r = reward_fn(pts_cache[b_idx], sol, batch_names[b_idx], length=n)
        rewards_flat.append(float(r))

    rewards        = torch.tensor(rewards_flat, dtype=torch.float32, device=device).view(B, K)
    log_probs      = all_log_probs_norm.view(B, K)
    log_probs_raw  = all_log_probs_raw.view(B, K)

    # -- optional episode logging (first 2 instances) --
    if episode_log and B > 0:
        for b in range(min(2, B)):
            best_k = int(rewards[b].argmax().item())
            n = int(lengths[b].item())
            sol = [idx for idx in all_idxs[b * K + best_k] if idx < n]
            cov = evaluate_polygon_visibility_numpy_wo_gt(
                pts_cache[b],
                np.array(sol, dtype=np.int64) if sol else np.array([], dtype=np.int64),
                batch_names[b],
            )
            print(
                f"  [pomo] {batch_names[b]} best-of-{K}: "
                f"cov={cov:.4f} guards={len(sol)} "
                f"r={rewards[b, best_k].item():.4f}"
            )

    return rewards, log_probs, log_probs_raw


# ===================================================================
#  4.  POMO training loop  (with v2 features)
# ===================================================================

def pomo_train(
    model: torch.nn.Module,
    dataset: Dataset,
    reward_fn: Callable,
    K: int = 8,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    checkpoint_params: dict | None = None,
    epoch_eval_fn: Callable | None = None,
    start_epoch: int = 0,
    episode_log: bool = False,
    max_grad_norm: float = 1.0,
    entropy_weight: float = 0.0,
    resume_optimizer_state: dict | None = None,
    save_best: bool = True,
    early_stop_on_collapse: bool = True,
    collapse_patience: int = 3,
    collapse_min_good_cov: float = 0.5,
    collapse_low_cov: float = 0.3,
    max_guard_ratio: float | None = None,
) -> None:
    """Train *model* with POMO (Algorithm 1 from the paper).

    For each mini-batch of B instances:
        1) Run K stochastic rollouts -> rewards  (B, K), log_probs (B, K)
        2) Shared baseline:  b_i = mean_k rewards_{i,k}
        3) Advantages:       A_{i,k} = rewards_{i,k} - b_i
        4) REINFORCE loss:   L = - mean_{i,k}[ A_{i,k} * log_probs_{i,k} ]
        5) Optional entropy bonus to prevent policy collapse.
        6) Gradient step with Adam + grad clipping.

    Additional v2 features:
        - Best-checkpoint saving (greedy + stochastic coverage)
        - Early-stop on coverage collapse after reaching good region
        - Optimizer state resume from checkpoint
    """
    if K < 2:
        print(
            "WARNING: K < 2 makes the shared baseline equal to the single "
            "reward, giving zero advantages and no gradient.  "
            "Setting K = max(K, 2)."
        )
        K = max(K, 2)

    print(
        f"\n{'='*60}\n"
        f"  POMO Training  |  {len(dataset)} instances  |  K={K}  |"
        f"  {epochs} epochs  |  bs={batch_size}  |"
        f"  ent={entropy_weight}\n"
        f"{'='*60}"
    )

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
        except Exception as e:
            print(f"[warn] could not restore optimizer state: {e}")

    device = next(model.parameters()).device

    lengths = get_lengths_from_dataset(dataset)
    sampler = BucketBatchSampler(
        lengths, batch_size, shuffle=True, bucket_size=10,
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_fn,
        pin_memory=True, num_workers=0,
    )

    last_ckpt_path: str | None = None
    best_cov_greedy  = -float("inf")
    best_cov_stoch   = -float("inf")
    collapse_streak  = 0
    reached_good     = False

    epoch_iter = (
        tqdm(range(epochs), desc="POMO epochs") if tqdm else range(epochs)
    )

    for epoch in epoch_iter:
        actual_epoch = start_epoch + epoch + 1
        total_loss    = 0.0
        total_reward  = 0.0
        n_instances   = 0

        batch_iter = (
            tqdm(loader, desc=f"Epoch {actual_epoch}", leave=False)
            if tqdm else loader
        )

        for batch_data, pad_mask, lens, names in batch_iter:
            batch_data = batch_data.to(device, non_blocking=True)
            pad_mask   = pad_mask.to(device, non_blocking=True)
            lens_t     = torch.tensor(lens, dtype=torch.long, device=device)

            # -- POMO rollouts --
            rewards, log_probs, log_probs_raw = _pomo_rollouts(
                model, batch_data, pad_mask, lens_t, names,
                reward_fn, K, episode_log=episode_log,
                max_guard_ratio=max_guard_ratio,
            )

            # -- shared baseline + advantages --
            baseline   = rewards.mean(dim=1, keepdim=True)   # (B, 1)
            advantages = rewards - baseline                   # (B, K)

            # -- per-instance advantage normalization --
            # Divide by std across the K rollouts for each instance.
            # Without this, easy instances (small advantage spread)
            # get drowned out by hard ones, and the gradient is
            # dominated by "just add more guards for hard polygons."
            # This is standard in POMO / modern PG implementations.
            adv_std = advantages.std(dim=1, keepdim=True) + 1e-8
            advantages = advantages / adv_std

            # -- REINFORCE loss (uses length-normalized log-probs) --
            pg_loss = -(log_probs * advantages).mean()

            # -- Entropy bonus --
            # Use per-step (length-normalized) log-probs to avoid an
            # artificial incentive for longer trajectories. With raw
            # summed log-probs, the model can reduce loss by increasing
            # decode length even when solution quality worsens.
            ent_bonus = entropy_weight * log_probs.mean() if entropy_weight > 0 else 0.0
            loss = pg_loss + ent_bonus

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_grad_norm,
                )
            optimizer.step()

            # -- bookkeeping --
            bs = batch_data.size(0)
            total_loss   += loss.item() * bs
            total_reward += rewards.max(dim=1).values.sum().item()
            n_instances  += bs

            if tqdm and hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(
                    loss=f"{loss.item():.4f}",
                    best_r=f"{rewards.max(dim=1).values.mean().item():.3f}",
                )

            del batch_data, pad_mask, lens_t, rewards, log_probs, log_probs_raw
            torch.cuda.empty_cache()

        # -- epoch summary --
        avg_loss = total_loss / max(1, n_instances)
        avg_best = total_reward / max(1, n_instances)
        print(
            f"Epoch {actual_epoch}/{start_epoch + epochs}  "
            f"loss={avg_loss:.4f}  best_reward_mean={avg_best:.3f}"
        )

        # -- epoch eval --
        eval_metrics = None
        if epoch_eval_fn is not None:
            eval_metrics = epoch_eval_fn(actual_epoch)
            model.train()

        # -- best-checkpoint saving (v2 feature) --
        if save_best and checkpoint_dir and isinstance(eval_metrics, dict):
            cov_g = eval_metrics.get("coverage_greedy_mean")
            cov_s = eval_metrics.get("coverage_stoch_mean")

            if cov_g is not None and cov_g > best_cov_greedy:
                best_cov_greedy = cov_g
                os.makedirs(checkpoint_dir, exist_ok=True)
                p_best_g = os.path.join(checkpoint_dir, "pomo_agp_best_greedy.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": actual_epoch,
                    "best_metric": "coverage_greedy_mean",
                    "best_value": float(cov_g),
                    "K": K,
                    "checkpoint_params": checkpoint_params,
                }, p_best_g)
                print(f"  Saved best-greedy: {p_best_g} (cov={cov_g:.3f})")

            if cov_s is not None and cov_s > best_cov_stoch:
                best_cov_stoch = cov_s
                os.makedirs(checkpoint_dir, exist_ok=True)
                p_best_s = os.path.join(checkpoint_dir, "pomo_agp_best_stoch.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": actual_epoch,
                    "best_metric": "coverage_stoch_mean",
                    "best_value": float(cov_s),
                    "K": K,
                    "checkpoint_params": checkpoint_params,
                }, p_best_s)
                print(f"  Saved best-stoch: {p_best_s} (cov={cov_s:.3f})")

            # -- early-stop on collapse (v2 feature) --
            if early_stop_on_collapse and cov_g is not None:
                if cov_g >= float(collapse_min_good_cov):
                    reached_good = True
                if reached_good and cov_g <= float(collapse_low_cov):
                    collapse_streak += 1
                else:
                    collapse_streak = 0
                if collapse_streak >= int(collapse_patience):
                    print(
                        f"[early-stop] collapse detected: greedy cov <= {collapse_low_cov} "
                        f"for {collapse_streak} epochs after reaching >= {collapse_min_good_cov}."
                    )
                    break

        # -- periodic checkpoint every 5 epochs --
        if checkpoint_dir and actual_epoch % 5 == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(
                checkpoint_dir, f"pomo_agp_epoch{actual_epoch}.pt",
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": actual_epoch,
                    "K": K,
                    "checkpoint_params": checkpoint_params,
                },
                ckpt_path,
            )
            print(f"  checkpoint -> {ckpt_path}")
            if last_ckpt_path and os.path.exists(last_ckpt_path):
                try:
                    os.remove(last_ckpt_path)
                except OSError:
                    pass
            last_ckpt_path = ckpt_path

        gc.collect()
        torch.cuda.empty_cache()

    print("POMO training complete.\n")


# ===================================================================
#  5.  Evaluation with optional augmentation + dual greedy/stoch
# ===================================================================

@torch.no_grad()
def evaluate_pomo(
    model: torch.nn.Module,
    dataset: Dataset,
    sol_dir: str,
    K: int = 8,
    aug_factor: int = 1,
    reward_fn: Callable | None = None,
    eval_k: int | None = None,
    max_guard_ratio: float | None = None,
) -> list[dict]:
    """Evaluate using POMO rollouts + optional dihedral augmentation.

    For each instance we report:
    - Greedy (deterministic) decode: coverage, guards, guard_ratio
    - Stochastic best-of-(aug*K): coverage, guards, guard_ratio
    - The overall best (max coverage, tie-break fewer guards)
    """
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    per_instance: list[dict] = []
    count = 0
    limit = eval_k if eval_k else len(dataset)

    for batch_data, pad_mask, lens, names in loader:
        if count >= limit:
            break
        batch_data = batch_data.to(device)
        pad_mask   = pad_mask.to(device)
        lens_t     = torch.tensor(lens, dtype=torch.long, device=device)

        name = names[0]
        n = int(lens_t[0].item())
        pts = batch_data[0, :n].detach().cpu().numpy()

        t0 = time.perf_counter()

        # -- deterministic (greedy) decode --
        budget = max(1, int(n * max_guard_ratio)) if max_guard_ratio is not None else None
        det_idxs, _ = model(
            batch_data, padding_mask=pad_mask,
            lengths=lens_t, deterministic=True,
            max_decode_steps=budget,
        )
        det_sol = [idx for idx in det_idxs[0] if idx < n]
        det_cov = 0.0
        if det_sol:
            try:
                det_cov = evaluate_polygon_visibility_numpy_wo_gt(
                    pts, np.array(det_sol, dtype=np.int64), name,
                )
            except Exception:
                det_cov = 0.0

        # -- stochastic best-of-(aug*K) --
        best_stoch_cov = -1.0
        best_stoch_guards: list[int] | None = None

        for aug_idx in range(aug_factor):
            aug_data = (
                batch_data if aug_idx == 0
                else augment_xy(batch_data, aug_idx)
            )
            exp_data    = aug_data.repeat_interleave(K, dim=0)
            exp_mask    = pad_mask.repeat_interleave(K, dim=0)
            exp_lengths = lens_t.repeat_interleave(K)

            all_idxs, _ = model(
                exp_data, padding_mask=exp_mask,
                lengths=exp_lengths, deterministic=False,
                max_decode_steps=budget,
            )

            for k_idx in range(K):
                sol = [idx for idx in all_idxs[k_idx] if idx < n]
                if not sol:
                    continue
                try:
                    cov = evaluate_polygon_visibility_numpy_wo_gt(
                        pts, np.array(sol, dtype=np.int64), name,
                    )
                except Exception:
                    cov = 0.0
                if (cov > best_stoch_cov
                        or (cov == best_stoch_cov
                            and len(sol) < len(best_stoch_guards or []))):
                    best_stoch_cov = cov
                    best_stoch_guards = sol

        dt = time.perf_counter() - t0

        # -- overall best --
        if best_stoch_cov > det_cov or (
                best_stoch_cov == det_cov
                and best_stoch_guards is not None
                and len(best_stoch_guards) < len(det_sol)):
            best_cov    = best_stoch_cov
            best_guards = best_stoch_guards or []
        else:
            best_cov    = det_cov
            best_guards = det_sol

        opt_sol  = _read_opt_solution(sol_dir, name)
        opt_size = len(opt_sol) if opt_sol else None
        n_guards = len(best_guards)

        per_instance.append({
            "name":             name,
            "n":                n,
            # Overall best
            "guards":           n_guards,
            "guard_ratio":      n_guards / max(1.0, float(n)),
            "coverage":         float(best_cov) if best_cov >= 0 else None,
            "opt_size":         opt_size,
            "approx_ratio":     (n_guards / opt_size) if opt_size else None,
            "time_s":           dt,
            # Greedy decode
            "guards_greedy":    len(det_sol),
            "guard_ratio_greedy": len(det_sol) / max(1.0, float(n)),
            "coverage_greedy":  float(det_cov),
            # Stochastic best
            "guards_stoch":     len(best_stoch_guards) if best_stoch_guards else 0,
            "guard_ratio_stoch": (len(best_stoch_guards) / max(1.0, float(n)))
                                 if best_stoch_guards else 0.0,
            "coverage_stoch":   float(best_stoch_cov) if best_stoch_cov >= 0 else None,
        })
        count += 1

    return per_instance


# ===================================================================
#  6.  CLI  /  main
# ===================================================================

def main() -> None:
    if load_dotenv is not None:
        load_dotenv()
    DATASET_PATH = os.getenv("DATASET_PATH")
    if not DATASET_PATH:
        raise EnvironmentError(
            "DATASET_PATH must be set (in .env or environment)."
        )

    p = argparse.ArgumentParser(
        description="POMO-style AGP solver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Model --
    g_model = p.add_argument_group("Model")
    g_model.add_argument("--embedding-size",   type=int,   default=128)
    g_model.add_argument("--hidden-size",      type=int,   default=128)
    g_model.add_argument("--n-glimpses",       type=int,   default=1)
    g_model.add_argument("--tanh-exploration", type=float, default=10)
    g_model.add_argument("--use-tanh",         action="store_true", default=True)
    g_model.add_argument("--temperature",      type=float, default=1.0)
    g_model.add_argument("--eos-bias-init",    type=float, default=None,
                         help="Initial EOS logit bias (e.g. 0.3). None = no bias.")
    g_model.add_argument("--eos-bias-learnable", action="store_true", default=False,
                         help="Make EOS bias a learnable parameter.")

    # -- POMO --
    g_pomo = p.add_argument_group("POMO")
    g_pomo.add_argument("--pomo-k",     type=int, default=8,
                        help="Stochastic rollouts per instance (>= 2)")
    g_pomo.add_argument("--aug-factor", type=int, default=1, choices=[1, 8],
                        help="Augmentations at inference (1=none, 8=full)")

    # -- Reward --
    g_rew = p.add_argument_group("Reward")
    g_rew.add_argument("--reward-lambda",      type=float, default=1.0,
                       help="Guard-sparsity penalty weight lambda")
    g_rew.add_argument("--coverage-threshold", type=float, default=0.99,
                       help="Coverage feasibility threshold tau")
    g_rew.add_argument("--tau-penalty",        type=float, default=3.0,
                       help="Linear penalty weight for coverage < tau")
    g_rew.add_argument("--cap-coverage",        action="store_true", default=True,
                       help="Cap coverage reward at tau (no credit above threshold).")
    g_rew.add_argument("--no-cap-coverage",     dest="cap_coverage", action="store_false",
                       help="Disable coverage cap (original uncapped reward).")
    g_rew.add_argument("--max-guard-ratio",     type=float, default=0.34,
                       help="Max guards as fraction of n (0.34 ≈ ⌊n/3⌋ from Art Gallery Thm). "
                            "Set to 0 or None to disable.")

    # -- Training --
    g_train = p.add_argument_group("Training")
    g_train.add_argument("--epochs",           type=int,   default=50)
    g_train.add_argument("--batch-size",       type=int,   default=64)
    g_train.add_argument("--lr",               type=float, default=2e-4)
    g_train.add_argument("--max-grad-norm",    type=float, default=0.5)
    g_train.add_argument("--entropy-weight",   type=float, default=0.0,
                         help="Entropy bonus weight to prevent policy collapse.")
    g_train.add_argument("--train-size",       type=int,   default=8000)
    g_train.add_argument("--epoch-eval-k",     type=int,   default=200,
                         help="Instances to evaluate per epoch (-1 = all)")

    # -- I/O --
    g_io = p.add_argument_group("IO")
    g_io.add_argument("--agp_train_dir",
                      type=str, default=os.path.join(DATASET_PATH, "train"))
    g_io.add_argument("--agp_val_dir",
                      type=str, default=os.path.join(DATASET_PATH, "dev"))
    g_io.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    g_io.add_argument("--resume-from",    type=str, default=None)
    g_io.add_argument("--verbose",        action="store_true")
    g_io.add_argument("--config",         type=str, default=None,
                      help="Path to JSON config (overrides defaults, CLI overrides config)")

    args = p.parse_args()

    # -- Apply JSON config (v2 feature) --
    defaults = {action.dest: p.get_default(action.dest)
                for action in p._actions if action.dest != 'help'}
    explicit_args = _get_explicit_args(p, sys.argv[1:])
    if args.config:
        config = _load_json_config(args.config)
        config_keys = [
            "epochs", "batch_size", "lr", "temperature", "train_size",
            "epoch_eval_k", "resume_from", "checkpoint_dir", "pomo_k",
            "aug_factor", "reward_lambda", "coverage_threshold",
            "tau_penalty", "cap_coverage", "eos_bias_init",
            "eos_bias_learnable", "entropy_weight", "max_grad_norm",
            "max_guard_ratio",
        ]
        _apply_config_to_args(args, config, defaults, config_keys, explicit_args)

    vprint = print if args.verbose else (lambda *a, **kw: None)

    # -- resolve max_guard_ratio --
    mgr = args.max_guard_ratio
    if mgr is not None and mgr <= 0:
        mgr = None
    vprint(f"[config] max_guard_ratio={mgr}")

    # -- datasets --
    vprint("[phase] preparing datasets")
    train_ds, val_ds = prepare_datasets(
        args.agp_train_dir, args.agp_val_dir, normalize=True,
    )
    sz = args.train_size
    small_train = (
        train_ds if len(train_ds) <= sz
        else Dataset(train_ds.samples[:sz])
    )
    small_val = (
        val_ds if len(val_ds) <= sz
        else Dataset(val_ds.samples[:sz])
    )

    # -- model (with EOS bias, v2 feature) --
    vprint("[phase] creating model")
    model = create_agp_model(
        args.embedding_size,
        args.hidden_size,
        args.n_glimpses,
        args.tanh_exploration,
        args.use_tanh,
        None,               # reward not stored/used inside model
        args.temperature,
        eos_logit_bias_init=args.eos_bias_init,
        eos_logit_bias_learnable=args.eos_bias_learnable,
    )

    # -- resume (v2 feature: restore optimizer state too) --
    start_epoch = 0
    resume_optimizer_state = None
    if args.resume_from:
        vprint(f"[phase] resuming from {args.resume_from}")
        device = next(model.parameters()).device
        ckpt = torch.load(
            args.resume_from, map_location=device, weights_only=False,
        )
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0))
            resume_optimizer_state = ckpt.get("optimizer_state_dict", None)
        vprint(f"  -> resumed at epoch {start_epoch}")

    # -- reward function (closed over CLI args) --
    lam      = args.reward_lambda
    tau      = args.coverage_threshold
    tau_pen  = args.tau_penalty
    cap_cov  = args.cap_coverage

    def reward_fn(
        points: np.ndarray,
        solution,
        name: str,
        length: int | None = None,
    ) -> float:
        return pomo_reward_smooth(
            points, solution, name, length=length,
            lam=lam, tau=tau, tau_penalty=tau_pen,
            cap_at_tau=cap_cov,
        )

    # -- epoch eval callback (v2 feature: dual greedy+stoch, returns dict) --
    def epoch_eval(epoch: int) -> dict:
        ek = (
            len(small_train) if args.epoch_eval_k < 0
            else min(args.epoch_eval_k, len(small_train))
        )
        per = evaluate_pomo(
            model, small_train, args.agp_train_dir,
            K=args.pomo_k, aug_factor=1,
            reward_fn=reward_fn, eval_k=ek,
            max_guard_ratio=mgr,
        )
        # Greedy metrics
        covs_g = [x["coverage_greedy"] for x in per if x.get("coverage_greedy") is not None]
        gr_g   = [x["guard_ratio_greedy"] for x in per if x.get("guard_ratio_greedy") is not None]
        cov_g_mean = float(np.mean(covs_g)) if covs_g else None
        gr_g_mean  = float(np.mean(gr_g)) if gr_g else None
        # Stochastic metrics
        covs_s = [x["coverage_stoch"] for x in per if x.get("coverage_stoch") is not None]
        gr_s   = [x["guard_ratio_stoch"] for x in per if x.get("guard_ratio_stoch") is not None]
        cov_s_mean = float(np.mean(covs_s)) if covs_s else None
        gr_s_mean  = float(np.mean(gr_s)) if gr_s else None
        # Approx ratio (overall best)
        rats = [x["approx_ratio"] for x in per if x.get("approx_ratio") is not None]
        rat_mean = float(np.mean(rats)) if rats else None

        msg = f"[epoch {epoch}] greedy"
        if cov_g_mean is not None:
            msg += f" | cov={cov_g_mean:.3f}"
        if gr_g_mean is not None:
            msg += f" | |S|/n={gr_g_mean:.3f}"
        msg += f"  stoch"
        if cov_s_mean is not None:
            msg += f" | cov={cov_s_mean:.3f}"
        if gr_s_mean is not None:
            msg += f" | |S|/n={gr_s_mean:.3f}"
        if rat_mean is not None:
            msg += f"  |S|/opt={rat_mean:.2f}"
        print(msg)

        return {
            "coverage_greedy_mean":     cov_g_mean,
            "coverage_stoch_mean":      cov_s_mean,
            "guard_ratio_greedy_mean":  gr_g_mean,
            "guard_ratio_stoch_mean":   gr_s_mean,
            "approx_ratio_greedy_mean": rat_mean,
        }

    # -- checkpoint params --
    checkpoint_params = {
        k: v
        for k, v in {
            "embedding_size":  args.embedding_size,
            "hidden_size":     args.hidden_size,
            "n_glimpses":      args.n_glimpses,
            "temperature":     args.temperature,
            "pomo_k":          args.pomo_k,
            "reward_lambda":   args.reward_lambda,
            "tau":             args.coverage_threshold,
            "eos_bias_init":   args.eos_bias_init,
            "entropy_weight":  args.entropy_weight,
        }.items()
        if v is not None
    }

    # -- train --
    vprint("[phase] training")
    remaining = args.epochs - start_epoch
    if remaining > 0:
        pomo_train(
            model, small_train, reward_fn,
            K=args.pomo_k,
            epochs=remaining,
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_params=checkpoint_params,
            epoch_eval_fn=epoch_eval,
            start_epoch=start_epoch,
            episode_log=args.verbose,
            max_grad_norm=args.max_grad_norm,
            entropy_weight=args.entropy_weight,
            resume_optimizer_state=resume_optimizer_state,
            max_guard_ratio=mgr,
        )
    else:
        vprint(
            f"[phase] training complete "
            f"(start_epoch={start_epoch} >= epochs={args.epochs})"
        )

    # -- save final --
    vprint("[phase] saving model")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    final_path = os.path.join(
        args.checkpoint_dir, f"pomo_agp_final_epoch{args.epochs}.pt",
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "epoch": args.epochs,
        },
        final_path,
    )
    print(f"Model saved -> {final_path}")

    # -- evaluate final model --
    vprint(f"[phase] evaluating (aug={args.aug_factor})")
    per_instance = evaluate_pomo(
        model, small_val, args.agp_val_dir,
        K=args.pomo_k,
        aug_factor=args.aug_factor,
        reward_fn=reward_fn,
        max_guard_ratio=mgr,
    )
    report = make_report(
        method="pomo_agp",
        per_instance=per_instance,
        args=vars(args),
        dataset={
            "path":    args.agp_val_dir,
            "eval_k":  len(small_val),
            "train_k": len(small_train),
        },
        oracle={
            "mode": "exact",
            "coverage_threshold": tau,
        },
        timing={},
    )
    report["checkpoint"] = final_path

    out_dir = os.path.join("results", "v3", "pomo_agp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pomo_agp_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # -- print summary --
    s = report["summary"]
    # Greedy summary
    covs_g = [x["coverage_greedy"] for x in per_instance if x.get("coverage_greedy") is not None]
    grs_g  = [x["guard_ratio_greedy"] for x in per_instance if x.get("guard_ratio_greedy") is not None]
    # Stoch summary
    covs_s = [x["coverage_stoch"] for x in per_instance if x.get("coverage_stoch") is not None]
    grs_s  = [x["guard_ratio_stoch"] for x in per_instance if x.get("guard_ratio_stoch") is not None]

    msg = f"\nPOMO eval  (K={args.pomo_k}, aug={args.aug_factor})"
    msg += f"\n  greedy | cov={np.mean(covs_g):.3f} | |S|/n={np.mean(grs_g):.3f}" if covs_g else ""
    msg += f"\n  stoch  | cov={np.mean(covs_s):.3f} | |S|/n={np.mean(grs_s):.3f}" if covs_s else ""
    msg += f"\n  best   | |S| mean={s['guards']['mean']:.2f} | |S|/n={s['guard_ratio']['mean']:.3f}"
    if s["coverage"]["mean"] is not None:
        msg += f" | cov={s['coverage']['mean']:.3f}"
    if s["approx_ratio"]["mean"] is not None:
        msg += f" | |S|/opt={s['approx_ratio']['mean']:.2f}"
    print(msg)
    print(f"\nReport -> {out_path}")


if __name__ == "__main__":
    main()
