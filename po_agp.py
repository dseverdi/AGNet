#!/usr/bin/env python3
"""po_agp.py — Preference Optimization for the Art Gallery Problem (AGP).

Trains a PointerNet policy with Preference Optimization (PO) from
Pan et al. (ICML 2025, arXiv:2505.08735).

Key idea:  For K stochastic rollouts per instance, PO uses *pairwise
preferences* instead of REINFORCE advantages.  For each pair (τ_i, τ_j)
with r(τ_i) > r(τ_j), the Bradley-Terry loss pushes π(τ_i) up and
π(τ_j) down:

    L = - Σ_{i≠j} 𝟙(r_i > r_j) · log σ(α [log π(τ_i) − log π(τ_j)])

Why PO over REINFORCE:
  - Binary preferences don't diminish as rewards converge.
  - Scale-invariant: no sensitivity to reward λ magnitude.
  - Inherent entropy regularisation via α — no separate bonus needed.

The α parameter controls exploration–exploitation (from the
entropy-regularised objective max E[r] + α·H(π)):
  - Large α  (0.05–0.10): more exploratory  (stronger entropy bonus).
  - Small α  (0.01–0.03): more exploitative  (weaker entropy bonus).
  Paper recommends 0.03–0.05 for POMO-style models that already
  incorporate built-in exploration mechanisms.

Variable-length outputs use length-normalised log-probs in the
pairwise comparison (paper §4.2).

Usage:
    python po_agp.py --alpha 0.05 --num-rollouts 8 --epochs 50 --verbose
    python po_agp.py --config configs/po_agp_train.json --verbose

Reference:
    Pan et al., "Preference Optimization for Combinatorial Optimization",
    ICML 2025, arXiv:2505.08735.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import argparse
import random
from typing import Callable, Dict, List

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

from dataset import Dataset, agp_read_samples, collate_fn
from models import create_actor
from utils import evaluate_polygon_visibility_numpy_wo_gt, _get_or_build_vis_cache, get_or_build_disc_vis
from eval_reporting import make_report


# ===================================================================
#  0.  Shared infrastructure
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


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_config_to_args(args, config: Dict, defaults: Dict,
                          keys: List[str], explicit_args: set) -> None:
    for key in keys:
        if key in config and key not in explicit_args and getattr(args, key) == defaults.get(key):
            setattr(args, key, _coerce_value(config[key], defaults.get(key)))


class BucketBatchSampler(Sampler):
    """Batch sampler that groups samples of similar length."""
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
                     use_tanh, temperature):
    return create_actor(
        embedding_size, hidden_size, None, n_glimpses,
        tanh_exploration, use_tanh, "Bahdanau", None,
        temperature=temperature,
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
#  1.  Reward function
# ===================================================================

def po_reward_smooth(
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
    """Smooth, monotone reward.

    r = min(cov, τ) − λ·(|S|/n) − π·max(0, τ − cov)

    With cap_at_tau, coverage above τ yields zero marginal reward,
    so the only way to improve is to reduce |S|.
    """
    n = length if length else len(points)
    sol = np.asarray(solution, dtype=np.int64)
    if len(sol) == 0:
        return float(-tau_penalty)

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


def po_reward_smooth_disc(
    points: np.ndarray,
    solution,
    name: str,
    length: int | None = None,
    *,
    lam: float = 0.2,
    tau: float = 0.99,
    tau_penalty: float = 5.0,
    cap_at_tau: bool = False,
    n_samples: int = 500,
) -> float:
    """Fast approximation of po_reward_smooth using discretised visibility.

    Uses precomputed binary visibility matrix for O(M) coverage estimation
    instead of exact CGAL computation.  Falls back to exact on cache miss.
    """
    n = length if length else len(points)
    sol = np.asarray(solution, dtype=np.int64)
    if len(sol) == 0:
        return float(-tau_penalty)

    pts = points[:n] if length else points

    disc = get_or_build_disc_vis(pts, name, n_samples=n_samples)
    if not disc.get("valid"):
        return po_reward_smooth(
            points, solution, name, length=length,
            lam=lam, tau=tau, tau_penalty=tau_penalty, cap_at_tau=cap_at_tau,
        )

    vis_matrix = disc["vis_matrix"]  # (n_guards, n_samples) bool
    M = disc["n_samples"]
    covered = np.zeros(M, dtype=np.bool_)
    for v in sol:
        if 0 <= v < vis_matrix.shape[0]:
            np.bitwise_or(covered, vis_matrix[v], out=covered)
    coverage = float(covered.sum()) / M

    guard_ratio = len(sol) / max(1, n)
    effective_cov = min(coverage, tau) if cap_at_tau else coverage
    r = effective_cov - lam * guard_ratio
    deficit = max(0.0, tau - coverage)
    r -= tau_penalty * deficit
    return float(r)


def permutation_reward(
    points: np.ndarray,
    permutation: list[int],
    name: str,
    length: int | None = None,
    *,
    lam: float = 0.2,
    tau: float = 0.99,
    tau_penalty: float = 5.0,
    cap_at_tau: bool = True,
) -> float:
    """Dense reward for a full vertex permutation (ranking formulation).

    Evaluates every prefix π[0..k] and returns the best smooth score:

        r_k = min(cov_k, τ) - λ·(k/n) - π·max(0, τ - cov_k)   (if cap_at_tau)
        r_k = cov_k         - λ·(k/n) - π·max(0, τ - cov_k)   (otherwise)

    Final reward is  max_k r_k, which provides dense learning signal even
    when no prefix reaches τ early in training.
    """
    n = length if length else len(points)
    pts = points[:n] if length else points

    if not permutation:
        return float(-tau_penalty)

    # Filter permutation to valid vertex indices
    perm = [v for v in permutation if 0 <= v < n]
    if not perm:
        return float(-tau_penalty)

    # Fast path: incremental union using prebuilt per-guard visibility cache.
    try:
        vcache = _get_or_build_vis_cache(pts, name)
        if not vcache.get("invalid"):
            import skgeom
            guard_vis = vcache["guard_visibility_cache"]
            poly_area = float(vcache["poly_area"])
            cov_cache = vcache["coverage_cache"]

            best_r = -float("inf")
            union = skgeom.PolygonSet()
            current_set: set[int] = set()
            hit_tau = False
            for k, v in enumerate(perm, start=1):
                if v not in current_set:
                    current_set.add(v)
                    if v in guard_vis:
                        union = union.union(guard_vis[v])
                cov = (_union_area(union) / poly_area) if poly_area > 0 else 0.0
                cov_cache[tuple(sorted(current_set))] = cov
                cov_eff = min(cov, tau) if cap_at_tau else cov
                r_k = cov_eff - lam * (k / max(1, n)) - tau_penalty * max(0.0, tau - cov)
                if r_k > best_r:
                    best_r = r_k
                # Early termination: once coverage ≥ τ, adding more guards
                # only increases the λ·(k/n) penalty, so reward can only decrease.
                if cov >= tau and not hit_tau:
                    hit_tau = True
                if hit_tau:
                    # Already past τ — one more step to confirm peak, then stop.
                    if r_k < best_r:
                        break

            return float(best_r)
    except Exception:
        pass

    # Fallback path: direct coverage calls per prefix.
    best_r = -float("inf")
    for k in range(1, len(perm) + 1):
        prefix = np.array(perm[:k], dtype=np.int64)
        try:
            cov = evaluate_polygon_visibility_numpy_wo_gt(pts, prefix, name)
        except Exception:
            cov = 0.0
        cov_eff = min(cov, tau) if cap_at_tau else cov
        r_k = cov_eff - lam * (k / max(1, n)) - tau_penalty * max(0.0, tau - cov)
        if r_k > best_r:
            best_r = r_k

    return float(best_r)


def permutation_reward_disc(
    points: np.ndarray,
    permutation: list[int],
    name: str,
    length: int | None = None,
    *,
    lam: float = 0.2,
    tau: float = 0.99,
    tau_penalty: float = 5.0,
    cap_at_tau: bool = True,
    n_samples: int = 500,
) -> float:
    """Fast approximation of permutation_reward using discretised visibility.

    Instead of O(n) CGAL PolygonSet unions per rollout, uses a precomputed
    binary visibility matrix (n_guards × n_samples).  Coverage of a guard
    subset is estimated as the fraction of sample points seen by at least
    one guard (bitwise OR + count), giving O(M) per prefix step.

    Falls back to exact permutation_reward on cache miss.
    """
    n = length if length else len(points)
    pts = points[:n] if length else points

    if not permutation:
        return float(-tau_penalty)

    perm = [v for v in permutation if 0 <= v < n]
    if not perm:
        return float(-tau_penalty)

    disc = get_or_build_disc_vis(pts, name, n_samples=n_samples)
    if not disc.get("valid"):
        # Fallback to exact computation
        return permutation_reward(
            points, permutation, name, length=length,
            lam=lam, tau=tau, tau_penalty=tau_penalty, cap_at_tau=cap_at_tau,
        )

    vis_matrix = disc["vis_matrix"]          # (n_guards, n_samples) bool
    M = disc["n_samples"]

    # Incremental coverage via bitwise OR
    covered = np.zeros(M, dtype=np.bool_)
    best_r = -float("inf")
    hit_tau = False

    for k, v in enumerate(perm, start=1):
        if v < vis_matrix.shape[0]:
            np.bitwise_or(covered, vis_matrix[v], out=covered)
        cov = float(covered.sum()) / M
        cov_eff = min(cov, tau) if cap_at_tau else cov
        r_k = cov_eff - lam * (k / max(1, n)) - tau_penalty * max(0.0, tau - cov)
        if r_k > best_r:
            best_r = r_k
        # Early termination: once coverage ≥ τ and reward is declining, stop
        if cov >= tau and not hit_tau:
            hit_tau = True
        if hit_tau and r_k < best_r:
            break

    return float(best_r)


# ===================================================================
#  2.  Augmentation utilities  (dihedral group of the square)
# ===================================================================

N_AUGMENTATIONS = 8


def augment_xy(xy: torch.Tensor, aug_idx: int) -> torch.Tensor:
    """Apply one of 8 dihedral-group transforms to 2-D coordinates."""
    x, y = xy[..., 0], xy[..., 1]
    if   aug_idx == 0: ax, ay = x, y
    elif aug_idx == 1: ax, ay = 1 - x, y
    elif aug_idx == 2: ax, ay = x, 1 - y
    elif aug_idx == 3: ax, ay = 1 - x, 1 - y
    elif aug_idx == 4: ax, ay = y, x
    elif aug_idx == 5: ax, ay = 1 - y, x
    elif aug_idx == 6: ax, ay = y, 1 - x
    elif aug_idx == 7: ax, ay = 1 - y, 1 - x
    else:
        raise ValueError(f"aug_idx must be 0-7, got {aug_idx}")
    return torch.stack([ax, ay], dim=-1)


# ===================================================================
#  2b.  Local search (remove / add / swap)
# ===================================================================

def _union_area(polygon_set) -> float:
    """Return the total area of a skgeom.PolygonSet."""
    total = 0.0
    for vis in polygon_set.polygons:
        outer = abs(float(vis.outer_boundary().area()))
        holes = sum(abs(float(h.area())) for h in vis.holes)
        total += outer - holes
    return total


def _coverage_from_guard_vis(
    guard_set: set,
    guard_vis: dict,
    poly_area: float,
    coverage_cache: dict,
) -> float:
    """Compute coverage from prebuilt per-guard visibility polygons.

    Uses `coverage_cache` (keyed by frozenset) to avoid redundant
    CGAL union operations.
    """
    key = tuple(sorted(guard_set))
    if key in coverage_cache:
        return coverage_cache[key]
    if not guard_set or poly_area <= 0:
        coverage_cache[key] = 0.0
        return 0.0
    try:
        import skgeom
        union = skgeom.PolygonSet()
        for g in key:
            if g in guard_vis:
                union = union.union(guard_vis[g])
        cov = _union_area(union) / poly_area
    except Exception:
        cov = 0.0
    coverage_cache[key] = cov
    return cov


def _reward_from_vis(
    guard_set: set,
    guard_vis: dict,
    poly_area: float,
    coverage_cache: dict,
    n: int,
    lam: float,
    tau: float,
    tau_penalty: float,
    cap_at_tau: bool,
) -> float:
    """Cheap reward computation using local vis-cache (no round-trip to reward_fn)."""
    if not guard_set:
        return float(-tau_penalty)
    cov = _coverage_from_guard_vis(guard_set, guard_vis, poly_area, coverage_cache)
    guard_ratio = len(guard_set) / max(1, n)
    effective_cov = min(cov, tau) if cap_at_tau else cov
    r = effective_cov - lam * guard_ratio
    deficit = max(0.0, tau - cov)
    r -= tau_penalty * deficit
    return float(r)


def local_search_improve(
    points: np.ndarray,
    initial_guards: list[int],
    name: str,
    n: int,
    reward_fn: Callable,
    *,
    max_iter: int = 50,
    enable_swap: bool = True,
    # Optional reward params for fast vis-cache path
    lam: float | None = None,
    tau: float | None = None,
    tau_penalty: float | None = None,
    cap_at_tau: bool = False,
) -> tuple[list[int], float, dict]:
    """First-improvement hill-climb on the PO reward function.

    Moves tried each iteration (in order, first improvement restarts):
      1. REMOVE — drop a guard  (saves λ/n, may lose coverage).
      2. ADD    — add a vertex   (costs λ/n, may gain coverage).
      3. SWAP   — replace a guard with a non-guard  (same |S|, may
                  improve coverage geometry).

    When lam/tau/tau_penalty are provided, uses the prebuilt per-guard
    visibility cache from _get_or_build_vis_cache for fast incremental
    coverage evaluation (skips CGAL polygon construction per candidate).

    Returns
    -------
    guards  : sorted list of guard vertex indices.
    reward  : final reward value.
    stats   : dict with iteration count and move counts.
    """
    # ── Try to get prebuilt vis cache for fast path ───────────────
    use_fast = (
        lam is not None and tau is not None and tau_penalty is not None
    )
    guard_vis: dict = {}
    poly_area: float = 0.0
    coverage_cache: dict = {}

    if use_fast:
        try:
            vcache = _get_or_build_vis_cache(points, name)
            if vcache.get("invalid"):
                use_fast = False
            else:
                guard_vis = vcache["guard_visibility_cache"]
                poly_area = vcache["poly_area"]
                coverage_cache = vcache["coverage_cache"]  # shared with global LRU
        except Exception:
            use_fast = False

    if use_fast:
        def _r(gs: set) -> float:
            return _reward_from_vis(
                gs, guard_vis, poly_area, coverage_cache,
                n, lam, tau, tau_penalty, cap_at_tau,  # type: ignore[arg-type]
            )
    else:
        def _r(gs: set) -> float:
            return reward_fn(points, sorted(gs), name, length=n)

    current = set(initial_guards)
    current_r = _r(current)

    n_remove = 0
    n_add    = 0
    n_swap   = 0
    iters    = 0

    for iters in range(1, max_iter + 1):
        improved = False

        # ── Phase 1: REMOVE ──────────────────────────────────────
        if len(current) > 1:
            for g in sorted(current):
                r = _r(current - {g})
                if r > current_r + 1e-9:
                    current.discard(g)
                    current_r = r
                    n_remove += 1
                    improved = True
                    break
        if improved:
            continue

        # ── Phase 2: ADD ─────────────────────────────────────────
        # Fast path: compute current_union once, then ADD is one extra union.
        if use_fast:
            try:
                import skgeom
                cur_key = tuple(sorted(current))
                # Build current union once (likely cached; only keys matter here)
                cur_union = skgeom.PolygonSet()
                for g in cur_key:
                    if g in guard_vis:
                        cur_union = cur_union.union(guard_vis[g])

                best_add_v = -1
                best_add_r = current_r
                for v in range(n):
                    if v in current:
                        continue
                    # Incremental: union of current + {v}
                    cand_key = tuple(sorted(current | {v}))
                    if cand_key in coverage_cache:
                        cov = coverage_cache[cand_key]
                    else:
                        cand_union = cur_union.union(guard_vis[v]) if v in guard_vis else cur_union
                        cov = _union_area(cand_union) / poly_area if poly_area > 0 else 0.0
                        coverage_cache[cand_key] = cov
                    cov_eff = min(cov, tau) if cap_at_tau else cov  # type: ignore[arg-type]
                    r = cov_eff - lam * (len(current) + 1) / max(1, n) - tau_penalty * max(0.0, tau - cov)  # type: ignore[operator]
                    if r > best_add_r + 1e-9:
                        best_add_v = v
                        best_add_r = r
            except Exception:
                # Fall back to generic path for this ADD phase
                best_add_v = -1
                best_add_r = current_r
                for v in range(n):
                    if v in current:
                        continue
                    r = _r(current | {v})
                    if r > best_add_r + 1e-9:
                        best_add_v = v
                        best_add_r = r
        else:
            best_add_v = -1
            best_add_r = current_r
            for v in range(n):
                if v in current:
                    continue
                r = _r(current | {v})
                if r > best_add_r + 1e-9:
                    best_add_v = v
                    best_add_r = r

        if best_add_v >= 0:
            current.add(best_add_v)
            current_r = best_add_r
            n_add += 1
            improved = True
            continue

        # ── Phase 3: SWAP ────────────────────────────────────────
        if enable_swap:
            for g in sorted(current):
                for v in range(n):
                    if v in current:
                        continue
                    r = _r((current - {g}) | {v})
                    if r > current_r + 1e-9:
                        current.discard(g)
                        current.add(v)
                        current_r = r
                        n_swap += 1
                        improved = True
                        break
                if improved:
                    break

        if not improved:
            break  # local optimum

    stats = {
        "iterations": iters,
        "n_remove": n_remove,
        "n_add":    n_add,
        "n_swap":   n_swap,
        "total_moves": n_remove + n_add + n_swap,
    }
    return sorted(current), current_r, stats


# ===================================================================
#  2c.  Teacher-forced log-prob  (for LS fine-tuning, paper §3.4)
# ===================================================================

def teacher_force_log_prob(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_seqs: list[list[int]],
    padding_mask: torch.Tensor | None = None,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute  log π_θ(target_seq | x)  via teacher forcing.

    Mirrors the PointerNet decoder loop but forces ``target_seqs[b]``
    at each step instead of sampling, collecting per-step log-probs.

    Parameters
    ----------
    model : CombinatorialRL wrapper (has ``.actor`` PointerNet).
    inputs : (B, N, 2)
    target_seqs : list of B index lists.  Each list ends with the
        per-sample EOS index (= n_vertices for that sample).
    padding_mask, lengths : same semantics as ``model.forward``.

    Returns
    -------
    log_probs : (B,) — sum of per-step log-probs (NOT length-normalised;
                caller normalises as needed).
    """
    actor = model.actor if hasattr(model, "actor") else model
    device = inputs.device
    batch_size = inputs.size(0)
    seq_len = inputs.size(1)

    # ── Encoder (same as PointerNet.forward) ─────────────────────
    eos_vec = torch.zeros(batch_size, 1, 2, device=device)
    inputs_ext = torch.cat([inputs, eos_vec], dim=1)  # (B, N+1, 2)
    embedded = actor.embedding(inputs_ext.transpose(1, 2))  # (B, N+1, emb)

    if lengths is not None:
        enc_lengths = (lengths + 1).cpu() if torch.is_tensor(lengths) else torch.tensor(
            [l + 1 for l in lengths], device="cpu"
        )
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, enc_lengths, batch_first=True, enforce_sorted=False,
        )
        packed_out, (hidden, context) = actor.encoder(packed)
        encoder_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=seq_len + 1,
        )
    else:
        encoder_outputs, (hidden, context) = actor.encoder(embedded)

    total_len = seq_len + 1
    # Build initial mask
    if padding_mask is not None:
        pad = ~padding_mask
        pad_eos = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        mask = torch.cat([pad, pad_eos], dim=1)
    else:
        mask = torch.zeros(batch_size, total_len, dtype=torch.bool, device=device)

    if lengths is not None:
        for b in range(batch_size):
            n = int(lengths[b].item()) if torch.is_tensor(lengths[b]) else int(lengths[b])
            if n + 1 < total_len:
                mask[b, n + 1:] = True

    # ── Decoder (teacher-forced) ─────────────────────────────────
    decoder_input = actor.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)
    idxs = None
    log_probs_accum = torch.zeros(batch_size, device=device)

    max_steps = max(len(s) for s in target_seqs)
    for step in range(max_steps):
        _, (hidden, context) = actor.decoder(
            decoder_input.unsqueeze(1), (hidden, context),
        )
        query = hidden.squeeze(0)
        for _ in range(actor.n_glimpses):
            ref, logits = actor.glimpse(query, encoder_outputs)
            logits, mask = actor.apply_mask_to_logits(logits, mask, idxs, lengths)
            logits = logits / actor.temperature
            query = torch.bmm(
                ref, F.softmax(logits, dim=1).unsqueeze(2),
            ).squeeze(2)
        _, logits = actor.pointer(query, encoder_outputs)
        logits, mask = actor.apply_mask_to_logits(logits, mask, idxs, lengths)

        if actor.eos_logit_bias is not None and lengths is not None:
            eos_bias_vec = torch.zeros_like(logits)
            for b in range(batch_size):
                eos_pos = int(lengths[b].item()) if torch.is_tensor(lengths[b]) else int(lengths[b])
                if not mask[b, eos_pos]:
                    eos_bias_vec[b, eos_pos] = 1.0
            logits = logits + eos_bias_vec * actor.eos_logit_bias

        probs = F.softmax(logits, dim=1)

        # Force target index
        forced = torch.zeros(batch_size, dtype=torch.long, device=device)
        for b in range(batch_size):
            if step < len(target_seqs[b]):
                forced[b] = target_seqs[b][step]
            else:
                # Past end of this target — use its EOS position
                n = int(lengths[b].item()) if lengths is not None and torch.is_tensor(lengths[b]) else int(lengths[b]) if lengths is not None else seq_len
                forced[b] = n

        # Collect log-prob for the forced index
        for b in range(batch_size):
            if step < len(target_seqs[b]):
                p = probs[b, forced[b]].clamp(min=1e-20)
                log_probs_accum[b] = log_probs_accum[b] + torch.log(p)

        # Update mask & decoder input (same as normal forward)
        idxs = forced
        selected_mask = F.one_hot(idxs, total_len).bool()
        mask = mask | selected_mask
        decoder_input = embedded[torch.arange(batch_size, device=device), idxs, :]

    return log_probs_accum


# ===================================================================
#  3.  Stochastic rollouts (batched, length-normalised log-probs)
# ===================================================================

def _po_rollouts(
    model: torch.nn.Module,
    batch_data: torch.Tensor,
    mask: torch.Tensor,
    lengths: torch.Tensor,
    batch_names: List[str],
    reward_fn: Callable,
    K: int,
    episode_log: bool = False,
    no_eos: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run K stochastic rollouts per instance.

    Parameters
    ----------
    no_eos : bool
        If True, model produces full permutations (ranking mode).
        All rollouts for one instance have the same length = n,
        so no length-normalisation is needed.

    Returns
    -------
    rewards   : (B, K)  float -- non-differentiable.
    log_probs : (B, K)  float -- (length-normalised if EOS mode).
    """
    B = batch_data.size(0)
    device = batch_data.device

    exp_data    = batch_data.repeat_interleave(K, dim=0)   # (B*K, N, 2)
    exp_mask    = mask.repeat_interleave(K, dim=0)
    exp_lengths = lengths.repeat_interleave(K)

    all_idxs, all_log_probs_raw = model(
        exp_data, padding_mask=exp_mask, lengths=exp_lengths,
        deterministic=False, no_eos=no_eos,
    )

    if no_eos:
        # Permutation mode: all rollouts for same instance have same length.
        # No length-normalisation needed.
        all_log_probs = all_log_probs_raw
    else:
        # EOS mode: length-normalise log-probs (paper §4.2).
        step_counts = torch.tensor(
            [max(1, len(idxs)) for idxs in all_idxs],
            dtype=torch.float32, device=device,
        )
        all_log_probs = all_log_probs_raw / step_counts

    # Pre-extract per-instance numpy points AND pre-warm visibility cache.
    # Building the vis cache once per instance avoids redundant rebuilds
    # across K rollouts when LRU eviction would otherwise discard them.
    pts_cache: list[np.ndarray] = []
    for b in range(B):
        n = int(lengths[b].item())
        pts_np = batch_data[b, :n].detach().cpu().numpy()
        pts_cache.append(pts_np)
        # Pre-warm: the reward_fn closure will trigger the right cache
        # (disc_vis for fast_reward, or vis_cache for exact).
        # Calling reward_fn once with an empty solution is cheap and
        # ensures the cache is built before the K-rollout loop.

    # Compute rewards
    rewards_flat: list[float] = []
    for i in range(B * K):
        b_idx = i // K
        n = int(lengths[b_idx].item())
        sol = [idx for idx in all_idxs[i] if idx < n]
        r = reward_fn(pts_cache[b_idx], sol, batch_names[b_idx], length=n)
        rewards_flat.append(float(r))

    rewards   = torch.tensor(rewards_flat, dtype=torch.float32, device=device).view(B, K)
    log_probs = all_log_probs.view(B, K)

    if episode_log and B > 0:
        for b in range(min(2, B)):
            best_k = int(rewards[b].argmax().item())
            n = int(lengths[b].item())
            sol = all_idxs[b * K + best_k]
            if no_eos:
                # Find the prefix that achieves coverage ≥ tau
                for k in range(1, len(sol) + 1):
                    prefix = [v for v in sol[:k] if v < n]
                    cov = evaluate_polygon_visibility_numpy_wo_gt(
                        pts_cache[b],
                        np.array(prefix, dtype=np.int64),
                        batch_names[b],
                    )
                    if cov >= 0.99:
                        sol = prefix
                        break
                else:
                    sol = [v for v in sol if v < n]
                    cov = evaluate_polygon_visibility_numpy_wo_gt(
                        pts_cache[b],
                        np.array(sol, dtype=np.int64) if sol else np.array([], dtype=np.int64),
                        batch_names[b],
                    )
            else:
                sol = [idx for idx in sol if idx < n]
                cov = evaluate_polygon_visibility_numpy_wo_gt(
                    pts_cache[b],
                    np.array(sol, dtype=np.int64) if sol else np.array([], dtype=np.int64),
                    batch_names[b],
                )
            print(
                f"  [po] {batch_names[b]} best-of-{K}: "
                f"cov={cov:.4f} guards={len(sol)} "
                f"r={rewards[b, best_k].item():.4f}"
            )

    return rewards, log_probs


# ===================================================================
#  4.  Preference Optimisation training loop
# ===================================================================

def po_train(
    model: torch.nn.Module,
    dataset: Dataset,
    reward_fn: Callable,
    K: int = 8,
    alpha: float = 0.05,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-4,
    checkpoint_dir: str = "checkpoints",
    checkpoint_params: dict | None = None,
    epoch_eval_fn: Callable | None = None,
    start_epoch: int = 0,
    episode_log: bool = False,
    max_grad_norm: float = 1.0,
    resume_optimizer_state: dict | None = None,
    save_best: bool = True,
    no_eos: bool = False,
    debug_stats: bool = False,
    preference_loss: str = "bt",
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 1e-4,
    lr_schedule: str = "none",
) -> None:
    """Train with Preference Optimisation (PO).

    For each mini-batch of B instances:
        1) K stochastic rollouts  →  rewards (B,K), log_probs (B,K)
        2) Pairwise preferences:  pref[b,i,j] = 𝟙(r[b,i] > r[b,j])
          3) Loss (BT): -Σ pref · log σ(α · Δlog π) / n_pairs
              Loss (Exp): -Σ pref · (α · Δlog π) / n_pairs
        4) Adam + grad clipping.
    """
    if K < 2:
        print("WARNING: K < 2 gives no pairwise preferences. Setting K=2.")
        K = max(K, 2)

    print(
        f"\n{'='*60}\n"
        f"  PO Training  |  {len(dataset)} instances  |  K={K}  |"
        f"  {epochs} epochs  |  bs={batch_size}  |  α={alpha}\n"
        f"{'='*60}"
    )

    preference_loss = str(preference_loss).lower().strip()
    if preference_loss not in {"bt", "exponential"}:
        raise ValueError(f"preference_loss must be one of {{'bt','exponential'}}, got {preference_loss!r}")
    if early_stop_patience < 0:
        early_stop_patience = 0

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
        except Exception as e:
            print(f"[warn] could not restore optimizer state: {e}")

    device = next(model.parameters()).device
    use_amp = (device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # -- optional cosine LR scheduler --
    scheduler = None
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01,
        )
    elif lr_schedule == "warmup_cosine":
        warmup_epochs = max(1, epochs // 20)
        def _lr_lambda(ep):
            if ep < warmup_epochs:
                return float(ep + 1) / float(warmup_epochs)
            progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    lengths = get_lengths_from_dataset(dataset)
    sampler = BucketBatchSampler(
        lengths, batch_size, shuffle=True, bucket_size=10,
    )
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_fn,
        pin_memory=True, num_workers=0,
    )

    last_ckpt_path: str | None = None
    best_cov_greedy = -float("inf")
    best_cov_stoch  = -float("inf")
    best_stop_score = -float("inf")
    no_improve_epochs = 0
    total_grad_steps = 0

    epoch_iter = (
        tqdm(range(epochs), desc="PO epochs") if tqdm else range(epochs)
    )

    for epoch in epoch_iter:
        actual_epoch = start_epoch + epoch + 1
        total_loss   = 0.0
        total_reward = 0.0
        n_instances  = 0
        epoch_pref_pairs = 0.0
        epoch_tie_pairs  = 0.0
        epoch_total_pairs = 0.0
        epoch_lp_abs_sum = 0.0
        epoch_lp_abs_count = 0.0
        epoch_reward_range_sum = 0.0
        epoch_reward_std_sum = 0.0
        epoch_reward_inst_count = 0.0
        epoch_tiny_gap_pairs = 0.0
        epoch_rank_corr_sum = 0.0
        epoch_rank_corr_count = 0.0
        epoch_top1_repeat_sum = 0.0
        epoch_top1_repeat_count = 0.0

        batch_iter = (
            tqdm(loader, desc=f"Epoch {actual_epoch}", leave=False)
            if tqdm else loader
        )

        for batch_data, pad_mask, lens, names in batch_iter:
            batch_data = batch_data.to(device, non_blocking=True)
            pad_mask   = pad_mask.to(device, non_blocking=True)
            lens_t     = torch.tensor(lens, dtype=torch.long, device=device)

            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
                amp_ctx = torch.amp.autocast("cuda", enabled=use_amp)
            else:
                amp_ctx = torch.cuda.amp.autocast(enabled=use_amp)

            with amp_ctx:
                rewards, log_probs = _po_rollouts(
                    model, batch_data, pad_mask, lens_t, names,
                    reward_fn, K, episode_log=episode_log,
                    no_eos=no_eos,
                )

                # ── PO loss (Bradley-Terry) ───────────────────────────
                pref = (rewards[:, :, None] > rewards[:, None, :]).float()   # (B,K,K)
                lp_diff = log_probs[:, :, None] - log_probs[:, None, :]      # (B,K,K)
                n_pairs = pref.sum().clamp(min=1.0)
                if preference_loss == "bt":
                    bt_log = F.logsigmoid(alpha * lp_diff)                   # (B,K,K)
                    loss = -(pref * bt_log).sum() / n_pairs
                else:
                    bt_log = None
                    loss = -(pref * (alpha * lp_diff)).sum() / n_pairs

                if debug_stats:
                    B = rewards.size(0)
                    diag = torch.eye(K, dtype=torch.bool, device=rewards.device).unsqueeze(0)
                    tie = (rewards[:, :, None] == rewards[:, None, :]) & (~diag)
                    non_diag = (~diag)
                    reward_diff = (rewards[:, :, None] - rewards[:, None, :]).abs()

                    # Instance-level reward spread diagnostics
                    reward_range = (rewards.max(dim=1).values - rewards.min(dim=1).values)
                    reward_std = rewards.std(dim=1, unbiased=False)

                    # Tiny pairwise reward gap rate (non-diagonal)
                    tiny_gap_mask = (reward_diff < 1e-3) & non_diag

                    # Rank alignment between reward and policy log-prob ranks (Spearman-like)
                    reward_rank = rewards.argsort(dim=1).argsort(dim=1).float()
                    logp_rank = log_probs.argsort(dim=1).argsort(dim=1).float()
                    rr = reward_rank - reward_rank.mean(dim=1, keepdim=True)
                    ll = logp_rank - logp_rank.mean(dim=1, keepdim=True)
                    denom = torch.sqrt((rr * rr).sum(dim=1) * (ll * ll).sum(dim=1)).clamp(min=1e-12)
                    rank_corr = ((rr * ll).sum(dim=1) / denom)

                    # How often best-reward rollout matches highest-logprob rollout
                    top_reward_idx = rewards.argmax(dim=1)
                    top_logp_idx = log_probs.argmax(dim=1)
                    top1_repeat = (top_reward_idx == top_logp_idx).float()

                    epoch_pref_pairs += float(pref.sum().item())
                    epoch_tie_pairs += float(tie.float().sum().item())
                    epoch_total_pairs += float(B * K * (K - 1))
                    epoch_lp_abs_sum += float(lp_diff.abs().masked_select(non_diag).sum().item())
                    epoch_lp_abs_count += float(B * K * (K - 1))
                    epoch_reward_range_sum += float(reward_range.sum().item())
                    epoch_reward_std_sum += float(reward_std.sum().item())
                    epoch_reward_inst_count += float(B)
                    epoch_tiny_gap_pairs += float(tiny_gap_mask.float().sum().item())
                    epoch_rank_corr_sum += float(rank_corr.sum().item())
                    epoch_rank_corr_count += float(B)
                    epoch_top1_repeat_sum += float(top1_repeat.sum().item())
                    epoch_top1_repeat_count += float(B)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            bs = batch_data.size(0)
            total_loss   += loss.item() * bs
            total_reward += rewards.max(dim=1).values.sum().item()
            n_instances  += bs
            total_grad_steps += 1

            if tqdm and hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(
                    loss=f"{loss.item():.4f}",
                    best_r=f"{rewards.max(dim=1).values.mean().item():.3f}",
                )

            del batch_data, pad_mask, lens_t, rewards, log_probs
            del pref, lp_diff, bt_log
            torch.cuda.empty_cache()

        # -- epoch summary --
        avg_loss = total_loss / max(1, n_instances)
        avg_best = total_reward / max(1, n_instances)
        epoch_batches = n_instances // max(1, batch_size) or 1
        print(
            f"Epoch {actual_epoch}/{start_epoch + epochs}  "
            f"loss={avg_loss:.4f}  best_reward_mean={avg_best:.3f}  "
            f"batches={epoch_batches}  grad_steps_total={total_grad_steps}"
        )

        if debug_stats and epoch_total_pairs > 0:
            pref_rate = epoch_pref_pairs / epoch_total_pairs
            tie_rate  = epoch_tie_pairs / epoch_total_pairs
            mean_lp_abs = epoch_lp_abs_sum / max(1.0, epoch_lp_abs_count)
            tiny_gap_rate = epoch_tiny_gap_pairs / epoch_total_pairs
            mean_reward_range = epoch_reward_range_sum / max(1.0, epoch_reward_inst_count)
            mean_reward_std = epoch_reward_std_sum / max(1.0, epoch_reward_inst_count)
            rank_align = epoch_rank_corr_sum / max(1.0, epoch_rank_corr_count)
            top1_match = epoch_top1_repeat_sum / max(1.0, epoch_top1_repeat_count)
            print(
                f"  [po-debug] pref_rate={pref_rate:.3f} "
                f"tie_rate={tie_rate:.3f} "
                f"tiny_gap_rate={tiny_gap_rate:.3f} "
                f"mean|Δlogπ|={mean_lp_abs:.4f} "
                f"r_range={mean_reward_range:.4f} "
                f"r_std={mean_reward_std:.4f} "
                f"rank_align={rank_align:.3f} "
                f"top1_match={top1_match:.3f}"
            )

        # -- epoch eval --
        eval_metrics = None
        if epoch_eval_fn is not None:
            eval_metrics = epoch_eval_fn(actual_epoch)
            model.train()

        # -- early stopping (optional) --
        if early_stop_patience > 0:
            if isinstance(eval_metrics, dict):
                # Composite score: coverage - guard_ratio.
                # Pure-coverage plateaus at 1.0 and can never trigger
                # further improvement once reached, so we fold in the
                # set-size cost to keep the criterion sensitive.
                # Use max of greedy and stoch composites so we keep
                # training whenever EITHER decoding mode is improving.
                _scores = []
                cov_g = eval_metrics.get("coverage_greedy_mean")
                gr_g  = eval_metrics.get("guard_ratio_greedy_mean")
                if cov_g is not None and gr_g is not None:
                    _scores.append(cov_g - gr_g)
                cov_s = eval_metrics.get("coverage_stoch_mean")
                gr_s  = eval_metrics.get("guard_ratio_stoch_mean")
                if cov_s is not None and gr_s is not None:
                    _scores.append(cov_s - gr_s)
                stop_score = max(_scores) if _scores else float(avg_best)
            else:
                # Fallback if epoch eval is disabled.
                stop_score = float(avg_best)

            if stop_score > best_stop_score + early_stop_min_delta:
                best_stop_score = stop_score
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= early_stop_patience:
                print(
                    f"[early-stop] no improvement for {no_improve_epochs} epochs "
                    f"(best_score={best_stop_score:.4f}, min_delta={early_stop_min_delta})"
                )
                break

        # -- best-checkpoint saving --
        if save_best and checkpoint_dir and isinstance(eval_metrics, dict):
            cov_g = eval_metrics.get("coverage_greedy_mean")
            cov_s = eval_metrics.get("coverage_stoch_mean")

            if cov_g is not None and cov_g > best_cov_greedy:
                best_cov_greedy = cov_g
                os.makedirs(checkpoint_dir, exist_ok=True)
                p = os.path.join(checkpoint_dir, "po_agp_best_greedy.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": actual_epoch,
                    "best_metric": "coverage_greedy_mean",
                    "best_value": float(cov_g),
                    "K": K, "alpha": alpha,
                    "checkpoint_params": checkpoint_params,
                }, p)
                print(f"  Saved best-greedy: {p} (cov={cov_g:.3f})")

            if cov_s is not None and cov_s > best_cov_stoch:
                best_cov_stoch = cov_s
                os.makedirs(checkpoint_dir, exist_ok=True)
                p = os.path.join(checkpoint_dir, "po_agp_best_stoch.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": actual_epoch,
                    "best_metric": "coverage_stoch_mean",
                    "best_value": float(cov_s),
                    "K": K, "alpha": alpha,
                    "checkpoint_params": checkpoint_params,
                }, p)
                print(f"  Saved best-stoch: {p} (cov={cov_s:.3f})")

        # -- periodic checkpoint every 5 epochs --
        if checkpoint_dir and actual_epoch % 5 == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(
                checkpoint_dir, f"po_agp_epoch{actual_epoch}.pt",
            )
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": actual_epoch,
                "K": K, "alpha": alpha,
                "checkpoint_params": checkpoint_params,
            }, ckpt_path)
            print(f"  checkpoint -> {ckpt_path}")
            if last_ckpt_path and os.path.exists(last_ckpt_path):
                try:
                    os.remove(last_ckpt_path)
                except OSError:
                    pass
            last_ckpt_path = ckpt_path

        if scheduler is not None:
            scheduler.step()

        gc.collect()
        torch.cuda.empty_cache()

    print("PO training complete.\n")


# ===================================================================
#  4b. LS Fine-tuning  (paper §3.4, Eq. 9)
# ===================================================================

def po_finetune(
    model: torch.nn.Module,
    dataset: Dataset,
    reward_fn: Callable,
    K: int = 4,
    alpha: float = 0.05,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-5,
    checkpoint_dir: str = "checkpoints",
    checkpoint_params: dict | None = None,
    max_grad_norm: float = 0.5,
    ls_max_iter: int = 50,
    ls_swap: bool = True,
    verbose: bool = False,
    # Reward params forwarded to local_search_improve for fast vis-cache path
    lam: float | None = None,
    tau: float | None = None,
    tau_penalty: float | None = None,
    cap_at_tau: bool = False,
) -> None:
    """Fine-tune via LS-based preference pairs (paper §3.4, Eq. 9).

    For each mini-batch:
      1) Sample K trajectories τ_k from π_θ (get solutions + log π(τ_k)).
      2) Apply local search: LS(τ_k) for each k.
      3) Teacher-force LS(τ_k) through model → get log π_θ(LS(τ_k)).
      4) For each (τ_k, LS(τ_k)) pair where r(LS) > r(τ):
              loss += -log σ(α · [log π(LS) − log π(τ)])
      5) Update θ.

    This teaches the policy to **internalise** local-search quality
    solutions, so at inference time LS is no longer needed.
    """
    print(
        f"\n{'='*60}\n"
        f"  LS Fine-tune (§3.4)  |  {len(dataset)} inst  |  K={K}  |"
        f"  {epochs} epochs  |  bs={batch_size}  |  α={alpha}\n"
        f"{'='*60}"
    )

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device

    lengths = get_lengths_from_dataset(dataset)
    sampler = BucketBatchSampler(lengths, batch_size, shuffle=True, bucket_size=10)
    loader = DataLoader(
        dataset, batch_sampler=sampler, collate_fn=collate_fn,
        pin_memory=True, num_workers=0,
    )

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_improved = 0
        n_total_pairs = 0
        n_instances = 0

        batch_iter = (
            tqdm(loader, desc=f"FT epoch {epoch}", leave=False)
            if tqdm else loader
        )

        for batch_data, pad_mask, lens, names in batch_iter:
            batch_data = batch_data.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True)
            lens_t = torch.tensor(lens, dtype=torch.long, device=device)
            B = batch_data.size(0)

            # ── 1) Sample K rollouts from current policy ─────────
            with torch.no_grad():
                exp_data = batch_data.repeat_interleave(K, dim=0)
                exp_mask = pad_mask.repeat_interleave(K, dim=0)
                exp_lens = lens_t.repeat_interleave(K)
                all_idxs, _ = model(
                    exp_data, padding_mask=exp_mask, lengths=exp_lens,
                    deterministic=False,
                )
                # Pre-extract points
                pts_cache = []
                for b in range(B):
                    n = int(lens_t[b].item())
                    pts_cache.append(batch_data[b, :n].detach().cpu().numpy())

            # ── 2) Apply LS to each rollout ──────────────────────
            tau_seqs: list[list[int]] = []      # original solution sequences
            ls_seqs:  list[list[int]] = []       # LS-improved sequences (+ EOS)
            tau_rewards: list[float]  = []
            ls_rewards:  list[float]  = []
            pair_b_idx:  list[int]    = []       # which batch element

            for i in range(B * K):
                b = i // K
                n = int(lens_t[b].item())
                sol = [idx for idx in all_idxs[i] if idx < n]
                r_tau = reward_fn(pts_cache[b], sol, names[b], length=n)

                ls_sol, r_ls, _ = local_search_improve(
                    pts_cache[b], sol, names[b], n, reward_fn,
                    max_iter=ls_max_iter, enable_swap=ls_swap,
                    lam=lam, tau=tau, tau_penalty=tau_penalty, cap_at_tau=cap_at_tau,
                )

                if r_ls > r_tau + 1e-9:
                    # Preference pair: LS wins over tau
                    tau_seqs.append(all_idxs[i])   # includes EOS from model
                    ls_seqs.append(ls_sol + [n])    # append EOS index
                    tau_rewards.append(r_tau)
                    ls_rewards.append(r_ls)
                    pair_b_idx.append(b)
                    n_improved += 1

                n_total_pairs += 1

            if not tau_seqs:
                # No improvements found in this batch — skip
                n_instances += B
                continue

            # ── 3) Teacher-force log π(LS(τ)) ────────────────────
            # Gather the batch elements that have improvements
            P = len(tau_seqs)
            tf_data = torch.stack([batch_data[pair_b_idx[j]] for j in range(P)])
            tf_mask = torch.stack([pad_mask[pair_b_idx[j]] for j in range(P)])
            tf_lens = torch.tensor(
                [int(lens_t[pair_b_idx[j]].item()) for j in range(P)],
                dtype=torch.long, device=device,
            )

            # log π(LS(τ)) via teacher forcing
            lp_ls = teacher_force_log_prob(model, tf_data, ls_seqs, tf_mask, tf_lens)

            # log π(τ) via teacher forcing (original sampled sequence)
            lp_tau = teacher_force_log_prob(model, tf_data, tau_seqs, tf_mask, tf_lens)

            # Length-normalise
            ls_steps = torch.tensor(
                [max(1, len(s)) for s in ls_seqs], dtype=torch.float32, device=device,
            )
            tau_steps = torch.tensor(
                [max(1, len(s)) for s in tau_seqs], dtype=torch.float32, device=device,
            )
            lp_ls_norm = lp_ls / ls_steps
            lp_tau_norm = lp_tau / tau_steps

            # ── 4) BT loss: LS preferred over τ ─────────────────
            # y=1 (LS always wins here), so loss = -log σ(α·(log π(LS) − log π(τ)))
            loss = -F.logsigmoid(alpha * (lp_ls_norm - lp_tau_norm)).mean()

            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item() * P
            n_instances += B

            if tqdm and hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(
                    loss=f"{loss.item():.4f}",
                    improved=f"{n_improved}/{n_total_pairs}",
                )

            del tf_data, tf_mask, tf_lens, lp_ls, lp_tau
            torch.cuda.empty_cache()

        avg_loss = total_loss / max(1, n_improved)
        pct = 100 * n_improved / max(1, n_total_pairs)
        print(
            f"FT epoch {epoch}/{epochs}  "
            f"loss={avg_loss:.4f}  "
            f"improved={n_improved}/{n_total_pairs} ({pct:.0f}%)"
        )

        # -- checkpoint --
        if checkpoint_dir and epoch % max(1, epochs // 2) == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            p = os.path.join(checkpoint_dir, f"po_agp_ft_epoch{epoch}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "finetune": True,
                "K": K, "alpha": alpha,
                "checkpoint_params": checkpoint_params,
            }, p)
            print(f"  FT checkpoint -> {p}")

        gc.collect()
        torch.cuda.empty_cache()

    print("LS fine-tuning complete.\n")


# ===================================================================
#  5.  Evaluation  (greedy + stochastic, optional augmentation + LS)
# ===================================================================

@torch.no_grad()
def evaluate_po(
    model: torch.nn.Module,
    dataset: Dataset,
    sol_dir: str,
    K: int = 8,
    aug_factor: int = 1,
    reward_fn: Callable | None = None,
    eval_k: int | None = None,
    local_search: bool = False,
    ls_max_iter: int = 50,
    ls_swap: bool = True,
    no_eos: bool = False,
    tau: float = 0.99,
    disc_vis_samples: int = 0,
) -> list[dict]:
    """Evaluate with greedy decode + stochastic best-of-(aug*K).

    If ``no_eos=True``, the model produces full permutations and the
    guard set is the shortest prefix achieving coverage ≥ τ.

    If ``local_search=True``, the best solution (greedy or stochastic)
    is refined via remove/add/swap hill-climbing on the reward function.

    If ``disc_vis_samples > 0``, use discretised visibility for fast
    approximate coverage computation (suitable for epoch-eval during
    training).  Set to 0 for exact CGAL evaluation.
    """
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    def _fast_coverage(pts: np.ndarray, guards: list[int],
                       name: str) -> float:
        """Compute coverage via discretised visibility matrix."""
        disc = get_or_build_disc_vis(pts, name, n_samples=disc_vis_samples)
        if not disc.get("valid"):
            # Fallback to exact
            try:
                return evaluate_polygon_visibility_numpy_wo_gt(
                    pts, np.array(guards, dtype=np.int64), name)
            except Exception:
                return 0.0
        vis_matrix = disc["vis_matrix"]  # (n_guards, n_samples)
        M = disc["n_samples"]
        covered = np.zeros(M, dtype=np.bool_)
        for v in guards:
            if 0 <= v < vis_matrix.shape[0]:
                np.bitwise_or(covered, vis_matrix[v], out=covered)
        return float(covered.sum()) / M

    def _eval_coverage(pts: np.ndarray, guards: list[int],
                       name: str) -> float:
        """Compute coverage — fast or exact depending on disc_vis_samples."""
        if disc_vis_samples > 0:
            return _fast_coverage(pts, guards, name)
        try:
            return evaluate_polygon_visibility_numpy_wo_gt(
                pts, np.array(guards, dtype=np.int64), name)
        except Exception:
            return 0.0

    def _perm_to_guards(perm: list[int], pts: np.ndarray, name: str,
                        n: int) -> tuple[list[int], float]:
        """Extract shortest prefix of permutation achieving coverage ≥ τ."""
        valid = [v for v in perm if 0 <= v < n]
        if not valid:
            return [], 0.0

        # Fast disc-vis path: incremental bitwise OR
        if disc_vis_samples > 0:
            disc = get_or_build_disc_vis(pts, name, n_samples=disc_vis_samples)
            if disc.get("valid"):
                vis_matrix = disc["vis_matrix"]
                M = disc["n_samples"]
                covered = np.zeros(M, dtype=np.bool_)
                current: list[int] = []
                for v in valid:
                    current.append(v)
                    if 0 <= v < vis_matrix.shape[0]:
                        np.bitwise_or(covered, vis_matrix[v], out=covered)
                    cov = float(covered.sum()) / M
                    if cov >= tau:
                        return current, cov
                return current, cov

        # Exact path: incremental CGAL union
        try:
            vcache = _get_or_build_vis_cache(pts, name)
            if not vcache.get("invalid"):
                import skgeom
                guard_vis = vcache["guard_visibility_cache"]
                poly_area = float(vcache["poly_area"])
                union = skgeom.PolygonSet()
                current2: list[int] = []
                current_set: set[int] = set()
                cov = 0.0
                for v in valid:
                    current2.append(v)
                    if v not in current_set:
                        current_set.add(v)
                        if v in guard_vis:
                            union = union.union(guard_vis[v])
                    cov = (_union_area(union) / poly_area) if poly_area > 0 else 0.0
                    if cov >= tau:
                        return current2, cov
                return current2, cov
        except Exception:
            pass

        for k in range(1, len(valid) + 1):
            prefix = valid[:k]
            try:
                cov = evaluate_polygon_visibility_numpy_wo_gt(
                    pts, np.array(prefix, dtype=np.int64), name)
            except Exception:
                cov = 0.0
            if cov >= tau:
                return prefix, cov
        # Never reached τ — return all
        try:
            cov = evaluate_polygon_visibility_numpy_wo_gt(
                pts, np.array(valid, dtype=np.int64), name) if valid else 0.0
        except Exception:
            cov = 0.0
        return valid, cov

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

        # -- greedy decode --
        det_idxs, _ = model(
            batch_data, padding_mask=pad_mask,
            lengths=lens_t, deterministic=True,
            no_eos=no_eos,
        )
        if no_eos:
            det_sol, det_cov = _perm_to_guards(det_idxs[0], pts, name, n)
        else:
            det_sol = [idx for idx in det_idxs[0] if idx < n]
            det_cov = 0.0
            if det_sol:
                det_cov = _eval_coverage(pts, det_sol, name)

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
                no_eos=no_eos,
            )

            for k_idx in range(K):
                if no_eos:
                    sol, cov = _perm_to_guards(all_idxs[k_idx], pts, name, n)
                else:
                    sol = [idx for idx in all_idxs[k_idx] if idx < n]
                    if not sol:
                        continue
                    cov = _eval_coverage(pts, sol, name)
                if (cov > best_stoch_cov
                        or (cov == best_stoch_cov
                            and len(sol) < len(best_stoch_guards or []))):
                    best_stoch_cov = cov
                    best_stoch_guards = sol

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

        # -- optional local search refinement --
        ls_info: dict = {}
        if local_search and best_guards and reward_fn is not None:
            pre_ls_guards = n_guards
            pre_ls_cov    = best_cov
            ls_guards, ls_r, ls_stats = local_search_improve(
                pts, best_guards, name, n, reward_fn,
                max_iter=ls_max_iter, enable_swap=ls_swap,
            )
            # Recompute coverage for the refined solution
            ls_cov = 0.0
            if ls_guards:
                try:
                    ls_cov = evaluate_polygon_visibility_numpy_wo_gt(
                        pts, np.array(ls_guards, dtype=np.int64), name,
                    )
                except Exception:
                    ls_cov = 0.0
            # Accept LS result (it can only be >= current reward)
            best_guards = ls_guards
            best_cov    = ls_cov
            n_guards    = len(best_guards)
            ls_info = {
                "ls_pre_guards":  pre_ls_guards,
                "ls_pre_cov":     float(pre_ls_cov) if pre_ls_cov >= 0 else None,
                "ls_post_guards": n_guards,
                "ls_post_cov":    float(ls_cov),
                "ls_delta_guards": n_guards - pre_ls_guards,
                "ls_iterations":  ls_stats["iterations"],
                "ls_removes":     ls_stats["n_remove"],
                "ls_adds":        ls_stats["n_add"],
                "ls_swaps":       ls_stats["n_swap"],
            }

        dt = time.perf_counter() - t0

        record = {
            "name":             name,
            "n":                n,
            "guards":           n_guards,
            "guard_ratio":      n_guards / max(1.0, float(n)),
            "coverage":         float(best_cov) if best_cov >= 0 else None,
            "opt_size":         opt_size,
            "approx_ratio":     (n_guards / opt_size) if opt_size else None,
            "time_s":           dt,
            "guards_greedy":    len(det_sol),
            "guard_ratio_greedy": len(det_sol) / max(1.0, float(n)),
            "coverage_greedy":  float(det_cov),
            "guards_stoch":     len(best_stoch_guards) if best_stoch_guards else 0,
            "guard_ratio_stoch": (len(best_stoch_guards) / max(1.0, float(n)))
                                 if best_stoch_guards else 0.0,
            "coverage_stoch":   float(best_stoch_cov) if best_stoch_cov >= 0 else None,
        }
        record.update(ls_info)
        per_instance.append(record)
        count += 1

    return per_instance


# ===================================================================
#  6.  CLI / main
# ===================================================================

def main() -> None:
    if load_dotenv is not None:
        load_dotenv()
    DATASET_PATH = os.getenv("DATASET_PATH")
    if not DATASET_PATH:
        raise EnvironmentError("DATASET_PATH must be set (in .env or environment).")

    p = argparse.ArgumentParser(
        description="PO-AGP: Preference Optimisation for the Art Gallery Problem",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Model --
    g = p.add_argument_group("Model")
    g.add_argument("--embedding-size",   type=int,   default=128)
    g.add_argument("--hidden-size",      type=int,   default=128)
    g.add_argument("--n-glimpses",       type=int,   default=1)
    g.add_argument("--tanh-exploration", type=float, default=10)
    g.add_argument("--use-tanh",         action="store_true", default=True)
    g.add_argument("--temperature",      type=float, default=1.0)

    # -- PO --
    g = p.add_argument_group("Preference Optimisation")
    g.add_argument("--num-rollouts", type=int, default=8,
                   help="Stochastic rollouts per instance K (>= 2)")
    g.add_argument("--aug-factor",   type=int, default=1, choices=[1, 8],
                   help="Dihedral augmentations at inference (1=none, 8=full)")
    g.add_argument("--alpha",        type=float, default=0.05,
                   help="PO scaling α (higher=more exploration). 0.03–0.05 recommended.")
    g.add_argument("--preference-loss", type=str, default="bt", choices=["bt", "exponential"],
                   help="Preference model loss. 'bt' uses log-sigmoid Bradley-Terry; 'exponential' uses linear exp-style objective.")
    g.add_argument("--ranking-mode", action="store_true", default=False,
                   help="Ranking formulation: model outputs full vertex permutation "
                        "(no EOS), guard set = shortest prefix achieving coverage ≥ τ.")
    g.add_argument("--po-debug-stats", action="store_true", default=False,
                   help="Print per-epoch PO diagnostics: preference rate, tie rate, and mean |Δlogπ|.")
    g.add_argument("--early-stop-patience", type=int, default=0,
                   help="Stop training if coverage_stoch_mean does not improve for this many epochs (0 disables).")
    g.add_argument("--early-stop-min-delta", type=float, default=1e-4,
                   help="Minimum improvement to reset early-stop patience.")
    g.add_argument("--lr-schedule", type=str, default="none",
                   choices=["none", "cosine", "warmup_cosine"],
                   help="LR schedule: none (constant), cosine, or warmup_cosine.")

    # -- Local search --
    g = p.add_argument_group("Local Search")
    g.add_argument("--local-search",    action="store_true", default=False,
                   help="Refine best solution with remove/add/swap local search.")
    g.add_argument("--ls-max-iter",     type=int, default=50,
                   help="Maximum LS iterations per instance.")
    g.add_argument("--ls-no-swap",      action="store_true", default=False,
                   help="Disable swap moves in LS (faster, less thorough).")

    # -- LS Fine-tuning (§3.4) --
    g = p.add_argument_group("LS Fine-tuning")
    g.add_argument("--finetune-only",   action="store_true", default=False,
                   help="Skip PO training, run only LS fine-tuning (requires --resume-from).")
    g.add_argument("--skip-finetune",   action="store_true", default=False,
                   help="Skip LS fine-tuning stage (useful for quick smoke runs).")
    g.add_argument("--finetune-epochs", type=int, default=0,
                   help="Post-training LS fine-tune epochs (0=disabled). "
                        "Paper recommends ~5%% of total epochs.")
    g.add_argument("--finetune-lr",     type=float, default=1e-5,
                   help="Learning rate for LS fine-tuning (lower than main training).")
    g.add_argument("--finetune-k",      type=int, default=4,
                   help="Rollouts per instance during fine-tuning.")
    g.add_argument("--ft-reward-lambda",  type=float, default=None,
                   help="Override reward λ during FT (default: same as --reward-lambda).")
    g.add_argument("--ft-tau-penalty",    type=float, default=None,
                   help="Override τ-penalty during FT (default: same as --tau-penalty).")
    g.add_argument("--ft-cap-coverage",   type=str, default=None, choices=["true", "false"],
                   help="Override cap-coverage during FT (default: same as --cap-coverage).")

    # -- Reward --
    g = p.add_argument_group("Reward")
    g.add_argument("--reward-lambda",      type=float, default=1.0,
                   help="Guard-sparsity penalty weight λ")
    g.add_argument("--coverage-threshold", type=float, default=0.99,
                   help="Coverage feasibility threshold τ")
    g.add_argument("--tau-penalty",        type=float, default=3.0,
                   help="Penalty weight for coverage < τ")
    g.add_argument("--cap-coverage",       action="store_true", default=True,
                   help="Cap coverage reward at τ.")
    g.add_argument("--no-cap-coverage",    dest="cap_coverage", action="store_false")
    g.add_argument("--fast-reward",        action="store_true", default=False,
                   help="Use discretised visibility (approx) for fast training reward.")
    g.add_argument("--disc-vis-samples",   type=int, default=500,
                   help="Number of sample points for discretised visibility (with --fast-reward).")

    # -- Training --
    g = p.add_argument_group("Training")
    g.add_argument("--epochs",        type=int,   default=50)
    g.add_argument("--batch-size",    type=int,   default=64)
    g.add_argument("--lr",            type=float, default=2e-4)
    g.add_argument("--max-grad-norm", type=float, default=0.5)
    g.add_argument("--train-size",    type=int,   default=8000)
    g.add_argument("--seed",          type=int,   default=1234,
                   help="Global random seed for reproducible runs.")
    g.add_argument("--epoch-eval-k",  type=int,   default=200,
                   help="Instances to evaluate per epoch (-1 = all)")

    # -- I/O --
    g = p.add_argument_group("IO")
    g.add_argument("--agp_train_dir", type=str,
                   default=os.path.join(DATASET_PATH, "train"))
    g.add_argument("--agp_val_dir",   type=str,
                   default=os.path.join(DATASET_PATH, "dev"))
    g.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    g.add_argument("--resume-from",    type=str, default=None)
    g.add_argument("--verbose",        action="store_true")
    g.add_argument("--config",         type=str, default=None,
                   help="JSON config (overrides defaults, CLI overrides config)")

    args = p.parse_args()

    # -- Apply JSON config --
    defaults = {action.dest: p.get_default(action.dest)
                for action in p._actions if action.dest != 'help'}
    explicit_args = _get_explicit_args(p, sys.argv[1:])
    if args.config:
        config = _load_json_config(args.config)
        config_keys = [
            "epochs", "batch_size", "lr", "temperature", "train_size",
            "seed", "epoch_eval_k", "resume_from", "checkpoint_dir",
            "num_rollouts", "aug_factor", "alpha", "ranking_mode",
            "po_debug_stats", "preference_loss",
            "early_stop_patience", "early_stop_min_delta",
            "lr_schedule",
            "reward_lambda", "coverage_threshold", "tau_penalty",
            "cap_coverage", "max_grad_norm",
            "local_search", "ls_max_iter",
            "finetune_epochs", "finetune_lr", "finetune_k",
            "finetune_only", "skip_finetune",
            "ft_reward_lambda", "ft_tau_penalty", "ft_cap_coverage",
            "fast_reward", "disc_vis_samples",
        ]
        _apply_config_to_args(args, config, defaults, config_keys, explicit_args)

    # Validate --finetune-only
    if args.finetune_only:
        if not args.resume_from:
            p.error("--finetune-only requires --resume-from")
        if args.finetune_epochs <= 0:
            args.finetune_epochs = max(args.finetune_epochs, 3)
            print(f"[warn] --finetune-only with finetune_epochs<=0, defaulting to {args.finetune_epochs}")
    if args.finetune_only and args.skip_finetune:
        p.error("--finetune-only and --skip-finetune cannot be used together")

    vprint = print if args.verbose else (lambda *a, **kw: None)
    vprint(
        f"[config] α={args.alpha}  K={args.num_rollouts}  loss={args.preference_loss}  "
        f"LS={args.local_search}  ranking={args.ranking_mode}"
    )
    vprint(f"[config] seed={args.seed}")

    _set_global_seed(args.seed)

    # -- datasets --
    vprint("[phase] preparing datasets")
    train_ds, val_ds = prepare_datasets(
        args.agp_train_dir, args.agp_val_dir, normalize=True,
    )
    sz = args.train_size
    small_train = train_ds if len(train_ds) <= sz else Dataset(train_ds.samples[:sz])
    small_val   = val_ds   if len(val_ds)   <= sz else Dataset(val_ds.samples[:sz])

    # -- model --
    vprint("[phase] creating model")
    model = create_agp_model(
        args.embedding_size, args.hidden_size, args.n_glimpses,
        args.tanh_exploration, args.use_tanh, args.temperature,
    )

    # -- resume --
    start_epoch = 0
    resume_optimizer_state = None
    if args.resume_from:
        vprint(f"[phase] resuming from {args.resume_from}")
        device = next(model.parameters()).device
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0))
            resume_optimizer_state = ckpt.get("optimizer_state_dict", None)
        vprint(f"  -> resumed at epoch {start_epoch}")

    # -- reward function --
    lam     = args.reward_lambda
    tau     = args.coverage_threshold
    tau_pen = args.tau_penalty
    cap_cov = args.cap_coverage

    if args.fast_reward:
        disc_n = args.disc_vis_samples
        vprint(f"[config] fast_reward=True  disc_vis_samples={disc_n}")
        if args.ranking_mode:
            def reward_fn(points, solution, name, length=None):
                return permutation_reward_disc(
                    points, solution, name, length=length,
                    lam=lam, tau=tau, tau_penalty=tau_pen,
                    cap_at_tau=cap_cov, n_samples=disc_n,
                )
        else:
            def reward_fn(points, solution, name, length=None):
                return po_reward_smooth_disc(
                    points, solution, name, length=length,
                    lam=lam, tau=tau, tau_penalty=tau_pen,
                    cap_at_tau=cap_cov, n_samples=disc_n,
                )
    elif args.ranking_mode:
        def reward_fn(points, solution, name, length=None):
            return permutation_reward(
                points, solution, name, length=length,
                lam=lam, tau=tau, tau_penalty=tau_pen,
                cap_at_tau=cap_cov,
            )
    else:
        def reward_fn(points, solution, name, length=None):
            return po_reward_smooth(
                points, solution, name, length=length,
                lam=lam, tau=tau, tau_penalty=tau_pen, cap_at_tau=cap_cov,
            )

    # -- epoch eval callback --
    def epoch_eval(epoch: int) -> dict:
        ek = (
            len(small_train) if args.epoch_eval_k < 0
            else min(args.epoch_eval_k, len(small_train))
        )
        per = evaluate_po(
            model, small_train, args.agp_train_dir,
            K=args.num_rollouts, aug_factor=1,
            reward_fn=reward_fn, eval_k=ek,
            no_eos=args.ranking_mode, tau=tau,
            disc_vis_samples=args.disc_vis_samples if args.fast_reward else 0,
        )
        covs_g = [x["coverage_greedy"] for x in per if x.get("coverage_greedy") is not None]
        gr_g   = [x["guard_ratio_greedy"] for x in per]
        covs_s = [x["coverage_stoch"] for x in per if x.get("coverage_stoch") is not None]
        gr_s   = [x["guard_ratio_stoch"] for x in per]
        rats   = [x["approx_ratio"] for x in per if x.get("approx_ratio") is not None]

        cov_g_mean = float(np.mean(covs_g)) if covs_g else None
        gr_g_mean  = float(np.mean(gr_g))   if gr_g   else None
        cov_s_mean = float(np.mean(covs_s)) if covs_s else None
        gr_s_mean  = float(np.mean(gr_s))   if gr_s   else None
        rat_mean   = float(np.mean(rats))   if rats   else None

        msg = f"[epoch {epoch}]"
        if cov_g_mean is not None:
            msg += f"  greedy cov={cov_g_mean:.3f} |S|/n={gr_g_mean:.3f}"
        if cov_s_mean is not None:
            msg += f"  stoch cov={cov_s_mean:.3f} |S|/n={gr_s_mean:.3f}"
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
        "embedding_size": args.embedding_size,
        "hidden_size":    args.hidden_size,
        "n_glimpses":     args.n_glimpses,
        "temperature":    args.temperature,
        "num_rollouts":   args.num_rollouts,
        "alpha":          args.alpha,
        "seed":           args.seed,
        "reward_lambda":  args.reward_lambda,
        "tau":            args.coverage_threshold,
    }

    # -- FT-specific reward overrides --
    ft_lam     = args.ft_reward_lambda if args.ft_reward_lambda is not None else lam
    ft_tau_pen = args.ft_tau_penalty   if args.ft_tau_penalty   is not None else tau_pen
    ft_cap_cov = (args.ft_cap_coverage.lower() == "true") if args.ft_cap_coverage is not None else cap_cov

    def ft_reward_fn(points, solution, name, length=None):
        return po_reward_smooth(
            points, solution, name, length=length,
            lam=ft_lam, tau=tau, tau_penalty=ft_tau_pen, cap_at_tau=ft_cap_cov,
        )

    if ft_lam != lam or ft_tau_pen != tau_pen or ft_cap_cov != cap_cov:
        vprint(f"[config] FT reward overrides: λ={ft_lam} π={ft_tau_pen} cap={ft_cap_cov}")

    # -- train --
    vprint("[phase] training")
    remaining = args.epochs - start_epoch
    if remaining > 0 and not args.finetune_only:
        po_train(
            model, small_train, reward_fn,
            K=args.num_rollouts,
            alpha=args.alpha,
            epochs=remaining,
            batch_size=args.batch_size,
            lr=args.lr,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_params=checkpoint_params,
            epoch_eval_fn=epoch_eval,
            start_epoch=start_epoch,
            episode_log=args.verbose,
            max_grad_norm=args.max_grad_norm,
            resume_optimizer_state=resume_optimizer_state,
            no_eos=args.ranking_mode,
            debug_stats=args.po_debug_stats,
            preference_loss=args.preference_loss,
            early_stop_patience=args.early_stop_patience,
            early_stop_min_delta=args.early_stop_min_delta,
            lr_schedule=args.lr_schedule,
        )
    elif args.finetune_only:
        vprint(f"[skip] --finetune-only, skipping PO training")
    else:
        vprint(f"[skip] start_epoch={start_epoch} >= epochs={args.epochs}")

    # -- LS fine-tuning (§3.4) --
    if args.skip_finetune:
        vprint("[skip] --skip-finetune set, skipping LS fine-tuning")
    elif args.finetune_epochs > 0:
        vprint(f"[phase] LS fine-tuning ({args.finetune_epochs} epochs)")
        po_finetune(
            model, small_train, ft_reward_fn,
            K=args.finetune_k,
            alpha=args.alpha,
            epochs=args.finetune_epochs,
            batch_size=args.batch_size,
            lr=args.finetune_lr,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_params=checkpoint_params,
            max_grad_norm=args.max_grad_norm,
            ls_max_iter=args.ls_max_iter,
            ls_swap=not args.ls_no_swap,
            verbose=args.verbose,
            lam=ft_lam,
            tau=tau,
            tau_penalty=ft_tau_pen,
            cap_at_tau=ft_cap_cov,
        )

    # -- save final --
    vprint("[phase] saving model")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    final_path = os.path.join(
        args.checkpoint_dir, f"po_agp_final_epoch{args.epochs}.pt",
    )
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "epoch": args.epochs,
    }, final_path)
    print(f"Model saved -> {final_path}")

    # -- final evaluation --
    ls_label = "+LS" if args.local_search else ""
    vprint(f"[phase] evaluating (aug={args.aug_factor}{ls_label})")
    per_instance = evaluate_po(
        model, small_val, args.agp_val_dir,
        K=args.num_rollouts,
        aug_factor=args.aug_factor,
        reward_fn=reward_fn,
        local_search=args.local_search,
        ls_max_iter=args.ls_max_iter,
        ls_swap=not args.ls_no_swap,
        no_eos=args.ranking_mode,
        tau=tau,
    )
    report = make_report(
        method="po_agp",
        per_instance=per_instance,
        args=vars(args),
        dataset={
            "path":    args.agp_val_dir,
            "eval_k":  len(small_val),
            "train_k": len(small_train),
        },
        oracle={"mode": "exact", "coverage_threshold": tau},
        timing={},
    )
    report["checkpoint"] = final_path

    out_dir = os.path.join("results", "v3", "po_agp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "po_agp_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # -- summary --
    s = report["summary"]
    covs_g = [x["coverage_greedy"] for x in per_instance if x.get("coverage_greedy") is not None]
    grs_g  = [x["guard_ratio_greedy"] for x in per_instance]
    covs_s = [x["coverage_stoch"] for x in per_instance if x.get("coverage_stoch") is not None]
    grs_s  = [x["guard_ratio_stoch"] for x in per_instance]

    msg = f"\nPO eval  (K={args.num_rollouts}, aug={args.aug_factor}, α={args.alpha})"
    if covs_g:
        msg += f"\n  greedy | cov={np.mean(covs_g):.3f} | |S|/n={np.mean(grs_g):.3f}"
    if covs_s:
        msg += f"\n  stoch  | cov={np.mean(covs_s):.3f} | |S|/n={np.mean(grs_s):.3f}"
    msg += f"\n  best   | |S| mean={s['guards']['mean']:.2f} | |S|/n={s['guard_ratio']['mean']:.3f}"
    if s["coverage"]["mean"] is not None:
        msg += f" | cov={s['coverage']['mean']:.3f}"
    if s["approx_ratio"]["mean"] is not None:
        msg += f" | |S|/opt={s['approx_ratio']['mean']:.2f}"

    # -- LS summary --
    if args.local_search:
        ls_removes = [x.get("ls_removes", 0) for x in per_instance]
        ls_adds    = [x.get("ls_adds", 0)    for x in per_instance]
        ls_swaps   = [x.get("ls_swaps", 0)   for x in per_instance]
        ls_deltas  = [x.get("ls_delta_guards", 0) for x in per_instance]
        msg += (
            f"\n  LS     | Δ|S| mean={np.mean(ls_deltas):+.2f}"
            f" | removes={sum(ls_removes)} adds={sum(ls_adds)} swaps={sum(ls_swaps)}"
        )

    print(msg)
    print(f"\nReport -> {out_path}")


if __name__ == "__main__":
    main()
