"""ss_agp_prune.py

Oracle-agnostic Neural Combinatorial Optimization (NCO) solver for the Vertex-Guard
Art Gallery Problem (AGP) using a *pruning* formulation.

Key constraints (by design):
- The policy does NOT use visibility graphs, triangulations, incidence matrices, or coverage values.
- All geometry-specific logic is encapsulated behind a black-box feasibility oracle F(s).
- Learning operates only by proposing removals and observing oracle feedback.

Problem:
- State is a binary selection mask s ∈ {0,1}^n indicating which polygon vertices are guards.
- Start from s = all-ones (all vertices selected).
- Action proposes removing one currently-active vertex.
- Oracle decides whether the new mask is feasible (fully guards the polygon).

Reward:
- +1 for each successful removal.
- Episode ends when no further feasible removals exist (i.e., all remaining active vertices
  have been attempted and rejected).

This file is intentionally modular so the feasibility oracle can be swapped.

"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from dataset import Dataset, agp_read_samples, collate_fn
from eval_reporting import make_report


def _summarize_seconds(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "total_s": 0.0,
            "mean_s": None,
            "median_s": None,
            "p95_s": None,
            "min_s": None,
            "max_s": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "total_s": float(arr.sum()),
        "mean_s": float(arr.mean()),
        "median_s": float(np.median(arr)),
        "p95_s": float(np.percentile(arr, 95)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
    }


# ---------------------------
# Oracle interface
# ---------------------------


class FeasibilityOracle:
    """Black-box feasibility oracle F(s) for AGP.

    Only required method is `is_feasible`. Implementations may optionally provide
    `coverage` for *evaluation only*; training/policy must not use it.
    """

    def is_feasible(self, points: np.ndarray, active: np.ndarray, name: str) -> bool:
        raise NotImplementedError

    def coverage(self, points: np.ndarray, active: np.ndarray, name: str) -> Optional[float]:
        return None


class SkgeomVisibilityOracle(FeasibilityOracle):
    """Default oracle using the existing geometric coverage implementation.

    Geometry is encapsulated here; the rest of the algorithm only consumes boolean
    feasibility feedback.
    """

    def __init__(self, coverage_threshold: float = 0.99, verbose: bool = False):
        self.coverage_threshold = float(coverage_threshold)
        self.verbose = bool(verbose)
        self._warned_import = False
        self._warned_runtime = False

    def coverage(self, points: np.ndarray, active: np.ndarray, name: str) -> Optional[float]:
        try:
            # Import locally to avoid accidental use outside the oracle.
            from utils import evaluate_polygon_visibility_numpy_wo_gt

            if active.size == 0:
                return 0.0
            return float(evaluate_polygon_visibility_numpy_wo_gt(points, active.astype(np.int64), name))
        except ImportError as e:
            if self.verbose and (not self._warned_import):
                self._warned_import = True
                print(
                    f"[oracle] coverage unavailable because utils/skgeom import failed: {e}",
                    file=sys.stderr,
                )
            return None
        except Exception as e:
            if self.verbose and (not self._warned_runtime):
                self._warned_runtime = True
                print(
                    f"[oracle] coverage computation failed (showing first failure only): {e}\n{traceback.format_exc()}",
                    file=sys.stderr,
                )
            return None

    def is_feasible(self, points: np.ndarray, active: np.ndarray, name: str) -> bool:
        cov = self.coverage(points, active, name)
        if cov is None:
            return False
        return cov >= self.coverage_threshold


class SampledVisibilityOracle(FeasibilityOracle):
    """Faster oracle via Monte-Carlo coverage on sampled interior points.

    Still uses skgeom visibility polygons (triangulation-based) to determine which
    sample points are visible from each vertex, but avoids expensive polygon unions.

    Guarantees:
      - A removal is accepted iff sampled_coverage >= coverage_threshold.
      - This is an approximation of true area-coverage (sanity-check / speed mode).
    """

    def __init__(
        self,
        coverage_threshold: float = 0.99,
        n_samples: int = 512,
        seed: int = 0,
        verbose: bool = False,
    ):
        self.coverage_threshold = float(coverage_threshold)
        self.n_samples = int(n_samples)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self._cache: Dict[str, Dict] = {}

    def _get_cache(self, points: np.ndarray, name: str) -> Optional[Dict]:
        cache = self._cache.get(name)
        if cache is not None and cache.get("n") == int(points.shape[0]):
            return cache

        try:
            import skgeom
            from utils import compute_visibility, createPolygon
        except Exception as e:
            if self.verbose:
                print(f"[oracle] sampled oracle unavailable (import failed): {e}", file=sys.stderr)
            self._cache[name] = {"invalid": True, "n": int(points.shape[0])}
            return self._cache[name]

        poly = createPolygon(points)
        if poly is None or poly is False:
            if self.verbose:
                print(f"[oracle] invalid polygon for {name}", file=sys.stderr)
            self._cache[name] = {"invalid": True, "n": int(points.shape[0])}
            return self._cache[name]

        # Build arrangement once.
        arr = skgeom.arrangement.Arrangement()
        try:
            for edge in poly.edges:
                arr.insert(edge)
        except Exception as e:
            if self.verbose:
                print(f"[oracle] arrangement build failed for {name}: {e}", file=sys.stderr)
            self._cache[name] = {"invalid": True, "n": int(points.shape[0])}
            return self._cache[name]

        vs = skgeom.TriangularExpansionVisibility(arr)
        edges = list(poly.edges)
        n = int(points.shape[0])

        # Sample interior points by rejection sampling in bounding box.
        rng = np.random.RandomState((abs(hash(name)) + 9973 * self.seed) % (2**32 - 1))
        xs = points[:, 0]
        ys = points[:, 1]
        minx, maxx = float(xs.min()), float(xs.max())
        miny, maxy = float(ys.min()), float(ys.max())
        samples: List[skgeom.Point2] = []
        max_tries = max(10_000, self.n_samples * 200)
        tries = 0
        while len(samples) < self.n_samples and tries < max_tries:
            tries += 1
            px = rng.uniform(minx, maxx)
            py = rng.uniform(miny, maxy)
            p = skgeom.Point2(float(px), float(py))
            try:
                # In this repo, POSITIVE is used for points inside the polygon.
                if poly.oriented_side(p) == skgeom.Sign.POSITIVE:
                    samples.append(p)
            except Exception:
                continue
        if len(samples) < max(16, self.n_samples // 4):
            if self.verbose:
                print(f"[oracle] too few interior samples for {name}: {len(samples)}/{self.n_samples}", file=sys.stderr)
            self._cache[name] = {"invalid": True, "n": n}
            return self._cache[name]

        # Precompute visibility polygon per vertex (can fail for some vertices).
        eps = 1e-8
        vis_polys: List[Optional[object]] = [None] * n
        for idx in range(n):
            vis_poly, err_idx, _ = compute_visibility(vs, arr, poly, eps, edges, idx)
            if vis_poly is None and self.verbose:
                print(f"[oracle] visibility failed at guard {err_idx} for {name}", file=sys.stderr)
            vis_polys[idx] = vis_poly

        # Boolean visibility matrix: [n, m]
        m = len(samples)
        vis = np.zeros((n, m), dtype=np.bool_)
        for i in range(n):
            vp = vis_polys[i]
            if vp is None:
                continue
            for j, sp in enumerate(samples):
                try:
                    side = vp.oriented_side(sp)
                    vis[i, j] = side != skgeom.Sign.NEGATIVE
                except Exception:
                    vis[i, j] = False

        cache = {
            "invalid": False,
            "n": n,
            "m": m,
            "vis": vis,
            "last_active": None,
            "counts": None,
            "last_cov": None,
        }
        self._cache[name] = cache
        return cache

    def coverage(self, points: np.ndarray, active: np.ndarray, name: str) -> Optional[float]:
        cache = self._get_cache(points, name)
        if cache is None or cache.get("invalid"):
            return None

        vis = cache["vis"]
        m = int(cache["m"])

        active_set = tuple(int(x) for x in active.tolist())
        last_active = cache.get("last_active")
        counts = cache.get("counts")

        if last_active is None or counts is None:
            if len(active_set) == 0:
                cov = 0.0
                cache["last_active"] = active_set
                cache["counts"] = np.zeros(m, dtype=np.int32)
                cache["last_cov"] = cov
                return cov
            counts = vis[np.array(active, dtype=np.int64)].sum(axis=0, dtype=np.int32)
        else:
            last_set = set(last_active)
            cur_set = set(active_set)
            if cur_set == last_set:
                cov = cache.get("last_cov")
                return float(cov) if cov is not None else None
            # Common case in pruning: remove exactly one index.
            if cur_set.issubset(last_set) and (len(last_set) - len(cur_set) == 1):
                removed = int(next(iter(last_set - cur_set)))
                counts = counts - vis[removed].astype(np.int32)
            else:
                # Fallback: recompute.
                if len(active_set) == 0:
                    counts = np.zeros(m, dtype=np.int32)
                else:
                    counts = vis[np.array(active, dtype=np.int64)].sum(axis=0, dtype=np.int32)

        cov = float(np.mean(counts > 0)) if m > 0 else 0.0
        cache["last_active"] = active_set
        cache["counts"] = counts
        cache["last_cov"] = cov
        return cov

    def is_feasible(self, points: np.ndarray, active: np.ndarray, name: str) -> bool:
        cov = self.coverage(points, active, name)
        if cov is None:
            return False
        return float(cov) >= self.coverage_threshold


class HybridOracle(FeasibilityOracle):
    """Fast precheck + exact confirm.

    is_feasible:
      1) Compute sampled coverage (fast). If below (thr + margin) -> reject.
      2) Otherwise run exact oracle to confirm feasibility.

    This keeps correctness w.r.t. the exact oracle, while reducing expensive exact
    calls when many candidate removals are clearly infeasible.
    """

    def __init__(
        self,
        fast_oracle: SampledVisibilityOracle,
        exact_oracle: SkgeomVisibilityOracle,
        margin: float = 0.0,
    ):
        self.fast_oracle = fast_oracle
        self.exact_oracle = exact_oracle
        self.margin = float(margin)

    def coverage(self, points: np.ndarray, active: np.ndarray, name: str) -> Optional[float]:
        # For reporting/sanity checks, return the exact area-based coverage.
        return self.exact_oracle.coverage(points, active, name)

    def is_feasible(self, points: np.ndarray, active: np.ndarray, name: str) -> bool:
        cov_fast = self.fast_oracle.coverage(points, active, name)
        if cov_fast is None:
            return False
        thr = float(self.exact_oracle.coverage_threshold)
        if float(cov_fast) + 1e-12 < (thr + self.margin):
            return False
        return self.exact_oracle.is_feasible(points, active, name)


def _oracle_threshold(oracle: FeasibilityOracle) -> Optional[float]:
    if isinstance(oracle, SkgeomVisibilityOracle):
        return float(oracle.coverage_threshold)
    if isinstance(oracle, SampledVisibilityOracle):
        return float(oracle.coverage_threshold)
    if isinstance(oracle, HybridOracle):
        return float(oracle.exact_oracle.coverage_threshold)
    return None


def _oracle_status_line(oracle: FeasibilityOracle) -> str:
    if isinstance(oracle, SkgeomVisibilityOracle):
        return f"oracle=exact cov=exact thr={oracle.coverage_threshold:.3f}"
    if isinstance(oracle, SampledVisibilityOracle):
        return (
            f"oracle=sampled cov=sampled thr={oracle.coverage_threshold:.3f} "
            f"samples={oracle.n_samples}"
        )
    if isinstance(oracle, HybridOracle):
        return (
            f"oracle=hybrid cov=exact thr={oracle.exact_oracle.coverage_threshold:.3f} "
            f"samples={oracle.fast_oracle.n_samples} margin={oracle.margin:.3f}"
        )
    return "oracle=unknown"


# ---------------------------
# Environment (per-instance)
# ---------------------------


@dataclass
class PruneStep:
    log_prob: torch.Tensor
    entropy: torch.Tensor
    reward: float


class PruningEpisodeEnv:
    """Oracle-driven pruning environment for a single polygon.

    State:
      - s: active guards mask (bool, length n)
      - blocked: vertices that have been tried for removal and rejected (bool, length n)

    Action mask prevents selecting removed or blocked vertices.

    Important monotonicity property:
      If removing vertex v fails at some state s, then v can be permanently blocked
      (it can never become removable later), because future states are subsets.
    """

    def __init__(self, n: int, oracle: FeasibilityOracle, max_steps: Optional[int] = None):
        self.n = int(n)
        self.oracle = oracle
        self.max_steps = int(max_steps) if max_steps is not None else self.n
        self.reset()

    def reset(self) -> None:
        self.s = np.ones(self.n, dtype=np.bool_)
        self.blocked = np.zeros(self.n, dtype=np.bool_)
        self.steps = 0

    def action_mask(self) -> np.ndarray:
        # Can only attempt to remove vertices that are still active and not blocked.
        return self.s & (~self.blocked)

    def done(self) -> bool:
        return (not self.action_mask().any()) or (self.steps >= self.max_steps)

    def step(self, points: np.ndarray, name: str, action: int) -> Tuple[float, bool, Dict]:
        """Attempt removal of `action`.

        Returns:
          reward, done, info
        """

        info: Dict = {}
        self.steps += 1

        if action < 0 or action >= self.n:
            info["invalid_action"] = True
            return 0.0, True, info

        if not self.s[action] or self.blocked[action]:
            info["invalid_action"] = True
            return 0.0, True, info

        # Propose removal.
        candidate = self.s.copy()
        candidate[action] = False
        active = np.flatnonzero(candidate).astype(np.int64)

        feasible = self.oracle.is_feasible(points, active, name)
        if feasible:
            self.s = candidate
            self.blocked[action] = True  # removed; also prevents reselection
            return 1.0, self.done(), {"feasible": True}

        # Not feasible: block permanently and continue.
        self.blocked[action] = True
        return 0.0, self.done(), {"feasible": False}


# ---------------------------
# Policy network (oracle-agnostic)
# ---------------------------


class PrunePolicyNet(nn.Module):
    """Permutation-invariant policy over vertices.

    Inputs:
      - points: [B, N, 2]
      - pad_mask: [B, N] True for real vertices, False for padding
      - s: [B, N] bool active mask
      - blocked: [B, N] bool blocked mask

    Output:
      - logits: [B, N] (masked to -inf outside valid actions)

    The policy does not use coverage/visibility; it only consumes coordinates and the
    combinatorial state bits.
    """

    def __init__(self, hidden_size: int = 128, use_coords: bool = True):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.use_coords = bool(use_coords)

        in_dim = (2 if self.use_coords else 0) + 2  # s, blocked

        self.vertex_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.score = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        points: torch.Tensor,
        pad_mask: torch.Tensor,
        s: torch.Tensor,
        blocked: torch.Tensor,
    ) -> torch.Tensor:
        device = points.device
        B, N, _ = points.shape

        feats: List[torch.Tensor] = []
        if self.use_coords:
            feats.append(points)
        feats.append(s.float().unsqueeze(-1))
        feats.append(blocked.float().unsqueeze(-1))
        x = torch.cat(feats, dim=-1)

        h = self.vertex_mlp(x)  # [B, N, H]

        # masked mean pooling for context
        m = pad_mask.float().unsqueeze(-1)
        denom = m.sum(dim=1).clamp_min(1.0)
        ctx = (h * m).sum(dim=1) / denom  # [B, H]
        ctx = ctx.unsqueeze(1).expand(B, N, self.hidden_size)

        logits = self.score(torch.cat([h, ctx], dim=-1)).squeeze(-1)  # [B, N]

        # Action mask: can remove only active & not blocked & real vertex
        action_mask = pad_mask & s & (~blocked)
        logits = logits.masked_fill(~action_mask, float("-inf"))
        return logits


# ---------------------------
# Helpers
# ---------------------------


def _list_pol_files(path: str) -> List[str]:
    if os.path.isfile(path) and path.endswith(".pol"):
        return [path]
    out: List[str] = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".pol"):
                out.append(os.path.join(root, f))
    out.sort()
    return out


def _read_opt_solution(sol_dir: Optional[str], name: str) -> Optional[List[int]]:
    if sol_dir is None:
        return None
    base = sol_dir
    if os.path.isfile(base):
        base = os.path.dirname(base)
    path = os.path.join(base, f"{name}.solution")
    if not os.path.exists(path):
        return None
    try:
        lines = open(path, "r").read().splitlines()
        if len(lines) >= 2:
            return [int(x) for x in lines[1].split()]
    except Exception:
        return None
    return None


def _make_oracle_from_config(cfg: Dict) -> FeasibilityOracle:
    if cfg["oracle"] == "exact":
        return SkgeomVisibilityOracle(
            coverage_threshold=float(cfg["coverage_threshold"]),
            verbose=bool(cfg.get("oracle_verbose", False)),
        )
    if cfg["oracle"] == "sampled":
        return SampledVisibilityOracle(
            coverage_threshold=float(cfg["coverage_threshold"]),
            n_samples=int(cfg["oracle_samples"]),
            seed=int(cfg["seed"]),
            verbose=bool(cfg.get("oracle_verbose", False)),
        )
    fast = SampledVisibilityOracle(
        coverage_threshold=float(cfg["coverage_threshold"]),
        n_samples=int(cfg["oracle_samples"]),
        seed=int(cfg["seed"]),
        verbose=bool(cfg.get("oracle_verbose", False)),
    )
    exact = SkgeomVisibilityOracle(
        coverage_threshold=float(cfg["coverage_threshold"]),
        verbose=bool(cfg.get("oracle_verbose", False)),
    )
    return HybridOracle(fast_oracle=fast, exact_oracle=exact, margin=float(cfg["oracle_margin"]))


_EVAL_WORKER_STATE: Dict[str, object] = {}


def _init_eval_worker(ckpt_path: str, model_cfg: Dict, oracle_cfg: Dict) -> None:
    global _EVAL_WORKER_STATE

    seed = int(oracle_cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(model_cfg.get("device", "cpu"))
    model = PrunePolicyNet(hidden_size=int(model_cfg["hidden_size"]), use_coords=bool(model_cfg["use_coords"]))

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval().to(device)

    oracle = _make_oracle_from_config(oracle_cfg)
    coverage_eval_oracle = SkgeomVisibilityOracle(
        coverage_threshold=0.0,
        verbose=bool(oracle_cfg.get("oracle_verbose", False)),
    )

    _EVAL_WORKER_STATE = {
        "device": device,
        "model": model,
        "oracle": oracle,
        "coverage_eval_oracle": coverage_eval_oracle,
        "oracle_free": bool(model_cfg.get("oracle_free", False)),
        "oracle_free_k": model_cfg.get("oracle_free_k"),
        "oracle_free_ratio": model_cfg.get("oracle_free_ratio"),
        "oracle_free_threshold": model_cfg.get("oracle_free_threshold"),
    }


def _process_eval_instance(
    idx: int,
    points: np.ndarray,
    name: str,
    max_steps: Optional[int],
    sol_dir_for_ratios: Optional[str],
) -> Dict:
    state = _EVAL_WORKER_STATE
    model: PrunePolicyNet = state["model"]
    oracle: FeasibilityOracle = state["oracle"]
    coverage_eval_oracle: SkgeomVisibilityOracle = state["coverage_eval_oracle"]
    device: torch.device = state["device"]
    oracle_free = bool(state.get("oracle_free", False))
    oracle_free_k = state.get("oracle_free_k")
    oracle_free_ratio = state.get("oracle_free_ratio")
    oracle_free_threshold = state.get("oracle_free_threshold")

    t0 = time.perf_counter()
    pts = torch.tensor(points, device=device)
    if oracle_free:
        active, info = run_policy_oracle_free(
            model,
            pts,
            name,
            top_k=oracle_free_k,
            ratio=oracle_free_ratio,
            threshold=oracle_free_threshold,
            max_steps=max_steps,
        )
    else:
        active, info = run_policy_pruning(model, oracle, pts, name, deterministic=True, max_steps=max_steps)
    dt = float(time.perf_counter() - t0)

    n = int(info.get("n", points.shape[0]))
    cov = None
    try:
        cov = coverage_eval_oracle.coverage(points, active, name)
    except Exception:
        cov = None
    if cov is None and "coverage" in info:
        cov = float(info["coverage"])
    opt_sol = _read_opt_solution(sol_dir_for_ratios, name)
    opt_size = len(opt_sol) if opt_sol is not None else None
    approx_ratio = (float(active.size) / float(opt_size)) if opt_size else None

    return {
        "_idx": int(idx),
        "name": name,
        "n": n,
        "guards": int(active.size),
        "guard_ratio": float(active.size) / max(1.0, float(n)),
        "coverage": cov,
        "opt_size": opt_size,
        "approx_ratio": approx_ratio,
        "time_s": dt,
    }


@torch.no_grad()
def run_policy_pruning(
    model: PrunePolicyNet,
    oracle: FeasibilityOracle,
    points: torch.Tensor,
    name: str,
    deterministic: bool = True,
    max_steps: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """Run a pruning episode to produce a final active set.

    Returns:
      active_indices (np.ndarray), info dict
    """

    device = next(model.parameters()).device
    pts_np = points.detach().cpu().numpy()
    n = int(points.shape[0])
    env = PruningEpisodeEnv(n=n, oracle=oracle, max_steps=max_steps)

    # Skip invalid instance if even all-ones is infeasible.
    init_active = np.arange(n, dtype=np.int64)
    if not oracle.is_feasible(pts_np, init_active, name):
        return init_active, {"initial_infeasible": True}

    while not env.done():
        s = torch.tensor(env.s, device=device, dtype=torch.bool).unsqueeze(0)
        blocked = torch.tensor(env.blocked, device=device, dtype=torch.bool).unsqueeze(0)
        pad_mask = torch.ones(1, n, device=device, dtype=torch.bool)
        pts = points.to(device).unsqueeze(0)

        logits = model(pts, pad_mask, s, blocked)  # [1, n]
        if torch.all(torch.isneginf(logits)):
            break

        if deterministic:
            action = int(torch.argmax(logits, dim=-1).item())
        else:
            dist = Categorical(logits=logits)
            action = int(dist.sample().item())

        _, _, _ = env.step(pts_np, name, action)

    active = np.flatnonzero(env.s).astype(np.int64)
    info = {
        "n": n,
        "guards": int(active.size),
        "removed": int(n - active.size),
    }
    cov = oracle.coverage(pts_np, active, name)
    if cov is not None:
        info["coverage"] = float(cov)
    return active, info


@torch.no_grad()
def run_policy_oracle_free(
    model: PrunePolicyNet,
    points: torch.Tensor,
    name: str,
    *,
    top_k: Optional[int],
    ratio: Optional[float],
    threshold: Optional[float],
    max_steps: Optional[int],
) -> Tuple[np.ndarray, Dict]:
    """Oracle-free selection based on model scores.

    If top-k or ratio is set, uses a single forward pass and keeps the top-scoring
    vertices. Otherwise, performs greedy removals without an oracle, stopping when
    the best score drops below the threshold or max_steps is reached.
    """

    device = next(model.parameters()).device
    n = int(points.shape[0])

    s = torch.ones(1, n, device=device, dtype=torch.bool)
    blocked = torch.zeros(1, n, device=device, dtype=torch.bool)
    pad_mask = torch.ones(1, n, device=device, dtype=torch.bool)
    pts = points.to(device).unsqueeze(0)

    if (top_k is not None and int(top_k) > 0) or (ratio is not None and float(ratio) > 0):
        logits = model(pts, pad_mask, s, blocked).squeeze(0)
        if top_k is not None and int(top_k) > 0:
            k = min(int(top_k), n)
        else:
            k = max(1, int(round(float(ratio) * n)))
        if k >= n:
            active = np.arange(n, dtype=np.int64)
        else:
            _, idx = torch.topk(logits, k)
            active = idx.detach().cpu().numpy().astype(np.int64)
    else:
        thresh = 0.0 if threshold is None else float(threshold)
        steps = 0
        max_steps_eff = n if max_steps is None else int(max_steps)
        while steps < max_steps_eff:
            logits = model(pts, pad_mask, s, blocked).squeeze(0)
            logits = logits.masked_fill(~s.squeeze(0), float("-inf"))
            best_val, best_idx = torch.max(logits, dim=0)
            if torch.isneginf(best_val) or float(best_val.item()) <= thresh:
                break
            s[0, int(best_idx.item())] = False
            steps += 1
        active = torch.nonzero(s.squeeze(0), as_tuple=False).squeeze(-1).detach().cpu().numpy().astype(np.int64)

    info = {
        "n": n,
        "guards": int(active.size),
        "removed": int(n - active.size),
        "oracle_free": True,
    }
    return active, info


# ---------------------------
# Training
# ---------------------------


def train_pruning_policy(
    model: PrunePolicyNet,
    oracle: FeasibilityOracle,
    train_loader: DataLoader,
    val_dataset: Dataset,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    entropy_weight: float,
    baseline_beta: float,
    max_steps: Optional[int],
    eval_k: int,
    log_every: int,
    sol_dir_for_ratios: Optional[str],
    verbose: bool = False,
) -> Dict:
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    baseline = 0.0
    global_step = 0

    history: Dict[str, List[float]] = {
        "train_return": [],
        "train_removed_ratio": [],
        "val_cov": [],
        "val_guard_ratio": [],
        "val_removed_ratio": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        if verbose:
            print(f"[phase] epoch {epoch}/{epochs}: training")
        epoch_returns: List[float] = []
        epoch_removed_ratio: List[float] = []
        epoch_guard_ratio: List[float] = []

        for batch_idx, (points_pad, pad_mask, lengths, names) in enumerate(train_loader, start=1):
            points_pad = points_pad.to(device)
            pad_mask = pad_mask.to(device)

            log_this_batch = bool(log_every > 0 and (batch_idx % log_every == 0))

            batch_losses: List[torch.Tensor] = []
            batch_returns: List[float] = []
            batch_guard_ratios: List[float] = []
            batch_removed_ratios: List[float] = []
            batch_covs: List[float] = []

            for i in range(points_pad.shape[0]):
                n = int(lengths[i])
                name = names[i]
                pts = points_pad[i, :n].detach()  # [n, 2]
                pts_np = pts.cpu().numpy()

                # Env per instance
                env = PruningEpisodeEnv(n=n, oracle=oracle, max_steps=max_steps)

                # Skip invalid instance if even all-ones is infeasible.
                if not oracle.is_feasible(pts_np, np.arange(n, dtype=np.int64), name):
                    continue

                steps: List[PruneStep] = []
                total_reward = 0.0

                while not env.done():
                    s = torch.tensor(env.s, device=device, dtype=torch.bool).unsqueeze(0)
                    blocked = torch.tensor(env.blocked, device=device, dtype=torch.bool).unsqueeze(0)
                    pm = torch.ones(1, n, device=device, dtype=torch.bool)
                    logits = model(pts.unsqueeze(0), pm, s, blocked)

                    if torch.all(torch.isneginf(logits)):
                        break

                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    entropy = dist.entropy()

                    r, _, _ = env.step(pts_np, name, int(action.item()))
                    total_reward += float(r)

                    steps.append(PruneStep(log_prob=log_prob.squeeze(0), entropy=entropy.squeeze(0), reward=float(r)))

                if len(steps) == 0:
                    continue

                # Coverage sanity-check for logging only (policy does NOT use this).
                if log_this_batch:
                    active_final = np.flatnonzero(env.s).astype(np.int64)
                    cov = oracle.coverage(pts_np, active_final, name)
                    if cov is not None:
                        batch_covs.append(float(cov))
                        thr = _oracle_threshold(oracle)
                        if thr is not None and float(cov) + 1e-9 < thr:
                            print(
                                f"[warn] train logged cov {float(cov):.6f} < threshold {thr:.6f} for {name}",
                                file=sys.stderr,
                            )

                # Episode-level REINFORCE
                R = float(total_reward)
                adv = R - baseline

                sum_logp = torch.stack([s_.log_prob for s_ in steps]).sum()
                sum_ent = torch.stack([s_.entropy for s_ in steps]).sum()

                loss = -(adv * sum_logp) - (entropy_weight * sum_ent)
                batch_losses.append(loss)
                batch_returns.append(R)
                batch_removed_ratios.append(R / max(1.0, float(n)))
                batch_guard_ratios.append((float(n) - R) / max(1.0, float(n)))

                epoch_returns.append(R)
                epoch_removed_ratio.append(R / max(1.0, float(n)))
                epoch_guard_ratio.append((float(n) - R) / max(1.0, float(n)))

            if len(batch_losses) == 0:
                continue

            loss_batch = torch.stack(batch_losses).mean()
            opt.zero_grad(set_to_none=True)
            loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            global_step += 1

            # EMA baseline update
            mean_R = float(np.mean(batch_returns)) if batch_returns else 0.0
            baseline = float(baseline_beta * baseline + (1.0 - baseline_beta) * mean_R)

            if log_this_batch:
                mean_guard_ratio = float(np.mean(batch_guard_ratios)) if batch_guard_ratios else float("nan")
                mean_removed_ratio = float(np.mean(batch_removed_ratios)) if batch_removed_ratios else float("nan")
                mean_cov = float(np.mean(batch_covs)) if batch_covs else float("nan")
                min_cov = float(np.min(batch_covs)) if batch_covs else float("nan")
                print(
                    f"[train] epoch {epoch:02d}/{epochs} batch {batch_idx:03d}/{len(train_loader)} "
                    f"| cov {'unavailable' if not batch_covs else f'{mean_cov:.3f} (min {min_cov:.3f})'} "
                    f"| |S|/n {mean_guard_ratio:.3f} "
                    f"| removed/n {mean_removed_ratio:.3f} "
                    f"| loss {loss_batch.item():.3f} "
                    f"| advantage {mean_R - baseline:.2f} "
                    f"| {_oracle_status_line(oracle)}"
                )

        # Epoch aggregates (kept for results.json / plotting if needed)
        mean_epoch_R = float(np.mean(epoch_returns)) if epoch_returns else 0.0
        mean_epoch_rr = float(np.mean(epoch_removed_ratio)) if epoch_removed_ratio else 0.0
        history["train_return"].append(mean_epoch_R)
        history["train_removed_ratio"].append(mean_epoch_rr)

        # Evaluation
        model.eval()
        if verbose:
            print(f"[phase] epoch {epoch}/{epochs}: validation")
        val_k = len(val_dataset) if eval_k is None or eval_k < 0 else min(int(eval_k), len(val_dataset))
        if val_k > 0:
            covs: List[float] = []
            guard_ratios: List[float] = []
            ratios: List[float] = []

            t_val0 = time.perf_counter()
            policy_secs: List[float] = []
            for j in range(val_k):
                pts, _, name = val_dataset[j]

                t0 = time.perf_counter()
                active, info = run_policy_pruning(model, oracle, pts, name, deterministic=True, max_steps=max_steps)
                policy_secs.append(float(time.perf_counter() - t0))

                guard_ratios.append(float(active.size) / max(1.0, float(info.get("n", pts.shape[0]))))
                if "coverage" in info:
                    covs.append(float(info["coverage"]))

                # OPT ratios if available
                opt_sol = _read_opt_solution(sol_dir_for_ratios, name)
                if opt_sol is not None and len(opt_sol) > 0:
                    ratios.append(float(active.size) / float(len(opt_sol)))

            mean_cov = float(np.mean(covs)) if covs else float("nan")
            mean_guard_ratio = float(np.mean(guard_ratios)) if guard_ratios else float("nan")
            mean_removed_ratio = float("nan")
            if guard_ratios:
                mean_removed_ratio = float(1.0 - mean_guard_ratio)

            history["val_cov"].append(mean_cov)
            history["val_guard_ratio"].append(mean_guard_ratio)
            history["val_removed_ratio"].append(mean_removed_ratio)

            msg = f"[val] k={val_k}"
            if covs:
                msg += f" | cov mean={mean_cov:.3f} (min {float(np.min(covs)):.3f})"
            else:
                msg += " | cov unavailable"
            msg += f" | |S|/n mean={mean_guard_ratio:.3f}"
            msg += f" | removed/n mean={mean_removed_ratio:.3f}"
            if ratios:
                msg += f" | |S|/opt mean={float(np.mean(ratios)):.3f}"

            val_wall_s = float(time.perf_counter() - t_val0)
            pstats = _summarize_seconds(policy_secs)
            msg += f" | time {val_wall_s:.2f}s (mean {pstats['mean_s']:.3f}s, p95 {pstats['p95_s']:.3f}s)"
            msg += f" | {_oracle_status_line(oracle)}"
            print(msg)

    return {
        "history": history,
        "final_baseline": baseline,
    }


def main() -> None:
    load_dotenv()
    DATASET_PATH = os.getenv("DATASET_PATH")
    if not DATASET_PATH:
        raise EnvironmentError("DATASET_PATH environment variable must be set in .env file.")
    
    default_train = os.path.join(DATASET_PATH, "train")
    default_val = os.path.join(DATASET_PATH, "dev")

    parser = argparse.ArgumentParser(description="Oracle-agnostic pruning NCO for AGP")

    parser.add_argument("--agp_train_dir", type=str, default=default_train, help="Training path (dir of .pol or a single .pol).")
    parser.add_argument("--agp_val_dir", type=str, default=default_val, help="Validation path (dir of .pol or a single .pol).")
    parser.add_argument("--normalize", action="store_true", help="Normalize polygon coordinates when loading .pol files.")

    parser.add_argument(
        "--train-size",
        type=int,
        default=-1,
        help="Cap on number of training samples. Use -1 for full training set (default: -1).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)

    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--no-coords", action="store_true", help="Do not feed vertex coordinates to the policy (state-only ablation).")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--baseline-beta", type=float, default=0.99)

    parser.add_argument("--coverage-threshold", type=float, default=0.99, help="Oracle feasibility threshold.")

    parser.add_argument(
        "--oracle",
        type=str,
        default="exact",
        choices=["exact", "sampled", "hybrid"],
        help=(
            "Feasibility oracle mode: "
            "exact (slow, area-based), "
            "sampled (fast, approximate), "
            "hybrid (fast sampled precheck + exact confirm)."
        ),
    )
    parser.add_argument(
        "--oracle-samples",
        type=int,
        default=512,
        help="Number of interior samples for --oracle sampled.",
    )

    parser.add_argument(
        "--oracle-margin",
        type=float,
        default=0.0,
        help="Extra margin for sampled precheck in --oracle hybrid (reject if sampled cov < thr + margin).",
    )

    parser.add_argument(
        "--oracle-verbose",
        action="store_true",
        help="Print a warning if coverage computation is unavailable/failing (first failure only).",
    )

    parser.add_argument("--max-steps", type=int, default=-1, help="Cap steps per episode (default: n).")

    parser.add_argument(
        "--eval-k",
        type=int,
        default=-1,
        help="How many polygons to evaluate each epoch. Use -1 for full validation set (default: -1).",
    )
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument("--evaluate", action="store_true", help="Skip training and only run evaluation with a checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a PrunePolicyNet checkpoint for --evaluate mode.")
    parser.add_argument(
        "--oracle-free",
        action="store_true",
        help="Run oracle-free one-pass inference during evaluation (no feasibility checks).",
    )
    parser.add_argument(
        "--oracle-free-k",
        type=int,
        default=-1,
        help="Oracle-free: keep top-k vertices by model score (default: -1).",
    )
    parser.add_argument(
        "--oracle-free-ratio",
        type=float,
        default=-1.0,
        help="Oracle-free: keep top ratio of vertices by model score (default: -1).",
    )
    parser.add_argument(
        "--oracle-free-threshold",
        type=float,
        default=0.0,
        help="Oracle-free: greedy removals stop when best score <= threshold (default: 0.0).",
    )

    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for evaluation (default: 1).")

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/v3",
        help="Directory to write evaluation JSON reports.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print high-level phases (data load, oracle init, training/val epochs, checkpoint, final eval).",
    )

    parser.add_argument("--seed", type=int, default=0)    

    args = parser.parse_args()

    def vprint(msg: str) -> None:
        if bool(args.verbose):
            print(msg)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vprint(f"[phase] dataset path: {DATASET_PATH}")
    vprint(f"[phase] train_dir: {args.agp_train_dir}")
    vprint(f"[phase] val_dir: {args.agp_val_dir}")

    if args.agp_val_dir is None:
        raise SystemExit("Please provide --agp_val_dir")
    if (not args.evaluate) and args.agp_train_dir is None:
        raise SystemExit("Please provide --agp_train_dir")

    train_paths: List[str] = []
    val_paths = _list_pol_files(args.agp_val_dir)

    if not args.evaluate:
        train_paths = _list_pol_files(args.agp_train_dir)

    vprint(f"[phase] discovered {len(train_paths)} train .pol files")
    vprint(f"[phase] discovered {len(val_paths)} val .pol files")

    if len(val_paths) == 0:
        raise SystemExit(f"No .pol files found under {args.agp_val_dir}")
    if (not args.evaluate) and len(train_paths) == 0:
        raise SystemExit(f"No .pol files found under {args.agp_train_dir}")

    if not args.evaluate and int(args.train_size) > 0:
        train_paths = train_paths[: int(args.train_size)]
        vprint(f"[phase] using {len(train_paths)} train files (train-size cap)")

    vprint(f"[phase] loading samples (normalize={bool(args.normalize)})")

    if not args.evaluate:
        train_samples = agp_read_samples(train_paths, normalize=bool(args.normalize))
        train_dataset = Dataset(train_samples)
        train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_fn)
        vprint(
            f"[phase] dataloader ready: batch_size={int(args.batch_size)}, batches/epoch≈{len(train_loader)}"
        )
    val_samples = agp_read_samples(val_paths, normalize=bool(args.normalize))
    val_dataset = Dataset(val_samples)

    vprint(
        f"[phase] oracle init: mode={args.oracle}, threshold={float(args.coverage_threshold):.3f}, samples={int(args.oracle_samples)}"
    )

    if args.oracle == "exact":
        oracle = SkgeomVisibilityOracle(
            coverage_threshold=float(args.coverage_threshold),
            verbose=bool(args.oracle_verbose),
        )
    elif args.oracle == "sampled":
        oracle = SampledVisibilityOracle(
            coverage_threshold=float(args.coverage_threshold),
            n_samples=int(args.oracle_samples),
            seed=int(args.seed),
            verbose=bool(args.oracle_verbose),
        )
    else:
        fast = SampledVisibilityOracle(
            coverage_threshold=float(args.coverage_threshold),
            n_samples=int(args.oracle_samples),
            seed=int(args.seed),
            verbose=bool(args.oracle_verbose),
        )
        exact = SkgeomVisibilityOracle(
            coverage_threshold=float(args.coverage_threshold),
            verbose=bool(args.oracle_verbose),
        )
        oracle = HybridOracle(fast_oracle=fast, exact_oracle=exact, margin=float(args.oracle_margin))

    # Coverage metric: always try exact coverage for comparable reporting.
    # This is independent of the feasibility oracle used during pruning.
    coverage_eval_oracle = SkgeomVisibilityOracle(coverage_threshold=0.0, verbose=bool(args.oracle_verbose))

    model = PrunePolicyNet(hidden_size=int(args.hidden_size), use_coords=(not args.no_coords))

    vprint(
        f"[phase] model init: hidden={int(args.hidden_size)}, use_coords={not args.no_coords}, lr={float(args.lr)}, ent={float(args.entropy_weight)}"
    )

    max_steps = None if int(args.max_steps) < 0 else int(args.max_steps)

    # For OPT ratio reporting we reuse val dir (expects .solution alongside .pol)
    sol_dir_for_ratios = args.agp_val_dir

    if args.evaluate:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required when using --evaluate")

        oracle_free = bool(args.oracle_free)
        oracle_free_k = int(args.oracle_free_k) if int(args.oracle_free_k) > 0 else None
        oracle_free_ratio = float(args.oracle_free_ratio) if float(args.oracle_free_ratio) > 0 else None
        oracle_free_threshold = float(args.oracle_free_threshold)

        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval().to(device)

        eval_k = len(val_dataset) if args.eval_k < 0 else min(int(args.eval_k), len(val_dataset))

        per_instance: List[Dict] = []

        t_eval0 = time.perf_counter()
        policy_secs: List[float] = []

        if int(args.workers) <= 1:
            for j in range(eval_k):
                pts, _, name = val_dataset[j]
                vprint(f"[eval] {j + 1}/{eval_k}: {name}")
                t0 = time.perf_counter()
                if oracle_free:
                    active, info = run_policy_oracle_free(
                        model,
                        pts,
                        name,
                        top_k=oracle_free_k,
                        ratio=oracle_free_ratio,
                        threshold=oracle_free_threshold,
                        max_steps=max_steps,
                    )
                else:
                    active, info = run_policy_pruning(model, oracle, pts, name, deterministic=True, max_steps=max_steps)
                dt = float(time.perf_counter() - t0)
                policy_secs.append(dt)

                n = int(info.get("n", pts.shape[0]))
                cov = None
                try:
                    cov = coverage_eval_oracle.coverage(pts.detach().cpu().numpy(), active, name)
                except Exception:
                    cov = None
                if cov is None and "coverage" in info:
                    cov = float(info["coverage"])
                opt_sol = _read_opt_solution(sol_dir_for_ratios, name)
                opt_size = len(opt_sol) if opt_sol is not None else None
                approx_ratio = (float(active.size) / float(opt_size)) if opt_size else None

                per_instance.append(
                    {
                        "name": name,
                        "n": n,
                        "guards": int(active.size),
                        "guard_ratio": float(active.size) / max(1.0, float(n)),
                        "coverage": cov,
                        "opt_size": opt_size,
                        "approx_ratio": approx_ratio,
                        "time_s": dt,
                    }
                )
        else:
            oracle_cfg = {
                "oracle": str(args.oracle),
                "coverage_threshold": float(args.coverage_threshold),
                "oracle_samples": int(args.oracle_samples),
                "oracle_margin": float(args.oracle_margin),
                "oracle_verbose": bool(args.oracle_verbose),
                "seed": int(args.seed),
            }
            model_cfg = {
                "hidden_size": int(args.hidden_size),
                "use_coords": not args.no_coords,
                "device": "cpu",
                "oracle_free": oracle_free,
                "oracle_free_k": oracle_free_k,
                "oracle_free_ratio": oracle_free_ratio,
                "oracle_free_threshold": oracle_free_threshold,
            }
            with ProcessPoolExecutor(
                max_workers=int(args.workers),
                initializer=_init_eval_worker,
                initargs=(args.checkpoint, model_cfg, oracle_cfg),
            ) as ex:
                jobs = []
                for j in range(eval_k):
                    pts, _, name = val_dataset[j]
                    pts_np = pts.detach().cpu().numpy() if hasattr(pts, "detach") else np.asarray(pts)
                    jobs.append(
                        ex.submit(
                            _process_eval_instance,
                            j,
                            pts_np,
                            name,
                            max_steps,
                            sol_dir_for_ratios,
                        )
                    )

                for fut in as_completed(jobs):
                    res = fut.result()
                    vprint(f"[eval] {res['_idx'] + 1}/{eval_k}: {res['name']}")
                    policy_secs.append(float(res["time_s"]))
                    per_instance.append(
                        {
                            "name": res["name"],
                            "n": int(res["n"]),
                            "guards": int(res["guards"]),
                            "guard_ratio": float(res["guard_ratio"]),
                            "coverage": res.get("coverage"),
                            "opt_size": res.get("opt_size"),
                            "approx_ratio": res.get("approx_ratio"),
                            "time_s": float(res["time_s"]),
                        }
                    )

        eval_wall_s = float(time.perf_counter() - t_eval0)

        report = make_report(
            method="ss_agp_prune",
            per_instance=per_instance,
            args=vars(args),
            dataset={
                "path": args.agp_val_dir,
                "eval_k": int(eval_k),
            },
            oracle={
                "mode": str(args.oracle),
                "coverage_threshold": float(args.coverage_threshold),
                "oracle_samples": int(args.oracle_samples),
                "oracle_margin": float(args.oracle_margin),
            },
            timing={
                "wall_total_s": eval_wall_s,
            },
        )
        report["checkpoint"] = args.checkpoint

        os.makedirs(args.results_dir, exist_ok=True)
        eval_tag = "full" if eval_k >= len(val_dataset) else str(eval_k)
        out_path = os.path.join(args.results_dir, f"ss_agp_prune_eval_only_val{eval_tag}.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

        print("\n--- Eval on dataset polygons (checkpoint mode) ---")
        s = report["summary"]
        msg = f"Dataset eval k={eval_k} | |S| mean={s['guards']['mean']:.2f} | |S|/n mean={s['guard_ratio']['mean']:.3f}"
        if s["coverage"]["mean"] is not None:
            msg += f" | geo-cov mean={s['coverage']['mean']:.3f}"
        if s["approx_ratio"]["mean"] is not None:
            msg += f" | |S|/opt mean={s['approx_ratio']['mean']:.2f}"
        msg += f" | time {report['timing']['wall_total_s']:.2f}s (mean {s['time_s']['mean']:.3f}s, p95 {s['time_s']['p95']:.3f}s)"
        print(msg)
        print(f"Results summary saved to {out_path}")
        return

    result = train_pruning_policy(
        model,
        oracle,
        train_loader,
        val_dataset,
        device,
        epochs=int(args.epochs),
        lr=float(args.lr),
        entropy_weight=float(args.entropy_weight),
        baseline_beta=float(args.baseline_beta),
        max_steps=max_steps,
        eval_k=int(args.eval_k),
        log_every=int(args.log_every),
        sol_dir_for_ratios=sol_dir_for_ratios,
        verbose=bool(args.verbose),
    )

    vprint("[phase] training complete")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Save model
    train_tag = "full" if int(args.train_size) <= 0 else str(len(train_paths))
    ckpt_path = os.path.join(
        "checkpoints",
        f"ss_agp_prune_h{args.hidden_size}_lr{args.lr}_ent{args.entropy_weight}_epochs{args.epochs}_train{train_tag}.pt",
    )
    torch.save({"model": model.state_dict(), "args": vars(args), "result": result}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    vprint("[phase] starting final evaluation")

    # Final eval on validation set
    model.eval().to(device)
    eval_k = len(val_dataset) if args.eval_k < 0 else min(int(args.eval_k), len(val_dataset))

    oracle_free = bool(args.oracle_free)
    oracle_free_k = int(args.oracle_free_k) if int(args.oracle_free_k) > 0 else None
    oracle_free_ratio = float(args.oracle_free_ratio) if float(args.oracle_free_ratio) > 0 else None
    oracle_free_threshold = float(args.oracle_free_threshold)

    per_instance: List[Dict] = []

    t_eval0 = time.perf_counter()
    policy_secs: List[float] = []

    if int(args.workers) <= 1:
        for j in range(eval_k):
            pts, _, name = val_dataset[j]
            vprint(f"[eval] {j + 1}/{eval_k}: {name}")
            t0 = time.perf_counter()
            if oracle_free:
                active, info = run_policy_oracle_free(
                    model,
                    pts,
                    name,
                    top_k=oracle_free_k,
                    ratio=oracle_free_ratio,
                    threshold=oracle_free_threshold,
                    max_steps=max_steps,
                )
            else:
                active, info = run_policy_pruning(model, oracle, pts, name, deterministic=True, max_steps=max_steps)
            dt = float(time.perf_counter() - t0)
            policy_secs.append(dt)

            n = int(info.get("n", pts.shape[0]))
            cov = None
            try:
                cov = coverage_eval_oracle.coverage(pts.detach().cpu().numpy(), active, name)
            except Exception:
                cov = None
            if cov is None and "coverage" in info:
                cov = float(info["coverage"])
            opt_sol = _read_opt_solution(sol_dir_for_ratios, name)
            opt_size = len(opt_sol) if opt_sol is not None else None
            approx_ratio = (float(active.size) / float(opt_size)) if opt_size else None

            per_instance.append(
                {
                    "name": name,
                    "n": n,
                    "guards": int(active.size),
                    "guard_ratio": float(active.size) / max(1.0, float(n)),
                    "coverage": cov,
                    "opt_size": opt_size,
                    "approx_ratio": approx_ratio,
                    "time_s": dt,
                }
            )
    else:
        oracle_cfg = {
            "oracle": str(args.oracle),
            "coverage_threshold": float(args.coverage_threshold),
            "oracle_samples": int(args.oracle_samples),
            "oracle_margin": float(args.oracle_margin),
            "oracle_verbose": bool(args.oracle_verbose),
            "seed": int(args.seed),
        }
        model_cfg = {
            "hidden_size": int(args.hidden_size),
            "use_coords": not args.no_coords,
            "device": "cpu",
            "oracle_free": oracle_free,
            "oracle_free_k": oracle_free_k,
            "oracle_free_ratio": oracle_free_ratio,
            "oracle_free_threshold": oracle_free_threshold,
        }
        with ProcessPoolExecutor(
            max_workers=int(args.workers),
            initializer=_init_eval_worker,
            initargs=(ckpt_path, model_cfg, oracle_cfg),
        ) as ex:
            jobs = []
            for j in range(eval_k):
                pts, _, name = val_dataset[j]
                pts_np = pts.detach().cpu().numpy() if hasattr(pts, "detach") else np.asarray(pts)
                jobs.append(
                    ex.submit(
                        _process_eval_instance,
                        j,
                        pts_np,
                        name,
                        max_steps,
                        sol_dir_for_ratios,
                    )
                )

            for fut in as_completed(jobs):
                res = fut.result()
                vprint(f"[eval] {res['_idx'] + 1}/{eval_k}: {res['name']}")
                policy_secs.append(float(res["time_s"]))
                per_instance.append(
                    {
                        "name": res["name"],
                        "n": int(res["n"]),
                        "guards": int(res["guards"]),
                        "guard_ratio": float(res["guard_ratio"]),
                        "coverage": res.get("coverage"),
                        "opt_size": res.get("opt_size"),
                        "approx_ratio": res.get("approx_ratio"),
                        "time_s": float(res["time_s"]),
                    }
                )

    eval_wall_s = float(time.perf_counter() - t_eval0)

    report = make_report(
        method="ss_agp_prune",
        per_instance=per_instance,
        args=vars(args),
        dataset={
            "path": args.agp_val_dir,
            "eval_k": int(eval_k),
            "train_k": int(len(train_paths)),
        },
        oracle={
            "mode": str(args.oracle),
            "coverage_threshold": float(args.coverage_threshold),
            "oracle_samples": int(args.oracle_samples),
            "oracle_margin": float(args.oracle_margin),
            "coverage_metric": "exact",
        },
        timing={
            "wall_total_s": eval_wall_s,
        },
    )
    report["checkpoint"] = ckpt_path

    eval_tag = "full" if eval_k >= len(val_dataset) else str(eval_k)
    out_path = os.path.join(args.results_dir, f"ss_agp_prune_evaluation_train{train_tag}_val{eval_tag}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print("\n--- Eval on dataset polygons ---")
    s = report["summary"]
    msg = f"Dataset eval k={eval_k} | |S| mean={s['guards']['mean']:.2f} | |S|/n mean={s['guard_ratio']['mean']:.3f}"
    if s["coverage"]["mean"] is not None:
        msg += f" | geo-cov mean={s['coverage']['mean']:.3f}"
    if s["approx_ratio"]["mean"] is not None:
        msg += f" | |S|/opt mean={s['approx_ratio']['mean']:.2f}"
    msg += f" | time {report['timing']['wall_total_s']:.2f}s (mean {s['time_s']['mean']:.3f}s, p95 {s['time_s']['p95']:.3f}s)"
    print(msg)
    print(f"Results summary saved to {out_path}")

    vprint("[phase] done")


if __name__ == "__main__":
    main()
