"""qlearning_prune.py

Tabular Q-learning baseline for AGP in a *pruning* formulation.

Goal: be comparable to ss_agp_prune.py (pruning/removal-based).

Key properties:
- Starts from all vertices active.
- Actions propose REMOVING a single vertex.
- Transition uses a feasibility oracle F(s): a removal is accepted iff F(s\{v}) is True.
- Reward: +1 for each successful removal (same objective as pruning RL).

Notes:
- The tabular state is compressed to an integer (number of active guards). This is an
  approximation (non-Markov w.r.t. the true subset state), but keeps the approach
  simple and fast.

"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

from dataset import Dataset, agp_read_samples
from ss_agp_prune import SampledVisibilityOracle, SkgeomVisibilityOracle, HybridOracle
from eval_reporting import make_report


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


def _oracle_status_line(oracle) -> str:
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


def _make_oracle_from_config(cfg: Dict) -> object:
    if cfg["oracle"] == "exact":
        return SkgeomVisibilityOracle(coverage_threshold=float(cfg["coverage_threshold"]), verbose=bool(cfg.get("oracle_verbose", False)))
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
    exact = SkgeomVisibilityOracle(coverage_threshold=float(cfg["coverage_threshold"]), verbose=bool(cfg.get("oracle_verbose", False)))
    return HybridOracle(fast_oracle=fast, exact_oracle=exact, margin=float(cfg["oracle_margin"]))


def _process_single_instance(
    idx: int,
    points: np.ndarray,
    name: str,
    n_total: int,
    cfg: Dict,
) -> Dict:
    oracle = _make_oracle_from_config(cfg)
    coverage_eval_oracle = SkgeomVisibilityOracle(coverage_threshold=0.0, verbose=bool(cfg.get("oracle_verbose", False)))

    res = evaluate_qlearning_prune_single_instance(
        oracle,
        coverage_eval_oracle,
        points,
        name,
        max_episodes=int(cfg["max_episodes"]),
        max_steps=cfg["max_steps"],
        lr=float(cfg["lr"]),
        epsilon=float(cfg["epsilon"]),
        gamma=float(cfg["gamma"]),
        epsilon_decay=float(cfg["epsilon_decay"]),
        epsilon_min=float(cfg["epsilon_min"]),
        seed=int(cfg["seed"]),
        patience=int(cfg["patience"]),
        verbose=bool(cfg.get("verbose", False)),
    )

    res["_idx"] = int(idx)
    res["_n_total"] = int(n_total)
    res["_status"] = _oracle_status_line(oracle)
    return res


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
            # file format used across repo: second line has indices
            parts = lines[1].strip().split()
            return [int(x) for x in parts]
    except Exception:
        return None
    return None


@dataclass
class EpisodeResult:
    active: np.ndarray
    removed: int
    steps: int


class QLearningPruneAGP:
    """Tabular Q-learning for pruning.

    State: k = number of active guards (int)
    Action: vertex index to try removing

    Environment dynamics (with monotonic blocking):
    - If removal fails at some state, we block that vertex for the rest of the episode
      since future active sets are subsets and the same removal cannot become feasible.
    """

    def __init__(
        self,
        n_vertices: int,
        learning_rate: float = 0.1,
        epsilon: float = 0.3,
        gamma: float = 0.9,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        seed: int = 0,
    ):
        self.n = int(n_vertices)
        self.alpha = float(learning_rate)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.epsilon_decay = float(epsilon_decay)
        self.epsilon_min = float(epsilon_min)
        self.rng = random.Random(int(seed))

        # q_table[state_k][action_v] -> q_value
        self.q_table: Dict[int, Dict[int, float]] = {}

        # best episode outcome
        self.best_active: Optional[np.ndarray] = None
        self.best_removed: int = -1

    def _q(self, k: int, v: int) -> float:
        return self.q_table.get(int(k), {}).get(int(v), 0.0)

    def _set_q(self, k: int, v: int, value: float) -> None:
        k = int(k)
        v = int(v)
        if k not in self.q_table:
            self.q_table[k] = {}
        self.q_table[k][v] = float(value)

    def _max_q_over_actions(self, k: int, actions: List[int]) -> float:
        if not actions:
            return 0.0
        return max(self._q(k, a) for a in actions)

    def _choose_action(self, k: int, actions: List[int]) -> int:
        # epsilon-greedy among valid actions
        if not actions:
            raise ValueError("No valid actions")
        if self.rng.random() < self.epsilon:
            return self.rng.choice(actions)
        # exploit
        best_a = actions[0]
        best_q = self._q(k, best_a)
        for a in actions[1:]:
            qa = self._q(k, a)
            if qa > best_q:
                best_q = qa
                best_a = a
        return best_a

    def train_episode(
        self,
        oracle,
        points: np.ndarray,
        name: str,
        max_steps: Optional[int],
    ) -> EpisodeResult:
        s = np.ones(self.n, dtype=np.bool_)
        blocked = np.zeros(self.n, dtype=np.bool_)
        steps = 0

        # Ensure initial all-ones is feasible; otherwise episode is degenerate.
        init_active = np.arange(self.n, dtype=np.int64)
        if not oracle.is_feasible(points, init_active, name):
            return EpisodeResult(active=init_active, removed=0, steps=0)

        while True:
            valid = np.flatnonzero(s & (~blocked)).astype(np.int64)
            if valid.size == 0:
                break
            if max_steps is not None and steps >= int(max_steps):
                break

            k = int(valid.size)  # approximate state: current number of active guards
            actions = valid.tolist()
            a = int(self._choose_action(k, actions))

            candidate = s.copy()
            candidate[a] = False
            active_idx = np.flatnonzero(candidate).astype(np.int64)

            feasible = oracle.is_feasible(points, active_idx, name)
            reward = 1.0 if feasible else 0.0

            if feasible:
                next_k = int(active_idx.size)
                next_valid = np.flatnonzero(candidate & (~blocked)).astype(np.int64).tolist()
            else:
                # action a is now known infeasible in this monotone pruning setting
                blocked[a] = True
                next_k = k
                next_valid = np.flatnonzero(s & (~blocked)).astype(np.int64).tolist()

            # Q-learning update
            current_q = self._q(k, a)
            max_next = self._max_q_over_actions(next_k, next_valid)
            target = reward + self.gamma * max_next
            self._set_q(k, a, current_q + self.alpha * (target - current_q))

            if feasible:
                s = candidate

            steps += 1

        active_final = np.flatnonzero(s).astype(np.int64)
        removed = int(self.n - active_final.size)
        return EpisodeResult(active=active_final, removed=removed, steps=steps)

    def train(
        self,
        oracle,
        points: np.ndarray,
        name: str,
        max_episodes: int,
        max_steps: Optional[int],
        target_removed: Optional[int] = None,
        patience: int = 30,
        verbose: bool = False,
    ) -> EpisodeResult:
        best_removed = -1
        episodes_without_improve = 0

        for ep in range(int(max_episodes)):
            res = self.train_episode(oracle, points, name, max_steps=max_steps)

            # track best
            if res.removed > best_removed:
                best_removed = res.removed
                self.best_active = res.active
                self.best_removed = res.removed
                episodes_without_improve = 0
            else:
                episodes_without_improve += 1

            # decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if verbose and (ep + 1) % 20 == 0:
                print(
                    f"  [ql-prune] ep {ep+1}/{max_episodes} | best removed={best_removed} | eps={self.epsilon:.3f}"
                )

            if target_removed is not None and best_removed >= int(target_removed):
                break
            if episodes_without_improve >= int(patience):
                break

        if self.best_active is None:
            # fallback
            self.best_active = np.arange(self.n, dtype=np.int64)
            self.best_removed = 0

        return EpisodeResult(active=self.best_active, removed=int(self.best_removed), steps=0)


def evaluate_qlearning_prune_single_instance(
    oracle,
    coverage_eval_oracle,
    points: np.ndarray,
    name: str,
    *,
    max_episodes: int,
    max_steps: Optional[int],
    lr: float,
    epsilon: float,
    gamma: float,
    epsilon_decay: float,
    epsilon_min: float,
    seed: int,
    patience: int,
    verbose: bool,
) -> Dict:
    n = int(points.shape[0])
    agent = QLearningPruneAGP(
        n_vertices=n,
        learning_rate=lr,
        epsilon=epsilon,
        gamma=gamma,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )

    t0 = time.perf_counter()
    res = agent.train(
        oracle,
        points,
        name,
        max_episodes=max_episodes,
        max_steps=max_steps,
        patience=patience,
        verbose=verbose,
    )
    wall_s = float(time.perf_counter() - t0)

    cov = None
    try:
        cov = coverage_eval_oracle.coverage(points, res.active, name)
    except Exception:
        cov = None
    if cov is None:
        cov = oracle.coverage(points, res.active, name)
    out = {
        "name": name,
        "n": n,
        "guards": int(res.active.size),
        "removed": int(res.removed),
        "removed_ratio": float(res.removed) / max(1.0, float(n)),
        "active": res.active.tolist(),
        "coverage": float(cov) if cov is not None else None,
        "time_s": wall_s,
        "hyperparameters": {
            "max_episodes": int(max_episodes),
            "max_steps": None if max_steps is None else int(max_steps),
            "lr": float(lr),
            "epsilon": float(epsilon),
            "gamma": float(gamma),
            "epsilon_decay": float(epsilon_decay),
            "epsilon_min": float(epsilon_min),
            "patience": int(patience),
            "seed": int(seed),
        },
    }
    return out


def main() -> None:
    load_dotenv()
    DATASET_PATH = os.getenv("DATASET_PATH")
    if not DATASET_PATH:
        raise EnvironmentError("DATASET_PATH environment variable must be set in .env file.")

    default_val = os.path.join(DATASET_PATH, "dev")

    parser = argparse.ArgumentParser(description="Tabular Q-learning pruning baseline for AGP")

    parser.add_argument("--agp_val_dir", type=str, default=default_val, help="Validation path (dir of .pol or a single .pol).")
    parser.add_argument("--normalize", action="store_true", help="Normalize polygon coordinates when loading .pol files.")

    parser.add_argument("--coverage-threshold", type=float, default=0.99, help="Oracle feasibility threshold.")
    parser.add_argument(
        "--oracle",
        type=str,
        default="exact",
        choices=["exact", "sampled", "hybrid"],
        help="Feasibility oracle mode.",
    )
    parser.add_argument("--oracle-samples", type=int, default=512, help="Number of interior samples for --oracle sampled.")
    parser.add_argument("--oracle-margin", type=float, default=0.0, help="Extra margin for sampled precheck in --oracle hybrid.")
    parser.add_argument("--oracle-verbose", action="store_true", help="Print oracle warnings if coverage computation fails.")

    parser.add_argument("--eval-k", type=int, default=-1, help="How many polygons to evaluate. Use -1 for full set.")
    parser.add_argument("--max-episodes", type=int, default=200, help="Max Q-learning episodes per instance.")
    parser.add_argument("--max-steps", type=int, default=-1, help="Cap steps per episode (default: n).")
    parser.add_argument("--patience", type=int, default=40, help="Stop if no improvement for this many episodes.")

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.01)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for evaluation (default: 1).")

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to write evaluation JSON reports.",
    )

    args = parser.parse_args()

    val_paths = _list_pol_files(args.agp_val_dir)
    val_samples = agp_read_samples(val_paths, normalize=args.normalize)
    val_dataset = Dataset(val_samples)
    eval_k = len(val_dataset) if args.eval_k < 0 else min(int(args.eval_k), len(val_dataset))
    max_steps = args.max_steps

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
            coverage_threshold=float(args.coverage_threshold) + float(args.oracle_margin),
            n_samples=int(args.oracle_samples),
            seed=int(args.seed),
            verbose=bool(args.oracle_verbose),
        )
        exact = SkgeomVisibilityOracle(
            coverage_threshold=float(args.coverage_threshold),
            verbose=bool(args.oracle_verbose),
        )
        oracle = HybridOracle(fast_oracle=fast, exact_oracle=exact)

    coverage_eval_oracle = oracle

    guards: List[int] = []
    guard_ratios: List[float] = []
    covs: List[float] = []
    ratios: List[float] = []
    times: List[float] = []

    per_instance: List[Dict] = []

    cfg = {
        "oracle": args.oracle,
        "oracle_samples": int(args.oracle_samples),
        "oracle_margin": float(args.oracle_margin),
        "coverage_threshold": float(args.coverage_threshold),
        "seed": int(args.seed),
        "oracle_verbose": bool(args.oracle_verbose),
        "max_episodes": int(args.max_episodes),
        "max_steps": max_steps,
        "patience": int(args.patience),
        "lr": float(args.lr),
        "epsilon": float(args.epsilon),
        "gamma": float(args.gamma),
        "epsilon_decay": float(args.epsilon_decay),
        "epsilon_min": float(args.epsilon_min),
        "verbose": bool(args.verbose),
    }

    t_all0 = time.perf_counter()
    if int(args.workers) <= 1:
        for j in range(eval_k):
            pts, _, name = val_dataset[j]
            pts_np = pts.detach().cpu().numpy() if hasattr(pts, "detach") else np.asarray(pts)

            res = evaluate_qlearning_prune_single_instance(
                oracle,
                coverage_eval_oracle,
                pts_np,
                name,
                max_episodes=int(args.max_episodes),
                max_steps=max_steps,
                lr=float(args.lr),
                epsilon=float(args.epsilon),
                gamma=float(args.gamma),
                epsilon_decay=float(args.epsilon_decay),
                epsilon_min=float(args.epsilon_min),
                seed=int(args.seed),
                patience=int(args.patience),
                verbose=bool(args.verbose),
            )

            guards.append(int(res["guards"]))
            guard_ratios.append(float(res["guards"]) / max(1.0, float(res["n"])))
            if res.get("coverage") is not None:
                covs.append(float(res["coverage"]))

            opt = _read_opt_solution(args.agp_val_dir, name)
            opt_size = len(opt) if opt is not None else None
            ratios.append((float(res["guards"]) / float(opt_size)) if opt_size else None)

            times.append(float(res["time_s"]))

            per_instance.append(
                {
                    "name": name,
                    "n": int(res["n"]),
                    "guards": int(res["guards"]),
                    "guard_ratio": float(res["guards"]) / max(1.0, float(res["n"])),
                    "coverage": float(res["coverage"]) if res.get("coverage") is not None else None,
                    "opt_size": opt_size,
                    "approx_ratio": (float(res["guards"]) / float(opt_size)) if opt_size else None,
                    "time_s": float(res["time_s"]),
                }
            )
    else:
        print(f"[qlearning] using {int(args.workers)} workers | {_oracle_status_line(oracle)}")
        jobs = []
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            for j in range(eval_k):
                pts, _, name = val_dataset[j]
                pts_np = pts.detach().cpu().numpy() if hasattr(pts, "detach") else np.asarray(pts)
                jobs.append(
                    ex.submit(
                        _process_single_instance,
                        j,
                        pts_np,
                        name,
                        eval_k,
                        cfg,
                    )
                )

            completed = 0
            for fut in as_completed(jobs):
                res = fut.result()
                completed += 1
                print(
                    f"[qlearning] {completed}/{eval_k}: {res['name']} | {res['_status']}"
                )

                guards.append(int(res["guards"]))
                guard_ratios.append(float(res["guards"]) / max(1.0, float(res["n"])))
                if res.get("coverage") is not None:
                    covs.append(float(res["coverage"]))

                opt = _read_opt_solution(args.agp_val_dir, res["name"])
                opt_size = len(opt) if opt is not None else None
                ratios.append((float(res["guards"]) / float(opt_size)) if opt_size else None)

                times.append(float(res["time_s"]))

                per_instance.append(
                    {
                        "name": res["name"],
                        "n": int(res["n"]),
                        "guards": int(res["guards"]),
                            "guard_ratio": float(res["guards"]) / max(1.0, float(res["n"])),
                            "coverage": float(res["coverage"]) if res.get("coverage") is not None else None,
                            "opt_size": opt_size,
                            "approx_ratio": (float(res["guards"]) / float(opt_size)) if opt_size else None,
                            "time_s": float(res["time_s"]),
                        }
                    )
    wall_total_s = float(time.perf_counter() - t_all0)

    summary = {
        "_deprecated": "Use standardized fields in this file; kept for forward-compat.",
    }

    report = make_report(
        method="qlearning_prune",
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
            "coverage_metric": "exact",
        },
        timing={
            "wall_total_s": wall_total_s,
        },
    )

    os.makedirs(args.results_dir, exist_ok=True)
    eval_tag = "full" if eval_k >= len(val_dataset) else str(eval_k)
    out_path = os.path.join(args.results_dir, f"qlearning_prune_eval_only_val{eval_tag}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n--- Q-learning pruning eval on dataset polygons ---")
    s = report["summary"]
    msg = f"Dataset eval k={eval_k} | |S| mean={s['guards']['mean']:.2f} | |S|/n mean={s['guard_ratio']['mean']:.3f}"
    if s["coverage"]["mean"] is not None:
        msg += f" | geo-cov mean={s['coverage']['mean']:.3f}"
    if s["approx_ratio"]["mean"] is not None:
        msg += f" | |S|/opt mean={s['approx_ratio']['mean']:.2f}"
    msg += f" | time {report['timing']['wall_total_s']:.2f}s (mean {s['time_s']['mean']:.3f}s, p95 {s['time_s']['p95']:.3f}s)"
    print(msg)
    print(f"Results summary saved to {out_path}")


if __name__ == "__main__":
    main()