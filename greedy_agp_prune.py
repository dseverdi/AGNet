"""
Compute greedy pruning baseline for AGP instances.

Pre-computes greedy solutions and saves them to disk for comparison during training/evaluation.
"""

import argparse
import json
import os
import time
from typing import List, Optional

import numpy as np
import torch
from dotenv import load_dotenv

from dataset import agp_read_samples, Dataset
from ss_agp_prune import SkgeomVisibilityOracle, SampledVisibilityOracle, HybridOracle
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


def oracle_greedy_prune(
    oracle,
    points: np.ndarray,
    name: str,
    *,
    order: Optional[np.ndarray] = None,
    max_steps: Optional[int] = None,
) -> np.ndarray:
    """Oracle-only greedy pruning baseline.

    Starts from all vertices active and attempts to remove vertices in a fixed order.
    A removal is accepted iff the oracle says the new set is feasible.
    """

    n = int(points.shape[0])
    s = np.ones(n, dtype=np.bool_)
    steps = 0

    if not oracle.is_feasible(points, np.arange(n, dtype=np.int64), name):
        return np.arange(n, dtype=np.int64)

    if order is None:
        order = np.arange(n, dtype=np.int64)

    for v in order:
        if max_steps is not None and steps >= int(max_steps):
            break
        if not s[int(v)]:
            continue
        candidate = s.copy()
        candidate[int(v)] = False
        active = np.flatnonzero(candidate).astype(np.int64)
        if oracle.is_feasible(points, active, name):
            s = candidate
        steps += 1

    return np.flatnonzero(s).astype(np.int64)


def _read_opt_solution(sol_dir: str, name: str) -> Optional[List[int]]:
    base = sol_dir
    if os.path.isfile(base):
        base = os.path.dirname(base)
    path = os.path.join(base, f"{name}.solution")
    if not os.path.exists(path):
        return None
    try:
        lines = open(path, "r").read().splitlines()
        if len(lines) >= 2:
            parts = lines[1].strip().split()
            return [int(x) for x in parts]
    except Exception:
        return None
    return None


def compute_and_save_greedy_solutions(
    dataset_path: str,
    output_dir: str,
    oracle,
    max_steps: Optional[int] = None,
    verbose: bool = False,
) -> None:
    """Compute greedy solutions for all polygons and save to disk."""
    
    pol_files = _list_pol_files(dataset_path)
    print(f"Found {len(pol_files)} polygon files in {dataset_path}")
    
    if len(pol_files) == 0:
        raise ValueError(f"No .pol files found in {dataset_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    samples = agp_read_samples(pol_files, normalize=False)
    dataset = Dataset(samples)

    # Coverage metric: always try exact coverage for comparable reporting.
    coverage_eval_oracle = SkgeomVisibilityOracle(coverage_threshold=0.0, verbose=False)
    
    results = {}
    per_instance = []
    t_all0 = time.perf_counter()
    
    for i, (pts, _, name) in enumerate(dataset):
        if verbose and (i + 1) % 10 == 0:
            print(f"Processing {i + 1}/{len(dataset)}: {name}")
        
        pts_np = pts.detach().cpu().numpy()
        t0 = time.perf_counter()
        greedy_active = oracle_greedy_prune(oracle, pts_np, name, order=None, max_steps=max_steps)
        t1 = time.perf_counter()

        # Compute coverage (prefer exact coverage oracle for consistent reporting)
        coverage = None
        t2 = t1
        try:
            coverage = coverage_eval_oracle.coverage(pts_np, greedy_active, name)
            if coverage is None:
                coverage = oracle.coverage(pts_np, greedy_active, name)
        finally:
            t2 = time.perf_counter()

        greedy_s = float(t1 - t0)
        cov_s = float(t2 - t1)
        tot_s = float(t2 - t0)
        
        results[name] = {
            "guards": greedy_active.tolist(),
            "n_guards": int(greedy_active.size),
            "n_vertices": int(pts_np.shape[0]),
            "guard_ratio": float(greedy_active.size) / max(1.0, float(pts_np.shape[0])),
            "coverage": float(coverage) if coverage is not None else None,
            "time_greedy_s": greedy_s,
            "time_coverage_s": cov_s,
            "time_total_s": tot_s,
        }

        opt = _read_opt_solution(dataset_path, name)
        opt_size = len(opt) if opt is not None else None
        approx_ratio = (float(greedy_active.size) / float(opt_size)) if opt_size else None
        per_instance.append(
            {
                "name": name,
                "n": int(pts_np.shape[0]),
                "guards": int(greedy_active.size),
                "guard_ratio": float(greedy_active.size) / max(1.0, float(pts_np.shape[0])),
                "coverage": float(coverage) if coverage is not None else None,
                "opt_size": opt_size,
                "approx_ratio": approx_ratio,
                "time_s": tot_s,
                "time_greedy_s": greedy_s,
                "time_coverage_s": cov_s,
            }
        )

    t_all1 = time.perf_counter()
    
    # Save results
    output_path = os.path.join(output_dir, "greedy_solutions.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved greedy solutions to {output_path}")

    # Standardized report for unified comparison
    report = make_report(
        method="greedy_prune",
        per_instance=per_instance,
        args={
            "dataset_dir": dataset_path,
            "output_dir": output_dir,
            "max_steps": max_steps,
            "oracle": getattr(oracle, "__class__", type(oracle)).__name__,
        },
        dataset={
            "path": dataset_path,
            "count": int(len(dataset)),
        },
        oracle={
            "type": getattr(oracle, "__class__", type(oracle)).__name__,
            "coverage_metric": "exact" if results else "exact",
        },
        timing={
            "wall_total_s": float(t_all1 - t_all0),
        },
    )

    report_path = os.path.join(output_dir, "greedy_prune_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    tstats = report["summary"]["time_s"]
    print(
        "[timing] total wall={:.2f}s | per instance mean={:.3f}s (p95 {:.3f}s)".format(
            report["timing"]["wall_total_s"],
            tstats["mean"] or 0.0,
            tstats["p95"] or 0.0,
        )
    )
    print(f"Unified report saved to {report_path}")
    
    # Print summary statistics
    n_guards = [r["n_guards"] for r in results.values()]
    guard_ratios = [r["guard_ratio"] for r in results.values()]
    coverages = [r["coverage"] for r in results.values() if r["coverage"] is not None]
    
    print(f"\nSummary statistics:")
    print(f"  Total instances: {len(results)}")
    print(f"  Mean guards: {np.mean(n_guards):.2f} ± {np.std(n_guards):.2f}")
    print(f"  Mean guard ratio: {np.mean(guard_ratios):.3f} ± {np.std(guard_ratios):.3f}")
    if coverages:
        print(f"  Mean coverage: {np.mean(coverages):.3f} (min: {np.min(coverages):.3f})")


def main() -> None:
    load_dotenv()
    DATASET_PATH = os.getenv("DATASET_PATH")
    if not DATASET_PATH:
        raise EnvironmentError("DATASET_PATH environment variable must be set in .env file.")
    
    default_val = os.path.join(DATASET_PATH, "dev")
    
    parser = argparse.ArgumentParser(
        description="Pre-compute greedy pruning baselines for AGP dataset"
    )
    
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=default_val,
        help="Directory containing .pol files",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/v3/greedy_prune",
        help="Directory to save greedy solutions",
    )
    
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.99,
        help="Oracle feasibility threshold",
    )
    
    parser.add_argument(
        "--oracle",
        type=str,
        default="exact",
        choices=["exact", "sampled", "hybrid"],
        help="Feasibility oracle mode",
    )
    
    parser.add_argument(
        "--oracle-samples",
        type=int,
        default=512,
        help="Number of interior samples for sampled oracle",
    )
    
    parser.add_argument(
        "--oracle-margin",
        type=float,
        default=0.0,
        help="Extra margin for hybrid oracle",
    )
    
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Cap steps per instance (default: n)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress",
    )
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize oracle
    if args.oracle == "exact":
        oracle = SkgeomVisibilityOracle(
            coverage_threshold=float(args.coverage_threshold),
            verbose=False,
        )
    elif args.oracle == "sampled":
        oracle = SampledVisibilityOracle(
            coverage_threshold=float(args.coverage_threshold),
            n_samples=int(args.oracle_samples),
            seed=int(args.seed),
            verbose=False,
        )
    else:
        fast = SampledVisibilityOracle(
            coverage_threshold=float(args.coverage_threshold),
            n_samples=int(args.oracle_samples),
            seed=int(args.seed),
            verbose=False,
        )
        exact = SkgeomVisibilityOracle(
            coverage_threshold=float(args.coverage_threshold),
            verbose=False,
        )
        oracle = HybridOracle(fast_oracle=fast, exact_oracle=exact, margin=float(args.oracle_margin))
    
    max_steps = None if int(args.max_steps) < 0 else int(args.max_steps)
    
    compute_and_save_greedy_solutions(
        dataset_path=args.dataset_dir,
        output_dir=args.output_dir,
        oracle=oracle,
        max_steps=max_steps,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
