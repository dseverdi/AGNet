import numpy as np
from   utils import evaluate_polygon_visibility_numpy_wo_gt
import torch
import math
from functools import lru_cache

# Global cache for value_net model to avoid reloading
_value_net_model = None
_value_net_device = None

def load_value_net_proxy(model_path='checkpoints/value_net_embedding_size128_hidden_size256_epochs50_best.pt'):
    """
    Load value network for coverage proxy predictions.
    Caches the model to avoid reloading.
    """
    global _value_net_model, _value_net_device

    if _value_net_model is None:
        try:
            # Import here to avoid circular imports
            from evaluate import load_reward_predictor
            _value_net_model, _value_net_device, _ = load_reward_predictor(model_path)
            _value_net_model.eval()
            print(f"✓ Loaded value_net proxy: {model_path}")
        except Exception as e:
            print(f"Warning: Could not load value_net proxy: {e}")
            return None, None

    return _value_net_model, _value_net_device

def predict_coverage_proxy(points: np.ndarray, solution: np.ndarray, name: str, length: int = None,
                          model_path='checkpoints/value_net_embedding_size128_hidden_size256_epochs50_best.pt'):
    """
    Predict coverage using value_net proxy (fast neural approximation).

    Args:
        points: np.ndarray, polygon vertices (N, 2)
        solution: np.ndarray, indices of selected guards
        name: str, instance name
        length: int, number of real vertices
        model_path: str, path to value_net checkpoint

    Returns:
        float: Predicted coverage [0,1] or None if prediction fails
    """
    try:
        model, device = load_value_net_proxy(model_path)
        if model is None:
            return None

        # Convert to tensors
        if length is not None:
            points_tensor = torch.tensor(points[:length], dtype=torch.float32).unsqueeze(0).to(device)
            solution_tensor = torch.tensor([solution], dtype=torch.long).to(device)
            poly_lengths = torch.tensor([length], dtype=torch.long).to(device)
        else:
            points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(device)
            solution_tensor = torch.tensor([solution], dtype=torch.long).to(device)
            poly_lengths = torch.tensor([len(points)], dtype=torch.long).to(device)

        sol_lengths = torch.tensor([len(solution)], dtype=torch.long).to(device)

        with torch.no_grad():
            # Get model prediction (returns dict with 'coverage' key)
            pred = model(points_tensor, solution_tensor, poly_lengths, sol_lengths)
            if isinstance(pred, dict) and 'coverage' in pred:
                coverage_pred = pred['coverage'].item()
                return max(0.0, min(1.0, coverage_pred))  # Ensure [0,1] range
            else:
                return None

    except Exception as e:
        print(f"Warning: Coverage proxy prediction failed: {e}")
        return None

def linear_reward(points: np.ndarray, solution: np.ndarray, name: str, length: int = None) -> float:
    """
    Reward function for AGP: negative of (1 - coverage) + penalty for number of guards used.
    Reward = - (1 - coverage) - alpha * (len(solution) / total_vertices)
    (You can tune alpha as needed; here we use alpha=1.0 for equal weighting)
    """
    alpha = 1.0
    # If length is provided, slice points to real vertices only
    if length is not None:
        points = points[:length]
    total_vertices = len(points)
    n_guards = len(solution)
    # Compute coverage using the existing function
    coverage = evaluate_polygon_visibility_numpy_wo_gt(points, solution, name)    
    rel_length = n_guards / total_vertices if total_vertices > 0 else 1.0
    # The reward is higher for high coverage and fewer guards
    return - (1.0 - coverage) - alpha * rel_length


def strict_reward(points: np.ndarray, solution: np.ndarray, name: str, length: int = None, alpha=1.0, M=1000.0):
    """
    Strict reward function for vertex guard optimization.
    Returns large penalty M if coverage < 0.99, otherwise exponential reward based on guard efficiency.

    Args:
        points: np.ndarray, polygon vertices (N, 2)
        solution: np.ndarray, indices of selected guards
        name: str, instance name (unused)
        length: int, number of real vertices (if points is padded)
        alpha: float, scaling factor for exponential reward (default: 1.0)
        M: float, large penalty for insufficient coverage (default: 1000.0)
    Returns:
        float: The computed reward.
    """
    n_vertices = length if length is not None else len(points)
    n_guards = len(solution)

    try:
        coverage = evaluate_polygon_visibility_numpy_wo_gt(points, solution, name)
    except Exception:
        coverage = 0.0  # fallback: assume nothing is covered

    if coverage < 0.99:
        return M  # large penalty
    else:
        return -math.exp(alpha * (1 - n_guards / n_vertices))


def smooth_reward(points: np.ndarray, solution: np.ndarray, name: str, length: int = None, alpha=1, p=1):
    """
    Smooth reward function for vertex guard optimization.
    Args:
        points: np.ndarray, polygon vertices (N, 2)
        solution: np.ndarray, indices of selected guards
        name: str, instance name (unused)
        length: int, number of real vertices (if points is padded)
        delta: float, tolerance for uncovered area (default: 1e-5)
        p: float, penalty exponent for guard usage (default: 1)
    Returns:
        float: The computed reward.
    """
    eps = 1e-8  # small epsilon to avoid division by zero
    n_vertices = length if length is not None else len(points)
    n_guards = len(solution)    
    try:        
        coverage = evaluate_polygon_visibility_numpy_wo_gt(points, solution, name)
        uncovered_area = 1.0 - coverage        
    except Exception:
        print(f"Warning: visibility failed for guards {solution} in polygon {name}")
        uncovered_area = 1.0  # fallback: assume nothing is covered

    return ( (uncovered_area + eps)**alpha * ((n_guards/n_vertices + eps) ** p) )


def enhanced_penalty(points: np.ndarray, solution: np.ndarray, name: str, length: int = None, alpha=1, p=2, delta=1.0, scale=1000, coverage_proxy=None):
    """
    Enhanced smooth penalty function for vertex guard optimization.
    Note: This function returns penalties (lower values = better solutions).
    Args:
        points: np.ndarray, polygon vertices (N, 2)
        solution: np.ndarray, indices of selected guards
        name: str, instance name (unused)
        length: int, number of real vertices (if points is padded)
        alpha: float, exponent for uncovered area penalty (default: 1)
        p: float, penalty exponent for guard usage (default: 2)
        delta: float, penalty for non-full coverage (default: 1.0)
        scale: float, scaling factor for base penalty (default: 1000)
        coverage_proxy: float, optional proxy coverage value [0,1] to use instead of geometric computation
    Returns:
        float: The computed penalty (lower is better).
    """
    eps = 1e-8
    n_vertices = length if length is not None else len(points)
    n_guards = len(solution)
    
    if coverage_proxy is not None:
        # Use proxy coverage prediction (fast neural network approximation)
        coverage = max(0.0, min(1.0, coverage_proxy))  # Ensure [0,1] range
        uncovered_area = 1.0 - coverage
    else:
        # Compute actual geometric coverage (expensive)
        try:
            coverage = evaluate_polygon_visibility_numpy_wo_gt(points, solution, name)
            uncovered_area = 1.0 - coverage
        except Exception:
            print(f"Warning: visibility failed for guards {solution} in polygon {name}")
            uncovered_area = 1.0

    # Base penalty: scaled to make guard count differences meaningful
    base_penalty = scale * ((uncovered_area + eps)**alpha * (n_guards/n_vertices + eps)**p)
    
    # Penalty for non-full coverage (instead of bonus for full coverage)
    coverage_penalty = delta if abs(uncovered_area) > 1e-5 else 0
    
    return base_penalty + coverage_penalty


def enhanced_penalty_with_proxy(points: np.ndarray, solution: np.ndarray, name: str, length: int = None,
                               alpha=1, p=2, delta=1.0, scale=1000,
                               use_proxy=True, model_path='checkpoints/value_net_embedding_size128_hidden_size256_epochs50_best.pt'):
    """
    Enhanced penalty with optional value_net proxy for coverage prediction.
    Falls back to geometric computation if proxy fails or is disabled.

    Args:
        points: np.ndarray, polygon vertices (N, 2)
        solution: np.ndarray, indices of selected guards
        name: str, instance name
        length: int, number of real vertices
        alpha, p, delta, scale: penalty parameters (same as enhanced_penalty)
        use_proxy: bool, whether to use value_net proxy (default: True)
        model_path: str, path to value_net checkpoint

    Returns:
        tuple: (penalty, coverage_used, proxy_success)
            - penalty: computed penalty value
            - coverage_used: actual coverage value used (proxy or geometric)
            - proxy_success: whether proxy was successfully used
    """
    if use_proxy:
        # Try to use value_net proxy for fast coverage prediction
        coverage_proxy = predict_coverage_proxy(points, solution, name, length, model_path)
        if coverage_proxy is not None:
            # Use proxy coverage
            penalty = enhanced_penalty(points, solution, name, length, alpha, p, delta, scale,
                                     coverage_proxy=coverage_proxy)
            return penalty, coverage_proxy, True
        else:
            print(f"Warning: Proxy failed for {name}, falling back to geometric computation")

    # Fallback to geometric computation
    penalty = enhanced_penalty(points, solution, name, length, alpha, p, delta, scale)

    # Also compute actual coverage for reference
    try:
        actual_coverage = evaluate_polygon_visibility_numpy_wo_gt(points, solution, name)
    except Exception:
        actual_coverage = 0.0

    return penalty, actual_coverage, False


def demo_proxy_vs_geometric():
    """
    Demonstration of proxy vs geometric coverage computation.
    Shows speed and accuracy comparison.
    """
    import time

    # Simple test polygon
    points = np.array([[0,0], [2,0], [2,2], [1,2], [1,1], [0,1]], dtype=np.float64)
    solution = np.array([0, 2, 4])  # Select 3 guards
    name = "demo_polygon"
    length = len(points)

    print("=== Value Net Proxy vs Geometric Coverage Demo ===")
    print(f"Polygon: {length} vertices, Solution: {len(solution)} guards")

    # Time geometric computation
    start_time = time.time()
    geometric_penalty = enhanced_penalty(points, solution, name, length)
    geometric_time = time.time() - start_time

    # Time proxy computation
    start_time = time.time()
    proxy_penalty, coverage_used, proxy_success = enhanced_penalty_with_proxy(
        points, solution, name, length, use_proxy=True
    )
    proxy_time = time.time() - start_time

    # Compute actual coverage for comparison
    try:
        actual_coverage = evaluate_polygon_visibility_numpy_wo_gt(points, solution, name)
    except Exception:
        actual_coverage = 0.0

    print("\nResults:")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")
    print(".4f")

    if proxy_success:
        print("\n✅ Proxy prediction successful!")
        print(".4f")
    else:
        print("\n❌ Proxy prediction failed, used geometric fallback")

    return proxy_success


def demo_active_search_with_proxy():
    """
    Demonstration of how to use the proxy in active search scenarios.
    Shows how to evaluate multiple solutions quickly using the proxy.
    """
    import time

    # Simple test polygon
    points = np.array([[0,0], [2,0], [2,2], [1,2], [1,1], [0,1]], dtype=np.float64)
    name = "demo_polygon"
    length = len(points)

    # Simulate K=5 sampled solutions during active search
    candidate_solutions = [
        np.array([0, 2, 4]),      # 3 guards
        np.array([0, 1, 2, 4]),   # 4 guards
        np.array([0, 1, 2, 3, 4]), # 5 guards
        np.array([0, 2]),         # 2 guards
        np.array([0, 1, 3, 4]),   # 4 guards
    ]

    print("=== Active Search with Value Net Proxy ===")
    print(f"Evaluating {len(candidate_solutions)} candidate solutions using proxy")

    # Time proxy-based evaluation (fast)
    start_time = time.time()
    proxy_results = []
    for i, solution in enumerate(candidate_solutions):
        penalty, coverage, success = enhanced_penalty_with_proxy(
            points, solution, f"{name}_sol_{i}", length, use_proxy=True
        )
        proxy_results.append((penalty, coverage, success))
        print(".4f")

    proxy_time = time.time() - start_time

    # Time geometric evaluation (slow)
    start_time = time.time()
    geometric_results = []
    for i, solution in enumerate(candidate_solutions):
        penalty = enhanced_penalty(points, solution, f"{name}_sol_{i}", length)
        geometric_results.append(penalty)

    geometric_time = time.time() - start_time

    print("\nTiming Comparison:")
    print(".3f")
    print(".3f")
    print(".1f")

    # Find best solution
    best_idx = np.argmin([r[0] for r in proxy_results])
    best_penalty, best_coverage, _ = proxy_results[best_idx]
    best_solution = candidate_solutions[best_idx]

    print("\nBest Solution (via proxy):")
    print(f"  Solution: {best_solution}")
    print(f"  Penalty: {best_penalty:.4f}")
    print(f"  Coverage: {best_coverage:.4f}")

    return proxy_time, geometric_time


def simulate_active_search_with_proxy(num_samples=100, K=5):
    """
    Simulate active search scenario: evaluate K solutions for each of num_samples instances.
    Compare proxy vs geometric computation performance.
    """
    import time
    import random

    print(f"=== Active Search Simulation: {num_samples} samples, K={K} ===")

    # Generate synthetic polygon data (simplified for demo)
    polygons = []
    for i in range(num_samples):
        # Create random polygons with 6-10 vertices
        n_vertices = random.randint(6, 10)
        points = np.random.rand(n_vertices, 2) * 10  # Scale to [0,10] range
        polygons.append((points, f"poly_{i}"))

    total_proxy_time = 0
    total_geometric_time = 0
    proxy_success_count = 0

    print(f"Evaluating {num_samples * K} total solution-polygon pairs...")

    for poly_idx, (points, name) in enumerate(polygons):
        if poly_idx % 20 == 0:
            print(f"  Progress: {poly_idx}/{num_samples} polygons")

        # Generate K random solutions for this polygon
        n_vertices = len(points)
        for k in range(K):
            # Random solution: select 2-4 guards
            n_guards = random.randint(2, min(4, n_vertices))
            solution = np.random.choice(n_vertices, n_guards, replace=False)
            solution = np.sort(solution)

            # Time proxy evaluation
            start_time = time.time()
            proxy_penalty, proxy_coverage, proxy_success = enhanced_penalty_with_proxy(
                points, solution, f"{name}_k{k}", n_vertices, use_proxy=True
            )
            proxy_time = time.time() - start_time
            total_proxy_time += proxy_time

            # Time geometric evaluation
            start_time = time.time()
            geometric_penalty = enhanced_penalty(
                points, solution, f"{name}_k{k}", n_vertices
            )
            geometric_time = time.time() - start_time
            total_geometric_time += geometric_time

            if proxy_success:
                proxy_success_count += 1

    # Results
    total_evaluations = num_samples * K
    avg_proxy_time = total_proxy_time / total_evaluations
    avg_geometric_time = total_geometric_time / total_evaluations
    speedup = avg_geometric_time / avg_proxy_time if avg_proxy_time > 0 else float('inf')
    success_rate = proxy_success_count / total_evaluations * 100

    print("\n=== Active Search Results ===")
    print(f"Total evaluations: {total_evaluations}")
    print(".6f")
    print(".6f")
    print(".1f")
    print(".1f")

    print("\nPerformance Summary:")
    print(".1f")
    print(".1f")
    print(".1f")

    return {
        'total_evaluations': total_evaluations,
        'avg_proxy_time': avg_proxy_time,
        'avg_geometric_time': avg_geometric_time,
        'speedup': speedup,
        'success_rate': success_rate
    }


if __name__ == "__main__":
    # Run both demos
    demo_proxy_vs_geometric()
    print("\n" + "="*50)
    demo_active_search_with_proxy()
    print("\n" + "="*50)
    simulate_active_search_with_proxy(num_samples=100, K=5)


if __name__ == "__main__":
    demo_proxy_vs_geometric()

