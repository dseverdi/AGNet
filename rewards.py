import numpy as np
from   utils import evaluate_polygon_visibility_numpy_wo_gt

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


def strict_reward(points: np.ndarray, solution: np.ndarray, name: str, length: int = None, delta=1e-5, p=1):
    """
    Strict reward function for vertex guard optimization.
    Args:
        points: np.ndarray, polygon vertices (N, 2)
        solution: np.ndarray, indices of selected guards
        name: str, instance name (unused)
        length: int, number of real vertices (if points is padded)
        uncovered_area: float, area not covered by guards (optional)
        delta: float, tolerance for uncovered area (default: 1e-5)
        p: float, penalty exponent for guard usage (default: 1)
    Returns:
        float: The computed reward.
    """
    n_vertices = length if length is not None else len(points)
    n_guards = len(solution)    
    try:        
        coverage = evaluate_polygon_visibility_numpy_wo_gt(points, solution, name)
        uncovered_area = 1.0 - coverage
    except Exception:
        uncovered_area = 1.0  # fallback: assume nothing is covered
    if uncovered_area < delta:
        return -((n_guards / n_vertices) ** p)
    else:
        return -1


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


def test_reward():
    reward = smooth_reward
    """Test the reward function with a simple polygon and guard sets, including an L-shaped polygon."""
    import numpy as np
    # Simple square polygon
    points = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float64)
    name = "square"
    # Case 1: All vertices as guards (should get full coverage, rel_length=1)
    guards_all = np.arange(4)
    r1 = reward(points, guards_all, name)
    print(f"Test 1 (square, all guards): reward={r1:.4f}")
    # Case 2: One guard (likely not full coverage, rel_length=0.25)
    guards_one = np.array([0])
    r2 = reward(points, guards_one, name)
    print(f"Test 2 (square, one guard): reward={r2:.4f}")
    # Case 3: Two guards (opposite corners)
    guards_two = np.array([0,2])
    r3 = reward(points, guards_two, name)
    print(f"Test 3 (square, two guards): reward={r3:.4f}")

    # L-shaped polygon
    l_points = np.array([
        [0,0], [2,0], [2,1], [1,1], [1,2], [0,2]
    ], dtype=np.float64)
    l_name = "L-shape"
    # All vertices as guards
    l_guards_all = np.arange(6)
    l_r1 = reward(l_points, l_guards_all, l_name)
    print(f"Test 4 (L-shape, all guards): reward={l_r1:.4f}")
    # One guard (corner)
    l_guards_one = np.array([1])
    l_r2 = reward(l_points, l_guards_one, l_name)
    print(f"Test 5 (L-shape, one guard): reward={l_r2:.4f}")
    # Two guards (corners)
    l_guards_two = np.array([0,2])
    l_r3 = reward(l_points, l_guards_two, l_name)
    print(f"Test 6 (L-shape, two guards): reward={l_r3:.4f}")
    # Three guards (corners)
    l_guards_three = np.array([0,2,4])
    l_r4 = reward(l_points, l_guards_three, l_name)
    print(f"Test 7 (L-shape, three guards): reward={l_r4:.4f}")

if __name__ == "__main__":
    test_reward()

