import os
import argparse
import numpy as np
import torch
import json
from utils import evaluate_polygon_visibility_numpy_wo_gt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import skgeom
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Precompute visibility polygons for all vertices
def precompute_visibility_polygons(points, name=None):
    poly = skgeom.Polygon([skgeom.Point2(float(x), float(y)) for x, y in points])
    arr = skgeom.arrangement.Arrangement()
    for edge in poly.edges:
        arr.insert(edge)
    vs = skgeom.TriangularExpansionVisibility(arr)
    edges = list(poly.edges)
    eps = 1e-8
    vis_polys = []
    def compute(idx):
        v_prev = edges[(idx - 1) % len(edges)].source()
        v = edges[idx % len(edges)].source()
        v_next = edges[idx % len(edges)].target()
        p = skgeom.Vector2(v, v_prev)
        p = p / math.sqrt(float(p.squared_length()))
        r = skgeom.Vector2(v, v_next)
        r = r / math.sqrt(float(r.squared_length()))
        q = skgeom.Point2(v.x() + eps * (p.x() + r.x()), v.y() + eps * (p.y() + r.y()))
        if poly.oriented_side(q) != skgeom.Sign.POSITIVE:
            q = skgeom.Point2(v.x() - eps * (p.x() + r.x()), v.y() - eps * (p.y() + r.y()))
        face = arr.find(q)
        if face is None or face.is_unbounded():
            return None
        try:
            vx = vs.compute_visibility(q, face)
            return skgeom.Polygon([vertex.point() for vertex in vx.vertices])
        except RuntimeError:
            return None
    # Limit number of workers to 2 for safety
    with ThreadPoolExecutor(max_workers=2) as executor:
        vis_polys = list(executor.map(compute, range(len(points))))
    return vis_polys, poly

# Fast greedy guard selection using precomputed visibility polygons
def greedy_guard_selection_fast(points, max_guards=None, coverage_threshold=1.0, verbose=False, name=None):
    vis_polys, poly = precompute_visibility_polygons(points, name)
    N = len(points)
    if max_guards is None:
        max_guards = N
    covered = skgeom.PolygonSet()
    guard_idxs = []
    coverages = []
    available = set(range(N))
    poly_area = abs(float(poly.area()))
    def covered_area(pset):
        return sum(abs(float(poly.outer_boundary().area())) - sum(abs(float(h.area())) for h in poly.holes) for poly in pset.polygons)
    while True:
        best_idx = None
        best_gain = 0.0
        best_union = None
        for idx in available:
            vis_poly = vis_polys[idx]
            if vis_poly is None:
                continue
            candidate_union = covered.union(skgeom.PolygonSet([vis_poly]))
            new_area = covered_area(candidate_union)
            old_area = covered_area(covered)
            gain = new_area - old_area
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                best_union = candidate_union
        if best_idx is None or best_gain == 0.0:
            break
        guard_idxs.append(best_idx)
        available.remove(best_idx)
        covered = best_union
        coverage = covered_area(covered) / poly_area if poly_area > 0 else 0.0
        coverages.append(coverage)
        if verbose:
            print(f"Added guard {best_idx}, coverage: {coverage:.4f}")
        if coverage >= coverage_threshold or len(guard_idxs) >= max_guards:
            break
    return guard_idxs, coverages

def greedy_guard_selection(points, max_guards=None, coverage_threshold=1.0, verbose=False):
    """
    Greedily select guards from polygon vertices to maximize coverage.
    Args:
        points: np.ndarray of shape (N, 2), polygon vertices
        max_guards: int or None, maximum number of guards to place
        coverage_threshold: float, stop when this coverage is reached (default: 1.0)
        verbose: bool, print progress
    Returns:
        guard_idxs: list of selected guard indices
        coverages: list of coverage after each guard is added
    """
    N = len(points)
    if max_guards is None:
        max_guards = N
    uncovered = 1.0
    guard_idxs = []
    coverages = []
    available = set(range(N))
    while uncovered > (1.0 - coverage_threshold) and len(guard_idxs) < max_guards and available:
        best_idx = None
        best_coverage = 0.0
        for idx in available:
            candidate = guard_idxs + [idx]
            coverage = evaluate_polygon_visibility_numpy_wo_gt(points, candidate, None)
            if coverage > best_coverage:
                best_coverage = coverage
                best_idx = idx
        if best_idx is None:
            break
        guard_idxs.append(best_idx)
        available.remove(best_idx)
        coverages.append(best_coverage)
        uncovered = 1.0 - best_coverage
        if verbose:
            print(f"Added guard {best_idx}, coverage: {best_coverage:.4f}")
    return guard_idxs, coverages

def read_single_pol_file(path, normalize=False):
    """
    Read a single .pol file and return points as a torch tensor.
    """
    with open(path, 'r') as f:
        tokens = f.read().split()
        num_points = int(tokens[0])
        points = []
        for i in range(1, 2 * num_points, 2):
            x_token = tokens[i]
            y_token = tokens[i + 1]
            if '/' in x_token:
                x_num, x_denom = map(float, x_token.split('/'))
                x = x_num / x_denom if x_denom != 0 else 0.0
            else:
                x = float(x_token)
            if '/' in y_token:
                y_num, y_denom = map(float, y_token.split('/'))
                y = y_num / y_denom if y_denom != 0 else 0.0
            else:
                y = float(y_token)
            points.append((x, y))
        points_tensor = torch.tensor(points, dtype=torch.float32)
        if normalize:
            min_xy = points_tensor.min(dim=0)[0]
            max_xy = points_tensor.max(dim=0)[0]
            denom = (max_xy - min_xy)
            denom[denom == 0] = 1.0  # avoid division by zero
            points_tensor = (points_tensor - min_xy) / denom
        return points_tensor

def process_single_sample(pol_path, normalize, agp_val_dir, verbose):
    import numpy as np
    try:
        points = read_single_pol_file(pol_path, normalize=normalize)
        n_vertices = len(points)
        guards, coverages = greedy_guard_selection_fast(points.numpy(), verbose=verbose, name=pol_path)
        final_coverage = coverages[-1] if coverages else 0.0
        rel_size = len(guards) / float(n_vertices) if n_vertices > 0 else float('nan')
        # --- Optimal solution comparison ---
        base_name = os.path.splitext(os.path.basename(pol_path))[0]
        sol_dir = agp_val_dir
        opt_sol_path = os.path.join(sol_dir, f"{base_name}.solution")
        opt_sizes = []
        try:
            with open(opt_sol_path, 'r') as f:
                lines = f.read().splitlines()
                for line in lines[1:]:
                    if line.strip() and not line.strip().startswith('#'):
                        opt_idxs = [int(x) for x in line.strip().split()]
                        opt_sizes.append(len(opt_idxs))
            if opt_sizes:
                min_opt_size = min(opt_sizes)
                rel_to_opt = len(guards) / min_opt_size if min_opt_size > 0 else float('nan')
            else:
                min_opt_size = float('nan')
                rel_to_opt = float('nan')
        except Exception as e:
            min_opt_size = float('nan')
            rel_to_opt = float('nan')
        print(f"[VERBOSE] Sample: {os.path.basename(pol_path)} | Guards: {len(guards)} | Opt: {min_opt_size} | Coverage: {final_coverage:.4f} | Rel_to_opt: {rel_to_opt}")
        if not np.isnan(min_opt_size) and len(guards) < min_opt_size:
            print(f"[SUSPICIOUS] Greedy uses fewer guards than 'optimal'! -> Guards: {len(guards)}, Opt: {min_opt_size}, Coverage: {final_coverage:.4f}")
        return (pol_path, len(guards), final_coverage, rel_size, rel_to_opt)
    except Exception as e:
        print(f"[ERROR] Sample: {os.path.basename(pol_path)} | Guards: nan | Opt: nan | Coverage: nan | Exception: {e}")
        return (pol_path, float('nan'), float('nan'), float('nan'), float('nan'))

def greedy_eval_comprehensive(pol_files, agp_val_dir, normalize=True, verbose=False):
    """Comprehensive evaluation of greedy algorithm with whisker plot statistics."""
    print(f"\n--- Comprehensive Greedy Evaluation on {len(pol_files)} samples ---")
    
    # Statistics for whisker plots (same as supervised learning and RL)
    pred_sizes = []
    true_sizes = []
    coverage_ratios = []  # What fraction of optimal guards are covered by greedy guards
    efficiency_ratios = []  # What fraction of greedy guards are in the optimal set
    size_ratios = []  # greedy_size / optimal_size
    overlap_counts = []  # absolute number of overlapping guards
    polygon_coverages = []  # geometric coverage of the polygon
    
    total_samples = len(pol_files)
    print(f"Processing {total_samples} samples...")
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_comprehensive_sample, pol_path, normalize, agp_val_dir, verbose) for pol_path in pol_files]
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is not None:
                pred_size, true_size, coverage_ratio, efficiency_ratio, size_ratio, overlap_count, polygon_coverage = result
                pred_sizes.append(pred_size)
                true_sizes.append(true_size)
                coverage_ratios.append(coverage_ratio)
                efficiency_ratios.append(efficiency_ratio)
                size_ratios.append(size_ratio)
                overlap_counts.append(overlap_count)
                polygon_coverages.append(polygon_coverage)
    
    # Compute statistics for whisker plots
    def compute_stats(data, name):
        data = np.array(data)
        if len(data) == 0:
            return {key: float('nan') for key in ['mean', 'median', 'std', 'min', 'max', 'q25', 'q75', 'iqr']}
        
        stats = {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25)
        }
        print(f"\n{name} Statistics:")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Median: {stats['median']:.3f}")
        print(f"  Std: {stats['std']:.3f}")
        print(f"  Min: {stats['min']:.3f}")
        print(f"  Max: {stats['max']:.3f}")
        print(f"  Q25: {stats['q25']:.3f}")
        print(f"  Q75: {stats['q75']:.3f}")
        print(f"  IQR: {stats['iqr']:.3f}")
        return stats
    
    print("\n=== EVALUATION RESULTS ===")
    
    # Compute all statistics
    if len(pred_sizes) > 0:
        size_stats = compute_stats(pred_sizes, "Greedy Solution Sizes")
        optimal_stats = compute_stats(true_sizes, "Optimal Solution Sizes")
        coverage_stats = compute_stats(coverage_ratios, "Coverage Ratios (fraction of optimal guards found)")
        efficiency_stats = compute_stats(efficiency_ratios, "Efficiency Ratios (fraction of greedy guards that are optimal)")
        ratio_stats = compute_stats(size_ratios, "Size Ratios (greedy/optimal)")
        overlap_stats = compute_stats(overlap_counts, "Overlap Counts (absolute number of matching guards)")
        
        # Summary metrics
        print(f"\n=== SUMMARY ===")
        print(f"Instances evaluated: {len(pred_sizes)}")
        print(f"Perfect solutions (100% coverage): {sum(1 for c in coverage_ratios if c >= 1.0)}")
        print(f"Good solutions (>=80% coverage): {sum(1 for c in coverage_ratios if c >= 0.8)}")
        print(f"Reasonable solutions (>=60% coverage): {sum(1 for c in coverage_ratios if c >= 0.6)}")
        if len(size_ratios) > 0:
            print(f"Average size inflation: {np.mean(size_ratios):.2f}x optimal")
    else:
        print("No valid instances with optimal solutions found for comparison.")
        size_stats = optimal_stats = coverage_stats = efficiency_stats = ratio_stats = overlap_stats = {}
    
    # Compute coverage statistics from polygon visibility
    coverage_array = np.array([c for c in polygon_coverages if not np.isnan(c)])
    coverage_vis_stats = compute_stats(coverage_array, "Polygon Coverage (visibility)") if len(coverage_array) > 0 else {}
    
    return {
        # Comprehensive metrics
        'pred_sizes': pred_sizes,
        'true_sizes': true_sizes,
        'coverage_ratios': coverage_ratios,
        'efficiency_ratios': efficiency_ratios,
        'size_ratios': size_ratios,
        'overlap_counts': overlap_counts,
        'polygon_coverages': coverage_array.tolist() if len(coverage_array) > 0 else [],
        'stats': {
            'size_stats': size_stats,
            'optimal_stats': optimal_stats,
            'coverage_stats': coverage_stats,
            'efficiency_stats': efficiency_stats,
            'ratio_stats': ratio_stats,
            'overlap_stats': overlap_stats,
            'coverage_vis_stats': coverage_vis_stats
        }
    }

def process_comprehensive_sample(pol_path, normalize, agp_val_dir, verbose):
    """Process a single sample for comprehensive evaluation."""
    try:
        points = read_single_pol_file(pol_path, normalize=normalize)
        n_vertices = len(points)
        pred_indices, coverages = greedy_guard_selection_fast(points.numpy(), verbose=verbose, name=pol_path)
        final_coverage = coverages[-1] if coverages else 0.0
        
        # Read optimal solution for comparison
        base_name = os.path.splitext(os.path.basename(pol_path))[0]
        opt_sol_path = os.path.join(agp_val_dir, f"{base_name}.solution")
        true_indices = []
        try:
            with open(opt_sol_path, 'r') as f:
                lines = f.read().splitlines()
                if len(lines) >= 2:
                    true_indices = [int(x) for x in lines[1].split()]
        except Exception:
            true_indices = []
        
        if len(true_indices) > 0:
            pred_set = set(pred_indices)
            true_set = set(true_indices)
            overlap = pred_set.intersection(true_set)
            
            pred_size = len(pred_indices)
            true_size = len(true_indices)
            overlap_count = len(overlap)
            
            # Coverage: what fraction of optimal guards are covered
            coverage_ratio = overlap_count / true_size if true_size > 0 else 0.0
            
            # Efficiency: what fraction of greedy guards are optimal
            efficiency_ratio = overlap_count / pred_size if pred_size > 0 else 0.0
            
            # Size ratio: how many times larger is the greedy vs optimal
            size_ratio = pred_size / true_size if true_size > 0 else float('inf')
            
            return (pred_size, true_size, coverage_ratio, efficiency_ratio, size_ratio, overlap_count, final_coverage)
        else:
            return None
    except Exception as e:
        if verbose:
            print(f"[ERROR] Sample: {os.path.basename(pol_path)} | Exception: {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agp_val_dir', type=str, default="/home/dseverdi/Radno/MLAG/dataset/AGPIL/dev", help='Directory with validation .pol files')
    parser.add_argument('--train-size', type=int, default=8000, help='Number of validation samples to use (default: 8000, or all if smaller)')
    parser.add_argument('--normalize', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--comprehensive', action='store_true', default=True, help='Use comprehensive evaluation with whisker plot statistics')
    args = parser.parse_args()

    agp_dir = args.agp_val_dir
    pol_files = [os.path.join(agp_dir, f) for f in os.listdir(agp_dir) if f.endswith('.pol')]
    if args.train_size is not None:
        pol_files = pol_files[:args.train_size]

    if args.comprehensive:
        # Use comprehensive evaluation
        eval_results = greedy_eval_comprehensive(pol_files, args.agp_val_dir, normalize=args.normalize, verbose=args.verbose)
        
        # Save evaluation results
        import json
        results_summary = {
            'args': vars(args),
            'num_samples': len(pol_files),
            'training_method': 'greedy_algorithm'
        }
        
        # Add statistics if available
        if 'stats' in eval_results and eval_results['stats']:
            if 'coverage_stats' in eval_results['stats']:
                results_summary['coverage_stats'] = eval_results['stats']['coverage_stats']
            if 'efficiency_stats' in eval_results['stats']:
                results_summary['efficiency_stats'] = eval_results['stats']['efficiency_stats']
            if 'ratio_stats' in eval_results['stats']:
                results_summary['size_ratio_stats'] = eval_results['stats']['ratio_stats']
            if 'coverage_vis_stats' in eval_results['stats']:
                results_summary['polygon_coverage_stats'] = eval_results['stats']['coverage_vis_stats']
        
        # Save to results directory
        os.makedirs('results', exist_ok=True)
        with open('results/greedy_agp_evaluation.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        print("Results summary saved to results/greedy_agp_evaluation.json")
        
    else:
        # Use legacy evaluation
        all_num_guards = []
        all_coverages = []
        all_rel_sizes = []
        all_rel_to_opt = []
        total_samples = len(pol_files)
        print(f"Processing {total_samples} samples in parallel...")
        suspicious_cases = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_single_sample, pol_path, args.normalize, args.agp_val_dir, args.verbose) for pol_path in pol_files]
            all_results = []
            for i, future in enumerate(as_completed(futures)):
                pol_path, num_guards, final_coverage, rel_size, rel_to_opt = future.result()
                # Print all results for debugging
                print(f"[RESULT] {os.path.basename(pol_path)} | Guards: {num_guards} | Rel_size: {rel_size} | Rel_to_opt: {rel_to_opt} | Coverage: {final_coverage:.4f}")
                # Only collect suspicious cases here
                if not np.isnan(num_guards) and not np.isnan(rel_to_opt) and num_guards < rel_to_opt:
                    suspicious_cases.append((os.path.basename(pol_path), num_guards, rel_to_opt, final_coverage))
                all_num_guards.append(num_guards)
                all_coverages.append(final_coverage)
                all_rel_sizes.append(rel_size)
                all_rel_to_opt.append(rel_to_opt)
                all_results.append((pol_path, num_guards, final_coverage, rel_size, rel_to_opt))

        # Print only suspicious cases
        if suspicious_cases:
            print("\nCases where greedy uses fewer guards than 'optimal':")
            print("| Polygon file | Greedy Guards | Opt Guards | Coverage |")
            for fname, greedy, opt, cov in suspicious_cases:
                print(f"| {fname:<20} | {greedy:^13} | {opt:^9} | {cov:^8.4f} |")
        else:
            print("\nNo cases where greedy uses fewer guards than 'optimal'.")

        # Print only number of processed instances and summary metrics
        print(f"\nProcessed {len(all_results)} instances.")
        # Print summary statistics in table format (like rl_agp.py)
        metrics = {
            'Coverage': np.array(all_coverages, dtype=np.float32),
            'Rel. Sol. Size': np.array(all_rel_sizes, dtype=np.float32),
            'Rel. to Opt. Sol. Size': np.array(all_rel_to_opt, dtype=np.float32)
        }
        stats = {}
        for key, arr in metrics.items():
            arr = arr[~np.isnan(arr)]
            stats[key] = {
                'avg': np.mean(arr) if arr.size else float('nan'),
                'min': np.min(arr) if arr.size else float('nan'),
                'max': np.max(arr) if arr.size else float('nan'),
                'std': np.std(arr) if arr.size else float('nan')
            }
        print("+-------------------------+----------+----------+----------+----------+")
        print("|        Metric           |   Avg    |   Min    |   Max    |   Std    |")
        print("+-------------------------+----------+----------+----------+----------+")
        for key in metrics.keys():
            print(f"| {key:<23} | {stats[key]['avg']:^8.4f} | {stats[key]['min']:^8.4f} | {stats[key]['max']:^8.4f} | {stats[key]['std']:^8.4f} |")
        print("+-------------------------+----------+----------+----------+----------+")

def get_optimal_solution_indices(pol_path, agp_val_dir):
    base_name = os.path.splitext(os.path.basename(pol_path))[0]
    sol_path = os.path.join(agp_val_dir, f"{base_name}.solution")
    opt_idxs = []
    try:
        with open(sol_path, 'r') as f:
            lines = f.read().splitlines()
            for line in lines[1:]:
                if line.strip() and not line.strip().startswith('#'):
                    opt_idxs = [int(x) for x in line.strip().split()]
                    break  # Take the first solution
    except Exception:
        pass
    return opt_idxs

def plot_polygon_with_guards_and_visibility(points, guard_idxs, vis_polys, ax, title=""): 
    # Draw polygon
    poly_np = np.array(points)
    ax.plot(np.append(poly_np[:,0], poly_np[0,0]), np.append(poly_np[:,1], poly_np[0,1]), color='black', linewidth=2)
    # Draw visibility regions
    for idx in guard_idxs:
        vis = vis_polys[idx]
        if vis is not None:
            vis_pts = np.array([[p.x(), p.y()] for p in vis.vertices])
            patch = mpatches.Polygon(vis_pts, closed=True, alpha=0.25, color='orange')
            ax.add_patch(patch)
    # Draw guards
    guards = poly_np[guard_idxs]
    ax.scatter(guards[:,0], guards[:,1], color='red', s=60, zorder=5, label='Guards')
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')

def visualize_greedy_vs_optimal(pol_path, greedy_idxs, greedy_vis_polys, opt_idxs, opt_vis_polys, out_dir=None):
    points = read_single_pol_file(pol_path, normalize=False).numpy()
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    plot_polygon_with_guards_and_visibility(points, greedy_idxs, greedy_vis_polys, axs[0], title="Greedy Solution")
    plot_polygon_with_guards_and_visibility(points, opt_idxs, opt_vis_polys, axs[1], title="Optimal Solution")
    plt.tight_layout()
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(pol_path))[0]
        out_path = os.path.join(out_dir, f"{base_name}_greedy_vs_optimal.png")
        plt.savefig(out_path, bbox_inches='tight')
        print(f"Saved comparison plot to {out_path}")
    else:
        plt.show()
    plt.close(fig)

def draw_all_greedy_vs_optimal(pol_files, agp_val_dir, rel_to_opts, out_dir=None):
    # Sequential processing, no parallelism
    for pol_path, rel_to_opt in zip(pol_files, rel_to_opts):
        if rel_to_opt < 1.0:
            points = read_single_pol_file(pol_path, normalize=False).numpy()
            greedy_idxs, _ = greedy_guard_selection_fast(points, name=pol_path)
            greedy_vis_polys, _ = precompute_visibility_polygons(points, pol_path)
            opt_idxs = get_optimal_solution_indices(pol_path, agp_val_dir)
            opt_vis_polys, _ = precompute_visibility_polygons(points, pol_path)
            visualize_greedy_vs_optimal(pol_path, greedy_idxs, greedy_vis_polys, opt_idxs, opt_vis_polys, out_dir=out_dir)

if __name__ == "__main__":
    main()
    # After main, visualize all polygons where greedy rel_size to optimal < 1
    agp_dir = "/home/dseverdi/Radno/MLAG/dataset/AGPIL/dev"  # or use args.agp_val_dir if available
    pol_files = [os.path.join(agp_dir, f) for f in os.listdir(agp_dir) if f.endswith('.pol')]
    rel_to_opts = []
    for pol_path in pol_files:
        points = read_single_pol_file(pol_path, normalize=False).numpy()
        greedy_idxs, _ = greedy_guard_selection_fast(points, name=pol_path)
        opt_idxs = get_optimal_solution_indices(pol_path, agp_dir)
        rel_to_opt = float('nan')
        if len(opt_idxs) > 0:
            rel_to_opt = len(greedy_idxs) / float(len(opt_idxs)) if len(opt_idxs) > 0 else float('nan')
        rel_to_opts.append(rel_to_opt)
    draw_all_greedy_vs_optimal(pol_files, agp_dir, rel_to_opts, out_dir="gfx")
