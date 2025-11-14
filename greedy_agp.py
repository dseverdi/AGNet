import os
from dotenv import load_dotenv
import argparse
import numpy as np
import torch
import json
import time
import pickle
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
    
    # Explicit cleanup of temporary skgeom objects
    del arr, vs, edges
    import gc
    gc.collect()
    
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
    
    # Explicit cleanup of skgeom objects to prevent memory leaks
    del vis_polys, poly, covered
    import gc
    gc.collect()
    
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

def process_single_sample(pol_path, normalize, agp_val_dir, verbose, coverage_threshold=1.0):
    import numpy as np
    try:
        points = read_single_pol_file(pol_path, normalize=normalize)
        n_vertices = len(points)
        guards, coverages = greedy_guard_selection_fast(points.numpy(), verbose=verbose, name=pol_path, coverage_threshold=coverage_threshold)
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

def process_single_polygon_for_cache(pol_path, normalize, coverage_threshold):
    """
    Process a single polygon for cache building (worker function for parallel processing).
    Returns tuple: (name, cache_entry) or (name, None) on failure
    """
    from rewards import strict_reward as reward_fn
    from functools import partial
    
    reward_func = partial(reward_fn, alpha=1.0, M=1000.0)
    
    try:
        points = read_single_pol_file(pol_path, normalize=normalize)
        name = os.path.splitext(os.path.basename(pol_path))[0]
        
        # Run greedy
        guards, coverages = greedy_guard_selection_fast(
            points.numpy(), verbose=False, name=pol_path, 
            coverage_threshold=coverage_threshold
        )
        
        # Compute reward
        reward = reward_func(points.numpy(), np.array(guards), name, length=len(points))
        
        # Create cache entry (clamp coverage to [0, 1] to avoid floating point errors)
        final_coverage = float(coverages[-1]) if coverages else 0.0
        cache_entry = {
            'guards': guards,
            'reward': float(reward),
            'num_guards': len(guards),
            'coverage': min(1.0, max(0.0, final_coverage))
        }
        
        return (name, cache_entry)
    except Exception as e:
        print(f"Failed to cache {os.path.basename(pol_path)}: {e}")
        return (os.path.splitext(os.path.basename(pol_path))[0], None)

def build_greedy_cache(pol_files, normalize=True, verbose=False, coverage_threshold=1.0, max_workers=4, checkpoint_every=500, start_offset=0, existing_cache=None):
    """
    Build a cache of greedy solutions for all polygons.
    
    Args:
        max_workers: Number of parallel workers (1 for sequential, >1 for parallel)
        checkpoint_every: Save checkpoint every N polygons (for crash recovery)
        start_offset: Starting index offset for progress reporting when resuming
        existing_cache: Existing cache to merge with (for resuming from checkpoint)
    """
    from rewards import strict_reward as reward_fn
    from functools import partial
    
    cache = existing_cache if existing_cache is not None else {}
    total = len(pol_files)
    failed = 0
    reward_func = partial(reward_fn, alpha=1.0, M=1000.0)
    
    if max_workers == 1:
        # Sequential processing - more stable, avoids memory issues
        print(f"\n[BUILD] Starting sequential cache building for {total} polygons...")
        print(f"[BUILD] Estimated time: ~{total * 10 / 60:.1f} minutes")
        print(f"[BUILD] Progress updates every 50 polygons\n")
        
        start_time = time.time()
        for i, pol_path in enumerate(pol_files, 1):
            try:
                points = read_single_pol_file(pol_path, normalize=normalize)
                name = os.path.splitext(os.path.basename(pol_path))[0]
                
                # Run greedy
                guards, coverages = greedy_guard_selection_fast(
                    points.numpy(), verbose=False, name=pol_path, 
                    coverage_threshold=coverage_threshold
                )
                
                # Force garbage collection after each polygon to prevent memory leaks
                import gc
                gc.collect()
                
                # Compute reward
                reward = reward_func(points.numpy(), np.array(guards), name, length=len(points))
                
                # Create cache entry (clamp coverage to [0, 1] to avoid floating point errors)
                final_coverage = float(coverages[-1]) if coverages else 0.0
                cache[name] = {
                    'guards': guards,
                    'reward': float(reward),
                    'num_guards': len(guards),
                    'coverage': min(1.0, max(0.0, final_coverage))
                }
                
                # Print progress (use start_offset + i for absolute position)
                abs_idx = start_offset + i
                elapsed = time.time() - start_time
                avg_time_per_poly = elapsed / i if i > 0 else 0
                eta_seconds = avg_time_per_poly * (total - i)
                eta_minutes = eta_seconds / 60
                
                if i % 50 == 0 or i == total:
                    progress_pct = 100 * abs_idx / (start_offset + total)
                    print(f"[{abs_idx}/{start_offset + total}] ({progress_pct:.1f}%) Cached {name}: "
                          f"{len(guards)} guards, coverage={coverages[-1] if coverages else 0:.3f}, "
                          f"reward={reward:.2f} | ETA: {eta_minutes:.1f}min")
                
                # Periodic checkpoint (based on absolute position)
                if checkpoint_every > 0 and abs_idx % checkpoint_every == 0:
                    checkpoint_path = f"data/greedy_cache_checkpoint_{abs_idx}.pkl"
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    print(f"  → [CHECKPOINT] Saving to {checkpoint_path}...")
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump(cache, f)
                    checkpoint_size = os.path.getsize(checkpoint_path) / 1024  # KB
                    print(f"  → [CHECKPOINT] Saved successfully ({checkpoint_size:.1f} KB, {len(cache)} entries)")
                
            except Exception as e:
                failed += 1
                name = os.path.splitext(os.path.basename(pol_path))[0]
                abs_idx = start_offset + i
                print(f"[{abs_idx}/{start_offset + total}] ⚠️  ERROR caching {name}: {e}")
                
                if failed > 100:
                    print(f"\n❌ Too many failures ({failed}), stopping")
                    break
        
        # Print actual timing for sequential processing
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / len(cache) if len(cache) > 0 else 0
        print(f"\n{'='*60}")
        print(f"✅ Cache building complete!")
        print(f"   Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
        print(f"   Avg time per polygon: {avg_time:.2f}s")
        print(f"   Processed: {len(cache) - start_offset} new polygons")
        print(f"   Failed: {failed} polygons")
        print(f"{'='*60}")
    
    else:
        # Parallel processing with error handling
        print(f"Building cache for {total} polygons using {max_workers} parallel workers...")
        print(f"Estimated time: ~{total * 4 / max_workers / 60:.1f} minutes")
        
        start_time = time.time()
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_polygon_for_cache, pol_path, normalize, coverage_threshold): pol_path
                for pol_path in pol_files
            }
            
            completed = 0
            for future in as_completed(futures):
                pol_path = futures[future]
                completed += 1
                
                try:
                    name, cache_entry = future.result(timeout=120)
                    
                    if cache_entry is not None:
                        cache[name] = cache_entry
                        if completed % 50 == 0 or completed == total:
                            print(f"[{completed}/{total}] Cached {name}: {cache_entry['num_guards']} guards, "
                                  f"coverage={cache_entry['coverage']:.3f}, reward={cache_entry['reward']:.2f}")
                    else:
                        failed += 1
                        print(f"[{completed}/{total}] Failed to cache {name}")
                        
                except Exception as e:
                    failed += 1
                    name = os.path.splitext(os.path.basename(pol_path))[0]
                    print(f"[{completed}/{total}] ERROR caching {name}: {e}")
                    
                    if failed > 100:
                        print(f"\nToo many failures ({failed}), stopping")
                        break
                
                # Periodic checkpoint
                if checkpoint_every > 0 and completed % checkpoint_every == 0:
                    checkpoint_path = f"data/greedy_cache_checkpoint_{completed}.pkl"
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump(cache, f)
                    print(f"  → Checkpoint saved: {checkpoint_path}")
        
        # Print actual timing for parallel processing
        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / len(cache) if len(cache) > 0 else 0
        print(f"\nActual cache building time: {elapsed_time:.1f}s ({avg_time:.2f}s per polygon)")
    
    print(f"Successfully cached {len(cache)}/{total} polygons ({failed} failed)")
    return cache

def greedy_eval_comprehensive(pol_files, agp_val_dir, normalize=True, verbose=False, coverage_threshold=1.0):
    """Comprehensive evaluation of greedy algorithm with essential metrics only."""
    print(f"\n--- Comprehensive Greedy Evaluation on {len(pol_files)} samples ---")
    
    # Essential metrics only
    size_ratios = []  # greedy_size / optimal_size (approx ratio)
    polygon_coverages = []  # geometric coverage of the polygon
    
    total_samples = len(pol_files)
    print(f"Processing {total_samples} samples...")
    
    # Limit workers to avoid memory issues and add error handling
    max_workers = min(4, (os.cpu_count() or 1))
    completed = 0
    failed = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_comprehensive_sample, pol_path, normalize, agp_val_dir, verbose, coverage_threshold): pol_path for pol_path in pol_files}
        
        for future in as_completed(futures):
            pol_path = futures[future]
            completed += 1
            
            try:
                result = future.result(timeout=60)  # 60 second timeout per polygon
                if result is not None:
                    size_ratio, polygon_coverage = result
                    size_ratios.append(size_ratio)
                    polygon_coverages.append(polygon_coverage)
                    
                # Progress feedback every 100 polygons
                if completed % 100 == 0:
                    print(f"Progress: {completed}/{total_samples} polygons processed ({failed} failed)")
                    
            except Exception as e:
                failed += 1
                print(f"[ERROR] Failed to process {os.path.basename(pol_path)}: {e}")
                if failed > 100:  # Stop if too many failures
                    print(f"Too many failures ({failed}), stopping evaluation")
                    break
    
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
    
    # Compute statistics for essential metrics only
    if len(size_ratios) > 0:
        ratio_stats = compute_stats(size_ratios, "Approximation Ratio (greedy/optimal)")
        
        # Summary metrics
        print(f"\n=== SUMMARY ===")
        print(f"Instances evaluated: {len(size_ratios)}")
        print(f"Average approximation ratio: {np.mean(size_ratios):.3f}")
        print(f"Median approximation ratio: {np.median(size_ratios):.3f}")
    else:
        print("No valid instances with optimal solutions found for comparison.")
        ratio_stats = {}
    
    # Compute coverage statistics from polygon visibility
    coverage_array = np.array([c for c in polygon_coverages if not np.isnan(c)])
    coverage_vis_stats = compute_stats(coverage_array, "Polygon Coverage") if len(coverage_array) > 0 else {}
    
    return {
        # Essential metrics only
        'size_ratios': size_ratios,
        'polygon_coverages': coverage_array.tolist() if len(coverage_array) > 0 else [],
        'stats': {
            'ratio_stats': ratio_stats,
            'coverage_vis_stats': coverage_vis_stats
        }
    }

def process_comprehensive_sample(pol_path, normalize, agp_val_dir, verbose, coverage_threshold=1.0):
    """Process a single sample for comprehensive evaluation."""
    try:
        points = read_single_pol_file(pol_path, normalize=normalize)
        n_vertices = len(points)
        pred_indices, coverages = greedy_guard_selection_fast(points.numpy(), verbose=verbose, name=pol_path, coverage_threshold=coverage_threshold)
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
            pred_size = len(pred_indices)
            true_size = len(true_indices)

            # Size ratio: how many times larger is the greedy vs optimal (approx ratio)
            size_ratio = pred_size / true_size if true_size > 0 else float('inf')

            return (size_ratio, final_coverage)
        else:
            return None
    except Exception as e:
        if verbose:
            print(f"[ERROR] Sample: {os.path.basename(pol_path)} | Exception: {e}")
        return None

def main():
    # Load environment variables from .env
    load_dotenv()
    DATASET_PATH = os.getenv("DATASET_PATH")
    if not DATASET_PATH:
        raise EnvironmentError("DATASET_PATH environment variable must be set in .env file.")
    parser = argparse.ArgumentParser()
    default_val = os.path.join(DATASET_PATH, "dev")
    parser.add_argument('--agp_val_dir', type=str, default=default_val, help='Directory with validation .pol files')
    parser.add_argument('--max-instances', type=int, default=10000, help='Maximum number of validation samples to use (default: 8000, or all if smaller)')
    parser.add_argument('--normalize', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--comprehensive', action='store_true', default=True, help='Use comprehensive evaluation with whisker plot statistics')
    parser.add_argument('--coverage-threshold', type=float, default=1.0, help='Stop greedy when polygon coverage >= threshold (0..1)')
    parser.add_argument('--cache', type=str, default=None, help='Save greedy solutions to cache file (e.g., data/greedy_baseline_cache.pkl)')
    parser.add_argument('--workers', type=int, default=1, help='Number of parallel workers for cache generation (default: 1, use 1 for sequential/stable processing)')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume from checkpoint file (e.g., data/greedy_cache_checkpoint_3500.pkl)')
    parser.add_argument('--cache-only', action='store_true', default=False, help='Only build cache, skip evaluation')
    args = parser.parse_args()

    agp_dir = args.agp_val_dir
    pol_files = [os.path.join(agp_dir, f) for f in os.listdir(agp_dir) if f.endswith('.pol')]
    if args.max_instances is not None:
        pol_files = pol_files[:args.max_instances]

    # If cache-only mode, skip evaluation and just build cache
    if args.cache_only and args.cache:
        print(f"\n{'='*60}")
        print(f"CACHE-ONLY MODE: Building cache for {len(pol_files)} polygons")
        print(f"{'='*60}")
        
        # Check if resuming from checkpoint
        cache = {}
        start_idx = 0
        if args.resume_from and os.path.exists(args.resume_from):
            print(f"\n[RESUME] Loading checkpoint from {args.resume_from}...")
            with open(args.resume_from, 'rb') as f:
                cache = pickle.load(f)
            start_idx = len(cache)
            print(f"[RESUME] Successfully loaded {start_idx} cached polygons")
            print(f"[RESUME] Will resume from index {start_idx}")
            
            # Filter pol_files to skip already cached polygons
            cached_names = set(cache.keys())
            pol_files_remaining = [p for p in pol_files 
                                  if os.path.splitext(os.path.basename(p))[0] not in cached_names]
            print(f"[RESUME] Remaining to process: {len(pol_files_remaining)}/{len(pol_files)}")
            print(f"[RESUME] Progress: {start_idx}/{len(pol_files)} ({100*start_idx/len(pol_files):.1f}% complete)")
        else:
            print(f"\n[START] Starting fresh cache generation...")
            pol_files_remaining = pol_files
        
        print(f"\n[INFO] Checkpoint will be saved every 500 polygons")
        print(f"[INFO] Estimated time remaining: ~{len(pol_files_remaining) * 8 / 60:.1f} minutes")
        print(f"{'='*60}\n")
        
        # Build cache for remaining polygons (merges with existing cache if resuming)
        print(f"[START] Calling build_greedy_cache with {len(pol_files_remaining)} polygons...\n")
        cache = build_greedy_cache(pol_files_remaining, args.normalize, args.verbose, 
                                  args.coverage_threshold, max_workers=args.workers, 
                                  checkpoint_every=500, start_offset=start_idx, 
                                  existing_cache=cache)
        
        # Save final cache
        cache_dir = os.path.dirname(args.cache)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(args.cache, 'wb') as f:
            pickle.dump(cache, f)
        print(f"\nSaved greedy baseline cache to {args.cache}")
        print(f"  - {len(cache)} polygons cached")
        if len(cache) > 0:
            print(f"  - Avg guards: {np.mean([d['num_guards'] for d in cache.values()]):.1f}")
            print(f"  - Avg coverage: {np.mean([d['coverage'] for d in cache.values()]):.3f}")
    
    elif args.comprehensive:
        # Use comprehensive evaluation with timing
        start_time = time.time()
        eval_results = greedy_eval_comprehensive(pol_files, args.agp_val_dir, normalize=args.normalize, verbose=args.verbose, coverage_threshold=args.coverage_threshold)
        total_time = time.time() - start_time
        avg_time_per_instance = total_time / len(pol_files) if len(pol_files) > 0 else 0
        
        print(f"Timing: total {total_time:.3f}s, avg {avg_time_per_instance:.4f}s/instance")
        
        # Save evaluation results
        import json
        results_summary = {
            'args': vars(args),
            'num_samples': len(pol_files),
            'training_method': 'greedy_algorithm',
            'total_evaluation_time': total_time,
            'avg_time_per_instance': avg_time_per_instance
        }
        
        # Add statistics if available
        if 'stats' in eval_results and eval_results['stats']:
            if 'ratio_stats' in eval_results['stats']:
                results_summary['approx_ratio'] = eval_results['stats']['ratio_stats']
            if 'coverage_vis_stats' in eval_results['stats']:
                results_summary['polygon_coverage'] = eval_results['stats']['coverage_vis_stats']
        
        # Save to results directory
        os.makedirs('results', exist_ok=True)
        with open('results/greedy_agp_evaluation.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        print("Results summary saved to results/greedy_agp_evaluation.json")
        
        # Save cache if requested
        if args.cache:
            print(f"\nGenerating greedy baseline cache...")
            
            # Check if resuming from checkpoint
            cache = {}
            start_idx = 0
            if args.resume_from and os.path.exists(args.resume_from):
                print(f"Loading checkpoint from {args.resume_from}...")
                with open(args.resume_from, 'rb') as f:
                    cache = pickle.load(f)
                start_idx = len(cache)
                print(f"Loaded {start_idx} cached polygons, resuming from index {start_idx}")
                # Filter pol_files to skip already cached polygons
                cached_names = set(cache.keys())
                pol_files_remaining = [p for p in pol_files 
                                      if os.path.splitext(os.path.basename(p))[0] not in cached_names]
                print(f"Remaining to process: {len(pol_files_remaining)}/{len(pol_files)}")
            else:
                pol_files_remaining = pol_files
            
            # Build cache for remaining polygons (merges with existing cache if resuming)
            cache = build_greedy_cache(pol_files_remaining, args.normalize, args.verbose, 
                                      args.coverage_threshold, max_workers=args.workers, 
                                      checkpoint_every=500, start_offset=start_idx, 
                                      existing_cache=cache)
            
            # Save final cache
            cache_dir = os.path.dirname(args.cache)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(args.cache, 'wb') as f:
                pickle.dump(cache, f)
            print(f"\nSaved greedy baseline cache to {args.cache}")
            print(f"  - {len(cache)} polygons cached")
            if len(cache) > 0:
                print(f"  - Avg guards: {np.mean([d['num_guards'] for d in cache.values()]):.1f}")
                print(f"  - Avg coverage: {np.mean([d['coverage'] for d in cache.values()]):.3f}")
        
    else:
        # Use legacy evaluation
        all_num_guards = []
        all_coverages = []
        all_rel_sizes = []
        all_rel_to_opt = []
        total_samples = len(pol_files)
        print(f"Processing {total_samples} samples in parallel...")
        # Start timing for legacy parallel evaluation
        legacy_start_time = time.time()
        suspicious_cases = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_single_sample, pol_path, args.normalize, args.agp_val_dir, args.verbose, args.coverage_threshold) for pol_path in pol_files]
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
        total_legacy_time = time.time() - legacy_start_time
        avg_time_per_instance_legacy = total_legacy_time / len(all_results) if len(all_results) > 0 else 0
        print(f"\nProcessed {len(all_results)} instances.")
        print(f"Timing: total {total_legacy_time:.3f}s, avg {avg_time_per_instance_legacy:.4f}s/instance")
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
        # Optionally, save legacy timing into a results file for backward compatibility
        try:
            legacy_results = {
                'args': vars(args),
                'num_samples': len(all_results),
                'training_method': 'greedy_algorithm_legacy',
                'total_evaluation_time': total_legacy_time,
                'avg_time_per_instance': avg_time_per_instance_legacy,
                'metrics': stats
            }
            os.makedirs('results', exist_ok=True)
            with open('results/greedy_agp_evaluation_all.json', 'w') as f:
                json.dump(legacy_results, f, indent=2)
            print("Legacy results (including timing) saved to results/greedy_agp_evaluation_all.json")
        except Exception:
            pass

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

def draw_all_greedy_vs_optimal(pol_files, agp_val_dir, rel_to_opts, out_dir=None, coverage_threshold=1.0):
    # Sequential processing, no parallelism
    for pol_path, rel_to_opt in zip(pol_files, rel_to_opts):
        if rel_to_opt < 1.0:
            points = read_single_pol_file(pol_path, normalize=False).numpy()
            greedy_idxs, _ = greedy_guard_selection_fast(points, name=pol_path, coverage_threshold=coverage_threshold)
            greedy_vis_polys, _ = precompute_visibility_polygons(points, pol_path)
            opt_idxs = get_optimal_solution_indices(pol_path, agp_val_dir)
            opt_vis_polys, _ = precompute_visibility_polygons(points, pol_path)
            visualize_greedy_vs_optimal(pol_path, greedy_idxs, greedy_vis_polys, opt_idxs, opt_vis_polys, out_dir=out_dir)

if __name__ == "__main__":
    main()
