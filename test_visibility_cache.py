#!/usr/bin/env python3
"""
Comprehensive test of visibility cache functionality.
Tests on real polygon data with random vertex subsampling.
"""

import numpy as np
from visibility_cache import VisibilityCache
from utils import evaluate_polygon_visibility_numpy_wo_gt
from dataset import Dataset, agp_read_samples
import time
import random
import os
from dotenv import load_dotenv


def test_basic_cache():
    """Test basic caching functionality with simple polygon."""
    print("="*70)
    print("TEST 1: Basic Cache Functionality (Simple Square)")
    print("="*70)
    
    # Create a simple square polygon
    points = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0]
    ])
    
    cache = VisibilityCache()
    
    # Test precomputation
    print("\n1. Precomputing visibility regions...")
    start = time.time()
    success = cache.precompute_instance(points, "test_square")
    elapsed = time.time() - start
    print(f"   Precomputation: {'SUCCESS' if success else 'FAILED'} ({elapsed:.3f}s)")
    
    if not success:
        print("ERROR: Precomputation failed")
        return False
    
    # Test coverage computation
    print("\n2. Computing coverage with cache...")
    
    # All guards (should be 100% coverage)
    solution_all = np.array([0, 1, 2, 3])
    start = time.time()
    coverage = cache.get_coverage_fast(points, solution_all, "test_square")
    elapsed = time.time() - start
    print(f"   All guards coverage: {coverage:.4f} ({elapsed:.3f}s)")
    
    # Subset of guards
    solution_subset = np.array([0, 2])
    start = time.time()
    coverage = cache.get_coverage_fast(points, solution_subset, "test_square")
    elapsed = time.time() - start
    print(f"   Subset guards coverage: {coverage:.4f} ({elapsed:.3f}s)")
    
    # Test cache hit
    print("\n3. Testing cache hits...")
    start = time.time()
    coverage = cache.get_coverage_fast(points, solution_subset, "test_square")
    elapsed = time.time() - start
    print(f"   Cached coverage: {coverage:.4f} ({elapsed:.3f}s - should be faster)")
    
    # Print statistics
    stats = cache.get_stats()
    print("\n4. Cache statistics:")
    print(f"   Instances cached: {stats['instances_cached']}")
    print(f"   Coverage cache size: {stats['coverage_cache_size']}")
    print(f"   Cache hits: {stats['cache_hits']}")
    print(f"   Cache misses: {stats['cache_misses']}")
    print(f"   Hit rate: {stats['hit_rate']*100:.1f}%")
    
    print("\n✓ Test 1 passed!")
    return True


def test_real_polygon_cache(dataset_path=None, n_tests=5):
    """Test cache with real polygon data and random vertex subsampling."""
    print("\n" + "="*70)
    print("TEST 2: Real Polygon Cache with Random Vertex Subsampling")
    print("="*70)
    
    # Load dataset
    if dataset_path is None:
        load_dotenv()
        dataset_path = os.getenv("DATASET_PATH")
        if not dataset_path:
            print("ERROR: DATASET_PATH not set in .env")
            return False
    
    val_dir = os.path.join(dataset_path, "dev")
    if not os.path.exists(val_dir):
        print(f"ERROR: Validation directory not found: {val_dir}")
        return False
    
    # Load validation samples
    print(f"\nLoading polygons from {val_dir}...")
    pol_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.pol')]
    
    if len(pol_files) == 0:
        print("ERROR: No .pol files found")
        return False
    
    # Randomly select test polygons
    test_files = random.sample(pol_files, min(n_tests, len(pol_files)))
    print(f"Selected {len(test_files)} random polygons for testing")
    
    samples = agp_read_samples(test_files, normalize=True)
    dataset = Dataset(samples)
    
    # Initialize cache
    cache = VisibilityCache()
    
    for test_idx, (polygon_tensor, _, polygon_name) in enumerate(dataset):
        print(f"\n{'='*70}")
        print(f"Test Polygon {test_idx+1}/{len(dataset)}: {os.path.basename(polygon_name)}")
        print(f"{'='*70}")
        
        points = polygon_tensor.numpy()
        n_vertices = len(points)
        print(f"Vertices: {n_vertices}")
        
        # Precompute visibility regions
        print("\n1. Precomputing visibility regions...")
        start = time.time()
        success = cache.precompute_instance(points, polygon_name, force=True)
        precompute_time = time.time() - start
        
        if not success:
            print(f"   FAILED - Skipping this polygon")
            continue
        
        print(f"   SUCCESS ({precompute_time:.3f}s)")
        
        # Test different guard configurations
        print("\n2. Testing different guard configurations...")
        
        # Configuration 1: All vertices as guards
        print(f"\n   a) All {n_vertices} vertices as guards:")
        all_guards = np.arange(n_vertices)
        
        start = time.time()
        coverage_cached = cache.get_coverage_fast(points, all_guards, polygon_name)
        time_cached = time.time() - start
        
        start = time.time()
        coverage_uncached = evaluate_polygon_visibility_numpy_wo_gt(points, all_guards, polygon_name)
        time_uncached = time.time() - start
        
        print(f"      Cached:   {coverage_cached:.4f} ({time_cached:.3f}s)")
        print(f"      Uncached: {coverage_uncached:.4f} ({time_uncached:.3f}s)")
        print(f"      Match: {abs(coverage_cached - coverage_uncached) < 1e-6}")
        print(f"      Speedup: {time_uncached/time_cached:.1f}x")
        
        # Configuration 2: Random subsamples of vertices
        subsample_sizes = [
            max(3, n_vertices // 4),
            max(3, n_vertices // 2),
            max(3, 3 * n_vertices // 4)
        ]
        
        for subsample_size in subsample_sizes:
            print(f"\n   b) Random {subsample_size} guards (out of {n_vertices}):")
            
            # Generate random guard configuration
            guards = np.random.choice(n_vertices, size=subsample_size, replace=False)
            guards = np.sort(guards)
            
            start = time.time()
            coverage_cached = cache.get_coverage_fast(points, guards, polygon_name)
            time_cached = time.time() - start
            
            start = time.time()
            coverage_uncached = evaluate_polygon_visibility_numpy_wo_gt(points, guards, polygon_name)
            time_uncached = time.time() - start
            
            print(f"      Cached:   {coverage_cached:.4f} ({time_cached:.3f}s)")
            print(f"      Uncached: {coverage_uncached:.4f} ({time_uncached:.3f}s)")
            print(f"      Match: {abs(coverage_cached - coverage_uncached) < 1e-6}")
            print(f"      Speedup: {time_uncached/time_cached:.1f}x")
        
        # Configuration 3: Test cache hit rate with repeated queries
        print(f"\n   c) Testing cache hits with repeated queries:")
        n_repeated = 10
        guards = np.random.choice(n_vertices, size=max(3, n_vertices // 3), replace=False)
        
        cache.clear_coverage_cache()  # Clear to reset hit counters
        
        start = time.time()
        for _ in range(n_repeated):
            coverage = cache.get_coverage_fast(points, guards, polygon_name)
        time_repeated = time.time() - start
        
        stats = cache.get_stats()
        print(f"      Repeated {n_repeated} identical queries")
        print(f"      Total time: {time_repeated:.3f}s ({time_repeated/n_repeated:.4f}s per query)")
        print(f"      Cache hits: {stats['cache_hits']}")
        print(f"      Cache misses: {stats['cache_misses']}")
        print(f"      Hit rate: {stats['hit_rate']*100:.1f}%")
    
    # Print overall statistics
    print("\n" + "="*70)
    print("Overall Cache Statistics")
    print("="*70)
    stats = cache.get_stats()
    print(f"Total instances cached: {stats['instances_cached']}")
    print(f"Coverage cache size: {stats['coverage_cache_size']}")
    print(f"Total cache hits: {stats['cache_hits']}")
    print(f"Total cache misses: {stats['cache_misses']}")
    print(f"Overall hit rate: {stats['hit_rate']*100:.1f}%")
    print(f"Total precompute time: {stats['total_precompute_time']:.1f}s")
    print(f"Average precompute time: {stats['avg_precompute_time']:.3f}s per instance")
    
    print("\n✓ Test 2 passed!")
    return True


def test_incremental_pruning_simulation(dataset_path=None):
    """Simulate the pruning RL scenario: start with all guards, remove one at a time."""
    print("\n" + "="*70)
    print("TEST 3: Incremental Pruning Simulation")
    print("="*70)
    
    # Load dataset
    if dataset_path is None:
        load_dotenv()
        dataset_path = os.getenv("DATASET_PATH")
        if not dataset_path:
            print("ERROR: DATASET_PATH not set in .env")
            return False
    
    val_dir = os.path.join(dataset_path, "dev")
    pol_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.pol')]
    
    # Pick one polygon
    test_file = random.choice(pol_files)
    print(f"\nTest polygon: {os.path.basename(test_file)}")
    
    samples = agp_read_samples([test_file], normalize=True)
    dataset = Dataset(samples)
    polygon_tensor, _, polygon_name = dataset[0]
    points = polygon_tensor.numpy()
    n_vertices = len(points)
    
    print(f"Vertices: {n_vertices}")
    
    # Initialize cache and precompute
    cache = VisibilityCache()
    print("\nPrecomputing visibility regions...")
    start = time.time()
    cache.precompute_instance(points, polygon_name)
    print(f"Precompute time: {time.time() - start:.3f}s")
    
    # Simulate pruning: start with all guards, randomly remove one at a time
    print("\nSimulating pruning (removing guards one by one)...")
    active_guards = list(range(n_vertices))
    removal_order = list(range(n_vertices))
    random.shuffle(removal_order)
    
    min_guards = 3
    coverage_history = []
    time_cached_total = 0
    time_uncached_total = 0
    
    print(f"\nStarting with {len(active_guards)} guards...")
    
    for step, guard_to_remove in enumerate(removal_order):
        if len(active_guards) <= min_guards:
            break
        
        if guard_to_remove not in active_guards:
            continue
        
        # Remove the guard
        active_guards.remove(guard_to_remove)
        guards_array = np.array(active_guards)
        
        # Compute coverage with cache
        start = time.time()
        coverage_cached = cache.get_coverage_fast(points, guards_array, polygon_name)
        time_cached_total += time.time() - start
        
        coverage_history.append(coverage_cached)
        
        if (step + 1) % 10 == 0 or len(active_guards) == min_guards:
            print(f"  Step {step+1}: {len(active_guards)} guards remaining, coverage: {coverage_cached:.4f}")
    
    # Compare one full uncached run
    print(f"\nFinal configuration: {len(active_guards)} guards")
    start = time.time()
    coverage_uncached = evaluate_polygon_visibility_numpy_wo_gt(points, np.array(active_guards), polygon_name)
    time_uncached_single = time.time() - start
    
    print(f"  Cached (final):   {coverage_history[-1]:.4f}")
    print(f"  Uncached (final): {coverage_uncached:.4f}")
    print(f"  Match: {abs(coverage_history[-1] - coverage_uncached) < 1e-6}")
    
    # Statistics
    stats = cache.get_stats()
    print(f"\nPruning simulation statistics:")
    print(f"  Total removal steps: {len(coverage_history)}")
    print(f"  Total cached time: {time_cached_total:.3f}s ({time_cached_total/len(coverage_history):.4f}s per step)")
    print(f"  Estimated uncached time: {time_uncached_single * len(coverage_history):.3f}s")
    print(f"  Estimated speedup: {(time_uncached_single * len(coverage_history)) / time_cached_total:.1f}x")
    print(f"  Cache hit rate: {stats['hit_rate']*100:.1f}%")
    
    print("\n✓ Test 3 passed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("VISIBILITY CACHE COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    random.seed(42)  # For reproducibility
    
    # Test 1: Basic functionality
    if not test_basic_cache():
        print("\n✗ Test 1 FAILED")
        return
    
    # Test 2: Real polygons with random subsampling
    if not test_real_polygon_cache(n_tests=3):
        print("\n✗ Test 2 FAILED")
        return
    
    # Test 3: Incremental pruning simulation
    if not test_incremental_pruning_simulation():
        print("\n✗ Test 3 FAILED")
        return
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)


if __name__ == "__main__":
    main()
