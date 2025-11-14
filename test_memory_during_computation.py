#!/usr/bin/env python3
"""
Test memory usage during greedy algorithm computation
"""
import os
import sys
import psutil
import time
import tracemalloc
from utils import evaluate_polygon_visibility_numpy_wo_gt
from greedy_agp import greedy_guard_selection_fast, read_single_pol_file
from rewards import strict_reward

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_single_polygon_memory(pol_file, verbose=True):
    """Test memory usage for a single polygon computation"""
    if verbose:
        print(f"\n{'='*50}")
        print(f"TESTING: {os.path.basename(pol_file)}")
        print(f"{'='*50}")

    # Start memory tracking
    tracemalloc.start()
    initial_memory = get_memory_usage()
    initial_snapshot = tracemalloc.take_snapshot()

    if verbose:
        print(f"[MEMORY] Initial: {initial_memory:.1f} MB")

    # Load polygon
    load_start = time.time()
    points = read_single_pol_file(pol_file, normalize=True)
    load_time = time.time() - load_start
    load_memory = get_memory_usage()

    if verbose:
        print(f"[LOAD] Time: {load_time:.3f}s, Memory: {load_memory:.1f} MB (+{load_memory - initial_memory:.1f} MB)")

    # Run greedy algorithm
    greedy_start = time.time()
    guards, coverages = greedy_guard_selection_fast(
        points.numpy(), verbose=False, name=pol_file, coverage_threshold=1.0
    )
    greedy_time = time.time() - greedy_start
    greedy_memory = get_memory_usage()

    if verbose:
        print(f"[GREEDY] Time: {greedy_time:.3f}s, Memory: {greedy_memory:.1f} MB (+{greedy_memory - load_memory:.1f} MB)")
        print(f"[GREEDY] Guards: {len(guards)}, Final coverage: {coverages[-1]:.3f}")

    # Compute reward
    reward_start = time.time()
    reward_func = lambda points, guards, name, length: strict_reward(points, guards, name, length, alpha=1.0, M=1000.0)
    reward = reward_func(points.numpy(), guards, os.path.basename(pol_file), len(points))
    reward_time = time.time() - reward_start
    reward_memory = get_memory_usage()

    if verbose:
        print(f"[REWARD] Time: {reward_time:.3f}s, Memory: {reward_memory:.1f} MB (+{reward_memory - greedy_memory:.1f} MB)")
        print(f"[REWARD] Value: {reward:.3f}")

    # Memory analysis
    final_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    if verbose:
        print(f"\n[MEMORY ANALYSIS]")
        print(f"  Peak memory: {get_memory_usage():.1f} MB")
        print(f"  Total increase: {get_memory_usage() - initial_memory:.1f} MB")

        # Top memory consumers
        stats = final_snapshot.compare_to(initial_snapshot, 'lineno')
        print(f"  Top memory allocations:")
        for stat in stats[:10]:  # Show more
            if stat.size_diff > 0:  # Only show allocations
                print(f"    +{stat.size_diff/1024/1024:.2f} MB: {stat.traceback.format()[-1]}")

    # Force cleanup
    import gc
    gc.collect()

    result = {
        'polygon': os.path.basename(pol_file),
        'load_time': load_time,
        'greedy_time': greedy_time,
        'reward_time': reward_time,
        'total_time': load_time + greedy_time + reward_time,
        'initial_memory': initial_memory,
        'peak_memory': get_memory_usage(),
        'memory_increase': get_memory_usage() - initial_memory,
        'num_guards': len(guards),
        'final_coverage': coverages[-1] if coverages else 0.0,
        'reward': reward
    }

    # Now delete variables
    del points, guards, coverages, reward
    gc.collect()

    return result

def test_multiple_polygons(pol_files, max_test=10):
    """Test memory usage on multiple polygons"""
    print(f"{'='*70}")
    print(f"MEMORY USAGE TEST: Testing {min(len(pol_files), max_test)} polygons")
    print(f"{'='*70}")

    results = []
    for i, pol_file in enumerate(pol_files[:max_test]):
        print(f"\n[TEST {i+1}/{min(len(pol_files), max_test)}]")
        result = test_single_polygon_memory(pol_file, verbose=True)
        results.append(result)

        # Force garbage collection between tests
        import gc
        gc.collect()
        time.sleep(0.1)  # Small delay

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")

    if results:
        times = [r['total_time'] for r in results]
        memories = [r['memory_increase'] for r in results]
        guards = [r['num_guards'] for r in results]

        print(f"Time per polygon: {sum(times)/len(times):.3f}s (range: {min(times):.3f}s - {max(times):.3f}s)")
        print(f"Memory increase: {sum(memories)/len(memories):.1f} MB (range: {min(memories):.1f} MB - {max(memories):.1f} MB)")
        print(f"Guards per polygon: {sum(guards)/len(guards):.1f} (range: {min(guards)} - {max(guards)})")

        # Check for memory leaks
        memory_trend = [r['peak_memory'] for r in results]
        if len(memory_trend) > 1:
            trend = memory_trend[-1] - memory_trend[0]
            print(f"Memory trend: {'increasing' if trend > 1 else 'stable'} ({trend:.1f} MB over {len(results)} tests)")

    return results

if __name__ == "__main__":
    # Get polygon files
    dataset_path = os.getenv("DATASET_PATH", "/home/dseverdi/Radno/MLAG/dataset")
    train_dir = os.path.join(dataset_path, "AGPIL", "train")

    if not os.path.exists(train_dir):
        print(f"Error: Training directory not found: {train_dir}")
        sys.exit(1)

    pol_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.pol')]
    pol_files.sort()  # For reproducible results

    if len(pol_files) == 0:
        print(f"Error: No .pol files found in {train_dir}")
        sys.exit(1)

    print(f"Found {len(pol_files)} polygon files")

    # Test memory usage
    results = test_multiple_polygons(pol_files, max_test=5)  # Test first 5 polygons

    # Save results
    import json
    with open('memory_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to memory_test_results.json")