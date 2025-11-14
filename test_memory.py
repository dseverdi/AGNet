#!/usr/bin/env python3
import os
import pickle
import time
import psutil
import gc
from greedy_agp import read_single_pol_file, greedy_guard_selection_fast
from rewards import strict_reward
from functools import partial
import numpy as np

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_checkpoint_loading():
    """Test loading the checkpoint"""
    print("Testing checkpoint loading...")
    start_time = time.time()
    start_mem = get_memory_usage()

    try:
        with open('data/greedy_cache_checkpoint_3500.pkl', 'rb') as f:
            cache = pickle.load(f)

        load_time = time.time() - start_time
        load_mem = get_memory_usage() - start_mem

        print(f"✅ Checkpoint loaded successfully!")
        print(f"   Entries: {len(cache)}")
        print(f"   Time: {load_time:.2f}s")
        print(f"   Memory: {load_mem:.1f} MB")
        print(f"   Current memory: {get_memory_usage():.1f} MB")

        return cache

    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return None

def test_single_polygon_processing():
    """Test processing a single polygon"""
    print("\nTesting single polygon processing...")

    # Get a polygon file
    pol_files = [f for f in os.listdir('/home/dseverdi/Radno/MLAG/dataset/AGPIL/train') if f.endswith('.pol')]
    if not pol_files:
        print("❌ No polygon files found")
        return

    pol_path = os.path.join('/home/dseverdi/Radno/MLAG/dataset/AGPIL/train', pol_files[0])
    print(f"Testing with: {pol_path}")

    start_time = time.time()
    start_mem = get_memory_usage()

    try:
        # Load polygon
        points = read_single_pol_file(pol_path, normalize=True)
        load_time = time.time() - start_time
        load_mem = get_memory_usage() - start_mem

        print(f"   Polygon loaded: {len(points)} vertices")
        print(f"   Load time: {load_time:.3f}s, memory: {load_mem:.1f} MB")

        # Test greedy algorithm
        from greedy_agp import greedy_guard_selection_fast
        start_time = time.time()
        start_mem = get_memory_usage()

        guards, coverages = greedy_guard_selection_fast(
            points.numpy(), verbose=False, name=pol_path,
            coverage_threshold=1.0
        )

        greedy_time = time.time() - start_time
        greedy_mem = get_memory_usage() - start_mem

        print(f"   Greedy result: {len(guards)} guards, coverage={coverages[-1]:.3f}")
        print(f"   Greedy time: {greedy_time:.3f}s, memory: {greedy_mem:.1f} MB")

        # Test reward computation
        reward_func = partial(strict_reward, alpha=1.0, M=1000.0)
        start_time = time.time()
        start_mem = get_memory_usage()

        reward = reward_func(points.numpy(), np.array(guards), os.path.splitext(os.path.basename(pol_path))[0], length=len(points))

        reward_time = time.time() - start_time
        reward_mem = get_memory_usage() - start_mem

        print(f"   Reward: {reward:.3f}")
        print(f"   Reward time: {reward_time:.3f}s, memory: {reward_mem:.1f} MB")

        print(f"   Total memory: {get_memory_usage():.1f} MB")

    except Exception as e:
        print(f"❌ Error processing polygon: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Memory usage test for greedy cache generation")
    print("=" * 50)

    # Test checkpoint loading
    cache = test_checkpoint_loading()

    # Test single polygon processing
    test_single_polygon_processing()

    # Final memory
    gc.collect()
    print(f"\nFinal memory usage: {get_memory_usage():.1f} MB")
    print("Test complete!")