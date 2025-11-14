#!/usr/bin/env python3
"""
Analyze cache memory usage and data structure
"""
import pickle
import sys
import os
import psutil
import gc
from collections import defaultdict

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def analyze_cache_structure(cache_file):
    """Analyze the cache data structure and memory usage"""
    print(f"{'='*60}")
    print(f"ANALYZING CACHE: {cache_file}")
    print(f"{'='*60}")

    # Load cache
    print(f"\n[LOADING] Loading cache from {cache_file}...")
    initial_memory = get_memory_usage()
    print(f"[MEMORY] Initial memory: {initial_memory:.1f} MB")

    try:
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to load cache: {e}")
        return

    loaded_memory = get_memory_usage()
    print(f"[MEMORY] After loading: {loaded_memory:.1f} MB (+{loaded_memory - initial_memory:.1f} MB)")

    # Basic statistics
    num_entries = len(cache)
    print(f"\n[CACHE STATS]")
    print(f"  Total entries: {num_entries}")
    print(f"  Cache type: {type(cache).__name__}")

    if num_entries == 0:
        print("  Cache is empty!")
        return

    # Analyze entry structure
    print(f"\n[ENTRY STRUCTURE ANALYSIS]")
    first_key = next(iter(cache.keys()))
    first_entry = cache[first_key]

    print(f"  Key type: {type(first_key).__name__}")
    print(f"  Key example: '{first_key}'")
    print(f"  Entry type: {type(first_entry).__name__}")
    print(f"  Entry fields: {list(first_entry.keys())}")

    # Analyze each field
    field_stats = defaultdict(list)
    total_memory_per_field = defaultdict(float)

    for name, entry in cache.items():
        for field, value in entry.items():
            field_stats[field].append(value)

            # Estimate memory usage per field
            if isinstance(value, list):
                # List of integers (guard indices)
                total_memory_per_field[field] += sys.getsizeof(value) + sum(sys.getsizeof(x) for x in value)
            else:
                # Simple types (int, float)
                total_memory_per_field[field] += sys.getsizeof(value)

    print(f"\n[FIELD ANALYSIS]")
    for field in first_entry.keys():
        values = field_stats[field]
        if field == 'guards':
            # Special handling for guards list
            guard_lengths = [len(guards) for guards in values]
            print(f"  '{field}': list of {len(values)} lists")
            print(f"    - Avg guards per polygon: {sum(guard_lengths)/len(guard_lengths):.1f}")
            print(f"    - Min guards: {min(guard_lengths)}")
            print(f"    - Max guards: {max(guard_lengths)}")
            print(f"    - Memory: {total_memory_per_field[field]/1024/1024:.2f} MB")
        elif field in ['reward', 'coverage']:
            print(f"  '{field}': {type(values[0]).__name__} values")
            print(f"    - Range: [{min(values):.3f}, {max(values):.3f}]")
            print(f"    - Memory: {total_memory_per_field[field]/1024/1024:.2f} MB")
        elif field == 'num_guards':
            print(f"  '{field}': {type(values[0]).__name__} values")
            print(f"    - Range: [{min(values)}, {max(values)}]")
            print(f"    - Memory: {total_memory_per_field[field]/1024/1024:.2f} MB")

    # Total memory breakdown
    print(f"\n[MEMORY BREAKDOWN]")
    total_data_memory = sum(total_memory_per_field.values())
    print(f"  Data memory: {total_data_memory/1024/1024:.2f} MB")

    # Dictionary overhead
    dict_overhead = sys.getsizeof(cache)
    for key in cache.keys():
        dict_overhead += sys.getsizeof(key)
    for entry in cache.values():
        dict_overhead += sys.getsizeof(entry)
    print(f"  Dictionary overhead: {dict_overhead/1024/1024:.2f} MB")

    total_estimated = (total_data_memory + dict_overhead) / 1024 / 1024
    print(f"  Total estimated: {total_estimated:.2f} MB")

    # File size
    file_size = os.path.getsize(cache_file) / 1024 / 1024
    print(f"  File size on disk: {file_size:.2f} MB")

    # Compression ratio
    compression_ratio = total_estimated / file_size if file_size > 0 else 0
    print(f"  Pickle compression ratio: {compression_ratio:.2f}x")

    # Memory per entry
    memory_per_entry = total_estimated / num_entries * 1024  # Convert to KB
    print(f"  Memory per entry: {memory_per_entry:.2f} KB")

    final_memory = get_memory_usage()
    print(f"\n[MEMORY USAGE]")
    print(f"  Final memory: {final_memory:.1f} MB")
    print(f"  Memory increase: {final_memory - initial_memory:.1f} MB")

    # Test garbage collection
    print(f"\n[GARBAGE COLLECTION TEST]")
    del cache
    gc.collect()
    gc_memory = get_memory_usage()
    print(f"  After GC: {gc_memory:.1f} MB")
    print(f"  Memory freed: {final_memory - gc_memory:.1f} MB")

    print(f"\n{'='*60}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_cache.py <cache_file>")
        sys.exit(1)

    cache_file = sys.argv[1]
    if not os.path.exists(cache_file):
        print(f"Error: Cache file '{cache_file}' does not exist")
        sys.exit(1)

    analyze_cache_structure(cache_file)