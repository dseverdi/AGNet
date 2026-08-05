#!/usr/bin/env python3
"""
Unify Evaluation Results Script
Standardizes evaluation JSON files for consistent analysis in results.py
"""

import json
import os
import shutil
from pathlib import Path

def unify_evaluation_results(results_dir="results"):
    """Unify all evaluation result files to have consistent structure."""

    print("🔧 Unifying evaluation result files...")
    print("="*50)

    # Create backup directory
    backup_dir = os.path.join(results_dir, "backup_original")
    os.makedirs(backup_dir, exist_ok=True)

    # Key mapping from various formats to standardized format
    key_mappings = {
        'best_coverages': 'coverage_stats',
        'best_rewards': 'efficiency_stats',  # Map rewards to efficiency for RL methods
        'best_sizes': 'size_stats',  # Keep for reference but not primary metric
        'size_ratios': 'size_ratio_stats',
        'polygon_coverage_stats': 'polygon_coverage_stats'
    }

    # Files to process and their target names
    file_unifications = {
        "greedy_agp_evaluation_all.json": "greedy_agp_evaluation.json",
        "active_search_all_K=5.json": "active_search_all_K=5.json",  # Keep same name
        "qlearning_evaluation.json": "qlearning_evaluation.json",    # Keep same name
        "sl_agp_evaluation.json": "sl_agp_evaluation.json",          # Keep same name
        "rl_agp_evaluation_critic.json": "rl_agp_evaluation.json"    # Rename to match expected
    }

    unified_count = 0

    for source_file, target_file in file_unifications.items():
        source_path = os.path.join(results_dir, source_file)
        target_path = os.path.join(results_dir, target_file)

        if not os.path.exists(source_path):
            print(f"⚠️  Source file not found: {source_file}")
            continue

        try:
            # Load original data
            with open(source_path, 'r') as f:
                data = json.load(f)

            print(f"📂 Processing {source_file} → {target_file}")

            # Create unified data structure
            unified_data = {}

            # Copy all original data first
            unified_data.update(data)

            # Apply key mappings
            for old_key, new_key in key_mappings.items():
                if old_key in data and new_key not in data:
                    unified_data[new_key] = data[old_key]
                    print(f"  🔄 Mapped {old_key} → {new_key}")

            # Ensure all expected keys exist with defaults if missing
            expected_keys = ['coverage_stats', 'efficiency_stats', 'size_ratio_stats', 'polygon_coverage_stats']

            for key in expected_keys:
                if key not in unified_data:
                    # Create default structure for missing keys
                    if 'stats' in key:
                        unified_data[key] = {
                            'mean': None,
                            'median': None,
                            'std': None,
                            'min': None,
                            'max': None,
                            'q25': None,
                            'q75': None,
                            'iqr': None
                        }
                        print(f"  ➕ Added missing key: {key} (with null values)")

            # Special handling for RL methods - map best_rewards to efficiency_stats if it doesn't exist
            if 'best_rewards' in data and 'efficiency_stats' not in data:
                unified_data['efficiency_stats'] = data['best_rewards']
                print(f"  🔄 Mapped best_rewards → efficiency_stats")

            # Backup original file
            backup_path = os.path.join(backup_dir, source_file)
            shutil.copy2(source_path, backup_path)
            print(f"  💾 Backed up original to {backup_path}")

            # Save unified file
            with open(target_path, 'w') as f:
                json.dump(unified_data, f, indent=2)

            print(f"  ✅ Saved unified file: {target_file}")
            unified_count += 1

        except Exception as e:
            print(f"  ❌ Error processing {source_file}: {e}")

    print(f"\n🎉 Unification complete!")
    print(f"  • Files processed: {unified_count}")
    print(f"  • Original files backed up to: {backup_dir}")
    print(f"  • All files now have consistent key structure for results.py")

    # List the final files
    print(f"\n📋 Final unified files:")
    for target_file in file_unifications.values():
        target_path = os.path.join(results_dir, target_file)
        if os.path.exists(target_path):
            print(f"  ✓ {target_file}")

def verify_unified_structure(results_dir="results"):
    """Verify that all unified files have consistent structure."""

    print(f"\n🔍 Verifying unified file structure...")

    expected_keys = ['coverage_stats', 'efficiency_stats', 'size_ratio_stats', 'polygon_coverage_stats']

    file_mappings = {
        "greedy_agp_evaluation.json": "Greedy",
        "sl_agp_evaluation.json": "Supervised Learning",
        "qlearning_evaluation.json": "Q-learning Reinforcement Learning",
        "active_search_all_K=5.json": "Active Search K=5",
        "rl_agp_evaluation.json": "RL with Critic"
    }

    all_good = True

    for filename, method_name in file_mappings.items():
        filepath = os.path.join(results_dir, filename)

        if not os.path.exists(filepath):
            print(f"  ⚠️  Missing: {filename}")
            continue

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            missing_keys = []
            for key in expected_keys:
                if key not in data:
                    missing_keys.append(key)

            if missing_keys:
                print(f"  ❌ {method_name}: Missing keys {missing_keys}")
                all_good = False
            else:
                print(f"  ✅ {method_name}: All expected keys present")

        except Exception as e:
            print(f"  ❌ {method_name}: Error reading file - {e}")
            all_good = False

    if all_good:
        print(f"\n🎯 All files have consistent structure!")
        print(f"  Ready to run: python results.py")
    else:
        print(f"\n⚠️  Some files still have structural issues.")

if __name__ == "__main__":
    # Unify the files
    unify_evaluation_results()

    # Verify the unification
    verify_unified_structure()