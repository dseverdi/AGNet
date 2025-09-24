#!/usr/bin/env python3
"""
Results Analysis Script for AGNet
Compares Greedy, Supervised Learning, and Reinforcement Learning approaches
using evaluation JSON files and generates comprehensive tables and plots.
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def safe_format(value, format_str=".3f", default="N/A"):
    """Safely format a value that might be None."""
    if value is None or (isinstance(value, float) and (str(value).lower() in ['nan', 'inf', '-inf'])):
        return default
    try:
        return f"{value:{format_str}}"
    except (ValueError, TypeError):
        return default

def load_evaluation_results(results_dir="results"):
    """Load evaluation results from JSON files."""
    results = {}
    
    # Define the expected files and their method names
    file_mappings = {
        "greedy_agp_evaluation.json": "Greedy",
        "sl_agp_evaluation.json": "Supervised Learning", 
        "qlearning_evaluation.json": "RL: Q-learning",
        "active_search_all_K=5.json": "RL: Active Search K=5",
        "sampling_evaluation_all_K=1.json": "RL: Solution Sampling K=1",
        "sampling_evaluation_all_K=5.json": "RL: Solution Sampling K=5",
    }
    
    for filename, method_name in file_mappings.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results[method_name] = data
                    print(f"✓ Loaded {method_name} results from {filename}")
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
        else:
            print(f"✗ File not found: {filepath}")
    
    return results

def create_summary_table(results):
    """Create a summary table comparing all methods with key metrics only."""
    summary_data = []
    
    for method, data in results.items():
        # Extract key statistics
        row = {
            'Method': method,
            'Samples': data.get('num_samples', data.get('num_val_samples', data.get('num_instances', 'N/A'))),
        }
        
        # Polygon coverage - handle both formats
        poly_cov_key = None
        if 'polygon_coverage' in data:
            poly_cov_key = 'polygon_coverage'
        elif 'polygon_coverage_stats' in data:
            poly_cov_key = 'polygon_coverage_stats'
        elif 'coverage_stats' in data:
            poly_cov_key = 'coverage_stats'
            
        if poly_cov_key:
            poly_cov = data[poly_cov_key]
            row.update({
                'Polygon Coverage Mean': safe_format(poly_cov.get('mean')),
                'Polygon Coverage Std': safe_format(poly_cov.get('std')),
                'Polygon Coverage Median': safe_format(poly_cov.get('median')),
            })
        
        # Approximation ratio - handle both formats
        ratio_key = None
        if 'approx_ratio' in data:
            ratio_key = 'approx_ratio'
        elif 'size_ratio_stats' in data:
            ratio_key = 'size_ratio_stats'
            
        if ratio_key:
            ratio = data[ratio_key]
            row.update({
                'Approx Ratio Mean': safe_format(ratio.get('mean')),
                'Approx Ratio Std': safe_format(ratio.get('std')),
                'Approx Ratio Median': safe_format(ratio.get('median')),
            })
        
        # Timing information
        if 'avg_time_per_instance' in data:
            row['Avg Time/Instance (s)'] = safe_format(data['avg_time_per_instance'], ".4f")
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def create_performance_comparison_table(results):
    """Create a focused performance comparison table with key metrics only."""
    comparison_data = []
    
    for method, data in results.items():
        row = {'Method': method}
        
        # Polygon coverage - handle multiple formats
        coverage_key = None
        if 'polygon_coverage' in data:
            coverage_key = 'polygon_coverage'
        elif 'polygon_coverage_stats' in data:
            coverage_key = 'polygon_coverage_stats'
        elif 'coverage_stats' in data:
            coverage_key = 'coverage_stats'
            
        if coverage_key:
            cov = data[coverage_key]
            row['Polygon Coverage (μ±σ)'] = f"{safe_format(cov.get('mean'))} ± {safe_format(cov.get('std'))}"
            row['Polygon Coverage Median'] = safe_format(cov.get('median'))
        
        # Approximation ratio - handle both formats
        ratio_key = None
        if 'approx_ratio' in data:
            ratio_key = 'approx_ratio'
        elif 'size_ratio_stats' in data:
            ratio_key = 'size_ratio_stats'
            
        if ratio_key:
            ratio = data[ratio_key]
            row['Approx Ratio (μ±σ)'] = f"{safe_format(ratio.get('mean'))} ± {safe_format(ratio.get('std'))}"
            row['Approx Ratio Median'] = safe_format(ratio.get('median'))
        
        # Timing information
        if 'avg_time_per_instance' in data:
            row['Avg Time/Instance (s)'] = safe_format(data['avg_time_per_instance'], ".3f")
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def create_pareto_frontier_plot(results, output_dir="results/gfx/pareto"):
    """Create a Pareto frontier plot showing trade-offs between coverage, approximation ratio, and timing."""
    
    # Extract data for all methods
    methods = []
    coverage_values = []
    approx_ratio_values = []
    timing_values = []
    
    for method, data in results.items():
        # Get polygon coverage
        coverage_found = False
        for cov_key in ['polygon_coverage', 'polygon_coverage_stats', 'coverage_stats']:
            if cov_key in data and data[cov_key].get('mean') is not None:
                coverage_values.append(data[cov_key]['mean'])
                coverage_found = True
                break
        if not coverage_found:
            coverage_values.append(0)
        
        # Get approximation ratio
        ratio_found = False
        for ratio_key in ['approx_ratio', 'size_ratio_stats']:
            if ratio_key in data and data[ratio_key].get('mean') is not None:
                approx_ratio_values.append(data[ratio_key]['mean'])
                ratio_found = True
                break
        if not ratio_found:
            approx_ratio_values.append(0)
        
        # Get timing
        timing_val = data.get('avg_time_per_instance', 0)
        timing_values.append(timing_val if timing_val is not None else 0)
        
        methods.append(method)
    
    # Create the plot with wider figure to accommodate legend
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Create scatter plot with timing as point size
    # Normalize timing values for better visualization (larger = slower)
    max_timing = max(timing_values) if timing_values else 1
    normalized_sizes = [(t / max_timing) * 500 + 50 for t in timing_values]  # 50-550 point size range
    
    # Create scatter plot with different colors for each method
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    scatter = ax.scatter(coverage_values, approx_ratio_values, 
                       s=normalized_sizes, c=colors, alpha=0.7, 
                       edgecolors='black', linewidth=1)
    
    # Add legend with method names and colors
    legend_elements = [plt.scatter([], [], s=100, color=colors[i], alpha=0.7, 
                                 edgecolors='black', linewidth=1, label=method) 
                      for i, method in enumerate(methods)]
    ax.legend(legend_elements, methods, loc='center left', bbox_to_anchor=(1.02, 0.5), 
             title='Methods', title_fontsize=12, fontsize=10)
    
    # Customize the plot
    ax.set_xlabel('Polygon Coverage', fontsize=12, fontweight='bold')
    ax.set_ylabel('Approximation Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Frontier: Coverage vs Approximation Ratio\n(Point size = Average Time per Instance)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add a size legend for timing
    timing_legend_sizes = [min(timing_values), max(timing_values)] if timing_values else [0, 1]
    timing_legend_labels = [f'{t:.4f}s' for t in timing_legend_sizes]
    timing_legend_points = [(t / max_timing) * 500 + 50 for t in timing_legend_sizes]
    
    legend_elements = [plt.scatter([], [], s=size, c='gray', alpha=0.7, 
                                 edgecolors='black', linewidth=1) 
                      for size in timing_legend_points]
    legend1 = ax.legend(legend_elements, timing_legend_labels, 
                       title='Timing', loc='upper right', 
                       bbox_to_anchor=(1.0, 1.0))
    legend1.set_title('Timing (seconds)', prop={'weight': 'bold'})
    
    # Highlight potential Pareto-optimal points
    # (Higher coverage and lower approximation ratio are better)
    pareto_mask = []
    for i in range(len(methods)):
        is_pareto = True
        for j in range(len(methods)):
            if i != j:
                # Point j dominates point i if it has higher coverage AND lower/equal approx_ratio
                if (coverage_values[j] >= coverage_values[i] and 
                    approx_ratio_values[j] <= approx_ratio_values[i] and
                    (coverage_values[j] > coverage_values[i] or approx_ratio_values[j] < approx_ratio_values[i])):
                    is_pareto = False
                    break
        pareto_mask.append(is_pareto)
    
    # Draw lines connecting Pareto-optimal points
    pareto_points = [(coverage_values[i], approx_ratio_values[i]) 
                     for i in range(len(methods)) if pareto_mask[i]]
    if len(pareto_points) > 1:
        # Sort points by coverage for proper line connection
        pareto_points.sort(key=lambda x: x[0])
        pareto_x, pareto_y = zip(*pareto_points)
        ax.plot(pareto_x, pareto_y, 'r--', alpha=0.5, linewidth=2, 
                label='Potential Pareto Frontier')
        ax.legend(loc='lower left')
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make room for the legend
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'pareto_frontier.png'), dpi=300, bbox_inches='tight')
    print(f"Pareto frontier plot saved to {os.path.join(output_dir, 'pareto_frontier.png')}")
    plt.close()

def create_whisker_plots(results, output_dir="results/gfx/whisker"):
    """Create whisker/box plots for coverage, approximation ratio, and timing."""
    
    # Extract data for all methods
    methods = []
    coverage_data = []
    approx_ratio_data = []
    timing_data = []
    
    for method, data in results.items():
        methods.append(method)
        
        # Get polygon coverage stats
        coverage_found = False
        for cov_key in ['polygon_coverage', 'polygon_coverage_stats', 'coverage_stats']:
            if cov_key in data and data[cov_key].get('mean') is not None:
                stats = data[cov_key]
                # Generate synthetic data for box plot visualization
                mean_val = stats.get('mean', 0)
                std_val = stats.get('std', 0)
                if std_val > 0:
                    # Generate sample points using normal distribution
                    sample_data = np.random.normal(mean_val, std_val, 100)
                    # Clip to reasonable bounds (0 to 1 for coverage)
                    sample_data = np.clip(sample_data, 0, 1)
                else:
                    sample_data = np.full(100, mean_val)
                coverage_data.append(sample_data)
                coverage_found = True
                break
        if not coverage_found:
            coverage_data.append(np.zeros(100))
        
        # Get approximation ratio stats
        ratio_found = False
        for ratio_key in ['approx_ratio', 'size_ratio_stats']:
            if ratio_key in data and data[ratio_key].get('mean') is not None:
                stats = data[ratio_key]
                mean_val = stats.get('mean', 0)
                std_val = stats.get('std', 0)
                if std_val > 0:
                    sample_data = np.random.normal(mean_val, std_val, 100)
                    # Clip to reasonable bounds (0 to max observed)
                    sample_data = np.clip(sample_data, 0, None)
                else:
                    sample_data = np.full(100, mean_val)
                approx_ratio_data.append(sample_data)
                ratio_found = True
                break
        if not ratio_found:
            approx_ratio_data.append(np.zeros(100))
        
        # Get timing (single values)
        timing_val = data.get('avg_time_per_instance', 0)
        timing_data.append([timing_val if timing_val is not None else 0] * 100)
    
    # Create three separate whisker plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Distribution Analysis: Key Performance Metrics', fontsize=16, fontweight='bold')
    
    # Coverage whisker plot
    bp1 = axes[0].boxplot(coverage_data, labels=methods, patch_artist=True)
    axes[0].set_title('Polygon Coverage Distribution', fontweight='bold')
    axes[0].set_ylabel('Coverage Ratio')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Color boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
    for patch, color in zip(bp1['boxes'], colors[:len(bp1['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Approximation ratio whisker plot
    bp2 = axes[1].boxplot(approx_ratio_data, labels=methods, patch_artist=True)
    axes[1].set_title('Approximation Ratio Distribution', fontweight='bold')
    axes[1].set_ylabel('Approx Ratio')
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', rotation=45)
    
    for patch, color in zip(bp2['boxes'], colors[:len(bp2['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Timing plot (bar chart since timing is single value per method)
    timing_means = [np.mean(t) for t in timing_data]
    bars = axes[2].bar(methods, timing_means, color=colors[:len(methods)], alpha=0.7)
    axes[2].set_title('Average Time per Instance', fontweight='bold')
    axes[2].set_ylabel('Time (seconds)')
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(axis='x', rotation=45)
    
    # Add value labels on timing bars
    for bar, mean_val in zip(bars, timing_means):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean_val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'whisker_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Whisker plots saved to {os.path.join(output_dir, 'whisker_comparison.png')}")
    plt.close()

def create_spider_plot(results, output_dir="results/gfx/spider"):
    """Create spider/radar chart comparing all methods across three key metrics."""
    
    # Extract data for all methods
    methods = []
    coverage_values = []
    approx_ratio_values = []
    timing_values = []
    
    for method, data in results.items():
        methods.append(method)
        
        # Get polygon coverage
        coverage_found = False
        for cov_key in ['polygon_coverage', 'polygon_coverage_stats', 'coverage_stats']:
            if cov_key in data and data[cov_key].get('mean') is not None:
                coverage_values.append(data[cov_key]['mean'])
                coverage_found = True
                break
        if not coverage_found:
            coverage_values.append(0)
        
        # Get approximation ratio (closer to 1 is better, so we use proximity to 1)
        ratio_found = False
        for ratio_key in ['approx_ratio', 'size_ratio_stats']:
            if ratio_key in data and data[ratio_key].get('mean') is not None:
                ratio_val = data[ratio_key]['mean']
                # Transform so values closer to 1 are better (higher score)
                # Use 1 / (1 + |ratio - 1|) so perfect ratio of 1.0 gives score of 1.0
                approx_ratio_values.append(1.0 / (1.0 + abs(ratio_val - 1.0)) if ratio_val > 0 else 0)
                ratio_found = True
                break
        if not ratio_found:
            approx_ratio_values.append(0)
        
        # Get timing (lower time is better, so we use reciprocal)
        timing_val = data.get('avg_time_per_instance', 0.001)
        if timing_val is None or timing_val <= 0:
            timing_val = 0.001  # Avoid division by zero
        # Reciprocal of time (higher = faster/better)
        timing_values.append(1.0 / timing_val)
    
    # Normalize all values to 0-1 scale for fair comparison
    if coverage_values:
        max_cov = max(coverage_values)
        coverage_values = [c/max_cov if max_cov > 0 else 0 for c in coverage_values]
    
    if approx_ratio_values:
        max_ratio = max(approx_ratio_values)
        approx_ratio_values = [r/max_ratio if max_ratio > 0 else 0 for r in approx_ratio_values]
    
    if timing_values:
        max_time = max(timing_values)
        timing_values = [t/max_time if max_time > 0 else 0 for t in timing_values]
    
    # Set up the radar chart
    categories = ['Coverage\n(Higher = Better)', 'Approximation\n(Closer to 1 = Better)', 'Speed\n(Higher = Faster)']
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Calculate angle for each category
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
    
    # Plot each method
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    for i, method in enumerate(methods):
        values = [coverage_values[i], approx_ratio_values[i], timing_values[i]]
        values += [values[0]]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True)
    
    plt.title('Performance Radar Chart\n(Coverage: Higher Better | Approximation: Closer to 1 Better | Speed: Higher Better)', 
              size=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'spider_comparison.png'), dpi=300, bbox_inches='tight')
    print(f"Spider chart saved to {os.path.join(output_dir, 'spider_comparison.png')}")
    plt.close()

def create_comparison_charts(results, output_dir="results/gfx"):
    """Create separate bar charts for each key metric."""
    
    # Extract data for plotting
    methods = []
    coverage_means = []
    coverage_stds = []
    approx_ratio_means = []
    approx_ratio_stds = []
    timing_means = []
    
    for method, data in results.items():
        methods.append(method)
        
        # Polygon Coverage - handle multiple formats
        coverage_found = False
        for cov_key in ['polygon_coverage', 'polygon_coverage_stats', 'coverage_stats']:
            if cov_key in data and data[cov_key].get('mean') is not None:
                coverage_means.append(data[cov_key]['mean'])
                coverage_stds.append(data[cov_key].get('std') or 0)
                coverage_found = True
                break
        if not coverage_found:
            coverage_means.append(0)
            coverage_stds.append(0)
        
        # Approximation Ratio - handle both formats
        ratio_found = False
        for ratio_key in ['approx_ratio', 'size_ratio_stats']:
            if ratio_key in data and data[ratio_key].get('mean') is not None:
                approx_ratio_means.append(data[ratio_key]['mean'])
                approx_ratio_stds.append(data[ratio_key].get('std') or 0)
                ratio_found = True
                break
        if not ratio_found:
            approx_ratio_means.append(0)
            approx_ratio_stds.append(0)
        
        # Timing
        timing_val = data.get('avg_time_per_instance', 0)
        timing_means.append(timing_val if timing_val is not None else 0)
    
    x = np.arange(len(methods))
    width = 0.6
    
    # Create separate plot for Coverage
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    bars1 = ax1.bar(x, coverage_means, width, yerr=coverage_stds, 
                   capsize=5, alpha=0.8, color='skyblue', edgecolor='navy')
    ax1.set_title('Polygon Coverage Comparison', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Coverage Ratio', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars1, coverage_means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'polygon_coverage_comparison.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Polygon coverage plot saved to {os.path.join(output_dir, 'polygon_coverage_comparison.png')}")
    plt.close()
    
    # Create separate plot for Approximation Ratio
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    bars2 = ax2.bar(x, approx_ratio_means, width, yerr=approx_ratio_stds,
                   capsize=5, alpha=0.8, color='lightcoral', edgecolor='darkred')
    ax2.set_title('Approximation Ratio Comparison', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Approx Ratio', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars2, approx_ratio_means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'approx_ratio_comparison.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Approximation ratio plot saved to {os.path.join(output_dir, 'approx_ratio_comparison.png')}")
    plt.close()
    
    # Create separate plot for Timing
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))
    bars3 = ax3.bar(x, timing_means, width, alpha=0.8, color='lightgreen', edgecolor='darkgreen')
    ax3.set_title('Average Time per Instance Comparison', fontweight='bold', fontsize=14)
    ax3.set_ylabel('Time (seconds)', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars3, timing_means):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timing_comparison.png'), 
                dpi=300, bbox_inches='tight')
    print(f"Timing plot saved to {os.path.join(output_dir, 'timing_comparison.png')}")
    plt.close()

def print_detailed_analysis(results):
    """Print detailed analysis and insights."""
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Best performer analysis
    def safe_get_mean(data, keys):
        """Safely get mean value from multiple possible keys, returning 0 for None values."""
        for key in keys:
            if key in data:
                stats = data[key]
                mean_val = stats.get('mean', 0)
                return mean_val if mean_val is not None else 0
        return 0
    
    # Find best performers using multiple possible key formats (prioritize polygon_coverage)
    best_coverage = max(results.items(), key=lambda x: safe_get_mean(x[1], ['polygon_coverage', 'polygon_coverage_stats', 'coverage_stats']))
    best_efficiency = max(results.items(), key=lambda x: safe_get_mean(x[1], ['efficiency_stats']))
    best_size_ratio = min(results.items(), key=lambda x: safe_get_mean(x[1], ['size_ratio_stats']) if safe_get_mean(x[1], ['size_ratio_stats']) > 0 else float('inf'))
    best_approx_ratio = max(results.items(), key=lambda x: safe_get_mean(x[1], ['approx_ratio']))
    fastest_method = min(results.items(), key=lambda x: x[1].get('avg_time_per_instance', float('inf')))
    
    print(f"\n🏆 BEST PERFORMERS:")
    print(f"  • Highest Coverage: {best_coverage[0]} ({safe_format(safe_get_mean(best_coverage[1], ['polygon_coverage', 'polygon_coverage_stats', 'coverage_stats']))})")
    print(f"  • Highest Efficiency: {best_efficiency[0]} ({safe_format(safe_get_mean(best_efficiency[1], ['efficiency_stats']))})")
    print(f"  • Best Size Ratio: {best_size_ratio[0]} ({safe_format(safe_get_mean(best_size_ratio[1], ['size_ratio_stats']))})")
    print(f"  • Best Approx Ratio: {best_approx_ratio[0]} ({safe_format(safe_get_mean(best_approx_ratio[1], ['approx_ratio']))})")
    print(f"  • Fastest Method: {fastest_method[0]} ({safe_format(fastest_method[1].get('avg_time_per_instance'))}s per instance)")
    
    print(f"\n📊 METHOD COMPARISON:")
    for method, data in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Samples: {data.get('num_samples', data.get('num_val_samples', data.get('num_instances', 'N/A')))}")
        
        # Handle multiple coverage key formats - prioritize polygon_coverage
        coverage_key = None
        for key in ['polygon_coverage', 'polygon_coverage_stats', 'coverage_stats']:
            if key in data:
                coverage_key = key
                break
        
        if coverage_key:
            cov = data[coverage_key]
            print(f"  Coverage: {safe_format(cov.get('mean'))} ± {safe_format(cov.get('std'))} (range: {safe_format(cov.get('min'))}-{safe_format(cov.get('max'))})")
        
        if 'efficiency_stats' in data:
            eff = data['efficiency_stats']
            print(f"  Efficiency: {safe_format(eff.get('mean'))} ± {safe_format(eff.get('std'))} (range: {safe_format(eff.get('min'))}-{safe_format(eff.get('max'))})")
        
        if 'size_ratio_stats' in data:
            ratio = data['size_ratio_stats']
            print(f"  Size Ratio: {safe_format(ratio.get('mean'))} ± {safe_format(ratio.get('std'))} (range: {safe_format(ratio.get('min'))}-{safe_format(ratio.get('max'))})")
        
        if 'approx_ratio' in data:
            approx = data['approx_ratio']
            print(f"  Approx Ratio: {safe_format(approx.get('mean'))} ± {safe_format(approx.get('std'))} (range: {safe_format(approx.get('min'))}-{safe_format(approx.get('max'))})")
        
        if 'avg_time_per_instance' in data:
            print(f"  Avg Time/Instance: {safe_format(data['avg_time_per_instance'])}s")
        
        if 'total_time' in data:
            print(f"  Total Time: {safe_format(data['total_time'])}s")
    
    print(f"\n💡 INSIGHTS:")
    print(f"  • Lower size ratios indicate more efficient solutions")
    print(f"  • Higher coverage ratios indicate better solution quality") 
    print(f"  • Higher efficiency ratios indicate fewer unnecessary guards")

def main():
    print("="*50)
    
    # Load results
    results = load_evaluation_results()
    
    if not results:
        print("❌ No evaluation results found. Please ensure the JSON files exist in the results/ directory.")
        return
    
    print(f"\n📋 Found results for {len(results)} methods")
    
    # Create and display summary table
    print("\n📊 SUMMARY TABLE:")
    summary_df = create_summary_table(results)
    print(summary_df.to_string(index=False))
    
    # Create and display performance comparison table
    print("\n🎯 PERFORMANCE COMPARISON:")
    comparison_df = create_performance_comparison_table(results)
    print(comparison_df.to_string(index=False))
    
    # Save tables to CSV
    os.makedirs("results/gfx", exist_ok=True)
    summary_df.to_csv("results/gfx/summary_table.csv", index=False)
    comparison_df.to_csv("results/gfx/performance_comparison.csv", index=False)
    print(f"\n💾 Tables saved to results/gfx/summary_table.csv and results/gfx/performance_comparison.csv")
    
    # Create visualizations
    print(f"\n📈 Generating visualizations...")
    try:
        create_comparison_charts(results)
        create_pareto_frontier_plot(results)
        create_whisker_plots(results)
        create_spider_plot(results)
        print(f"✅ Plots saved to results/gfx/ subdirectories (pareto/, whisker/, spider/)")
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        print("Note: Make sure matplotlib and seaborn are installed")
    
    # Detailed analysis
    print_detailed_analysis(results)
    
    print(f"\n🎉 Analysis complete!")

def print_detailed_analysis(results):
    """Print detailed analysis and insights."""
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Best performer analysis
    def safe_get_mean(data, keys):
        """Safely get mean value from multiple possible keys, returning 0 for None values."""
        for key in keys:
            if key in data:
                stats = data[key]
                mean_val = stats.get('mean', 0)
                return mean_val if mean_val is not None else 0
        return 0
    
    # Find best performers using multiple possible key formats (prioritize polygon_coverage)
    best_coverage = max(results.items(), key=lambda x: safe_get_mean(x[1], ['polygon_coverage', 'polygon_coverage_stats', 'coverage_stats']))
    best_efficiency = max(results.items(), key=lambda x: safe_get_mean(x[1], ['efficiency_stats']))
    best_size_ratio = min(results.items(), key=lambda x: safe_get_mean(x[1], ['size_ratio_stats']) if safe_get_mean(x[1], ['size_ratio_stats']) > 0 else float('inf'))
    best_approx_ratio = max(results.items(), key=lambda x: safe_get_mean(x[1], ['approx_ratio']))
    fastest_method = min(results.items(), key=lambda x: x[1].get('avg_time_per_instance', float('inf')))
    
    print(f"\n🏆 BEST PERFORMERS:")
    print(f"  • Highest Coverage: {best_coverage[0]} ({safe_format(safe_get_mean(best_coverage[1], ['polygon_coverage', 'polygon_coverage_stats', 'coverage_stats']))})")
    print(f"  • Highest Efficiency: {best_efficiency[0]} ({safe_format(safe_get_mean(best_efficiency[1], ['efficiency_stats']))})")
    print(f"  • Best Size Ratio: {best_size_ratio[0]} ({safe_format(safe_get_mean(best_size_ratio[1], ['size_ratio_stats']))})")
    print(f"  • Best Approx Ratio: {best_approx_ratio[0]} ({safe_format(safe_get_mean(best_approx_ratio[1], ['approx_ratio']))})")
    print(f"  • Fastest Method: {fastest_method[0]} ({safe_format(fastest_method[1].get('avg_time_per_instance'))}s per instance)")
    
    print(f"\n📊 METHOD COMPARISON:")
    for method, data in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Samples: {data.get('num_samples', data.get('num_val_samples', data.get('num_instances', 'N/A')))}")
        
        # Handle multiple coverage key formats - prioritize polygon_coverage
        coverage_key = None
        for key in ['polygon_coverage', 'polygon_coverage_stats', 'coverage_stats']:
            if key in data:
                coverage_key = key
                break
        
        if coverage_key:
            cov = data[coverage_key]
            print(f"  Coverage: {safe_format(cov.get('mean'))} ± {safe_format(cov.get('std'))} (range: {safe_format(cov.get('min'))}-{safe_format(cov.get('max'))})")
        
        if 'efficiency_stats' in data:
            eff = data['efficiency_stats']
            print(f"  Efficiency: {safe_format(eff.get('mean'))} ± {safe_format(eff.get('std'))} (range: {safe_format(eff.get('min'))}-{safe_format(eff.get('max'))})")
        
        if 'size_ratio_stats' in data:
            ratio = data['size_ratio_stats']
            print(f"  Size Ratio: {safe_format(ratio.get('mean'))} ± {safe_format(ratio.get('std'))} (range: {safe_format(ratio.get('min'))}-{safe_format(ratio.get('max'))})")
        
        if 'approx_ratio' in data:
            approx = data['approx_ratio']
            print(f"  Approx Ratio: {safe_format(approx.get('mean'))} ± {safe_format(approx.get('std'))} (range: {safe_format(approx.get('min'))}-{safe_format(approx.get('max'))})")
        
        if 'avg_time_per_instance' in data:
            print(f"  Avg Time/Instance: {safe_format(data['avg_time_per_instance'])}s")
        
        if 'total_time' in data:
            print(f"  Total Time: {safe_format(data['total_time'])}s")
    
    print(f"\n💡 INSIGHTS:")
    print(f"  • Lower size ratios indicate more efficient solutions")
    print(f"  • Higher coverage ratios indicate better solution quality") 
    print(f"  • Higher efficiency ratios indicate fewer unnecessary guards")

def main():
    """Main function to run the complete analysis."""
    print("🔬 AGNet Results Analysis")
    print("="*50)
    
    # Load results
    results = load_evaluation_results()
    
    if not results:
        print("❌ No evaluation results found. Please ensure the JSON files exist in the results/ directory.")
        return
    
    print(f"\n📋 Found results for {len(results)} methods")
    
    # Create and display summary table
    print("\n📊 SUMMARY TABLE:")
    summary_df = create_summary_table(results)
    print(summary_df.to_string(index=False))
    
    # Create and display performance comparison table
    print("\n🎯 PERFORMANCE COMPARISON:")
    comparison_df = create_performance_comparison_table(results)
    print(comparison_df.to_string(index=False))
    
    # Save tables to CSV
    os.makedirs("results/gfx", exist_ok=True)
    summary_df.to_csv("results/gfx/summary_table.csv", index=False)
    comparison_df.to_csv("results/gfx/performance_comparison.csv", index=False)
    print(f"\n💾 Tables saved to results/gfx/summary_table.csv and results/gfx/performance_comparison.csv")
    
    # Create visualizations
    print(f"\n📈 Generating visualizations...")
    try:
        create_comparison_charts(results)
        create_pareto_frontier_plot(results)
        create_whisker_plots(results)
        create_spider_plot(results)
        print(f"✅ Plots saved to results/gfx/ subdirectories (pareto/, whisker/, spider/)")
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        print("Note: Make sure matplotlib and seaborn are installed")
    
    # Detailed analysis
    print_detailed_analysis(results)
    
    print(f"\n🎉 Analysis complete!")

if __name__ == "__main__":
    main()
