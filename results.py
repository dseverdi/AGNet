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
        "greedy_agp_evaluation_all.json": "Greedy",
        "sl_agp_evaluation.json": "Supervised Learning", 
        "qlearning_evaluation.json": "Q-learning Reinforcement Learning",
        "active_search_all_K=5.json": "Active Search K=5",
        "sampling_evaluation_all_K=1.json": "Solution Sampling K=1  RL",
        "sampling_evaluation_all_K=5.json": "Solution Sampllig K=5 RL",
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
    """Create a summary table comparing all methods."""
    summary_data = []
    
    for method, data in results.items():
        # Extract key statistics
        row = {
            'Method': method,
            'Samples': data.get('num_samples', data.get('num_val_samples', data.get('num_instances', 'N/A'))),
            'Training Method': data.get('training_method', 'N/A'),
        }
        
        # Coverage statistics (prioritize polygon_coverage over coverage_stats)
        coverage_key = None
        if 'polygon_coverage' in data:
            coverage_key = 'polygon_coverage'
        elif 'polygon_coverage_stats' in data:
            coverage_key = 'polygon_coverage_stats'
        elif 'coverage_stats' in data:
            coverage_key = 'coverage_stats'
            
        if coverage_key:
            cov = data[coverage_key]
            row.update({
                'Coverage Mean': safe_format(cov.get('mean')),
                'Coverage Std': safe_format(cov.get('std')),
                'Coverage Min': safe_format(cov.get('min')),
                'Coverage Max': safe_format(cov.get('max')),
            })
        
        # Efficiency statistics
        if 'efficiency_stats' in data:
            eff = data['efficiency_stats']
            row.update({
                'Efficiency Mean': safe_format(eff.get('mean')),
                'Efficiency Std': safe_format(eff.get('std')),
                'Efficiency Min': safe_format(eff.get('min')),
                'Efficiency Max': safe_format(eff.get('max')),
            })
        
        # Size ratio statistics
        if 'size_ratio_stats' in data:
            ratio = data['size_ratio_stats']
            row.update({
                'Size Ratio Mean': safe_format(ratio.get('mean')),
                'Size Ratio Std': safe_format(ratio.get('std')),
                'Size Ratio Min': safe_format(ratio.get('min')),
                'Size Ratio Max': safe_format(ratio.get('max')),
            })
        
        # Polygon coverage (for greedy) - handle both formats
        poly_cov_key = None
        if 'polygon_coverage' in data:
            poly_cov_key = 'polygon_coverage'
        elif 'polygon_coverage_stats' in data:
            poly_cov_key = 'polygon_coverage_stats'
            
        if poly_cov_key:
            poly_cov = data[poly_cov_key]
            row.update({
                'Polygon Coverage Mean': safe_format(poly_cov.get('mean')),
                'Polygon Coverage Std': safe_format(poly_cov.get('std')),
            })
        
        # Approximation ratio (for sampling methods)
        if 'approx_ratio' in data:
            approx = data['approx_ratio']
            row.update({
                'Approx Ratio Mean': safe_format(approx.get('mean')),
                'Approx Ratio Std': safe_format(approx.get('std')),
                'Approx Ratio Min': safe_format(approx.get('min')),
                'Approx Ratio Max': safe_format(approx.get('max')),
            })
        
        # Timing information
        if 'avg_time_per_instance' in data:
            row['Avg Time per Instance'] = safe_format(data['avg_time_per_instance'])
        if 'total_time' in data:
            row['Total Time'] = safe_format(data['total_time'])
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def create_performance_comparison_table(results):
    """Create a focused performance comparison table."""
    comparison_data = []
    
    for method, data in results.items():
        row = {'Method': method}
        
        # Extract key performance metrics - prioritize polygon_coverage
        coverage_key = None
        if 'polygon_coverage' in data:
            coverage_key = 'polygon_coverage'
        elif 'polygon_coverage_stats' in data:
            coverage_key = 'polygon_coverage_stats'
        elif 'coverage_stats' in data:
            coverage_key = 'coverage_stats'
            
        if coverage_key:
            cov = data[coverage_key]
            row['Coverage (μ±σ)'] = f"{safe_format(cov.get('mean'))} ± {safe_format(cov.get('std'))}"
            row['Coverage Median'] = safe_format(cov.get('median'))
        
        if 'efficiency_stats' in data:
            eff = data['efficiency_stats']
            row['Efficiency (μ±σ)'] = f"{safe_format(eff.get('mean'))} ± {safe_format(eff.get('std'))}"
            row['Efficiency Median'] = safe_format(eff.get('median'))
        
        if 'size_ratio_stats' in data:
            ratio = data['size_ratio_stats']
            row['Size Ratio (μ±σ)'] = f"{safe_format(ratio.get('mean'))} ± {safe_format(ratio.get('std'))}"
            row['Size Ratio Median'] = safe_format(ratio.get('median'))
        
        # Approximation ratio (for sampling methods)
        if 'approx_ratio' in data:
            approx = data['approx_ratio']
            row['Approx Ratio (μ±σ)'] = f"{safe_format(approx.get('mean'))} ± {safe_format(approx.get('std'))}"
            row['Approx Ratio Median'] = safe_format(approx.get('median'))
        
        # Timing information
        if 'avg_time_per_instance' in data:
            row['Avg Time/Instance (s)'] = safe_format(data['avg_time_per_instance'], ".3f")
        
        # Additional metrics
        row['Samples'] = data.get('num_samples', data.get('num_val_samples', data.get('num_instances', 'N/A')))
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def create_whisker_plots(results, output_dir="results"):
    """Create whisker plots for key metrics."""
    
    # Prepare data for plotting - prioritize polygon_coverage
    metrics = ['polygon_coverage', 'polygon_coverage_stats', 'coverage_stats', 'efficiency_stats', 'size_ratio_stats', 'approx_ratio']
    metric_names = ['Coverage Ratio', 'Coverage Ratio', 'Coverage Ratio', 'Efficiency Ratio', 'Size Ratio', 'Approx Ratio']
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Performance Comparison: Whisker Plots', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        
        # Collect data for each method
        plot_data = []
        methods = []
        
        for method, data in results.items():
            if metric in data and data[metric].get('mean') is not None:
                stats = data[metric]
                # Create synthetic data points based on statistics for box plot
                # Using quartiles and min/max to approximate distribution
                q25 = stats.get('q25') or stats.get('mean', 0)
                median = stats.get('median') or stats.get('mean', 0)
                q75 = stats.get('q75') or stats.get('mean', 0)
                minimum = stats.get('min') or stats.get('mean', 0)
                maximum = stats.get('max') or stats.get('mean', 0)
                
                # Generate sample points (approximation for visualization)
                n_points = 100
                # Use beta distribution to approximate the data distribution
                if stats['std'] > 0:
                    # Create points around quartiles
                    lower_half = np.random.uniform(minimum, median, n_points//2)
                    upper_half = np.random.uniform(median, maximum, n_points//2)
                    sample_data = np.concatenate([lower_half, upper_half])
                else:
                    # If std is 0, all points are the same
                    sample_data = np.full(n_points, median)
                
                plot_data.append(sample_data)
                methods.append(method)
        
        # Create box plot
        if plot_data:
            bp = ax.boxplot(plot_data, labels=methods, patch_artist=True)
            
            # Customize box plot
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_title(f'{metric_name} Distribution', fontweight='bold')
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    
    # Remove unused subplots if needed
    total_subplots = 6
    used_subplots = len([m for m in metrics if any(m in data for data in results.values())])
    
    for i in range(used_subplots, total_subplots):
        if i < total_subplots:
            fig.delaxes(axes.flatten()[i])
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'performance_whisker_plots.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

def create_comparison_charts(results, output_dir="results"):
    """Create comparison bar charts for key metrics."""
    
    # Extract data for plotting
    methods = []
    coverage_means = []
    coverage_stds = []
    efficiency_means = []
    efficiency_stds = []
    size_ratio_means = []
    size_ratio_stds = []
    approx_ratio_means = []
    approx_ratio_stds = []
    timing_means = []
    
    for method, data in results.items():
        methods.append(method)
        
        # Coverage - prioritize polygon_coverage
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
        
        # Efficiency
        if 'efficiency_stats' in data and data['efficiency_stats'].get('mean') is not None:
            efficiency_means.append(data['efficiency_stats']['mean'])
            efficiency_stds.append(data['efficiency_stats']['std'] or 0)
        else:
            efficiency_means.append(0)
            efficiency_stds.append(0)
        
        # Size Ratio
        if 'size_ratio_stats' in data and data['size_ratio_stats'].get('mean') is not None:
            size_ratio_means.append(data['size_ratio_stats']['mean'])
            size_ratio_stds.append(data['size_ratio_stats']['std'] or 0)
        else:
            size_ratio_means.append(0)
            size_ratio_stds.append(0)
        
        # Approximation Ratio
        if 'approx_ratio' in data and data['approx_ratio'].get('mean') is not None:
            approx_ratio_means.append(data['approx_ratio']['mean'])
            approx_ratio_stds.append(data['approx_ratio']['std'] or 0)
        else:
            approx_ratio_means.append(0)
            approx_ratio_stds.append(0)
        
        # Timing
        timing_val = data.get('avg_time_per_instance', 0)
        timing_means.append(timing_val if timing_val is not None else 0)
    
    # Create bar charts - updated layout for more metrics
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold')
    
    x = np.arange(len(methods))
    width = 0.6
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Coverage chart
    bars1 = axes[0].bar(x, coverage_means, width, yerr=coverage_stds, 
                       capsize=5, alpha=0.8, color='skyblue', edgecolor='navy')
    axes[0].set_title('Coverage Ratio', fontweight='bold')
    axes[0].set_ylabel('Coverage Ratio')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars1, coverage_means):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.3f}', ha='center', va='bottom')
    
    # Efficiency chart
    bars2 = axes[1].bar(x, efficiency_means, width, yerr=efficiency_stds,
                       capsize=5, alpha=0.8, color='lightgreen', edgecolor='darkgreen')
    axes[1].set_title('Efficiency Ratio', fontweight='bold')
    axes[1].set_ylabel('Efficiency Ratio')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars2, efficiency_means):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.3f}', ha='center', va='bottom')
    
    # Size Ratio chart
    bars3 = axes[2].bar(x, size_ratio_means, width, yerr=size_ratio_stds,
                       capsize=5, alpha=0.8, color='lightcoral', edgecolor='darkred')
    axes[2].set_title('Size Ratio (Lower is Better)', fontweight='bold')
    axes[2].set_ylabel('Size Ratio')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(methods, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars3, size_ratio_means):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.3f}', ha='center', va='bottom')
    
    # Approximation Ratio chart
    bars4 = axes[3].bar(x, approx_ratio_means, width, yerr=approx_ratio_stds,
                       capsize=5, alpha=0.8, color='gold', edgecolor='orange')
    axes[3].set_title('Approximation Ratio', fontweight='bold')
    axes[3].set_ylabel('Approx Ratio')
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(methods, rotation=45, ha='right')
    axes[3].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars4, approx_ratio_means):
        height = bar.get_height()
        axes[3].text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.3f}', ha='center', va='bottom')
    
    # Timing chart
    bars5 = axes[4].bar(x, timing_means, width, alpha=0.8, color='purple', edgecolor='indigo')
    axes[4].set_title('Avg Time per Instance (s)', fontweight='bold')
    axes[4].set_ylabel('Time (seconds)')
    axes[4].set_xticks(x)
    axes[4].set_xticklabels(methods, rotation=45, ha='right')
    axes[4].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars5, timing_means):
        height = bar.get_height()
        axes[4].text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.3f}', ha='center', va='bottom')
    
    # Remove unused subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'performance_comparison_bars.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()

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
    os.makedirs("results", exist_ok=True)
    summary_df.to_csv("results/summary_table.csv", index=False)
    comparison_df.to_csv("results/performance_comparison.csv", index=False)
    print(f"\n💾 Tables saved to results/summary_table.csv and results/performance_comparison.csv")
    
    # Create visualizations
    print(f"\n📈 Generating visualizations...")
    try:
        create_comparison_charts(results)
        create_whisker_plots(results)
        print(f"✅ Plots saved to results/ directory")
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
    os.makedirs("results", exist_ok=True)
    summary_df.to_csv("results/summary_table.csv", index=False)
    comparison_df.to_csv("results/performance_comparison.csv", index=False)
    print(f"\n💾 Tables saved to results/summary_table.csv and results/performance_comparison.csv")
    
    # Create visualizations
    print(f"\n📈 Generating visualizations...")
    try:
        create_comparison_charts(results)
        create_whisker_plots(results)
        print(f"✅ Plots saved to results/ directory")
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        print("Note: Make sure matplotlib and seaborn are installed")
    
    # Detailed analysis
    print_detailed_analysis(results)
    
    print(f"\n🎉 Analysis complete!")

if __name__ == "__main__":
    main()
