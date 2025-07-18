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

def load_evaluation_results(results_dir="results"):
    """Load evaluation results from JSON files."""
    results = {}
    
    # Define the expected files and their method names
    file_mappings = {
        "greedy_agp_evaluation.json": "Greedy",
        "sl_agp_evaluation.json": "Supervised Learning", 
        "rl_agp_evaluation.json": "Reinforcement Learning"
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
            'Samples': data.get('num_samples', data.get('num_val_samples', 'N/A')),
            'Training Method': data.get('training_method', 'N/A'),
        }
        
        # Coverage statistics
        if 'coverage_stats' in data:
            cov = data['coverage_stats']
            row.update({
                'Coverage Mean': f"{cov['mean']:.3f}",
                'Coverage Std': f"{cov['std']:.3f}",
                'Coverage Min': f"{cov['min']:.3f}",
                'Coverage Max': f"{cov['max']:.3f}",
            })
        
        # Efficiency statistics
        if 'efficiency_stats' in data:
            eff = data['efficiency_stats']
            row.update({
                'Efficiency Mean': f"{eff['mean']:.3f}",
                'Efficiency Std': f"{eff['std']:.3f}",
                'Efficiency Min': f"{eff['min']:.3f}",
                'Efficiency Max': f"{eff['max']:.3f}",
            })
        
        # Size ratio statistics
        if 'size_ratio_stats' in data:
            ratio = data['size_ratio_stats']
            row.update({
                'Size Ratio Mean': f"{ratio['mean']:.3f}",
                'Size Ratio Std': f"{ratio['std']:.3f}",
                'Size Ratio Min': f"{ratio['min']:.3f}",
                'Size Ratio Max': f"{ratio['max']:.3f}",
            })
        
        # Polygon coverage (for greedy)
        if 'polygon_coverage_stats' in data:
            poly_cov = data['polygon_coverage_stats']
            row.update({
                'Polygon Coverage Mean': f"{poly_cov['mean']:.3f}",
                'Polygon Coverage Std': f"{poly_cov['std']:.3f}",
            })
        
        summary_data.append(row)
    
    return pd.DataFrame(summary_data)

def create_performance_comparison_table(results):
    """Create a focused performance comparison table."""
    comparison_data = []
    
    for method, data in results.items():
        row = {'Method': method}
        
        # Extract key performance metrics
        if 'coverage_stats' in data:
            row['Coverage (μ±σ)'] = f"{data['coverage_stats']['mean']:.3f} ± {data['coverage_stats']['std']:.3f}"
            row['Coverage Median'] = f"{data['coverage_stats']['median']:.3f}"
        
        if 'efficiency_stats' in data:
            row['Efficiency (μ±σ)'] = f"{data['efficiency_stats']['mean']:.3f} ± {data['efficiency_stats']['std']:.3f}"
            row['Efficiency Median'] = f"{data['efficiency_stats']['median']:.3f}"
        
        if 'size_ratio_stats' in data:
            row['Size Ratio (μ±σ)'] = f"{data['size_ratio_stats']['mean']:.3f} ± {data['size_ratio_stats']['std']:.3f}"
            row['Size Ratio Median'] = f"{data['size_ratio_stats']['median']:.3f}"
        
        # Additional metrics
        row['Samples'] = data.get('num_samples', data.get('num_val_samples', 'N/A'))
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def create_whisker_plots(results, output_dir="results"):
    """Create whisker plots for key metrics."""
    
    # Prepare data for plotting
    metrics = ['coverage_stats', 'efficiency_stats', 'size_ratio_stats']
    metric_names = ['Coverage Ratio', 'Efficiency Ratio', 'Size Ratio']
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Performance Comparison: Whisker Plots', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx]
        
        # Collect data for each method
        plot_data = []
        methods = []
        
        for method, data in results.items():
            if metric in data:
                stats = data[metric]
                # Create synthetic data points based on statistics for box plot
                # Using quartiles and min/max to approximate distribution
                q25, median, q75 = stats['q25'], stats['median'], stats['q75']
                minimum, maximum = stats['min'], stats['max']
                
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
    
    # Remove the fourth subplot if not needed
    if len(metrics) < 4:
        fig.delaxes(axes[3])
    
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
    
    for method, data in results.items():
        methods.append(method)
        
        # Coverage
        if 'coverage_stats' in data:
            coverage_means.append(data['coverage_stats']['mean'])
            coverage_stds.append(data['coverage_stats']['std'])
        else:
            coverage_means.append(0)
            coverage_stds.append(0)
        
        # Efficiency
        if 'efficiency_stats' in data:
            efficiency_means.append(data['efficiency_stats']['mean'])
            efficiency_stds.append(data['efficiency_stats']['std'])
        else:
            efficiency_means.append(0)
            efficiency_stds.append(0)
        
        # Size Ratio
        if 'size_ratio_stats' in data:
            size_ratio_means.append(data['size_ratio_stats']['mean'])
            size_ratio_stds.append(data['size_ratio_stats']['std'])
        else:
            size_ratio_means.append(0)
            size_ratio_stds.append(0)
    
    # Create bar charts
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold')
    
    x = np.arange(len(methods))
    width = 0.6
    
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
    best_coverage = max(results.items(), key=lambda x: x[1].get('coverage_stats', {}).get('mean', 0))
    best_efficiency = max(results.items(), key=lambda x: x[1].get('efficiency_stats', {}).get('mean', 0))
    best_size_ratio = min(results.items(), key=lambda x: x[1].get('size_ratio_stats', {}).get('mean', float('inf')))
    
    print(f"\n🏆 BEST PERFORMERS:")
    print(f"  • Highest Coverage: {best_coverage[0]} ({best_coverage[1].get('coverage_stats', {}).get('mean', 0):.3f})")
    print(f"  • Highest Efficiency: {best_efficiency[0]} ({best_efficiency[1].get('efficiency_stats', {}).get('mean', 0):.3f})")
    print(f"  • Best Size Ratio: {best_size_ratio[0]} ({best_size_ratio[1].get('size_ratio_stats', {}).get('mean', 0):.3f})")
    
    print(f"\n📊 METHOD COMPARISON:")
    for method, data in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Samples: {data.get('num_samples', data.get('num_val_samples', 'N/A'))}")
        
        if 'coverage_stats' in data:
            cov = data['coverage_stats']
            print(f"  Coverage: {cov['mean']:.3f} ± {cov['std']:.3f} (range: {cov['min']:.3f}-{cov['max']:.3f})")
        
        if 'efficiency_stats' in data:
            eff = data['efficiency_stats']
            print(f"  Efficiency: {eff['mean']:.3f} ± {eff['std']:.3f} (range: {eff['min']:.3f}-{eff['max']:.3f})")
        
        if 'size_ratio_stats' in data:
            ratio = data['size_ratio_stats']
            print(f"  Size Ratio: {ratio['mean']:.3f} ± {ratio['std']:.3f} (range: {ratio['min']:.3f}-{ratio['max']:.3f})")
    
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
