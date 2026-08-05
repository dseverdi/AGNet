#!/usr/bin/env python3
import json
import os
import sys
import matplotlib.pyplot as plt


def main(json_path: str, out_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle both old and new JSON formats
    if 'value_performance' in data:
        perf = data['value_performance']
        metrics = perf.get('overall_metrics', {})
        pred_stats = perf.get('prediction_stats', {})
        tgt_stats = perf.get('target_stats', {})
        
        mae = float(metrics.get('mae', 0.0))
        rmse = float(metrics.get('rmse', 0.0))
        r2 = float(metrics.get('r2_score', 0.0))
        corr = float(metrics.get('correlation', 0.0))
        mean_pred = float(pred_stats.get('mean', 0.0))
        mean_tgt = float(tgt_stats.get('mean', 0.0))
        std_pred = float(pred_stats.get('std', 0.0))
        std_tgt = float(tgt_stats.get('std', 0.0))
    else:
        # Legacy format
        metrics = data.get('overall_metrics', {})
        mae = float(metrics.get('mae', 0.0))
        rmse = float(metrics.get('rmse', 0.0))
        r2 = float(metrics.get('r2_score', 0.0))
        corr = float(metrics.get('correlation', 0.0))
        mean_pred = float(metrics.get('mean_prediction', 0.0))
        mean_tgt = float(metrics.get('mean_target', 0.0))
        std_pred = float(metrics.get('std_prediction', 0.0))
        std_tgt = float(metrics.get('std_target', 0.0))

    # Build figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    # Panel A: Quality metrics (scale-independent)
    ax = axes[0]
    # Calculate relative error metrics
    rel_mae = mae / mean_tgt * 100 if mean_tgt > 0 else 0
    rel_rmse = rmse / mean_tgt * 100 if mean_tgt > 0 else 0
    
    ax.bar(['R² Score', 'Correlation', 'Rel. MAE (%)', 'Rel. RMSE (%)'], 
           [r2, corr, rel_mae, rel_rmse], 
           color=['#577590', '#f3722c', '#43aa8b', '#277da1'])
    ax.set_title('Proxy quality metrics')
    ax.set_ylabel('Value')
    ax.set_ylim(0, max(1.0, r2, corr, rel_mae, rel_rmse) * 1.1)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    # Annotate bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.3f}",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9, xytext=(0, 3), textcoords='offset points')

    # Panel B: Distribution comparison
    ax = axes[1]
    labels = ['Prediction', 'Target']
    means = [mean_pred, mean_tgt]
    stds = [std_pred, std_tgt]
    colors = ['#43aa8b', '#277da1']
    ax.bar(labels, means, yerr=stds, capsize=6, color=colors)
    ax.set_title('Value distribution')
    ax.set_ylabel('Reward value (with std)')
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    # Add text with absolute errors
    ax.text(0.5, 0.05, f"MAE: {mae:.1f}  |  RMSE: {rmse:.1f}",
            transform=ax.transAxes, ha='center', va='bottom', fontsize=9)

    fig.suptitle('Value proxy (value_net) fidelity summary', fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved figure to {out_path}")


if __name__ == '__main__':
    # Usage: python tools/plot_value_net_proxy.py results/value_net_evaluation_all.json paper/gfx/value_net_proxy_metrics.png
    if len(sys.argv) < 3:
        print("Usage: plot_value_net_proxy.py <json_path> <out_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
