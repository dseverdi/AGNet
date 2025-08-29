#!/usr/bin/env python3
"""
Evaluation script for RL AGP model with active search.
Loads a trained RL model and evaluates it on validation instances using active search:
- For each instance, sample K solutions
- Pick the one with the best (smallest) reward
"""

import os
import copy
import argparse
import torch
import numpy as np
import json
from dotenv import load_dotenv
from torch.utils.data import DataLoader

# Import from project files
from dataset import Dataset, agp_read_samples, collate_fn
from models import create_actor
from rewards import enhanced_penalty as reward_fn
from utils import evaluate_polygon_visibility_numpy_wo_gt
from rl_agp import BucketBatchSampler, get_lengths_from_dataset
from train_value_net import SolutionValueNet, solution_collate_fn
from torch.nn.utils.rnn import pad_sequence
from functools import partial


def find_best_value_net_checkpoint(checkpoints_dir='checkpoints'):
    """
    Automatically find the best value network checkpoint in the checkpoints directory.
    Looks for files matching pattern: value_net_*_best.pt
    """
    import glob
    pattern = os.path.join(checkpoints_dir, 'value_net_*_best.pt')
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def load_value_net(value_net_path, embedding_size=128, hidden_size=256):
    """Load the trained value network."""
    print(f"Loading value network from: {value_net_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SolutionValueNet(embedding_size=embedding_size, hidden_size=hidden_size)
    model.load_state_dict(torch.load(value_net_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"Value network loaded on device: {device}")
    return model, device


def fast_active_search_with_value_net(actor, value_net, data_tensor, mask, length, name, K=10, device='cpu'):
    """
    Active search using learned value network for fast evaluation.
    
    Args:
        actor: Trained RL actor model
        value_net: Trained value network
        data_tensor: Instance data [1, max_len, 2]
        mask: Padding mask [1, max_len]
        length: Actual length of the instance
        name: Instance name
        K: Number of solutions to sample
        device: Device to run on
    
    Returns:
        best_solution: List of guard indices for best solution
        best_pred_reward: Predicted reward of best solution
        best_actual_reward: Actual reward of best solution (for validation)
        all_pred_rewards: Array of all K predicted rewards
        prediction_error: Absolute error between predicted and actual best reward
    """
    actor.eval()
    value_net.eval()
    
    data_tensor = data_tensor.to(device)
    mask = mask.to(device)
    length_tensor = torch.tensor([length], dtype=torch.long, device=device)
    
    solutions = []
    
    with torch.no_grad():
        # Sample K solutions from actor
        for k in range(K):
            selected_idxs, _ = actor(data_tensor, padding_mask=mask, lengths=length_tensor)
            solution = selected_idxs[0].cpu().numpy()  # Get first (and only) batch item
            # Filter to valid indices
            solution = [idx for idx in solution if idx < length]
            solutions.append(solution)
        
        # Batch predict rewards using value network (FAST!)
        if len(solutions) > 0:
            # Prepare batch data for value network
            batch_polygons = data_tensor.repeat(len(solutions), 1, 1)  # [K, max_len, 2]
            batch_solutions = pad_sequence([torch.tensor(sol, dtype=torch.long) for sol in solutions], 
                                         batch_first=True, padding_value=0).to(device)
            poly_lengths = torch.tensor([length] * len(solutions), dtype=torch.long, device=device)
            sol_lengths = torch.tensor([len(sol) for sol in solutions], dtype=torch.long, device=device)
            
            # Get predicted rewards
            predicted_rewards = value_net(batch_polygons, batch_solutions, poly_lengths, sol_lengths)
            predicted_rewards = predicted_rewards.cpu().numpy()
            
            # Pick best solution according to value network
            best_idx = np.argmin(predicted_rewards)
            best_solution = solutions[best_idx]
            best_pred_reward = predicted_rewards[best_idx]
            
            # Compute actual reward for validation
            real_points = data_tensor[0, :length].cpu().numpy()
            reward_func = partial(reward_fn, alpha=5.0, p=0.0)
            actual_reward = reward_func(real_points, best_solution, name, length=length)
            
            prediction_error = abs(best_pred_reward - actual_reward)
            
            return best_solution, best_pred_reward, actual_reward, predicted_rewards, prediction_error
        else:
            # Fallback if no valid solutions
            return [], 0.0, float('inf'), np.array([]), float('inf')


def load_rl_model(checkpoint_path, embedding_size=128, hidden_size=128, n_glimpses=1, 
                  tanh_exploration=10, use_tanh=True, temperature=1.0):
    """Load the trained RL model from checkpoint."""
    print(f"Loading RL model from: {checkpoint_path}")
    
    # Create model with same architecture as training
    model = create_actor(
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        seq_len=None,  # Will be set automatically for variable length
        n_glimpses=n_glimpses,
        tanh_exploration=tanh_exploration,
        use_tanh=use_tanh,
        attention_type="Bahdanau",
        reward_fn=reward_fn,
        temperature=temperature
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract model state dict from checkpoint
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        model_state_dict = checkpoint
    
    model.load_state_dict(model_state_dict)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    return model, device


def active_search_train_on_instance(model, data_tensor, mask, length, name, K=64, B=8, lr=3e-4, alpha=0.9, device='cpu'):
    """
    Bello et al. Active Search: policy-gradient fine-tuning on a single instance.
    Returns the best solution found and its metrics.
    """
    # Clone the model so global weights aren't modified
    as_model = copy.deepcopy(model)
    as_model.train()
    optimizer = torch.optim.Adam(as_model.parameters(), lr=lr)
    
    data_tensor = data_tensor.to(device)
    mask = mask.to(device)
    length_tensor = torch.tensor([length], dtype=torch.long, device=device)
    real_points = data_tensor[0, :length].detach().cpu().numpy()
    reward_func = partial(reward_fn, alpha=5.0, p=0.0)
    
    # Budget control
    samples_remaining = int(K)
    best_solution = []
    best_reward = float('inf')
    all_rewards_this_instance = []
    b = None  # EMA baseline
    
    while samples_remaining > 0:
        batch = min(B, samples_remaining)
        samples_remaining -= batch
        log_probs_list = []
        losses = []
        sols = []
        # Sample batch
        for _ in range(batch):
            selected_idxs, log_probs = as_model(data_tensor, padding_mask=mask, lengths=length_tensor)
            solution = [idx for idx in selected_idxs[0] if idx < length]
            # skip degenerate empty solution by forcing EOS-only samples to empty
            L = reward_func(real_points, solution, name, length=length)
            losses.append(float(L))
            log_probs_list.append(log_probs.squeeze())  # shape: ()
            sols.append(solution)
            all_rewards_this_instance.append(float(L))
            if L < best_reward:
                best_reward = float(L)
                best_solution = solution
        # Policy gradient step
        if len(losses) > 0:
            L_tensor = torch.tensor(losses, dtype=torch.float32, device=device)
            lp_tensor = torch.stack(log_probs_list)  # [batch]
            mean_L = L_tensor.mean().item()
            b = mean_L if b is None else alpha * b + (1.0 - alpha) * mean_L
            advantage = L_tensor - b
            pg_loss = (advantage * lp_tensor).mean()
            optimizer.zero_grad()
            pg_loss.backward()
            torch.nn.utils.clip_grad_norm_(as_model.parameters(), 1.0)
            optimizer.step()
    
    # Compute coverage for best solution
    try:
        best_coverage = float(evaluate_polygon_visibility_numpy_wo_gt(real_points, best_solution, name))
    except Exception:
        best_coverage = 0.0
    
    return best_solution, best_reward, best_coverage


def evaluate_with_as_training(model, dataset, val_dir, K=64, B=8, lr=3e-4, alpha=0.9, device='cpu'):
    """Run Active Search training per instance and report aggregated stats."""
    import time
    model.eval()
    lengths = get_lengths_from_dataset(dataset)
    batch_sampler = BucketBatchSampler(lengths, batch_size=1, shuffle=False, bucket_size=10)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, pin_memory=True, num_workers=0)
    
    all_best_rewards, all_best_coverages, all_best_sizes, all_size_ratios = [], [], [], []
    start_time = time.time()
    for i, (batch_data, mask, lengths, batch_names) in enumerate(loader):
        data_tensor = batch_data
        instance_mask = mask
        length = lengths[0]
        name = batch_names[0]
        # Run AS on this instance
        best_solution, best_reward, best_coverage = active_search_train_on_instance(
            model, data_tensor, instance_mask, length, name, K=K, B=B, lr=lr, alpha=alpha, device=device
        )
        # Read optimal size for ratio if available
        base_name = os.path.splitext(os.path.basename(name))[0]
        opt_sol_path = os.path.join(val_dir, f"{base_name}.solution")
        try:
            with open(opt_sol_path, 'r') as f:
                lines = f.read().splitlines()
                if len(lines) >= 2:
                    optimal_indices = [int(x) for x in lines[1].split()]
                    optimal_size = len(optimal_indices)
                    size_ratio = len(best_solution) / optimal_size if optimal_size > 0 else float('inf')
                else:
                    size_ratio = float('nan')
        except Exception:
            size_ratio = float('nan')
        # Collect
        all_best_rewards.append(best_reward)
        all_best_coverages.append(best_coverage)
        all_best_sizes.append(len(best_solution))
        all_size_ratios.append(size_ratio)
    total_time = time.time() - start_time
    
    # Stats helper
    def compute_stats(data):
        data = np.array([x for x in data if not np.isnan(x)])
        if len(data) == 0:
            return {k: float('nan') for k in ['mean','median','std','min','max','q25','q75','iqr']}
        return {
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75)),
            'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
        }
    results = {
        'num_instances': len(all_best_rewards),
        'K': K,
        'best_reward_stats': compute_stats(all_best_rewards),
        'best_coverage_stats': compute_stats(all_best_coverages),
        'best_size_stats': compute_stats(all_best_sizes),
        'size_ratio_stats': compute_stats(all_size_ratios),
        'use_value_net': False,
        'evaluation_time': total_time,
        'avg_time_per_instance': total_time / len(all_best_rewards) if len(all_best_rewards) > 0 else 0.0
    }
    return results


def active_search_single_instance(model, data_tensor, mask, length, name, K=10, device='cpu', value_net=None):
    """
    Perform active search on a single instance.
    Sample K solutions and return the best one (lowest reward).
    
    Args:
        model: Trained RL model
        data_tensor: Instance data [1, max_len, 2]
        mask: Padding mask [1, max_len]
        length: Actual length of the instance
        name: Instance name for reward computation
        K: Number of solutions to sample
        device: Device to run on
        value_net: Optional value network for fast evaluation
    
    Returns:
        best_solution: List of guard indices for best solution
        best_reward: Reward value of best solution (true reward of selected)
        best_coverage: Polygon coverage of best solution
        all_rewards: List of all K true rewards
        all_coverages: List of all K coverages
        pred_err_selected: |pred(selected) - true(selected)| if value_net else None
        pred_err_opt: |pred(selected) - true(best_actual)| if value_net else None
        regret: true(selected) - min(true over K) if value_net else None
        corr: Pearson correlation between predicted and true rewards over K (if value_net), else None
    """
    model.eval()
    
    data_tensor = data_tensor.to(device)
    mask = mask.to(device)
    length_tensor = torch.tensor([length], dtype=torch.long, device=device)
    
    solutions = []
    rewards = []
    coverages = []
    pred_err_selected = None
    pred_err_opt = None
    regret = None
    corr = None
    
    with torch.no_grad():
        for k in range(K):
            # Sample a solution
            selected_idxs, log_probs = model(data_tensor, padding_mask=mask, lengths=length_tensor)
            solution = selected_idxs[0]  # Get first (and only) batch item
            
            # Convert to numpy for reward computation
            real_points = data_tensor[0, :length].cpu().numpy()
            real_solution = [idx for idx in solution if idx < length]
            
            # Compute reward (smaller is better)
            r = reward_fn(real_points, real_solution, name, length=length)
            
            # Compute polygon coverage
            try:
                coverage = evaluate_polygon_visibility_numpy_wo_gt(real_points, real_solution, name)
            except:
                coverage = 0.0
            
            solutions.append(real_solution)
            rewards.append(r)
            coverages.append(coverage)
    
    # Find best solution
    if value_net is not None:
        # Predict rewards for all K solutions using the value net (batched, with guard indices)
        K_solutions = len(solutions)
        predicted_values = np.full(K_solutions, np.inf, dtype=float)

        # Indices of non-empty solutions
        valid_idxs = [i for i, sol in enumerate(solutions) if len(sol) > 0]
        if len(valid_idxs) > 0:
            # Prepare batch tensors
            batch_polygons = data_tensor.repeat(len(valid_idxs), 1, 1)
            batch_solutions = pad_sequence(
                [torch.tensor(solutions[i], dtype=torch.long) for i in valid_idxs],
                batch_first=True,
                padding_value=0,
            ).to(device)
            poly_lengths = torch.tensor([length] * len(valid_idxs), dtype=torch.long, device=device)
            sol_lengths = torch.tensor([len(solutions[i]) for i in valid_idxs], dtype=torch.long, device=device)

            with torch.no_grad():
                preds = value_net(batch_polygons, batch_solutions, poly_lengths, sol_lengths)
                preds = preds.detach().cpu().numpy()
            for j, i in enumerate(valid_idxs):
                predicted_values[i] = float(preds[j])

        # Select best according to predicted value
        best_idx = int(np.argmin(predicted_values)) if K_solutions > 0 else 0

        # Compute additional metrics
        actual_best_reward = float(np.min(rewards)) if len(rewards) > 0 else float('nan')
        predicted_best_value = float(predicted_values[best_idx]) if K_solutions > 0 else float('nan')
        selected_true_reward = float(rewards[best_idx]) if len(rewards) > 0 else float('nan')

        pred_err_selected = abs(predicted_best_value - selected_true_reward) if np.isfinite(predicted_best_value) else float('nan')
        pred_err_opt = abs(predicted_best_value - actual_best_reward) if np.isfinite(predicted_best_value) else float('nan')
        regret = selected_true_reward - actual_best_reward

        # Correlation over K between predicted and true if at least 2 valid pairs
        try:
            mask = np.isfinite(predicted_values)
            if np.sum(mask) >= 2:
                corr = float(np.corrcoef(np.array(rewards)[mask], predicted_values[mask])[0, 1])
            else:
                corr = float('nan')
        except Exception:
            corr = float('nan')
    else:
        # Use actual rewards to select best solution (minimum reward)
        best_idx = int(np.argmin(rewards)) if len(rewards) > 0 else 0
    
    best_solution = solutions[best_idx]
    best_reward = rewards[best_idx]
    best_coverage = coverages[best_idx]
    
    return best_solution, best_reward, best_coverage, rewards, coverages, pred_err_selected, pred_err_opt, regret, corr


def evaluate_with_active_search(model, dataset, val_dir, K=10, batch_size=1, device='cpu', value_net=None, use_value_net=False):
    """
    Evaluate the model on validation dataset using active search.
    
    Args:
        model: Trained RL model
        dataset: Validation dataset
        val_dir: Directory containing .solution files for optimal comparison
        K: Number of solutions to sample per instance
        batch_size: Batch size for evaluation (should be 1 for active search)
        device: Device to run on
        value_net: Optional trained value network for fast evaluation
        use_value_net: Whether to use value network for solution selection
    
    Returns:
        results: Dictionary with evaluation results
    """
    import time
    
    search_type = "fast" if use_value_net else "standard"
    print(f"\n--- {search_type.title()} Active Search Evaluation (K={K}) on {len(dataset)} validation samples ---")
    
    model.eval()
    
    # Use bucket sampler for consistent batching
    lengths = get_lengths_from_dataset(dataset)
    batch_sampler = BucketBatchSampler(lengths, batch_size=1, shuffle=False, bucket_size=10)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, 
                       pin_memory=True, num_workers=0)  # num_workers=0 for deterministic sampling
    
    all_best_rewards = []
    all_best_coverages = []
    all_best_sizes = []
    all_size_ratios = []
    all_reward_variances = []
    all_coverage_variances = []
    all_pred_err_selected = []
    all_pred_err_opt = []
    all_regrets = []
    all_corrs = []
    
    # Timing statistics
    total_time = 0
    total_selection_time = 0
    total_reward_computation_time = 0
    
    start_time = time.time()
    
    for i, (batch_data, mask, lengths, batch_names) in enumerate(loader):
        instance_start = time.time()
        
        # Process one instance at a time
        data_tensor = batch_data  # [1, max_len, 2]
        instance_mask = mask      # [1, max_len]
        length = lengths[0]       # Actual length
        name = batch_names[0]     # Instance name
        
        # Perform active search
        best_solution, best_reward, best_coverage, all_rewards, all_coverages, pred_err_selected, pred_err_opt, regret, corr = \
            active_search_single_instance(model, data_tensor, instance_mask, length, name, K, device, value_net)
        
        instance_time = time.time() - instance_start
        total_time += instance_time
        
        # Load optimal solution for size ratio comparison
        base_name = os.path.splitext(os.path.basename(name))[0]
        opt_sol_path = os.path.join(val_dir, f"{base_name}.solution")
        try:
            with open(opt_sol_path, 'r') as f:
                lines = f.read().splitlines()
                if len(lines) >= 2:
                    optimal_indices = [int(x) for x in lines[1].split()]
                    optimal_size = len(optimal_indices)
                    size_ratio = len(best_solution) / optimal_size if optimal_size > 0 else float('inf')
                else:
                    size_ratio = float('nan')
        except Exception:
            size_ratio = float('nan')
        
        # Collect statistics
        all_best_rewards.append(best_reward)
        all_best_coverages.append(best_coverage)
        all_best_sizes.append(len(best_solution))
        all_size_ratios.append(size_ratio)
        all_reward_variances.append(np.var(all_rewards))
        all_coverage_variances.append(np.var(all_coverages))
        
        # Collect value net quality metrics if using value network
        if pred_err_selected is not None:
            all_pred_err_selected.append(pred_err_selected)
        if pred_err_opt is not None:
            all_pred_err_opt.append(pred_err_opt)
        if regret is not None:
            all_regrets.append(regret)
        if corr is not None:
            all_corrs.append(corr)
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            avg_time_per_instance = elapsed / (i + 1)
            estimated_total = avg_time_per_instance * len(loader)
            estimated_remaining = estimated_total - elapsed
            print(f"Processed {i + 1}/{len(loader)} instances... "
                  f"(avg: {avg_time_per_instance:.2f}s/instance, "
                  f"ETA: {estimated_remaining/60:.1f}m)")
    
    total_evaluation_time = time.time() - start_time
    
    # Compute statistics using the same format as RL evaluation
    def compute_stats(data, name):
        data = np.array([x for x in data if not np.isnan(x)])
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
        return stats
    
    # Compute all statistics
    # Calculate statistics for all metrics
    best_reward_stats = compute_stats(all_best_rewards, "best_reward")
    best_coverage_stats = compute_stats(all_best_coverages, "best_coverage")
    best_size_stats = compute_stats(all_best_sizes, "best_size")
    size_ratio_stats = compute_stats(all_size_ratios, "size_ratio")
    
    # Calculate statistics for optional metrics (only if present)
    reward_diversity_stats = compute_stats(all_reward_variances, "reward_diversity") if len(all_reward_variances) > 0 else None
    coverage_diversity_stats = compute_stats(all_coverage_variances, "coverage_diversity") if len(all_coverage_variances) > 0 else None
    pred_err_selected_stats = compute_stats(all_pred_err_selected, "pred_err_selected") if len(all_pred_err_selected) > 0 else None
    pred_err_opt_stats = compute_stats(all_pred_err_opt, "pred_err_opt") if len(all_pred_err_opt) > 0 else None
    regret_stats = compute_stats(all_regrets, "regret") if len(all_regrets) > 0 else None
    # For correlations, report mean/std/min/max, leave percentiles as NaN if empty
    corr_stats = {
        'mean': float(np.nanmean(all_corrs)) if len(all_corrs) > 0 else float('nan'),
        'median': float(np.nanmedian(all_corrs)) if len(all_corrs) > 0 else float('nan'),
        'std': float(np.nanstd(all_corrs)) if len(all_corrs) > 0 else float('nan'),
        'min': float(np.nanmin(all_corrs)) if len(all_corrs) > 0 else float('nan'),
        'max': float(np.nanmax(all_corrs)) if len(all_corrs) > 0 else float('nan'),
        'q25': float('nan'), 'q75': float('nan'), 'iqr': float('nan')
    }
    
    # Compile results
    results_summary = {
        'num_instances': len(all_best_rewards),
        'K': K,
        'best_reward_stats': best_reward_stats,
        'best_coverage_stats': best_coverage_stats,
        'best_size_stats': best_size_stats,
        'size_ratio_stats': size_ratio_stats,
        'use_value_net': value_net is not None,
        'evaluation_time': total_evaluation_time,
        'avg_time_per_instance': total_evaluation_time / len(all_best_rewards) if len(all_best_rewards) > 0 else 0
    }
    
    # Add optional statistics if available
    if reward_diversity_stats:
        results_summary['reward_diversity_stats'] = reward_diversity_stats
    if coverage_diversity_stats:
        results_summary['coverage_diversity_stats'] = coverage_diversity_stats
    if pred_err_selected_stats:
        results_summary['pred_err_selected_stats'] = pred_err_selected_stats
    if pred_err_opt_stats:
        results_summary['pred_err_opt_stats'] = pred_err_opt_stats
    if regret_stats:
        results_summary['regret_stats'] = regret_stats
    if np.isfinite(corr_stats['mean']):
        results_summary['corr_stats'] = corr_stats
    
    return results_summary


def print_results(results):
    """Print evaluation results in a nice format."""
    use_value_net = results.get('use_value_net', False)
    search_type = "Fast" if use_value_net else "Standard"
    print(f"\n=== {search_type.upper()} ACTIVE SEARCH EVALUATION RESULTS (K={results['K']}) ===")
    print(f"Number of instances: {results['num_instances']}")
    
    # Print timing information
    if 'evaluation_time' in results:
        total_time = results['evaluation_time']
        avg_time = results.get('avg_time_per_instance', 0)
        print(f"\nTiming Information:")
        print(f"  Total evaluation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"  Average time per instance: {avg_time:.3f} seconds")
        if use_value_net:
            print(f"  🚀 Using value network for fast evaluation!")
        else:
            print(f"  ⏳ Using standard evaluation (computing actual rewards for all K samples)")
    
    def print_stat_section(stats, title):
        if all(key in stats for key in ['mean', 'median', 'std', 'min', 'max', 'q25', 'q75', 'iqr']):
            print(f"\n{title}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Median: {stats['median']:.4f}")
            print(f"  Std: {stats['std']:.4f}")
            print(f"  Min: {stats['min']:.4f}")
            print(f"  Max: {stats['max']:.4f}")
            print(f"  Q25: {stats['q25']:.4f}")
            print(f"  Q75: {stats['q75']:.4f}")
            print(f"  IQR: {stats['iqr']:.4f}")
    
    print_stat_section(results['best_reward_stats'], "Best Reward Statistics")
    print_stat_section(results['best_coverage_stats'], "Best Coverage Statistics") 
    print_stat_section(results['best_size_stats'], "Best Solution Size Statistics")
    print_stat_section(results['size_ratio_stats'], "Size Ratio Statistics (best/optimal)")
    
    # Print value net quality stats if using value network
    if use_value_net:
        if 'pred_err_selected_stats' in results:
            print_stat_section(results['pred_err_selected_stats'], "|pred(selected) - true(selected)|")
        if 'pred_err_opt_stats' in results:
            print_stat_section(results['pred_err_opt_stats'], "|pred(selected) - true(best over K)|")
        if 'regret_stats' in results:
            print_stat_section(results['regret_stats'], "Regret: true(selected) - min(true over K)")
        if 'corr_stats' in results:
            print(f"\nCorrelation (pred vs true over K):")
            print(f"  Mean: {results['corr_stats']['mean']:.4f}")
            print(f"  Std:  {results['corr_stats']['std']:.4f}")
            print(f"  Min:  {results['corr_stats']['min']:.4f}")
            print(f"  Max:  {results['corr_stats']['max']:.4f}")
    
    # Print diversity stats if available
    if 'reward_diversity_stats' in results and 'coverage_diversity_stats' in results:
        print(f"\nActive Search Diversity:")
        print(f"  Mean Reward Variance:   {results['reward_diversity_stats']['mean']:.6f}")
        print(f"  Mean Coverage Variance: {results['coverage_diversity_stats']['mean']:.6f}")


def main():
    # Load environment variables
    load_dotenv()
    DATASET_PATH = os.getenv("DATASET_PATH")
    if not DATASET_PATH:
        raise EnvironmentError("DATASET_PATH environment variable must be set in .env file.")
    
    parser = argparse.ArgumentParser(description='Evaluate RL AGP model with active search')
    parser.add_argument('--checkpoint', type=str, 
                       default='checkpoints/rl_agp_model_embedding_size128_hidden_size128_n_glimpses1_tanh_exploration10_temperature1.0_use_tanhTrue_epochs30.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--val-dir', type=str, 
                       default=os.path.join(DATASET_PATH, "dev"),
                       help='Directory with validation .pol files')
    parser.add_argument('--K', type=int, default=10,
                       help='Number of solutions to sample per instance')
    parser.add_argument('--max-instances', type=int, default=None,
                       help='Maximum number of instances to evaluate (for testing)')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize coordinates')
    parser.add_argument('--output', type=str, default='results/active_search_evaluation_K={K}.json',
                       help='Output file for results (K will be replaced with actual value)')
    
    # Active Search (test-time training) parameters
    parser.add_argument('--use-as', action='store_true',
                       help='Use Active Search test-time training on each instance')
    parser.add_argument('--as-batch-size', type=int, default=8,
                       help='Batch size B for Active Search (solutions per gradient step)')
    parser.add_argument('--as-lr', type=float, default=3e-4,
                       help='Learning rate for Active Search Adam optimizer')
    parser.add_argument('--as-alpha', type=float, default=0.9,
                       help='EMA baseline alpha for Active Search (0.9-0.99)')
    
    # Value network parameters
    parser.add_argument('--use-value-net', action='store_true',
                       help='Use trained value network for fast active search')
    parser.add_argument('--value-net-path', type=str, 
                       default='auto',
                       help='Path to trained value network ("auto" to auto-detect best checkpoint)')
    parser.add_argument('--value-net-embedding-size', type=int, default=128,
                       help='Embedding size for value network (should match training)')
    parser.add_argument('--value-net-hidden-size', type=int, default=256,
                       help='Hidden size for value network (should match training)')
    parser.add_argument('--value-net-fallback', action='store_true', default=True,
                       help='Fall back to standard evaluation if value network fails')
    
    # Model architecture parameters (should match training)
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--n-glimpses', type=int, default=1)
    parser.add_argument('--tanh-exploration', type=int, default=10)
    parser.add_argument('--use-tanh', action='store_true', default=True)
    parser.add_argument('--temperature', type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Load validation dataset
    print(f"Loading validation dataset from: {args.val_dir}")
    val_files = [os.path.join(args.val_dir, f) for f in os.listdir(args.val_dir) if f.endswith('.pol')]
    if args.max_instances:
        val_files = val_files[:args.max_instances]
    
    val_samples = agp_read_samples(val_files, normalize=args.normalize)
    val_dataset = Dataset(val_samples)
    print(f"Loaded {len(val_dataset)} validation instances")
    
    # Load model
    model, device = load_rl_model(
        args.checkpoint,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        n_glimpses=args.n_glimpses,
        tanh_exploration=args.tanh_exploration,
        use_tanh=args.use_tanh,
        temperature=args.temperature
    )
    
    # Load value network if requested
    value_net = None
    if args.use_value_net:
        # Auto-detect value network if needed
        value_net_path = args.value_net_path
        if value_net_path == 'auto':
            auto_path = find_best_value_net_checkpoint()
            if auto_path:
                value_net_path = auto_path
                print(f"🔍 Auto-detected value network: {value_net_path}")
            else:
                print("✗ No value network checkpoints found in 'checkpoints/' directory")
                if args.value_net_fallback:
                    print("  Falling back to standard active search")
                    args.use_value_net = False
                else:
                    raise FileNotFoundError("No value network checkpoints found and fallback disabled")
        
        if args.use_value_net and os.path.exists(value_net_path):
            try:
                value_net, _ = load_value_net(value_net_path, 
                                            embedding_size=args.value_net_embedding_size,
                                            hidden_size=args.value_net_hidden_size)
                print(f"✓ Value network loaded successfully from {value_net_path}")
                print(f"  Architecture: embedding_size={args.value_net_embedding_size}, hidden_size={args.value_net_hidden_size}")
            except Exception as e:
                print(f"✗ Error loading value network: {e}")
                if args.value_net_fallback:
                    print("  Falling back to standard active search")
                    args.use_value_net = False
                else:
                    raise
        elif args.use_value_net:
            print(f"✗ Value network not found at {value_net_path}")
            if args.value_net_fallback:
                print("  Falling back to standard active search")
                args.use_value_net = False
            else:
                raise FileNotFoundError(f"Value network not found: {value_net_path}")
    
    # Evaluate with chosen strategy
    if args.use_as:
        results = evaluate_with_as_training(
            model, val_dataset, args.val_dir,
            K=args.K,
            B=args.as_batch_size,
            lr=args.as_lr,
            alpha=args.as_alpha,
            device=device
        )
    else:
        results = evaluate_with_active_search(model, val_dataset, args.val_dir, K=args.K, device=device,
                                            value_net=value_net, use_value_net=args.use_value_net)
    
    # Print results
    print_results(results)
    
    # Prepare results for saving (match RL evaluation format)
    if args.use_as:
        training_method = 'active_search_training'
    else:
        method_suffix = "_fast" if args.use_value_net else "_standard"
        training_method = f'active_search_rl{method_suffix}'
    results_summary = {
        'args': vars(args),
        'K': args.K,
        'use_value_net': args.use_value_net,
        'num_instances': results['num_instances'],
        'training_method': training_method,
        'best_reward_stats': results['best_reward_stats'],
        'best_coverage_stats': results['best_coverage_stats'], 
        'best_size_stats': results['best_size_stats'],
        'size_ratio_stats': results['size_ratio_stats']
    }
    
    # Add prediction error stats if using value network
    if args.use_value_net and 'prediction_error_stats' in results:
        results_summary['prediction_error_stats'] = results['prediction_error_stats']
    
    # Generate output filename with K value
    output_file = args.output.replace('{K}', str(args.K))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy_types(results_summary)
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
