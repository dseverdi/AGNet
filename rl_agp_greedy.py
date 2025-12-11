#!/usr/bin/env python3
"""
RL-AGP with Greedy Baseline

This variant uses precomputed greedy solutions as the baseline for advantage estimation,
providing more stable and informed training compared to EMA or learned critic baselines.

Key difference from rl_agp.py:
- Loads greedy cache (data/greedy_baseline_train.pkl) at startup
- Uses greedy rewards as baseline: advantage = rl_reward - greedy_reward
- Provides immediate feedback on whether RL solution is better than greedy
"""

import os
from dotenv import load_dotenv
import argparse
from dataset import Dataset, agp_read_samples, collate_fn
from models import create_actor, create_critic
from utils import createPolygon, compute_visibility
import torch
from torch.utils.data import DataLoader, Sampler

import skgeom

from utils import evaluate_polygon_visibility_numpy_wo_gt  # reuse coverage evaluation
import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from rewards import coverage_smooth_reward as reward  # Smooth reward with no discontinuity
from functools import wraps, partial

# use torch's nn and functional via existing torch import
nn = torch.nn
F = torch.nn.functional

# --- Utility ---
def get_checkpoint_path(folder, model_name, params, n_epochs):
    """Generate a checkpoint path based on model name, parameters, and epoch count."""
    param_str = "_".join([f"{k}{v}" for k, v in sorted(params.items())])
    filename = f"{model_name}_{param_str}_epochs{n_epochs}.pt"
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)

def load_greedy_cache(cache_path):
    """Load precomputed greedy solutions cache."""
    print(f"\n[GREEDY CACHE] Loading from {cache_path}...")
    try:
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        print(f"[GREEDY CACHE] Successfully loaded {len(cache)} greedy solutions")
        
        # Print sample statistics
        if len(cache) > 0:
            sample_entries = list(cache.items())[:3]
            print(f"[GREEDY CACHE] Sample entries:")
            for name, data in sample_entries:
                print(f"  - {name}: {data['num_guards']} guards, coverage={data['coverage']:.3f}, reward={data['reward']:.2f}")
        
        return cache
    except Exception as e:
        print(f"[GREEDY CACHE] ERROR: Failed to load cache: {e}")
        print(f"[GREEDY CACHE] Continuing without greedy baseline (will fall back to EMA)")
        return None

# --- Data Preparation ---
def prepare_datasets(train_path, val_path, normalize=True):
    # Accepts either a directory or a single .pol file for both train and val
    def get_pol_files(path):
        if os.path.isdir(path):
            return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pol')]
        elif os.path.isfile(path) and path.endswith('.pol'):
            return [path]
        else:
            raise ValueError(f"Provided path {path} is neither a .pol file nor a directory containing .pol files.")
    agp_train_paths = get_pol_files(train_path)
    agp_val_paths = get_pol_files(val_path)
    print(f"Found {len(agp_train_paths)} training and {len(agp_val_paths)} validation AGP .pol files.")
    train_samples = agp_read_samples(agp_train_paths, normalize=normalize)
    val_samples = agp_read_samples(agp_val_paths, normalize=normalize)
    return Dataset(train_samples), Dataset(val_samples)

# --- Model Creation ---
def create_agp_model(embedding_size, hidden_size, n_glimpses, tanh_exploration, use_tanh, reward, temperature):
    return create_actor(
        embedding_size, hidden_size, None, n_glimpses,
        tanh_exploration, use_tanh, "Bahdanau", reward, temperature=temperature
    )

def create_critic_model(embedding_size, hidden_size, n_glimpses, attention_type):
    return create_critic(embedding_size, hidden_size, n_glimpses, attention_type)

class BucketBatchSampler(Sampler):
    """Batch sampler that groups samples of similar length into the same batch."""
    def __init__(self, lengths, batch_size, shuffle=True, drop_last=False, bucket_size=10):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.bucket_size = bucket_size
        # Sort indices by length
        self.sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        # Create buckets
        self.buckets = []
        for i in range(0, len(self.sorted_indices), bucket_size):
            self.buckets.append(self.sorted_indices[i:i+bucket_size])

    def __iter__(self):
        # Shuffle buckets if needed
        buckets = self.buckets.copy()
        if self.shuffle:
            np.random.shuffle(buckets)
        for bucket in buckets:
            # Shuffle within bucket
            if self.shuffle:
                np.random.shuffle(bucket)
            # Yield batches from this bucket
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i+self.batch_size]
                if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
                    yield batch

    def __len__(self):
        total = 0
        for bucket in self.buckets:
            n = len(bucket)
            total += n // self.batch_size
            if not self.drop_last and n % self.batch_size != 0:
                total += 1
        return total

def get_lengths_from_dataset(dataset):
    # Assumes dataset[i][0] is a tensor of shape [num_points, 2]
    return [sample[0].shape[0] for sample in dataset]


def reinforce_train_greedy_baseline(model, dataset, reward_fn, greedy_cache, epochs=2, batch_size=1, lr=1e-3, beta=0.99):
    """
    Train the model using REINFORCE with GREEDY BASELINE for variance reduction.
    
    Uses precomputed greedy solutions as baseline instead of EMA or learned critic.
    Advantage = RL_reward - Greedy_reward
    
    Args:
        model: The actor model to train.
        dataset: The dataset containing samples.
        reward_fn: The reward function to compute rewards.
        greedy_cache: Dictionary mapping polygon names to greedy solutions {name: {'guards': [...], 'reward': float, ...}}
        epochs: Number of training epochs.
        batch_size: Size of each training batch.
        lr: Learning rate for the optimizer.
        beta: EMA decay factor for fallback baseline when greedy solution not available.
    """
    print(f"\n--- Training with Greedy Baseline on {len(dataset)} samples for {epochs} epochs (batch size {batch_size}) ---")
    
    if greedy_cache is None:
        print("[WARNING] No greedy cache available, falling back to EMA baseline")
    else:
        print(f"[GREEDY BASELINE] Using {len(greedy_cache)} precomputed greedy solutions as baseline")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lengths = get_lengths_from_dataset(dataset)
    batch_sampler = BucketBatchSampler(lengths, batch_size, shuffle=True, bucket_size=10)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, pin_memory=True, num_workers=2)
    device = next(model.parameters()).device
    
    # Fallback EMA baseline for samples without greedy solution
    ema_baseline = 0.0
    
    # Statistics
    total_greedy_hits = 0
    total_greedy_misses = 0
    total_improvements = 0  # Count when RL beats greedy
    
    # Use tqdm for epoch progress if available
    epoch_iterator = tqdm(range(epochs), desc="Training Epochs") if tqdm else range(epochs)
    
    for epoch in epoch_iterator:
        total_loss = 0
        batch_count = 0
        epoch_greedy_hits = 0
        epoch_greedy_misses = 0
        epoch_improvements = 0
        
        # Use tqdm for batch progress if available
        batch_iterator = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} Batches", leave=False) if tqdm else loader
        
        for batch_data, mask, lengths, batch_names in batch_iterator:
            batch_data = batch_data.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            lengths = torch.tensor(lengths, dtype=torch.long, device=device)
            
            # Forward pass: model should return (selected_idxs, log_probs)
            selected_idxs, log_probs = model(batch_data, padding_mask=mask, lengths=lengths)
            
            # Compute RL rewards
            rewards_list = []
            baselines_list = []
            
            for i, (data_tensor, idxs, name) in enumerate(zip(batch_data.cpu(), selected_idxs, batch_names)):
                n = lengths[i].item() if lengths is not None else len(data_tensor)
                real_points = data_tensor[:n].detach().cpu().numpy()
                real_solution = [idx for idx in idxs if idx < n]
                
                # Compute RL reward (strict_reward returns reward directly, higher = better)
                rl_reward = reward(real_points, real_solution, name, length=n)
                rewards_list.append(rl_reward)
                
                # Get greedy baseline for this sample
                # Extract polygon name without extension
                base_name = os.path.splitext(os.path.basename(name))[0]
                
                if greedy_cache is not None and base_name in greedy_cache:
                    # Recompute greedy reward using current reward function (not cached value)
                    # This ensures consistency when reward function changes
                    greedy_guards = greedy_cache[base_name]['guards']
                    greedy_reward = reward(real_points, greedy_guards, name, length=n)
                    baselines_list.append(greedy_reward)
                    epoch_greedy_hits += 1
                    
                    # Track improvements over greedy
                    if rl_reward > greedy_reward:
                        epoch_improvements += 1
                else:
                    # Fallback to EMA baseline
                    baselines_list.append(ema_baseline)
                    epoch_greedy_misses += 1
            
            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
            baselines = torch.tensor(baselines_list, dtype=torch.float32, device=device)
            
            # Update EMA baseline for fallback
            batch_mean = rewards.mean().item()
            ema_baseline = beta * ema_baseline + (1 - beta) * batch_mean
            
            # Compute advantages using greedy baseline (or EMA fallback)
            advantages = rewards - baselines
            
            # REINFORCE loss using advantages
            loss = -(log_probs * advantages).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_data.size(0)
            batch_count += 1
            
            # Update tqdm postfix with current loss and statistics
            if tqdm and hasattr(batch_iterator, 'set_postfix'):
                batch_iterator.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_reward': f"{batch_mean:.3f}",
                    'greedy_hits': f"{epoch_greedy_hits}/{epoch_greedy_hits + epoch_greedy_misses}"
                })
            
            # Free up memory after each batch
            del batch_data, mask, lengths, selected_idxs, log_probs, rewards, baselines
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        avg_loss = total_loss / len(dataset)
        total_greedy_hits += epoch_greedy_hits
        total_greedy_misses += epoch_greedy_misses
        total_improvements += epoch_improvements
        
        # Epoch summary
        hit_rate = 100.0 * epoch_greedy_hits / (epoch_greedy_hits + epoch_greedy_misses) if (epoch_greedy_hits + epoch_greedy_misses) > 0 else 0
        improvement_rate = 100.0 * epoch_improvements / epoch_greedy_hits if epoch_greedy_hits > 0 else 0
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} | "
              f"Greedy hits: {epoch_greedy_hits}/{epoch_greedy_hits + epoch_greedy_misses} ({hit_rate:.1f}%) | "
              f"RL > Greedy: {epoch_improvements}/{epoch_greedy_hits} ({improvement_rate:.1f}%)")
        
        torch.cuda.empty_cache()
    
    # Final summary
    print(f"\n[TRAINING SUMMARY]")
    print(f"  Total samples with greedy baseline: {total_greedy_hits}")
    print(f"  Total samples with EMA fallback: {total_greedy_misses}")
    print(f"  Times RL beat greedy: {total_improvements}/{total_greedy_hits} ({100.0*total_improvements/total_greedy_hits:.1f}%)")
    print("Training done.")

# --- Evaluation ---
def reinforce_eval(model, dataset, reward_fn, batch_size=1, sol_dir=None, greedy_cache=None):
    """Evaluate the model on the validation dataset with comprehensive metrics for whisker plots."""
    print(f"\n--- Evaluation on {len(dataset)} validation samples (batch size {batch_size}) ---")
    
    if greedy_cache is not None:
        print(f"[EVAL] Will compare RL solutions against {len(greedy_cache)} greedy baselines")
    
    model.eval()
    lengths = get_lengths_from_dataset(dataset)
    batch_sampler = BucketBatchSampler(lengths, batch_size, shuffle=False, bucket_size=10)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=2)
    device = next(model.parameters()).device
    
    # Statistics for whisker plots (same as supervised learning)
    pred_sizes = []
    true_sizes = []
    coverage_ratios = []  # What fraction of optimal guards are covered by predicted guards
    efficiency_ratios = []  # What fraction of predicted guards are in the optimal set
    size_ratios = []  # predicted_size / optimal_size
    overlap_counts = []  # absolute number of overlapping guards
    all_rewards = []
    all_coverages = []
    
    # Greedy comparison statistics
    greedy_rewards = []
    rl_vs_greedy_improvements = 0
    greedy_comparisons = 0
    
    if sol_dir is None:
        # Default: use the directory of the first validation .pol file
        if hasattr(dataset, 'samples') and len(dataset.samples) > 0 and hasattr(dataset.samples[0], 'name'):
            sample_name = dataset.samples[0].name
            sol_dir = os.path.dirname(sample_name)
        else:
            sol_dir = '.'
    
    with torch.no_grad():
        for batch_data, mask, lengths, batch_names in loader:
            batch_data = batch_data.to(device)
            mask = mask.to(device)
            lengths = torch.tensor(lengths, dtype=torch.long, device=device)
            selected_idxs, log_probs = model(batch_data, padding_mask=mask, lengths=lengths)
            
            for i, (data_tensor, pred_indices, name) in enumerate(zip(batch_data.cpu(), selected_idxs, batch_names)):
                points = data_tensor.numpy()
                length = lengths[i].item() if lengths is not None else len(points)
                
                # Compute reward and coverage (strict_reward returns reward directly)
                r = reward_fn(points, np.array(pred_indices), name, length=length)
                all_rewards.append(r)
                
                # Compare with greedy if available (recompute greedy reward with current reward function)
                base_name = os.path.splitext(os.path.basename(name))[0]
                if greedy_cache is not None and base_name in greedy_cache:
                    greedy_guards = greedy_cache[base_name]['guards']
                    greedy_reward = reward_fn(points, np.array(greedy_guards), name, length=length)
                    greedy_rewards.append(greedy_reward)
                    greedy_comparisons += 1
                    if r > greedy_reward:  # r is RL reward (higher=better)
                        rl_vs_greedy_improvements += 1
                
                # Extract coverage from reward if available
                if isinstance(r, tuple) and len(r) == 2:
                    coverage = r[1]
                else:
                    try:
                        from utils import evaluate_polygon_visibility_numpy_wo_gt
                        coverage = evaluate_polygon_visibility_numpy_wo_gt(points, np.array(pred_indices), name)
                    except Exception:
                        coverage = float('nan')
                all_coverages.append(coverage)
                
                # Read optimal solution for comparison
                opt_sol_path = os.path.join(sol_dir, f"{base_name}.solution")
                true_indices = []
                try:
                    with open(opt_sol_path, 'r') as f:
                        lines = f.read().splitlines()
                        if len(lines) >= 2:
                            true_indices = [int(x) for x in lines[1].split()]
                except Exception:
                    true_indices = []
                
                if len(true_indices) > 0:
                    pred_set = set(pred_indices)
                    true_set = set(true_indices)
                    overlap = pred_set.intersection(true_set)
                    
                    pred_size = len(pred_indices)
                    true_size = len(true_indices)
                    overlap_count = len(overlap)
                    
                    # Coverage: what fraction of optimal guards are covered
                    coverage_ratio = overlap_count / true_size if true_size > 0 else 0.0
                    
                    # Efficiency: what fraction of predicted guards are optimal
                    efficiency_ratio = overlap_count / pred_size if pred_size > 0 else 0.0
                    
                    # Size ratio: how many times larger is the prediction vs optimal
                    size_ratio = pred_size / true_size if true_size > 0 else float('inf')
                    
                    pred_sizes.append(pred_size)
                    true_sizes.append(true_size)
                    coverage_ratios.append(coverage_ratio)
                    efficiency_ratios.append(efficiency_ratio)
                    size_ratios.append(size_ratio)
                    overlap_counts.append(overlap_count)
    
    # Compute statistics for whisker plots
    def compute_stats(data, name):
        data = np.array(data)
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
        print(f"\n{name} Statistics:")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Median: {stats['median']:.3f}")
        print(f"  Std: {stats['std']:.3f}")
        print(f"  Min: {stats['min']:.3f}")
        print(f"  Max: {stats['max']:.3f}")
        print(f"  Q25: {stats['q25']:.3f}")
        print(f"  Q75: {stats['q75']:.3f}")
        print(f"  IQR: {stats['iqr']:.3f}")
        return stats
    
    print("\n=== EVALUATION RESULTS ===")
    
    # Greedy comparison summary
    if greedy_comparisons > 0:
        improvement_rate = 100.0 * rl_vs_greedy_improvements / greedy_comparisons
        print(f"\n[GREEDY COMPARISON]")
        print(f"  Comparisons: {greedy_comparisons}")
        print(f"  RL > Greedy: {rl_vs_greedy_improvements} ({improvement_rate:.1f}%)")
        print(f"  Avg RL reward: {np.mean(all_rewards):.3f}")
        if len(greedy_rewards) > 0:
            print(f"  Avg Greedy reward: {np.mean(greedy_rewards):.3f}")
    
    # Compute all statistics
    if len(pred_sizes) > 0:
        size_stats = compute_stats(pred_sizes, "Predicted Solution Sizes")
        optimal_stats = compute_stats(true_sizes, "Optimal Solution Sizes")
        coverage_stats = compute_stats(coverage_ratios, "Guard Set Coverage Ratios (fraction of optimal guards covered by predicted guards)")
        efficiency_stats = compute_stats(efficiency_ratios, "Efficiency Ratios (fraction of predicted guards that are optimal)")
        ratio_stats = compute_stats(size_ratios, "Size Ratios (predicted/optimal)")
        overlap_stats = compute_stats(overlap_counts, "Overlap Counts (absolute number of matching guards)")
        
        # Summary metrics
        print(f"\n=== SUMMARY ===")
        print(f"Instances evaluated: {len(pred_sizes)}")
        print(f"Perfect solutions (100% coverage): {sum(1 for c in coverage_ratios if c >= 1.0)}")
        print(f"Good solutions (>=80% coverage): {sum(1 for c in coverage_ratios if c >= 0.8)}")
        print(f"Reasonable solutions (>=60% coverage): {sum(1 for c in coverage_ratios if c >= 0.6)}")
        print(f"Average coverage: {np.mean(coverage_ratios):.3f}")
        if len(size_ratios) > 0:
            print(f"Average size inflation: {np.mean(size_ratios):.2f}x optimal")
    else:
        print("No valid instances with optimal solutions found for comparison.")
        size_stats = optimal_stats = coverage_stats = efficiency_stats = ratio_stats = overlap_stats = {}
    
    # Compute coverage statistics from polygon visibility
    coverage_array = np.array([c for c in all_coverages if not np.isnan(c)])
    coverage_vis_stats = compute_stats(coverage_array, "Polygon Coverage (visibility)") if len(coverage_array) > 0 else {}
    
    # Backward compatibility: old format metrics (skip for now as it's complex to compute)
    rel_sizes = []
    
    return {
        # New comprehensive metrics
        'pred_sizes': pred_sizes,
        'true_sizes': true_sizes,
        'coverage_ratios': coverage_ratios,
        'efficiency_ratios': efficiency_ratios,
        'size_ratios': size_ratios,
        'overlap_counts': overlap_counts,
        'polygon_coverages': coverage_array.tolist() if len(coverage_array) > 0 else [],
        'stats': {
            'size_stats': size_stats,
            'optimal_stats': optimal_stats,
            'coverage_stats': coverage_stats,
            'efficiency_stats': efficiency_stats,
            'ratio_stats': ratio_stats,
            'overlap_stats': overlap_stats,
            'coverage_vis_stats': coverage_vis_stats
        },
        # Greedy comparison metrics
        'greedy_comparisons': greedy_comparisons,
        'rl_vs_greedy_improvements': rl_vs_greedy_improvements,
        'avg_greedy_reward': np.mean(greedy_rewards) if len(greedy_rewards) > 0 else float('nan'),
        # Legacy compatibility
        'rewards': all_rewards,
        'coverages': all_coverages,
        'rel_sizes': rel_sizes
    }


def main():
    # Load environment variables from .env
    load_dotenv()
    DATASET_PATH = os.getenv("DATASET_PATH")
    if not DATASET_PATH:
        raise EnvironmentError("DATASET_PATH environment variable must be set in .env file.")
    
    parser = argparse.ArgumentParser(description='RL-AGP with Greedy Baseline')
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--n-glimpses', type=int, default=1)
    parser.add_argument('--tanh-exploration', type=float, default=10)
    parser.add_argument('--use-tanh', action='store_true', default=True)
    parser.add_argument('--beta', type=float, default=0.99, help='EMA decay for fallback baseline')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=1.0)
    default_train = os.path.join(DATASET_PATH, "train")
    default_val = os.path.join(DATASET_PATH, "dev")
    parser.add_argument('--agp_train_dir', type=str, default=default_train)
    parser.add_argument('--agp_val_dir', type=str, default=default_val)
    parser.add_argument('--train-size', type=int, default=8000, help="Number of training samples to use (default: 8000, or all if smaller)")
    parser.add_argument('--greedy-cache', type=str, default='data/greedy_baseline_train.pkl', 
                       help='Path to greedy baseline cache file')
    
    args = parser.parse_args()

    # Load greedy cache
    greedy_cache = load_greedy_cache(args.greedy_cache)
    
    # Load datasets
    train_dataset, val_dataset = prepare_datasets(args.agp_train_dir, args.agp_val_dir, normalize=True)
    
    # Create model
    agp_model = create_agp_model(
        args.embedding_size, args.hidden_size, args.n_glimpses, args.tanh_exploration, args.use_tanh, reward, args.temperature
    )

    # Use only args.train_size samples for training if available
    size = args.train_size
    small_train_dataset = train_dataset if len(train_dataset) <= size else Dataset(train_dataset.samples[:size])
    small_val_dataset = val_dataset if len(val_dataset) <= size else Dataset(val_dataset.samples[:size])
    
    # Define reward function - coverage_smooth_reward: NO discontinuity, stable for RL
    # coverage_weight: base weight for coverage^exponent term
    # guard_weight: penalty for guards (scaled by coverage)
    # coverage_exponent: higher = stronger emphasis on reaching 100% coverage
    reward_fn = partial(reward, coverage_weight=100.0, guard_weight=5.0, coverage_exponent=4.0)

    # Train with greedy baseline
    reinforce_train_greedy_baseline(agp_model, small_train_dataset, reward_fn, greedy_cache,
                                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, beta=args.beta)
    
    training_method = 'reinforcement_learning_greedy_baseline'

    # Save the trained model
    checkpoint_params = {
        'embedding_size': args.embedding_size,
        'hidden_size': args.hidden_size,
        'n_glimpses': args.n_glimpses,
        'tanh_exploration': args.tanh_exploration,
        'use_tanh': args.use_tanh,
        'beta': args.beta,
        'temperature': args.temperature,
        'greedy_baseline': True
    }
    
    checkpoint_path = get_checkpoint_path('checkpoints', 'rl_agp_greedy_model', checkpoint_params, args.epochs)
    model_checkpoint = {
        'model_state_dict': agp_model.state_dict(),
        'args': vars(args),
        'training_method': training_method,
        'num_train_samples': len(small_train_dataset),
        'num_val_samples': len(small_val_dataset),
        'greedy_cache_size': len(greedy_cache) if greedy_cache else 0
    }
    
    torch.save(model_checkpoint, checkpoint_path)
    print(f"\nModel saved to {checkpoint_path}")

    # Evaluate the model on the validation dataset
    eval_results = reinforce_eval(agp_model, small_val_dataset, reward_fn, batch_size=1, 
                                  sol_dir=args.agp_val_dir, greedy_cache=greedy_cache)
    
    # Save evaluation results
    import json
    results_summary = {
        'args': vars(args),
        'num_train_samples': len(small_train_dataset),
        'num_val_samples': len(small_val_dataset),
        'training_method': training_method,
        'greedy_cache_size': len(greedy_cache) if greedy_cache else 0
    }
    
    # Add statistics if available
    if 'stats' in eval_results and eval_results['stats']:
        if 'coverage_stats' in eval_results['stats']:
            results_summary['coverage_stats'] = eval_results['stats']['coverage_stats']
        if 'efficiency_stats' in eval_results['stats']:
            results_summary['efficiency_stats'] = eval_results['stats']['efficiency_stats']
        if 'ratio_stats' in eval_results['stats']:
            results_summary['size_ratio_stats'] = eval_results['stats']['ratio_stats']
        if 'coverage_vis_stats' in eval_results['stats']:
            results_summary['polygon_coverage_stats'] = eval_results['stats']['coverage_vis_stats']
    
    # Add greedy comparison results
    results_summary['greedy_comparisons'] = eval_results.get('greedy_comparisons', 0)
    results_summary['rl_vs_greedy_improvements'] = eval_results.get('rl_vs_greedy_improvements', 0)
    results_summary['avg_greedy_reward'] = eval_results.get('avg_greedy_reward', float('nan'))
    
    # Save to results directory
    os.makedirs('results', exist_ok=True)
    with open('results/rl_agp_greedy_evaluation.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print("\nResults summary saved to results/rl_agp_greedy_evaluation.json")


if __name__ == "__main__":
    main()
