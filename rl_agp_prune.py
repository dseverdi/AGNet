#!/usr/bin/env python3
"""
Subtractive (Pruning) RL approach for Art Gallery Problem.
Symmetric to rl_agp.py:
- Model outputs guards to REMOVE (vs rl_agp.py outputs guards to SELECT)
- One forward pass generates complete removal set
- One coverage check on final solution
- Same training efficiency as additive approach
"""

import os
from dotenv import load_dotenv
import argparse
from dataset import Dataset, agp_read_samples, collate_fn
from models import create_actor
from utils import evaluate_polygon_visibility_numpy_wo_gt
import torch
from torch.utils.data import DataLoader, Sampler
import numpy as np
import sys
from rewards import strict_reward as reward
from functools import partial
import json
import time
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    print("Warning: tqdm not available, progress bars will be disabled")

# use torch's nn via existing torch import
nn = torch.nn

# --- Utility ---
def get_checkpoint_path(folder, model_name, params, n_epochs):
    """Generate a checkpoint path based on model name, parameters, and epoch count."""
    param_str = "_".join([f"{k}{v}" for k, v in sorted(params.items())])
    filename = f"{model_name}_{param_str}_epochs{n_epochs}.pt"
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)

# --- Data Preparation ---
def prepare_datasets(train_path, val_path, normalize=True):
    """Load training and validation datasets."""
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
def create_pruning_model(embedding_size, hidden_size, n_glimpses, tanh_exploration, use_tanh, temperature):
    """Create pruning policy network (same architecture as rl_agp.py actor)."""
    return create_actor(
        embedding_size, hidden_size, None, n_glimpses,
        tanh_exploration, use_tanh, "Bahdanau", reward, temperature=temperature
    )

def get_lengths_from_dataset(dataset):
    """Extract polygon lengths from dataset."""
    return [len(polygon) for polygon, _, _ in dataset]

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
            total += (len(bucket) + self.batch_size - 1) // self.batch_size
        return total

# --- Training with REINFORCE (Pruning) - Symmetric to rl_agp.py ---
def reinforce_prune_train(model, dataset, reward_fn, epochs=10, batch_size=1, lr=1e-3, beta=0.99):
    """
    Train pruning policy using REINFORCE with EMA baseline (SYMMETRIC to rl_agp.py).
    
    Key difference:
    - rl_agp.py: Model SELECTS guards to include (additive)
    - rl_agp_prune.py: Model SELECTS guards to remove (subtractive)
    
    """
    print(f"\n--- Pruning Training on {len(dataset)} samples for {epochs} epochs (batch size {batch_size}) ---")
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lengths = get_lengths_from_dataset(dataset)
    batch_sampler = BucketBatchSampler(lengths, batch_size, shuffle=True, bucket_size=10)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, pin_memory=True, num_workers=2)
    device = next(model.parameters()).device
    baseline = 0.0  # EMA baseline
    
    epoch_iterator = tqdm(range(epochs), desc="Training Epochs") if tqdm else range(epochs)
    
    for epoch in epoch_iterator:
        total_loss = 0
        epoch_rewards = []
        
        batch_iterator = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} Batches", leave=False) if tqdm else loader
        
        for batch_data, mask, lengths_tensor, batch_names in batch_iterator:
            batch_data = batch_data.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            lengths_tensor = torch.tensor(lengths_tensor, dtype=torch.long, device=device)
            
            # Forward pass: model outputs guards to REMOVE (symmetric to rl_agp.py which outputs guards to SELECT)
            selected_idxs, log_probs = model(batch_data, padding_mask=mask, lengths=lengths_tensor)
            
            rewards_list = []
            for i, (data_tensor, removal_idxs, name) in enumerate(zip(batch_data.cpu(), selected_idxs, batch_names)):
                n = lengths_tensor[i].item()
                real_points = data_tensor[:n].detach().cpu().numpy()
                
                # Start with ALL guards (100% coverage guaranteed)
                all_guards = set(range(n))
                
                # Remove the guards selected by model
                removal_set = set(idx for idx in removal_idxs if idx < n)
                final_guards = list(all_guards - removal_set)
                
                # Compute reward ONCE on final solution
                # This is SYMMETRIC to rl_agp.py - same number of coverage checks!
                r = reward_fn(real_points, np.array(final_guards), name, length=n)
                rewards_list.append(r)
            
            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
            
            # Update EMA baseline and compute advantages
            batch_mean = rewards.mean().item()
            baseline = beta * baseline + (1 - beta) * batch_mean
            advantages = rewards - baseline
            
            # REINFORCE loss using advantages
            loss = -(log_probs * advantages).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_data.size(0)
            epoch_rewards.extend(rewards_list)
            
            # Update tqdm postfix
            if tqdm and hasattr(batch_iterator, 'set_postfix'):
                batch_iterator.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'reward': f"{batch_mean:.3f}",
                    'baseline': f"{baseline:.3f}"
                })
            
            # Memory cleanup
            del batch_data, mask, lengths_tensor, selected_idxs, log_probs, rewards
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        avg_loss = total_loss / len(dataset)
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0.0
        print(f"Epoch {epoch+1}/{epochs} - Avg loss: {avg_loss:.4f}, Avg reward: {avg_reward:.4f}")
        torch.cuda.empty_cache()
    
    print("Training done!")

# --- Evaluation ---
def reinforce_prune_eval(model, dataset, reward_fn, batch_size=1, sol_dir=None):
    """
    Evaluate the pruning policy (symmetric to rl_agp.py evaluation).
    Model generates removal sequence, evaluate final solution only.
    """
    print(f"\n--- Evaluating Pruning Policy on {len(dataset)} samples ---")
    
    model.eval()
    device = next(model.parameters()).device
    
    all_rewards = []
    all_coverages = []
    all_guard_counts = []
    all_initial_guards = []
    guard_ratios = []
    polygon_sizes = []
    pred_sizes = []
    true_sizes = []
    coverage_ratios = []
    efficiency_ratios = []
    size_ratios = []
    overlap_counts = []
    optimal_solutions = {}
    
    solutions = {}
    
    start_time = time.time()
    
    with torch.no_grad():
        pbar = tqdm(range(len(dataset)), desc="Evaluating") if tqdm else range(len(dataset))
        for idx in pbar:
            polygon_tensor, _, name = dataset[idx]
            polygon = polygon_tensor.unsqueeze(0).to(device)
            length = torch.tensor([len(polygon_tensor)], dtype=torch.long, device=device)
            n_vertices = length.item()
            
            # Start with all guards
            all_guards = set(range(n_vertices))
            all_initial_guards.append(n_vertices)
            
            # ONE forward pass - get removal sequence
            polygon_input = polygon  # Use full polygon as input
            mask = torch.zeros(1, n_vertices, dtype=torch.bool, device=device)
            removal_idxs, _ = model(polygon_input, padding_mask=mask, lengths=length)
            
            # Apply ALL removals at once
            removal_set = set(idx for idx in removal_idxs[0] if idx < n_vertices)
            final_guards = list(all_guards - removal_set)
            
            # Compute coverage and reward ONCE on final solution
            final_solution = np.array(final_guards)
            points = polygon[0, :n_vertices].cpu().numpy()
            
            final_coverage = evaluate_polygon_visibility_numpy_wo_gt(points, final_solution, name)
            final_reward = reward_fn(points, final_solution, name, length=n_vertices)
            
            all_rewards.append(final_reward)
            all_coverages.append(final_coverage)
            all_guard_counts.append(len(final_guards))
            
            solutions[name] = final_solution.tolist()
            polygon_sizes.append(n_vertices)
            pred_sizes.append(len(final_guards))
            guard_ratios.append(len(final_guards) / n_vertices if n_vertices > 0 else 0.0)
            
            # Update progress bar
            if tqdm and hasattr(pbar, 'set_postfix') and (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                pbar.set_postfix({
                    'avg_time': f'{avg_time:.2f}s',
                    'coverage': f'{np.mean(all_coverages):.3f}',
                    'guards': f'{np.mean(all_guard_counts):.1f}'
                })
            
            # Optional optimal comparison
            if sol_dir:
                base_name = os.path.splitext(os.path.basename(name))[0]
                opt_sol_path = os.path.join(sol_dir, f"{base_name}.solution")
                try:
                    with open(opt_sol_path, 'r') as f:
                        lines = f.read().splitlines()
                    if len(lines) >= 2:
                        opt_indices = [int(x) for x in lines[1].split() if x.strip()]
                        opt_set = set(opt_indices)
                        if opt_set:
                            optimal_solutions[name] = opt_indices
                            true_sizes.append(len(opt_set))
                            pred_set = set(final_guards)
                            overlap = len(pred_set & opt_set)
                            overlap_counts.append(overlap)
                            coverage_ratios.append(overlap / len(opt_set))
                            efficiency_ratios.append(overlap / len(pred_set) if pred_set else 0.0)
                            size_ratios.append(len(pred_set) / len(opt_set))
                except Exception:
                    continue
    
    total_time = time.time() - start_time
    print(f"\nTotal evaluation time: {total_time/60:.1f} minutes ({total_time/len(dataset):.2f}s per sample)")
    
    def compute_stats(data, name):
        data = np.array(data, dtype=np.float64)
        if len(data) == 0:
            print(f"\n{name} Statistics: No data available.")
            return {key: float('nan') for key in ['mean', 'median', 'std', 'min', 'max', 'q25', 'q75', 'iqr']}
        stats = {
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75)),
            'iqr': float(np.percentile(data, 75) - np.percentile(data, 25))
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
    size_stats = compute_stats(pred_sizes, "Predicted Solution Sizes") if pred_sizes else {}
    optimal_stats = compute_stats(true_sizes, "Optimal Solution Sizes") if true_sizes else {}
    coverage_stats = compute_stats(coverage_ratios, "Guard Set Coverage Ratios (fraction of optimal guards covered by predicted guards)") if coverage_ratios else {}
    efficiency_stats = compute_stats(efficiency_ratios, "Efficiency Ratios (fraction of predicted guards that are optimal)") if efficiency_ratios else {}
    ratio_stats = compute_stats(size_ratios, "Size Ratios (predicted/optimal)") if size_ratios else {}
    overlap_stats = compute_stats(overlap_counts, "Overlap Counts (absolute number of matching guards)") if overlap_counts else {}
    guard_count_stats = compute_stats(all_guard_counts, "Predicted Guard Counts") if all_guard_counts else {}
    guard_ratio_stats = compute_stats(guard_ratios, "Guard Ratios (guards / vertices)") if guard_ratios else {}
    reward_stats = compute_stats(all_rewards, "Reward Distribution") if all_rewards else {}
    
    coverage_array = np.array([c for c in all_coverages if not np.isnan(c)], dtype=np.float64)
    coverage_vis_stats = compute_stats(coverage_array, "Polygon Coverage (visibility)") if len(coverage_array) > 0 else {}
    
    if coverage_ratios:
        print(f"\n=== SUMMARY ===")
        print(f"Instances evaluated: {len(pred_sizes)}")
        print(f"Perfect solutions (100% coverage): {sum(1 for c in coverage_ratios if c >= 1.0)}")
        print(f"Good solutions (>=80% coverage): {sum(1 for c in coverage_ratios if c >= 0.8)}")
        print(f"Reasonable solutions (>=60% coverage): {sum(1 for c in coverage_ratios if c >= 0.6)}")
        print(f"Average coverage ratio: {np.mean(coverage_ratios):.3f}")
        if size_ratios:
            print(f"Average size inflation: {np.mean(size_ratios):.2f}x optimal")
    else:
        print("No optimal solutions available for comparison.")
    
    stats = {
        'evaluation_time': float(total_time),
        'num_samples': len(dataset),
        'size_stats': size_stats,
        'optimal_stats': optimal_stats,
        'coverage_stats': coverage_stats,
        'efficiency_stats': efficiency_stats,
        'ratio_stats': ratio_stats,
        'overlap_stats': overlap_stats,
        'coverage_vis_stats': coverage_vis_stats,
        'guard_count_stats': guard_count_stats,
        'guard_ratio_stats': guard_ratio_stats,
        'reward_stats': reward_stats
    }
    
    return {
        'stats': stats,
        'solutions': solutions,
        'optimal_solutions': optimal_solutions,
        'rewards': all_rewards,
        'coverages': all_coverages,
        'guard_counts': all_guard_counts,
        'guard_ratios': guard_ratios,
        'pred_sizes': pred_sizes,
        'true_sizes': true_sizes,
        'coverage_ratios': coverage_ratios,
        'efficiency_ratios': efficiency_ratios,
        'size_ratios': size_ratios,
        'overlap_counts': overlap_counts,
        'polygon_sizes': polygon_sizes,
        'evaluation_time': float(total_time)
    }

# --- Main ---
def main():
    load_dotenv()
    DATASET_PATH = os.getenv("DATASET_PATH")
    if not DATASET_PATH:
        raise EnvironmentError("DATASET_PATH environment variable must be set in .env file.")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--n-glimpses', type=int, default=1)
    parser.add_argument('--tanh-exploration', type=float, default=10)
    parser.add_argument('--use-tanh', action='store_true', default=True)
    parser.add_argument('--beta', type=float, default=0.99)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=1.0)
    default_train = os.path.join(DATASET_PATH, "train")
    default_val = os.path.join(DATASET_PATH, "dev")
    parser.add_argument('--agp_train_dir', type=str, default=default_train)
    parser.add_argument('--agp_val_dir', type=str, default=default_val)
    parser.add_argument('--train-size', type=int, default=8000, help="Number of training samples to use (default: 8000, or all if smaller)")
    
    args = parser.parse_args()

    train_dataset, val_dataset = prepare_datasets(args.agp_train_dir, args.agp_val_dir, normalize=True)
    
    pruning_model = create_pruning_model(
        args.embedding_size, args.hidden_size, args.n_glimpses, args.tanh_exploration, args.use_tanh, args.temperature
    )

    # Use only args.train_size samples for training if available
    size = args.train_size
    small_train_dataset = train_dataset if len(train_dataset) <= size else Dataset(train_dataset.samples[:size])
    small_val_dataset = val_dataset if len(val_dataset) <= size else Dataset(val_dataset.samples[:size])
    
    # Define reward function
    reward_fn = partial(reward, alpha=1.0, M=1000.0)

    # Train model
    reinforce_prune_train(pruning_model, small_train_dataset, reward_fn,
                        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, beta=args.beta)
    training_method = 'pruning_reinforcement_learning'

    # Save the trained model
    checkpoint_params = {
        'embedding_size': args.embedding_size,
        'hidden_size': args.hidden_size,
        'n_glimpses': args.n_glimpses,
        'tanh_exploration': args.tanh_exploration,
        'use_tanh': args.use_tanh,
        'beta': args.beta,
        'temperature': args.temperature
    }
    
    checkpoint_path = get_checkpoint_path('checkpoints', 'rl_agp_prune_model', checkpoint_params, args.epochs)
    model_checkpoint = {
        'model_state_dict': pruning_model.state_dict(),
        'args': vars(args),
        'training_method': training_method,
        'num_train_samples': len(small_train_dataset),
        'num_val_samples': len(small_val_dataset)
    }
    
    torch.save(model_checkpoint, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # Evaluate the model on the validation dataset
    eval_results = reinforce_prune_eval(pruning_model, small_val_dataset, reward_fn, batch_size=1, sol_dir=args.agp_val_dir)
    
    # Save evaluation results
    results_summary = {
        'args': vars(args),
        'num_train_samples': len(small_train_dataset),
        'num_val_samples': len(small_val_dataset),
        'training_method': training_method
    }
    
    # Add statistics if available
    if 'stats' in eval_results and eval_results['stats']:
        stats = eval_results['stats']
        if 'coverage_stats' in stats:
            results_summary['coverage_stats'] = stats['coverage_stats']
        if 'efficiency_stats' in stats:
            results_summary['efficiency_stats'] = stats['efficiency_stats']
        if 'ratio_stats' in stats:
            results_summary['size_ratio_stats'] = stats['ratio_stats']
        if 'coverage_vis_stats' in stats:
            results_summary['polygon_coverage_stats'] = stats['coverage_vis_stats']
        if 'guard_count_stats' in stats:
            results_summary['guard_count_stats'] = stats['guard_count_stats']
        if 'guard_ratio_stats' in stats:
            results_summary['guard_ratio_stats'] = stats['guard_ratio_stats']
        if 'reward_stats' in stats:
            results_summary['reward_stats'] = stats['reward_stats']
        results_summary['evaluation_time'] = stats.get('evaluation_time', eval_results.get('evaluation_time'))
    
    # Save to results directory
    os.makedirs('results', exist_ok=True)
    with open('results/rl_agp_prune_evaluation.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print("Results summary saved to results/rl_agp_prune_evaluation.json")


if __name__ == "__main__":
    main()
