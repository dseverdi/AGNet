#!/usr/bin/env python3
"""
Subtractive (Pruning) RL approach for Art Gallery Problem - Version 2.
Optimized to match rl_agp.py efficiency:
- One forward pass per polygon generates full removal sequence
- One coverage check per polygon (on final solution)
- Proper batching
"""

import os
from dotenv import load_dotenv
import argparse
from dataset import Dataset, agp_read_samples, collate_fn
from utils import evaluate_polygon_visibility_numpy_wo_gt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
import numpy as np
import sys
import matplotlib.pyplot as plt
from rewards import strict_reward as reward
from functools import partial
import json
from models import create_actor
import time
from tqdm import tqdm

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
        buckets = self.buckets.copy()
        if self.shuffle:
            np.random.shuffle(buckets)
        
        for bucket in buckets:
            if self.shuffle:
                np.random.shuffle(bucket)
            
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
    """Extract lengths from dataset."""
    return [sample[0].shape[0] for sample in dataset]


# --- Pruning Policy Network ---
def create_pruning_model(embedding_size, hidden_size, n_glimpses, tanh_exploration, use_tanh, temperature):
    """Create actor model for pruning (same architecture as additive approach)."""
    return create_actor(
        embedding_size, hidden_size, None, n_glimpses,
        tanh_exploration, use_tanh, "Bahdanau", reward, temperature=temperature
    )


# --- Training with REINFORCE (Pruning) - Optimized ---
def reinforce_prune_train(model, dataset, reward_fn, epochs=10, batch_size=1, lr=1e-3, 
                         beta=0.99, min_guards=3, coverage_threshold=0.99):
    """
    Train pruning policy - OPTIMIZED version matching rl_agp.py efficiency.
    Key idea: Model generates removal sequence in ONE forward pass,
    then we validate the final solution.
    """
    print(f"\n--- Pruning Training on {len(dataset)} samples for {epochs} epochs (batch size {batch_size}) ---")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    lengths = get_lengths_from_dataset(dataset)
    batch_sampler = BucketBatchSampler(lengths, batch_size, shuffle=True, bucket_size=10)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, 
                       pin_memory=True, num_workers=2)
    
    device = next(model.parameters()).device
    baseline = 0.0  # EMA baseline
    
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        epoch_loss = 0.0
        epoch_rewards = []
        epoch_removals = []
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch_data, mask, lengths_tensor, batch_names in pbar:
            batch_data = batch_data.to(device, non_blocking=True)
            lengths_tensor = torch.tensor(lengths_tensor, dtype=torch.long, device=device)
            
            batch_loss = 0.0
            batch_rewards_list = []
            
            # Process each polygon - similar to rl_agp.py
            for b_idx in range(batch_data.size(0)):
                polygon = batch_data[b_idx:b_idx+1]
                length = lengths_tensor[b_idx:b_idx+1]
                name = batch_names[b_idx]
                n_vertices = length.item()
                
                # Start with all vertices as guards
                all_guards = list(range(n_vertices))
                
                # Create input: coordinates of ALL guards
                guard_coords = polygon[0, all_guards, :].unsqueeze(0)  # [1, n_vertices, 2]
                guard_mask = torch.ones(1, n_vertices, dtype=torch.bool, device=device)
                
                # ONE forward pass - model generates removal sequence
                removal_sequence, log_probs_sum = model(guard_coords, padding_mask=guard_mask, lengths=length)
                
                # Apply removals from the sequence
                removal_idxs = removal_sequence[0]  # List of indices to remove
                final_guards = all_guards.copy()
                valid_removals = 0
                
                points = polygon[0, :n_vertices].cpu().numpy()
                
                # Apply each removal if it maintains coverage
                for rem_idx in removal_idxs:
                    if rem_idx >= len(final_guards) or len(final_guards) <= min_guards:
                        break  # EOS or min guards reached
                    
                    guard_to_remove = final_guards[rem_idx]
                    candidate_guards = [g for g in final_guards if g != guard_to_remove]
                    
                    # Check coverage (this is the bottleneck, but only done once per removal)
                    coverage = evaluate_polygon_visibility_numpy_wo_gt(points, np.array(candidate_guards), name)
                    
                    if coverage >= coverage_threshold:
                        final_guards = candidate_guards
                        valid_removals += 1
                    else:
                        break  # Can't remove more without violating coverage
                
                # Compute reward on FINAL solution
                r = reward_fn(points, np.array(final_guards), name, length=n_vertices)
                
                # Update baseline
                baseline = beta * baseline + (1 - beta) * r
                
                # Compute advantage
                advantage = r - baseline
                
                # Policy gradient loss
                loss = -(log_probs_sum[0] * advantage)
                
                batch_loss += loss
                batch_rewards_list.append(r)
                epoch_removals.append(valid_removals)
            
            # Backward pass
            if batch_loss != 0:
                optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += batch_loss.item()
                epoch_rewards.extend(batch_rewards_list)
            
            # Update progress bar
            if epoch_rewards:
                pbar.set_postfix({
                    'loss': f'{epoch_loss / max(1, len(epoch_rewards)):.4f}',
                    'reward': f'{np.mean(epoch_rewards):.4f}',
                    'removals': f'{np.mean(epoch_removals):.1f}'
                })
            
            # Cleanup
            torch.cuda.empty_cache()
        
        # Epoch summary
        avg_loss = epoch_loss / len(dataset) if len(dataset) > 0 else 0
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        avg_removals = np.mean(epoch_removals) if epoch_removals else 0
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, "
              f"Avg Reward: {avg_reward:.4f}, Avg Removals: {avg_removals:.2f}, Baseline: {baseline:.4f}")
    
    print("Training complete!")


# --- Evaluation ---
def evaluate_pruning_policy(model, dataset, reward_fn, batch_size=1, min_guards=3, 
                           coverage_threshold=0.99, sol_dir=None):
    """
    Evaluate the pruning policy - OPTIMIZED.
    Model generates removal sequence, we trust it and only check final coverage.
    """
    print(f"\n--- Evaluating Pruning Policy on {len(dataset)} samples ---")
    model.eval()
    
    device = next(model.parameters()).device
    
    all_rewards = []
    all_coverages = []
    all_guard_counts = []
    all_removal_counts = []
    all_initial_guards = []
    
    solutions = {}
    
    start_time = time.time()
    
    with torch.no_grad():
        pbar = tqdm(range(len(dataset)), desc="Evaluating")
        for idx in pbar:
            polygon_tensor, _, name = dataset[idx]
            polygon = polygon_tensor.unsqueeze(0).to(device)
            length = torch.tensor([len(polygon_tensor)], dtype=torch.long, device=device)
            n_vertices = length.item()
            
            # Start with all guards
            all_guards = list(range(n_vertices))
            all_initial_guards.append(n_vertices)
            
            # ONE forward pass - get removal sequence
            guard_coords = polygon[0, all_guards, :].unsqueeze(0)
            guard_mask = torch.ones(1, n_vertices, dtype=torch.bool, device=device)
            removal_sequence, _ = model(guard_coords, padding_mask=guard_mask, lengths=length)
            
            # Apply removals (trust the model, don't check coverage each step)
            removal_idxs = removal_sequence[0]
            final_guards = all_guards.copy()
            removals = 0
            
            for rem_idx in removal_idxs:
                if rem_idx >= len(final_guards) or len(final_guards) <= min_guards:
                    break
                guard_to_remove = final_guards[rem_idx]
                final_guards = [g for g in final_guards if g != guard_to_remove]
                removals += 1
            
            # NOW compute coverage only ONCE on final solution
            final_solution = np.array(final_guards)
            points = polygon[0, :n_vertices].cpu().numpy()
            
            final_coverage = evaluate_polygon_visibility_numpy_wo_gt(points, final_solution, name)
            final_reward = reward_fn(points, final_solution, name, length=n_vertices)
            
            all_rewards.append(final_reward)
            all_coverages.append(final_coverage)
            all_guard_counts.append(len(final_guards))
            all_removal_counts.append(removals)
            
            solutions[name] = final_solution.tolist()
            
            # Update progress bar
            if (idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                pbar.set_postfix({
                    'avg_time': f'{avg_time:.2f}s',
                    'coverage': f'{np.mean(all_coverages):.3f}',
                    'guards': f'{np.mean(all_guard_counts):.1f}'
                })
    
    total_time = time.time() - start_time
    print(f"\nTotal evaluation time: {total_time/60:.1f} minutes ({total_time/len(dataset):.2f}s per sample)")
    
    # Compute statistics
    stats = {
        'mean_reward': float(np.mean(all_rewards)),
        'std_reward': float(np.std(all_rewards)),
        'mean_coverage': float(np.mean(all_coverages)),
        'std_coverage': float(np.std(all_coverages)),
        'mean_guards': float(np.mean(all_guard_counts)),
        'std_guards': float(np.std(all_guard_counts)),
        'mean_removals': float(np.mean(all_removal_counts)),
        'std_removals': float(np.std(all_removal_counts)),
        'mean_initial_guards': float(np.mean(all_initial_guards)),
        'mean_guard_ratio': float(np.mean([g/n for g, n in zip(all_guard_counts, all_initial_guards)])),
        'num_samples': len(dataset)
    }
    
    print("\nEvaluation Results:")
    print(f"  Mean Coverage: {stats['mean_coverage']:.4f} ± {stats['std_coverage']:.4f}")
    print(f"  Mean Guards: {stats['mean_guards']:.2f} ± {stats['std_guards']:.2f}")
    print(f"  Mean Removals: {stats['mean_removals']:.2f} ± {stats['std_removals']:.2f}")
    print(f"  Mean Guard Ratio (guards/vertices): {stats['mean_guard_ratio']:.4f}")
    print(f"  Mean Reward: {stats['mean_reward']:.4f} ± {stats['std_reward']:.4f}")
    
    # Report approximation ratio if optimal solutions available
    if sol_dir:
        optimal_ratios = []
        for idx in range(len(dataset)):
            _, _, name = dataset[idx]
            base_name = os.path.splitext(os.path.basename(name))[0]
            opt_sol_path = os.path.join(sol_dir, f"{base_name}.solution")
            try:
                with open(opt_sol_path, 'r') as f:
                    lines = f.read().splitlines()
                    if len(lines) >= 2:
                        opt_guards = len([int(x) for x in lines[1].split()])
                        pred_guards = all_guard_counts[idx]
                        if opt_guards > 0:
                            optimal_ratios.append(pred_guards / opt_guards)
            except Exception:
                continue
        
        if optimal_ratios:
            stats['mean_approx_ratio'] = float(np.mean(optimal_ratios))
            stats['std_approx_ratio'] = float(np.std(optimal_ratios))
            print(f"  Mean Approximation Ratio (pred/optimal): {stats['mean_approx_ratio']:.4f} ± {stats['std_approx_ratio']:.4f}")
    
    return {
        'stats': stats,
        'solutions': solutions,
        'rewards': all_rewards,
        'coverages': all_coverages,
        'guard_counts': all_guard_counts
    }


# --- Main ---
def main():
    load_dotenv()
    DATASET_PATH = os.getenv("DATASET_PATH")
    if not DATASET_PATH:
        raise EnvironmentError("DATASET_PATH environment variable must be set in .env file.")
    
    parser = argparse.ArgumentParser(description='Pruning-based RL for AGP (Optimized)')
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--n-glimpses', type=int, default=1)
    parser.add_argument('--tanh-exploration', type=float, default=10)
    parser.add_argument('--use-tanh', action='store_true', default=True)
    parser.add_argument('--beta', type=float, default=0.99, help='EMA baseline decay')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=1.0)
    default_train = os.path.join(DATASET_PATH, "train")
    default_val = os.path.join(DATASET_PATH, "dev")
    parser.add_argument('--agp-train-dir', type=str, default=default_train)
    parser.add_argument('--agp-val-dir', type=str, default=default_val)
    parser.add_argument('--train-size', type=int, default=8000, 
                       help='Number of training samples to use')
    parser.add_argument('--min-guards', type=int, default=3, 
                       help='Minimum guards to keep')
    parser.add_argument('--coverage-threshold', type=float, default=0.99,
                       help='Minimum coverage threshold')
    parser.add_argument('--eval-size', type=int, default=None,
                       help='Number of validation samples to evaluate (default: all)')
    
    args = parser.parse_args()
    
    # Load datasets
    train_dataset, val_dataset = prepare_datasets(args.agp_train_dir, args.agp_val_dir, normalize=True)
    
    # Limit training size
    if len(train_dataset) > args.train_size:
        train_dataset = Dataset(train_dataset.samples[:args.train_size])
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_pruning_model(
        args.embedding_size, args.hidden_size, args.n_glimpses,
        args.tanh_exploration, args.use_tanh, args.temperature
    )
    model = model.to(device)
    
    print(f"\nPruning Policy Network (optimized - same arch as rl_agp.py):")
    print(f"  Embedding size: {args.embedding_size}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  N-glimpses: {args.n_glimpses}")
    print(f"  Tanh exploration: {args.tanh_exploration}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Define reward function
    reward_fn = partial(reward, alpha=1.0, M=1000.0)
    
    # Check if loading pre-trained model
    checkpoint_params = {
        'embedding_size': args.embedding_size,
        'hidden_size': args.hidden_size,
        'n_glimpses': args.n_glimpses,
        'tanh_exploration': args.tanh_exploration,
        'use_tanh': args.use_tanh,
        'beta': args.beta,
        'temperature': args.temperature
    }
    checkpoint_path = get_checkpoint_path('checkpoints', 'rl_agp_prune_v2', checkpoint_params, args.epochs)
    
    if args.epochs == 0 and os.path.exists(checkpoint_path):
        print(f"\nLoading pre-trained model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Pre-trained model loaded successfully")
    elif args.epochs > 0:
        # Train model
        reinforce_prune_train(
            model, train_dataset, reward_fn,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            beta=args.beta,
            min_guards=args.min_guards,
            coverage_threshold=args.coverage_threshold
        )
        
        # Save model
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'args': vars(args),
            'training_method': 'pruning_reinforce_optimized'
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"\nModel saved to {checkpoint_path}")
    
    # Evaluate
    print("\n" + "="*60)
    
    # Optionally limit evaluation set size
    eval_dataset = val_dataset
    if args.eval_size is not None and args.eval_size < len(val_dataset):
        print(f"Evaluating on subset: {args.eval_size} / {len(val_dataset)} samples")
        eval_dataset = Dataset(val_dataset.samples[:args.eval_size])
    
    eval_results = evaluate_pruning_policy(
        model, eval_dataset, reward_fn,
        batch_size=1,
        min_guards=args.min_guards,
        coverage_threshold=args.coverage_threshold,
        sol_dir=args.agp_val_dir
    )
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_summary = {
        'args': vars(args),
        'stats': eval_results['stats'],
        'training_method': 'pruning_reinforce_optimized',
        'num_train_samples': len(train_dataset),
        'num_val_samples': len(eval_dataset)
    }
    
    with open('results/rl_agp_prune_v2_evaluation.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print("\nResults saved to results/rl_agp_prune_v2_evaluation.json")


if __name__ == "__main__":
    main()
