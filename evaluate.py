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
import time
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
from train_value_net import SolutionValueNet, MultiTaskValueNet, solution_collate_fn
from train_ranker import RankerNet
from torch.nn.utils.rnn import pad_sequence
from functools import partial


def _flatten_lstm_parameters(module: torch.nn.Module):
    """Recursively call flatten_parameters() on all LSTM modules.
    Helps avoid PyTorch RNN contiguity warnings and improves performance.
    Call this after moving the module to its device and after deepcopy/load_state.
    """
    for m in module.modules():
        if isinstance(m, torch.nn.LSTM):
            try:
                m.flatten_parameters()
            except Exception:
                # Safe to ignore; PyTorch will still run, just less optimal
                pass

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


def load_reward_predictor(model_path, embedding_size=128, hidden_size=None):
    """
    Load either a ranker network or value network for reward prediction.
    Automatically detects the model type and hidden size from filename.
    
    Args:
        model_path: Path to the model checkpoint
        embedding_size: Embedding size (default: 128)  
        hidden_size: Hidden size (auto-detected if None)
        
    Returns:
        (model, device, model_info): Loaded model, device, and info dict
    """
    print(f"🔧 Loading reward predictor from: {model_path}")
    
    # Auto-detect hidden size from filename if not provided
    if hidden_size is None:
        import re
        match = re.search(r'hidden_size(\d+)', os.path.basename(model_path))
        if match:
            hidden_size = int(match.group(1))
            print(f"  Auto-detected hidden_size: {hidden_size}")
        else:
            hidden_size = 256  # Default fallback
            print(f"  Using default hidden_size: {hidden_size}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(model_path, map_location=device)
    
    # Determine model type
    model_type = 'unknown'
    if 'ranker' in os.path.basename(model_path).lower():
        model_type = 'ranker'
    elif 'value_net' in os.path.basename(model_path).lower():
        model_type = 'value_net'
    
    # Load model based on checkpoint format
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
        md = ckpt.get('metadata', {})
        detected_type = md.get('model_type', model_type)
        
        if detected_type == 'multitask':
            from train_value_net import MultiTaskValueNet
            model = MultiTaskValueNet(embedding_size=embedding_size, hidden_size=hidden_size)
            model_type = 'multitask_value_net'
        elif detected_type == 'solution':
            from train_value_net import SolutionValueNet
            model = SolutionValueNet(embedding_size=embedding_size, hidden_size=hidden_size)
            model_type = 'solution_value_net'
        elif detected_type == 'ranker' or model_type == 'ranker':
            model = RankerNet(embedding_size=embedding_size, hidden_size=hidden_size)
            model_type = 'ranker'
        else:
            # Fallback
            from train_value_net import SolutionValueNet
            model = SolutionValueNet(embedding_size=embedding_size, hidden_size=hidden_size)
            model_type = 'solution_value_net'
        
        model.load_state_dict(state, strict=False)
    else:
        # Legacy format - guess from filename
        if model_type == 'ranker':
            model = RankerNet(embedding_size=embedding_size, hidden_size=hidden_size)
        else:
            from train_value_net import SolutionValueNet
            model = SolutionValueNet(embedding_size=embedding_size, hidden_size=hidden_size)
            model_type = 'solution_value_net'
        model.load_state_dict(ckpt)
    
    model = model.to(device)
    model.eval()
    
    model_info = {
        'type': model_type,
        'embedding_size': embedding_size,
        'hidden_size': hidden_size,
        'path': model_path
    }
    
    print(f"✓ {model_type} loaded successfully")
    print(f"  Architecture: embedding_size={embedding_size}, hidden_size={hidden_size}")
    print(f"  Device: {device}")
    
    return model, device, model_info


def predict_rewards(model, polygons, solutions, poly_lengths, sol_lengths):
    """
    Unified function to predict rewards using any model type (ranker, value_net, etc.)
    
    Args:
        model: Loaded model (ranker or value network)
        polygons: Batch of polygon coordinates
        solutions: Batch of solution indices  
        poly_lengths: Polygon lengths
        sol_lengths: Solution lengths
        
    Returns:
        tensor: Predicted rewards/scores
    """
    with torch.no_grad():
        if hasattr(model, 'predict_reward'):
            preds = model.predict_reward(polygons, solutions, poly_lengths, sol_lengths)
        else:
            preds = model(polygons, solutions, poly_lengths, sol_lengths)
            # Handle dict output from MultiTaskValueNet
            if isinstance(preds, dict):
                preds = preds['reward']  # Use reward prediction
    
    return preds
    
    if not checkpoints:
        return None
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def find_best_ranker_checkpoint(checkpoints_dir='checkpoints'):
    """
    Automatically find the best ranker network checkpoint in the checkpoints directory.
    Looks for files matching pattern: ranker_net_*_best.pt
    """
    import glob
    pattern = os.path.join(checkpoints_dir, 'ranker_net_*_best.pt')
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        return None
    
    # Sort by modification time (newest first)
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def find_best_rl_checkpoint(checkpoints_dir='checkpoints'):
    """
    Automatically find the best RL model checkpoint in the checkpoints directory.
    Looks for files matching pattern: rl_agp_model_*_epochs*.pt
    """
    import glob
    pattern = os.path.join(checkpoints_dir, 'rl_agp_model_*_epochs*.pt')
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
    ckpt = torch.load(value_net_path, map_location=device)
    # Backward compatibility: raw state_dict vs checkpoint with metadata
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
        md = ckpt.get('metadata', {})
        # Try to infer model type by metadata; choose based on marker
        model_type = md.get('model_type', 'ranker')
        if model_type == 'multitask':
            from train_value_net import MultiTaskValueNet
            model = MultiTaskValueNet(embedding_size=embedding_size, hidden_size=hidden_size)
        elif model_type == 'solution':
            from train_value_net import SolutionValueNet
            model = SolutionValueNet(embedding_size=embedding_size, hidden_size=hidden_size)
        elif model_type == 'ranker':
            model = RankerNet(embedding_size=embedding_size, hidden_size=hidden_size)
        else:
            # Fallback: default to SolutionValueNet
            from train_value_net import SolutionValueNet
            model = SolutionValueNet(embedding_size=embedding_size, hidden_size=hidden_size)
        model.load_state_dict(state, strict=False)
        # Attach helper for predict_reward if missing
        model.predict_reward = getattr(model, 'predict_reward', None)
    else:
        from train_value_net import SolutionValueNet
        model = SolutionValueNet(embedding_size=embedding_size, hidden_size=hidden_size)
        model.load_state_dict(ckpt)
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
            solution = selected_idxs[0]  # Get first (and only) batch item
            
            # Convert to numpy if it's a tensor, otherwise keep as is
            if hasattr(solution, 'cpu'):
                solution = solution.cpu().numpy()
            elif isinstance(solution, (list, tuple)):
                solution = np.array(solution)
                
            # Filter to valid indices
            solution = sorted([idx for idx in solution if idx < length])
            solutions.append(solution)
        
        # Batch predict rewards using value network (FAST!) with empty-solution safety
        if len(solutions) > 0:
            K_solutions = len(solutions)
            predicted_rewards = np.full(K_solutions, np.inf, dtype=float)
            valid_idxs = [i for i, sol in enumerate(solutions) if len(sol) > 0]
            if len(valid_idxs) > 0:
                # Prepare batch data only for non-empty solutions
                batch_polygons = data_tensor.repeat(len(valid_idxs), 1, 1)  # [K_valid, max_len, 2]
                batch_solutions = pad_sequence(
                    [torch.tensor(solutions[i], dtype=torch.long) for i in valid_idxs],
                    batch_first=True,
                    padding_value=0,
                ).to(device)
                poly_lengths = torch.tensor([length] * len(valid_idxs), dtype=torch.long, device=device)
                sol_lengths = torch.tensor([len(solutions[i]) for i in valid_idxs], dtype=torch.long, device=device)

                # Get predicted rewards (support MultiTaskValueNet)
                if hasattr(value_net, 'predict_reward'):
                    preds = value_net.predict_reward(batch_polygons, batch_solutions, poly_lengths, sol_lengths)
                else:
                    preds = value_net(batch_polygons, batch_solutions, poly_lengths, sol_lengths)
                    # Handle different model types: dict output (MultiTaskValueNet) vs tensor output
                    if isinstance(preds, dict):
                        preds = preds['reward']  # Use reward for evaluation
                preds = preds.detach().cpu().numpy()
                for j, i in enumerate(valid_idxs):
                    predicted_rewards[i] = float(preds[j])

            # Pick best solution according to value network (empties remain +inf)
            best_idx = int(np.argmin(predicted_rewards))
            best_solution = solutions[best_idx]
            best_pred_reward = float(predicted_rewards[best_idx])
            
            # Compute actual reward for validation
            real_points = data_tensor[0, :length].cpu().numpy()
            reward_func = partial(reward_fn, alpha=5.0, p=0.0)
            actual_reward = reward_func(real_points, best_solution, name, length=length)
            
            # Compute coverage
            try:
                coverage = evaluate_polygon_visibility_numpy_wo_gt(real_points, best_solution, name)
            except Exception:
                coverage = 0.0
            
            # Calculate statistics (compute reward variance from all predictions)
            reward_variance = np.var(predicted_rewards[predicted_rewards != np.inf]) if np.sum(predicted_rewards != np.inf) > 1 else 0.0
            
            # Calculate prediction accuracy metrics
            prediction_error = abs(best_pred_reward - actual_reward)
            best_size = len(best_solution)
            size_ratio = best_size / max(1, length) if length > 0 else 1.0
            
            return {
                'best_reward': actual_reward,
                'best_coverage': coverage,
                'best_size': best_size,
                'size_ratio': size_ratio,
                'reward_variance': reward_variance,
                'coverage_variance': 0.0,  # Not computed for fast evaluation
                'pred_err_selected': prediction_error,
                'pred_err_opt': prediction_error,  # Same since we only evaluate selected solution
                'regret': 0.0,  # Would need to compute actual rewards for all K solutions
                'correlation': 0.0  # Would need actual rewards for all K solutions
            }
        else:
            # Fallback if no valid solutions
            return {
                'best_reward': float('inf'),
                'best_coverage': 0.0,
                'best_size': 0,
                'size_ratio': 0.0,
                'reward_variance': 0.0,
                'coverage_variance': 0.0,
                'pred_err_selected': 0.0,
                'pred_err_opt': 0.0,
                'regret': 0.0,
                'correlation': 0.0
            }


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
    _flatten_lstm_parameters(model)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    return model, device


def active_search_train_on_instance(model, data_tensor, mask, length, name, K=64, B=8, lr=3e-4, alpha=0.9, device='cpu', no_geometry=False):
    """
    Bello et al. Active Search: policy-gradient fine-tuning on a single instance.
    Returns the best solution found and its metrics.
    """
    # Clone the model so global weights aren't modified
    as_model = copy.deepcopy(model)
    as_model = as_model.to(device)
    _flatten_lstm_parameters(as_model)
    as_model.train()
    optimizer = torch.optim.Adam(as_model.parameters(), lr=lr)
    
    data_tensor = data_tensor.to(device)
    mask = mask.to(device)
    length_tensor = torch.tensor([length], dtype=torch.long, device=device)
    real_points = data_tensor[0, :length].detach().cpu().numpy()
    reward_func = None if no_geometry else partial(reward_fn, alpha=5.0, p=0.0)
    
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
            solution = sorted([idx for idx in selected_idxs[0] if idx < length])
            # skip degenerate empty solution by forcing EOS-only samples to empty
            if no_geometry:
                # Geometry-free: use simple heuristic proxy (solution size) as loss if no value_net
                L = float(len(solution))
            else:
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
    if no_geometry:
        best_coverage = float('nan')
    else:
        try:
            best_coverage = float(evaluate_polygon_visibility_numpy_wo_gt(real_points, best_solution, name))
        except Exception:
            best_coverage = 0.0
    
    return best_solution, best_reward, best_coverage


def active_search_train_proxy_on_instance(model, value_net, data_tensor, mask, length, name, K=64, B=8, lr=3e-4, alpha=0.9, device='cpu', no_geometry=False):
    """
    Active Search using value_net predictions as proxy rewards for REINFORCE.
    Avoids calling the expensive true reward; gradients are computed on predicted reward.
    """
    as_model = copy.deepcopy(model)
    as_model = as_model.to(device)
    _flatten_lstm_parameters(as_model)
    as_model.train()
    value_net = value_net.to(device)
    value_net.eval()
    optimizer = torch.optim.Adam(as_model.parameters(), lr=lr)

    data_tensor = data_tensor.to(device)
    mask = mask.to(device)
    length_tensor = torch.tensor([length], dtype=torch.long, device=device)
    
    samples_remaining = int(K)
    best_solution = []
    best_pred_reward = float('inf')
    
    while samples_remaining > 0:
        batch = min(B, samples_remaining)
        samples_remaining -= batch
        log_probs_list = []
        proxy_losses = None
        sols = []
        # Sample batch
        for _ in range(batch):
            selected_idxs, log_probs = as_model(data_tensor, padding_mask=mask, lengths=length_tensor)
            solution = [idx for idx in selected_idxs[0] if idx < length]
            sols.append(solution)
            log_probs_list.append(log_probs.squeeze())
        # Predict rewards for batch solutions (skip empties and assign penalty)
        if len(sols) > 0:
            batch_size = len(sols)
            empty_penalty = 10.0  # large loss so empties are not selected
            proxy_losses = torch.full((batch_size,), empty_penalty, dtype=torch.float32, device=device)
            valid_idxs = [i for i, s in enumerate(sols) if len(s) > 0]
            if len(valid_idxs) > 0:
                batch_polygons = data_tensor.repeat(len(valid_idxs), 1, 1)
                batch_solutions = pad_sequence(
                    [torch.tensor(sols[i], dtype=torch.long) for i in valid_idxs],
                    batch_first=True,
                    padding_value=0,
                ).to(device)
                poly_lengths = torch.tensor([length] * len(valid_idxs), dtype=torch.long, device=device)
                sol_lengths = torch.tensor([len(sols[i]) for i in valid_idxs], dtype=torch.long, device=device)
                with torch.no_grad():
                    if hasattr(value_net, 'predict_reward'):
                        preds = value_net.predict_reward(batch_polygons, batch_solutions, poly_lengths, sol_lengths)
                    else:
                        preds = value_net(batch_polygons, batch_solutions, poly_lengths, sol_lengths)
                        # Handle different model types: dict output (MultiTaskValueNet) vs tensor output
                        if isinstance(preds, dict):
                            preds = preds['reward']  # Use reward for evaluation
                preds = preds.detach().float()
                # scatter back into full proxy_losses
                proxy_losses[torch.tensor(valid_idxs, dtype=torch.long, device=device)] = preds
            # Update incumbent based on predicted loss (considering penalties for empties)
            min_idx = int(torch.argmin(proxy_losses).item())
            if float(proxy_losses[min_idx]) < best_pred_reward:
                best_pred_reward = float(proxy_losses[min_idx])
                best_solution = sols[min_idx]
        # Policy gradient step on proxy loss
        if len(log_probs_list) > 0 and proxy_losses is not None:
            L_tensor = proxy_losses.to(device)
            lp_tensor = torch.stack(log_probs_list)
            mean_L = L_tensor.mean().item()
            # EMA over proxy loss
            b = mean_L  # simple moving baseline per step is fine here
            advantage = L_tensor - b
            pg_loss = (advantage * lp_tensor).mean()
            optimizer.zero_grad()
            pg_loss.backward()
            torch.nn.utils.clip_grad_norm_(as_model.parameters(), 1.0)
            optimizer.step()

    # Compute coverage estimate for the best proxy-selected solution
    real_points = data_tensor[0, :length].detach().cpu().numpy()
    if no_geometry:
        best_coverage = float('nan')
    else:
        try:
            best_coverage = float(evaluate_polygon_visibility_numpy_wo_gt(real_points, best_solution, name))
        except Exception:
            best_coverage = 0.0
    return best_solution, best_pred_reward, best_coverage

def evaluate_with_as_training(model, dataset, val_dir, K=64, B=8, lr=3e-4, alpha=0.9, device='cpu', use_proxy=False, value_net=None, no_geometry=False):
    """Run Active Search training per instance and report aggregated stats.
    If use_proxy=True, uses value_net predictions as proxy rewards (no true reward calls).
    """
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
        if use_proxy:
            if value_net is None:
                raise ValueError("--as-proxy requires a loaded value_net (use --value-net-path or auto)")
            # Run AS with proxy rewards (no true reward calls inside)
            best_solution, _proxy_best, _proxy_cov = active_search_train_proxy_on_instance(
                model, value_net, data_tensor, instance_mask, length, name, K=K, B=B, lr=lr, alpha=alpha, device=device, no_geometry=no_geometry
            )
            # For metrics, compute TRUE reward and coverage once for the selected solution
            if not no_geometry:
                real_points = data_tensor[0, :length].detach().cpu().numpy()
                reward_func = partial(reward_fn, alpha=5.0, p=0.0)
                try:
                    best_reward = float(reward_func(real_points, best_solution, name, length=length))
                except Exception:
                    best_reward = float('inf')
                try:
                    best_coverage = float(evaluate_polygon_visibility_numpy_wo_gt(real_points, best_solution, name))
                except Exception:
                    best_coverage = 0.0
            else:
                best_reward = float(len(best_solution))
                best_coverage = float('nan')
        else:
            best_solution, best_reward, best_coverage = active_search_train_on_instance(
                model, data_tensor, instance_mask, length, name, K=K, B=B, lr=lr, alpha=alpha, device=device, no_geometry=no_geometry
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


def active_search_single_instance(model, data_tensor, mask, length, name, K=10, device='cpu', value_net=None, no_geometry=False):
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
            real_solution = sorted([idx for idx in solution if idx < length])
            
            # Compute reward (smaller is better)
            if no_geometry:
                r = float(len(real_solution))
                coverage = float('nan')
            else:
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
                # Handle different model types: dict output (MultiTaskValueNet) vs tensor output (RankerNet/SolutionValueNet)
                if isinstance(preds, dict):
                    # MultiTaskValueNet returns {'reward': tensor, 'coverage': tensor}
                    preds = preds['reward']  # Use reward for evaluation
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


def evaluate_with_sampling(model, dataset, val_dir, K=10, device='cpu', value_net=None, use_value_net=False, no_geometry=False):
    """
    Evaluate the model using simple sampling (no active search).
    
    Args:
        model: Trained RL model
        dataset: Validation dataset
        val_dir: Directory containing .solution files for optimal comparison
        K: Number of solutions to sample per instance
        device: Device to run on
        value_net: Optional trained value network for solution selection
        use_value_net: Whether to use value network for solution selection
        no_geometry: Skip geometry-based reward computations
    
    Returns:
        results: Dictionary with evaluation results
    """
    import time
    
    evaluation_type = "Value Network Sampling" if use_value_net else "Standard Sampling"
    print(f"\n--- {evaluation_type} Evaluation (K={K}) on {len(dataset)} validation samples ---")
    
    model.eval()
    
    # Use bucket sampler for consistent batching
    lengths = get_lengths_from_dataset(dataset)
    batch_sampler = BucketBatchSampler(lengths, batch_size=1, shuffle=False, bucket_size=10)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, 
                       pin_memory=True, num_workers=0)
    
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
    
    start_time = time.time()
    
    for batch_idx, (batch_data, mask, lengths, batch_names) in enumerate(loader):
        batch_data, mask = batch_data.to(device), mask.to(device)
        
        # Process single instance
        data_tensor = batch_data[0:1]  # [1, max_len, 2]
        instance_mask = mask[0:1]  # [1, max_len]
        length = lengths[0]  # lengths is a list, not tensor
        name = batch_names[0]
        
        if use_value_net and value_net is not None:
            # Use value network for fast sampling evaluation
            result = fast_active_search_with_value_net(
                model, value_net, data_tensor, instance_mask, length, name, K, device
            )
        else:
            # Use standard sampling evaluation
            result = standard_sampling_single_instance(
                model, data_tensor, instance_mask, length, name, K, device, no_geometry
            )
        
        # Collect results
        all_best_rewards.append(result['best_reward'])
        all_best_coverages.append(result['best_coverage']) 
        all_best_sizes.append(result['best_size'])
        all_size_ratios.append(result['size_ratio'])
        all_reward_variances.append(result.get('reward_variance', 0.0))
        all_coverage_variances.append(result.get('coverage_variance', 0.0))
        all_pred_err_selected.append(result.get('pred_err_selected', 0.0))
        all_pred_err_opt.append(result.get('pred_err_opt', 0.0))
        all_regrets.append(result.get('regret', 0.0))
        all_corrs.append(result.get('correlation', 0.0))
        
        print(f"Instance {batch_idx+1}/{len(dataset)}: {name} - Best reward: {result['best_reward']:.4f}, Coverage: {result['best_coverage']:.4f}")
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    def safe_stats(values):
        """Calculate statistics safely handling NaN values"""
        clean_values = [v for v in values if not (np.isnan(v) or np.isinf(v))]
        if not clean_values:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
        
        return {
            'mean': float(np.mean(clean_values)),
            'std': float(np.std(clean_values)),
            'min': float(np.min(clean_values)),
            'max': float(np.max(clean_values)),
            'median': float(np.median(clean_values))
        }
    
    results = {
        'evaluation_type': evaluation_type,
        'num_instances': len(dataset),
        'K': K,
        'total_time': total_time,
        'avg_time_per_instance': total_time / len(dataset),
        'best_rewards': safe_stats(all_best_rewards),
        'best_coverages': safe_stats(all_best_coverages),
        'best_sizes': safe_stats(all_best_sizes),
        'size_ratios': safe_stats(all_size_ratios),
        'reward_variances': safe_stats(all_reward_variances),
        'coverage_variances': safe_stats(all_coverage_variances),
        'pred_err_selected': safe_stats(all_pred_err_selected),
        'pred_err_opt': safe_stats(all_pred_err_opt),
        'regrets': safe_stats(all_regrets),
        'correlations': safe_stats(all_corrs),
        'raw_results': {
            'best_rewards': all_best_rewards,
            'best_coverages': all_best_coverages,
            'best_sizes': all_best_sizes,
            'size_ratios': all_size_ratios
        }
    }
    
    return results


def standard_sampling_single_instance(model, data_tensor, mask, length, name, K, device, no_geometry=False):
    """
    Standard sampling evaluation for a single instance.
    Sample K solutions and pick the best one based on true rewards.
    """
    solutions = []
    rewards = []
    coverages = []
    
    # Sample K solutions
    for _ in range(K):
        with torch.no_grad():
            selected_idxs, log_probs = model(data_tensor, mask, lengths=torch.tensor([length], device=device))
            solution = selected_idxs[0]  # Get first batch item
            
            # Convert to numpy if it's a tensor, otherwise keep as is
            if hasattr(solution, 'cpu'):
                solution = solution.cpu().numpy()
            elif isinstance(solution, (list, tuple)):
                solution = np.array(solution)
            
            # Remove EOS token and padding
            if length < len(solution):
                solution = solution[:length+1]  # Keep up to EOS
            eos_pos = np.where(solution == length)[0]
            if len(eos_pos) > 0:
                solution = solution[:eos_pos[0]]  # Remove EOS and everything after
            
            solutions.append(solution)
    
    # Evaluate all solutions
    polygon_coords = data_tensor[0, :length, :].cpu().numpy()
    
    for solution in solutions:
        if not no_geometry and len(solution) > 0:
            reward = reward_fn(polygon_coords, solution, name, length=length)
            # Compute coverage separately
            try:
                coverage = evaluate_polygon_visibility_numpy_wo_gt(polygon_coords, solution, name)
            except Exception:
                coverage = 0.0
        else:
            reward = 0.0
            coverage = 1.0
        
        rewards.append(reward)
        coverages.append(coverage)
    
    # Find best solution
    if len(rewards) > 0:
        best_idx = np.argmin(rewards)
        best_reward = rewards[best_idx]
        best_coverage = coverages[best_idx]
        best_solution = solutions[best_idx]
        best_size = len(best_solution)
    else:
        best_reward = float('inf')
        best_coverage = 0.0
        best_size = 0
    
    # Calculate statistics
    reward_variance = np.var(rewards) if len(rewards) > 1 else 0.0
    coverage_variance = np.var(coverages) if len(coverages) > 1 else 0.0
    
    # Size ratio (compared to polygon size as rough optimal estimate)
    size_ratio = best_size / max(1, length) if length > 0 else 1.0
    
    return {
        'best_reward': best_reward,
        'best_coverage': best_coverage,
        'best_size': best_size,
        'size_ratio': size_ratio,
        'reward_variance': reward_variance,
        'coverage_variance': coverage_variance,
        'regret': 0.0,  # No regret calculation in simple sampling
        'correlation': 0.0,  # No prediction correlation in simple sampling
        'pred_err_selected': 0.0,
        'pred_err_opt': 0.0
    }


def evaluate_with_qlearning(dataset, val_dir, device='cpu', no_geometry=False, 
                           max_episodes=50, verbose=False):
    """
    Evaluate using Q-learning tabular approach on validation dataset.
    
    Args:
        dataset: Validation dataset
        val_dir: Directory containing .pol files
        device: Device (unused for Q-learning)
        no_geometry: Whether to disable geometry computations
        max_episodes: Maximum episodes for Q-learning training per instance
        verbose: Whether to print progress
        
    Returns:
        dict: Evaluation results
    """
    from qlearning_agp import evaluate_qlearning_single_instance
    
    all_best_rewards = []
    all_best_coverages = []
    all_best_sizes = []
    all_size_ratios = []
    all_training_times = []
    all_cache_hit_rates = []
    
    start_time = time.time()
    
    print(f"--- Optimized Q-Learning Evaluation on {len(dataset)} validation samples ---")
    
    for i, sample in enumerate(dataset.samples):
        polygon_coords = sample.data
        instance_name = sample.name
        
        print(f"Instance {i+1}/{len(dataset)}: {instance_name}")
        
        # Run Q-learning on this instance with optimized parameters
        result = evaluate_qlearning_single_instance(
            polygon_coords, instance_name,
            max_episodes=max_episodes,
            max_steps_per_episode=15,  # Reduced steps for faster evaluation
            target_coverage=0.95,  # Stop early at 90% coverage
            patience=20,  # Reduced patience
            verbose=False  # Reduce verbosity for batch evaluation
        )
        
        # Collect results
        all_best_rewards.append(result['best_coverage'])  # Coverage is our reward
        all_best_coverages.append(result['best_coverage'])
        all_best_sizes.append(result['best_size'])
        all_size_ratios.append(result['size_ratio'])
        all_training_times.append(result['training_time'])
        all_cache_hit_rates.append(result['cache_stats']['hit_rate'])
        
        if verbose:
            print(f"  Coverage: {result['best_coverage']:.4f}, "
                  f"Solution size: {result['best_size']}, "
                  f"Training time: {result['training_time']:.2f}s, "
                  f"Cache hit rate: {result['cache_stats']['hit_rate']*100:.1f}%")
    
    evaluation_time = time.time() - start_time
    avg_time = evaluation_time / len(dataset)
    avg_training_time = np.mean(all_training_times)
    avg_cache_hit_rate = np.mean(all_cache_hit_rates)
    
    # Compute statistics
    def compute_stats(values):
        if not values:
            return {}
        values = np.array(values)
        return {
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
            'iqr': float(np.percentile(values, 75) - np.percentile(values, 25))
        }
    
    results = {
        'num_instances': len(dataset),
        'K': 1,  # Q-learning finds one solution per instance
        'evaluation_time': evaluation_time,
        'avg_time_per_instance': avg_time,
        'avg_training_time_per_instance': avg_training_time,
        'avg_cache_hit_rate': avg_cache_hit_rate,
        'best_reward_stats': compute_stats(all_best_rewards),
        'best_coverage_stats': compute_stats(all_best_coverages),
        'best_size_stats': compute_stats(all_best_sizes),
        'size_ratio_stats': compute_stats(all_size_ratios),
        'training_time_stats': compute_stats(all_training_times),
        'cache_hit_rate_stats': compute_stats(all_cache_hit_rates),
        # Q-learning specific stats
        'method': 'qlearning_optimized',
        'max_episodes': max_episodes,
        'optimization_features': [
            'visibility_region_caching',
            'precomputed_guard_visibility',
            'incremental_coverage_computation',
            'early_stopping',
            'reduced_episodes_and_steps'
        ]
    }
    
    return results


def evaluate_with_active_search(model, dataset, val_dir, K=10, batch_size=1, device='cpu', value_net=None, use_value_net=False, no_geometry=False):
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
            active_search_single_instance(model, data_tensor, instance_mask, length, name, K, device, value_net, no_geometry)
        
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
        if not no_geometry:
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


def print_results(results, args=None):
    """Print evaluation results in a nice format."""
    # Use new simplified proxy argument structure
    if args and hasattr(args, 'proxy'):
        proxy = args.proxy
        mode = getattr(args, 'mode', 'sampling')
    else:
        # Backward compatibility fallback
        use_value_net = results.get('use_value_net', False)
        use_ranker = results.get('use_ranker', False)
        if use_ranker:
            proxy = 'ranker'
        elif use_value_net:
            proxy = 'value_net'
        else:
            proxy = 'reward'
        mode = 'sampling'
    
    if proxy == 'ranker':
        search_type = "Ranker"
    elif proxy == 'value_net':
        search_type = "Value Net"
    else:
        search_type = "Standard"
    
    print(f"\n=== {search_type.upper()} {mode.upper()} EVALUATION RESULTS (K={results['K']}) ===")
    print(f"Number of instances: {results['num_instances']}")
    
    # Print timing information
    if 'evaluation_time' in results:
        total_time = results['evaluation_time']
        avg_time = results.get('avg_time_per_instance', 0)
        print(f"\nTiming Information:")
        print(f"  Total evaluation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"  Average time per instance: {avg_time:.3f} seconds")
        if proxy in ['value_net', 'ranker']:
            network_name = "ranker network" if proxy == 'ranker' else "value network"
            print(f"  🚀 Using {network_name} for fast evaluation!")
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
    
    print_stat_section(results.get('best_rewards', results.get('best_reward_stats', {})), "Best Reward Statistics")
    print_stat_section(results.get('best_coverages', results.get('best_coverage_stats', {})), "Best Coverage Statistics") 
    print_stat_section(results.get('best_sizes', results.get('best_size_stats', {})), "Best Solution Size Statistics")
    print_stat_section(results.get('size_ratios', results.get('size_ratio_stats', {})), "Size Ratio Statistics (best/optimal)")
    
    # Print value net quality stats if using value network
    if proxy == 'value_net':
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
                       default='auto',
                       help='Path to model checkpoint ("auto" to auto-detect best RL checkpoint)')
    parser.add_argument('--val-dir', type=str, 
                       default=os.path.join(DATASET_PATH, "dev"),
                       help='Directory with validation .pol files')
    parser.add_argument('--K', type=int, default=1,
                       help='Number of solutions to sample per instance')
    parser.add_argument('--max-instances', type=int, default=None,
                       help='Maximum number of instances to evaluate (for testing)')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize coordinates')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results. If not specified, will be auto-generated based on mode and parameters.')
    parser.add_argument('--no-geometry', action='store_true',
                       help='Disable all geometry-based computations (true rewards and coverage). Useful for fully geometry-free RL demos.')
    
    # === CORE EVALUATION PARAMETERS ===
    # Evaluation Mode: What strategy to use
    parser.add_argument('--mode', choices=['sampling', 'search', 'qlearning'], default='sampling',
                       help='Evaluation mode: "sampling" (sample K solutions, pick best), "search" (active search with test-time training), or "qlearning" (Q-learning tabular approach)')
    
    # Evaluation Proxy: How to evaluate/rank solutions  
    parser.add_argument('--proxy', choices=['reward', 'value_net', 'ranker'], default='reward',
                       help='Evaluation proxy: "reward" (true geometry), "value_net" (trained value network), "ranker" (trained ranker)')
    
    # Proxy model path (auto-detected if not specified)
    parser.add_argument('--proxy-path', type=str, default='auto',
                       help='Path to proxy model checkpoint ("auto" to auto-detect best)')

    args = parser.parse_args()    # === ARGUMENT VALIDATION AND CLEANUP ===
    # Simplify the logic - convert old-style args to new style for backward compatibility
    if hasattr(args, 'use_as') and args.use_as:
        args.mode = 'search'
    if hasattr(args, 'use_value_net') and args.use_value_net:
        args.proxy = 'value_net'
    if hasattr(args, 'use_ranker') and args.use_ranker:
        args.proxy = 'ranker'
    
    # Set proxy path from old arguments if available
    if hasattr(args, 'value_net_path') and args.proxy == 'value_net' and args.proxy_path == 'auto':
        args.proxy_path = args.value_net_path
    if hasattr(args, 'ranker_path') and args.proxy == 'ranker' and args.proxy_path == 'auto':
        args.proxy_path = args.ranker_path
    
    print(f"🎯 Evaluation Mode: {args.mode.upper()}")
    print(f"🔧 Evaluation Proxy: {args.proxy.upper()}")
    if args.proxy != 'reward':
        print(f"📁 Proxy Model Path: {args.proxy_path}")
    
    # Load validation dataset
    print(f"\nLoading validation dataset from: {args.val_dir}")
    val_files = [os.path.join(args.val_dir, f) for f in os.listdir(args.val_dir) if f.endswith('.pol')]
    if args.max_instances:
        val_files = val_files[:args.max_instances]
    
    val_samples = agp_read_samples(val_files, normalize=args.normalize)
    val_dataset = Dataset(val_samples)
    print(f"Loaded {len(val_dataset)} validation instances")
    
    # Load model
    checkpoint_path = args.checkpoint
    if checkpoint_path == 'auto':
        auto_checkpoint = find_best_rl_checkpoint()
        if auto_checkpoint:
            checkpoint_path = auto_checkpoint
            print(f"🔍 Auto-detected RL model: {checkpoint_path}")
        else:
            raise FileNotFoundError("No RL model checkpoints found in 'checkpoints/' directory")
    
    model, device = load_rl_model(checkpoint_path)
    
    # === LOAD PROXY MODEL (if needed) ===
    proxy_model = None
    if args.proxy != 'reward':
        print(f"\n🔧 Loading {args.proxy} proxy model...")
        
        # Auto-detect proxy path if needed
        if args.proxy_path == 'auto':
            if args.proxy == 'value_net':
                auto_path = find_best_value_net_checkpoint()
                if auto_path:
                    args.proxy_path = auto_path
                    print(f"🔍 Auto-detected value network: {auto_path}")
                else:
                    print("✗ No value network checkpoints found")
                    print("  Falling back to reward-based evaluation")
                    args.proxy = 'reward'
            elif args.proxy == 'ranker':
                auto_path = find_best_ranker_checkpoint()
                if auto_path:
                    args.proxy_path = auto_path
                    print(f"🔍 Auto-detected ranker network: {auto_path}")
                else:
                    print("✗ No ranker network checkpoints found")
                    print("  Falling back to reward-based evaluation")
                    args.proxy = 'reward'
        
        # Load proxy model if we have a valid path
        if args.proxy != 'reward' and os.path.exists(args.proxy_path):
            try:
                proxy_model, _, model_info = load_reward_predictor(args.proxy_path)
                print(f"✓ {args.proxy} loaded successfully")
                
            except Exception as e:
                print(f"✗ Error loading {args.proxy}: {e}")
                print("  Falling back to reward-based evaluation")
                args.proxy = 'reward'
                proxy_model = None
        elif args.proxy != 'reward':
            print(f"✗ Proxy model not found at {args.proxy_path}")
            print("  Falling back to reward-based evaluation")
            args.proxy = 'reward'
            proxy_model = None
    
    # === EVALUATE WITH CHOSEN STRATEGY ===
    if args.mode == 'qlearning':
        # Q-learning tabular approach (no neural network needed)
        print("🧠 Using optimized Q-learning tabular approach")
        print("📋 Q-Learning Configuration:")
        print(f"  • Max episodes per instance: 1000")
        print(f"  • Max steps per episode: 15") 
        print(f"  • Target coverage: 90%")
        print(f"  • Learning rate: 0.1")
        print(f"  • Epsilon decay: 0.995 → 0.01")
        print(f"  • Early stopping: enabled")
        print(f"  • Visibility region caching: enabled")
        print(f"  • State representation: number of guards")
        print(f"  • Action space: toggle guard at vertex")
        results = evaluate_with_qlearning(
            val_dataset, args.val_dir,
            device=device,
            no_geometry=args.no_geometry,
            max_episodes=1000,
            verbose=True
        )
        training_method = 'qlearning_optimized'
    elif args.mode == 'search':
        # Active Search with test-time training
        print("🚀 Using Active Search with test-time training")
        print("📋 Active Search Configuration:")
        print(f"  • Model: {os.path.basename(checkpoint_path)}")
        print(f"  • Proxy for ranking: {args.proxy}")
        print(f"  • Solutions per instance (K): {args.K}")
        print(f"  • Test-time gradient steps: enabled")
        print(f"  • Policy gradient optimization: REINFORCE")
        if args.proxy != 'reward':
            print(f"  • Proxy model: {os.path.basename(args.proxy_path)}")
        results = evaluate_with_as_training(
            model, val_dataset, args.val_dir,
            K=args.K,
            device=device,
            value_net=proxy_model,
            no_geometry=args.no_geometry
        )
        training_method = 'search_' + args.proxy
    else:
        # Default: Simple sampling evaluation
        use_proxy = args.proxy != 'reward'
        print("🎲 Using sampling evaluation")
        print("📋 Sampling Configuration:")
        print(f"  • Model: {os.path.basename(checkpoint_path)}")
        print(f"  • Solutions per instance (K): {args.K}")
        print(f"  • Evaluation proxy: {args.proxy}")
        if use_proxy:
            print(f"  • Proxy model: {os.path.basename(args.proxy_path)}")
            print(f"  • Selection strategy: best predicted {args.proxy}")
        else:
            print(f"  • Selection strategy: best true reward")
        
        # Extract model architecture info from filename or checkpoint
        try:
            if hasattr(model, 'embedding_size'):
                print(f"  • Model architecture: {model.embedding_size}d embeddings, {model.hidden_size}d hidden")
            else:
                # Try to extract from filename
                import re
                filename = os.path.basename(checkpoint_path)
                emb_match = re.search(r'embedding_size(\d+)', filename)
                hid_match = re.search(r'hidden_size(\d+)', filename)
                if emb_match and hid_match:
                    print(f"  • Model architecture: {emb_match.group(1)}d embeddings, {hid_match.group(1)}d hidden")
            
            if hasattr(model, 'n_glimpses'):
                print(f"  • Attention glimpses: {model.n_glimpses}")
            elif 'n_glimpses' in filename:
                glimpse_match = re.search(r'n_glimpses(\d+)', filename)
                if glimpse_match:
                    print(f"  • Attention glimpses: {glimpse_match.group(1)}")
        except Exception:
            pass
        
        if use_proxy:
            print(f"🎯 Using {args.proxy} for solution selection")
        else:
            print("🎲 Using true rewards for solution selection")
            
        results = evaluate_with_sampling(
            model, val_dataset, args.val_dir, 
            K=args.K, 
            device=device,
            value_net=proxy_model, 
            use_value_net=use_proxy, 
            no_geometry=args.no_geometry
        )
        
        training_method = 'sampling_' + args.proxy
    
    # Print results
    print_results(results, args)
    
    # Prepare results for saving
    results_summary = {
        'args': vars(args),
        'K': args.K,
        'mode': args.mode,
        'proxy': args.proxy,
        'num_instances': results['num_instances'],
        'training_method': training_method,
        'evaluation_type': results.get('evaluation_type', training_method),
        'total_time': results.get('total_time', 0.0),
        'avg_time_per_instance': results.get('avg_time_per_instance', 0.0),
        'best_rewards': results.get('best_rewards', results.get('best_reward_stats', {})),
        'best_coverages': results.get('best_coverages', results.get('best_coverage_stats', {})), 
        'best_sizes': results.get('best_sizes', results.get('best_size_stats', {})),
        'size_ratios': results.get('size_ratios', results.get('size_ratio_stats', {}))
    }
    
    # Add additional stats if available
    if 'prediction_error_stats' in results:
        results_summary['prediction_error_stats'] = results['prediction_error_stats']
    if 'regrets' in results:
        results_summary['regrets'] = results['regrets']
    if 'correlations' in results:
        results_summary['correlations'] = results['correlations']
    
    # Generate mode-specific output filename if not provided
    if args.output is None:
        if args.mode == 'qlearning':
            output_file = f'results/qlearning_evaluation.json'
        elif args.mode == 'search':
            output_file = f'results/search_evaluation_{args.proxy}_K={args.K}.json'
        else:  # sampling mode
            output_file = f'results/sampling_evaluation_{args.proxy}_K={args.K}.json'
    else:
        # Use user-provided filename, replace placeholders
        output_file = args.output.replace('{K}', str(args.K))
        output_file = output_file.replace('{mode}', args.mode)
        output_file = output_file.replace('{proxy}', args.proxy)
    
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
