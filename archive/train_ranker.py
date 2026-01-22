#!/usr/bin/env python3
"""
Train a simple inference-time RankerNet for AGP: given polygon coords and a guard set, output a scalar score.
Use pairwise logistic ranking loss within each polygon. Higher score = better solution.
The file also supports data generation from a trained actor.
"""

import os
import argparse
import json
import time
import pickle
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

from dataset import Dataset as AGPDataset, agp_read_samples
from models import create_actor
from rewards import enhanced_penalty as reward_fn


class RankerNet(nn.Module):
    def __init__(self, embedding_size=128, hidden_size=256, max_vertices=1000):
        super().__init__()
        self.polygon_encoder = nn.LSTM(2, embedding_size, batch_first=True)
        self.guard_embedding = nn.Embedding(max_vertices, embedding_size)
        self.solution_encoder = nn.LSTM(embedding_size, embedding_size, batch_first=True)
        fusion_in = embedding_size * 2 + 2  # add meta: |S|/N, N_norm
        self.backbone = nn.Sequential(
            nn.Linear(fusion_in, hidden_size), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.1),
        )
        self.score_head = nn.Linear(hidden_size, 1)

    def forward(self, polygon_coords, guard_indices, polygon_lengths, solution_lengths):
        packed_poly = pack_padded_sequence(polygon_coords, polygon_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (poly_hidden, _) = self.polygon_encoder(packed_poly)
        poly_features = poly_hidden[-1]
        guard_embeds = self.guard_embedding(guard_indices)
        packed_sol = pack_padded_sequence(guard_embeds, solution_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (sol_hidden, _) = self.solution_encoder(packed_sol)
        sol_features = sol_hidden[-1]
        n_vertices = polygon_lengths.to(polygon_coords.device).float()
        n_guards = solution_lengths.to(polygon_coords.device).float()
        n_ratio = torch.where(n_vertices > 0, n_guards / n_vertices, torch.zeros_like(n_vertices))
        N_norm = torch.clamp(n_vertices / 1000.0, 0.0, 1.0)
        meta = torch.stack([n_ratio, N_norm], dim=1)
        h = self.backbone(torch.cat([poly_features, sol_features, meta], dim=1))
        return self.score_head(h).squeeze(-1)

    def predict_reward(self, polygon_coords, guard_indices, polygon_lengths, solution_lengths):
        # Compatibility with evaluate.py (lower is better): treat reward as -score
        return -self.forward(polygon_coords, guard_indices, polygon_lengths, solution_lengths)


class PairBatchSampler:
    """Custom batch sampler that keeps solution pairs together in the same batch"""
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by pair_id
        self.pair_groups = {}
        for idx, item in enumerate(dataset.data):
            pair_id = item.get('pair_id')
            if pair_id:
                self.pair_groups.setdefault(pair_id, []).append(idx)
        
        # Convert to list of pair index lists
        self.pairs = list(self.pair_groups.values())
        
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.pairs)
        
        batch = []
        for pair_indices in self.pairs:
            # Add all indices from this pair to current batch
            batch.extend(pair_indices)
            
            # If batch is big enough, yield it and start new batch
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
        
        # Yield remaining items if any
        if batch:
            yield batch
    
    def __len__(self):
        total_items = sum(len(pair) for pair in self.pairs)
        return (total_items + self.batch_size - 1) // self.batch_size


class SolutionDataset(Dataset):
    def __init__(self, samples):
        self.data = samples
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.tensor(item['polygon'], dtype=torch.float32),
            torch.tensor(item['solution'], dtype=torch.long),
            torch.tensor(item['reward'], dtype=torch.float32),
            item['polygon_length'],
            item['solution_length'],
            item.get('polygon_name', None),
            item.get('pair_id', None),
        )
    def __init__(self, samples):
        self.data = samples
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.tensor(item['polygon'], dtype=torch.float32),
            torch.tensor(item['solution'], dtype=torch.long),
            torch.tensor(item['reward'], dtype=torch.float32),
            item['polygon_length'],
            item['solution_length'],
            item.get('polygon_name', None),
            item.get('pair_id', None),
        )


def collate_fn(batch):
    polygons, solutions, rewards, poly_lengths, sol_lengths, names, pair_ids = zip(*batch)
    polygons_padded = pad_sequence(polygons, batch_first=True, padding_value=0.0)
    solutions_padded = pad_sequence(solutions, batch_first=True, padding_value=0)
    rewards_tensor = torch.stack(rewards)
    poly_lengths_tensor = torch.tensor(poly_lengths, dtype=torch.long)
    sol_lengths_tensor = torch.tensor(sol_lengths, dtype=torch.long)
    return (
        polygons_padded,
        solutions_padded,
        rewards_tensor,
        poly_lengths_tensor,
        sol_lengths_tensor,
        names,
        pair_ids,
    )


def load_trained_actor(checkpoint_path, embedding_size=128, hidden_size=128, n_glimpses=1,
                       tanh_exploration=10, use_tanh=True, temperature=1.0):
    print(f"Loading trained actor from: {checkpoint_path}")
    model = create_actor(
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        seq_len=None,
        n_glimpses=n_glimpses,
        tanh_exploration=tanh_exploration,
        use_tanh=use_tanh,
        attention_type="Bahdanau",
        reward_fn=reward_fn,
        temperature=temperature,
    )
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return model.to(device).eval(), device


def generate_training_data(actor_model, dataset, num_pairs_per_polygon=32, device='cpu', max_time_per_polygon=300):
    print(f"Generating training data: {num_pairs_per_polygon} solution pairs per polygon...")
    training_data = []
    r_fn = partial(reward_fn, alpha=5.0, p=0.0)
    actor_model.eval()
    
    with torch.no_grad():
        for i, (polygon_data, _, polygon_name) in enumerate(tqdm(dataset, desc="Processing polygons")):
            polygon_start_time = time.time()
            
            try:
                polygon_coords = polygon_data.numpy()
                N = len(polygon_coords)
                
                # Skip if polygon is too small or too large
                if N < 3:
                    continue
                if N > 1000:
                    continue
            
                # Strategy: Generate solution pairs explicitly for guaranteed ranking training
                solution_pairs = []
                
                # Generate base solutions using different strategies
                base_solutions = []
                
                # 1. Get solutions from actor model
                for sample_idx in range(min(8, num_pairs_per_polygon // 4)):
                    try:
                        data_tensor = polygon_data.unsqueeze(0).to(device)
                        mask = torch.zeros(1, N, dtype=torch.bool, device=device)
                        lengths = torch.tensor([N], device=device)
                        torch.manual_seed(42 + i * 1000 + sample_idx)
                        guard_indices, _ = actor_model(data_tensor, padding_mask=mask, lengths=lengths)
                        sol = guard_indices[0] if isinstance(guard_indices, list) else guard_indices[0].cpu().numpy()
                        # Sanitize solution
                        if N > 0:
                            seen = set()
                            clamped = []
                            for idx in sol:
                                if not isinstance(idx, (int, np.integer)):
                                    continue
                                idx = max(0, min(int(idx), N-1))
                                if idx not in seen:
                                    seen.add(idx)
                                    clamped.append(idx)
                            sol = sorted(clamped)
                        if len(sol) > 0:
                            base_solutions.append(sol)
                    except Exception:
                        continue
                
                # 2. Generate systematic solutions for variety
                systematic_solutions = []
                
                # Random solutions of different sizes
                for size in [1, 2, 3, 5, min(N//10, 8), min(N//5, 15)]:
                    if size >= N:
                        continue
                    for seed_offset in range(3):
                        np.random.seed(42 + i * 1000 + size * 100 + seed_offset)
                        indices = np.random.choice(N, size=size, replace=False)
                        sol = sorted(indices.tolist())
                        systematic_solutions.append(sol)
                
                # Systematic patterns
                for k in [max(1, N//20), max(1, N//10), max(1, N//5)]:
                    sol = list(range(0, N, k))
                    if len(sol) > 0:
                        systematic_solutions.append(sol)
                
                # Local clusters
                for center in [0, N//4, N//2, 3*N//4]:
                    for radius in [2, 5]:
                        sol = []
                        for r in range(radius):
                            if center + r < N:
                                sol.append(center + r)
                            if center - r >= 0 and center - r not in sol:
                                sol.append(center - r)
                        if len(sol) > 0:
                            systematic_solutions.append(sorted(sol))
                
                # Combine all solutions and remove duplicates
                all_solutions = base_solutions + systematic_solutions
                unique_solutions = []
                seen_solutions = set()
                for sol in all_solutions:
                    sol_tuple = tuple(sol)
                    if sol_tuple not in seen_solutions and len(sol) > 0:
                        seen_solutions.add(sol_tuple)
                        unique_solutions.append(sol)
                
                # Ensure we have enough solutions for pairing
                min_solutions_needed = max(10, int(np.sqrt(num_pairs_per_polygon * 2)))
                while len(unique_solutions) < min_solutions_needed:
                    np.random.seed(42 + i * 10000 + len(unique_solutions))
                    size = np.random.randint(1, min(N//3, 10) + 1)
                    indices = np.random.choice(N, size=size, replace=False)
                    sol = sorted(indices.tolist())
                    sol_tuple = tuple(sol)
                    if sol_tuple not in seen_solutions:
                        seen_solutions.add(sol_tuple)
                        unique_solutions.append(sol)
                
                # Pre-compute rewards for all solutions
                solution_rewards = []
                for sol in unique_solutions:
                    try:
                        r = r_fn(polygon_coords, sol, polygon_name, length=N)
                    except Exception:
                        r = float('inf')
                    solution_rewards.append(r)
                
                # Generate solution pairs with meaningful reward differences
                pairs_generated = 0
                max_attempts = num_pairs_per_polygon * 3
                attempts = 0
                
                while pairs_generated < num_pairs_per_polygon and attempts < max_attempts:
                    attempts += 1
                    
                    # Select two different solutions
                    if len(unique_solutions) < 2:
                        break
                    
                    np.random.seed(42 + i * 100000 + attempts)
                    idx1, idx2 = np.random.choice(len(unique_solutions), size=2, replace=False)
                    
                    sol1, sol2 = unique_solutions[idx1], unique_solutions[idx2]
                    r1, r2 = solution_rewards[idx1], solution_rewards[idx2]
                    
                    # Skip pairs with very similar rewards (ties)
                    if abs(r1 - r2) < 1e-6:
                        continue
                    
                    # Add both solutions to training data (order matters for ranking)
                    training_data.append({
                        'polygon': polygon_coords,
                        'solution': sol1,
                        'reward': r1,
                        'polygon_length': N,
                        'solution_length': len(sol1),
                        'polygon_name': polygon_name,
                        'pair_id': f"{polygon_name}_pair_{pairs_generated}",
                    })
                    
                    training_data.append({
                        'polygon': polygon_coords,
                        'solution': sol2,
                        'reward': r2,
                        'polygon_length': N,
                        'solution_length': len(sol2),
                        'polygon_name': polygon_name,
                        'pair_id': f"{polygon_name}_pair_{pairs_generated}",
                    })
                    
                    pairs_generated += 1
                    
                    # Check timeout
                    if time.time() - polygon_start_time > max_time_per_polygon:
                        break
                    
            except Exception as e:
                continue
    print(f"Generated {len(training_data)} training samples total")
    return training_data


def evaluate_ranker_performance(model, test_data, device, max_samples=1000):
    """Evaluate ranker performance metrics"""
    if not test_data:
        return {}
    
    model.eval()
    test_loader = DataLoader(
        SolutionDataset(test_data[:max_samples]), 
        batch_size=32, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    all_scores = []
    all_rewards = []
    polygon_stats = {}
    ranking_accuracy = []
    
    with torch.no_grad():
        eval_pbar = tqdm(test_loader, desc="Evaluating ranker performance", leave=False)
        for batch in eval_pbar:
            poly, sol, rewards, poly_lens, sol_lens, names, pair_ids = batch
            poly = poly.to(device)
            sol = sol.to(device)
            poly_lens = poly_lens.to(device)
            sol_lens = sol_lens.to(device)
            
            scores = model(poly, sol, poly_lens, sol_lens)
            
            # Collect data for analysis
            for i, name in enumerate(names):
                score_val = scores[i].item()
                reward_val = rewards[i].item()
                all_scores.append(score_val)
                all_rewards.append(reward_val)
                
                if name not in polygon_stats:
                    polygon_stats[name] = {'scores': [], 'rewards': []}
                polygon_stats[name]['scores'].append(score_val)
                polygon_stats[name]['rewards'].append(reward_val)
    
    # Calculate ranking accuracy per polygon
    for name, stats in polygon_stats.items():
        if len(stats['scores']) < 2:
            continue
        
        scores = np.array(stats['scores'])
        rewards = np.array(stats['rewards'])
        
        # Calculate pairwise ranking accuracy
        correct_pairs = 0
        total_pairs = 0
        
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                if abs(rewards[i] - rewards[j]) > 1e-8:  # Skip ties
                    # Higher score should correspond to lower reward (better solution)
                    score_preference = scores[i] > scores[j]
                    reward_preference = rewards[i] < rewards[j]
                    
                    if score_preference == reward_preference:
                        correct_pairs += 1
                    total_pairs += 1
        
        if total_pairs > 0:
            ranking_accuracy.append(correct_pairs / total_pairs)
    
    # Calculate correlation
    if len(all_scores) > 1:
        correlation = np.corrcoef(all_scores, all_rewards)[0, 1]
        # Since higher scores should correspond to lower rewards, we expect negative correlation
        correlation = -correlation if correlation < 0 else correlation
    else:
        correlation = 0.0
    
    performance_stats = {
        'num_samples': len(all_scores),
        'score_stats': {
            'mean': float(np.mean(all_scores)),
            'std': float(np.std(all_scores)),
            'min': float(np.min(all_scores)),
            'max': float(np.max(all_scores))
        },
        'reward_stats': {
            'mean': float(np.mean(all_rewards)),
            'std': float(np.std(all_rewards)),
            'min': float(np.min(all_rewards)),
            'max': float(np.max(all_rewards))
        },
        'ranking_accuracy': float(np.mean(ranking_accuracy)) if ranking_accuracy else 0.0,
        'score_reward_correlation': float(correlation),
        'num_polygons_evaluated': len(polygon_stats)
    }
    
    return performance_stats


def train_ranker(training_data, epochs=30, batch_size=32, lr=1e-3, val_split=0.1,
                 embedding_size=128, hidden_size=256):
    os.makedirs('checkpoints', exist_ok=True)
    np.random.shuffle(training_data)
    
    # Debug: check reward distribution
    rewards = [item['reward'] for item in training_data]
    print(f"Training data: {len(training_data)} samples")
    print(f"Reward stats: min={min(rewards):.4f}, max={max(rewards):.4f}, mean={np.mean(rewards):.4f}")
    
    split = int(len(training_data) * (1 - val_split))
    train_data = training_data[:split]
    val_data = training_data[split:]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RankerNet(embedding_size=embedding_size, hidden_size=hidden_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Create datasets
    train_dataset = SolutionDataset(train_data)
    val_dataset = SolutionDataset(val_data)
    
    # Use custom batch sampler for training to keep pairs together
    train_batch_sampler = PairBatchSampler(train_dataset, batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, collate_fn=collate_fn)
    
    # For validation, use regular batching since we don't need shuffling
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    best_val = float('inf')
    patience = 10
    no_improve = 0
    ckpt_name = f"ranker_net_embedding_size{embedding_size}_hidden_size{hidden_size}_epochs{epochs}.pt"
    best_path = os.path.join('checkpoints', ckpt_name.replace('.pt', '_best.pt'))
    final_path = os.path.join('checkpoints', ckpt_name)

    # Track training statistics
    training_stats = {
        'train_losses': [],
        'val_losses': [],
        'epochs_completed': 0,
        'best_epoch': 0,
        'early_stopped': False,
        'training_time': 0.0
    }
    
    training_start_time = time.time()

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        total_batches = 0
        skipped_batches = 0
        
        # Training progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False)
        for batch in train_pbar:
            total_batches += 1
            poly, sol, rewards, poly_lens, sol_lens, names, pair_ids = batch
            poly = poly.to(device); sol = sol.to(device)
            rewards = rewards.to(device); poly_lens = poly_lens.to(device); sol_lens = sol_lens.to(device)
            scores = model(poly, sol, poly_lens, sol_lens)
            
            # New approach: use pair_ids to group solutions from the same pair
            groups = {}
            if pair_ids[0] is not None:  # New pair-based data
                for idx, pair_id in enumerate(pair_ids):
                    if pair_id is not None:
                        groups.setdefault(pair_id, []).append(idx)
            else:  # Fallback to polygon-based grouping for compatibility
                for idx, nm in enumerate(names):
                    groups.setdefault(nm, []).append(idx)
            
            pair_loss = torch.tensor(0.0, device=device)
            pair_cnt = 0
            total_pairs_this_batch = 0
            
            for group_key, idxs in groups.items():
                if len(idxs) < 2:
                    continue
                    
                # For pair-based data, we expect exactly 2 items per group
                if pair_ids[0] is not None and len(idxs) == 2:
                    # Direct pair comparison
                    i, j = idxs[0], idxs[1]
                    total_pairs_this_batch += 1
                    diff = rewards[j] - rewards[i]
                    abs_diff = torch.abs(diff)
                    
                    # Skip only exact ties
                    if abs_diff < 1e-8:
                        continue
                    
                    # Weight near-ties less heavily
                    weight = 0.1 if abs_diff < 0.01 else 1.0
                    
                    y = torch.sign(diff)
                    pair_loss = pair_loss + weight * F.softplus(-y * (scores[i] - scores[j]))
                    pair_cnt += 1
                else:
                    # Fallback: all pairs within group (original approach)
                    pairs = [(i, j) for ii, i in enumerate(idxs) for j in idxs[ii+1:]]
                    if len(pairs) > 32:
                        ridx = np.random.choice(len(pairs), size=32, replace=False)
                        pairs = [pairs[k] for k in ridx]
                    for i, j in pairs:
                        total_pairs_this_batch += 1
                        diff = rewards[j] - rewards[i]
                        abs_diff = torch.abs(diff)
                        
                        if abs_diff < 1e-8:
                            continue
                        elif abs_diff < 0.01:
                            weight = 0.1
                        else:
                            weight = 1.0
                        
                        y = torch.sign(diff)
                        pair_loss = pair_loss + weight * F.softplus(-y * (scores[i] - scores[j]))
                        pair_cnt += 1
            
            if pair_cnt == 0:
                skipped_batches += 1
                continue
                
            loss = pair_loss / pair_cnt
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Pairs': pair_cnt,
                'Skipped': skipped_batches
            })
        
        processed_batches = total_batches - skipped_batches
        tr_loss /= max(1, processed_batches)

        # Validation
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation", leave=False)
            for batch in val_pbar:
                poly, sol, rewards, poly_lens, sol_lens, names, pair_ids = batch
                poly = poly.to(device); sol = sol.to(device)
                rewards = rewards.to(device); poly_lens = poly_lens.to(device); sol_lens = sol_lens.to(device)
                scores = model(poly, sol, poly_lens, sol_lens)
                
                groups = {}
                if pair_ids[0] is not None:  # New pair-based data
                    for idx, pair_id in enumerate(pair_ids):
                        if pair_id is not None:
                            groups.setdefault(pair_id, []).append(idx)
                else:  # Fallback to polygon-based grouping
                    for idx, nm in enumerate(names):
                        groups.setdefault(nm, []).append(idx)
                
                pair_loss = torch.tensor(0.0, device=device)
                pair_cnt = 0
                
                for group_key, idxs in groups.items():
                    if len(idxs) < 2:
                        continue
                        
                    if pair_ids[0] is not None and len(idxs) == 2:
                        # Direct pair comparison
                        i, j = idxs[0], idxs[1]
                        diff = rewards[j] - rewards[i]
                        abs_diff = torch.abs(diff)
                        
                        if abs_diff < 1e-8:
                            continue
                            
                        y = torch.sign(diff)
                        pair_loss = pair_loss + F.softplus(-y * (scores[i] - scores[j]))
                        pair_cnt += 1
                    else:
                        # Fallback: all pairs within group
                        pairs = [(i, j) for ii, i in enumerate(idxs) for j in idxs[ii+1:]]
                        if len(pairs) > 64:
                            ridx = np.random.choice(len(pairs), size=64, replace=False)
                            pairs = [pairs[k] for k in ridx]
                        for i, j in pairs:
                            diff = rewards[j] - rewards[i]
                            abs_diff = torch.abs(diff)
                            
                            if abs_diff < 1e-8:
                                continue
                                
                            y = torch.sign(diff)
                            pair_loss = pair_loss + F.softplus(-y * (scores[i] - scores[j]))
                            pair_cnt += 1
                
                if pair_cnt > 0:
                    va_loss += (pair_loss / pair_cnt).item()
                    
                # Update validation progress bar
                val_pbar.set_postfix({'Val Loss': f'{(pair_loss / max(1, pair_cnt)).item():.4f}'})
                    
        va_loss /= max(1, len(val_loader))
        dt = time.time() - t0
        
        # Record statistics
        training_stats['train_losses'].append(float(tr_loss))
        training_stats['val_losses'].append(float(va_loss))
        training_stats['epochs_completed'] = epoch + 1
        
        print(f"Epoch {epoch+1}/{epochs} ({dt:.1f}s) - Train Loss: {tr_loss:.4f}, Val Loss: {va_loss:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            no_improve = 0
            training_stats['best_epoch'] = epoch + 1
            meta = {
                'model_type': 'ranker',
                'embedding_size': embedding_size,
                'hidden_size': hidden_size,
                'target': 'rank',
            }
            torch.save({'model_state_dict': model.state_dict(), 'metadata': meta}, best_path)
            print(f"  → New best model saved to {best_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                training_stats['early_stopped'] = True
                break

    training_stats['training_time'] = time.time() - training_start_time

    meta = {
        'model_type': 'ranker',
        'embedding_size': embedding_size,
        'hidden_size': hidden_size,
        'target': 'rank',
    }
    torch.save({'model_state_dict': model.state_dict(), 'metadata': meta}, final_path)
    print(f"Final model saved to {final_path}")
    # Load best for return
    best_state = torch.load(best_path, map_location='cpu')
    model.load_state_dict(best_state['model_state_dict'])
    return model, best_path, final_path, training_stats


def main():
    load_dotenv()
    DATASET_PATH = os.getenv('DATASET_PATH')
    if not DATASET_PATH:
        raise EnvironmentError("DATASET_PATH environment variable must be set in .env file.")
    parser = argparse.ArgumentParser(description='Train RankerNet for AGP')
    parser.add_argument('--mode', choices=['generate', 'train', 'both'], default='both')
    parser.add_argument('--actor-checkpoint', type=str, default='checkpoints/rl_agp_model_embedding_size128_hidden_size128_n_glimpses1_tanh_exploration10_temperature1.0_use_tanhTrue_epochs30.pt')
    parser.add_argument('--train-dir', type=str, default=os.path.join(DATASET_PATH, 'train'))
    parser.add_argument('--samples-per-polygon', type=int, default=32,
                       help='Number of solution pairs to generate per polygon')
    parser.add_argument('--max-polygons', type=int, default=None,
                       help='Maximum number of polygons to process (for testing)')
    parser.add_argument('--data-file', type=str, default='data/ranker_training_data.pkl')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='checkpoints/ranker_net.pt')
    parser.add_argument('--value-embedding-size', type=int, default=128)
    parser.add_argument('--value-hidden-size', type=int, default=256)
    # actor arch (for sampling)
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--n-glimpses', type=int, default=1)
    parser.add_argument('--tanh-exploration', type=int, default=10)
    parser.add_argument('--use-tanh', action='store_true', default=True)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--normalize', action='store_true', default=True)
    # Evaluation and output arguments
    parser.add_argument('--stats-output', type=str, default='checkpoints/ranker_training_stats.json',
                       help='Output file for training statistics JSON')
    parser.add_argument('--evaluate', action='store_true', default=True,
                       help='Run internal ranking evaluation after training')
    args = parser.parse_args()

    if args.mode in ['generate', 'both']:
        print("=== DATA GENERATION PHASE ===")
        train_files = [os.path.join(args.train_dir, f) for f in os.listdir(args.train_dir) if f.endswith('.pol')]
        if args.max_polygons:
            train_files = train_files[:args.max_polygons]
        train_samples = agp_read_samples(train_files, normalize=args.normalize)
        train_dataset = AGPDataset(train_samples)
        print(f"Loaded {len(train_dataset)} training polygons")
        actor_model, device = load_trained_actor(
            args.actor_checkpoint,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            n_glimpses=args.n_glimpses,
            tanh_exploration=args.tanh_exploration,
            use_tanh=args.use_tanh,
            temperature=args.temperature,
        )
        training_data = generate_training_data(actor_model, train_dataset, args.samples_per_polygon, device)
        with open(args.data_file, 'wb') as f:
            pickle.dump(training_data, f)
        print(f"Training data saved to {args.data_file}")
        metadata = {
            'num_samples': len(training_data),
            'num_polygons': len(train_dataset),
            'samples_per_polygon': args.samples_per_polygon,
            'actor_checkpoint': args.actor_checkpoint,
            'args': vars(args),
        }
        with open(args.data_file.replace('.pkl', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    if args.mode in ['train', 'both']:
        print("=== TRAINING PHASE ===")
        with open(args.data_file, 'rb') as f:
            training_data = pickle.load(f)
        print(f"Loaded {len(training_data)} training samples")
        model, best_path, final_path, training_stats = train_ranker(
            training_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_split=args.val_split,
            embedding_size=args.value_embedding_size,
            hidden_size=args.value_hidden_size,
        )
        # Save a copy to args.output
        torch.save({'model_state_dict': model.state_dict(), 'metadata': {'model_type': 'ranker'}}, args.output)
        print(f"\nCheckpoint summary:\n  Best model saved to: {best_path}\n  Final model saved to: {final_path}\n  Copy also saved to: {args.output}")
        
        # Compile comprehensive statistics
        stats = {
            'training_configuration': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'val_split': args.val_split,
                'embedding_size': args.value_embedding_size,
                'hidden_size': args.value_hidden_size,
                'num_training_samples': len(training_data),
            },
            'training_stats': training_stats,
            'model_paths': {
                'best_model': best_path,
                'final_model': final_path,
                'output_copy': args.output
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        # Evaluate ranker performance if requested
        if args.evaluate:
            print("\n=== EVALUATION PHASE ===")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Use validation split for internal evaluation
            split = int(len(training_data) * (1 - args.val_split))
            val_data = training_data[split:]
            
            # Internal ranker performance evaluation
            print("Evaluating ranker performance on validation data...")
            ranker_performance = evaluate_ranker_performance(model, val_data, device)
            stats['ranker_performance'] = ranker_performance
        else:
            stats['ranker_performance'] = {'evaluated': False, 'reason': 'Evaluation disabled'}
        
        # Save comprehensive statistics to JSON
        print(f"\nSaving comprehensive training and evaluation statistics to {args.stats_output}")
        with open(args.stats_output, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n=== SUMMARY ===")
        print(f"Training completed in {training_stats['training_time']:.1f}s over {training_stats['epochs_completed']} epochs")
        if training_stats['early_stopped']:
            print(f"Early stopping triggered, best model at epoch {training_stats['best_epoch']}")
        
        if args.evaluate and 'ranker_performance' in stats and stats['ranker_performance'].get('evaluated', True):
            rp = stats['ranker_performance']
            print(f"Ranker Performance:")
            print(f"  - Ranking accuracy: {rp['ranking_accuracy']:.3f}")
            print(f"  - Score-reward correlation: {rp['score_reward_correlation']:.3f}")
            print(f"  - Evaluated on {rp['num_samples']} samples from {rp['num_polygons_evaluated']} polygons")
        
        print(f"All statistics saved to: {args.stats_output}")


if __name__ == '__main__':
    main()
