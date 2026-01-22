#!/usr/bin/env python3
"""
Simplified training script for Value Network.
Generates random guard placements and trains network to predict coverage.
No dependency on trained actor - just pure supervised learning.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import json
from dotenv import load_dotenv

from dataset import agp_read_samples
from utils import evaluate_polygon_visibility_numpy_wo_gt


class SimpleValueNet(nn.Module):
    """
    Simple network: polygon coords + guard indices -> coverage prediction
    """
    def __init__(self, embedding_size=128, hidden_size=256):
        super().__init__()
        # Encode polygon coordinates
        self.polygon_encoder = nn.LSTM(2, embedding_size, batch_first=True)
        # Encode guard positions (as embeddings of indices)
        self.guard_embedding = nn.Embedding(2000, embedding_size)  # support up to 2000 vertices
        self.guard_encoder = nn.LSTM(embedding_size, embedding_size, batch_first=True)
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Coverage is in [0, 1]
        )
    
    def forward(self, polygon_coords, guard_indices, poly_lengths, sol_lengths):
        # Encode polygon
        packed_poly = pack_padded_sequence(polygon_coords, poly_lengths.cpu(), 
                                          batch_first=True, enforce_sorted=False)
        _, (poly_h, _) = self.polygon_encoder(packed_poly)
        poly_features = poly_h[-1]
        
        # Encode guards
        guard_embeds = self.guard_embedding(guard_indices)
        packed_guards = pack_padded_sequence(guard_embeds, sol_lengths.cpu(), 
                                            batch_first=True, enforce_sorted=False)
        _, (guard_h, _) = self.guard_encoder(packed_guards)
        guard_features = guard_h[-1]
        
        # Fuse and predict
        combined = torch.cat([poly_features, guard_features], dim=1)
        coverage = self.fusion(combined).squeeze(-1)
        return coverage


def generate_random_guards(polygon_length, min_guards=1, max_guards=None):
    """Generate random guard placement for a polygon."""
    if max_guards is None:
        max_guards = max(polygon_length // 3, 1)  # Use at most 1/3 of vertices
    
    num_guards = np.random.randint(min_guards, max_guards + 1)
    guards = sorted(np.random.choice(polygon_length, size=num_guards, replace=False).tolist())
    return guards


def generate_training_data(polygon_files, samples_per_polygon=50, normalize=True):
    """
    Generate training data: random guard placements + actual coverage.
    Much simpler than sampling from actor.
    """
    print(f"Generating training data from {len(polygon_files)} polygons...")
    
    training_data = []
    
    for i, pol_file in enumerate(polygon_files):
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(polygon_files)} polygons...")
        try:
            # Read polygon using existing dataset functions
            with open(pol_file, 'r') as f:
                tokens = f.read().split()
                num_points = int(tokens[0])
                points = []
                for i in range(1, 2 * num_points, 2):
                    x_token = tokens[i]
                    y_token = tokens[i + 1]
                    # Parse coordinates
                    if '/' in x_token:
                        x_num, x_denom = map(float, x_token.split('/'))
                        x = x_num / x_denom
                    else:
                        x = float(x_token)
                    if '/' in y_token:
                        y_num, y_denom = map(float, y_token.split('/'))
                        y = y_num / y_denom
                    else:
                        y = float(y_token)
                    points.append([x, y])
            
            polygon_coords = np.array(points, dtype=np.float32)
            polygon_name = os.path.splitext(os.path.basename(pol_file))[0]
            
            # Normalize if needed
            if normalize:
                center = polygon_coords.mean(axis=0)
                polygon_coords = polygon_coords - center
                scale = np.abs(polygon_coords).max()
                if scale > 0:
                    polygon_coords = polygon_coords / scale
            
            polygon_length = len(polygon_coords)
            
            # Generate multiple random guard placements
            for _ in range(samples_per_polygon):
                guards = generate_random_guards(polygon_length)
                
                # Compute coverage
                coverage = float(evaluate_polygon_visibility_numpy_wo_gt(
                    polygon_coords, guards, polygon_name
                ))
                
                training_data.append({
                    'polygon': polygon_coords,
                    'guards': guards,
                    'coverage': coverage,
                    'polygon_length': polygon_length,
                    'num_guards': len(guards),
                    'polygon_name': polygon_name
                })
        
        except Exception:
            continue
    
    print(f"Generated {len(training_data)} valid training samples")
    return training_data


class CoverageDataset(Dataset):
    """Simple dataset for coverage prediction."""
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'polygon': torch.FloatTensor(item['polygon']),
            'guards': torch.LongTensor(item['guards']),
            'coverage': torch.FloatTensor([item['coverage']]),
            'poly_length': item['polygon_length'],
            'num_guards': item['num_guards']
        }


def collate_batch(batch):
    """Collate function for DataLoader."""
    polygons = [item['polygon'] for item in batch]
    guards = [item['guards'] for item in batch]
    coverages = torch.cat([item['coverage'] for item in batch])
    poly_lengths = torch.LongTensor([item['poly_length'] for item in batch])
    guard_lengths = torch.LongTensor([item['num_guards'] for item in batch])
    
    # Pad sequences
    polygons_padded = pad_sequence(polygons, batch_first=True, padding_value=0)
    guards_padded = pad_sequence(guards, batch_first=True, padding_value=0)
    
    return polygons_padded, guards_padded, coverages, poly_lengths, guard_lengths


def compute_metrics(predictions, targets):
    """Compute regression metrics."""
    pred_np = predictions.cpu().numpy() if torch.is_tensor(predictions) else predictions
    target_np = targets.cpu().numpy() if torch.is_tensor(targets) else targets
    
    mae = float(np.mean(np.abs(pred_np - target_np)))
    mse = float(np.mean((pred_np - target_np) ** 2))
    rmse = float(np.sqrt(mse))
    
    ss_res = np.sum((target_np - pred_np) ** 2)
    ss_tot = np.sum((target_np - np.mean(target_np)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    correlation = float(np.corrcoef(target_np, pred_np)[0, 1]) if len(target_np) > 1 else 0.0
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'correlation': correlation
    }


def grid_search(train_data, val_data, param_grid=None):
    """Perform grid search over hyperparameters."""
    
    if param_grid is None:
        param_grid = {
            'embedding_size': [64, 128, 256],
            'hidden_size': [128, 256, 512],
            'batch_size': [16, 32, 64],
            'lr': [1e-4, 1e-3, 1e-2]
        }
    
    import itertools
    import time
    from tqdm import tqdm
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    print(f"\n{'='*80}")
    print(f"GRID SEARCH: Testing {len(combinations)} hyperparameter combinations")
    print(f"{'='*80}")
    
    best_score = float('inf')
    best_params = None
    best_stats = None
    best_model_path = None
    
    results = []
    
    # Overall progress bar for grid search
    grid_pbar = tqdm(total=len(combinations), desc="Grid Search", unit="config", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for i, combo in enumerate(combinations):
        params = dict(zip(param_names, combo))
        
        grid_pbar.set_description(f"Grid Search (Config {i+1}/{len(combinations)})")
        grid_pbar.set_postfix({
            'emb': params['embedding_size'],
            'hid': params['hidden_size'], 
            'bs': params['batch_size'],
            'lr': f"{params['lr']:.0e}"
        })
        
        print(f"\n--- Combination {i+1}/{len(combinations)} ---")
        print(f"Parameters: {params}")
        
        start_time = time.time()
        
        try:
            # Train model with these parameters
            model, model_path, stats = train_model(
                train_data, val_data,
                embedding_size=params['embedding_size'],
                hidden_size=params['hidden_size'],
                epochs=30,  # Reduced epochs for grid search
                batch_size=params['batch_size'],
                lr=params['lr'],
                show_progress=False  # Disable progress bars during grid search
            )
            
            val_loss = stats['best_val_loss']
            val_mae = stats['final_val_metrics']['mae']
            val_r2 = stats['final_val_metrics']['r2']
            val_corr = stats['final_val_metrics']['correlation']
            
            combo_result = {
                'params': params,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_r2': val_r2,
                'val_corr': val_corr,
                'training_time': time.time() - start_time,
                'model_path': model_path
            }
            
            results.append(combo_result)
            
            print(f"Results: Loss={val_loss:.6f}, MAE={val_mae:.6f}, R²={val_r2:.4f}, Corr={val_corr:.4f}")
            
            # Check if this is the best so far
            if val_loss < best_score:
                best_score = val_loss
                best_params = params
                best_stats = stats
                best_model_path = model_path
                print("  → NEW BEST!")
            
        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue
        
        # Update overall progress
        grid_pbar.update(1)
    
    grid_pbar.close()
    
    # Print summary
    print(f"\n{'='*80}")
    print("GRID SEARCH RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Sort results by validation loss
    results.sort(key=lambda x: x['val_loss'])
    
    print(f"{'Rank':<4} {'Embedding':<9} {'Hidden':<7} {'Batch':<6} {'LR':<8} "
          f"{'Val Loss':<10} {'MAE':<8} {'R²':<6} {'Corr':<6} {'Time':<8}")
    print("-" * 90)
    
    for rank, result in enumerate(results[:10], 1):  # Show top 10
        params = result['params']
        print(f"{rank:<4} {params['embedding_size']:<9} {params['hidden_size']:<7} "
              f"{params['batch_size']:<6} {params['lr']:<8.0e} "
              f"{result['val_loss']:<10.6f} {result['val_mae']:<8.4f} "
              f"{result['val_r2']:<6.3f} {result['val_corr']:<6.3f} "
              f"{result['training_time']:<8.1f}s")
    
    print(f"\n{'='*80}")
    print("BEST HYPERPARAMETERS FOUND:")
    print(f"{'='*80}")
    print(f"Embedding Size: {best_params['embedding_size']}")
    print(f"Hidden Size: {best_params['hidden_size']}")
    print(f"Batch Size: {best_params['batch_size']}")
    print(f"Learning Rate: {best_params['lr']}")
    print(f"Best Validation Loss: {best_score:.6f}")
    print(f"Model saved at: {best_model_path}")
    
    # Save grid search results
    grid_results = {
        'grid_search_summary': {
            'total_combinations': len(combinations),
            'best_params': best_params,
            'best_val_loss': best_score,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'all_results': results
    }
    
    os.makedirs('results', exist_ok=True)
    grid_filename = f'grid_search_results_{time.strftime("%Y%m%d_%H%M%S")}.json'
    grid_path = os.path.join('results', grid_filename)
    
    with open(grid_path, 'w') as f:
        json.dump(grid_results, f, indent=2)
    
    print(f"Grid search results saved to: {grid_path}")
    
    return best_params, best_stats, best_model_path


def train_model(train_data, val_data, embedding_size=128, hidden_size=256, 
                epochs=50, batch_size=32, lr=1e-3, show_progress=True):
    """Train the simple value network."""
    
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Create datasets
    train_dataset = CoverageDataset(train_data)
    val_dataset = CoverageDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, collate_fn=collate_batch)
    
    # Create model
    model = SimpleValueNet(embedding_size=embedding_size, hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 10
    patience_counter = 0
    
    # Track statistics
    train_losses = []
    val_losses = []
    epoch_times = []
    
    os.makedirs('checkpoints', exist_ok=True)
    best_path = f'checkpoints/simple_value_net_best.pt'
    
    print(f"\nTraining for {epochs} epochs...")
    training_start_time = time.time()
    
    # Set up progress bar if requested
    epoch_iterator = range(epochs)
    if show_progress:
        from tqdm import tqdm
        epoch_iterator = tqdm(epoch_iterator, desc="Training", unit="epoch")
    
    for epoch in epoch_iterator:
        epoch_start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        # Training progress bar
        train_iterator = train_loader
        if show_progress:
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False, unit="batch")
        
        for poly, guards, cov, poly_len, guard_len in train_iterator:
            poly = poly.to(device)
            guards = guards.to(device)
            cov = cov.to(device)
            
            optimizer.zero_grad()
            pred = model(poly, guards, poly_len, guard_len)
            loss = criterion(pred, cov)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.append(pred.detach().cpu())
            train_targets.append(cov.cpu())
            
            if show_progress:
                train_iterator.set_postfix({'train_loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_metrics = compute_metrics(train_preds, train_targets)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        # Validation progress bar
        val_iterator = val_loader
        if show_progress:
            val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1} Val", leave=False, unit="batch")
        
        with torch.no_grad():
            for poly, guards, cov, poly_len, guard_len in val_iterator:
                poly = poly.to(device)
                guards = guards.to(device)
                cov = cov.to(device)
                
                pred = model(poly, guards, poly_len, guard_len)
                loss = criterion(pred, cov)
                val_loss += loss.item()
                val_preds.append(pred.cpu())
                val_targets.append(cov.cpu())
                
                if show_progress:
                    val_iterator.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader)
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_metrics = compute_metrics(val_preds, val_targets)
        
        epoch_time = time.time() - epoch_start_time
        
        # Store statistics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        epoch_times.append(epoch_time)
        
        # Print comprehensive epoch statistics
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train MAE: {train_metrics['mae']:.4f} | Val MAE: {train_metrics['mae']:.4f} | "
              f"Train R²: {train_metrics['r2']:.3f} | Val R²: {val_metrics['r2']:.3f} | "
              f"Train Corr: {train_metrics['correlation']:.3f} | Val Corr: {val_metrics['correlation']:.3f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'embedding_size': embedding_size,
                'hidden_size': hidden_size,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, best_path)
            print(f"  → New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    total_training_time = time.time() - training_start_time
    
    # Final evaluation on best model
    print("\n" + "="*80)
    print("FINAL EVALUATION ON BEST MODEL")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate on train set
    train_preds = []
    train_targets = []
    with torch.no_grad():
        for poly, guards, cov, poly_len, guard_len in train_loader:
            poly = poly.to(device)
            guards = guards.to(device)
            pred = model(poly, guards, poly_len, guard_len)
            train_preds.append(pred.cpu())
            train_targets.append(cov)
    
    train_preds = torch.cat(train_preds)
    train_targets = torch.cat(train_targets)
    final_train_metrics = compute_metrics(train_preds, train_targets)
    
    # Evaluate on val set
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for poly, guards, cov, poly_len, guard_len in val_loader:
            poly = poly.to(device)
            guards = guards.to(device)
            pred = model(poly, guards, poly_len, guard_len)
            val_preds.append(pred.cpu())
            val_targets.append(cov)
    
    val_preds = torch.cat(val_preds)
    val_targets = torch.cat(val_targets)
    final_val_metrics = compute_metrics(val_preds, val_targets)
    
    # Compute final statistics
    stats = {
        'best_epoch': best_epoch + 1,
        'total_epochs': len(train_losses),
        'best_val_loss': best_val_loss,
        'final_train_metrics': final_train_metrics,
        'final_val_metrics': final_val_metrics,
        'total_training_time_seconds': total_training_time,
        'avg_epoch_time_seconds': np.mean(epoch_times)
    }
    
    # Print comprehensive final results
    print(f"{'Dataset':<10} {'MAE':<8} {'RMSE':<8} {'R²':<8} {'Corr':<8}")
    print("-" * 50)
    print(f"{'Train':<10} {final_train_metrics['mae']:<8.4f} {final_train_metrics['rmse']:<8.4f} "
          f"{final_train_metrics['r2']:<8.3f} {final_train_metrics['correlation']:<8.3f}")
    print(f"{'Val':<10} {final_val_metrics['mae']:<8.4f} {final_val_metrics['rmse']:<8.4f} "
          f"{final_val_metrics['r2']:<8.3f} {final_val_metrics['correlation']:<8.3f}")
    
    print(f"\n{'='*80}")
    print(f"TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Best Epoch: {stats['best_epoch']}/{stats['total_epochs']}")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Total Training Time: {total_training_time:.1f}s ({total_training_time/60:.1f}min)")
    print(f"Average Epoch Time: {stats['avg_epoch_time_seconds']:.1f}s")
    print(f"Model saved to: {best_path}")
    print(f"{'='*80}")
    
    return model, best_path, stats


def main():
    load_dotenv()
    DATASET_PATH = os.getenv("DATASET_PATH")
    
    parser = argparse.ArgumentParser(description='Train Simple Value Network')
    parser.add_argument('--mode', choices=['generate', 'train', 'both', 'grid_search'], default='both')
    parser.add_argument('--train-dir', type=str, 
                       default=os.path.join(DATASET_PATH, "train"))
    parser.add_argument('--samples-per-polygon', type=int, default=50)
    parser.add_argument('--max-polygons', type=int, default=None)
    parser.add_argument('--data-file', type=str, default='data/simple_value_net_data.pkl')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--hidden-size', type=int, default=256)
    parser.add_argument('--normalize', action='store_true', default=True)
    
    args = parser.parse_args()
    
    if args.mode in ['generate', 'both']:
        print("=== GENERATING DATA ===")
        
        # Get polygon files
        pol_files = [os.path.join(args.train_dir, f) 
                    for f in os.listdir(args.train_dir) if f.endswith('.pol')]
        
        if args.max_polygons:
            pol_files = pol_files[:args.max_polygons]
        
        # Generate training data
        training_data = generate_training_data(
            pol_files, 
            samples_per_polygon=args.samples_per_polygon,
            normalize=args.normalize
        )
        
        # Save
        with open(args.data_file, 'wb') as f:
            pickle.dump(training_data, f)
        
        print(f"Saved to {args.data_file}")
        
        # Save metadata
        metadata = {
            'num_samples': len(training_data),
            'num_polygons': len(pol_files),
            'samples_per_polygon': args.samples_per_polygon,
            'args': vars(args)
        }
        with open(args.data_file.replace('.pkl', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    if args.mode in ['train', 'both', 'grid_search']:
        print("\n=== TRAINING MODEL ===")
        
        # Load data
        with open(args.data_file, 'rb') as f:
            training_data = pickle.load(f)
        
        print(f"Loaded {len(training_data)} samples")
        
        # Split
        np.random.shuffle(training_data)
        split_idx = int(len(training_data) * (1 - args.val_split))
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        print(f"Train: {len(train_data)}, Val: {len(val_data)}")
        
        if args.mode == 'grid_search':
            # Perform grid search
            best_params, best_stats, best_model_path = grid_search(train_data, val_data)
            
            # Train final model with best parameters for full epochs
            print(f"\n{'='*80}")
            print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
            print(f"{'='*80}")
            
            model, best_path, training_stats = train_model(
                train_data, val_data,
                embedding_size=best_params['embedding_size'],
                hidden_size=best_params['hidden_size'],
                epochs=args.epochs,  # Use full epochs for final training
                batch_size=best_params['batch_size'],
                lr=best_params['lr']
            )
        else:
            # Train with specified parameters
            model, best_path, training_stats = train_model(
                train_data, val_data,
                embedding_size=args.embedding_size,
                hidden_size=args.hidden_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr
            )
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Save comprehensive results to JSON
        import time as time_module
        timestamp = time_module.strftime('%Y%m%d_%H%M%S')
        
        # Use best parameters if grid search was performed
        if args.mode == 'grid_search':
            final_embedding_size = best_params['embedding_size']
            final_hidden_size = best_params['hidden_size']
            final_batch_size = best_params['batch_size']
            final_lr = best_params['lr']
        else:
            final_embedding_size = args.embedding_size
            final_hidden_size = args.hidden_size
            final_batch_size = args.batch_size
            final_lr = args.lr
        
        results = {
            'experiment': {
                'name': 'simple_value_net_training' + ('_grid_search' if args.mode == 'grid_search' else ''),
                'timestamp': time_module.strftime('%Y-%m-%d %H:%M:%S'),
                'model_type': 'SimpleValueNet'
            },
            'configuration': {
                'embedding_size': final_embedding_size,
                'hidden_size': final_hidden_size,
                'epochs': args.epochs,
                'batch_size': final_batch_size,
                'learning_rate': final_lr,
                'samples_per_polygon': args.samples_per_polygon,
                'max_polygons': args.max_polygons,
                'grid_search_performed': args.mode == 'grid_search'
            },
            'data_statistics': {
                'total_samples': len(training_data),
                'train_samples': len(train_data),
                'val_samples': len(val_data)
            },
            'training_results': {
                'best_epoch': training_stats['best_epoch'],
                'best_val_loss': training_stats['best_val_loss'],
                'total_training_time_seconds': training_stats['total_training_time_seconds']
            },
            'evaluation_metrics': {
                'train_set': training_stats['final_train_metrics'],
                'val_set': training_stats['final_val_metrics']
            },
            'model_checkpoint': {
                'path': best_path
            }
        }
        
        # Save with descriptive filename
        results_filename = f'simple_value_net_emb{final_embedding_size}_hid{final_hidden_size}_epochs{args.epochs}.json'
        results_path = os.path.join('results', results_filename)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*40}")
        print(f"RESULTS SAVED")
        print(f"{'='*40}")
        print(f"Results file: {results_path}")
        print(f"Model checkpoint: {best_path}")
        print(f"\nFinal Validation Metrics:")
        print(f"  MAE: {results['evaluation_metrics']['val_set']['mae']:.6f}")
        print(f"  R²:  {results['evaluation_metrics']['val_set']['r2']:.4f}")
        print(f"{'='*40}")
        print("\nTraining complete!")


if __name__ == "__main__":
    main()
