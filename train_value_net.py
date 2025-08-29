#!/usr/bin/env python3
"""
Training script for Solution-Aware Value Network.
This script generates training data from a trained RL actor and trains a value network
to predict rewards given polygon coordinates and guard solutions.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import json
import time
from dotenv import load_dotenv
from tqdm import tqdm

# Import from project files
from dataset import Dataset as AGPDataset, agp_read_samples, collate_fn
from models import create_actor
from rewards import enhanced_penalty as reward_fn
from rl_agp import BucketBatchSampler, get_lengths_from_dataset
from functools import partial


def compute_simple_metrics(predictions, targets):
    """Compute simple regression metrics without sklearn."""
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions_np - targets_np))
    
    # R² Score (coefficient of determination)
    ss_res = np.sum((targets_np - predictions_np) ** 2)
    ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((predictions_np - targets_np) ** 2))
    
    # Correlation coefficient
    correlation = np.corrcoef(targets_np, predictions_np)[0, 1] if len(targets_np) > 1 else 0
    
    return mae, r2, rmse, correlation


def print_training_statistics(train_losses, val_losses, train_maes, val_maes, 
                             train_r2s, val_r2s, epoch_times, best_epoch):
    """Print comprehensive training statistics."""
    
    print("\n" + "="*60)
    print("TRAINING STATISTICS SUMMARY")
    print("="*60)
    
    # Basic training info
    total_epochs = len(train_losses)
    total_time = sum(epoch_times)
    avg_time_per_epoch = np.mean(epoch_times)
    
    print(f"\nTRAINING OVERVIEW:")
    print(f"  Total Epochs: {total_epochs}")
    print(f"  Best Epoch: {best_epoch + 1}")
    print(f"  Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"  Average Time per Epoch: {avg_time_per_epoch:.2f} seconds")
    
    # Loss statistics
    print(f"\nLOSS STATISTICS:")
    print(f"  Initial Train Loss: {train_losses[0]:.6f}")
    print(f"  Final Train Loss: {train_losses[-1]:.6f}")
    print(f"  Best Train Loss: {min(train_losses):.6f}")
    print(f"  Initial Val Loss: {val_losses[0]:.6f}")
    print(f"  Final Val Loss: {val_losses[-1]:.6f}")
    print(f"  Best Val Loss: {min(val_losses):.6f}")
    
    # Improvement metrics
    train_improvement = ((train_losses[0] - min(train_losses)) / train_losses[0]) * 100
    val_improvement = ((val_losses[0] - min(val_losses)) / val_losses[0]) * 100
    
    print(f"\nIMPROVEMENT:")
    print(f"  Train Loss Improvement: {train_improvement:.2f}%")
    print(f"  Val Loss Improvement: {val_improvement:.2f}%")
    
    # Best epoch performance
    print(f"\nBEST MODEL PERFORMANCE (Epoch {best_epoch + 1}):")
    print(f"  Train Loss: {train_losses[best_epoch]:.6f}")
    print(f"  Val Loss: {val_losses[best_epoch]:.6f}")
    print(f"  Train MAE: {train_maes[best_epoch]:.6f}")
    print(f"  Val MAE: {val_maes[best_epoch]:.6f}")
    print(f"  Train R²: {train_r2s[best_epoch]:.6f}")
    print(f"  Val R²: {val_r2s[best_epoch]:.6f}")
    
    # Final performance
    print(f"\nFINAL MODEL PERFORMANCE:")
    print(f"  Train Loss: {train_losses[-1]:.6f}")
    print(f"  Val Loss: {val_losses[-1]:.6f}")
    print(f"  Train MAE: {train_maes[-1]:.6f}")
    print(f"  Val MAE: {val_maes[-1]:.6f}")
    print(f"  Train R²: {train_r2s[-1]:.6f}")
    print(f"  Val R²: {val_r2s[-1]:.6f}")
    
    # Overfitting analysis
    final_overfitting = val_losses[-1] / train_losses[-1]
    best_overfitting = val_losses[best_epoch] / train_losses[best_epoch]
    
    print(f"\nOVERFITTING ANALYSIS:")
    print(f"  Final Val/Train Loss Ratio: {final_overfitting:.4f}")
    print(f"  Best Epoch Val/Train Loss Ratio: {best_overfitting:.4f}")
    
    if final_overfitting < 1.2:
        overfitting_status = "Minimal overfitting"
    elif final_overfitting < 2.0:
        overfitting_status = "Moderate overfitting"
    else:
        overfitting_status = "Significant overfitting"
    
    print(f"  Overfitting Assessment: {overfitting_status}")
    
    # Training efficiency
    efficiency = min(val_losses) / total_time
    print(f"\nTRAINING EFFICIENCY:")
    print(f"  Best Loss per Second: {efficiency:.8f}")
    print(f"  Fastest Epoch: {min(epoch_times):.2f}s")
    print(f"  Slowest Epoch: {max(epoch_times):.2f}s")
    
    print("="*60)


class SolutionValueNet(nn.Module):
    """
    Neural network that predicts reward given polygon coordinates and guard solution.
    Input: polygon vertices + selected guard indices
    Output: predicted reward value
    """
    def __init__(self, embedding_size=128, hidden_size=256, max_vertices=1000):
        super().__init__()
        
        # Polygon encoder (similar to actor's encoder)
        self.polygon_encoder = nn.LSTM(2, embedding_size, batch_first=True)
        
        # Solution encoder - convert guard indices to features
        self.guard_embedding = nn.Embedding(max_vertices, embedding_size)
        self.solution_encoder = nn.LSTM(embedding_size, embedding_size, batch_first=True)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)  # single reward value
        )
        
    def forward(self, polygon_coords, guard_indices, polygon_lengths, solution_lengths):
        """
        Args:
            polygon_coords: [batch_size, max_poly_len, 2]
            guard_indices: [batch_size, max_sol_len]  
            polygon_lengths: [batch_size]
            solution_lengths: [batch_size]
        """
        batch_size = polygon_coords.size(0)
        device = polygon_coords.device
        
        # Encode polygon
        packed_poly = pack_padded_sequence(polygon_coords, polygon_lengths.cpu(), 
                                         batch_first=True, enforce_sorted=False)
        poly_out, (poly_hidden, _) = self.polygon_encoder(packed_poly)
        poly_features = poly_hidden[-1]  # [batch_size, embedding_size]
        
        # Encode solution
        guard_embeds = self.guard_embedding(guard_indices)  # [batch_size, max_guards, embedding_size]
        packed_sol = pack_padded_sequence(guard_embeds, solution_lengths.cpu(), 
                                        batch_first=True, enforce_sorted=False)
        sol_out, (sol_hidden, _) = self.solution_encoder(packed_sol)
        sol_features = sol_hidden[-1]  # [batch_size, embedding_size]
        
        # Fuse and predict
        combined = torch.cat([poly_features, sol_features], dim=1)
        reward_pred = self.fusion(combined)
        return reward_pred.squeeze(-1)


class SolutionDataset(Dataset):
    """Dataset for (polygon, solution, reward) triplets."""
    
    def __init__(self, training_data):
        self.data = training_data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.tensor(item['polygon'], dtype=torch.float32),
            torch.tensor(item['solution'], dtype=torch.long),
            torch.tensor(item['reward'], dtype=torch.float32),
            item['polygon_length'],
            item['solution_length']
        )


def solution_collate_fn(batch):
    """Collate function for solution dataset."""
    polygons, solutions, rewards, poly_lengths, sol_lengths = zip(*batch)
    
    # Pad sequences
    polygons_padded = pad_sequence(polygons, batch_first=True, padding_value=0.0)
    solutions_padded = pad_sequence(solutions, batch_first=True, padding_value=0)
    rewards_tensor = torch.stack(rewards)
    poly_lengths_tensor = torch.tensor(poly_lengths, dtype=torch.long)
    sol_lengths_tensor = torch.tensor(sol_lengths, dtype=torch.long)
    
    return polygons_padded, solutions_padded, rewards_tensor, poly_lengths_tensor, sol_lengths_tensor


def load_trained_actor(checkpoint_path, embedding_size=128, hidden_size=128, n_glimpses=1, 
                      tanh_exploration=10, use_tanh=True, temperature=1.0):
    """Load the trained RL actor from checkpoint."""
    print(f"Loading trained actor from: {checkpoint_path}")
    
    # Create model with same architecture as training
    model = create_actor(
        embedding_size=embedding_size,
        hidden_size=hidden_size,
        seq_len=None,
        n_glimpses=n_glimpses,
        tanh_exploration=tanh_exploration,
        use_tanh=use_tanh,
        attention_type="Bahdanau",
        reward_fn=reward_fn,
        temperature=temperature
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    print(f"Actor loaded on device: {device}")
    return model, device


def generate_training_data(actor_model, dataset, num_samples_per_polygon=100, device='cpu'):
    """Generate training data: (polygon, solution, reward) triplets"""
    print(f"Generating training data: {num_samples_per_polygon} samples per polygon...")
    training_data = []
    reward_func = partial(reward_fn, alpha=5.0, p=0.0)
    
    actor_model.eval()
    with torch.no_grad():
        for i, (polygon_data, _, polygon_name) in enumerate(tqdm(dataset, desc="Processing polygons")):
            polygon_coords = polygon_data.numpy()
            polygon_length = len(polygon_coords)
            
            # Sample multiple solutions for this polygon
            for _ in range(num_samples_per_polygon):
                try:
                    # Sample solution from trained actor
                    data_tensor = polygon_data.unsqueeze(0).to(device)  # add batch dim
                    mask = torch.zeros(1, len(polygon_coords), dtype=torch.bool, device=device)
                    lengths = torch.tensor([len(polygon_coords)], device=device)
                    
                    guard_indices, _ = actor_model(data_tensor, padding_mask=mask, lengths=lengths)
                    # guard_indices is a list with one element (list of indices)
                    solution = guard_indices[0]
                    
                    # Sanitize indices: clamp to [0, polygon_length-1] and remove duplicates while preserving order
                    if polygon_length > 0:
                        clamped = []
                        seen = set()
                        for idx in solution:
                            # Only accept non-negative integers
                            if not isinstance(idx, (int, np.integer)):
                                continue
                            if idx < 0:
                                idx = 0
                            if idx >= polygon_length:
                                idx = polygon_length - 1
                            if idx not in seen:
                                seen.add(idx)
                                clamped.append(idx)
                        solution = clamped
                    
                    if len(solution) == 0:
                        # If still empty, skip
                        continue
                    
                    # Compute exact reward (expensive)
                    reward = reward_func(polygon_coords, solution, polygon_name, length=polygon_length)
                    
                    training_data.append({
                        'polygon': polygon_coords,
                        'solution': solution,
                        'reward': reward,
                        'polygon_length': polygon_length,
                        'solution_length': len(solution),
                        'polygon_name': polygon_name
                    })
                    
                except Exception as e:
                    print(f"Error processing polygon {polygon_name}: {e}")
                    continue
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(dataset)} polygons, generated {len(training_data)} samples")
    
    print(f"Generated {len(training_data)} training samples total")
    return training_data


def train_value_net(training_data, epochs=50, batch_size=32, lr=1e-3, val_split=0.1, 
                    embedding_size=128, hidden_size=256):
    """Train the solution-aware value network"""
    print(f"Training value network on {len(training_data)} samples...")
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Split data
    np.random.shuffle(training_data)
    split_idx = int(len(training_data) * (1 - val_split))
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    # Create datasets
    train_dataset = SolutionDataset(train_data)
    val_dataset = SolutionDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             collate_fn=solution_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           collate_fn=solution_collate_fn)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SolutionValueNet(embedding_size=embedding_size, hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Statistics tracking
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    train_r2s = []
    val_r2s = []
    epoch_times = []
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Generate checkpoint filename similar to rl_agp.py and sl_agp.py
    checkpoint_name = f"value_net_embedding_size{embedding_size}_hidden_size{hidden_size}_epochs{epochs}.pt"
    best_checkpoint_path = os.path.join('checkpoints', checkpoint_name.replace('.pt', '_best.pt'))
    final_checkpoint_path = os.path.join('checkpoints', checkpoint_name)
    
    print(f"Training on device: {device}")
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    print(f"Will save best model to: {best_checkpoint_path}")
    print(f"Will save final model to: {final_checkpoint_path}")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_predictions = []
        train_targets = []
        
        for batch in train_loader:
            polygon_coords, guard_indices, rewards, poly_lengths, sol_lengths = \
                [x.to(device) for x in batch]
            
            # Forward pass
            pred_rewards = model(polygon_coords, guard_indices, poly_lengths, sol_lengths)
            loss = criterion(pred_rewards, rewards)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_predictions.append(pred_rewards.detach())
            train_targets.append(rewards.detach())
        
        # Compute training metrics
        train_loss /= len(train_loader)
        train_preds_tensor = torch.cat(train_predictions)
        train_targets_tensor = torch.cat(train_targets)
        train_mae, train_r2, _, _ = compute_simple_metrics(train_preds_tensor, train_targets_tensor)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                polygon_coords, guard_indices, rewards, poly_lengths, sol_lengths = \
                    [x.to(device) for x in batch]
                
                pred_rewards = model(polygon_coords, guard_indices, poly_lengths, sol_lengths)
                loss = criterion(pred_rewards, rewards)
                val_loss += loss.item()
                val_predictions.append(pred_rewards)
                val_targets.append(rewards)
        
        # Compute validation metrics
        val_loss /= len(val_loader)
        val_preds_tensor = torch.cat(val_predictions)
        val_targets_tensor = torch.cat(val_targets)
        val_mae, val_r2, _, _ = compute_simple_metrics(val_preds_tensor, val_targets_tensor)
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Store statistics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_maes.append(train_mae)
        val_maes.append(val_mae)
        train_r2s.append(train_r2)
        val_r2s.append(val_r2)
        epoch_times.append(epoch_time)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, "
              f"Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}")
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model to checkpoints folder
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  → New best model saved to {best_checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Save final model to checkpoints folder
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Final model saved to {final_checkpoint_path}")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(best_checkpoint_path))
    
    # Print comprehensive training statistics
    best_epoch = np.argmin(val_losses)
    print_training_statistics(train_losses, val_losses, train_maes, val_maes, 
                             train_r2s, val_r2s, epoch_times, best_epoch)
    
    # Print final model evaluation on validation set
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION ON VALIDATION SET")
    print("="*60)
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            polygon_coords, guard_indices, rewards, poly_lengths, sol_lengths = \
                [x.to(device) for x in batch]
            
            pred_rewards = model(polygon_coords, guard_indices, poly_lengths, sol_lengths)
            all_predictions.extend(pred_rewards.cpu().numpy())
            all_targets.extend(rewards.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Compute final metrics
    final_mae, final_r2, final_rmse, correlation = compute_simple_metrics(
        torch.tensor(all_predictions), torch.tensor(all_targets)
    )
    
    print(f"\nFINAL VALIDATION METRICS:")
    print(f"  Mean Absolute Error (MAE): {final_mae:.6f}")
    print(f"  Root Mean Square Error (RMSE): {final_rmse:.6f}")
    print(f"  R² Score: {final_r2:.6f}")
    print(f"  Correlation Coefficient: {correlation:.6f}")
    print(f"  Mean Prediction: {np.mean(all_predictions):.6f}")
    print(f"  Mean Target: {np.mean(all_targets):.6f}")
    print(f"  Std Prediction: {np.std(all_predictions):.6f}")
    print(f"  Std Target: {np.std(all_targets):.6f}")
    
    # Prediction quality assessment
    if final_r2 > 0.9:
        quality = "Excellent"
    elif final_r2 > 0.7:
        quality = "Good"
    elif final_r2 > 0.5:
        quality = "Fair"
    else:
        quality = "Poor"
    
    print(f"\nMODEL QUALITY ASSESSMENT: {quality}")
    print("="*60)
    
    return model, best_checkpoint_path, final_checkpoint_path


def main():
    # Load environment variables
    load_dotenv()
    DATASET_PATH = os.getenv("DATASET_PATH")
    if not DATASET_PATH:
        raise EnvironmentError("DATASET_PATH environment variable must be set in .env file.")
    
    parser = argparse.ArgumentParser(description='Train Solution-Aware Value Network')
    
    # Mode selection
    parser.add_argument('--mode', choices=['generate', 'train', 'both'], default='both',
                       help='Generate data, train model, or both')
    
    # Data generation parameters
    parser.add_argument('--actor-checkpoint', type=str, 
                       default='checkpoints/rl_agp_model_embedding_size128_hidden_size128_n_glimpses1_tanh_exploration10_temperature1.0_use_tanhTrue_epochs30.pt',
                       help='Path to trained actor checkpoint')
    parser.add_argument('--train-dir', type=str, 
                       default=os.path.join(DATASET_PATH, "train"),
                       help='Directory with training .pol files')
    parser.add_argument('--samples-per-polygon', type=int, default=50,
                       help='Number of solutions to sample per polygon')
    parser.add_argument('--max-polygons', type=int, default=None,
                       help='Maximum number of polygons to use (for testing)')
    parser.add_argument('--data-file', type=str, default='value_net_training_data.pkl',
                       help='File to save/load training data')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--output', type=str, default='value_net.pt',
                       help='Output file for trained model (in addition to checkpoints)')
    
    # Value network architecture parameters
    parser.add_argument('--value-embedding-size', type=int, default=128,
                       help='Embedding size for value network')
    parser.add_argument('--value-hidden-size', type=int, default=256,
                       help='Hidden size for value network')
    
    # Model architecture parameters (should match actor)
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--n-glimpses', type=int, default=1)
    parser.add_argument('--tanh-exploration', type=int, default=10)
    parser.add_argument('--use-tanh', action='store_true', default=True)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--normalize', action='store_true', default=True)
    
    args = parser.parse_args()
    
    if args.mode in ['generate', 'both']:
        print("=== DATA GENERATION PHASE ===")
        
        # Load training dataset
        train_files = [os.path.join(args.train_dir, f) for f in os.listdir(args.train_dir) if f.endswith('.pol')]
        if args.max_polygons:
            train_files = train_files[:args.max_polygons]
        
        train_samples = agp_read_samples(train_files, normalize=args.normalize)
        train_dataset = AGPDataset(train_samples)
        print(f"Loaded {len(train_dataset)} training polygons")
        
        # Load trained actor
        actor_model, device = load_trained_actor(
            args.actor_checkpoint,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            n_glimpses=args.n_glimpses,
            tanh_exploration=args.tanh_exploration,
            use_tanh=args.use_tanh,
            temperature=args.temperature
        )
        
        # Generate training data
        training_data = generate_training_data(
            actor_model, train_dataset, 
            num_samples_per_polygon=args.samples_per_polygon,
            device=device
        )
        
        # Save training data
        with open(args.data_file, 'wb') as f:
            pickle.dump(training_data, f)
        print(f"Training data saved to {args.data_file}")
        
        # Save metadata
        metadata = {
            'num_samples': len(training_data),
            'num_polygons': len(train_dataset),
            'samples_per_polygon': args.samples_per_polygon,
            'actor_checkpoint': args.actor_checkpoint,
            'args': vars(args)
        }
        with open(args.data_file.replace('.pkl', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    if args.mode in ['train', 'both']:
        print("=== TRAINING PHASE ===")
        
        # Load training data
        with open(args.data_file, 'rb') as f:
            training_data = pickle.load(f)
        print(f"Loaded {len(training_data)} training samples")
        
        # Train value network
        model, best_checkpoint_path, final_checkpoint_path = train_value_net(
            training_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_split=args.val_split,
            embedding_size=args.value_embedding_size,
            hidden_size=args.value_hidden_size
        )
        
        # Also save final model to specified location for backward compatibility
        torch.save(model.state_dict(), args.output)
        print(f"\nCheckpoint summary:")
        print(f"  Best model saved to: {best_checkpoint_path}")
        print(f"  Final model saved to: {final_checkpoint_path}")
        print(f"  Copy also saved to: {args.output} (for backward compatibility)")


if __name__ == "__main__":
    main()
