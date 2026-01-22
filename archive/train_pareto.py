#!/usr/bin/env python3
"""
Pareto Training for AGP: Multi-objective optimization across coverage-size trade-offs.

This script trains a single RL model using a progressive alpha schedule that explores
the Pareto frontier between polygon coverage and solution size. The model learns to
balance these objectives by training across     print(f"\n📊 Pareto Training Performance Summary:")
    print(f"{'='*60}")
    print(f"🕒 Training time: {training_time:.2f} seconds")
    print(f"🔄 Total epochs: {total_epochs}")
    print(f"📈 Alpha progression: {alpha_values[0]:.1f} → {alpha_values[-1]:.1f} ({len(alpha_values)} steps)")
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📉 Final loss: {training_stats['loss_history'][-1]:.4f}")
    print(f"\n📈 Pareto Frontier Results:")
    print(f"{'Alpha':<8} {'Coverage':<10} {'Size Ratio':<12} {'Trade-off'}")
    print(f"{'-'*40}") alpha values in the reward function:

Reward = -α(1-coverage) - (1-α) * rel_length

where rel_length = n_guards / total_vertices

Training progression:
- α=1.0 → 0.9 → 0.8 → ... → 0.1 → 0.0 (11 iterations)
- 10 epochs per alpha value
- Shared model weights across all alpha values
- Saves checkpoints at key alpha milestones
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import tempfile
import shutil
from functools import partial
from tqdm import tqdm

# Local imports
from dataset import agp_read_samples, Dataset, collate_fn
from models import create_actor, create_critic  
from rl_agp import reinforce_train_ema, reinforce_eval, prepare_datasets
from utils import evaluate_polygon_visibility_numpy_wo_gt


def create_pareto_reward_fn(alpha):
    """
    Create a reward function with specific alpha value for Pareto training.
    
    Args:
        alpha: Weight for coverage term (0.0 = pure size focus, 1.0 = pure coverage focus)
        
    Returns:
        Reward function: -α(1-coverage) - (1-α) * rel_length
    """
    def pareto_reward(points: np.ndarray, solution: np.ndarray, name: str, length: int = None) -> float:
        # If length is provided, slice points to real vertices only
        if length is not None:
            points = points[:length]
        
        total_vertices = len(points)
        n_guards = len(solution)
        
        # Compute coverage using the existing function
        try:
            coverage = evaluate_polygon_visibility_numpy_wo_gt(points, solution, name)
        except Exception:
            coverage = 0.0  # Fallback for invalid solutions
            
        # Relative length (guards per vertex)
        rel_length = n_guards / total_vertices if total_vertices > 0 else 1.0
        
        # Pareto reward: balance coverage and size based on alpha
        return -alpha * (1.0 - coverage) - (1.0 - alpha) * rel_length
    
    return pareto_reward


def reinforce_train_ema_with_tqdm(model, dataset, reward_fn, epochs=2, batch_size=1, lr=1e-3, beta=0.99, alpha_desc=""):
    """
    Train the model using REINFORCE with EMA baseline and tqdm progress bars.
    """
    from torch.utils.data import DataLoader
    from rl_agp import BucketBatchSampler, get_lengths_from_dataset
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lengths = get_lengths_from_dataset(dataset)
    batch_sampler = BucketBatchSampler(lengths, batch_size, shuffle=True, bucket_size=10)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, pin_memory=True, num_workers=2)
    device = next(model.parameters()).device
    
    # Initialize EMA baseline for variance reduction
    baseline = 0.0
    
    # Loop over epochs; use inner batch progress which already shows "Epoch X/Y" so we avoid duplicate epoch bars
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        # Progress bar for batches within each epoch (shows Epoch X/Y)
        batch_progress = tqdm(loader, desc=f"🔄 Epoch {epoch+1}/{epochs}", 
                             leave=False, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}] {postfix}")
        
        for batch_data, mask, lengths, batch_names in batch_progress:
            batch_data = batch_data.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            lengths = torch.tensor(lengths, dtype=torch.long, device=device)
            
            # Forward pass: model should return (selected_idxs, log_probs)
            selected_idxs, log_probs = model(batch_data, padding_mask=mask, lengths=lengths)
            rewards_list = []
            
            for i, (data_tensor, idxs, name) in enumerate(zip(batch_data.cpu(), selected_idxs, batch_names)):
                n = lengths[i].item() if lengths is not None else len(data_tensor)
                real_points = data_tensor[:n].detach().cpu().numpy()
                real_solution = [idx for idx in idxs if idx < n]
                r = reward_fn(real_points, real_solution, name, length=n)
                rewards_list.append(r)
            
            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
            
            # Update EMA baseline and compute advantages
            batch_mean = rewards.mean().item()
            baseline = beta * baseline + (1 - beta) * batch_mean
            advantages = rewards - baseline
            
            # REINFORCE loss using advantages instead of raw rewards
            loss = -(log_probs * advantages).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_data.size(0)
            num_batches += 1
            
            # Update batch progress bar
            batch_progress.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Reward': f"{batch_mean:.3f}",
                'Baseline': f"{baseline:.3f}"
            })
            
            # Free up memory after each batch
            del batch_data, mask, lengths, selected_idxs, log_probs, rewards
            torch.cuda.empty_cache()
            
            # Gather garbage
            import gc
            gc.collect()
        
        batch_progress.close()
        
        avg_loss = total_loss / len(dataset) if len(dataset) > 0 else 0.0
        
    # Print a concise epoch summary (avoids a second progress bar)
    tqdm.write(f"Epoch {epoch+1}/{epochs} - Avg_Loss: {avg_loss:.4f} - Baseline: {baseline:.3f}")

    torch.cuda.empty_cache()
    return baseline, avg_loss


def evaluate_pareto_model(model, val_dataset, alpha_values, sol_dir, device='cpu'):
    """
    Evaluate the trained Pareto model across different alpha values to show Pareto frontier.
    
    Args:
        model: Trained RL model
        val_dataset: Validation dataset
        alpha_values: List of alpha values to evaluate
        sol_dir: Directory containing .solution files
        device: Device for evaluation
        
    Returns:
        Dict with evaluation results for each alpha
    """
    results = {}
    model.eval()
    
    # Use progress bar for evaluation
    eval_progress = tqdm(alpha_values, desc="🔍 Evaluating Pareto frontier", 
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    
    for alpha in eval_progress:
        eval_progress.set_description(f"🔍 Evaluating α={alpha:.1f}")
        reward_fn = create_pareto_reward_fn(alpha)
        
        # Use smaller batch size for evaluation
        eval_results = reinforce_eval(model, val_dataset, reward_fn, batch_size=1, sol_dir=sol_dir)
        
        results[f"alpha_{alpha:.1f}"] = {
            'alpha': alpha,
            'stats': eval_results.get('stats', {}),
            'coverage_mean': eval_results['stats']['coverage_vis_stats']['mean'] if 'stats' in eval_results and 'coverage_vis_stats' in eval_results['stats'] else 0.0,
            'size_ratio_mean': eval_results['stats']['ratio_stats']['mean'] if 'stats' in eval_results and 'ratio_stats' in eval_results['stats'] else 0.0
        }
        
        # Update progress bar with current results
        eval_progress.set_postfix({
            'Coverage': f"{results[f'alpha_{alpha:.1f}']['coverage_mean']:.3f}",
            'Size_Ratio': f"{results[f'alpha_{alpha:.1f}']['size_ratio_mean']:.3f}"
        })
    
    eval_progress.close()
    return results


def train_pareto_model(args):
    """
    Main training function for Pareto optimization.
    """
    print("🎯 Starting Pareto Training for AGP")
    print(f"📊 Training progression: α=1.0 → 0.0 (steps: {args.alpha_steps})")
    print(f"🔄 Epochs per alpha: {args.epochs_per_alpha}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🎓 Learning rate: {args.lr}")
    
    # Load datasets
    print(f"\n📂 Loading datasets...")
    train_dataset, val_dataset = prepare_datasets(
        args.agp_train_dir, args.agp_val_dir, 
        args.normalize
    )
    
    # Apply max_instances limit if specified
    if args.max_instances:
        train_samples = train_dataset.samples[:args.max_instances]
        val_samples = val_dataset.samples[:args.max_instances]
        train_dataset = Dataset(train_samples)
        val_dataset = Dataset(val_samples)
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Create model
    print(f"\n🧠 Creating RL model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Create a dummy reward function for model initialization (we'll use different ones during training)
    dummy_reward_fn = create_pareto_reward_fn(1.0)
    
    model = create_actor(
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size, 
        seq_len=None,  # Dynamic based on input
        n_glimpses=args.n_glimpses,
        tanh_exploration=args.tanh_exploration,
        use_tanh=args.use_tanh,
        attention_type=args.attention,
        reward_fn=dummy_reward_fn,
        temperature=args.temperature
    )
    
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Generate alpha schedule
    alpha_values = np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps)
    print(f"\n📈 Alpha schedule: {alpha_values}")
    
    # Training statistics
    training_stats = {
        'alpha_history': [],
        'epoch_history': [],
        'loss_history': [],
        'coverage_history': [],
        'size_history': []
    }
    
    total_epochs = 0
    start_time = time.time()
    
    # Train across alpha values with progress bar
    alpha_progress = tqdm(enumerate(alpha_values), total=len(alpha_values), 
                         desc="🎯 Pareto Training Progress", 
                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
    
    # Ensure checkpoint directory exists and prepare global-best tracking
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_metric = float('-inf')
    best_ckpt_path = None

    for i, alpha in alpha_progress:
        alpha_progress.set_description(f"🎯 Training α={alpha:.1f}")

        # Create reward function for this alpha
        reward_fn = create_pareto_reward_fn(alpha)

        # Adjust learning rate for later phases (optional)
        current_lr = args.lr * (args.lr_decay ** i) if args.lr_decay < 1.0 else args.lr

        # Save incumbent weights to a temporary on-disk checkpoint to avoid holding two large models in RAM
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=f"incumbent_alpha{i}_", suffix='.pt', dir=args.checkpoint_dir)
        os.close(tmp_fd)
        torch.save({'model_state_dict': model.state_dict()}, tmp_path)
        incumbent_tmp_path = tmp_path

        # Evaluate incumbent performance on validation
        try:
            inc_eval = reinforce_eval(model, val_dataset, reward_fn, batch_size=1, sol_dir=args.agp_val_dir)
            inc_rewards = inc_eval.get('rewards', [])
            inc_metric = float(np.mean(inc_rewards)) if len(inc_rewards) > 0 else float(inc_eval.get('stats', {}).get('coverage_vis_stats', {}).get('mean', float('-inf')))
        except Exception as e:
            tqdm.write(f"⚠️ Incumbent evaluation failed: {e}; defaulting to very low incumbent metric")
            inc_metric = float('-inf')

        # Train with current alpha — concise header
        tqdm.write(f"🎯 Phase {i+1}/{len(alpha_values)} — α={alpha:.1f}, epochs={args.epochs_per_alpha}, lr={current_lr:.2e}")

        # Train with current alpha using tqdm-enhanced training
        alpha_desc = f"α={alpha:.1f}"
        baseline, avg_loss = reinforce_train_ema_with_tqdm(
            model=model,
            dataset=train_dataset,
            reward_fn=reward_fn,
            epochs=args.epochs_per_alpha,
            batch_size=args.batch_size,
            lr=current_lr,
            beta=args.beta,
            alpha_desc=alpha_desc
        )

        total_epochs += args.epochs_per_alpha

        # Store training progress for final statistics
        training_stats['alpha_history'].append(alpha)
        training_stats['epoch_history'].append(total_epochs)
        training_stats['loss_history'].append(avg_loss)

        tqdm.write(f"✓ Phase {i+1}/{len(alpha_values)} completed: α={alpha:.1f}, "
                   f"Total epochs: {total_epochs}, Loss: {avg_loss:.4f}, Baseline: {baseline:.3f}")

        # Evaluate candidate model and accept weights only if performance improved (higher reward is better)
        try:
            cand_eval = reinforce_eval(model, val_dataset, reward_fn, batch_size=1, sol_dir=args.agp_val_dir)
            cand_rewards = cand_eval.get('rewards', [])
            cand_metric = float(np.mean(cand_rewards)) if len(cand_rewards) > 0 else float(cand_eval.get('stats', {}).get('coverage_vis_stats', {}).get('mean', float('-inf')))
        except Exception as e:
            tqdm.write(f"⚠️ Candidate evaluation failed: {e}; keeping incumbent weights")
            cand_metric = float('-inf')

        if cand_metric >= inc_metric:
            tqdm.write(f"✅ Accepted candidate weights for α={alpha:.1f} (incumbent: {inc_metric:.4f}, candidate: {cand_metric:.4f})")
            # Optionally, save checkpoint for accepted model
            if args.save_intermediate:
                ckpt_path = os.path.join(args.checkpoint_dir, f"pareto_alpha{alpha:.2f}_epochs{total_epochs}.pt")
                torch.save({'model_state_dict': model.state_dict(), 'alpha': alpha, 'total_epochs': total_epochs, 'metric': cand_metric}, ckpt_path)

            # Update global best if improved
            if cand_metric > best_metric:
                best_metric = cand_metric
                best_ckpt_path = os.path.join(args.checkpoint_dir, f"pareto_best_alpha{alpha:.2f}_metric{cand_metric:.4f}.pt")
                torch.save({'model_state_dict': model.state_dict(), 'alpha': alpha, 'metric': cand_metric, 'total_epochs': total_epochs}, best_ckpt_path)

            # Remove temporary incumbent checkpoint
            try:
                if os.path.exists(incumbent_tmp_path):
                    os.remove(incumbent_tmp_path)
            except Exception:
                pass
        else:
            # Revert to incumbent weights by loading the temporary checkpoint
            try:
                ck = torch.load(incumbent_tmp_path, map_location=device)
                model.load_state_dict(ck['model_state_dict'])
            except Exception as e:
                tqdm.write(f"⚠️ Failed to reload incumbent from disk: {e}")
            tqdm.write(f"⛔ Reverted to incumbent weights for α={alpha:.1f} (incumbent: {inc_metric:.4f}, candidate: {cand_metric:.4f})")
            try:
                if os.path.exists(incumbent_tmp_path):
                    os.remove(incumbent_tmp_path)
            except Exception:
                pass

    alpha_progress.close()
    
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time:.2f} seconds ({total_epochs} total epochs)")
    
    # Final comprehensive evaluation and statistics
    print(f"\n🔍 Final Pareto frontier evaluation and performance statistics...")
    eval_alphas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]  # Key points on frontier
    pareto_results = evaluate_pareto_model(model, val_dataset, eval_alphas, args.agp_val_dir, device)
    
    # Print performance summary
    print(f"\n� Pareto Training Performance Summary:")
    print(f"{'='*60}")
    print(f"🕒 Training time: {training_time:.2f} seconds")
    print(f"🔄 Total epochs: {total_epochs}")
    print(f"📈 Alpha progression: {alpha_values[0]:.1f} → {alpha_values[-1]:.1f} ({len(alpha_values)} steps)")
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n� Pareto Frontier Results:")
    print(f"{'Alpha':<8} {'Coverage':<10} {'Size Ratio':<12} {'Trade-off'}")
    print(f"{'-'*40}")
    
    for alpha_key, result in pareto_results.items():
        alpha_val = result['alpha']
        coverage = result['coverage_mean']
        size_ratio = result['size_ratio_mean']
        tradeoff = f"C:{coverage:.2f}/S:{size_ratio:.2f}"
        print(f"{alpha_val:<8.1f} {coverage:<10.3f} {size_ratio:<12.3f} {tradeoff}")
    
    print(f"{'='*60}")
    
    # Save final results
    final_results = {
        'args': vars(args),
        'training_time': training_time,
        'total_epochs': total_epochs,
        'alpha_schedule': alpha_values.tolist(),
        'training_stats': training_stats,
        'pareto_evaluation': pareto_results,
        'model_info': {
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'device': str(device),
            'embedding_size': args.embedding_size,
            'hidden_size': args.hidden_size
        }
    }
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results_path = 'results/pareto_training_results.json'
    # Sanitize final_results to ensure all numpy types are converted to native Python types
    def sanitize_for_json(obj):
        # Recursively convert numpy types and arrays to Python native types
        if isinstance(obj, dict):
            return {k: sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize_for_json(v) for v in obj]
        if isinstance(obj, tuple):
            return [sanitize_for_json(v) for v in obj]
        try:
            import numpy as _np
        except Exception:
            _np = None

        if _np is not None:
            if isinstance(obj, _np.generic):
                return obj.item()
            if isinstance(obj, _np.ndarray):
                return obj.tolist()

        # Fallback: return as-is (json.dump will raise if still not serializable)
        return obj

    sanitized = sanitize_for_json(final_results)
    with open(results_path, 'w') as f:
        json.dump(sanitized, f, indent=2)
    print(f"📊 Results saved to {results_path}")
    
    # Save final model only
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    final_checkpoint_path = os.path.join(args.checkpoint_dir, 
                                        f"pareto_model_final_epochs{total_epochs}.pt")
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'total_epochs': total_epochs,
        'alpha_schedule': alpha_values.tolist(),
        'training_stats': training_stats,
        'training_time': training_time,
        'metadata': {
            'model_type': 'pareto_rl',
            'training_complete': True,
            'pareto_frontier_trained': True
        }
    }
    
    torch.save(final_checkpoint, final_checkpoint_path)
    print(f"💾 Final model saved: {final_checkpoint_path}")
    
    return model, final_results


def main():
    parser = argparse.ArgumentParser(description='Pareto Training for AGP')
    
    # Dataset arguments
    parser.add_argument('--agp-train-dir', type=str, 
                       default='/home/dseverdi/Radno/MLAG/dataset/AGPIL/train',
                       help='Directory containing training .pol files')
    parser.add_argument('--agp-val-dir', type=str, 
                       default='/home/dseverdi/Radno/MLAG/dataset/AGPIL/dev',
                       help='Directory containing validation .pol and .solution files')
    parser.add_argument('--max-instances', type=int, default=None,
                       help='Maximum number of instances to use (None = all)')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize polygon coordinates')
    
    # Model architecture arguments
    parser.add_argument('--embedding-size', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='Hidden layer size')
    parser.add_argument('--n-glimpses', type=int, default=1,
                       help='Number of attention glimpses')
    parser.add_argument('--tanh-exploration', type=float, default=10.0,
                       help='Tanh exploration parameter')
    parser.add_argument('--use-tanh', action='store_true', default=True,
                       help='Use tanh in the model')
    parser.add_argument('--attention', type=str, default='Bahdanau',
                       choices=['Bahdanau', 'Dot'],
                       help='Attention mechanism type')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for softmax sampling')
    
    # Training arguments
    parser.add_argument('--epochs-per-alpha', type=int, default=10,
                       help='Number of epochs to train at each alpha value')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--lr-decay', type=float, default=1.0,
                       help='Learning rate decay factor per alpha phase (1.0 = no decay)')
    parser.add_argument('--beta', type=float, default=0.99,
                       help='EMA baseline decay factor')
    
    # Pareto schedule arguments
    parser.add_argument('--alpha-start', type=float, default=1.0,
                       help='Starting alpha value (coverage focus)')
    parser.add_argument('--alpha-end', type=float, default=0.0,
                       help='Ending alpha value (size focus)')
    parser.add_argument('--alpha-steps', type=int, default=11,
                       help='Number of alpha steps (including start and end)')
    
    # Output arguments
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for final model (default: auto-generate)')
    parser.add_argument('--save-intermediate', action='store_true', default=False,
                       help='Save intermediate accepted models after each alpha phase')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.alpha_start < args.alpha_end:
        raise ValueError("alpha_start should be >= alpha_end (we go from coverage to size focus)")
    
    if args.alpha_steps < 2:
        raise ValueError("alpha_steps should be >= 2")
    
    print("🚀 Pareto Training Configuration:")
    print(f"   Dataset: {args.agp_train_dir} → {args.agp_val_dir}")
    print(f"   Alpha schedule: {args.alpha_start} → {args.alpha_end} ({args.alpha_steps} steps)")
    print(f"   Training: {args.epochs_per_alpha} epochs/alpha, batch_size={args.batch_size}")
    print(f"   Model: emb={args.embedding_size}, hidden={args.hidden_size}")
    print(f"   Output: {args.checkpoint_dir}")
    
    # Start training
    model, results = train_pareto_model(args)
    
    print("\n🎉 Pareto training completed successfully!")
    print("📊 Check results/pareto_training_results.json for detailed evaluation")


if __name__ == "__main__":
    main()
