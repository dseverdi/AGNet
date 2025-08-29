import os
from dotenv import load_dotenv
import argparse
from dataset import Dataset, agp_read_samples, collate_fn
from models import create_actor, create_critic
from utils import createPolygon, compute_visibility
import torch
try:
    import skgeom
except ImportError:
    skgeom = None
    print("Warning: skgeom not available, some visualization functions may not work")
from torch.utils.data import DataLoader, Sampler

from utils import evaluate_polygon_visibility_numpy_wo_gt  # reuse coverage evaluation
import numpy as np
import sys
import matplotlib.pyplot as plt
#from rewards import linear_reward as reward  # Use the new reward function
#from rewards import strict_reward as reward  # Use the strict reward function
from rewards import enhanced_penalty as reward  # Use the smooth reward function
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



def reinforce_train_ema(model, dataset, reward_fn, epochs=2, batch_size=1, lr=1e-3, beta=0.99):
    """
    Train the model using REINFORCE with exponential moving average (EMA) baseline for variance reduction.
    Uses a simple bucket sampler to group samples by length.
    Args:
        model: The actor model to train.
        dataset: The dataset containing samples.
        reward_fn: The reward function to compute rewards.
        epochs: Number of training epochs.
        batch_size: Size of each training batch.
        lr: Learning rate for the optimizer.
        beta: EMA decay factor for baseline updates.
    """
    print(f"\n--- Training on {len(dataset)} samples for {epochs} epochs (batch size {batch_size}) ---")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lengths = get_lengths_from_dataset(dataset)
    batch_sampler = BucketBatchSampler(lengths, batch_size, shuffle=True, bucket_size=10)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, pin_memory=True, num_workers=2)
    device = next(model.parameters()).device
    # Initialize EMA baseline for variance reduction
    baseline = 0.0
    for epoch in range(epochs):
        total_loss = 0
        for batch_data, mask, lengths, batch_names in loader:
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
                r = reward(real_points, real_solution, name, length=n)
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
            # Free up memory after each batch
            del batch_data, mask, lengths, selected_idxs, log_probs, rewards
            torch.cuda.empty_cache()
            # gather garbage
            import gc
            gc.collect()
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Avg loss: {avg_loss:.4f}")
        torch.cuda.empty_cache()
    print("Training done.")

# --- Reinforce with learned critic ---
def reinforce_train_critic(actor, critic, dataset, reward_fn,
                          epochs=2, batch_size=1,
                          lr_actor=1e-3, lr_critic=1e-3):
    """
    Actor-critic training: policy network (actor) and value network (critic).
    Critic is trained via MSE to match observed return; actor via advantage.
    """
    print(f"\n--- AC Training on {len(dataset)} samples for {epochs} epochs (batch {batch_size}) ---")
    actor.train(); critic.train()
    opt_actor = torch.optim.Adam(actor.parameters(), lr=lr_actor)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=lr_critic)
    lengths_list = get_lengths_from_dataset(dataset)
    sampler = BucketBatchSampler(lengths_list, batch_size, shuffle=True, bucket_size=10)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn,
                        pin_memory=True, num_workers=2)
    device = next(actor.parameters()).device
    for epoch in range(epochs):
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        for batch_data, mask, lengths, batch_names in loader:
            batch_data = batch_data.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            lengths = torch.tensor(lengths, dtype=torch.long, device=device)
            # Actor forward
            selected_idxs, log_probs = actor(batch_data, padding_mask=mask, lengths=lengths)
            # Critic forward
            values = critic(batch_data, mask, lengths)
            # Compute rewards
            rewards_list = []
            for i, (data_tensor, idxs, name) in enumerate(zip(batch_data.cpu(), selected_idxs, batch_names)):
                n = lengths[i].item() if lengths is not None else len(data_tensor)
                real_points = data_tensor[:n].detach().cpu().numpy()
                real_solution = [idx for idx in idxs if idx < n]
                r = reward(real_points, real_solution, name, length=n)
                rewards_list.append(r)
            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
            # Critic loss: fit to returns
            critic_loss = F.mse_loss(values, rewards)
            # Advantages for actor
            advantages = rewards - values.detach()
            actor_loss = -(log_probs * advantages).mean()
            # Optimize
            opt_actor.zero_grad(); opt_critic.zero_grad()
            actor_loss.backward(); critic_loss.backward()
            opt_actor.step(); opt_critic.step()
            total_actor_loss += actor_loss.item() * batch_data.size(0)
            total_critic_loss += critic_loss.item() * batch_data.size(0)
            # Mem cleanup
            del batch_data, mask, lengths, selected_idxs, log_probs, values, rewards
            torch.cuda.empty_cache(); import gc; gc.collect()
        print(f"Epoch {epoch+1}/{epochs} - Actor loss: {total_actor_loss/len(dataset):.4f}" \
              f", Critic loss: {total_critic_loss/len(dataset):.4f}")
    print("Actor-critic training done.")

# --- Evaluation ---
def reinforce_eval(model, dataset, reward_fn, batch_size=1, sol_dir=None):
    """Evaluate the model on the validation dataset with comprehensive metrics for whisker plots."""
    print(f"\n--- Evaluation on {len(dataset)} validation samples (batch size {batch_size}) ---")
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
                
                # Compute reward and coverage
                r = reward_fn(points, np.array(pred_indices), name, length=length)
                all_rewards.append(r)
                
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
                base_name = os.path.splitext(os.path.basename(name))[0]
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
        # Legacy compatibility
        'rewards': all_rewards,
        'coverages': all_coverages,
        'rel_sizes': rel_sizes
    }



# --- Visualization and Coverage Testing ---
# --- Test Coverage ---
def test_coverage_on_sample(dataset, sol_dir, index=0, regime="opt", n_random_guards=None):
    """Load the optimal solution or a random guard set and evaluate coverage on one sample
    n_random_guards: if set and regime=='random', use this many guards (default: match optimal solution)
    """
    import random
    print(f"\n--- Coverage test on a single sample (regime: {regime}) ---")
    if len(dataset) == 0:
        print("No validation samples available for coverage test.")
        return
    # Get raw polygon points and sample name
    sample_data, _, sample_name = dataset[index]
    points = sample_data.numpy()
    n_points = len(points)
    true_idxs = []
    if regime == "opt":
        # Read optimal guard indices from .solution file (second line)
        sol_path = os.path.join(sol_dir, f"{sample_name}.solution")
        try:
            with open(sol_path, 'r') as f:
                lines = f.read().splitlines()
                if len(lines) >= 2:
                    true_idxs = [int(x) for x in lines[1].split()]
        except Exception as e:
            print(f"Could not read solution file {sol_path}: {e}", file=sys.stderr)
            return
        if not true_idxs:
            print(f"No guards found in solution file for sample {sample_name}")
            return
        guard_idxs = np.array(true_idxs)
        label = "True (optimal)"
    elif regime == "random":
        # Try to match the number of guards in the optimal solution if possible, unless overridden
        if n_random_guards is not None:
            n_guards = min(max(1, int(n_random_guards)), n_points)
        else:
            sol_path = os.path.join(sol_dir, f"{sample_name}.solution")
            n_guards = 1
            try:
                with open(sol_path, 'r') as f:
                    lines = f.read().splitlines()
                    if len(lines) >= 2:
                        n_guards = max(1, len([int(x) for x in lines[1].split()]))
            except Exception:
                n_guards = max(1, n_points // 10)  # fallback: 10% of vertices
        guard_idxs = np.array(sorted(random.sample(range(n_points), min(n_guards, n_points))))
        label = f"Random ({len(guard_idxs)} guards)"
    else:
        print(f"Unknown regime: {regime}")
        return
    # Evaluate coverage
    coverage = evaluate_polygon_visibility_numpy_wo_gt(points, guard_idxs, sample_name)
    print(f"Sample: {sample_name}  {label} coverage: {coverage:.4f}")

    # --- Visualization with visibility regions ---
    # Compute visibility polygons for each guard
    if skgeom is None:
        print("Warning: skgeom not available, skipping visibility visualization")
        return
    
    from concurrent.futures import ThreadPoolExecutor
    eps = 1e-8
    poly_obj = createPolygon(points)
    if poly_obj is None:
        print(f"Invalid polygon in {sample_name}: less than 3 vertices or zero area", file=sys.stderr)
        return
    arr = skgeom.arrangement.Arrangement()
    for edge in poly_obj.edges:
        arr.insert(edge)
    vs = skgeom.TriangularExpansionVisibility(arr)
    edges = list(poly_obj.edges)
    vis_polys = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_visibility, vs, arr, poly_obj, eps, edges, idx) for idx in guard_idxs]
        for future in futures:
            vis_poly, err_idx, err_q = future.result()
            if vis_poly:
                vis_polys.append(vis_poly)
            else:
                vis_polys.append(None)

    fig, ax = plt.subplots()
    poly = np.array(points)
    if poly.shape[0] > 2:
        ax.plot(np.append(poly[:,0], poly[0,0]), np.append(poly[:,1], poly[0,1]), 'k-', lw=1, label='Polygon')
    # Plot each guard's visibility region
    for i, vis_poly in enumerate(vis_polys):
        if vis_poly is not None:
            vis_pts = np.array([[p.x(), p.y()] for p in vis_poly.vertices])
            ax.fill(vis_pts[:,0], vis_pts[:,1], alpha=0.25, label=f'Guard {i} vis' if i==0 else None)
    # Guards
    guards = poly[guard_idxs]
    ax.scatter(guards[:,0], guards[:,1], c='red', s=60, marker='*', label='Guards')
    ax.set_aspect('equal')
    ax.set_title(f"{sample_name} ({label})\nCoverage: {coverage:.2f}")
    ax.legend()
    out_dir = os.path.join(os.path.dirname(__file__), 'gfx')
    os.makedirs(out_dir, exist_ok=True)
    # Add number of guards to the filename
    n_guards_str = f"{len(guard_idxs)}_guards"
    out_path = os.path.join(out_dir, f"{sample_name}_{regime}_{n_guards_str}_coverage.png")
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved coverage plot to {out_path}")
    plt.close(fig)


def main():
    # Load environment variables from .env
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
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ema', action='store_true', help='Use EMA baseline (reinforce_train_ema)')
    group.add_argument('--critic', action='store_true', help='Use learned critic (reinforce_train_critic)')
    args = parser.parse_args()

    train_dataset, val_dataset = prepare_datasets(args.agp_train_dir, args.agp_val_dir, normalize=True)
    agp_model = create_agp_model(
        args.embedding_size, args.hidden_size, args.n_glimpses, args.tanh_exploration, args.use_tanh, reward, args.temperature
    )

    # Create critic model (optional, can be used for actor-critic training)
    critic_model = create_critic_model(
        args.embedding_size, args.hidden_size, args.n_glimpses, "Bahdanau"
    )

    # Use only args.train_size samples for training if available
    size = args.train_size
    small_train_dataset = train_dataset if len(train_dataset) <= size else Dataset(train_dataset.samples[:size])
    small_val_dataset = val_dataset if len(val_dataset) <= size else Dataset(val_dataset.samples[:size])
    
    # define reward function
    reward_fn = partial(reward, alpha=5.0, p=0.0)  # Use smooth reward with alpha=5.0 and p=0.0

    if args.ema:
        reinforce_train_ema(agp_model, small_train_dataset, reward_fn,
                            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, beta=args.beta)
        training_method = 'reinforcement_learning_ema'
    elif args.critic:
        reinforce_train_critic(agp_model, critic_model, small_train_dataset, reward_fn,
                               epochs=args.epochs, batch_size=args.batch_size, lr_actor=args.lr, lr_critic=args.lr)
        training_method = 'reinforcement_learning_critic'

    # Save the trained model
    checkpoint_params = {
        'embedding_size': args.embedding_size,
        'hidden_size': args.hidden_size,
        'n_glimpses': args.n_glimpses,
        'tanh_exploration': args.tanh_exploration,
        'use_tanh': args.use_tanh,
        'beta': args.beta if args.ema else None,
        'temperature': args.temperature
    }
    # Remove None values from checkpoint params
    checkpoint_params = {k: v for k, v in checkpoint_params.items() if v is not None}
    
    checkpoint_path = get_checkpoint_path('checkpoints', 'rl_agp_model', checkpoint_params, args.epochs)
    model_checkpoint = {
        'model_state_dict': agp_model.state_dict(),
        'args': vars(args),
        'training_method': training_method,
        'num_train_samples': len(small_train_dataset),
        'num_val_samples': len(small_val_dataset)
    }
    
    # Also save critic if used
    if args.critic:
        model_checkpoint['critic_state_dict'] = critic_model.state_dict()
    
    torch.save(model_checkpoint, checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    # evaluate the model on the validation dataset
    eval_results = reinforce_eval(agp_model, small_val_dataset, reward_fn, batch_size=1, sol_dir=args.agp_val_dir)
    
    # Save evaluation results
    import json
    results_summary = {
        'args': vars(args),
        'num_train_samples': len(small_train_dataset),
        'num_val_samples': len(small_val_dataset),
        'training_method': 'reinforcement_learning'
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
    
    # Save to results directory
    os.makedirs('results', exist_ok=True)
    with open('results/rl_agp_evaluation.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print("Results summary saved to results/rl_agp_evaluation.json")


if __name__ == "__main__":
    main()
