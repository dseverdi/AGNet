import os
from dotenv import load_dotenv
import argparse
from dataset import Dataset, agp_read_samples, collate_fn
from models import create_actor
import torch
from torch.utils.data import DataLoader, Sampler
import numpy as np
import sys
from functools import partial

nn = torch.nn
F = torch.nn.functional

# --- Utility ---
def get_checkpoint_path(folder, model_name, params, n_epochs):
    param_str = "_".join([f"{k}{v}" for k, v in sorted(params.items())])
    filename = f"{model_name}_{param_str}_epochs{n_epochs}.pt"
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)

# --- Data Preparation ---
def prepare_datasets_with_targets(train_path, val_path, normalize=True):
    def get_pol_files(path):
        if os.path.isdir(path):
            return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pol')]
        elif os.path.isfile(path) and path.endswith('.pol'):
            return [path]
        else:
            raise ValueError(f"Provided path {path} is neither a .pol file nor a directory containing .pol files.")
    
    def agp_read_samples_with_targets(paths, normalize=False):
        """Read AGP samples and their corresponding targets from .solution files"""
        samples = []
        for i, path in enumerate(paths):
            # Read polygon data
            with open(path, 'r') as f:
                tokens = f.read().split()
                num_points = int(tokens[0])
                points = []
                for i in range(1, 2 * num_points, 2):
                    x_token = tokens[i]
                    y_token = tokens[i + 1]
                    # Parse x coordinate (handle x/1 format)
                    if '/' in x_token:
                        x_num, x_denom = map(float, x_token.split('/'))
                        x = x_num / x_denom if x_denom != 0 else 0.0
                    else:
                        x = float(x_token)
                    # Parse y coordinate (handle y/1 format)
                    if '/' in y_token:
                        y_num, y_denom = map(float, y_token.split('/'))
                        y = y_num / y_denom if y_denom != 0 else 0.0
                    else:
                        y = float(y_token)
                    points.append((x, y))
                points_tensor = torch.tensor(points, dtype=torch.float32)
                if normalize:
                    min_xy = points_tensor.min(dim=0)[0]
                    max_xy = points_tensor.max(dim=0)[0]
                    denom = (max_xy - min_xy)
                    denom[denom == 0] = 1.0  # avoid division by zero
                    points_tensor = (points_tensor - min_xy) / denom
            
            # Read target indices from .solution file
            base = os.path.splitext(os.path.basename(path))[0]
            sol_path = os.path.join(os.path.dirname(path), f"{base}.solution")
            try:
                with open(sol_path, 'r') as f:
                    lines = f.read().splitlines()
                    if len(lines) >= 2:
                        target_indices = [int(x) for x in lines[1].split()]
                    else:
                        target_indices = []
            except Exception:
                target_indices = []
            
            name = os.path.splitext(os.path.basename(path))[0]
            # Create Sample with target indices as label
            from dataset import Sample
            samples.append(Sample(data=points_tensor, label=target_indices, name=name))
        return samples
    
    agp_train_paths = get_pol_files(train_path)
    agp_val_paths = get_pol_files(val_path)
    print(f"Found {len(agp_train_paths)} training and {len(agp_val_paths)} validation AGP .pol files.")
    print("Reading training samples with targets...")
    train_samples = agp_read_samples_with_targets(agp_train_paths, normalize=normalize)
    print("Reading validation samples with targets...")
    val_samples = agp_read_samples_with_targets(agp_val_paths, normalize=normalize)
    return train_samples, val_samples

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

# --- Supervised Training Loop ---
def supervised_train(model, dataset, epochs=10, batch_size=128, lr=1e-3):
    print(f"\n--- Supervised Training on {len(dataset)} samples for {epochs} epochs (batch size {batch_size}) ---")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Convert to standard dataset format for efficient batching
    standard_dataset = []
    for sample in dataset:
        # Convert Sample objects to (data, label, name) tuples for collate_fn compatibility
        standard_dataset.append((sample.data, sample.label, sample.name))
    
    lengths = [sample[0].shape[0] for sample in standard_dataset]
    batch_sampler = BucketBatchSampler(lengths, batch_size, shuffle=True, bucket_size=10)
    loader = DataLoader(standard_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, pin_memory=True, num_workers=2)
    device = next(model.parameters()).device
    try:
        from tqdm import tqdm
        epoch_iter = tqdm(range(epochs), desc='Training Epochs', leave=True)
        use_tqdm = True
    except ImportError:
        epoch_iter = range(epochs)
        use_tqdm = False
    for epoch in epoch_iter:
        total_loss = 0
        for batch_data, mask, lengths, batch_names in loader:
            # Extract targets for this batch - need to map from names to target indices
            targets = []
            for name in batch_names:
                for sample in standard_dataset:
                    if sample[2] == name:
                        targets.append(sample[1])
                        break
            
            batch_data = batch_data.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            lengths = torch.tensor(lengths, dtype=torch.long, device=device)
            # Forward pass: model should return (selected_idxs, log_probs)
            selected_idxs, log_probs = model(batch_data, padding_mask=mask, lengths=lengths)
            
            # Simplified supervised loss: use the log probabilities from the model
            # The model already computed log probabilities for the selected sequences
            # We want to maximize the probability of sequences that match the targets better
            
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            for i, (pred_indices, target_indices) in enumerate(zip(selected_idxs, targets)):
                if len(target_indices) > 0:
                    # Convert to sets for overlap computation
                    pred_set = set(pred_indices)
                    target_set = set(target_indices)
                    
                    # Compute overlap ratio
                    overlap = len(pred_set.intersection(target_set))
                    total_target = len(target_set)
                    overlap_ratio = overlap / total_target if total_target > 0 else 0.0
                    
                    # Loss: negative log probability weighted by inverse overlap
                    # Higher overlap -> lower loss
                    weight = 1.0 - overlap_ratio + 0.1  # Add small constant to avoid zero weight
                    sample_loss = -log_probs[i] * weight
                    loss = loss + sample_loss
                else:
                    # No target indices - penalize long sequences
                    penalty = len(pred_indices) * 0.1
                    loss = loss + torch.tensor(penalty, device=device)
            
            loss = loss / batch_data.size(0)  # Average over batch
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_data.size(0)
            
            del batch_data, mask, lengths, selected_idxs, log_probs, targets
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        avg_loss = total_loss / len(dataset)
        if use_tqdm:
            epoch_iter.set_postfix({'Avg loss': f'{avg_loss:.4f}'})
        else:
            print(f"Epoch {epoch+1}/{epochs} - Avg loss: {avg_loss:.4f}")
        
        # Report detailed metrics every 10th epoch
        if (epoch + 1) % 10 == 0:
            print(f"\n📊 Training Set Metrics (Epoch {epoch+1}):")
            train_loss, train_coverage, train_size_ratio = compute_dataset_metrics(model, dataset, batch_size=batch_size, device=device)
            print(f"  Loss: {train_loss:.4f}")
            print(f"  Coverage: {train_coverage:.3f} ({train_coverage*100:.1f}%)")
            print(f"  Size Ratio (pred/target): {train_size_ratio:.2f}")
            model.train()  # Set back to training mode
        
        torch.cuda.empty_cache()
    print("Supervised training done.")

# --- Evaluation ---
def compute_dataset_metrics(model, dataset, batch_size=32, device=None):
    """Compute average loss, coverage, and size ratio on a dataset."""
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Convert to standard dataset format for efficient batching
    standard_dataset = []
    for sample in dataset:
        standard_dataset.append((sample.data, sample.label, sample.name))
    
    lengths = [sample[0].shape[0] for sample in standard_dataset]
    batch_sampler = BucketBatchSampler(lengths, batch_size, shuffle=False, bucket_size=10)
    loader = DataLoader(standard_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, pin_memory=True, num_workers=2)
    
    total_loss = 0
    all_coverages = []
    all_size_ratios = []
    
    with torch.no_grad():
        for batch_data, mask, lengths, batch_names in loader:
            # Extract targets for this batch
            targets = []
            for name in batch_names:
                for sample in standard_dataset:
                    if sample[2] == name:
                        targets.append(sample[1])
                        break
            
            batch_data = batch_data.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            lengths = torch.tensor(lengths, dtype=torch.long, device=device)
            selected_idxs, log_probs = model(batch_data, padding_mask=mask, lengths=lengths)
            
            # Compute loss (same as training)
            loss = torch.tensor(0.0, device=device, requires_grad=False)
            for i, (pred_indices, target_indices) in enumerate(zip(selected_idxs, targets)):
                if len(target_indices) > 0:
                    pred_set = set(pred_indices)
                    target_set = set(target_indices)
                    overlap = len(pred_set.intersection(target_set))
                    total_target = len(target_set)
                    overlap_ratio = overlap / total_target if total_target > 0 else 0.0
                    
                    # Coverage and size ratio for this sample
                    all_coverages.append(overlap_ratio)
                    size_ratio = len(pred_indices) / len(target_indices) if len(target_indices) > 0 else 0.0
                    all_size_ratios.append(size_ratio)
                    
                    weight = 1.0 - overlap_ratio + 0.1
                    sample_loss = -log_probs[i] * weight
                    loss = loss + sample_loss
                else:
                    penalty = len(pred_indices) * 0.1
                    loss = loss + torch.tensor(penalty, device=device)
                    all_coverages.append(0.0)
                    all_size_ratios.append(0.0)
            
            loss = loss / batch_data.size(0)
            total_loss += loss.item() * batch_data.size(0)
            
            del batch_data, mask, lengths, selected_idxs, log_probs, targets
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / len(dataset)
    avg_coverage = np.mean(all_coverages) if all_coverages else 0.0
    avg_size_ratio = np.mean(all_size_ratios) if all_size_ratios else 0.0
    
    return avg_loss, avg_coverage, avg_size_ratio

def supervised_eval(model, dataset, batch_size=1):
    print(f"\n--- Evaluation on {len(dataset)} validation samples (batch size {batch_size}) ---")
    model.eval()
    
    # Convert to standard dataset format for efficient batching
    standard_dataset = []
    for sample in dataset:
        # Convert Sample objects to (data, label, name) tuples for collate_fn compatibility
        standard_dataset.append((sample.data, sample.label, sample.name))
    
    lengths = [sample[0].shape[0] for sample in standard_dataset]
    batch_sampler = BucketBatchSampler(lengths, batch_size, shuffle=False, bucket_size=10)
    loader = DataLoader(standard_dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=2)
    device = next(model.parameters()).device
    
    # Statistics for whisker plots
    pred_sizes = []
    true_sizes = []
    coverage_ratios = []  # What fraction of optimal guards are covered by predicted guards
    efficiency_ratios = []  # What fraction of predicted guards are in the optimal set
    size_ratios = []  # predicted_size / optimal_size
    overlap_counts = []  # absolute number of overlapping guards
    
    with torch.no_grad():
        batch_count = 0
        for batch_data, mask, lengths, batch_names in loader:
            # Extract targets for this batch
            targets = []
            for name in batch_names:
                for sample in standard_dataset:
                    if sample[2] == name:
                        targets.append(sample[1])
                        break
            
            batch_data = batch_data.to(device)
            mask = mask.to(device)
            lengths = torch.tensor(lengths, dtype=torch.long, device=device)
            selected_idxs, log_probs = model(batch_data, padding_mask=mask, lengths=lengths)
            
            # Debug: Print first few predictions
            if batch_count < 3:
                print(f"\n🔍 Debug - Batch {batch_count}:")
                for i, (pred_indices, true_indices, name) in enumerate(zip(selected_idxs[:2], targets[:2], batch_names[:2])):
                    print(f"  Sample {name}:")
                    print(f"    Predicted: {pred_indices} (count: {len(pred_indices)})")
                    print(f"    True: {true_indices} (count: {len(true_indices)})")
                    if len(pred_indices) > 0 and len(true_indices) > 0:
                        coverage = len(set(pred_indices).intersection(set(true_indices))) / len(true_indices)
                        print(f"    Coverage: {coverage:.3f}")
            batch_count += 1
            
            for pred_indices, true_indices, name in zip(selected_idxs, targets, batch_names):
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
    size_stats = compute_stats(pred_sizes, "Predicted Solution Sizes")
    optimal_stats = compute_stats(true_sizes, "Optimal Solution Sizes")
    coverage_stats = compute_stats(coverage_ratios, "Coverage Ratios (fraction of optimal guards found)")
    efficiency_stats = compute_stats(efficiency_ratios, "Efficiency Ratios (fraction of predicted guards that are optimal)")
    ratio_stats = compute_stats(size_ratios, "Size Ratios (predicted/optimal)")
    overlap_stats = compute_stats(overlap_counts, "Overlap Counts (absolute number of matching guards)")
    
    # Summary metrics
    print(f"\n=== SUMMARY ===")
    print(f"Instances evaluated: {len(pred_sizes)}")
    print(f"Perfect solutions (100% coverage): {sum(1 for c in coverage_ratios if c >= 1.0)}")
    print(f"Good solutions (>=80% coverage): {sum(1 for c in coverage_ratios if c >= 0.8)}")
    print(f"Reasonable solutions (>=60% coverage): {sum(1 for c in coverage_ratios if c >= 0.6)}")
    print(f"Average size inflation: {np.mean(size_ratios):.2f}x optimal")
    
    return {
        'pred_sizes': pred_sizes,
        'true_sizes': true_sizes,
        'coverage_ratios': coverage_ratios,
        'efficiency_ratios': efficiency_ratios,
        'size_ratios': size_ratios,
        'overlap_counts': overlap_counts,
        'stats': {
            'size_stats': size_stats,
            'optimal_stats': optimal_stats,
            'coverage_stats': coverage_stats,
            'efficiency_stats': efficiency_stats,
            'ratio_stats': ratio_stats,
            'overlap_stats': overlap_stats
        }
    }


def main():
    # Load environment variables from .env
    load_dotenv()
    DATASET_PATH = os.getenv("DATASET_PATH")
    print("Starting main() in sl_agp.py...")
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--embedding-size', type=int, default=128)
        parser.add_argument('--hidden-size', type=int, default=128)
        parser.add_argument('--n-glimpses', type=int, default=1)
        parser.add_argument('--tanh-exploration', type=float, default=10)
        parser.add_argument('--use-tanh', action='store_true', default=True)
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--batch-size', type=int, default=128)
        parser.add_argument('--lr', type=float, default=1e-3)
        # Always use DATASET_PATH from environment variable, no fallback
        if not DATASET_PATH:
            raise EnvironmentError("DATASET_PATH environment variable must be set in .env file.")
        default_train = os.path.join(DATASET_PATH, "train")
        default_val = os.path.join(DATASET_PATH, "dev")
        parser.add_argument('--agp_train_dir', type=str, default=default_train)
        parser.add_argument('--agp_val_dir', type=str, default=default_val)
        parser.add_argument('--train-size', type=int, default=8000, help="Number of training samples to use (default: 8000, or all if smaller)")
        args = parser.parse_args()

        print(f"Arguments: {args}")
        print("Loading datasets...")
        train_dataset, val_dataset = prepare_datasets_with_targets(args.agp_train_dir, args.agp_val_dir, normalize=True)
        print("Datasets loaded.")
        print("Creating model...")
        # For supervised learning, we don't need a reward function, but need to provide a dummy one
        def dummy_reward(points, solution, name, length):
            return 0.0
        
        # Set seq_len to None (will be handled dynamically in the model)
        agp_model = create_actor(
            args.embedding_size, args.hidden_size, None, args.n_glimpses,
            args.tanh_exploration, args.use_tanh, "Bahdanau", dummy_reward, temperature=1.0
        )
        print("Model created.")

        size = args.train_size
        small_train_dataset = train_dataset if len(train_dataset) <= size else train_dataset[:size]
        small_val_dataset = val_dataset if len(val_dataset) <= size else val_dataset[:size]

        print("Starting supervised training...")
        supervised_train(agp_model, small_train_dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        print("Training complete. Starting evaluation...")
        
        # Save the trained model
        checkpoint_params = {
            'embedding_size': args.embedding_size,
            'hidden_size': args.hidden_size,
            'n_glimpses': args.n_glimpses,
            'tanh_exploration': args.tanh_exploration,
            'use_tanh': args.use_tanh
        }
        checkpoint_path = get_checkpoint_path('checkpoints', 'sl_agp_model', checkpoint_params, args.epochs)
        torch.save({
            'model_state_dict': agp_model.state_dict(),
            'args': vars(args),
            'training_method': 'supervised_learning',
            'num_train_samples': len(small_train_dataset),
            'num_val_samples': len(small_val_dataset)
        }, checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
        
        eval_results = supervised_eval(agp_model, small_val_dataset, batch_size=1)
        print("Evaluation complete.")
        
        # Optionally save results for further analysis
        import json
        results_summary = {
            'args': vars(args),
            'num_train_samples': len(small_train_dataset),
            'num_val_samples': len(small_val_dataset),
            'coverage_stats': eval_results['stats']['coverage_stats'],
            'efficiency_stats': eval_results['stats']['efficiency_stats'],
            'size_ratio_stats': eval_results['stats']['ratio_stats']
        }
        
        # Save to results directory
        os.makedirs('results', exist_ok=True)
        with open('results/sl_agp_evaluation.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        print("Results summary saved to results/sl_agp_evaluation.json")
    except Exception as e:
        print(f"Exception occurred: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
