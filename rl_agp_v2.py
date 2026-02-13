import os
from dotenv import load_dotenv
import argparse
from dataset import Dataset, agp_read_samples, collate_fn
from models import create_actor, create_critic
from utils import createPolygon, compute_visibility
import torch
import json
import time
from typing import Dict, List, Optional
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
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    print("Warning: tqdm not available, progress bars will be disabled")
#from rewards import linear_reward as reward  # Use the new reward function
#from rewards import strict_reward as reward  # Use the strict reward function
#from rewards import enhanced_penalty as reward  # Use the smooth reward function
#from rewards import strict_reward as reward  # Use the smooth reward function
from rewards import coverage_gated_hysteresis_reward as reward  # Coverage-gated hysteresis reward

from eval_reporting import make_report

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


def _load_json_config(path: str) -> Dict:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config JSON must be an object at the top level.")
    return data


def _coerce_value(value, default):
    if default is None:
        return value
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, str):
        return str(value)
    return value


def _get_explicit_args(parser: argparse.ArgumentParser, argv: List[str]) -> set:
    explicit = set()
    for action in parser._actions:
        if not action.option_strings:
            continue
        for opt in action.option_strings:
            if opt in argv:
                explicit.add(action.dest)
                break
            prefix = opt + "="
            if any(arg.startswith(prefix) for arg in argv):
                explicit.add(action.dest)
                break
    return explicit


def _apply_config_to_args(args, config: Dict, defaults: Dict, keys: List[str], explicit_args: set) -> None:
    for key in keys:
        if key in config and key not in explicit_args and getattr(args, key) == defaults.get(key):
            setattr(args, key, _coerce_value(config[key], defaults.get(key)))

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
def create_agp_model(embedding_size, hidden_size, n_glimpses, tanh_exploration, use_tanh, reward, temperature, eos_logit_bias_init=None, eos_logit_bias_learnable=False):
    return create_actor(
        embedding_size, hidden_size, None, n_glimpses,
        tanh_exploration, use_tanh, "Bahdanau", reward, temperature=temperature,
        eos_logit_bias_init=eos_logit_bias_init,
        eos_logit_bias_learnable=eos_logit_bias_learnable,
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


def _best_of_k_rollouts(
    model,
    batch_data,
    mask,
    lengths,
    batch_names,
    reward_fn,
    multi_start,
    episode_log,
):
    device = batch_data.device
    bsz = batch_data.size(0)
    best_rewards = [-float("inf")] * bsz
    best_log_probs: List[torch.Tensor] = [None] * bsz
    best_steps: List[int] = [1] * bsz
    best_meta: List[Optional[Dict]] = [None] * bsz

    # Pre-extract per-instance points for reuse
    points_list = []
    lengths_list = []
    for i in range(bsz):
        n = lengths[i].item() if lengths is not None else len(batch_data[i])
        lengths_list.append(int(n))
        points_list.append(batch_data[i, :n].detach().cpu().numpy())

    for _ in range(int(multi_start)):
        selected_idxs, log_probs = model(batch_data, padding_mask=mask, lengths=lengths, deterministic=False)
        for i, (idxs, name) in enumerate(zip(selected_idxs, batch_names)):
            n = lengths_list[i]
            real_solution = [idx for idx in idxs if idx < n]
            r_out = reward_fn(points_list[i], real_solution, name, length=n, return_meta=bool(episode_log))
            if isinstance(r_out, tuple) and len(r_out) == 2:
                r, meta = r_out
            else:
                r, meta = r_out, None
            if float(r) > best_rewards[i]:
                best_rewards[i] = float(r)
                best_log_probs[i] = log_probs[i]
                best_steps[i] = max(1, len(idxs))
                best_meta[i] = meta

    # Fallback if any log_prob missing
    for i in range(bsz):
        if best_log_probs[i] is None:
            best_log_probs[i] = torch.tensor(0.0, device=device)
            best_steps[i] = 1
    rewards = torch.tensor(best_rewards, dtype=torch.float32, device=device)
    # Normalize trajectory log-prob by decoded length to reduce update scale variance
    # between short (early-EOS) and long (near-all-guards) rollouts.
    step_counts = torch.tensor(best_steps, dtype=torch.float32, device=device)
    log_probs = torch.stack(best_log_probs) / step_counts

    if episode_log:
        for i, meta in enumerate(best_meta):
            if meta is not None:
                print(
                    f"[episode] {batch_names[i]} | cov={meta['coverage']:.4f} | guards={meta['num_guards']} "
                    f"| reward={meta['reward']:.4f} | regime={meta['regime']}"
                )

    return rewards, log_probs


def _greedy_baseline_rewards(
    model,
    batch_data,
    mask,
    lengths,
    batch_names,
    reward_fn,
):
    device = batch_data.device
    selected_idxs, _ = model(batch_data, padding_mask=mask, lengths=lengths, deterministic=True)
    rewards_list = []
    for i, (data_tensor, idxs, name) in enumerate(zip(batch_data.cpu(), selected_idxs, batch_names)):
        n = lengths[i].item() if lengths is not None else len(data_tensor)
        real_points = data_tensor[:n].detach().cpu().numpy()
        real_solution = [idx for idx in idxs if idx < n]
        r = reward_fn(real_points, real_solution, name, length=n, return_meta=False)
        rewards_list.append(float(r))
    return torch.tensor(rewards_list, dtype=torch.float32, device=device)



def reinforce_train_ema(
    model,
    dataset,
    reward_fn,
    epochs=20,
    batch_size=20,
    lr=1e-3,
    beta=0.99,
    checkpoint_dir=None,
    checkpoint_params=None,
    epoch_eval_fn=None,
    start_epoch=0,
    episode_log=False,
    multi_start=1,
    baseline_mode="ema",
    epoch_start_fn=None,
    entropy_weight=0.01,
    grad_clip=1.0,
    resume_optimizer_state=None,
    resume_baseline=None,
    save_best=True,
    early_stop_on_collapse=True,
    collapse_patience=2,
    collapse_min_good_cov=0.5,
    collapse_low_cov=0.3,
):
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
        checkpoint_dir: Directory to save intermediate checkpoints (optional).
        checkpoint_params: Parameters dict for checkpoint naming (optional).
        epoch_eval_fn: Optional callback(epoch:int) to report training metrics per epoch.
        start_epoch: Starting epoch number (for resume).
        entropy_weight: Weight for entropy bonus to prevent policy collapse.
        grad_clip: Max gradient norm for clipping (0 = no clipping).
    """
    print(f"\n--- Training on {len(dataset)} samples for {epochs} epochs (batch size {batch_size}) ---")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lengths = get_lengths_from_dataset(dataset)
    batch_sampler = BucketBatchSampler(lengths, batch_size, shuffle=True, bucket_size=10)
    loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, pin_memory=True, num_workers=0)
    device = next(model.parameters()).device
    # Initialize EMA baseline (resume if provided, else set from first batch)
    baseline = resume_baseline
    if resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
        except Exception as e:
            print(f"[warn] could not restore optimizer state from checkpoint: {e}")
    
    # Track last checkpoint path for cleanup
    last_checkpoint_path = None
    best_cov_greedy = -float("inf")
    best_cov_stoch = -float("inf")
    collapse_streak = 0
    reached_good_region = False
    
    # Use tqdm for epoch progress if available
    epoch_iterator = tqdm(range(epochs), desc="Training Epochs") if tqdm else range(epochs)
    
    for epoch in epoch_iterator:
        actual_epoch = start_epoch + epoch + 1
        if epoch_start_fn is not None:
            epoch_start_fn(actual_epoch)
        total_loss = 0
        batch_count = 0
        
        # Use tqdm for batch progress if available
        batch_iterator = tqdm(loader, desc=f"Epoch {actual_epoch}/{start_epoch + epochs} Batches", leave=False) if tqdm else loader
        
        for batch_data, mask, lengths, batch_names in batch_iterator:
            batch_data = batch_data.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            lengths = torch.tensor(lengths, dtype=torch.long, device=device)
            # Forward pass: multi-start rollouts, pick best reward per instance
            rewards, log_probs = _best_of_k_rollouts(
                model,
                batch_data,
                mask,
                lengths,
                batch_names,
                reward_fn,
                multi_start=multi_start,
                episode_log=episode_log,
            )

            if baseline_mode == "greedy":
                greedy_rewards = _greedy_baseline_rewards(
                    model,
                    batch_data,
                    mask,
                    lengths,
                    batch_names,
                    reward_fn,
                )
                advantages = rewards - greedy_rewards
            else:
                # EMA baseline (cold-start safe: init from first batch)
                batch_mean = rewards.mean().item()
                if baseline is None:
                    baseline = batch_mean
                else:
                    baseline = beta * baseline + (1 - beta) * batch_mean
                advantages = rewards - baseline
            # REINFORCE loss using advantages instead of raw rewards
            pg_loss = -(log_probs * advantages).mean()
            # Entropy bonus: encourage exploration, prevent policy collapse
            # log_probs are negative sums of per-step log P(action).
            # More negative = more spread out (higher entropy) = good.
            # We want to MAXIMIZE entropy, i.e. push log_probs more negative.
            # loss += entropy_weight * log_probs.mean()  (adding a positive * negative = subtracting)
            # This rewards the model for maintaining high-entropy (exploratory) policies.
            ent_bonus = entropy_weight * log_probs.mean() if entropy_weight > 0 else 0.0
            loss = pg_loss + ent_bonus
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * batch_data.size(0)
            batch_count += 1
            
            # Update tqdm postfix with current loss
            if tqdm and hasattr(batch_iterator, 'set_postfix'):
                baseline_disp = baseline if baseline is not None else 0.0
                batch_iterator.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'baseline': f"{baseline_disp:.3f}"
                })
            
            # Free up memory after each batch
            del batch_data, mask, lengths, log_probs, rewards
            torch.cuda.empty_cache()
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {actual_epoch}/{start_epoch + epochs} - Avg loss: {avg_loss:.4f}")
        eval_metrics = None
        if epoch_eval_fn is not None:
            eval_metrics = epoch_eval_fn(actual_epoch)
            model.train()

        # Save best checkpoints based on eval coverage
        if save_best and checkpoint_dir and isinstance(eval_metrics, dict):
            cov_g = eval_metrics.get("coverage_greedy_mean", None)
            cov_s = eval_metrics.get("coverage_stoch_mean", None)

            if cov_g is not None and cov_g > best_cov_greedy:
                best_cov_greedy = cov_g
                os.makedirs(checkpoint_dir, exist_ok=True)
                best_g_path = os.path.join(checkpoint_dir, "rl_agp_ema_best_greedy.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': actual_epoch,
                    'baseline': baseline,
                    'best_metric': 'coverage_greedy_mean',
                    'best_value': float(cov_g),
                    'checkpoint_params': checkpoint_params
                }, best_g_path)
                print(f"  Saved best-greedy checkpoint: {best_g_path} (cov={cov_g:.3f})")

            if cov_s is not None and cov_s > best_cov_stoch:
                best_cov_stoch = cov_s
                os.makedirs(checkpoint_dir, exist_ok=True)
                best_s_path = os.path.join(checkpoint_dir, "rl_agp_ema_best_stoch.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': actual_epoch,
                    'baseline': baseline,
                    'best_metric': 'coverage_stoch_mean',
                    'best_value': float(cov_s),
                    'checkpoint_params': checkpoint_params
                }, best_s_path)
                print(f"  Saved best-stoch checkpoint: {best_s_path} (cov={cov_s:.3f})")

            # Early stop if model collapses after reaching good greedy coverage
            if early_stop_on_collapse and cov_g is not None:
                if cov_g >= float(collapse_min_good_cov):
                    reached_good_region = True
                if reached_good_region and cov_g <= float(collapse_low_cov):
                    collapse_streak += 1
                else:
                    collapse_streak = 0
                if collapse_streak >= int(collapse_patience):
                    print(
                        f"[early-stop] collapse detected: greedy cov <= {collapse_low_cov} for "
                        f"{collapse_streak} epochs after reaching >= {collapse_min_good_cov}."
                    )
                    break
        
        # Save checkpoint every 5 epochs
        if checkpoint_dir and checkpoint_params and actual_epoch % 5 == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"rl_agp_ema_intermediate_epoch{actual_epoch}.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': actual_epoch,
                'baseline': baseline,
                'checkpoint_params': checkpoint_params
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")
            
            # Remove previous checkpoint
            if last_checkpoint_path and os.path.exists(last_checkpoint_path):
                try:
                    os.remove(last_checkpoint_path)
                    print(f"  Removed old checkpoint: {last_checkpoint_path}")
                except Exception as e:
                    print(f"  Warning: Could not remove old checkpoint: {e}")
            last_checkpoint_path = ckpt_path
        
        torch.cuda.empty_cache()
    print("Training done.")

# --- Reinforce with learned critic ---
def reinforce_train_critic(
    actor,
    critic,
    dataset,
    reward_fn,
    epochs=2,
    batch_size=1,
    lr_actor=1e-3,
    lr_critic=1e-3,
    checkpoint_dir=None,
    checkpoint_params=None,
    epoch_eval_fn=None,
    start_epoch=0,
    episode_log=False,
    multi_start=1,
    epoch_start_fn=None,
):
    """
    Actor-critic training: policy network (actor) and value network (critic).
    Critic is trained via MSE to match observed return; actor via advantage.
    Args:
        checkpoint_dir: Directory to save intermediate checkpoints (optional).
        checkpoint_params: Parameters dict for checkpoint naming (optional).
        epoch_eval_fn: Optional callback(epoch:int) to report training metrics per epoch.
        start_epoch: Starting epoch number (for resume).
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
    
    # Track last checkpoint path for cleanup
    last_checkpoint_path = None
    
    # Use tqdm for epoch progress if available
    epoch_iterator = tqdm(range(epochs), desc="AC Training Epochs") if tqdm else range(epochs)
    
    for epoch in epoch_iterator:
        actual_epoch = start_epoch + epoch + 1
        if epoch_start_fn is not None:
            epoch_start_fn(actual_epoch)
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        batch_count = 0
        
        # Use tqdm for batch progress if available
        batch_iterator = tqdm(loader, desc=f"Epoch {actual_epoch}/{start_epoch + epochs} Batches", leave=False) if tqdm else loader
        
        for batch_data, mask, lengths, batch_names in batch_iterator:
            batch_data = batch_data.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            lengths = torch.tensor(lengths, dtype=torch.long, device=device)
            # Actor forward (multi-start best-of-K)
            rewards, log_probs = _best_of_k_rollouts(
                actor,
                batch_data,
                mask,
                lengths,
                batch_names,
                reward_fn,
                multi_start=multi_start,
                episode_log=episode_log,
            )
            # Critic forward
            values = critic(batch_data, mask, lengths)
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
            batch_count += 1
            
            # Update tqdm postfix with current losses
            if tqdm and hasattr(batch_iterator, 'set_postfix'):
                batch_iterator.set_postfix({
                    'actor_loss': f"{actor_loss.item():.4f}",
                    'critic_loss': f"{critic_loss.item():.4f}"
                })
            
            # Mem cleanup
            del batch_data, mask, lengths, selected_idxs, log_probs, values, rewards
            torch.cuda.empty_cache(); import gc; gc.collect()
        print(f"Epoch {actual_epoch}/{start_epoch + epochs} - Actor loss: {total_actor_loss/len(dataset):.4f}" \
              f", Critic loss: {total_critic_loss/len(dataset):.4f}")
        if epoch_eval_fn is not None:
            epoch_eval_fn(actual_epoch)
            actor.train(); critic.train()
        
        # Save checkpoint every 5 epochs
        if checkpoint_dir and checkpoint_params and actual_epoch % 5 == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"rl_agp_critic_intermediate_epoch{actual_epoch}.pt")
            torch.save({
                'actor_state_dict': actor.state_dict(),
                'critic_state_dict': critic.state_dict(),
                'optimizer_actor_state_dict': opt_actor.state_dict(),
                'optimizer_critic_state_dict': opt_critic.state_dict(),
                'epoch': actual_epoch,
                'checkpoint_params': checkpoint_params
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")
            
            # Remove previous checkpoint
            if last_checkpoint_path and os.path.exists(last_checkpoint_path):
                try:
                    os.remove(last_checkpoint_path)
                    print(f"  Removed old checkpoint: {last_checkpoint_path}")
                except Exception as e:
                    print(f"  Warning: Could not remove old checkpoint: {e}")
            last_checkpoint_path = ckpt_path
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


def _read_opt_solution(sol_dir: str, name: str):
    base_name = os.path.splitext(os.path.basename(name))[0]
    opt_sol_path = os.path.join(sol_dir, f"{base_name}.solution")
    try:
        with open(opt_sol_path, 'r') as f:
            lines = f.read().splitlines()
            if len(lines) >= 2:
                return [int(x) for x in lines[1].split()]
    except Exception:
        return None
    return None


@torch.no_grad()
def evaluate_rl_agp_simple(model, dataset, sol_dir: str, eval_k: int):
    model.eval()
    per_instance = []
    device = next(model.parameters()).device

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)
    count = 0
    for batch_data, mask, lengths, batch_names in loader:
        if count >= eval_k:
            break
        batch_data = batch_data.to(device)
        mask = mask.to(device)
        lengths = torch.tensor(lengths, dtype=torch.long, device=device)

        # Deterministic (greedy) decode
        t0 = time.perf_counter()
        selected_idxs, _ = model(batch_data, padding_mask=mask, lengths=lengths, deterministic=True)
        dt = float(time.perf_counter() - t0)

        # Also do one stochastic decode for comparison
        selected_idxs_stoch, _ = model(batch_data, padding_mask=mask, lengths=lengths, deterministic=False)

        name = batch_names[0]
        n = int(lengths[0].item())

        pred_indices = [int(idx) for idx in selected_idxs[0] if int(idx) < n]
        pred_indices_stoch = [int(idx) for idx in selected_idxs_stoch[0] if int(idx) < n]

        try:
            coverage = evaluate_polygon_visibility_numpy_wo_gt(
                batch_data[0, :n].detach().cpu().numpy(),
                np.array(pred_indices, dtype=np.int64),
                name,
            )
        except Exception:
            coverage = None

        try:
            coverage_stoch = evaluate_polygon_visibility_numpy_wo_gt(
                batch_data[0, :n].detach().cpu().numpy(),
                np.array(pred_indices_stoch, dtype=np.int64),
                name,
            ) if len(pred_indices_stoch) > 0 else 0.0
        except Exception:
            coverage_stoch = None

        opt_sol = _read_opt_solution(sol_dir, name)
        opt_size = len(opt_sol) if opt_sol else None
        approx_ratio = (float(len(pred_indices)) / float(opt_size)) if opt_size else None

        per_instance.append(
            {
                "name": name,
                "n": n,
                "guards": int(len(pred_indices)),
                "guard_ratio": float(len(pred_indices)) / max(1.0, float(n)),
                "coverage": coverage,
                "opt_size": opt_size,
                "approx_ratio": approx_ratio,
                "time_s": dt,
                "guards_stoch": int(len(pred_indices_stoch)),
                "guard_ratio_stoch": float(len(pred_indices_stoch)) / max(1.0, float(n)),
                "coverage_stoch": coverage_stoch,
            }
        )
        count += 1

    return per_instance



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
    parser.add_argument('--epoch-eval-k', type=int, default=-1, help="Training samples to evaluate per epoch (-1 = all)")
    parser.add_argument('--resume-from', type=str, default=None, help="Resume training from intermediate checkpoint (path to .pt file)")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--multi-start', type=int, default=1, help='Number of rollouts per instance (best-of-K).')
    parser.add_argument('--baseline', type=str, default='ema', choices=['ema', 'greedy'], help='Baseline mode for EMA training.')
    parser.add_argument('--guard-weight', type=float, default=0.5, help='Weight for guard penalty in R = coverage - gw*(|S|/n).')
    parser.add_argument('--eos-bias-init', type=float, default=None, help='Initial EOS logit bias (e.g. 2.0). None = no bias.')
    parser.add_argument('--eos-bias-final', type=float, default=None, help='Final EOS logit bias for linear decay schedule.')
    parser.add_argument('--eos-bias-decay-epochs', type=int, default=0, help='Epochs for linear EOS bias decay (<=0 uses total epochs when final is set).')
    parser.add_argument('--eos-bias-learnable', action='store_true', default=False, help='Make EOS bias a learnable parameter (default: fixed buffer).')
    parser.add_argument('--entropy-weight', type=float, default=0.01, help='Entropy bonus weight to prevent policy collapse.')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Max gradient norm for clipping (0 = no clipping).')
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config with training parameters.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--ema', action='store_true', help='Use EMA baseline (reinforce_train_ema)')
    group.add_argument('--critic', action='store_true', help='Use learned critic (reinforce_train_critic)')
    args = parser.parse_args()

    defaults = {action.dest: parser.get_default(action.dest) for action in parser._actions if action.dest != 'help'}
    explicit_args = _get_explicit_args(parser, sys.argv[1:])
    if args.config:
        config = _load_json_config(args.config)
        training_keys = [
            'epochs',
            'batch_size',
            'lr',
            'beta',
            'temperature',
            'train_size',
            'epoch_eval_k',
            'resume_from',
            'checkpoint_dir',
            'multi_start',
            'baseline',
            'guard_weight',
            'eos_bias_init',
            'eos_bias_final',
            'eos_bias_decay_epochs',
            'entropy_weight',
            'grad_clip',
        ]
        _apply_config_to_args(args, config, defaults, training_keys, explicit_args)

    def vprint(msg: str) -> None:
        if bool(args.verbose):
            print(msg)

    vprint("[phase] preparing datasets")
    train_dataset, val_dataset = prepare_datasets(args.agp_train_dir, args.agp_val_dir, normalize=True)
    vprint("[phase] creating models")
    agp_model = create_agp_model(
        args.embedding_size, args.hidden_size, args.n_glimpses, args.tanh_exploration, args.use_tanh, reward, args.temperature,
        eos_logit_bias_init=args.eos_bias_init,
        eos_logit_bias_learnable=args.eos_bias_learnable,
    )

    # Create critic model (optional, can be used for actor-critic training)
    critic_model = create_critic_model(
        args.embedding_size, args.hidden_size, args.n_glimpses, "Bahdanau"
    )

    # Use only args.train_size samples for training if available
    size = args.train_size
    small_train_dataset = train_dataset if len(train_dataset) <= size else Dataset(train_dataset.samples[:size])
    small_val_dataset = val_dataset if len(val_dataset) <= size else Dataset(val_dataset.samples[:size])
    
    # Load from intermediate checkpoint if resuming
    start_epoch = 0
    resume_optimizer_state = None
    resume_baseline = None
    if args.resume_from:
        vprint(f"[phase] resuming from checkpoint: {args.resume_from}")
        device = next(agp_model.parameters()).device
        ckpt = torch.load(args.resume_from, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                agp_model.load_state_dict(ckpt['model_state_dict'])
            elif 'actor_state_dict' in ckpt:
                agp_model.load_state_dict(ckpt['actor_state_dict'])
                if 'critic_state_dict' in ckpt and args.critic:
                    critic_model.load_state_dict(ckpt['critic_state_dict'])
            start_epoch = int(ckpt.get('epoch', 0))
            resume_optimizer_state = ckpt.get('optimizer_state_dict', None)
            resume_baseline = ckpt.get('baseline', None)
            vprint(f"[phase] resuming from epoch {start_epoch}")
    
    # --- Weighted-sum reward: R = coverage - guard_weight * (|S| / n) ---
    def reward_fn(points, solution, name, length=None, return_meta=False):
        try:
            coverage = evaluate_polygon_visibility_numpy_wo_gt(
                points,
                np.array(solution, dtype=np.int64),
                name,
            )
        except Exception:
            coverage = 0.0
        num_guards = int(len(solution))
        n = float(max(1, length if length is not None else num_guards))
        cov = float(coverage)
        guard_cost = float(args.guard_weight) * (float(num_guards) / n)
        reward_value = cov - guard_cost
        if return_meta:
            return reward_value, {
                "coverage": cov,
                "num_guards": num_guards,
                "guard_ratio": float(num_guards) / n,
                "reward": reward_value,
            }
        return reward_value

    def _set_eos_bias_value(model, value: float) -> bool:
        actor = getattr(model, "actor", None)
        if actor is None:
            return False
        eos_bias = getattr(actor, "eos_logit_bias", None)
        if eos_bias is None:
            return False
        try:
            with torch.no_grad():
                if torch.is_tensor(eos_bias):
                    eos_bias.copy_(torch.tensor(float(value), device=eos_bias.device, dtype=eos_bias.dtype))
                else:
                    actor.eos_logit_bias = float(value)
            return True
        except Exception:
            return False

    eos_decay_active = args.eos_bias_init is not None and args.eos_bias_final is not None
    eos_decay_total_epochs = int(args.eos_bias_decay_epochs) if int(args.eos_bias_decay_epochs) > 0 else int(args.epochs)
    if eos_decay_active and eos_decay_total_epochs <= 0:
        eos_decay_total_epochs = 1

    def epoch_start_fn(epoch: int) -> None:
        if not eos_decay_active:
            return
        # epoch is 1-based in training loop
        progress = min(1.0, max(0.0, float(epoch - 1) / float(max(1, eos_decay_total_epochs - 1))))
        current_bias = float(args.eos_bias_init) + progress * (float(args.eos_bias_final) - float(args.eos_bias_init))
        if _set_eos_bias_value(agp_model, current_bias) and bool(args.verbose):
            print(f"[sched] epoch={epoch} eos_bias={current_bias:.4f}")

    def epoch_eval_train_metrics(epoch: int):
        eval_k = len(small_train_dataset) if int(args.epoch_eval_k) < 0 else min(int(args.epoch_eval_k), len(small_train_dataset))
        per_instance = evaluate_rl_agp_simple(
            agp_model,
            small_train_dataset,
            args.agp_train_dir,
            eval_k=eval_k,
        )
        # Deterministic metrics
        covs = [x.get("coverage") for x in per_instance if x.get("coverage") is not None]
        ratios = [x.get("approx_ratio") for x in per_instance if x.get("approx_ratio") is not None]
        guard_ratios = [x.get("guard_ratio") for x in per_instance if x.get("guard_ratio") is not None]
        cov_mean = float(np.mean(covs)) if covs else None
        ratio_mean = float(np.mean(ratios)) if ratios else None
        guard_ratio_mean = float(np.mean(guard_ratios)) if guard_ratios else None
        # Stochastic metrics
        covs_s = [x.get("coverage_stoch") for x in per_instance if x.get("coverage_stoch") is not None]
        guard_ratios_s = [x.get("guard_ratio_stoch") for x in per_instance if x.get("guard_ratio_stoch") is not None]
        cov_s_mean = float(np.mean(covs_s)) if covs_s else None
        gr_s_mean = float(np.mean(guard_ratios_s)) if guard_ratios_s else None
        msg = f"[epoch {epoch}] greedy"
        if cov_mean is not None:
            msg += f" | cov={cov_mean:.3f}"
        if ratio_mean is not None:
            msg += f" | |S|/opt={ratio_mean:.2f}"
        if guard_ratio_mean is not None:
            msg += f" | |S|/n={guard_ratio_mean:.3f}"
        msg += f"  stoch"
        if cov_s_mean is not None:
            msg += f" | cov={cov_s_mean:.3f}"
        if gr_s_mean is not None:
            msg += f" | |S|/n={gr_s_mean:.3f}"
        print(msg)
        return {
            "coverage_greedy_mean": cov_mean,
            "coverage_stoch_mean": cov_s_mean,
            "guard_ratio_greedy_mean": guard_ratio_mean,
            "guard_ratio_stoch_mean": gr_s_mean,
            "approx_ratio_greedy_mean": ratio_mean,
        }

    # Define checkpoint params early (needed for training phase)
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

    vprint("[phase] training")
    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
    # If resuming, adjust epochs to be relative to the starting epoch
    remaining_epochs = args.epochs - start_epoch
    if remaining_epochs <= 0:
        vprint(f"[phase] training already complete (start_epoch={start_epoch} >= epochs={args.epochs})")
        training_method = 'reinforcement_learning_ema' if args.ema else 'reinforcement_learning_critic'
    else:
        vprint(f"[phase] training {remaining_epochs} more epochs (from epoch {start_epoch} to {args.epochs})")
        if args.ema:
            reinforce_train_ema(agp_model, small_train_dataset, reward_fn,
                                epochs=remaining_epochs, batch_size=args.batch_size, lr=args.lr, beta=args.beta,
                                checkpoint_dir=args.checkpoint_dir, checkpoint_params=checkpoint_params,
                                epoch_eval_fn=epoch_eval_train_metrics, start_epoch=start_epoch,
                                episode_log=bool(args.verbose),
                                multi_start=int(args.multi_start),
                                baseline_mode=str(args.baseline),
                                epoch_start_fn=epoch_start_fn,
                                entropy_weight=float(args.entropy_weight),
                                grad_clip=float(args.grad_clip),
                                resume_optimizer_state=resume_optimizer_state,
                                resume_baseline=resume_baseline)
            training_method = 'reinforcement_learning_ema'
        elif args.critic:
            reinforce_train_critic(agp_model, critic_model, small_train_dataset, reward_fn,
                                   epochs=remaining_epochs, batch_size=args.batch_size, lr_actor=args.lr, lr_critic=args.lr,
                                   checkpoint_dir=args.checkpoint_dir, checkpoint_params=checkpoint_params,
                                   epoch_eval_fn=epoch_eval_train_metrics, start_epoch=start_epoch,
                                   episode_log=bool(args.verbose),
                                   multi_start=int(args.multi_start))
            training_method = 'reinforcement_learning_critic'

    vprint("[phase] saving model")
    # Save the trained model
    # checkpoint_params already defined earlier
    
    checkpoint_path = get_checkpoint_path(args.checkpoint_dir, 'rl_agp_model', checkpoint_params, args.epochs)
    model_checkpoint = {
        'model_state_dict': agp_model.state_dict(),
        'epoch': args.epochs,
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

    vprint("[phase] evaluating")
    # evaluate the model on the validation dataset (simple report format)
    eval_k = len(small_val_dataset)
    per_instance = evaluate_rl_agp_simple(agp_model, small_val_dataset, args.agp_val_dir, eval_k)
    report = make_report(
        method="rl_agp",
        per_instance=per_instance,
        args=vars(args),
        dataset={
            "path": args.agp_val_dir,
            "eval_k": int(eval_k),
            "train_k": int(len(small_train_dataset)),
        },
        oracle={
            "mode": "exact",
            "coverage_threshold": 0.99,
            "coverage_metric": "exact",
        },
        timing={},
    )
    report["checkpoint"] = checkpoint_path

    vprint("[phase] saving report")
    out_dir = os.path.join("results", "v3", "rl_agp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "rl_agp_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n--- RL AGP eval on dataset polygons ---")
    s = report["summary"]
    msg = f"Dataset eval k={eval_k}  greedy | |S| mean={s['guards']['mean']:.2f} | |S|/n mean={s['guard_ratio']['mean']:.3f}"
    if s["coverage"]["mean"] is not None:
        msg += f" | cov mean={s['coverage']['mean']:.3f}"
    if s["approx_ratio"]["mean"] is not None:
        msg += f" | |S|/opt mean={s['approx_ratio']['mean']:.2f}"
    # Stochastic summary
    covs_s = [x.get("coverage_stoch") for x in per_instance if x.get("coverage_stoch") is not None]
    grs_s = [x.get("guard_ratio_stoch") for x in per_instance if x.get("guard_ratio_stoch") is not None]
    if covs_s:
        msg += f"  stoch | cov mean={float(np.mean(covs_s)):.3f}"
    if grs_s:
        msg += f" | |S|/n mean={float(np.mean(grs_s)):.3f}"
    print(msg)
    print(f"Results summary saved to {out_path}")


if __name__ == "__main__":
    main()
