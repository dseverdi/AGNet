import argparse
import pandas as pd
import os
import time
import random
import threading
import multiprocessing as mp
from multiprocessing import Queue, Process

import torch
import torch.optim as optim
import torch.multiprocessing as torch_mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_types import VisSample, VisDataset, collate_fn
from demo_parallel import evaluate_polygon_visibility_numpy_wo_gt, CustomError
from losses import Reward, RewardLH

# architecture
import PtrNet

# visualization
from tqdm import tqdm


def batch_eval_coverage(seq, seq_lens, pointer_indices, sample_names):
    start_time = time.time()
    points_batch = seq[1:, :, :2].numpy()
    batch_size = points_batch.shape[1]

    coverages = torch.zeros(batch_size, dtype=torch.float)
    relative_lengths = torch.ones(batch_size, dtype=torch.float)
    fine = torch.zeros(batch_size, dtype=torch.float)

    for i in range(batch_size):
        seq_len = seq_lens[i] - 1
        points = points_batch[:seq_len, i]

        pointers_i = pointer_indices[:, i]
        zero_indices = (pointers_i == 0).nonzero()
        has_zero = zero_indices.size(0) > 0
        if has_zero:
            first_zero_index = zero_indices[0, 0].item()
            pointers_i = pointers_i[:first_zero_index]

        try:
            coverage = evaluate_polygon_visibility_numpy_wo_gt(points, (pointers_i - 1).cpu().numpy(), sample_names[i])
            coverages[i] = coverage
            relative_lengths[i] = pointers_i.size(0) / seq_len
            fine[i] = 1.0
        except:
            raise CustomError("evaluate_polygon_visibility_numpy_wo_gt")

    end_time = time.time()
    avg_time = (end_time - start_time)

    return coverages, relative_lengths, fine, avg_time


def worker(rank, args, device_id, shared_model, critic_model, counter, log_dir, reward_type, cov_wt, train_data_dir, valid_data_dir):
    # Create lock inside the worker process instead of passing it
    lock = mp.Lock()
    
    torch.manual_seed(args.seed + rank)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"Worker {rank} started on device: {device}")
    
    # Initialize reward function locally in each worker
    if reward_type == "reward_manual":
        reward_function = Reward(cov_wt)
    elif reward_type == "reward_learnable":
        reward_function = RewardLH()
        
    # Create a separate writer for each worker
    writer = None
    if rank == 0:  # Only rank 0 needs to log to TensorBoard
        writer = SummaryWriter(log_dir=log_dir)
    
    # Initialize model with parameters
    model_args = {
        'hidden_size': args.hidden_size,
        'bidirectional': args.bidirectional,
        'device': device,
        'teacher_forcing_ratio': 0.0,
        'max_decoded_length': args.max_decoded_length,
        'bello_et_al': True,
        'num_sols': 1
    }
    
    # Create local models
    local_model = PtrNet.PointerNetwork(model_args).to(device)
    local_critic = PtrNet.Critic(model_args).to(device)
    
    # Load state dictionaries from shared models
    local_model.load_state_dict(shared_model.state_dict())
    local_critic.load_state_dict(critic_model.state_dict())
    
    # Create optimizer for local models
    optimizer = optim.Adam(
        list(local_model.parameters()) + list(local_critic.parameters()),
        lr=args.lr, weight_decay=args.wd
    )
    
    # Load training paths for this worker
    train_files = [f for f in os.listdir(train_data_dir) if f.endswith('.pol')]
    
    # Distribute files based on worker rank
    worker_files = []
    for i in range(rank, len(train_files), args.processes):
        worker_files.append(train_files[i])
    
    if rank == 0:
        print(f"Total training files: {len(train_files)}, distributed across {args.processes} workers")
    
    # Load some validation paths for rank 0
    valid_generator = None
    if rank == 0:
        valid_files = [f for f in os.listdir(valid_data_dir) if f.endswith('.pol')]
        val_sample_size = min(20, len(valid_files))
        valid_sample_files = random.sample(valid_files, val_sample_size)
        
        validation_set = VisDataset()
        for file in valid_sample_files:
            try:
                path = os.path.join(valid_data_dir, file)
                validation_set.extend([VisDataset(VisSample.read_samples(path=path, normalize=args.normalize))])
            except Exception as e:
                print(f"Error loading validation sample {file}: {e}")
                
        dataloader_params = {
            'batch_size': args.batch_size, 
            'shuffle': False,
            'num_workers': 0
        }
        valid_generator = DataLoader(validation_set, **dataloader_params, collate_fn=collate_fn)
        print(f"Worker {rank} loaded {len(validation_set)} validation samples")
    
    # Define chunk processing parameters
    chunk_size = min(args.chunk_size, len(worker_files))
    mini_epochs = args.num_updates // max((len(worker_files) // chunk_size), 1) + 1
    
    if rank == 0:
        print(f"Processing in chunks of {chunk_size} files for up to {mini_epochs} mini-epochs")
    
    # Training loop
    updates_since_report = 0
    last_report_time = time.time()
    
    for epoch in range(mini_epochs):
        # Load shared model state at the beginning of each epoch
        local_model.load_state_dict(shared_model.state_dict())
        local_critic.load_state_dict(critic_model.state_dict())
        
        if counter.value >= args.num_updates:
            break
            
        # Shuffle files at the start of each epoch
        random.shuffle(worker_files)
        
        # Process files in chunks
        for chunk_start in range(0, len(worker_files), chunk_size):
            if counter.value >= args.num_updates:
                break
                
            chunk_end = min(chunk_start + chunk_size, len(worker_files))
            chunk_files = worker_files[chunk_start:chunk_end]
            
            # Load this chunk of data quietly
            with torch.no_grad():
                # Load chunk data                    
                chunk_dataset = VisDataset()
                load_errors = 0
                for file in chunk_files:
                    try:
                        path = os.path.join(train_data_dir, file)
                        chunk_dataset.extend([VisDataset(VisSample.read_samples(path=path, normalize=args.normalize))])
                    except Exception:
                        load_errors += 1
                
                if len(chunk_dataset) == 0:
                    continue  # Skip empty chunks silently
                
                # Configure data loader with pinning memory disabled to save RAM
                dataloader_params = {
                    'batch_size': args.batch_size, 
                    'shuffle': True,
                    'num_workers': 0,
                    'pin_memory': False
                }
                train_generator = DataLoader(chunk_dataset, **dataloader_params, collate_fn=collate_fn)
            
            # Process batches
            values = []
            log_probs = []
            rewards = []
            entropies = []
            batch_sizes = []
            
            for batch_idx, (seq, seq_lens, positions, sample_names) in enumerate(train_generator):
                seq, positions = seq.to(device), positions.to(device)
                batch_sizes.append(seq.size(1))  # Store the batch size
                
                # Sample solution from model
                pointer_indices, pointer_log_scores = local_model(seq, seq_lens)[0]
                
                # Compute coverage and relative length
                coverages, relative_lengths, fine, avg_time = batch_eval_coverage(seq.cpu(), seq_lens, pointer_indices, sample_names)
                coverages, relative_lengths, fine = coverages.to(device), relative_lengths.to(device), fine.to(device)
                
                # Calculate rewards
                batch_rewards = reward_function(coverages, relative_lengths)
                
                # Get critic values
                critic_values = local_critic(seq, seq_lens)
                
                # Compute log probabilities and entropy
                batch_size = seq.size(1)
                batch_log_probs = torch.empty(batch_size, dtype=coverages.dtype, device=device)
                batch_entropy = torch.empty(batch_size, dtype=coverages.dtype, device=device)
                
                for b_i in range(batch_size):
                    pointers_i = pointer_indices[:, b_i]
                    zero_indices = (pointers_i == 0).nonzero()
                    has_zero = zero_indices.size(0) > 0
                    if has_zero:
                        first_zero_index = zero_indices[0, 0].item()
                        pointers_i = pointers_i[:first_zero_index]
                    
                    pointers_i_y = pointers_i
                    pointers_i_x = torch.arange(pointers_i.size(0))
                    
                    # Compute log probability sum
                    batch_log_probs[b_i] = pointer_log_scores[pointers_i_x, pointers_i_y, b_i].sum()
                    
                    # Compute entropy (fix the incomplete line)
                    probs = torch.exp(pointer_log_scores[pointers_i_x, pointers_i_y, b_i])
                    batch_entropy[b_i] = -torch.sum(probs * pointer_log_scores[pointers_i_x, pointers_i_y, b_i])
                
                # Store batch data
                values.append(critic_values)
                log_probs.append(batch_log_probs)
                rewards.append(batch_rewards)
                entropies.append(batch_entropy)
                
                # Perform A3C updates here
                if len(values) >= args.update_freq or batch_idx == len(train_generator) - 1:
                    # Process accumulated updates
                    actor_loss = 0
                    critic_loss = 0
                    entropy_loss = 0
                    
                    # Process each batch with its own return accumulation
                    for i in reversed(range(len(values))):
                        # Initialize returns
                        R = torch.zeros(batch_sizes[i], device=device)
                        
                        # Compute returns
                        R = rewards[i] + args.gamma * R
                        advantage = R - values[i]
                        critic_loss += 0.5 * advantage.pow(2).mean()
                        
                        # Compute actor loss
                        actor_loss -= (log_probs[i] * advantage.detach()).mean()
                        
                        # Add entropy term for exploration
                        entropy_loss -= args.entropy_weight * entropies[i].mean()
                    
                    # Total loss
                    total_loss = actor_loss + args.value_loss_coef * critic_loss + entropy_loss
                    
                    # Compute gradients
                    optimizer.zero_grad()
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(), args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(local_critic.parameters(), args.max_grad_norm)
                    optimizer.step()
                    
                    # Clear accumulated data
                    values.clear()
                    log_probs.clear()
                    rewards.clear()
                    entropies.clear()
                    batch_sizes.clear()
                
                # After processing the batch and updating models
                # Sync with the shared model periodically to avoid divergence
                if batch_idx % 5 == 0:  # Sync every 5 batches
                    with lock:
                        # Update shared dictionaries
                        shared_model.load_state_dict(local_model.state_dict())
                        critic_model.load_state_dict(local_critic.state_dict())
                        
                        # Update counter for progress tracking
                        counter.value += 1
                        updates_since_report += 1
                        
                        # Progress reporting
                        current_time = time.time()
                        if rank == 0 and current_time - last_report_time > 10:
                            updates_per_second = updates_since_report / (current_time - last_report_time)
                            progress_percentage = (counter.value / args.num_updates) * 100
                            est_remaining_time = (args.num_updates - counter.value) / updates_per_second if updates_per_second > 0 else float('inf')
                            
                            print(f"Progress: {counter.value}/{args.num_updates} updates ({progress_percentage:.1f}%) | "
                                  f"Speed: {updates_per_second:.2f} updates/s | "
                                  f"Est. time remaining: {est_remaining_time/60:.1f} minutes | "
                                  f"Coverage: {coverages.mean().item():.4f} | "
                                  f"Length: {relative_lengths.mean().item():.4f}")
                            
                            last_report_time = current_time
                            updates_since_report = 0
    
    # Free memory
    del train_generator
    del chunk_dataset
    torch.cuda.empty_cache()
    
    print(f"Worker {rank} completed training")
    if writer is not None:
        writer.close()


def evaluate_model(model, data_loader, device, writer, step, reward_function, mode='Validation'):
    model.eval()
    total_coverage = 0
    total_relative_length = 0
    total_rewards = 0
    total_samples = 0
    
    start_time = time.time()
    print(f"Starting {mode} evaluation...")

    with torch.no_grad():
        for batch_idx, (seq, seq_lens, positions, sample_names) in enumerate(data_loader):
            seq, positions = seq.to(device), positions.to(device)
            pointer_indices, pointer_log_scores = model(seq, seq_lens)[0]

            coverages, relative_lengths, fine, _ = batch_eval_coverage(seq.cpu(), seq_lens, pointer_indices, sample_names)
            coverages = coverages.to(device)
            relative_lengths = relative_lengths.to(device)
            fine = fine.to(device)

            rewards = reward_function(coverages, relative_lengths)

            total_coverage += coverages.sum().item()
            total_relative_length += relative_lengths.sum().item()
            total_rewards += rewards.sum().item()
            total_samples += seq.size(1)
            
            # Progress indicator for large validation sets
            if batch_idx % 10 == 0 and batch_idx > 0:
                progress = batch_idx / len(data_loader) * 100
                elapsed = time.time() - start_time
                est_total = elapsed / (batch_idx / len(data_loader))
                est_remaining = est_total - elapsed
                print(f"{mode} progress: {progress:.1f}% | Time remaining: {est_remaining:.1f}s", end='\r')

    avg_coverage = total_coverage / total_samples
    avg_relative_length = total_relative_length / total_samples
    avg_reward = total_rewards / total_samples
    
    eval_time = time.time() - start_time
    print(f"\n{mode} evaluation completed in {eval_time:.2f}s")
    print(f"{mode} set - avg. coverage: {avg_coverage:.4f}, avg. rel. length: {avg_relative_length:.4f}, avg. reward: {avg_reward:.4f}")

    # Log statistics to TensorBoard if writer exists
    if writer is not None:
        writer.add_scalar(f'{mode}/Coverage', avg_coverage, step)
        writer.add_scalar(f'{mode}/Relative_Length', avg_relative_length, step)
        writer.add_scalar(f'{mode}/Reward', avg_reward, step)

    return avg_coverage


def main():
    parser = argparse.ArgumentParser(description='A3C training for pointer network terrain guarding.')
    parser.add_argument('--batch-size', type=int, default=32, help='training batch size')
    parser.add_argument('--normalize', action='store_true', help='normalize inputs to unit square')
    parser.add_argument('--bidirectional', action='store_true', help='Bidirectional encoder LSTM')
    parser.add_argument('--hidden-size', type=int, default=256, help='LSTM hidden dimension size')
    parser.add_argument('--hidden-v', type=int, default=256, help='Attention layer hidden size')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay for Adam optimizer')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for optimizer')
    parser.add_argument('--max-decoded-length', type=int, default=200, help='Maximum allowed sequence length for decoder')
    parser.add_argument('--num-updates', type=int, default=20000, help='Number of total parameter updates to perform')
    parser.add_argument('--update-freq', type=int, default=5, help='Frequency of updates (in batches)')
    parser.add_argument('--log-interval', type=int, default=100, help='Interval for logging')
    parser.add_argument('--train-data', type=str, default='train', help='path to training samples')
    parser.add_argument('--valid-data', type=str, default='dev', help='path to validation samples')
    parser.add_argument('--test-data', type=str, default='test', help='path to test samples')
    parser.add_argument('--sample-size', type=int, default=1e6, help='Number of samples to use')
    parser.add_argument('--comment', type=str, default='', help='Additional comment for the model name')
    parser.add_argument('--cov-wt', type=float, default=0.6, help='Coverage weight')
    parser.add_argument('--reward-function', type=str, default='reward_manual', help='reward function type')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--entropy-weight', type=float, default=0.01, help='weight for entropy')
    parser.add_argument('--value-loss-coef', type=float, default=1, help='value loss coefficient')
    parser.add_argument('--max-grad-norm', type=float, default=2.0, help='max norm for gradients')
    parser.add_argument('--processes', type=int, default=4, help='Number of processes for A3C')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--chunk-size', type=int, default=100, help='Number of samples to load at once to save memory')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    dataset_dir = "/mnt/nvme0n1/dseverdi/MLAG/dataset/AG/development"

    train_path = os.path.join(dataset_dir, f'{args.train_data}')
    valid_path = os.path.join(dataset_dir, f'{args.valid_data}')
    test_path = os.path.join(dataset_dir, f'{args.test_data}')
    
    model_name = f'a3c_nu-{args.num_updates}_bs-{args.batch_size}_hs-{args.hidden_size}_hv-{args.hidden_v}_wd-{args.wd}_lr-{args.lr}_ss-{args.sample_size}_wt_cov-{args.cov_wt}'
    if args.bidirectional:
        model_name += '_bidirectional'
    
    if args.normalize:
        model_name += '_normalized'

    if args.comment:
        model_name += f'_{args.comment}'

    model_path = f'./models/a3c/trained_models/{model_name}/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    log_dir = f'./logs/{model_name}'
    
    # Print training configuration
    print("\n" + "="*50)
    print("A3C Training Configuration:")
    print("="*50)
    print(f"Model name: {model_name}")
    print(f"Number of updates: {args.num_updates}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.wd}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Bidirectional: {args.bidirectional}")
    print(f"Entropy weight: {args.entropy_weight}")
    print(f"Gamma (discount factor): {args.gamma}")
    print(f"Value loss coefficient: {args.value_loss_coef}")
    print(f"Reward function: {args.reward_function}")
    print(f"Coverage weight: {args.cov_wt}")
    print(f"Number of processes: {args.processes}")
    print("="*50 + "\n")
    
    # Create dataset
    print("Loading datasets...")
    samples = {s: [f for f in os.listdir(f"{dataset_dir}/{s}") if f.endswith('.pol')] for s in ['train', 'dev', 'test']}
    df = pd.DataFrame.from_dict(samples, orient='index').transpose()
    
    sample_size = min(int(args.sample_size), len(df["train"].dropna()))
    
    # Define training, validation and test datasets
    train_paths = [f"{train_path}/{filename}" for filename in df["train"].tolist() if filename is not None]
    valid_paths = [f"{valid_path}/{filename}" for filename in df["dev"].tolist() if filename is not None]
    test_paths = [f"{test_path}/{filename}" for filename in df["test"].tolist() if filename is not None]

    # Sample specific number of paths
    train_paths = random.sample(train_paths, min(sample_size, len(train_paths)))
    valid_paths = random.sample(valid_paths, min(sample_size, len(valid_paths)))
    test_paths = random.sample(test_paths, min(sample_size, len(test_paths)))
    
    print(f"Dataset loaded: {len(train_paths)} train, {len(valid_paths)} validation, {len(test_paths)} test")
    
    # Create datasets
    print("Creating training dataset...")
    training_set = VisDataset()
    training_set.extend([VisDataset(VisSample.read_samples(path=path, normalize=args.normalize)) for path in train_paths])
    
    validation_set = VisDataset()
    validation_set.extend([VisDataset(VisSample.read_samples(path=path, normalize=args.normalize)) for path in valid_paths])
    
    test_set = VisDataset()
    test_set.extend([VisDataset(VisSample.read_samples(path=path, normalize=args.normalize)) for path in test_paths])
    
    # Setup shared model for A3C
    mp.set_start_method('spawn', force=True)
    torch_mp.set_sharing_strategy('file_system')
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    # Initialize shared model for actor
    model_args = {'hidden_size': args.hidden_size,
                  'bidirectional': args.bidirectional,
                  'device': device,
                  'teacher_forcing_ratio': 0.0,
                  'max_decoded_length': args.max_decoded_length,
                  'bello_et_al': True,
                  'num_sols': 1}
    
    shared_model = PtrNet.PointerNetwork(model_args).to(device)
    shared_model.share_memory()
    
    # Initialize shared critic model
    critic_model = PtrNet.Critic(model_args).to(device)
    critic_model.share_memory()
    
    # Get state dictionary references that can be shared
    shared_model_state_dict = shared_model.state_dict()
    critic_model_state_dict = critic_model.state_dict()
    
    # Counter for global updates
    counter = mp.Value('i', 0)
    
    print(f"A3C training with {args.processes} processes")
    print(f"Training samples: {len(training_set)}")
    print(f"Validation samples: {len(validation_set)}")
    print(f"Test samples: {len(test_set)}")
    
    print("Setting up models...")
    
    print("Launching worker processes...")
    start_time = time.time()
    # Setup and launch worker processes
    processes = []
    try:
        for rank in range(args.processes):
            # Always use cuda:0 for all processes
            device_id = 0 if use_cuda else -1
            
            p = mp.Process(
                target=worker,
                args=(rank, args, device_id, shared_model, critic_model, 
                      counter, log_dir, args.reward_function, args.cov_wt, 
                      train_path, valid_path)  # Pass only paths to directories
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes to finish
        print("Waiting for all processes to complete...")
        for p in processes:
            p.join()
    except (KeyboardInterrupt, Exception) as e:
        print(f"Error occurred: {e}")
        for p in processes:
            if p.is_alive():
                p.terminate()
        raise
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Create a writer for the final evaluation
    final_writer = SummaryWriter(log_dir=log_dir)
    
    # Initialize reward function for final evaluation
    if args.reward_function == "reward_manual":
        reward_function = Reward(args.cov_wt)
    elif args.reward_function == "reward_learnable":
        reward_function = RewardLH()
    
    # Load test data only for final evaluation
    print("\nLoading test dataset for final evaluation...")
    test_dataset = VisDataset()
    
    # Load test data in chunks to avoid memory issues
    test_chunk_size = min(100, len(test_paths))
    for i in range(0, len(test_paths), test_chunk_size):
        chunk_paths = test_paths[i:i+test_chunk_size]
        for path in chunk_paths:
            try:
                test_dataset.extend([VisDataset(VisSample.read_samples(path=path, normalize=args.normalize))])
            except Exception as e:
                print(f"Error loading test sample {path}: {e}")
        
        #print(f"Loaded {len(test_dataset)} test samples so far...")
    
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # Evaluate final model on test set
    print("\nEvaluating final model on test set...")
    test_coverage = evaluate_model(shared_model, test_dataloader, device, final_writer, args.num_updates, reward_function, mode='Test')
    
    # Save the final model
    torch.save(shared_model.state_dict(), os.path.join(model_path, 'final_model.pth'))
    print(f"Final model saved with test coverage {test_coverage:.4f}")
    
    final_writer.close()
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()

