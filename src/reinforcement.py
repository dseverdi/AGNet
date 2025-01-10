import argparse
import pandas as pd
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from data_types import VisSample, VisDataset, collate_fn

import numpy as np

import PtrNet

def get_model_name_and_create_paths(args):
    0

from demo import evaluate_polygon_visibility_numpy_wo_gt, CustomError


def batch_eval_coverage(seq, seq_lens, pointer_indices):
        
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
            coverage = evaluate_polygon_visibility_numpy_wo_gt(points, (pointers_i - 1).cpu().numpy())
            coverages[i] = coverage
            relative_lengths[i] = pointers_i.size(0) / seq_len
            fine[i] = 1.0
        except:
            raise CustomError("evaluate_polygon_visibility_numpy_wo_gt")
    
    return coverages, relative_lengths, fine

def evaluate_model(model, data_loader, device, writer, epoch_i, mode='Validation'):
    model.eval()
    total_coverage = 0
    total_relative_length = 0
    total_rewards = 0
    total_samples = 0

    with torch.no_grad():
        for seq, seq_lens, positions in tqdm(data_loader, desc=f"Evaluating {mode}"):
            seq, positions = seq.to(device), positions.to(device)
            pointer_indices, pointer_log_scores = model(seq, seq_lens)[0]

            coverages, relative_lengths, fine = batch_eval_coverage(seq.cpu(), seq_lens, pointer_indices)
            coverages = coverages.to(device)
            relative_lengths = relative_lengths.to(device)
            fine = fine.to(device)

            gamma_1 = 0.5
            gamma_2 = 0.5

            rewards = gamma_1 * (1 - coverages) + gamma_2 * relative_lengths

            total_coverage += coverages.sum().item()
            total_relative_length += relative_lengths.sum().item()
            total_rewards += rewards.sum().item()
            total_samples += seq.size(1)

    avg_coverage = total_coverage / total_samples
    avg_relative_length = total_relative_length / total_samples
    avg_reward = total_rewards / total_samples

    print(f"{mode} set - avg. coverage: {avg_coverage}, avg. rel. length: {avg_relative_length}, avg. reward: {avg_reward}")

    # Log statistics to TensorBoard
    writer.add_scalar(f'{mode}/Coverage', avg_coverage, epoch_i)
    writer.add_scalar(f'{mode}/Relative Length', avg_relative_length, epoch_i)
    writer.add_scalar(f'{mode}/Reward', avg_reward, epoch_i)

    model.train()

def train_model(model, training_generator, validation_generator, optimizer, device, writer, num_epochs):
    max_grad_norm = 2
    best_coverage = 0
    best_epoch = 0

    for epoch_i in range(num_epochs):
        critic_exp_mvg_avg = torch.zeros(1).to(device)
        
        for batch_idx, (seq, seq_lens, positions) in enumerate(tqdm(training_generator, desc=f"Epoch {epoch_i+1}/{num_epochs}")):
            seq, positions = seq.to(device), positions.to(device)
            
            # In RL setup, we dont have labeled data, therefore, we don't use positions
            # solution is sampled from the model
            pointer_indices, pointer_log_scores = model(seq, seq_lens)[0]
            
            coverages, relative_lengths, fine = batch_eval_coverage(seq.cpu(), seq_lens, pointer_indices)
            coverages = coverages.to(device)
            relative_lengths = relative_lengths.to(device)
            fine = fine.to(device)
            
            gamma_1 = 0.5
            gamma_2 = 0.5
            
            rewards = gamma_1 * (1 - coverages) + gamma_2 * relative_lengths
            
            beta = 0.9
            
            if batch_idx == 0:
                critic_exp_mvg_avg = rewards.mean()
            else:    
                critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * rewards.mean())                    
            
            advantage = rewards - critic_exp_mvg_avg
            
            batch_size = seq.size(1)
            log_probs = torch.empty(batch_size, dtype=advantage.dtype, device=device)
            for b_i in range(batch_size):
                pointers_i = pointer_indices[:, b_i]
                zero_indices = (pointers_i == 0).nonzero()
                has_zero = zero_indices.size(0) > 0
                if has_zero:
                    first_zero_index = zero_indices[0, 0].item()
                    pointers_i = pointers_i[:first_zero_index]
                
                pointers_i_y = pointers_i
                pointers_i_x = torch.arange(pointers_i.size(0))
                log_probs_sum_i = pointer_log_scores[pointers_i_x, pointers_i_y, b_i].sum()
                log_probs[b_i] = log_probs_sum_i
            
            reinforce = advantage * log_probs            
            loss = reinforce.mean()

            optimizer.zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm), norm_type=2)

            optimizer.step()

            critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

            if batch_idx % 10 == 0:
                avg_coverage = coverages.mean().item()
                avg_rel_length = relative_lengths.mean().item()
                avg_reward = rewards.mean().item()
                #print(f"epoch: {epoch_i}, batch_idx: {batch_idx}, avg. coverage: {avg_coverage}, avg. rel. length: {avg_rel_length}, avg. reward: {avg_reward}")

                # Log statistics to TensorBoard
                writer.add_scalar('Train/Loss', loss.item(), epoch_i * len(training_generator) + batch_idx)
                writer.add_scalar('Train/Coverage', avg_coverage, epoch_i * len(training_generator) + batch_idx)
                writer.add_scalar('Train/Relative Length', avg_rel_length, epoch_i * len(training_generator) + batch_idx)
                writer.add_scalar('Train/Reward', avg_reward, epoch_i * len(training_generator) + batch_idx)

        # Evaluate the model on the validation set
        evaluate_model(model, validation_generator, device, writer, epoch_i, mode='Validation')
        # save checkpoint
        torch.save(model, os.path.join(model_path, f'model_epoch_{epoch_i}.pt'))
        # find best model
       # Check if the current model has the best coverage
        if avg_coverage > best_coverage:
            best_coverage = avg_coverage
            best_epoch = epoch_i
            torch.save(model.state_dict(), os.path.join(model_path, 'best_model.pth'))
            print(f"New best model saved at epoch {epoch_i} with coverage {avg_coverage}")

    print(f"Best model found at epoch {best_epoch} with coverage {best_coverage}")

    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pointer network for Terrain guarding predictions.')
    parser.add_argument('--batch-size', type=int, default=64, help='training batch size')
    parser.add_argument('--normalize', action='store_true', help='normalize inputs to unit square')
    parser.add_argument('--bidirectional', action='store_true', help='Bidirectional encoder LSTM')
    parser.add_argument('--hidden-size', type=int, default=256, help='LSTM hidden dimension size')
    parser.add_argument('--hidden-v', type=int, default=256, help='Attention layer hidden size')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay for Adam optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for Adam optimizer')
    parser.add_argument('--max-decoded-length', type=int, default=200, help='Maximum allowed sequence length for decoder')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--log-interval', type=int, default=200, help='Print epoch state every log-interval interval mini batches')
    parser.add_argument('--train-data', type=str, default='train', help='path to training samples')
    parser.add_argument('--valid-data', type=str, default='dev', help='path to validation samples')
    parser.add_argument('--test-data', type=str, default='test', help='path to test samples')
    parser.add_argument('--sample-size', type=int, default=100, help='Number of samples to use')
    args = parser.parse_args()

    dataset_dir = "dataset/development"

    train_path = os.path.join(dataset_dir,f'{args.train_data}')
    valid_path = os.path.join(dataset_dir,f'{args.valid_data}')
    test_path = os.path.join(dataset_dir,f'{args.test_data}')
    
    model_name = f'ne-{args.num_epochs}_bs-{args.batch_size}_hs-{args.hidden_size}_hv-{args.hidden_v}_wd-{args.wd}_lr-{args.lr}_ss-{args.sample_size}'
    if args.bidirectional:
        model_name += '_bidirectional'
    
    if args.normalize:
        model_name += '_normalized'

    model_path = f'./trained_models/{model_name}/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)    
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    dataloader_params = {'batch_size': args.batch_size,
                         'shuffle': False,
                         'num_workers' : 6}
    
    samples = {s : [f for f in os.listdir(f"{dataset_dir}/{s}") if f.endswith('.pol')] for s in ['train','dev','test']}
    
    df = pd.DataFrame.from_dict(samples, orient='index').transpose()
    
         
    sample_size = 100
    training_set = VisDataset()
    paths = [f"{train_path}/{filename}" for filename in df["train"].tolist() if filename is not None]
    training_set.extend([VisDataset(VisSample.read_samples(path=path,normalize = args.normalize)[:sample_size]) for path in paths]) 
    training_generator = DataLoader(training_set, **dataloader_params, collate_fn=collate_fn)   
    
    validation_set = VisDataset()
    valid_paths = [f"{valid_path}/{filename}" for filename in df["dev"].tolist() if filename is not None]
    
    validation_set.extend([VisDataset(VisSample.read_samples(path=path, normalize=args.normalize)[:sample_size]) for path in valid_paths])
    validation_generator = DataLoader(validation_set, **dataloader_params, collate_fn=collate_fn)

    model_args = {'hidden_size': args.hidden_size,
                  'bidirectional': args.bidirectional,
                  'device': device,
                  'teacher_forcing_ratio': 0.0,
                  'max_decoded_length': args.max_decoded_length,
                  'num_sols': 1}
    
    model = PtrNet.PointerNetwork(model_args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f'./logs/{model_name}')

    # Train the model
    train_model(model, training_generator, validation_generator, optimizer, device, writer, args.num_epochs)

    writer.close()

    # Evaluate the model on the test set
    test_set = VisDataset()
    test_paths = [f"{test_path}/{filename}" for filename in df["test"].tolist() if filename is not None]
    test_set.extend([VisDataset(VisSample.read_samples(path=path, normalize=args.normalize)[:sample_size]) for path in test_paths])
    test_generator = DataLoader(test_set, **dataloader_params, collate_fn=collate_fn)

    evaluate_model(model, test_generator, device, writer, args.num_epochs, mode='Test')