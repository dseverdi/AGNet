import argparse
import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from   torch.utils.data import DataLoader

import tsp_ptrnet as PtrNet
from tsp_dataset import TSPDataset, tsp_examples, collate_fn as tsp_collate_fn

   
def init_parser():  
    parser = argparse.ArgumentParser(description='Pointer network for Terrain guarding predictions.')
    parser.add_argument('--batch-size', type=int, default=64, help='training batch size')
    parser.add_argument('--normalize', action='store_true', default=True, help='normalize inputs to unit square')
    parser.add_argument('--bidirectional', action='store_true', default=True, help='Bidirectional encoder LSTM')
    parser.add_argument('--hidden-size', type=int, default=128, help='LSTM hidden dimension size')
    parser.add_argument('--hidden-v', type=int, default=128, help='Attention layer hidden size')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay for Adam optimizer')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate for Adam optimizer')
    parser.add_argument('--max-decoded-length', type=int, default=10, help='Maximum allowed sequence length for decoder')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--log-interval', type=int, default=200, help='Print epoch state every log-interval interval mini batches')
    parser.add_argument('--train-data', type=str, default='train', help='path to training samples')
    parser.add_argument('--valid-data', type=str, default='dev', help='path to validation samples')
    parser.add_argument('--test-data', type=str, default='test', help='path to test samples')
    parser.add_argument('--sample-size', type=int, default=1e6, help='Number of samples to use')
    parser.add_argument('--use-critic', action='store_true', default=True, help='Use Bello critic instead of exp_mvg_avg')
    parser.add_argument('--comment', type=str, default='', help='Additional comment for the model name')
    
    # parse arguments
    args = parser.parse_args()
    return args


def create_model_name(args):
    model_name = f'ne-{args.num_epochs}_bs-{args.batch_size}_hs-{args.hidden_size}_hv-{args.hidden_v}_wd-{args.wd}_lr-{args.lr}_ss-{args.sample_size}'
    if args.use_critic:
        model_name += '_critic'
    if args.bidirectional:
        model_name += '_bidirectional'
    if args.normalize:
        model_name += '_normalized'
    if args.comment:
        model_name += f'_{args.comment}'
        
    return model_name    


def calculate_tour_lengths(coords, tours):
    n_steps, batch_size = tours.shape

    tour_lengths_ = []

    for b in range(batch_size):
        tour = tours[:, b]
        
        # Find first 0, which marks end of tour
        zero_idx = (tour == 0).nonzero(as_tuple=True)[0]
        end = zero_idx[0].item() if len(zero_idx) > 0 else n_steps

        tour_trimmed = tour[:end]
        #print(tour_trimmed, tour)

        if len(tour_trimmed) < 2:
            tour_lengths_.append(torch.tensor(0.0, dtype=coords.dtype, device=coords.device))
            continue

        points = coords[tour_trimmed, :2]

        segment_dists = torch.norm(points[1:] - points[:-1], dim=1)
        loop_closure = torch.norm(points[0] - points[-1])
        total_length = segment_dists.sum() + loop_closure

        tour_lengths_.append(total_length)
        
    return torch.stack(tour_lengths_)

@torch.no_grad()
def calculate_ground_truth_tour_lengths(coords, tours):

    max_len, batch_size = tours.shape
    tour_lengths = []

    for b in range(batch_size):
        tour = tours[:, b]

        # Find first 0 (EOS)
        eos_pos = (tour == 0).nonzero(as_tuple=True)[0]
        end = eos_pos[0].item() if eos_pos.numel() > 0 else max_len
        tour_trimmed = tour[:end]

        # Skip empty or invalid tours
        if len(tour_trimmed) < 2:
            tour_lengths.append(torch.tensor(0.0, device=coords.device, dtype=coords.dtype))
            continue

        # Get the coordinate points for the tour
        points = coords[tour_trimmed, b, :2]  # use only x, y dimensions

        # Compute pairwise distances
        dists = torch.norm(points[1:] - points[:-1], dim=1)
        loop_closure = torch.norm(points[0] - points[-1])
        total_length = dists.sum() + loop_closure

        tour_lengths.append(total_length)

    return torch.stack(tour_lengths)  # shape: (batch_size,)


if __name__ == '__main__':
    args = init_parser()    
    model_name = create_model_name(args)

    model_path = f'./models/reinforce/trained_models/{model_name}/'
    os.system(f"mkdir -p {model_path}")
    
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    # Initialize model and optimizer
    model_args = {'hidden_size': args.hidden_size,
                  'bidirectional': args.bidirectional,
                  'device': device,
                  'teacher_forcing_ratio': 0.0,
                  'max_decoded_length': args.max_decoded_length,
                  'bello_et_al' : True, 
                  'num_sols': 1}
    
    model = PtrNet.PointerNetwork(model_args).to(device)
    critic = PtrNet.Critic(model_args).to(device)

    coord_tensors, tour_tensors = tsp_examples()

    dataset = TSPDataset(
        coord_tensors=coord_tensors,
        tour_tensors=tour_tensors
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=tsp_collate_fn
    )
    
    entropy_weight_initial = 0.1
    max_grad_norm = 1000
    num_epochs = args.num_epochs

    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
    critic_pretraining_epochs = 20
    for epoch in range(1, critic_pretraining_epochs + 1):
        for batch in dataloader:
            coords, tours, lengths = batch
            coords, tours = coords.to(device), tours.to(device)

            with torch.no_grad():
                gt_rewards = -calculate_ground_truth_tour_lengths(coords, tours)

            pred_values = critic(coords, lengths)
            loss = F.mse_loss(pred_values, gt_rewards)

            critic_optimizer.zero_grad()
            loss.backward()
            critic_optimizer.step()

    optimizer = optim.Adam([{"params": model.parameters()}, {"params": critic.parameters()}], lr=1e-4)#, weight_decay=args.wd)
    
    lr_lambda = lambda epoch: 0.1 if epoch == 10 else 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    losses_epochs = []
    differences_epochs = []
    
    for epoch in range(1, num_epochs + 1):
        losses, tour_lengths_, differences = [], [], []
        for batch in dataloader:
            coords, tours, lengths = batch
            coords, tours = coords.to(device), tours.to(device)
            seq, seq_lens = coords, lengths
            
            pointer_indices, pointer_log_scores = model(seq, seq_lens)[0]
            tour_lengths = calculate_tour_lengths(coords, pointer_indices)
            ground_truth_lengths = calculate_ground_truth_tour_lengths(coords, tours)
            rewards = -tour_lengths

            critic_values = critic(seq, seq_lens)
            advantage = rewards - critic_values.detach()

            batch_size = seq.size(1)
            log_probs = torch.empty(batch_size, dtype=advantage.dtype, device=device)
            entropy = torch.empty(batch_size, dtype=advantage.dtype, device=device)

            for b_i in range(batch_size):
                pointers_i = pointer_indices[:, b_i]
                zero_indices = (pointers_i == 0).nonzero()
                if zero_indices.numel() > 0:
                    pointers_i = pointers_i[:zero_indices[0, 0].item()]
                pointers_i_y = pointers_i
                pointers_i_x = torch.arange(pointers_i.size(0), device=device)
                log_probs[b_i] = pointer_log_scores[pointers_i_x, pointers_i_y, b_i].sum()
                log_probs_i = pointer_log_scores[pointers_i_x, :, b_i]
                probs_i = torch.exp(log_probs_i)
                entropy[b_i] = -torch.sum(probs_i * log_probs_i)

            reinforce = -torch.mean(advantage * log_probs)
            critic_loss = torch.mean((rewards - critic_values) ** 2)
            entropy_weight = entropy_weight_initial * (0.99 ** epoch)
            loss = reinforce + critic_loss - entropy_weight * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, norm_type=2)
            optimizer.step()

            losses.append(loss.item())
            tour_lengths_ += [t.item() for t in tour_lengths]
            differences += [(t - g).item() for t, g in zip(tour_lengths, ground_truth_lengths)]
        scheduler.step()
        print(f"Epoch {epoch}, Loss: {torch.tensor(losses).mean():.4f}, Average Tour Length: {torch.tensor(tour_lengths_).mean():.4f}, Average Difference: {torch.tensor(differences).mean():.4f}")
        losses_epochs.append(torch.tensor(losses).mean().item())
        differences_epochs.append(torch.tensor(differences).mean().item())
        
    
    np.save("./losses.npy", np.array(losses_epochs, dtype=float))
    np.save("./differences.npy", np.array(differences_epochs, dtype=float))