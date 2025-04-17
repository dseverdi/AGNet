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
    parser.add_argument('--batch-size', type=int, default=1024, help='training batch size')
    parser.add_argument('--normalize', action='store_true', default=True, help='normalize inputs to unit square')
    parser.add_argument('--bidirectional', action='store_true', default=True, help='Bidirectional encoder LSTM')
    parser.add_argument('--hidden-size', type=int, default=256, help='LSTM hidden dimension size')
    parser.add_argument('--hidden-v', type=int, default=256, help='Attention layer hidden size')
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

        if len(tour_trimmed) < 2:
            tour_lengths_.append(torch.tensor(0.0, dtype=coords.dtype, device=coords.device))
            continue

        points = coords[tour_trimmed, b, :2]


        segment_dists = torch.norm(points[1:] - points[:-1], dim=1)
        segment_dists
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
    
    entropy_weight_initial = 0.05
    max_grad_norm = 100
    num_epochs = args.num_epochs

    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
    critic_pretraining_epochs = 10
    for epoch in range(1, critic_pretraining_epochs + 1):
        for batch in dataloader:
            coords, tours, lengths = batch
            coords, tours = coords.to(device), tours.to(device)

            with torch.no_grad():
                gt_rewards = calculate_ground_truth_tour_lengths(coords, tours)

            pred_values = critic(coords, lengths)
            loss = F.mse_loss(pred_values, gt_rewards)

            critic_optimizer.zero_grad()
            loss.backward()
            critic_optimizer.step()

    optimizer = optim.Adam([{"params": model.parameters()}, {"params": critic.parameters(), "lr": 1e-5}], lr=1e-4)#, weight_decay=args.wd)
    
    lr_lambda = lambda epoch: 0.1 if epoch == 10 else 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    losses_epochs = []
    differences_epochs = []
    model.train()
    critic.train()
    for epoch in range(1, num_epochs + 1):
        losses, tour_lengths_, differences = [], [], []
        for batch in dataloader:
            coords, tours, lengths = batch
            coords, tours = coords.to(device), tours.to(device)
            seq, seq_lens = coords, lengths
            
            pointer_indices, pointer_log_scores = model(seq, seq_lens)[0]
            #pointer_indices = tours.clone()
            tour_lengths = calculate_tour_lengths(coords, pointer_indices)
            ground_truth_lengths = calculate_ground_truth_tour_lengths(coords, tours)
            
            baseline = critic(seq, seq_lens)
    
            advantage = tour_lengths - baseline.detach()
            
            batch_size = seq.size(1)

            T, B = pointer_indices.shape

            # Gather log probs from [T, B, N] using [T, B] indices
            selected_log_probs = pointer_log_scores.gather(dim=2, index=pointer_indices.unsqueeze(-1)).squeeze(-1)  # [T, B]

            # Create mask to ignore entries after EOS
            is_eos = pointer_indices == 0  # EOS token index is 0
            eos_seen = is_eos.float().cumsum(dim=0) > 1
            valid_mask = ~eos_seen

            # Mask and sum
            log_probs = (selected_log_probs * valid_mask).sum(dim=0)  # [B]
            
            print("log_probs: mean =", log_probs.mean().item(), "std =", log_probs.std().item())
            print("advantage: mean =", advantage.mean().item(), "std =", advantage.std().item())

            
            actor_loss = -torch.mean(advantage * log_probs)  # p. 5, Algorithm 1, line 8
            
            any_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"{name}: {param.grad.norm().item()}")
                    any_grad = True
            if not any_grad:
                print("❌ No actor gradients flowing!")            
            
            critic_loss = F.mse_loss(baseline, tour_lengths)
            
            #for name, param in model.named_parameters():
            #    if param.grad is not None:
            #        print(f"{name}: grad norm = {param.grad.norm().item()}")
            #exit()
            optimizer.zero_grad()
            critic_loss.backward()
            actor_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'] + optimizer.param_groups[1]['params'], max_grad_norm)

            optimizer.step()

            losses.append(actor_loss.item())
            tour_lengths_ += [t.item() for t in tour_lengths]
            differences += [(t - g).item() for t, g in zip(tour_lengths, ground_truth_lengths)]
        scheduler.step()
        print(f"Epoch {epoch}, Loss: {torch.tensor(losses).mean():.4f}, Average Tour Length: {torch.tensor(tour_lengths_).mean():.4f}, Average Difference: {torch.tensor(differences).mean():.4f}")
        losses_epochs.append(torch.tensor(losses).mean().item())
        differences_epochs.append(torch.tensor(differences).mean().item())
        
    
    #np.save("./losses.npy", np.array(losses_epochs, dtype=float))
    #np.save("./differences.npy", np.array(differences_epochs, dtype=float))
