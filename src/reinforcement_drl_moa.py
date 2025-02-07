import argparse
import pandas as pd
import os

import torch
import torch.optim                as optim
from   torch.utils.data           import DataLoader
from   torch.utils.tensorboard    import SummaryWriter

from tqdm       import tqdm
from data_types import VisSample, VisDataset, collate_fn
import numpy as np
import PtrNet

from demo import evaluate_polygon_visibility_numpy_wo_gt, CustomError

from losses import RewardTwoParams

def get_model_name_and_create_paths(args):
    0


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


def train_model(
        model: PtrNet.PointerNetwork,
        training_generator: DataLoader,
        validation_generator: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
        writer: SummaryWriter,
        num_epochs: int,
        use_critic: bool = False,
        entropy_weight: float = 0.01  # Entropy weight
    ):

    max_grad_norm = 2
    best_coverage = 0
    best_epoch = 0

    print(f"Using critic: {use_critic}")
    betas_list = torch.stack([
        torch.linspace(0, 1, args.drl_moa_steps),
        torch.linspace(1, 0, args.drl_moa_steps)
    ], dim=1)
    
    for betas in betas_list:
        print(betas)
    
    exit()

    for epoch_i in range(num_epochs):
        critic_exp_mvg_avg = torch.zeros(1).to(device)

        for batch_idx, (seq, seq_lens, positions) in enumerate(tqdm(training_generator, desc=f"Epoch {epoch_i+1}/{num_epochs}")):
            seq, positions = seq.to(device), positions.to(device)

            # Sample solution from model
            pointer_indices, pointer_log_scores = model(seq, seq_lens)[0]

            # Compute coverage and relative length
            coverages, relative_lengths, fine = batch_eval_coverage(seq.cpu(), seq_lens, pointer_indices)
            coverages, relative_lengths, fine = coverages.to(device), relative_lengths.to(device), fine.to(device)

            # Define rewards
            rewards = reward_function(coverages, relative_lengths)
            #rewards = cov_wt * (1 - coverages) + opt_wt * relative_lengths

            # Exponential moving average for baseline
            if batch_idx == 0:
                critic_exp_mvg_avg = rewards.mean()
            else:
                beta = 0.9
                critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * rewards.mean())

            # Compute advantage
            if use_critic:
                critic_values = critic(seq, seq_lens)
                advantage = (rewards - critic_values).detach()
            else:
                advantage = (rewards - critic_exp_mvg_avg).detach()

            # Compute log probabilities and entropy
            batch_size = seq.size(1)
            log_probs = torch.empty(batch_size, dtype=advantage.dtype, device=device)
            entropy = torch.empty(batch_size, dtype=advantage.dtype, device=device)

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
                log_probs[b_i] = pointer_log_scores[pointers_i_x, pointers_i_y, b_i].sum()

                # Compute entropy 
                probs = torch.exp(pointer_log_scores[pointers_i_x, pointers_i_y, b_i])  # Convert log-prob to prob
                entropy[b_i] = -torch.sum(probs * pointer_log_scores[pointers_i_x, pointers_i_y, b_i])

            # Compute loss with entropy regularization
            reinforce = torch.mean(advantage * log_probs)
            critic_loss = torch.mean(advantage ** 2)
            loss = reinforce + (critic_loss if use_critic else 0) - entropy_weight * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, norm_type=2)
            optimizer.step()

            critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

            if batch_idx % 10 == 0:
                avg_coverage = coverages.mean().item()
                avg_rel_length = relative_lengths.mean().item()
                avg_reward = rewards.mean().item()
                avg_entropy = entropy.mean().item()

                # Log statistics to TensorBoard
                writer.add_scalar('Train/Loss', loss.item(), epoch_i * len(training_generator) + batch_idx)
                writer.add_scalar('Train/Coverage', avg_coverage, epoch_i * len(training_generator) + batch_idx)
                writer.add_scalar('Train/Relative Length', avg_rel_length, epoch_i * len(training_generator) + batch_idx)
                writer.add_scalar('Train/Reward', avg_reward, epoch_i * len(training_generator) + batch_idx)
                writer.add_scalar('Train/Entropy', avg_entropy, epoch_i * len(training_generator) + batch_idx)

        # Evaluate the model on the validation set
        evaluate_model(model, validation_generator, device, writer, epoch_i, mode='Validation')
         # save checkpoint
        #torch.save(model, os.path.join(model_path, f'model_epoch_{epoch_i}.pt'))
        # Save the best model based on coverage
        if avg_coverage > best_coverage:
            best_coverage = avg_coverage
            best_epoch = epoch_i
            torch.save(model.state_dict(), os.path.join(model_path, 'best_model.pth'))
            
            if args.reward_function == "reward_learnable":
                torch.save(reward_function, os.path.join(alpha_path, 'best_model.pt'))
                
            print(f"New best model saved at epoch {epoch_i} with coverage {avg_coverage}")

    print(f"Best model found at epoch {best_epoch} with coverage {best_coverage}")


def evaluate_model(
        model: PtrNet.PointerNetwork, 
        data_loader : DataLoader, 
        device : torch.device, 
        writer : SummaryWriter, 
        epoch_i : int, 
        mode : str ='Validation'
    ):
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

            #rewards = cov_wt * (1 - coverages) + opt_wt * relative_lengths
            rewards = reward_function(coverages, relative_lengths)

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
    parser.add_argument('--drl-moa-steps', type=int, default=11, help='Number of DRL-MOA steps')
    parser.add_argument('--train-data', type=str, default='train', help='path to training samples')
    parser.add_argument('--valid-data', type=str, default='dev', help='path to validation samples')
    parser.add_argument('--test-data', type=str, default='test', help='path to test samples')
    parser.add_argument('--sample-size', type=int, default=100, help='Number of samples to use')
    parser.add_argument('--use-critic', action='store_true', help='Use Bello critic instead of exp_mvg_avg')
    parser.add_argument('--comment', type=str, default='', help='Additional comment for the model name')
    parser.add_argument('--cov-wt', type=float, default=0.6, help='Coverage weight')
    #parser.add_argument('--reward-function', type=str, default='reward_manual', help='Set a defined reward function (e.g. manual hyperparams reward (reward_manual) or learnable hyperparams reward (reward_learnable))')
    args = parser.parse_args()
    
    """
    if args.reward_function == "reward_manual":
        reward_function = Reward(args.cov_wt)
    elif args.reward_function == "reward_learnable":
        reward_function = RewardLH()
    else:
        raise ValueError("No reward function is set")
    """
    
    dataset_dir = "/mnt/nvme0n1/dseverdi/MLAG/dataset/AG/development"
    #dataset_dir = "dataset/development"

    train_path = os.path.join(dataset_dir,f'{args.train_data}')
    valid_path = os.path.join(dataset_dir,f'{args.valid_data}')
    test_path = os.path.join(dataset_dir,f'{args.test_data}')
    
    model_name = f'ne-{args.num_epochs}_bs-{args.batch_size}_hs-{args.hidden_size}_hv-{args.hidden_v}_wd-{args.wd}_lr-{args.lr}_ss-{args.sample_size}_wt_cov-{args.cov_wt}'
    if args.use_critic:
        model_name += '_critic'
    if args.bidirectional:
        model_name += '_bidirectional'
    
    if args.normalize:
        model_name += '_normalized'

    if args.comment:
        model_name += f'_{args.comment}'

    if args.drl_moa_steps:
        model_name += f'_{args.drl_moa_steps}'

    model_path = f'./trained_models/{model_name}/'
    alpha_path = f'./trained_models/alpha/{model_name}/'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    if not os.path.exists(alpha_path):
        os.makedirs(alpha_path)    
    
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
    critic = PtrNet.Critic(model_args).to(device)
    #optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)    
    optimizer = optim.Adam([ {"params": model.parameters()}, { "params": critic.parameters() }], lr=1e-3)
    #optimizer_critic = optim.Adam(critic.parameters(), lr=1e-4)
    


    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f'./logs/{model_name}')

    # Train the model
    train_model(model, training_generator, validation_generator, optimizer, device, writer, args.num_epochs, args.use_critic)

    writer.close()

    # Evaluate the model on the test set
    test_set = VisDataset()
    test_paths = [f"{test_path}/{filename}" for filename in df["test"].tolist() if filename is not None]
    test_set.extend([VisDataset(VisSample.read_samples(path=path, normalize=args.normalize)[:sample_size]) for path in test_paths])
    test_generator = DataLoader(test_set, **dataloader_params, collate_fn=collate_fn)

    evaluate_model(model, test_generator, device, writer, args.num_epochs, mode='Test')