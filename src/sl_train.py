import os
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from data_types import VisSample, VisDataset, collate_fn
import PtrNet

dataset_dir = '/home/dseverdi/Radno/MLTG/dataset/TGPIL/my_walk'
# dataset_dir = '/home/dseverdi/Radno/MLTG/dataset/TGPIL/my_walk_discretized'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pointer network for Terrain guarding predictions.')
    parser.add_argument('--batch-size', type=int, default=128, help='training batch size')
    parser.add_argument('--normalize', action='store_true', help='normalize inputs to unit square')
    parser.add_argument('--bidirectional', action='store_true', help='Bidirectional encoder LSTM')
    parser.add_argument('--hidden-size', type=int, default=256, help='LSTM hidden dimension size')
    parser.add_argument('--hidden-v', type=int, default=256, help='Attention layer hidden size')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay for Adam optimizer')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for Adam optimizer')
    parser.add_argument('--max-decoded-length', type=int, default=20, help='Maximum allowed sequence length for decoder')
    parser.add_argument('--num-epochs', type=int, default=30, help='Number of epochs for training')
    parser.add_argument('--log-interval', type=int, default=200, help='Print epoch state every log-interval interval mini batches')
    parser.add_argument('--train-data', type=str, default='train', help='path to training samples')
    parser.add_argument('--valid-data', type=str, default='dev', help='path to validation samples')
    args = parser.parse_args()

    train_path = os.path.join(dataset_dir,f'{args.train_data}')
    valid_path = os.path.join(dataset_dir,f'{args.valid_data}')
    
    model_name = f'ne-{args.num_epochs}_bs-{args.batch_size}_hs-{args.hidden_size}_hv-{args.hidden_v}_wd-{args.wd}_lr-{args.lr}'
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
                         'shuffle': True,
                         'num_workers' : 6}    

    # training set
    sizes = ['5','10','15','20','30','40','50']
    sample_size = 10000
    training_set = VisDataset()
    paths = [os.path.join(train_path,size) for size in sizes] if sizes else [os.path.join(root,dir) for root,dirs,files in os.walk(train_path) for dir in dirs]
    print(paths)
    exit()
    training_set.extend([VisDataset(VisSample.read_samples(path=path,normalize = args.normalize)[:sample_size]) for path in paths])     
    training_generator = DataLoader(training_set, **dataloader_params, collate_fn=collate_fn)

    
    # validation set
    validation_set = VisDataset()
    paths = [os.path.join(valid_path,size) for size in sizes] if sizes else [os.path.join(root,dir) for root,dirs,files in os.walk(valid_path) for dir in dirs]
    validation_set.extend([VisDataset(VisSample.read_samples(path=path, normalize = args.normalize)[:sample_size]) for path in paths])
    validation_generator = DataLoader(validation_set, **dataloader_params, collate_fn=collate_fn)

    encoder_args = {'hidden_size': args.hidden_size,
                    'bidirectional': args.bidirectional,
                    'device': device}

    decoder_args = {'hidden_size': args.hidden_size if not args.bidirectional else 2 * args.hidden_size,
                    'hidden_v': args.hidden_v,
                    'max_length': args.max_decoded_length,
                    'device': device}
    
    model = PtrNet.PointerNetwork(encoder_args, decoder_args).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(1, args.num_epochs + 1):
        print(f'epoch {epoch}')

        # TRAINING
        model.train()
        epoch_loss = 0
        for batch_idx, (seq, seq_lens, positions) in tqdm(enumerate(training_generator)):
            #pdb.set_trace()
            seq, positions = seq.to(device), positions.to(device)     
            output = model(seq, seq_lens, positions)
            optimizer.zero_grad()
            for loss in output[1]: # output[1] is list of losses
                epoch_loss += loss.item()
                loss.backward(retain_graph=True)
            optimizer.step()
            if batch_idx and batch_idx % args.log_interval == 0:
                batch_total = len(training_set) // args.batch_size
                loss = epoch_loss / (batch_idx * args.batch_size)
                print(f'Epoch {epoch} | {batch_idx} / {batch_total} batches | cur loss {loss}')
        print(f' * Training epoch loss: {(epoch_loss / len(training_set)):8.3f}')

        # VALIDATION
        model.eval()
        validation_loss = 0
        for batch_idx, (seq, seq_lens, positions) in enumerate(validation_generator):
            seq, positions = seq.to(device), positions.to(device)
            output = model(seq, seq_lens, positions)
            for loss in output[1]:
                validation_loss += loss.item()
        print(f' * Validation epoch loss: {(validation_loss / len(validation_set)):8.3f}')
                
        torch.save(model, os.path.join(model_path, f'model_epoch_{epoch}.pt'))
