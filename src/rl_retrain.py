import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_types import VisSample, VisDataset, collate_fn
from PtrNet import PointerNetwork
from torch.utils.tensorboard import SummaryWriter

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    return model

def train_model(model, train_loader, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for seq, seq_lens, _, _ in train_loader:
            seq = seq.to(device)
            optimizer.zero_grad()
            output = model(seq, seq_lens)
            loss = -output[0].sum()  # Dummy loss for illustration
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs} completed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Retrain Pointer Network model.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model to be retrained')
    parser.add_argument('--train-data', type=str, required=True, help='Path to training data')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs for retraining')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(args.model_path, device)
    model.to(device)

    # Prepare data
    train_samples = VisSample.read_samples(path=args.train_data, sol_sample=1)
    train_dataset = VisDataset(train_samples)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

    # Prepare optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train model
    train_model(model, train_loader, optimizer, device, args.num_epochs)

    # Save retrained model
    model_name = os.path.basename(args.model_path)
    save_path = os.path.join('./models/retrained_models/reinforce', model_name)
    torch.save(model, save_path)
    print(f"Retraining completed and model saved as '{save_path}'.")
