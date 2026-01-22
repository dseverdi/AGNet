import sys
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from IPython.display import clear_output
from tqdm import tqdm

from utils import USE_CUDA

class TrainModel:
    def __init__(self, model, train_dataset, val_dataset, batch_size=128, threshold=None, max_grad_norm=1.0, beta=0.9, lr=1e-3, decay_steps=5000, decay_rate=0.96):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset   = val_dataset
        self.batch_size = batch_size
        self.threshold = threshold
        self.beta = beta  # Store beta as an instance variable        
        self.lr = lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
        self.val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

        self.actor_optim   = optim.Adam(model.actor.parameters(), lr=lr)
        self.max_grad_norm = max_grad_norm
        
        self.train_tour = []
        self.val_tour   = []
        
        self.epochs = 0
        self.global_step = 0

    def adjust_learning_rate(self):
        """Decay learning rate every decay_steps by decay_rate."""
        lr = self.lr * (self.decay_rate ** (self.global_step // self.decay_steps))
        for param_group in self.actor_optim.param_groups:
            param_group['lr'] = lr

    def train_and_validate(self, n_epochs=None, total_epochs=None):
        if n_epochs is None:
            n_epochs = self.n_epochs
        if total_epochs is None:
            total_epochs = n_epochs
        critic_exp_mvg_avg = torch.zeros(1)
        if USE_CUDA: 
            critic_exp_mvg_avg = critic_exp_mvg_avg.cuda()

        for epoch in range(n_epochs):
            for batch_id, sample_batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {self.epochs+1}/{total_epochs}", disable=False)):
                self.model.train()
                self.global_step += 1
                self.adjust_learning_rate()

                inputs = Variable(sample_batch)
                if USE_CUDA:
                    inputs = inputs.cuda()

                R, probs, actions, actions_idxs = self.model(inputs)

                if batch_id == 0:
                    critic_exp_mvg_avg = R.mean()
                else:
                    critic_exp_mvg_avg = (critic_exp_mvg_avg * self.beta) + ((1. - self.beta) * R.mean())

                advantage = R - critic_exp_mvg_avg

                logprobs = 0
                for prob in probs: 
                    logprob = torch.log(prob)
                    logprobs += logprob
                logprobs[logprobs < -1000] = 0.  

                reinforce = advantage * logprobs
                actor_loss = reinforce.mean()

                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.actor.parameters(), 1.0, norm_type=2)  # Clip to 1.0

                self.actor_optim.step()

                critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

                self.train_tour.append(R.mean().item())

                if batch_id % 10 == 0:
                    self.plot(self.epochs)

                if batch_id % 100 == 0:    
                    self.model.eval()
                    for val_batch in self.val_loader:
                        inputs = Variable(val_batch)
                        if USE_CUDA:
                            inputs = inputs.cuda()
                        R, probs, actions, actions_idxs = self.model(inputs)
                        self.val_tour.append(R.mean().item())

            if self.threshold and self.train_tour[-1] < self.threshold:
                print("EARLY STOPPAGE!")
                break
                
            self.epochs += 1
                
    def plot(self, epoch):
        pass
