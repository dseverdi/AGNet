import torch
import torch.nn

class Reward(torch.nn.Module):
    def __init__(self, cov_wt):
        super().__init__()
        self.cov_wt = cov_wt
        self.opt_wt = 1 - cov_wt
    
    def forward(self, coverages, relative_lengths):
        return self.cov_wt * (1 - coverages) + self.opt_wt * relative_lengths
    
# LH - learnable hyperparameter
class RewardLH(torch.nn.Module):
    def __init__(self, alpha_initial = 0.0):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.tensor(alpha_initial, dtype=torch.float32))
        
    def forward(self, coverages, relative_lengths):
        beta = torch.special.expit(self.alpha) * 0.9 + 0.1
        return beta * coverages + (1 - beta) * relative_lengths