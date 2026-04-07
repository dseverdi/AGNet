import torch
import torch.nn

class Reward(torch.nn.Module):
    def __init__(self, cov_wt, exp_factor=None):
        super().__init__()
        self.cov_wt = cov_wt
        self.opt_wt = 1 - cov_wt
        self.exp_factor = exp_factor  # Exponential growth factor
    
    def forward(self, coverages, relative_lengths):
        if self.exp_factor:
            rel = torch.pow(relative_lengths, self.exp_factor)
        else:
            rel = relative_lengths
        # Higher = better: full coverage + few guards → high reward
        return self.cov_wt * coverages - self.opt_wt * rel

class RewardTwoParams(torch.nn.Module):
    def __init__(self, beta_1, beta_2):
        super().__init__()
        self.beta_1 = beta_1
        self.beta_2 = beta_2
    
    def forward(self, coverages, relative_lengths):
        # Higher = better: reward coverage, penalize guards
        return self.beta_1 * coverages - self.beta_2 * relative_lengths
    
# LH - learnable hyperparameter
class RewardLH(torch.nn.Module):
    def __init__(self, alpha_initial = 0.0):
        super().__init__()
        self.alpha = torch.tensor(alpha_initial, dtype=torch.float32)
        
    def forward(self, coverages, relative_lengths):
        beta = torch.special.expit(self.alpha) * 0.9 + 0.1
        # Higher = better: reward coverage, penalize guards
        return beta * coverages - (1 - beta) * relative_lengths
    
    
class RewardHardCoverage(torch.nn.Module):
    """Reward = -n_guards if coverage == 100%, else large penalty.

    Only rewards the model for reducing guards once full coverage is achieved.
    Normalized by n_vertices so reward is in [-1, 0] range.
    """
    def __init__(self, coverage_threshold: float = 0.999):
        super().__init__()
        self.coverage_threshold = coverage_threshold

    def forward(self, coverages, relative_lengths):
        full_cov = (coverages >= self.coverage_threshold).float()
        # Higher = better: full coverage → -guard_ratio (closer to 0 = fewer guards = better)
        # No coverage → -2.0 (heavy penalty)
        return full_cov * (-relative_lengths) + (1 - full_cov) * (-2.0)


class RewardOptimalRatio(torch.nn.Module):
    """Reward based on ratio to optimal guard count (requires GT n_guards).

    reward = coverage_bonus + optimal_n_guards / predicted_n_guards
    Ranges roughly in [0, 1] when coverage is full and guard count matches optimal.
    """
    def __init__(self, coverage_threshold: float = 0.999, cov_penalty: float = 5.0):
        super().__init__()
        self.coverage_threshold = coverage_threshold
        self.cov_penalty = cov_penalty

    def forward(self, coverages, relative_lengths, n_guards_gt, n_guards_pred):
        """
        Args:
            coverages: (batch,) coverage fraction per room
            relative_lengths: (batch,) not used directly
            n_guards_gt: (batch,) optimal guard count from ILP
            n_guards_pred: (batch,) number of guards the model selected
        """
        # Ratio: optimal / predicted, clamped to [0, 1]
        ratio = (n_guards_gt.float() / n_guards_pred.float().clamp(min=1)).clamp(max=1.0)
        # Coverage bonus: 1 if full, 0 otherwise
        full_cov = (coverages >= self.coverage_threshold).float()
        return full_cov * ratio - (1 - full_cov) * self.cov_penalty


class RewardExpGuardPenalty(torch.nn.Module):
    """Exponential penalty on guard count to strongly discourage over-selection.

    reward = coverage - exp(guard_ratio * scale)
    The exponential makes using many guards very costly.
    """
    def __init__(self, scale: float = 3.0, coverage_weight: float = 2.0):
        super().__init__()
        self.scale = scale
        self.coverage_weight = coverage_weight

    def forward(self, coverages, relative_lengths):
        return self.coverage_weight * coverages - torch.exp(self.scale * relative_lengths)


class RewardOptimalThreshold(torch.nn.Module):
    """Coverage-first reward that only penalizes guards *above* optimal count.

    When n_pred <= n_optimal: reward = coverage   (pure coverage maximisation)
    When n_pred >  n_optimal: reward = coverage - excess_penalty * (n_excess / n_optimal)

    This lets the model improve placement quality without being punished for
    already being at or below the optimal guard count.
    """
    def __init__(self, excess_penalty: float = 2.0):
        super().__init__()
        self.excess_penalty = excess_penalty

    def forward(self, coverages, relative_lengths, n_guards_gt, n_guards_pred):
        excess = (n_guards_pred.float() - n_guards_gt.float()).clamp(min=0.0)
        excess_ratio = excess / n_guards_gt.float().clamp(min=1.0)
        return coverages - self.excess_penalty * excess_ratio


if __name__ == "__main__":
    r = RewardLH(alpha_initial=0.77)
    torch.save(r, "./temp.pt")
    r2 = torch.load("./temp.pt")
    print(r2.alpha)

