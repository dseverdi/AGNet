import torch
from losses import RewardTwoParams


if __name__ == "__main__":
    epoch_avg_coverages = [0.1, 0.05, 0.005, 0.75, 0.99]
    x = torch.tensor(epoch_avg_coverages).mean()
    print(x)