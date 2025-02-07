import torch

if __name__ == "__main__":
    drl_moa_steps = 11
    betas_list = torch.stack([
        torch.linspace(0, 1, drl_moa_steps),
        torch.linspace(1, 0, drl_moa_steps)
    ], dim=1)
    
    
    for betas in betas_list:
        print(betas)