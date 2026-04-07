import torch

if __name__ == "__main__":
    a = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
    mask = torch.tensor([0, 0, 1, 0, 1], dtype=torch.bool)
    a.masked_fill_(mask, float("-inf"))
    
    
    print(a)
    print(torch.isinf(a))