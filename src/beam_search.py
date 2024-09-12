import torch
from typing import Optional, Type

class BeamNode():
    def __init__(self, score, x, h, c, prev, idx, log_ps):
        self.score = score
        self.state = x, h, c
        self.prev = prev
        self.idx = idx
        self.log_ps = log_ps
    
    def eval(self, alpha):
        return sum(self.log_ps).item() / (len(self.log_ps)**alpha)

    def to_seq(self):
        seq = []
        node = self
        while node.prev != None:
            seq.append(node.idx)
            node = node.prev
        seq.reverse()
        return torch.tensor(seq)
