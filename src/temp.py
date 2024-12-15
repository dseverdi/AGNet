import torch
from torch.nn import LSTMCell, LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import random

def lstm_cell_example():
    sequence_length = 10
    batch_size = 4
    input_size = 3
    hidden_size = 5
    
    a = torch.randn(sequence_length, batch_size, input_size)
    
    lstm = LSTMCell(input_size, hidden_size)
    
    output = []
    hx, cx = torch.randn(batch_size, hidden_size), torch.randn(batch_size, hidden_size)
    for a_t in a:
        hx, cx = lstm(a_t, (hx, cx))
        output.append(hx)
    output = torch.stack(output)
    print(output.shape)
    
def lstm_example():
    sequence_length = 10
    batch_size = 4
    input_size = 3
    hidden_size = 5
    
    n_layers = 1
    
    a = torch.randn(sequence_length, batch_size, input_size)
    lstm = LSTM(input_size, hidden_size, n_layers)
    h0 = torch.randn(n_layers, batch_size, hidden_size)
    c0 = torch.randn(n_layers, batch_size, hidden_size)
    
    output, (hn, cn) = lstm(a, (h0, c0))
    
    print(a.shape)
    print(output.shape)
    print(hn.shape)
    print(cn.shape)
    
def padded_sequence_example():
    sequence_length = 3
    batch_size = 4
    input_size = 5
    
    padding_from = torch.randint(sequence_length // 2, sequence_length, (batch_size, ))
    
    paddings = []
    for i in range(batch_size):
        for j in range(padding_from[i], sequence_length):
            paddings.append([i, j])
            
    paddings = torch.tensor(paddings, dtype=torch.long)
    
    
    a = torch.rand((sequence_length, batch_size, input_size))
    a.transpose_(0, 1)
    a[paddings[:, 0], paddings[:, 1]] = 0
    a.transpose_(0, 1)
    
    ps = pack_padded_sequence(a, padding_from, enforce_sorted=False)

    a_, lens_unpacked = pad_packed_sequence(ps)
    
    print(a)
    print(a_)
    
if __name__ == "__main__":
    
    
    pointer_indices = torch.tensor([
        [1, 5, 0, 7, 0],
        [2, 5, 4, 3, 0],
        [7, 1, 1, 1, 1]
    ])
    
    pointer_indices_list = []
    for p in pointer_indices:
        zero_indices = (p == 0).nonzero()
        has_zero = zero_indices.size(0) > 0
        if has_zero:
            first_zero_index = zero_indices[0, 0].item()
            p = p[:first_zero_index]
        p -= 1
        pointer_indices_list.append(p)
    
    
    exit()
    
    # pointer_indices.size(): out_seq_len x batch_size        
    # pointer_log_scores.size(): out_seq_len x in_seq_len x batch_size
    batch_size = 2
    out_seq_len = 24
    in_seq_len = 151
    
    pointer_indices = torch.empty((out_seq_len, batch_size), dtype=torch.long)
    pointer_indices[:, 0] = torch.from_numpy(np.random.choice(range(1, in_seq_len), out_seq_len))
    pointer_indices[:, 1] = torch.from_numpy(np.random.choice(range(1, in_seq_len), out_seq_len))

    for i in range(batch_size):
        put_eos_earlier_prob = 0.5 
        put_eos_earlier = random.random() < put_eos_earlier_prob
        if put_eos_earlier:
            pointer_indices[random.randint(1, out_seq_len), i] = 0
    
    print(pointer_indices.transpose(0, 1))
            