import torch
from torch.nn import LSTMCell, LSTM
import numpy as np

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
    print(output.shape)
    print(hn.shape)
    print(cn.shape)
    
    
if __name__ == "__main__":
    a = torch.tensor([1, 7, 8, 0, -1, 11, 13, 15, 23], dtype=torch.float32).numpy()
    n = np.random.randint(1, 5)
    subset = np.random.choice(a, n)
    print(subset)