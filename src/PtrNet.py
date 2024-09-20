import torch
import torch.nn as nn
from   torch.utils.data import Dataset
import pandas as pd
import numpy as np

from skgeom import Point2 # type: ignore
from skgeom import Polygon, PolygonSet # type: ignore
from skgeom import arrangement, RotationalSweepVisibility, TriangularExpansionVisibility # type: ignore
from typing import Optional, Type # type: ignore

from data_types import VisSequence
from beam_search import BeamNode

def create_mask(seq_lens, device):
    mask = torch.zeros(max(seq_lens), len(seq_lens), device=device)
    for i, seq_len in enumerate(seq_lens):
        mask[:seq_len, i] = 1
    return mask.bool()

def masked_log_softmax(vector, mask, dim):
    return nn.functional.log_softmax(vector + (mask.float() + 1e-45).log(), dim=dim)

def masked_max(vector, mask, dim, keepdim=False, min_val=-1e7):
    one_minus_mask = ~mask
    replaced_vector = vector.masked_fill(one_minus_mask, min_val)
    max_value, max_index = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value, max_index

class Encoder(nn.Module):
    def __init__(self, hidden_size, bidirectional):
        super().__init__()
        self.lstm = nn.LSTM(input_size=VisSequence.input_size, hidden_size=hidden_size, bidirectional=bidirectional)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

    def forward(self, seq, seq_lens):
        """ Forward call for PtrNet encoder
        :param seq: torch.tensor of dimension (seq_len, batch_size, VisSequence.input_size)
        :param seq_lens: list of length batch_size, contains lengths of unpadded sequences
        :returns: tuple (h_n, c_n), concatenated fwd/bkd hidden and cell states, and tensor *output* of all hidden states
        """                

        # Pack sequences and feed into LSTM
        seq_packed = nn.utils.rnn.pack_padded_sequence(seq, seq_lens, enforce_sorted=False)
        output, (h_n, c_n) = self.lstm(seq_packed)

        output, _ = nn.utils.rnn.pad_packed_sequence(output, padding_value=-10000)
        if self.bidirectional:
            h_n = torch.cat((h_n[0], h_n[1]), dim=1)
            c_n = torch.cat((c_n[0], c_n[1]), dim=1)
        else:
            h_n = h_n.squeeze(0)
            c_n = c_n.squeeze(0)
            
        return (h_n, c_n), output

    
class Pointer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.W_e = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_d = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_size = hidden_size
        self.tanh = nn.Tanh()

    def forward(self, decoder_state, encoder_states, mask):
        """ Forward call for Pointer
        :param decoder_state: last state from decoder
        :param encoder_states: tensor of all encoder states
        :param mask: bool tensor for softmax masking
        :returns: log-softmax scores for every input sequence
        """
        encoder_transform = self.W_e(encoder_states)
        decoder_transform = self.W_d(decoder_state)
        u = self.v(self.tanh(encoder_transform + decoder_transform)).squeeze(-1)
        return masked_log_softmax(u, mask, dim=0)


class PointerNetwork(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.bidirectional = model_args['bidirectional']
        self.encoder_hidden_size = model_args['hidden_size']
        self.decoder_hidden_size = 2 * self.encoder_hidden_size if self.bidirectional else self.encoder_hidden_size 
        self.teacher_forcing_ratio = model_args['teacher_forcing_ratio']
        self.max_decoded_length = model_args['max_decoded_length']
        self.num_sols = model_args['num_sols']

        self.encoders = nn.ModuleList([Encoder(self.encoder_hidden_size, self.bidirectional) for _ in range(self.num_sols)])
        self.decoder_rnns = nn.ModuleList([nn.LSTMCell(self.decoder_hidden_size, self.decoder_hidden_size) for _ in range(self.num_sols)])
        self.pointers = nn.ModuleList([Pointer(self.decoder_hidden_size) for _ in range(self.num_sols)])

    def forward(self, seq, seq_lens, target=None, beam_width=None, alpha=None, beta=None):
        sols = []
        for i in range(self.num_sols):
            if beam_width is None:
                sols.append(self.greedy_decode(seq, seq_lens, target, i))
            else:
                sols.append(self.beam_search_decode(seq, seq_lens, target, i, beam_width, alpha, beta))
        return sols

    def greedy_decode(self, seq, seq_lens, target, model_idx):
        
        
        
        batch_size = seq.shape[1]

        (h, c), encoder_states = self.encoders[model_idx](seq, seq_lens)
        
        print(seq.size())
        print(encoder_states.size())
        print(h.size())
        print(c.size())
        exit()
        x = torch.zeros_like(h)

        pointer_log_scores = []
        pointer_indices = []

        use_teacher_forcing = torch.rand(1) <= self.teacher_forcing_ratio
        mask = create_mask(seq_lens, device=seq.device)

        for i in range(target.shape[0] if target is not None else self.max_decoded_length):
            h, c = self.decoder_rnns[model_idx](x, (h, c))

            pointer_log_score = self.pointers[model_idx](h, encoder_states, mask) # (seq_len, batch_size)
            pointer_log_scores.append(pointer_log_score)

            _, pointer_index = masked_max(pointer_log_score, mask, dim=0) # (batch_size)
            pointer_indices.append(pointer_index)

            if use_teacher_forcing and target is not None:
                idx = torch.where(target[i, :, model_idx] == -1, 0, target[i, :, model_idx]).reshape(1, -1, 1).expand(1, -1, self.decoder_hidden_size)
            else:
                idx = pointer_index.reshape(1, -1, 1).expand(1, -1, self.decoder_hidden_size)
            x = torch.gather(encoder_states, dim=0, index=idx).squeeze(0)

        return torch.stack(pointer_indices, dim=0), torch.stack(pointer_log_scores, dim=0)

    def beam_search_decode(self, seq, seq_lens, target, model_idx, beam_width, alpha, beta):
        batch_size = seq.shape[1]
        mask = create_mask(seq_lens, device=seq.device)
        best_nodes = []

        (h_encoder, c_encoder), encoder_states = self.encoders[model_idx](seq, seq_lens)

        for batch_idx in range(batch_size):
            x = torch.zeros_like(h_encoder[batch_idx]).unsqueeze(0)
            h = h_encoder[batch_idx].unsqueeze(0)
            c = c_encoder[batch_idx].unsqueeze(0)
            
            node = BeamNode(1e9, x, h, c, None, None, torch.empty(0, device=seq.device))
            nodes = [node]
            finished_nodes = []
            
            submask = mask[:, batch_idx]

            for i in range(self.max_decoded_length):
                next_nodes = []
                for node in nodes:
                    x, h, c = node.state
                    h, c = self.decoder_rnns[model_idx](x, (h, c))
                    pointer_log_score = self.pointers[model_idx](h, encoder_states[:, batch_idx, :], submask)[:seq_lens[batch_idx]]
                    pointer_log_score = torch.topk(pointer_log_score, k=beam_width)
                    for j in range(beam_width):
                        pointer_index = pointer_log_score.indices[j]
                        log_score = pointer_log_score.values[j]
                        
                        x = encoder_states[pointer_index, batch_idx, :].unsqueeze(0)
                        next_log_score = log_score.reshape(-1) if len(node.log_ps) == 0 else torch.cat((node.log_ps, log_score.reshape(1)))
                        next_node = BeamNode(None, x, h, c, node, pointer_index, next_log_score)
                        next_node.score = -next_node.eval(alpha)
                        
                        if pointer_index == 0 and i == 0:
                            continue
                        if pointer_index == 0:
                            finished_nodes.append(next_node)
                        if pointer_index > 0:
                            next_nodes.append(next_node)
                nodes = sorted(next_nodes, key=lambda x : x.score)[:beam_width]
            best_nodes.append(sorted(finished_nodes, key=lambda x : x.score)[0])
        return best_nodes
