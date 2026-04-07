import random
import torch
import torch.nn as nn
import torch.nn.functional as F

PADDING = -1.0

def num_to_one_hot(x, num_classes, seq_lens, to_pad=False):
    oh = F.one_hot(x.long().transpose(0, 1), num_classes)
    
    if to_pad:
        for i in range(oh.size(1)):
            oh[seq_lens[i]:, i, :] = PADDING
                
    return oh

def one_hot_to_num(x, seq_lens):
    x_ = x.clone()
    if x_.dtype.is_floating_point:
        min_ = torch.finfo(x_.dtype).min
    else:
        min_ = torch.iinfo(x_.dtype).min
        
    for i in range(x_.size(1)):
        x_[seq_lens[i]:, i, :] = min_
        
    return x_.argmax(dim=-1).transpose(0, 1).float()

class Seq2Seq(nn.Module):
    
    def __init__(
            self, input_size, hidden_size, output_size, 
            num_layers=1, bidirectional=False, padding_value=0,
            categorical=False, batch_first=False,
            n_categories=None, attention_dim=None
        ):
        super().__init__()
    
        
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional, batch_first=False)    

        self.decoder_hidden_size = hidden_size * (2 if bidirectional else 1)

        self.decoder_cell = nn.LSTMCell(output_size, self.decoder_hidden_size)
        self.decoder_out = nn.Linear(self.decoder_hidden_size, output_size)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.padding_value = padding_value
        
        self.tf_probability = 1.0
        
        self.categorical = categorical
        self.n_categories = n_categories
        self.batch_first = batch_first
    
        
        if attention_dim is not None:
            self.attention_dim = attention_dim        
            self.decoder_out = nn.Linear(self.decoder_hidden_size * 2, output_size)
            self.v = nn.Linear(attention_dim, 1, bias=False)
            self.W_1 = nn.Linear(self.decoder_hidden_size, attention_dim)
            self.W_2 = nn.Linear(self.decoder_hidden_size, attention_dim)
            self.tanh = nn.Tanh()
        
        self.layernorm = nn.LayerNorm(self.decoder_hidden_size * 2)    
    

    def _prepare_h_and_c_for_decoder(self, h, c):
        num_layers = self.encoder.num_layers
        num_directions = 2 if self.encoder.bidirectional else 1        

        batch_size_ = h.size(1)
        hidden_size_ = h.size(2)

        # h.shape: (num_layers * num_directions, batch_size, hidden_size)

        h = h.view(num_layers, num_directions, batch_size_, hidden_size_)
        c = c.view(num_layers, num_directions, batch_size_, hidden_size_)

        # Take last layer
        h = h[-1]  # (num_directions, batch_size, hidden_size)
        c = c[-1]
        
        # (batch_size, num_directions, hidden_size)
        h = h.permute(1, 0, 2).reshape(batch_size_, -1)
        c = c.permute(1, 0, 2).reshape(batch_size_, -1) 
    
        return h, c

    def attention(self, out_enc_all, out_dec, seq_lens):
        
        seq_len, batch_size = out_enc_all.size(0), out_enc_all.size(1)

        # Apply linear layer to encoder outputs
        W_1_out = self.W_1(out_enc_all)  # [seq_len, batch, attn_dim]

        # Compute valid sequence length including BOS and EOS
        valid_len = seq_lens + 2  # original + BOS + EOS

        # Create mask for positions ≥ valid_len
        mask = torch.arange(seq_len, device=W_1_out.device).unsqueeze(1) >= valid_len.unsqueeze(0)  # [seq_len, batch]

        # Mask encoder projections
        W_1_out = W_1_out.masked_fill(mask.unsqueeze(2), 0)

        # Project decoder hidden state
        W_2_out = self.W_2(out_dec)  # [batch, attn_dim]

        # Compute energy scores and apply attention
        energy = self.tanh(W_1_out + W_2_out.unsqueeze(0))  # [seq_len, batch, attn_dim]
        scores = self.v(energy).squeeze(2)  # [seq_len, batch]

        # Mask out padding positions
        scores = scores.masked_fill(mask, float('-inf'))

        alpha = torch.softmax(scores, dim=0)
        context = torch.sum(alpha.unsqueeze(-1) * out_enc_all, dim=0)  # [batch, hidden]

        return context

    def forward(self, x, y, seq_lens):
        y = torch.nn.functional.one_hot(y, self.n_categories).float().to(x.device)
        
        if self.categorical:            
            x = torch.nn.functional.one_hot(x, self.n_categories)    
            
        if self.batch_first:
            x = x.transpose(0, 1)
            y = y.transpose(0, 1)            
        
        # encoder forward-propagation
        x_packed = nn.utils.rnn.pack_padded_sequence(x, seq_lens, enforce_sorted=False)
        
        out_enc_packed, (h, c) = self.encoder(x_packed)
        
        out_enc, out_enc_seq_lens = nn.utils.rnn.pad_packed_sequence(out_enc_packed, batch_first=False, padding_value=self.padding_value)
        
        h, c = self._prepare_h_and_c_for_decoder(h, c)
                
        out_dec_list = []
        for i in range(y.size(0)):
            is_first = i == 0
            use_tf = random.random() < self.tf_probability
            if is_first or use_tf:
                inp_dec_i = y[i]
            else:
                logits = self.decoder_out(out_dec_list[-1])
                # ChatGPT says:
                # Soft vs Hard Sampling
                # Instead of argmax, you can do sampling for a more stochastic decoder:
                hard_sampling = True
                probs = torch.softmax(logits, dim=-1)
                if hard_sampling:
                    idx = probs.argmax(dim=-1)
                else:
                    idx = torch.multinomial(probs, num_samples=1).squeeze(1)
            
                # Detach to prevent gradient through sampling
                #inp_dec_i = torch.nn.functional.one_hot(idx, self.n_categories)
                ##inp_dec_i = self.embedding(idx).detach()
                #inp_dec_i = torch.nn.functional.one_hot(idx, self.n_categories).float().to(x.device).detach()
                inp_dec_i = torch.nn.functional.gumbel_softmax(logits, tau=1.0, hard=False)  # use `hard=True` if you want a discrete forward pass


            h, c = self.decoder_cell(inp_dec_i.float(), (h, c))
            
            
            
            context = self.attention(out_enc, h, seq_lens)
            
            # Pointer Networks paper
            # p. 3, eq. (3) & the penultimate paragraph on this page
            # there it is denoted with: d and d_dash
            h_and_context = torch.cat((h, context), dim=1)
            
            #h_and_context = self.layernorm(h_and_context)
            out_dec_list.append(h_and_context)        
            #out_dec_list.append(h)
        out_dec = torch.stack(out_dec_list, dim=0)
        
        # Return raw logits (softmax will be applied by loss function externally)
        return self.decoder_out(out_dec)
    

class PointerNetwork(Seq2Seq):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Supervised-mode special tokens
        self.bos_token = nn.Parameter(torch.randn(1, 1, self.input_size))
        self.eos_token = nn.Parameter(torch.randn(1, 1, self.input_size))

        # RL-mode: learned stop token prepended to encoder input
        self.stop_token = nn.Parameter(torch.randn(1, 1, self.input_size))
        # Project encoder state → decoder cell input size for RL autoregressive feeding
        self.enc_to_dec = nn.Linear(self.decoder_hidden_size, self.output_size)

    def attention(self, W_1_out, out_enc_all, out_dec, seq_lens):
        seq_len, batch_size = out_enc_all.size(0), out_enc_all.size(1)

        mask = torch.arange(seq_len, device=W_1_out.device).unsqueeze(1) >= seq_lens.unsqueeze(0).to(W_1_out.device)

        W_2_out = self.W_2(out_dec)

        scores = self.v(self.tanh(W_1_out + W_2_out)).squeeze(2)
        scores = scores.masked_fill(mask, float('-inf'))

        return scores

    def forward(self, x, y, seq_lens):
        y = torch.nn.functional.one_hot(y, self.n_categories).float().to(x.device)
        
        if self.categorical:            
            x = torch.nn.functional.one_hot(x, self.n_categories)    
            
        if self.batch_first:
            x = x.transpose(0, 1)
            y = y.transpose(0, 1)            
            
        # encoder forward-propagation
        x_packed = nn.utils.rnn.pack_padded_sequence(x, seq_lens, enforce_sorted=False)
        
        out_enc_packed, (h, c) = self.encoder(x_packed)
        
        out_enc, out_enc_seq_lens = nn.utils.rnn.pad_packed_sequence(out_enc_packed, batch_first=False, padding_value=self.padding_value)
        
        h, c = self._prepare_h_and_c_for_decoder(h, c)

        # Precompute W_1 projection (encoder outputs don't change across decoder steps)
        W_1_out = self.W_1(out_enc)
                
        out_dec_list = []
        for i in range(y.size(0)):
            ### 
            # TODO: revisit this once a reinforcement learning is about
            # to be implemented. Fine for now.
            is_first = i == 0
            use_tf = random.random() < self.tf_probability
            if is_first or use_tf:
                inp_dec_i = y[i]
            else:
                logits = out_dec_list[-1]
                # ChatGPT says:
                # Soft vs Hard Sampling
                # Instead of argmax, you can do sampling for a more stochastic decoder:
                hard_sampling = True
                probs = torch.softmax(logits, dim=0)
                if hard_sampling:
                    idx = probs.argmax(dim=0)                    
                else:
                    idx = torch.multinomial(probs.T, num_samples=1).squeeze(1)
                                
                inp_dec_i = torch.nn.functional.one_hot(idx, self.n_categories).float().to(x.device)



            h, c = self.decoder_cell(inp_dec_i, (h, c))
            logits = self.attention(W_1_out, out_enc, h, seq_lens)
            out_dec_list.append(logits)        
                
        out_dec = torch.stack(out_dec_list, dim=0)
        out_dec = out_dec.permute(0, 2, 1)
        
        # Return raw logits (softmax will be applied by loss function externally)
        return out_dec

    def rl_forward(self, x, seq_lens, max_steps=None):
        """Autoregressive decoding with sampling for REINFORCE.

        A learned *stop token* is prepended at encoder position 0.
        When the model points to position 0 it signals "done".

        Args:
            x:        (max_seq_len, batch, input_size)  – vertex coordinates
            seq_lens: (batch,)                          – actual vertex counts
            max_steps: upper bound on decode steps (default: max seq_len)

        Returns:
            actions:   (batch, n_steps) – vertex indices (0-based), -1 = stop
            log_probs: (batch, n_steps) – per-step log-probability
            entropies: (batch, n_steps) – per-step entropy
        """
        device = x.device
        batch_size = x.size(1)
        max_steps = max_steps or seq_lens.max().item()

        # ── prepend stop token (position 0) ──────────────────────────
        stop = self.stop_token.expand(1, batch_size, -1)
        x_aug = torch.cat([stop, x], dim=0)           # (seq_len+1, B, D)
        seq_lens_aug = seq_lens + 1

        # ── encode ────────────────────────────────────────────────────
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x_aug, seq_lens_aug.cpu(), enforce_sorted=False)
        out_enc_packed, (h, c) = self.encoder(x_packed)
        out_enc, _ = nn.utils.rnn.pad_packed_sequence(
            out_enc_packed, batch_first=False, padding_value=self.padding_value)
        h, c = self._prepare_h_and_c_for_decoder(h, c)
        W_1_out = self.W_1(out_enc)

        # ── masks (plain bool tensors, never attached to the graph) ────
        enc_len = out_enc.size(0)
        batch_idx = torch.arange(batch_size, device=device)

        # Padding mask (True = invalid)
        pad_mask = (torch.arange(enc_len, device=device).unsqueeze(1)
                    >= seq_lens_aug.unsqueeze(0))
        # Selection mask – accumulates chosen positions (detached bool)
        sel_mask = pad_mask.clone()

        inp = torch.zeros(batch_size, self.output_size, device=device)
        active = torch.ones(batch_size, dtype=torch.bool, device=device)

        all_actions, all_log_probs, all_entropies = [], [], []

        for _ in range(max_steps):
            h, c = self.decoder_cell(inp, (h, c))
            scores = self.attention(W_1_out, out_enc, h, seq_lens_aug)

            # Mask already-selected positions, but keep stop (pos 0) open
            mask_step = sel_mask.clone()
            mask_step[0, :] = False
            scores = scores.masked_fill(mask_step, float('-inf'))

            probs = F.softmax(scores, dim=0).T             # (B, enc_len)
            probs = probs.clamp(min=1e-8)
            probs = probs / probs.sum(dim=1, keepdim=True)

            dist = torch.distributions.Categorical(probs)
            idx = dist.sample()                             # (B,)

            all_actions.append(idx - 1)                     # 0→-1 (stop)
            all_log_probs.append(dist.log_prob(idx) * active.float())
            all_entropies.append(dist.entropy() * active.float())

            # Update active set
            active = active & (idx != 0)
            if not active.any():
                break

            # Mask selected (non-stop) positions for next step
            non_stop = idx > 0
            if non_stop.any():
                sel_mask = sel_mask.clone()
                sel_mask[idx[non_stop], batch_idx[non_stop]] = True

            # Feed encoder state at selected position into decoder;
            # zero out input for inactive samples to avoid polluting their h, c
            selected_enc = out_enc[idx, batch_idx]
            inp = self.enc_to_dec(selected_enc)
            inp = inp * active.float().unsqueeze(1)

        return (torch.stack(all_actions, dim=1),
                torch.stack(all_log_probs, dim=1),
                torch.stack(all_entropies, dim=1))


class Critic(nn.Module):
    """Value-function estimator (baseline) for REINFORCE.

    Encodes the input sequence, applies *n_glimpses* rounds of
    attention to produce a context vector, then maps it to a scalar.
    """

    def __init__(self, input_size, hidden_size, bidirectional=False, n_glimpses=2):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            bidirectional=bidirectional)
        enc_dim = hidden_size * (2 if bidirectional else 1)

        self.W_q = nn.Linear(enc_dim, enc_dim)
        self.W_k = nn.Linear(enc_dim, enc_dim)
        self.v   = nn.Linear(enc_dim, 1, bias=False)
        self.tanh = nn.Tanh()

        self.n_glimpses = n_glimpses
        self.value_head = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.ReLU(),
            nn.Linear(enc_dim, 1),
        )

    def forward(self, x, seq_lens):
        """
        x:        (max_seq_len, batch, input_size)
        seq_lens: (batch,)
        Returns:  (batch,)  scalar value estimates
        """
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x, seq_lens.cpu(), enforce_sorted=False)
        out_packed, (h, _) = self.encoder(x_packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed)

        seq_len, batch_size = out.size(0), out.size(1)
        mask = (torch.arange(seq_len, device=out.device).unsqueeze(1)
                >= seq_lens.unsqueeze(0).to(out.device))

        # Initial query from final hidden state
        if self.encoder.bidirectional:
            q = torch.cat([h[-2], h[-1]], dim=1)
        else:
            q = h[-1]

        K = self.W_k(out)                                   # (S, B, D)
        for _ in range(self.n_glimpses):
            Q = self.W_q(q)                                  # (B, D)
            energy = self.tanh(K + Q.unsqueeze(0))           # (S, B, D)
            scores = self.v(energy).squeeze(2)               # (S, B)
            scores = scores.masked_fill(mask, float('-inf'))
            alpha = F.softmax(scores, dim=0)                 # (S, B)
            q = (alpha.unsqueeze(2) * out).sum(dim=0)        # (B, D)

        return self.value_head(q).squeeze(1)                 # (B,)