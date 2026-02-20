import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import USE_CUDA

class Attention(nn.Module):
    def __init__(self, hidden_size, use_tanh=False, C=10, name='Bahdanau', use_cuda=USE_CUDA):
        super(Attention, self).__init__()
        
        self.use_tanh = use_tanh
        self.C = C
        self.name = name
        
        if name == 'Bahdanau':
            self.W_query = nn.Linear(hidden_size, hidden_size)
            self.W_ref   = nn.Conv1d(hidden_size, hidden_size, 1, 1)

            V = torch.FloatTensor(hidden_size)
            if use_cuda:
                V = V.cuda()  
            self.V = nn.Parameter(V)
            self.V.data.uniform_(-0.08, 0.08)
            self.W_query.weight.data.uniform_(-0.08, 0.08)
            self.W_query.bias.data.uniform_(-0.08, 0.08)
            self.W_ref.weight.data.uniform_(-0.08, 0.08)
            self.W_ref.bias.data.uniform_(-0.08, 0.08)
            
        
    def forward(self, query, ref):
        """
        Args: 
            query: [batch_size x hidden_size]
            ref:   ]batch_size x seq_len x hidden_size]
        """
        
        batch_size = ref.size(0)
        seq_len    = ref.size(1)
        
        if self.name == 'Bahdanau':
            ref = ref.permute(0, 2, 1)
            query = self.W_query(query).unsqueeze(2)  # [batch_size x hidden_size x 1]
            ref   = self.W_ref(ref)  # [batch_size x hidden_size x seq_len] 
            expanded_query = query.repeat(1, 1, seq_len) # [batch_size x hidden_size x seq_len]
            V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size x 1 x hidden_size]
            logits = torch.bmm(V, F.tanh(expanded_query + ref)).squeeze(1)
            
        elif self.name == 'Dot':
            query  = query.unsqueeze(2)
            logits = torch.bmm(ref, query).squeeze(2) #[batch_size x seq_len x 1]
            ref = ref.permute(0, 2, 1)
        
        else:
            raise NotImplementedError
        
        if self.use_tanh:
            logits = self.C * F.tanh(logits)
        else:
            logits = logits  
        return ref, logits

class GraphEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size, use_cuda=USE_CUDA):
        super(GraphEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.use_cuda = use_cuda
        
        self.embedding = nn.Parameter(torch.FloatTensor(input_size, embedding_size)) 
        self.embedding.data.uniform_(-0.08, 0.08)
        
    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len    = inputs.size(2)
        embedding = self.embedding.repeat(batch_size, 1, 1)  
        embedded = []
        inputs = inputs.unsqueeze(1)
        for i in range(seq_len):
            embedded.append(torch.bmm(inputs[:, :, :, i].float(), embedding))
        embedded = torch.cat(embedded, 1)
        return embedded

class PointerNet(nn.Module):
    def __init__(self, 
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            attention,
            use_cuda=USE_CUDA,
            temperature=1.0,
            eos_logit_bias_init=None,
            eos_logit_bias_learnable=False):
        super(PointerNet, self).__init__()
        
        self.embedding_size = embedding_size
        self.hidden_size    = hidden_size
        self.n_glimpses     = n_glimpses
        self.seq_len        = seq_len
        self.use_cuda       = use_cuda
        self.temperature = temperature
        
        self.embedding = GraphEmbedding(2, embedding_size, use_cuda=use_cuda)
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.pointer = Attention(hidden_size, use_tanh=use_tanh, C=tanh_exploration, name=attention, use_cuda=use_cuda)
        self.glimpse = Attention(hidden_size, use_tanh=False, name=attention, use_cuda=use_cuda)
        
        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-0.08, 0.08)
        
        for name, param in self.encoder.named_parameters():
            if 'weight' in name or 'bias' in name:
                param.data.uniform_(-0.08, 0.08)
        for name, param in self.decoder.named_parameters():
            if 'weight' in name or 'bias' in name:
                param.data.uniform_(-0.08, 0.08)
        
        self.eos_token = -1  # EOS index (will be appended to input)

        # Optional EOS logit bias — positive init encourages stopping
        # By default registered as a buffer (non-learnable, but saved in state_dict)
        if eos_logit_bias_init is not None:
            if eos_logit_bias_learnable:
                self.eos_logit_bias = nn.Parameter(torch.tensor(float(eos_logit_bias_init)))
            else:
                self.register_buffer('eos_logit_bias', torch.tensor(float(eos_logit_bias_init)))
        else:
            self.eos_logit_bias = None
        
    def apply_mask_to_logits(self, logits, mask, idxs, lengths=None): 
        batch_size = logits.size(0)
        # Create a new mask to avoid in-place modifications
        clone_mask = mask.clone()
        # Mask already-selected indices
        if idxs is not None:
            clone_mask[torch.arange(batch_size), idxs] = True
        
        # For each sample, ensure its EOS position remains unmasked
        if lengths is not None:
            for b in range(batch_size):
                actual_len = lengths[b].item() if torch.is_tensor(lengths[b]) else lengths[b]
                eos_pos = actual_len  # EOS position for this sample
                clone_mask[b, eos_pos] = False  # Ensure EOS is unmasked for this sample
        
        # Check if any sample has all positions masked except EOS
        for b in range(batch_size):
            if lengths is not None:
                actual_len = lengths[b].item() if torch.is_tensor(lengths[b]) else lengths[b]
                eos_pos = actual_len
                if clone_mask[b].sum() == clone_mask.size(1) - 1:  # All but one position masked
                    clone_mask[b, eos_pos] = False  # Ensure EOS is unmasked
        
        # Apply mask to logits
        masked_logits = logits.masked_fill(clone_mask, float('-inf'))
        
        # Additional safety: ensure no row has all -inf values
        for b in range(batch_size):
            if torch.all(torch.isinf(masked_logits[b]) & (masked_logits[b] < 0)):
                # If all values are -inf, unmask EOS position for this sample
                if lengths is not None:
                    actual_len = lengths[b].item() if torch.is_tensor(lengths[b]) else lengths[b]
                    eos_pos = actual_len
                    masked_logits[b, eos_pos] = logits[b, eos_pos]
                    clone_mask[b, eos_pos] = False
                else:
                    # Fallback: unmask the first position
                    masked_logits[b, 0] = logits[b, 0]
                    clone_mask[b, 0] = False
        
        return masked_logits, clone_mask
            
    def forward(self, inputs, padding_mask=None, lengths=None, deterministic: bool = False, max_decode_steps=None, no_eos: bool = False):
        """
        Args: 
            inputs: [batch_size x num_points x 2]
            padding_mask: [batch_size x num_points] (True for real, False for pad)
            lengths: list or tensor of ints, true number of vertices per sample
            max_decode_steps: int, 1-D LongTensor of shape (batch_size,), or
                None.  Limits the maximum number of guard selections per
                instance (excluding the EOS step itself).  A scalar int is
                broadcast to every sample; a per-sample tensor lets each
                polygon use its own ⌊n/3⌋ budget.
                When a sample reaches its budget, EOS is forced.
            no_eos: bool.  If True, the model produces a full permutation of
                the n real vertices (no EOS appended/available).  Each sample
                decodes for exactly lengths[b] steps.  Used for the
                ranking-based AGP formulation.
        Returns:
            output_idxs: list of [num_selected_guards] tensors, one per instance, with EOS as end
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        device = inputs.device

        if no_eos:
            # ── Permutation mode: no EOS token ──────────────────
            embedded = self.embedding(inputs.transpose(1, 2))  # [B, N, emb]

            if lengths is not None:
                enc_lengths = lengths.cpu() if torch.is_tensor(lengths) else torch.tensor(lengths, device='cpu')
                packed_embedded = nn.utils.rnn.pack_padded_sequence(
                    embedded, enc_lengths, batch_first=True, enforce_sorted=False)
                packed_outputs, (hidden, context) = self.encoder(packed_embedded)
                encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_outputs, batch_first=True, total_length=seq_len)
            else:
                encoder_outputs, (hidden, context) = self.encoder(embedded)

            # Build mask: True = invalid
            if padding_mask is not None:
                mask = ~padding_mask  # True for padded positions
            else:
                mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
            if lengths is not None:
                for b in range(batch_size):
                    n = lengths[b].item() if torch.is_tensor(lengths[b]) else lengths[b]
                    if n < seq_len:
                        mask[b, n:] = True

            idxs = None
            decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)
            output_idxs = [[] for _ in range(batch_size)]
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            per_sample_steps = lengths if lengths is not None else torch.full(
                (batch_size,), seq_len, dtype=torch.long, device=device)
            if not torch.is_tensor(per_sample_steps):
                per_sample_steps = torch.tensor(per_sample_steps, dtype=torch.long, device=device)
            max_steps = int(per_sample_steps.max().item())

            log_probs_list = [[] for _ in range(batch_size)]
            for step in range(max_steps):
                _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))
                query = hidden.squeeze(0)
                for _ in range(self.n_glimpses):
                    ref, logits = self.glimpse(query, encoder_outputs)
                    # Simple mask for permutation mode (no EOS logic)
                    clone_mask = mask.clone()
                    if idxs is not None:
                        clone_mask[torch.arange(batch_size), idxs] = True
                    masked_logits = logits.masked_fill(clone_mask, float('-inf'))
                    # Safety: ensure no all-inf rows
                    for b in range(batch_size):
                        if not finished[b] and torch.all(torch.isinf(masked_logits[b]) & (masked_logits[b] < 0)):
                            # Unmask first valid position
                            n = per_sample_steps[b].item()
                            for v in range(n):
                                if not mask[b, v]:
                                    masked_logits[b, v] = logits[b, v]
                                    clone_mask[b, v] = False
                                    break
                    logits = masked_logits / self.temperature
                    query = torch.bmm(ref, F.softmax(logits, dim=1).unsqueeze(2)).squeeze(2)

                _, logits = self.pointer(query, encoder_outputs)
                clone_mask = mask.clone()
                if idxs is not None:
                    clone_mask[torch.arange(batch_size), idxs] = True
                logits = logits.masked_fill(clone_mask, float('-inf'))
                for b in range(batch_size):
                    if not finished[b] and torch.all(torch.isinf(logits[b]) & (logits[b] < 0)):
                        n = per_sample_steps[b].item()
                        for v in range(n):
                            if not mask[b, v]:
                                logits[b, v] = 0.0
                                break

                probs = F.softmax(logits, dim=1)
                nan_rows = torch.isnan(probs).any(dim=1, keepdim=True)
                if nan_rows.any():
                    fallback = torch.zeros_like(probs)
                    fallback[:, 0] = 1.0
                    probs = torch.where(nan_rows, fallback, probs)

                if deterministic:
                    idxs = torch.argmax(probs, dim=1)
                else:
                    idxs = probs.multinomial(1).squeeze(1)

                for b in range(batch_size):
                    if not finished[b]:
                        output_idxs[b].append(idxs[b].item())
                        log_prob = torch.log(probs[b, idxs[b]].clamp(min=1e-20))
                        log_probs_list[b].append(log_prob)
                        if len(output_idxs[b]) >= per_sample_steps[b].item():
                            finished[b] = True

                selected_mask = F.one_hot(idxs, seq_len).bool()
                mask = mask | selected_mask
                decoder_input = embedded[torch.arange(batch_size), idxs, :]
                if finished.all():
                    break

            log_probs = [torch.stack(lp).sum() if len(lp) > 0
                         else torch.tensor(0., device=device) for lp in log_probs_list]
            log_probs = torch.stack(log_probs)
            return output_idxs, log_probs
        
        # ── Original EOS mode below ─────────────────────────────
        # Add EOS token to input (as zeros)
        eos_vec = torch.zeros(batch_size, 1, 2, device=inputs.device)
        inputs_ext = torch.cat([inputs, eos_vec], dim=1)  # [B, N+1, 2]
        embedded = self.embedding(inputs_ext.transpose(1,2))  # [B, N+1, emb]
        # Use packed sequence for encoder
        if lengths is not None:
            # Add 1 to each length for EOS
            enc_lengths = (lengths + 1).cpu() if torch.is_tensor(lengths) else torch.tensor([l+1 for l in lengths], device='cpu')
            packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, enc_lengths, batch_first=True, enforce_sorted=False)
            packed_outputs, (hidden, context) = self.encoder(packed_embedded)
            encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, total_length=seq_len+1)
            
        else:
            encoder_outputs, (hidden, context) = self.encoder(embedded)
        total_len = seq_len + 1  # including shared EOS
        # Build mask: True for masked (invalid), False for valid
        if padding_mask is not None:
            pad = ~padding_mask  # True for padded positions
            pad_eos = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
            mask = torch.cat([pad, pad_eos], dim=1)
        else:
            mask = torch.zeros(batch_size, total_len, dtype=torch.bool, device=device)
        
        # For each sample, mask positions beyond its actual sequence length
        if lengths is not None:
            for b in range(batch_size):
                actual_len = lengths[b].item() if torch.is_tensor(lengths[b]) else lengths[b]
                # Mask positions beyond actual_len (but leave EOS at actual_len unmasked)
                if actual_len + 1 < total_len:
                    mask[b, actual_len + 1:] = True
        
        idxs = None
        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)
        output_idxs = [[] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        max_steps = total_len  # allow up to N+1 selections

        # Per-sample guard budget from Art Gallery Theorem bound or user limit.
        # When a sample selects this many non-EOS guards, force EOS next step.
        if max_decode_steps is not None:
            if torch.is_tensor(max_decode_steps):
                per_sample_budget = max_decode_steps.to(dtype=torch.long, device=device)
            else:
                per_sample_budget = torch.full((batch_size,), int(max_decode_steps),
                                              dtype=torch.long, device=device)
        else:
            per_sample_budget = None
        non_eos_counts = torch.zeros(batch_size, dtype=torch.long, device=device)

        log_probs_list = [[] for _ in range(batch_size)]
        for step in range(max_steps):
            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))
            query = hidden.squeeze(0)
            for _ in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, idxs, lengths)
                logits = logits / self.temperature
                query = torch.bmm(ref, F.softmax(logits, dim=1).unsqueeze(2)).squeeze(2)
            _, logits = self.pointer(query, encoder_outputs)
            logits, mask = self.apply_mask_to_logits(logits, mask, idxs, lengths)
            # Apply learnable EOS logit bias (makes stopping more likely)
            if self.eos_logit_bias is not None and lengths is not None:
                eos_bias_vec = torch.zeros_like(logits)
                for b in range(batch_size):
                    if not finished[b]:
                        eos_pos = lengths[b].item() if torch.is_tensor(lengths[b]) else lengths[b]
                        if not mask[b, eos_pos]:
                            eos_bias_vec[b, eos_pos] = 1.0
                logits = logits + eos_bias_vec * self.eos_logit_bias

            # Budget enforcement: if a sample has used all its non-EOS budget,
            # mask everything except EOS so only EOS can be selected.
            if per_sample_budget is not None and lengths is not None:
                for b in range(batch_size):
                    if not finished[b] and non_eos_counts[b] >= per_sample_budget[b]:
                        eos_pos = lengths[b].item() if torch.is_tensor(lengths[b]) else lengths[b]
                        logits[b, :] = float('-inf')
                        logits[b, eos_pos] = 0.0  # only EOS allowed
            # Safety: if all logits are -inf for a sample, unmask EOS
            for b in range(batch_size):
                actual_eos = lengths[b].item() if lengths is not None and torch.is_tensor(lengths[b]) else (lengths[b] if lengths is not None else seq_len)
                if torch.all(torch.isinf(logits[b]) & (logits[b] < 0)):
                    logits[b, actual_eos] = 0.0  # Only EOS for this sample is valid
            probs = F.softmax(logits, dim=1)
            # Safety: replace any NaN rows with one-hot on their respective EOS
            nan_rows = torch.isnan(probs).any(dim=1, keepdim=True)
            if nan_rows.any():
                fallback = torch.zeros_like(probs)
                if lengths is not None:
                    for b in range(batch_size):
                        actual_eos = lengths[b].item() if torch.is_tensor(lengths[b]) else lengths[b]
                        fallback[b, actual_eos] = 1.0
                else:
                    fallback[:, 0] = 1.0
                probs = torch.where(nan_rows, fallback, probs)
            if deterministic:
                idxs = torch.argmax(probs, dim=1)
            else:
                idxs = probs.multinomial(1).squeeze(1)
            
            for b in range(batch_size):
                if not finished[b]:
                    output_idxs[b].append(idxs[b].item())
                    log_prob = torch.log(probs[b, idxs[b]])
                    log_probs_list[b].append(log_prob)
                    # Check if this sample reached its EOS
                    actual_eos = lengths[b].item() if lengths is not None and torch.is_tensor(lengths[b]) else (lengths[b] if lengths is not None else seq_len)
                    if idxs[b].item() == actual_eos:
                        finished[b] = True
                    else:
                        non_eos_counts[b] += 1
            selected_mask = F.one_hot(idxs, total_len).bool()
            mask = mask | selected_mask
            decoder_input = embedded[torch.arange(batch_size), idxs, :]
            if finished.all():
                break
        log_probs = [torch.stack(lp).sum() if len(lp) > 0 else torch.tensor(0., device=inputs.device) for lp in log_probs_list]
        log_probs = torch.stack(log_probs)  # shape: (batch_size,)
        return output_idxs, log_probs

class CombinatorialRL(nn.Module):
    def __init__(self, 
            embedding_size,
            hidden_size,
            seq_len,
            n_glimpses,
            tanh_exploration,
            use_tanh,
            reward,
            attention,
            use_cuda=USE_CUDA,
            temperature=1.0,
            eos_logit_bias_init=None,
            eos_logit_bias_learnable=False):
        super(CombinatorialRL, self).__init__()
        self.reward = reward
        self.use_cuda = use_cuda
        
        self.actor = PointerNet(
                embedding_size,
                hidden_size,
                seq_len,
                n_glimpses,
                tanh_exploration,
                use_tanh,
                attention,
                use_cuda,
                temperature,
                eos_logit_bias_init=eos_logit_bias_init,
                eos_logit_bias_learnable=eos_logit_bias_learnable)


    def forward(self, inputs, padding_mask=None, lengths=None, deterministic: bool = False, max_decode_steps=None, no_eos: bool = False):
        """
        Run the PointerNet actor with padding_mask and lengths to ignore padded vertices and return selected guard indices and log-probabilities for REINFORCE.
        """
        action_idxs, log_probs = self.actor(inputs, padding_mask=padding_mask, lengths=lengths, deterministic=deterministic, max_decode_steps=max_decode_steps, no_eos=no_eos)
        return action_idxs, log_probs


# --- CriticNet: LSTM encoder + process block + 2-layer decoder ---
class CriticNet(nn.Module):
    """
    Critic with LSTM encoder, process block (glimpse attention), and 2-layer decoder.
    Mirrors the architecture from Neural Combinatorial Optimization.
    """
    def __init__(self, embedding_size, hidden_size, n_glimpses, attention_type, use_cuda=USE_CUDA):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_glimpses = n_glimpses
        self.use_cuda = use_cuda
        self.embedding = GraphEmbedding(2, embedding_size, use_cuda)
        self.encoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.glimpse = Attention(hidden_size, name=attention_type, use_cuda=use_cuda)
        self.P = n_glimpses
        self.dec_fc1 = nn.Linear(hidden_size, hidden_size)
        self.dec_fc2 = nn.Linear(hidden_size, 1)

    def forward(self, inputs, mask, lengths):
        # inputs: [B, L, 2], mask: [B, L], lengths: [B]
        batch_size, seq_len, _ = inputs.size()
        device = inputs.device
        # 1) Embedding
        emb = self.embedding(inputs.transpose(1,2))  # [B, L, emb]
        # 2) LSTM encoder (packed)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.encoder(packed)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)
        # h: [1, B, hidden_size]
        h_state = h  # [1, B, hidden_size]
        # 3) Process block: repeat glimpses
        for _ in range(self.P):
            query = h_state.squeeze(0)  # [B, hidden_size]
            _, logits = self.glimpse(query, encoder_outputs)
            # Use context vector as new hidden state (weighted sum of encoder outputs)
            attn_weights = F.softmax(logits, dim=1)  # [B, L]
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # [B, hidden_size]
            h_state = context.unsqueeze(0)  # [1, B, hidden_size]
        # 4) 2-layer decoder
        h_final = h_state.squeeze(0)  # [B, hidden_size]
        x = F.relu(self.dec_fc1(h_final))
        b = self.dec_fc2(x).squeeze(-1)  # [B]
        return b


def create_actor(embedding_size, hidden_size, seq_len, n_glimpses, 
                tanh_exploration, use_tanh, attention_type, reward_fn, temperature=1.0,
                eos_logit_bias_init=None, eos_logit_bias_learnable=False):
    """Create and initialize a TSP model"""
    model = CombinatorialRL(
        embedding_size,
        hidden_size,
        seq_len,
        n_glimpses, 
        tanh_exploration,
        use_tanh,
        reward_fn,
        attention=attention_type,
        use_cuda=USE_CUDA,
        temperature=temperature,
        eos_logit_bias_init=eos_logit_bias_init,
        eos_logit_bias_learnable=eos_logit_bias_learnable)
    
    if USE_CUDA:
        model = model.cuda()
    
    return model 



def create_critic(embedding_size, hidden_size, n_glimpses, attention_type):
    """Create and initialize a CriticNet model with LSTM encoder/process block."""
    model = CriticNet(embedding_size, hidden_size, n_glimpses, attention_type, use_cuda=USE_CUDA)
    if USE_CUDA:
        model = model.cuda()
    return model