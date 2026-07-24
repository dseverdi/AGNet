import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
            eos_logit_bias_learnable=False,
            marg_cov_inject_enabled=True):
        super(PointerNet, self).__init__()
        self.marg_cov_inject_enabled = marg_cov_inject_enabled
        
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

        # Auxiliary head for conditional marginal-coverage prediction.
        # Given (decoder query, encoder per-vertex output), predicts a
        # scalar — the marginal coverage v would add to the current
        # partial guard set. Trained with a regression loss against
        # disc-vis ground truth at each decode step. Inference unchanged;
        # this only contributes to the gradient during fine-tuning.
        self.marg_cov_head = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        for m in self.marg_cov_head:
            if isinstance(m, nn.Linear):
                m.weight.data.uniform_(-0.08, 0.08)
                m.bias.data.uniform_(-0.08, 0.08)

        # Marginal-coverage feature injection. Projects the scalar
        # "marginal coverage of v given the partial guard set so far"
        # into the encoder's feature dim, so the pointer attention
        # scores vertices as a learned function of
        # (query, encoder_output_v, dynamic marginal coverage of v).
        # This is the LS-style placement signal made available to the
        # pointer at each decode step.
        #
        # Init scale: the feature added to encoder_outputs needs to be
        # comparable in magnitude to those outputs (~O(1)) so the BT
        # gradient sees the signal and shapes it. With marg_cov ∈ [0, 1]
        # and a `1 / sqrt(H)` weight scale, output magnitude ~
        # |marg_cov| · 1 / sqrt(H) · sqrt(H) = O(marg_cov) ≈ O(0.1).
        # Bias=0 so a vertex with zero marginal coverage adds no signal.
        # A learnable scalar gain is multiplied on top so the model can
        # learn to amplify or attenuate the contribution; init=1.0 so
        # the feature is visible from epoch 1.
        self.marg_cov_inject_proj = nn.Linear(1, hidden_size)
        proj_scale = 1.0 / (hidden_size ** 0.5)
        self.marg_cov_inject_proj.weight.data.uniform_(-proj_scale, proj_scale)
        self.marg_cov_inject_proj.bias.data.zero_()
        self.marg_cov_inject_gain = nn.Parameter(torch.tensor(1.0))

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
        # Vectorised, semantics-identical replacement for
        # _apply_mask_to_logits_legacy below. The legacy version ran three
        # `for b in range(batch_size)` loops (EOS-unmask, all-but-one guard,
        # all-inf safety), each doing per-element .item()/.sum() host<->device
        # syncs; called twice per decode step, this dominated training
        # wall-clock (~60% of a step, GPU ~17% util). Same result, tensor ops,
        # at most one .any() sync per call. Set self._legacy_mask=True to
        # revert. Equivalence is asserted in tools/smoke_fast_decode.py.
        if getattr(self, "_legacy_mask", False):
            return self._apply_mask_to_logits_legacy(logits, mask, idxs, lengths)
        B = logits.size(0)
        device = logits.device
        clone_mask = mask.clone()
        if idxs is not None:
            clone_mask[torch.arange(B, device=device), idxs] = True
        eos = None
        if lengths is not None:
            eos = (lengths if torch.is_tensor(lengths)
                   else torch.as_tensor(lengths, device=device)).to(device).long().view(-1)
            # Keep every sample's EOS position open (legacy steps 2 & 3; step 3
            # is a no-op once EOS is unconditionally unmasked here).
            clone_mask.scatter_(1, eos.view(-1, 1), False)
        masked_logits = logits.masked_fill(clone_mask, float('-inf'))
        # Safety: an entirely -inf row gets its EOS (or col 0) logit restored.
        all_neg_inf = (torch.isinf(masked_logits) & (masked_logits < 0)).all(dim=1)
        if bool(all_neg_inf.any()):
            fix = eos if eos is not None else torch.zeros(B, dtype=torch.long, device=device)
            r = all_neg_inf.nonzero(as_tuple=True)[0]
            c = fix[r]
            masked_logits[r, c] = logits[r, c]
            clone_mask[r, c] = False
        return masked_logits, clone_mask

    def _apply_mask_to_logits_legacy(self, logits, mask, idxs, lengths=None):
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
            
    def forward(self, inputs, padding_mask=None, lengths=None, deterministic: bool = False, max_decode_steps=None, no_eos: bool = False, eos_cov_threshold: float = 0.0, vis_matrices_list=None):
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
            eos_cov_threshold: float.  When > 0 and vis_matrices_list is
                provided, the EOS logit is masked (-inf) at any step where
                the partial coverage of already-selected guards is below
                this threshold.  Forces the model to keep picking until
                feasibility — closes the "early-EOS escape" gradient.
            vis_matrices_list: optional list of length batch_size, one
                np.ndarray per sample of shape (n_b, M) with bool dtype,
                from disc_vis cache.  Used only with eos_cov_threshold > 0.
                Entries may be None for samples without a vis cache.
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

        # Coverage-based EOS gating: track per-sample running coverage so we
        # can mask the EOS logit until partial coverage ≥ eos_cov_threshold.
        cov_gate_active = (
            eos_cov_threshold > 0.0
            and vis_matrices_list is not None
            and lengths is not None
        )
        # Marginal-coverage injection into the pointer attention. Active
        # whenever disc_vis is supplied; covered bitsets are shared with
        # the EOS-gate path above when both are on.
        cov_inject_active = (
            self.marg_cov_inject_enabled
            and vis_matrices_list is not None
            and lengths is not None
        )
        cov_track_active = cov_gate_active or cov_inject_active
        covered_per_sample = None
        cov_M_per_sample = None
        if cov_track_active:
            covered_per_sample = []
            cov_M_per_sample = []
            for b in range(batch_size):
                vm = vis_matrices_list[b] if b < len(vis_matrices_list) else None
                if vm is None:
                    covered_per_sample.append(None)
                    cov_M_per_sample.append(0)
                else:
                    covered_per_sample.append(np.zeros(vm.shape[1], dtype=bool))
                    cov_M_per_sample.append(int(vm.shape[1]))

        log_probs_list = [[] for _ in range(batch_size)]

        # Fast decode: for the plain training/eval case (no coverage gating,
        # no marginal-cov injection, no per-sample budget, no EOS-logit bias,
        # lengths given) the per-step per-sample .item() loops below are pure
        # host<->device syncs. When use_fast, we vectorise the safety/NaN guards
        # and DEFER the append: sampled indices and (differentiable) step
        # log-probs are accumulated as tensors and materialised once after the
        # loop, so ~3*B syncs/step become ~2. Sampling, masking and state
        # updates are byte-for-byte the legacy path, so the result is identical
        # (asserted in tools/smoke_fast_decode.py). AGNET_LEGACY_DECODE=1 /
        # self._legacy_mask=True forces the legacy loops.
        use_fast = (
            not getattr(self, "_legacy_mask", False)
            and not cov_track_active
            and per_sample_budget is None
            and self.eos_logit_bias is None
            and lengths is not None
        )
        if use_fast:
            eos_pos_t = (lengths if torch.is_tensor(lengths)
                         else torch.as_tensor(lengths, device=device)).to(device).long().view(-1)
            step_idx_list: list = []
            step_lp_list: list = []

        for step in range(max_steps):
            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))
            query = hidden.squeeze(0)
            for _ in range(self.n_glimpses):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask_to_logits(logits, mask, idxs, lengths)
                logits = logits / self.temperature
                query = torch.bmm(ref, F.softmax(logits, dim=1).unsqueeze(2)).squeeze(2)
            # Optionally augment encoder refs with per-vertex marginal
            # coverage of the current partial guard set.  This injects the
            # LS-style placement signal directly into the pointer score
            # (glimpse/query path is unchanged).
            if cov_inject_active:
                marg_full = np.zeros((batch_size, total_len), dtype=np.float32)
                for b in range(batch_size):
                    vm_b = vis_matrices_list[b] if b < len(vis_matrices_list) else None
                    if vm_b is None or covered_per_sample[b] is None:
                        continue
                    M_b = cov_M_per_sample[b]
                    if M_b <= 0:
                        continue
                    not_covered = ~covered_per_sample[b]
                    n_b = vm_b.shape[0]
                    marg_full[b, :n_b] = (vm_b & not_covered).sum(axis=1) / float(M_b)
                marg_t = torch.from_numpy(marg_full).to(
                    device=encoder_outputs.device, dtype=encoder_outputs.dtype,
                )
                feat = self.marg_cov_inject_proj(marg_t.unsqueeze(-1))
                pointer_refs = encoder_outputs + self.marg_cov_inject_gain * feat
            else:
                pointer_refs = encoder_outputs
            _, logits = self.pointer(query, pointer_refs)
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

            # Coverage-based EOS gating: mask EOS while partial coverage <
            # threshold, but only if at least one non-EOS vertex still has
            # finite logit (so we don't strand the sample with all -inf).
            if cov_gate_active:
                for b in range(batch_size):
                    if finished[b] or covered_per_sample[b] is None:
                        continue
                    M_b = cov_M_per_sample[b]
                    if M_b <= 0:
                        continue
                    cov_b = float(covered_per_sample[b].sum()) / M_b
                    if cov_b >= eos_cov_threshold:
                        continue
                    eos_pos = lengths[b].item() if torch.is_tensor(lengths[b]) else lengths[b]
                    # Only block EOS if some non-EOS vertex remains valid.
                    row = logits[b]
                    has_alt = False
                    for v in range(row.shape[0]):
                        if v == eos_pos:
                            continue
                        if torch.isfinite(row[v]):
                            has_alt = True
                            break
                    if has_alt:
                        logits[b, eos_pos] = float('-inf')
            # Safety: if all logits are -inf for a sample, unmask EOS
            if use_fast:
                all_neg_inf = (torch.isinf(logits) & (logits < 0)).all(dim=1)
                if bool(all_neg_inf.any()):
                    r = all_neg_inf.nonzero(as_tuple=True)[0]
                    logits[r, eos_pos_t[r]] = 0.0
            else:
                for b in range(batch_size):
                    actual_eos = lengths[b].item() if lengths is not None and torch.is_tensor(lengths[b]) else (lengths[b] if lengths is not None else seq_len)
                    if torch.all(torch.isinf(logits[b]) & (logits[b] < 0)):
                        logits[b, actual_eos] = 0.0  # Only EOS for this sample is valid
            probs = F.softmax(logits, dim=1)
            # Safety: replace any NaN rows with one-hot on their respective EOS
            nan_rows = torch.isnan(probs).any(dim=1, keepdim=True)
            if nan_rows.any():
                fallback = torch.zeros_like(probs)
                if use_fast:
                    fallback.scatter_(1, eos_pos_t.view(-1, 1), 1.0)
                elif lengths is not None:
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
            
            if use_fast:
                # Defer per-sample bookkeeping: accumulate sampled indices and
                # differentiable step log-probs as tensors; reconstruct after
                # the loop. `finished` is updated tensor-wise for the early
                # break. Same log-prob (no clamp) as the legacy branch below.
                step_idx_list.append(idxs)
                step_lp_list.append(torch.log(probs.gather(1, idxs.unsqueeze(1)).squeeze(1)))
                finished = finished | (idxs == eos_pos_t)
            else:
                for b in range(batch_size):
                    if not finished[b]:
                        picked = idxs[b].item()
                        output_idxs[b].append(picked)
                        log_prob = torch.log(probs[b, idxs[b]])
                        log_probs_list[b].append(log_prob)
                        # Check if this sample reached its EOS
                        actual_eos = lengths[b].item() if lengths is not None and torch.is_tensor(lengths[b]) else (lengths[b] if lengths is not None else seq_len)
                        if picked == actual_eos:
                            finished[b] = True
                        else:
                            non_eos_counts[b] += 1
                            # Update running coverage with the picked vertex.
                            if (cov_track_active
                                    and covered_per_sample[b] is not None
                                    and vis_matrices_list[b] is not None):
                                vm_b = vis_matrices_list[b]
                                if 0 <= picked < vm_b.shape[0]:
                                    np.bitwise_or(
                                        covered_per_sample[b], vm_b[picked],
                                        out=covered_per_sample[b],
                                    )
            selected_mask = F.one_hot(idxs, total_len).bool()
            mask = mask | selected_mask
            decoder_input = embedded[torch.arange(batch_size), idxs, :]
            if finished.all():
                break

        if use_fast:
            # Reconstruct output_idxs (each sample keeps steps up to and
            # including its first EOS pick; if it never picks EOS it keeps all
            # steps run) and per-sample log-prob sums, matching the legacy
            # append semantics exactly.
            S = len(step_idx_list)
            if S == 0:
                log_probs = torch.zeros(batch_size, device=inputs.device)
                output_idxs = [[] for _ in range(batch_size)]
            else:
                idx_mat = torch.stack(step_idx_list, dim=0)        # [S, B]
                lp_mat = torch.stack(step_lp_list, dim=0)          # [S, B]
                picked_eos = idx_mat == eos_pos_t.unsqueeze(0)     # [S, B]
                step_ar = torch.arange(S, device=device).unsqueeze(1)        # [S,1]
                masked_step = torch.where(picked_eos, step_ar,
                                          torch.full_like(step_ar, S + 1))
                first_eos = masked_step.min(dim=0).values          # [B]; S+1 if none
                ever = first_eos <= S
                fs = torch.where(ever, first_eos,
                                 torch.full_like(first_eos, S - 1))  # last-step fallback
                keep = step_ar <= fs.unsqueeze(0)                  # [S, B] bool
                log_probs = (lp_mat * keep.to(lp_mat.dtype)).sum(dim=0)  # [B], differentiable
                idx_cpu = idx_mat.t().tolist()                     # [B][S]
                keep_cpu = keep.t().tolist()                       # [B][S]
                output_idxs = [[idx_cpu[b][s] for s in range(S) if keep_cpu[b][s]]
                               for b in range(batch_size)]
        else:
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


    def forward(self, inputs, padding_mask=None, lengths=None, deterministic: bool = False, max_decode_steps=None, no_eos: bool = False, eos_cov_threshold: float = 0.0, vis_matrices_list=None):
        """
        Run the PointerNet actor with padding_mask and lengths to ignore padded vertices and return selected guard indices and log-probabilities for REINFORCE.
        """
        action_idxs, log_probs = self.actor(inputs, padding_mask=padding_mask, lengths=lengths, deterministic=deterministic, max_decode_steps=max_decode_steps, no_eos=no_eos, eos_cov_threshold=eos_cov_threshold, vis_matrices_list=vis_matrices_list)
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