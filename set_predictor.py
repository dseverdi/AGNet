"""Single-shot set-membership predictor for AGP editor.

Different formulation from edit_head.EditHead (which is iterative). Here the
editor outputs, in a single forward pass, P(v in LS_final | polygon, seed)
for every vertex v. The final set is obtained by thresholding those
probabilities.

Why this exists: the iterative editor has three structural pathologies
(compounding errors, STOP-action coupling, train/test distribution drift)
that we couldn't resolve with topology features, aux losses, or DAgger. This
formulation eliminates all three by construction:

  • No iteration → no compounding.
  • No STOP head → no STOP-action coupling.
  • Same (polygon, pointer_seed) input distribution at train and inference
    → no drift.

The model uses the pretrained pointer's encoder embeddings as per-vertex
features. Those embeddings were learned from coordinates only (the pointer
sees no disc_vis); reusing them is consistent with the "no geometric oracle
at inference" constraint. At deployment, the pointer encoder runs once per
polygon and the SetPredictor runs once on its output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the self-attention block from edit_head.py.
from edit_head import _SelfAttnBlock


# ──────────────────────────────────────────────────────────────────────
#  Pointer encoder access
# ──────────────────────────────────────────────────────────────────────
def extract_pointer_embeddings(pointer, inputs, lengths):
    """Run only the pointer's encoder; return per-vertex hidden states.

    Bypasses the pointer's decoder/EOS logic entirely. Uses pointer.embedding
    and pointer.encoder directly.

    Args:
        pointer:  the loaded PointerNet (CombinatorialRL wrapping PointerNet)
        inputs:   (B, L, 2) float — polygon coordinates (already padded)
        lengths:  (B,) long — real vertex counts per polygon

    Returns:
        (B, L, hidden_size) per-vertex encoder embeddings. Positions beyond
        each polygon's length are arbitrary (caller should mask them).
    """
    # CombinatorialRL → PointerNet (`.actor`). Drill through.
    pn = pointer.actor if hasattr(pointer, "actor") else pointer

    # GraphEmbedding expects (B, 2, L) (it does Conv1d-style projection).
    embedded = pn.embedding(inputs.transpose(1, 2))           # (B, L, emb)

    L = inputs.size(1)
    enc_lengths = lengths.cpu() if torch.is_tensor(lengths) else torch.tensor(lengths, device="cpu")
    packed = nn.utils.rnn.pack_padded_sequence(
        embedded, enc_lengths, batch_first=True, enforce_sorted=False,
    )
    packed_out, _ = pn.encoder(packed)
    enc_out, _ = nn.utils.rnn.pad_packed_sequence(
        packed_out, batch_first=True, total_length=L,
    )
    return enc_out                                            # (B, L, H_ptr)


# ──────────────────────────────────────────────────────────────────────
#  SetPredictor
# ──────────────────────────────────────────────────────────────────────
class SetPredictor(nn.Module):
    """Per-vertex set-membership predictor.

    Forward inputs:
        ptr_emb: (B, L, H_ptr) — pointer encoder embeddings
        xy:      (B, L, 2)    — normalised polygon coordinates
        in_S:    (B, L) bool  — True for vertices in the pointer's seed
        pad:     (B, L) bool  — True for padded positions

    Returns: (B, L) logits for P(v in S_final).

    Padded positions have their logits set to -inf so the threshold rule
    naturally excludes them.
    """

    def __init__(self, ptr_emb_dim: int = 128, hidden: int = 128,
                 n_attn_layers: int = 3, heads: int = 8,
                 disable_ptr_emb: bool = False):
        super().__init__()
        self._ptr_emb_dim = ptr_emb_dim
        self._hidden = hidden
        # When True, the frozen pointer embedding is masked to zeros at every
        # forward pass — the architecture and parameter count are unchanged
        # but the encoder carries no information. Used for the coordinate-only
        # ablation that isolates representation signal from raw-coordinate
        # signal (paper §6, C1 ablation).
        self._disable_ptr_emb = bool(disable_ptr_emb)

        # Per-vertex input is (ptr_emb || xy || in_S) ∈ ℝ^{H_ptr + 3}.
        d_in = ptr_emb_dim + 3
        self.embed = nn.Linear(d_in, hidden)

        self.attn_blocks = nn.ModuleList([
            _SelfAttnBlock(hidden, heads=heads) for _ in range(n_attn_layers)
        ])
        self.norm = nn.LayerNorm(hidden)

        # Per-vertex output head sees (v || S_pool || G_pool) ∈ ℝ^{3H}.
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    @staticmethod
    def _pool_masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool along dim=1; rows with no valid entries return zeros."""
        m = mask.unsqueeze(-1).to(x.dtype)
        s = (x * m).sum(dim=1)
        cnt = m.sum(dim=1).clamp(min=1.0)
        return s / cnt

    def forward(self, ptr_emb: torch.Tensor, xy: torch.Tensor,
                in_S: torch.Tensor, pad: torch.Tensor | None = None
                ) -> torch.Tensor:
        if ptr_emb.dim() == 2:
            # Single-example convenience: add batch dim.
            ptr_emb = ptr_emb.unsqueeze(0)
            xy = xy.unsqueeze(0)
            in_S = in_S.unsqueeze(0)
            if pad is not None:
                pad = pad.unsqueeze(0)

        B, L, _ = ptr_emb.shape
        if pad is None:
            pad = torch.zeros(B, L, dtype=torch.bool, device=ptr_emb.device)

        in_S_f = in_S.float().unsqueeze(-1)
        if self._disable_ptr_emb:
            ptr_emb = torch.zeros_like(ptr_emb)
        feats = torch.cat([ptr_emb, xy, in_S_f], dim=-1)      # (B, L, H_ptr+3)
        h = self.embed(feats)
        for blk in self.attn_blocks:
            h = blk(h, key_padding_mask=pad)
        h = self.norm(h)

        valid = ~pad
        in_S_eff = in_S & valid

        S_ctx = self._pool_masked(h, in_S_eff)               # (B, H)
        G_ctx = self._pool_masked(h, valid)                  # (B, H)

        S_ctx_b = S_ctx.unsqueeze(1).expand(-1, L, -1)
        G_ctx_b = G_ctx.unsqueeze(1).expand(-1, L, -1)
        h_full = torch.cat([h, S_ctx_b, G_ctx_b], dim=-1)    # (B, L, 3H)
        logits = self.head(h_full).squeeze(-1)               # (B, L)

        # Mask padded positions to -inf.
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(pad, neg_inf)
        return logits

    @torch.no_grad()
    def predict_set(self, ptr_emb: torch.Tensor, xy: torch.Tensor,
                    in_S: torch.Tensor, pad: torch.Tensor | None = None,
                    threshold: float = 0.5) -> list[list[int]]:
        """Threshold to a list of vertex indices, one list per batch element."""
        logits = self.forward(ptr_emb, xy, in_S, pad)
        probs = torch.sigmoid(logits)
        out = []
        for b in range(logits.shape[0]):
            keep = (probs[b] >= threshold).nonzero(as_tuple=True)[0]
            out.append(keep.cpu().tolist())
        return out

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────────────
#  Loss
# ──────────────────────────────────────────────────────────────────────
def set_loss(logits: torch.Tensor, target_mask: torch.Tensor,
             pad: torch.Tensor | None = None,
             pos_weight: float | None = None) -> dict:
    """Binary cross-entropy per vertex against ``target_mask`` (1 if v ∈ S_final).

    Pad positions are excluded from the loss. Optional pos_weight rebalances
    positives (kept vertices) vs negatives (removed vertices).
    """
    device = logits.device
    if pad is None:
        pad = torch.zeros_like(target_mask, dtype=torch.bool)
    valid = (~pad).float()

    pw = (
        torch.tensor(pos_weight, device=device, dtype=logits.dtype)
        if pos_weight is not None else None
    )
    losses = F.binary_cross_entropy_with_logits(
        logits, target_mask.float(), reduction="none", pos_weight=pw,
    )
    losses = losses * valid
    n_valid = valid.sum().clamp(min=1.0)
    return {"loss": losses.sum() / n_valid}
