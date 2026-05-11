"""Learned LS-editor head.

Sits next to the (frozen) pretrained pointer. Given a polygon and the
current guard set S_t, predicts a single edit action:

    STOP                 - terminate
    REMOVE v             - remove v in S_t
    SWAP (v_out, v_in)   - replace v_out in S_t with v_in not in S_t

Trained on (state, action) tuples from `tools/build_ls_trajectories.py`.

Set-equivariance: no positional encoding; per-vertex features go through
a shared encoder + 1 self-attention layer + DeepSets-style global pool.

Architecture (~90k params):

    per-vertex features (D_in=6)
        |
        v
    Linear(6 -> H)             [vertex embedding]
        |
        v
    Self-Attention(H, heads=4) [1 layer, set-equivariant]
        |
        v
    LayerNorm + FFN(H -> 2H -> H)
        |
        v
    pool(in_S vertices) -> S-context (H)
    pool(all vertices)  -> G-context (H)
        |
        v
    For each vertex: concat(v_emb, S_ctx, G_ctx) -> 3H
        |
        +-- remove_head:   3H -> H -> 1   (valid only for v in S)
        +-- swap_out_head: 3H -> H -> 1   (valid only for v in S)
        +-- swap_in_head:  3H -> H -> 1   (valid only for v not in S)

    stop_head: G-context concat S-context -> 2H -> H -> 1
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
#  Per-vertex feature computation
# ──────────────────────────────────────────────────────────────────────
@dataclass
class VertexFeatures:
    feats:  torch.Tensor  # (n, D_in)
    in_S:   torch.Tensor  # (n,) bool

    D_IN = 6


def compute_vertex_features(
    points: np.ndarray,        # (n, 2) float32 — normalized polygon coords
    state: list[int],          # current guard set S_t
    vis_matrix: np.ndarray,    # (n, M) bool — disc_vis cache
    device: torch.device | str = "cpu",
) -> VertexFeatures:
    """Build the 6-dim per-vertex feature tensor.

    Features per vertex v:
        0: x          (already in [0, 1])
        1: y          (already in [0, 1])
        2: in_S       (1 if v in S, else 0)
        3: vis_frac   (vis count of v / M)         — how much v sees
        4: marg_cov   (new samples v would cover if added; 0 if v in S)
        5: redundancy (for v in S: fraction of v's vis already covered by
                       others in S; for v not in S: 0)
    """
    n, M = vis_matrix.shape
    in_S = np.zeros(n, dtype=np.bool_)
    in_S[list(state)] = True

    # Coverage by S excluding each vertex.
    if state:
        S_arr = np.array(state, dtype=np.int64)
        gc_S = vis_matrix[S_arr].astype(np.int32).sum(axis=0)  # (M,)
    else:
        gc_S = np.zeros(M, dtype=np.int32)
    covered_by_S = gc_S > 0

    vis_frac = vis_matrix.sum(axis=1).astype(np.float32) / max(1, M)

    # Marginal coverage (new samples covered if v added).
    new_samples = (~covered_by_S[None, :]) & vis_matrix       # (n, M)
    marg_cov = new_samples.sum(axis=1).astype(np.float32) / max(1, M)
    marg_cov = np.where(in_S, 0.0, marg_cov)

    # Redundancy for v in S: fraction of v's covered samples that have
    # gc_S > 1 (i.e. covered by at least one other guard). Vectorised over
    # all members of S simultaneously: (|S|, M) elementwise ops.
    redundancy = np.zeros(n, dtype=np.float32)
    if state:
        S_arr = np.array(list(state), dtype=np.int64)
        v_cov = vis_matrix[S_arr]                                    # (|S|, M) bool
        v_cov_int = v_cov.astype(np.int32)
        v_cov_total = v_cov_int.sum(axis=1)                          # (|S|,)
        others_count = gc_S[None, :] - v_cov_int                     # (|S|, M)
        others_mask = others_count > 0                               # (|S|, M)
        overlap = (v_cov & others_mask).sum(axis=1).astype(np.float32)  # (|S|,)
        denom = np.maximum(v_cov_total, 1)
        redundancy[S_arr] = (overlap / denom).astype(np.float32)

    feats = np.stack([
        points[:, 0],
        points[:, 1],
        in_S.astype(np.float32),
        vis_frac,
        marg_cov,
        redundancy,
    ], axis=1).astype(np.float32)

    return VertexFeatures(
        feats=torch.from_numpy(feats).to(device),
        in_S=torch.from_numpy(in_S).to(device),
    )


# ──────────────────────────────────────────────────────────────────────
#  Editor model
# ──────────────────────────────────────────────────────────────────────
class _SelfAttnBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Linear(2 * dim, dim),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None
                ) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x


class EditHead(nn.Module):
    """Lightweight set-equivariant editor.

    Forward inputs (single example):
        feats:    (n, D_in)
        in_S:     (n,) bool

    Forward inputs (batched, padded to L vertices):
        feats:    (B, L, D_in)
        in_S:     (B, L) bool
        pad_mask: (B, L) bool — True for padded positions

    Returns dict of logits (real-valued, raw, masked-out positions are
    set to -inf so they don't compete with valid positions).
    """

    D_IN = VertexFeatures.D_IN

    def __init__(self, hidden: int = 64, n_attn_layers: int = 1,
                 heads: int = 4):
        super().__init__()
        H = hidden
        self.embed = nn.Linear(self.D_IN, H)
        self.attn_blocks = nn.ModuleList([
            _SelfAttnBlock(H, heads=heads) for _ in range(n_attn_layers)
        ])
        self.norm = nn.LayerNorm(H)

        # Per-vertex heads operate on (v_emb || S_ctx || G_ctx) ∈ ℝ^{3H}.
        def _vhead():
            return nn.Sequential(
                nn.Linear(3 * H, H),
                nn.GELU(),
                nn.Linear(H, 1),
            )
        self.remove_head = _vhead()
        self.swap_out_head = _vhead()
        self.swap_in_head = _vhead()

        # Global STOP head over (S_ctx || G_ctx).
        self.stop_head = nn.Sequential(
            nn.Linear(2 * H, H),
            nn.GELU(),
            nn.Linear(H, 1),
        )

    def _pool_masked(self, x: torch.Tensor, mask: torch.Tensor
                     ) -> torch.Tensor:
        """Mean-pool along dim=1 (sequence), ignoring positions where
        mask=False. Returns (B, H). If a row has no True, returns zeros."""
        m = mask.unsqueeze(-1).to(x.dtype)
        s = (x * m).sum(dim=1)
        cnt = m.sum(dim=1).clamp(min=1.0)
        return s / cnt

    def forward(self, feats: torch.Tensor, in_S: torch.Tensor,
                pad_mask: torch.Tensor | None = None) -> dict:
        """
        feats:    (B, L, D_in)
        in_S:     (B, L) bool — True for vertices currently in the set
        pad_mask: (B, L) bool — True for padded (invalid) positions
        """
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)
            in_S = in_S.unsqueeze(0)
            if pad_mask is not None:
                pad_mask = pad_mask.unsqueeze(0)

        B, L, _ = feats.shape
        if pad_mask is None:
            pad_mask = torch.zeros(B, L, dtype=torch.bool, device=feats.device)

        h = self.embed(feats)
        for blk in self.attn_blocks:
            h = blk(h, key_padding_mask=pad_mask)
        h = self.norm(h)

        valid = ~pad_mask
        in_S_eff = in_S & valid
        not_S_eff = (~in_S) & valid

        S_ctx = self._pool_masked(h, in_S_eff)        # (B, H)
        G_ctx = self._pool_masked(h, valid)           # (B, H)

        S_ctx_b = S_ctx.unsqueeze(1).expand(-1, L, -1)  # (B, L, H)
        G_ctx_b = G_ctx.unsqueeze(1).expand(-1, L, -1)
        h_full = torch.cat([h, S_ctx_b, G_ctx_b], dim=-1)  # (B, L, 3H)

        rm_logits = self.remove_head(h_full).squeeze(-1)         # (B, L)
        sw_out_logits = self.swap_out_head(h_full).squeeze(-1)
        sw_in_logits = self.swap_in_head(h_full).squeeze(-1)

        # Mask invalid positions to -inf so softmax/argmax ignore them.
        neg_inf = torch.finfo(rm_logits.dtype).min
        rm_logits = rm_logits.masked_fill(~in_S_eff, neg_inf)
        sw_out_logits = sw_out_logits.masked_fill(~in_S_eff, neg_inf)
        sw_in_logits = sw_in_logits.masked_fill(~not_S_eff, neg_inf)

        stop_logit = self.stop_head(
            torch.cat([S_ctx, G_ctx], dim=-1)
        ).squeeze(-1)                                            # (B,)

        return {
            "remove_logits":  rm_logits,
            "swap_out_logits": sw_out_logits,
            "swap_in_logits":  sw_in_logits,
            "stop_logit":      stop_logit,
        }

    @torch.no_grad()
    def predict(self, feats: torch.Tensor, in_S: torch.Tensor,
                pad_mask: torch.Tensor | None = None,
                stop_threshold: float = 0.5) -> dict:
        """Greedy decode of one edit step. Single example or batch.

        Returns:
            kind: list[str] of {"stop","remove","swap"} — len B
            remove_idx: (B,) int  (or -1)
            add_idx: (B,) int     (or -1)
        """
        out = self.forward(feats, in_S, pad_mask)
        p_stop = torch.sigmoid(out["stop_logit"])
        rm_argmax = out["remove_logits"].argmax(dim=-1)
        # Compare best REMOVE-only score vs best SWAP-pair score.
        best_remove = out["remove_logits"].max(dim=-1).values
        best_sw_out = out["swap_out_logits"].max(dim=-1).values
        best_sw_in_v = out["swap_in_logits"].max(dim=-1)
        best_swap = best_sw_out + best_sw_in_v.values
        sw_out_argmax = out["swap_out_logits"].argmax(dim=-1)
        sw_in_argmax = best_sw_in_v.indices

        B = feats.shape[0] if feats.dim() == 3 else 1
        kinds: list[str] = []
        rm_out = torch.full((B,), -1, dtype=torch.long, device=feats.device)
        ad_out = torch.full((B,), -1, dtype=torch.long, device=feats.device)
        for b in range(B):
            if p_stop[b].item() >= stop_threshold:
                kinds.append("stop")
                continue
            if best_remove[b] >= best_swap[b]:
                kinds.append("remove")
                rm_out[b] = rm_argmax[b]
            else:
                kinds.append("swap")
                rm_out[b] = sw_out_argmax[b]
                ad_out[b] = sw_in_argmax[b]
        return {"kind": kinds, "remove_idx": rm_out, "add_idx": ad_out,
                "p_stop": p_stop}

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ──────────────────────────────────────────────────────────────────────
#  Loss helpers
# ──────────────────────────────────────────────────────────────────────
def edit_loss(
    out: dict,
    target: dict,
    *,
    w_stop: float = 1.0,
    w_action: float = 1.0,
    stop_pos_weight: float | None = None,
) -> dict:
    """Cross-entropy loss for one (state, action) example.

    target keys (per batch element):
        kind:       "stop" | "remove" | "swap"
        remove_idx: int (vertex removed when kind in {"remove","swap"})
        add_idx:    int (vertex added when kind == "swap")

    Loss components:
        L_stop   = BCE(stop_logit, kind == "stop")
        if kind == "remove": L_action = CE(remove_logits,  remove_idx)
        if kind == "swap":   L_action = CE(swap_out_logits,remove_idx)
                                       + CE(swap_in_logits, add_idx)
        if kind == "stop":   L_action = 0
    """
    device = out["stop_logit"].device
    is_stop = torch.tensor(
        [k == "stop" for k in target["kind"]],
        device=device, dtype=torch.float32,
    )
    pw = (
        torch.tensor(stop_pos_weight, device=device, dtype=out["stop_logit"].dtype)
        if stop_pos_weight is not None else None
    )
    L_stop = F.binary_cross_entropy_with_logits(
        out["stop_logit"], is_stop, reduction="mean", pos_weight=pw,
    )

    rm_idx = target["remove_idx"].to(device)
    ad_idx = target["add_idx"].to(device)

    # Compute per-example action losses, masking by kind.
    B = out["remove_logits"].shape[0]
    L_action_sum = torch.zeros((), device=device)
    n_action = 0
    for b in range(B):
        k = target["kind"][b]
        if k == "stop":
            continue
        if k == "remove":
            L_action_sum = L_action_sum + F.cross_entropy(
                out["remove_logits"][b:b+1], rm_idx[b:b+1],
            )
            n_action += 1
        elif k == "swap":
            L_action_sum = L_action_sum + F.cross_entropy(
                out["swap_out_logits"][b:b+1], rm_idx[b:b+1],
            )
            L_action_sum = L_action_sum + F.cross_entropy(
                out["swap_in_logits"][b:b+1], ad_idx[b:b+1],
            )
            n_action += 1
    L_action = L_action_sum / max(1, n_action)

    L = w_stop * L_stop + w_action * L_action
    return {"loss": L, "L_stop": L_stop.detach(), "L_action": L_action.detach()}
