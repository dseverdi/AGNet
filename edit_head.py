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
    D_IN_GEO_FREE = 3  # (x, y, in_S) only — no visibility oracle


def compute_vertex_features(
    points: np.ndarray,        # (n, 2) float32 — normalized polygon coords
    state: list[int],          # current guard set S_t
    vis_matrix: np.ndarray,    # (n, M) bool — disc_vis cache
    device: torch.device | str = "cpu",
) -> VertexFeatures:
    """Build the 6-dim per-vertex feature tensor (oracle-assisted).

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


def compute_vertex_features_geo_free(
    points: np.ndarray,        # (n, 2) float32 — normalized polygon coords
    state: list[int],          # current guard set S_t
    device: torch.device | str = "cpu",
    topology: bool = False,
) -> VertexFeatures:
    """Build a per-vertex feature tensor WITHOUT any visibility oracle.

    Two variants:
      • ``topology=False`` (default, 3-dim):
          0: x          (normalised polygon coordinate)
          1: y
          2: in_S       (1 if v in current guard set, else 0)
      • ``topology=True`` (8-dim):
          0–2:   x, y, in_S (as above)
          3:     pos_norm = vertex_index / n    (cyclic position in the
                 boundary; gives the model knowledge that vertices form a
                 closed cycle)
          4–5:   (dx_prev, dy_prev) — vector to previous boundary neighbour
          6–7:   (dx_next, dy_next) — vector to next boundary neighbour

    The topology variant gives the model the polygon's *shape* (edges and
    cyclic order) without giving it visibility. Visibility must still be
    inferred — that is the research question.
    """
    n = len(points)
    in_S = np.zeros(n, dtype=np.bool_)
    in_S[list(state)] = True

    cols = [points[:, 0], points[:, 1], in_S.astype(np.float32)]
    if topology:
        idx = np.arange(n, dtype=np.float32)
        pos_norm = idx / max(1.0, float(n))
        prev_pts = points[(np.arange(n) - 1) % n]
        next_pts = points[(np.arange(n) + 1) % n]
        dx_prev = (points[:, 0] - prev_pts[:, 0]).astype(np.float32)
        dy_prev = (points[:, 1] - prev_pts[:, 1]).astype(np.float32)
        dx_next = (next_pts[:, 0] - points[:, 0]).astype(np.float32)
        dy_next = (next_pts[:, 1] - points[:, 1]).astype(np.float32)
        cols.extend([pos_norm, dx_prev, dy_prev, dx_next, dy_next])

    feats = np.stack(cols, axis=1).astype(np.float32)

    return VertexFeatures(
        feats=torch.from_numpy(feats).to(device),
        in_S=torch.from_numpy(in_S).to(device),
    )


def compute_visibility_targets(
    points: np.ndarray,
    state: list[int],
    vis_matrix: np.ndarray,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Per-vertex visibility targets for the auxiliary loss.

    Returns ``(n, 3)`` tensor of ``[vis_frac, marg_cov, redundancy]`` — same
    quantities the oracle feature builder produces, but used as *supervision
    targets* for the auxiliary head rather than model inputs. This is what
    "encode visibility in latent space" means concretely: latents are
    trained to be linearly predictive of these targets.

    Uses disc_vis (allowed during training).
    """
    n, M = vis_matrix.shape
    in_S = np.zeros(n, dtype=np.bool_)
    in_S[list(state)] = True

    vis_frac = vis_matrix.sum(axis=1).astype(np.float32) / max(1, M)

    if state:
        S_arr = np.array(state, dtype=np.int64)
        gc_S = vis_matrix[S_arr].astype(np.int32).sum(axis=0)
    else:
        gc_S = np.zeros(M, dtype=np.int32)
    covered_by_S = gc_S > 0
    new_samples = (~covered_by_S[None, :]) & vis_matrix
    marg_cov = new_samples.sum(axis=1).astype(np.float32) / max(1, M)
    marg_cov = np.where(in_S, 0.0, marg_cov)

    redundancy = np.zeros(n, dtype=np.float32)
    if state:
        S_arr = np.array(state, dtype=np.int64)
        v_cov = vis_matrix[S_arr]
        v_cov_int = v_cov.astype(np.int32)
        v_cov_total = v_cov_int.sum(axis=1)
        others_count = gc_S[None, :] - v_cov_int
        others_mask = others_count > 0
        overlap = (v_cov & others_mask).sum(axis=1).astype(np.float32)
        redundancy[S_arr] = (overlap / np.maximum(v_cov_total, 1)).astype(np.float32)

    targets = np.stack([vis_frac, marg_cov, redundancy], axis=1).astype(np.float32)
    return torch.from_numpy(targets).to(device)


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
                 heads: int = 4, d_in: int | None = None,
                 aux_visibility: bool = False):
        """d_in overrides the class-level D_IN (default 6).
        Use d_in=3 for the geometry-free (no visibility oracle) variant,
        d_in=8 for the geometry-free + polygon-topology variant.

        aux_visibility: if True, instantiate an auxiliary head that predicts
        (vis_frac, marg_cov, redundancy) per vertex from the post-attention
        latents. The head is supervised at training time with disc_vis-derived
        targets and discarded at inference — it exists only to push visibility
        information into the shared latents."""
        super().__init__()
        H = hidden
        self._d_in = d_in if d_in is not None else self.D_IN
        self._aux_visibility = aux_visibility
        self.embed = nn.Linear(self._d_in, H)
        self.attn_blocks = nn.ModuleList([
            _SelfAttnBlock(H, heads=heads) for _ in range(n_attn_layers)
        ])
        self.norm = nn.LayerNorm(H)

        if aux_visibility:
            # Predicts [vis_frac, marg_cov, redundancy] per vertex from latents.
            self.aux_head = nn.Sequential(
                nn.Linear(H, H),
                nn.GELU(),
                nn.Linear(H, 3),
            )
        else:
            self.aux_head = None

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

        out = {
            "remove_logits":  rm_logits,
            "swap_out_logits": sw_out_logits,
            "swap_in_logits":  sw_in_logits,
            "stop_logit":      stop_logit,
        }
        if self.aux_head is not None:
            # Predicts (vis_frac, marg_cov, redundancy) per vertex.
            # Per-vertex latent h (post-attention, post-norm) is the same
            # representation feeding the action heads — supervising it to
            # be predictive of visibility pushes those quantities into the
            # shared latent space.
            out["aux_vis"] = self.aux_head(h)                    # (B, L, 3)
        return out

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
    aux_target: torch.Tensor | None = None,
    aux_pad_mask: torch.Tensor | None = None,
    w_aux: float = 0.5,
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
    out_dict = {"loss": L, "L_stop": L_stop.detach(), "L_action": L_action.detach()}

    # Auxiliary visibility-prediction loss (training only). Targets are
    # (B, L, 3) per-vertex (vis_frac, marg_cov, redundancy); pad_mask
    # excludes padded vertices. The loss pushes the editor's latents to
    # encode visibility information so the action heads can read it
    # implicitly at inference (where no targets are available).
    if "aux_vis" in out and aux_target is not None:
        pred = out["aux_vis"]                                # (B, L, 3)
        if aux_pad_mask is not None:
            valid = (~aux_pad_mask).to(pred.dtype).unsqueeze(-1)
            diff = (pred - aux_target) * valid
            denom = valid.sum().clamp(min=1.0) * 3
            L_aux = (diff * diff).sum() / denom
        else:
            L_aux = F.mse_loss(pred, aux_target)
        L = L + w_aux * L_aux
        out_dict["loss"] = L
        out_dict["L_aux"] = L_aux.detach()
    return out_dict
