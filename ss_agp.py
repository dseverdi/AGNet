"""ss_agp.py

Solution-Sampler (SS) model for the Art Gallery Problem (AGP), specialized to vertex guarding
in simple polygons.

Implements a neural combinatorial optimization (NCO) sampler aligned with Caramanis et al.
(2023) "Optimizing Solution-Samplers for Combinatorial Problems: The Landscape of
Policy-Gradient Methods".

Core formulation:
- Input instance I: a simple polygon with n vertices (np.ndarray [n,2]).
- Triangulate the polygon into n-2 triangles.
- Build a vertex-visibility graph G_V among polygon vertices.
- Build a hypergraph hitting-set instance where each triangle i yields a hyperedge G_i
  (vertices that can guard that triangle), with incidence matrix M in {0,1}^{T x n}.
- Sample a guard set S via a pointer-style policy; optimize with REINFORCE + entropy
  regularization and a fast/slow (learned/uniform) mixture to avoid premature collapse
  ("Pareto traps") into over-guarding.

This file is intentionally self-contained, but it reuses existing project utilities when possible:
- Coverage evaluation (geometric) via utils.evaluate_polygon_visibility_numpy_wo_gt if skgeom is installed.

Usage (smoke):
  python ss_agp.py --smoke

Usage (train small, synthetic):
    python ss_agp.py --num-train 200 --epochs 5 --min-n 12 --max-n 24

Usage (train on your existing AGP .pol dataset, like rl_agp.py):
    python ss_agp.py --agp_train_dir /path/to/train_dir --normalize --epochs 30
    python ss_agp.py --agp_train_dir /path/to/single.pol --normalize --epochs 5

By default evaluation uses dataset polygons (dev if available). Synthetic comb evaluation is optional.
"""

from __future__ import annotations

import argparse
import math
import random
import os
import json
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Reuse the project's AGP .pol parsing utilities.
    from dataset import Dataset as _BaseDataset, agp_read_samples
except Exception:  # pragma: no cover
    _BaseDataset = None
    agp_read_samples = None

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None

try:
    from shapely.geometry import LineString, Polygon as ShapelyPolygon
except Exception:  # pragma: no cover
    LineString = None
    ShapelyPolygon = None

try:
    from shapely.prepared import prep as _shapely_prep
except Exception:  # pragma: no cover
    _shapely_prep = None

try:
    from utils import evaluate_polygon_visibility_numpy_wo_gt
except Exception:  # pragma: no cover
    evaluate_polygon_visibility_numpy_wo_gt = None


# ----------------------------
# Geometry / triangulation
# ----------------------------

def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _polygon_signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ensure_ccw(points: np.ndarray) -> np.ndarray:
    if _polygon_signed_area(points) < 0:
        return points[::-1].copy()
    return points


def _point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-12) -> bool:
    # Barycentric sign checks (works for CCW triangles)
    v0 = c - a
    v1 = b - a
    v2 = p - a

    den = _cross(v1, v0)
    if abs(den) < eps:
        return False
    u = _cross(v1, v2) / den
    v = _cross(v2, v0) / den
    return (u >= -eps) and (v >= -eps) and (u + v <= 1.0 + eps)


def triangulate_polygon(points: np.ndarray) -> List[Tuple[int, int, int]]:
    """Ear-clipping triangulation for a simple polygon.

    Returns triangles as tuples of vertex indices into `points`.
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    if n < 3:
        return []

    pts = _ensure_ccw(points)

    # Map back to original indices if we reversed.
    if pts is not points and not np.shares_memory(pts, points):
        # We reversed by copy; detect reversal by signed area sign.
        # If reversed, index map is n-1-i.
        if _polygon_signed_area(points) < 0:
            idx_map = list(range(n - 1, -1, -1))
        else:
            idx_map = list(range(n))
    else:
        idx_map = list(range(n))

    remaining = list(range(n))  # indices into pts
    triangles: List[Tuple[int, int, int]] = []

    def is_convex(i_prev: int, i: int, i_next: int) -> bool:
        a = pts[i_prev]
        b = pts[i]
        c = pts[i_next]
        return _cross(c - b, a - b) < 0  # because pts is CCW: convex if turn is left; using reversed order

    # For CCW polygon, convex at vertex b if cross((c-b),(a-b)) > 0.
    # Above uses < 0 because of that ordering; safer explicit:
    def is_convex_ccw(i_prev: int, i: int, i_next: int) -> bool:
        a = pts[i_prev]
        b = pts[i]
        c = pts[i_next]
        return _cross(b - a, c - b) > 0

    max_iters = n * n
    iters = 0
    while len(remaining) > 3 and iters < max_iters:
        ear_found = False
        m = len(remaining)
        for k in range(m):
            i_prev = remaining[(k - 1) % m]
            i = remaining[k]
            i_next = remaining[(k + 1) % m]

            if not is_convex_ccw(i_prev, i, i_next):
                continue

            a = pts[i_prev]
            b = pts[i]
            c = pts[i_next]
            # Check if any other point lies inside this ear triangle.
            contains_other = False
            for j in remaining:
                if j in (i_prev, i, i_next):
                    continue
                if _point_in_triangle(pts[j], a, b, c):
                    contains_other = True
                    break
            if contains_other:
                continue

            triangles.append((idx_map[i_prev], idx_map[i], idx_map[i_next]))
            remaining.pop(k)
            ear_found = True
            break

        if not ear_found:
            # Fallback: if ear clipping stalls (degenerate), stop.
            break
        iters += 1

    if len(remaining) == 3:
        i_prev, i, i_next = remaining
        triangles.append((idx_map[i_prev], idx_map[i], idx_map[i_next]))

    return triangles


# ----------------------------
# Visibility graph
# ----------------------------

def _segment_intersects_proper(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, eps: float = 1e-12) -> bool:
    """Proper intersection excluding shared endpoints."""

    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return _cross(q - p, r - p)

    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        # q on pr
        if abs(orient(p, q, r)) > eps:
            return False
        return (
            min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
            and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    # General case
    if (o1 * o2 < -eps) and (o3 * o4 < -eps):
        return True

    # Collinear cases
    if abs(o1) <= eps and on_segment(a, c, b):
        return True
    if abs(o2) <= eps and on_segment(a, d, b):
        return True
    if abs(o3) <= eps and on_segment(c, a, d):
        return True
    if abs(o4) <= eps and on_segment(c, b, d):
        return True

    return False


def _segment_in_polygon_fallback(points: np.ndarray, i: int, j: int) -> bool:
    """Visibility check without shapely: segment must not cross polygon edges, and its midpoint must lie inside."""
    n = len(points)
    a = points[i]
    b = points[j]

    # Adjacent vertices are always visible along boundary.
    if (i - j) % n == 1 or (j - i) % n == 1:
        return True

    # Reject if segment properly intersects any polygon edge (excluding incident edges).
    for k in range(n):
        k2 = (k + 1) % n
        # Skip edges incident to endpoints.
        if k in (i, j) or k2 in (i, j):
            continue
        c = points[k]
        d = points[k2]
        if _segment_intersects_proper(a, b, c, d):
            return False

    # Midpoint inside polygon test via winding number / ray casting.
    mid = 0.5 * (a + b)
    x, y = float(mid[0]), float(mid[1])
    inside = False
    for k in range(n):
        x1, y1 = float(points[k][0]), float(points[k][1])
        x2, y2 = float(points[(k + 1) % n][0]), float(points[(k + 1) % n][1])
        cond = (y1 > y) != (y2 > y)
        if cond:
            x_int = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-18) + x1
            if x < x_int:
                inside = not inside
    return inside


def compute_visibility_graph(points: np.ndarray) -> np.ndarray:
    """Compute mutual vertex visibility adjacency matrix (n x n)."""
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=bool)

    adj = np.zeros((n, n), dtype=bool)
    np.fill_diagonal(adj, True)

    poly = None
    poly_prepared = None
    if ShapelyPolygon is not None:
        try:
            poly = ShapelyPolygon(pts)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly is not None and _shapely_prep is not None:
                poly_prepared = _shapely_prep(poly)
        except Exception:
            poly = None
            poly_prepared = None

    for i in range(n):
        for j in range(i + 1, n):
            visible = False
            if poly is not None and LineString is not None:
                try:
                    seg = LineString([tuple(pts[i]), tuple(pts[j])])
                    # covers allows boundary visibility.
                    if poly_prepared is not None:
                        visible = bool(poly_prepared.covers(seg))
                    else:
                        visible = bool(poly.covers(seg))
                except Exception:
                    visible = _segment_in_polygon_fallback(pts, i, j)
            else:
                visible = _segment_in_polygon_fallback(pts, i, j)
            adj[i, j] = visible
            adj[j, i] = visible

    return adj


# ----------------------------
# Hypergraph instance builder
# ----------------------------

def build_hypergraph(triangles: Sequence[Tuple[int, int, int]], vis_adj: np.ndarray) -> np.ndarray:
    """Build incidence matrix M (T x n) where M[t, v]=1 if vertex v guards triangle t."""
    n = vis_adj.shape[0]
    T = len(triangles)
    M = np.zeros((T, n), dtype=np.int8)
    for t, (a, b, c) in enumerate(triangles):
        eligible = vis_adj[a] & vis_adj[b] & vis_adj[c]
        M[t, eligible] = 1
    return M


@dataclass
class HypergraphInstance:
    points: np.ndarray  # (n,2)
    triangles: List[Tuple[int, int, int]]
    vis_adj: np.ndarray  # (n,n) bool
    incidence: np.ndarray  # (T,n) int8


# ----------------------------
# Objective / oracle
# ----------------------------


class BilinearOracle:
    """Bilinear-ish scalarized objective for hitting-set coverage + sparsity."""

    def __init__(
        self,
        lambda_card: float = 0.05,
        penalty_weight: float = 10.0,
        penalty_power: float = 1.0,
        coverage_gate: float = 0.0,
    ):
        self.lambda_card = float(lambda_card)
        self.penalty_weight = float(penalty_weight)
        self.penalty_power = float(penalty_power)
        # Coverage-gating: if > 0, use gated cost instead of additive
        self.coverage_gate = float(coverage_gate)

    def oracle_value(self, s: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """O(s;I) = lambda*|s| + sum_i relu(1 - (M s)_i). Lower is better."""
        # s: [n] in {0,1}
        # M: [T,n] in {0,1}
        Ms = torch.matmul(M.float(), s.float())  # [T]
        penalties = F.relu(1.0 - Ms)
        if self.penalty_power != 1.0:
            penalties = penalties.clamp_min(0.0) ** float(self.penalty_power)

        # Normalize by number of constraints (triangles) so penalty scale doesn't grow with n.
        # This avoids the degenerate "pick almost everything" policy when lambda_card is small.
        penalty_term = penalties.mean() if penalties.numel() > 0 else torch.tensor(0.0, device=penalties.device)
        return self.lambda_card * s.float().sum() + self.penalty_weight * penalty_term

    def gated_cost(self, s: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Coverage-gated cost: separates coverage and cardinality objectives.

        Uses soft gating with a small cardinality term always present:
        - Below gate: primarily coverage penalty + small cardinality term
        - Above gate: primarily cardinality penalty + small coverage term

        This ensures the model learns efficient coverage even before crossing the gate.
        """
        coverage = BilinearOracle.discrete_coverage(s, M)
        card = s.float().sum()
        n = float(M.shape[1])  # number of vertices for normalization
        gate = float(self.coverage_gate)

        if gate <= 0.0:
            # Fall back to standard additive cost
            return self.oracle_value(s, M)

        # Soft gating with sigmoid
        sharpness = 30.0  # steepness of transition (reduced from 50 for smoother gradient)
        gate_val = torch.sigmoid(sharpness * (coverage - gate))

        # Coverage penalty: large when coverage is low (scale by 100 to dominate below gate)
        cov_penalty = 100.0 * (1.0 - coverage)
        
        # Cardinality penalty: normalized by n so it's comparable across instances
        # Target: |S|/n around 0.1-0.2 for good solutions (OPT is typically ~15% of vertices)
        card_penalty = self.lambda_card * (card / n)

        # Always add a small cardinality term (10% weight) even in coverage mode
        # This teaches efficient coverage from the start
        min_card_weight = 0.1
        
        # Gated combination with minimum cardinality weight
        cov_weight = (1.0 - gate_val) * (1.0 - min_card_weight)
        card_weight = gate_val + (1.0 - gate_val) * min_card_weight

        return cov_weight * cov_penalty + card_weight * card_penalty

    @staticmethod
    def discrete_coverage(s: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        Ms = torch.matmul(M.float(), s.float())
        covered = (Ms > 0).float().mean()
        return covered

    def surrogate_reward(self, s: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """Smooth bi-criteria reward surrogate (maximize).

        Key property: `lambda_card` must affect training.

        reward = 100 * coverage^4 - (100 * lambda_card) * (|s|/n) * coverage
        """
        n = float(s.numel())
        coverage = BilinearOracle.discrete_coverage(s, M)
        size_ratio = s.float().sum() / max(1.0, n)
        size_weight = 100.0 * float(self.lambda_card)
        return 100.0 * (coverage ** 4) - size_weight * size_ratio * coverage


# ----------------------------
# Baselines
# ----------------------------

def greedy_hitting_set(M: np.ndarray) -> List[int]:
    """Simple greedy hitting set on incidence M (T x n)."""
    T, n = M.shape
    uncovered = np.ones(T, dtype=bool)
    chosen: List[int] = []
    remaining = set(range(n))

    while uncovered.any() and remaining:
        best_v = None
        best_gain = -1
        for v in remaining:
            gain = int(M[uncovered, v].sum())
            if gain > best_gain:
                best_gain = gain
                best_v = v
        if best_v is None or best_gain <= 0:
            break
        chosen.append(best_v)
        remaining.remove(best_v)
        uncovered = uncovered & (M[:, best_v] == 0)

    return chosen


def triangulation_three_color_guard_set(triangles: Sequence[Tuple[int, int, int]], n_vertices: int) -> List[int]:
    """Attempt a 3-coloring based baseline (Fisk) using NetworkX greedy coloring.

    For a valid 3-coloring of a triangulation, the smallest color class hits all triangles.
    Greedy coloring usually finds 3 colors for triangulation graphs, but this is not guaranteed.
    """
    if nx is None:
        return []

    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))
    for a, b, c in triangles:
        G.add_edge(a, b)
        G.add_edge(b, c)
        G.add_edge(a, c)

    coloring = nx.coloring.greedy_color(G, strategy="largest_first")
    # Group by color
    by_color: Dict[int, List[int]] = {}
    for v, col in coloring.items():
        by_color.setdefault(col, []).append(v)

    smallest = min(by_color.values(), key=len) if by_color else []
    return sorted(smallest)


# ----------------------------
# Pointer-style sampler
# ----------------------------


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.Wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, query: torch.Tensor, keys: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # query: [B,H], keys: [B,N,H], mask: [B,N] where True=masked
        q = self.Wq(query).unsqueeze(1)  # [B,1,H]
        k = self.Wk(keys)  # [B,N,H]
        scores = self.v(torch.tanh(q + k)).squeeze(-1)  # [B,N]
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        return scores


class SolutionSamplerPointerNet(nn.Module):
    """Pointer-style subset sampler with an explicit EOS action.

    Designed for variable n; operates on a batch by looping per instance to keep the code simple
    and robust for research.
    """

    def __init__(
        self,
        feature_dim: int = 3,
        embedding_size: int = 128,
        hidden_size: int = 128,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.embedding_size = int(embedding_size)
        self.hidden_size = int(hidden_size)
        self.temperature = float(temperature)

        self.embed = nn.Linear(self.feature_dim, self.embedding_size)
        self.encoder = nn.LSTM(self.embedding_size, self.hidden_size, batch_first=True)
        self.decoder_cell = nn.LSTMCell(self.embedding_size, self.hidden_size)

        self.attn = BahdanauAttention(self.hidden_size)
        self.eos_head = nn.Linear(self.hidden_size, 1)

        self.decoder_start = nn.Parameter(torch.zeros(self.embedding_size))

    def sample(
        self,
        feats: torch.Tensor,
        alpha_slow: float = 0.1,
        rho_slow: float = 0.1,
        max_steps: Optional[int] = None,
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """Sample a guard set for a single instance.

        Args:
            feats: [n, feature_dim]
            alpha_slow: mixture weight for uniform distribution (slow component)
            max_steps: cap on selections (default: n+1)

        Returns:
            chosen: list of selected vertex indices (no EOS)
            logp_sum: scalar tensor
            entropy_sum: scalar tensor
        """
        device = feats.device
        n = feats.size(0)

        x = self.embed(feats).unsqueeze(0)  # [1,n,E]
        enc_out, (h_n, c_n) = self.encoder(x)
        h = h_n.squeeze(0)
        c = c_n.squeeze(0)

        if max_steps is None:
            max_steps = int(n + 1)

        chosen: List[int] = []
        chosen_mask = torch.zeros(1, n, dtype=torch.bool, device=device)
        logp_sum = torch.tensor(0.0, device=device)
        reg_sum = torch.tensor(0.0, device=device)

        dec_inp = self.decoder_start.unsqueeze(0)
        for _ in range(max_steps):
            h, c = self.decoder_cell(dec_inp, (h, c))

            logits_vertices = self.attn(h, enc_out, mask=chosen_mask)  # [1,n]
            logits_vertices = logits_vertices / max(1e-6, self.temperature)
            logits_eos = self.eos_head(h).view(-1)  # [1]

            # Combine into action logits: [n + 1] where last is EOS
            logits = torch.cat([logits_vertices.squeeze(0), logits_eos], dim=0)  # [n+1]

            # Build availability mask: vertices not yet chosen + EOS
            avail = torch.cat([(~chosen_mask.squeeze(0)), torch.tensor([True], device=device)], dim=0)  # [n+1]

            # Softmax over all actions (masked)
            masked_logits = logits.clone()
            masked_logits[~avail] = float("-inf")
            probs_fast = F.softmax(masked_logits, dim=0)

            # Slow component (paper-style): softmax(rho * logits) with rho << 1 (more uniform-like)
            rho = float(rho_slow)
            rho = max(1e-6, rho)
            probs_slow = F.softmax(masked_logits * rho, dim=0)

            beta = float(alpha_slow)
            probs = (1.0 - beta) * probs_fast + beta * probs_slow
            probs = probs / probs.sum().clamp_min(1e-12)

            # Caramanis-style negative entropy regularizer:
            # H(W) := E_{a~pi}[log pi(a)] = sum_a pi(a) log pi(a)
            neg_ent_fast = (probs_fast * probs_fast.clamp_min(1e-12).log()).sum()
            neg_ent_slow = (probs_slow * probs_slow.clamp_min(1e-12).log()).sum()
            reg = (1.0 - beta) * neg_ent_fast + (beta / rho) * neg_ent_slow

            action = torch.multinomial(probs, 1).item()
            logp_sum = logp_sum + probs[action].clamp_min(1e-12).log()
            reg_sum = reg_sum + reg

            if action == n:
                break  # EOS

            chosen.append(int(action))
            # Avoid in-place mutation on tensors referenced by autograd graphs.
            chosen_mask = chosen_mask.clone()
            chosen_mask[0, action] = True
            dec_inp = x[0, action, :].unsqueeze(0)

        return chosen, logp_sum, reg_sum

    def forward(
        self,
        feats_batch: torch.Tensor,
        lengths: Sequence[int],
        alpha_slow: float = 0.1,
        rho_slow: float = 0.1,
    ) -> Tuple[List[List[int]], torch.Tensor, torch.Tensor]:
        """Sample for a batch (loops per instance for simplicity)."""
        all_chosen: List[List[int]] = []
        logps: List[torch.Tensor] = []
        ents: List[torch.Tensor] = []
        for b, n in enumerate(lengths):
            feats = feats_batch[b, :n, :]
            chosen, logp, reg = self.sample(feats, alpha_slow=alpha_slow, rho_slow=rho_slow)
            all_chosen.append(chosen)
            logps.append(logp)
            ents.append(reg)
        return all_chosen, torch.stack(logps), torch.stack(ents)


# ----------------------------
# Dataset
# ----------------------------


def _generate_star_shaped_polygon(n: int, radius: float = 1.0, noise: float = 0.25) -> np.ndarray:
    """Generate a random simple polygon by sampling points in polar coordinates.

    This produces a simple (non-self-intersecting) polygon by construction: points are ordered
    by increasing angle.
    """
    angles = np.sort(np.random.rand(n) * 2 * np.pi)
    radii = radius * (1.0 + noise * (2 * np.random.rand(n) - 1))
    radii = np.clip(radii, 0.05 * radius, None)
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    pts = np.stack([x, y], axis=1)
    # Translate to positive quadrant-ish and scale to [0,1]
    pts = pts - pts.min(axis=0, keepdims=True)
    denom = pts.max(axis=0, keepdims=True) - pts.min(axis=0, keepdims=True)
    denom[denom == 0] = 1.0
    pts = pts / denom
    return pts.astype(np.float64)


def _generate_comb_polygon(n_teeth: int = 6) -> np.ndarray:
    """Simple orthogonal comb-like polygon (common hard-ish AGP instance)."""
    n_teeth = max(2, int(n_teeth))
    pts = [(0.0, 0.0), (1.0, 0.0)]
    # Build the top boundary with teeth.
    x = 1.0
    y0 = 0.0
    y1 = 1.0
    tooth_w = 1.0 / (2 * n_teeth)
    for t in range(n_teeth):
        pts.append((x, y1))
        x -= tooth_w
        pts.append((x, y1))
        pts.append((x, 0.4))
        x -= tooth_w
        pts.append((x, 0.4))
    pts.append((0.0, y1))
    pts.append((0.0, 0.0))
    pts = np.array(pts[:-1], dtype=np.float64)
    # Normalize
    pts = pts - pts.min(axis=0, keepdims=True)
    denom = pts.max(axis=0, keepdims=True) - pts.min(axis=0, keepdims=True)
    denom[denom == 0] = 1.0
    pts = pts / denom
    return pts


class AGPDataset(torch.utils.data.Dataset):
    """AGP dataset that generates random simple polygons and (optionally) caches hypergraph instances."""

    def __init__(
        self,
        num_samples: int = 1000,
        min_n: int = 12,
        max_n: int = 30,
        seed: int = 0,
        cache_instances: bool = True,
    ):
        super().__init__()
        self.num_samples = int(num_samples)
        self.min_n = int(min_n)
        self.max_n = int(max_n)
        self.seed = int(seed)
        self.cache_instances = bool(cache_instances)

        self._rng = np.random.RandomState(self.seed)
        self._polygons: List[np.ndarray] = []
        for _ in range(self.num_samples):
            n = int(self._rng.randint(self.min_n, self.max_n + 1))
            # Use global np.random for vectorized generation, but reseed per sample for determinism.
            state = np.random.get_state()
            np.random.seed(self._rng.randint(0, 2**31 - 1))
            poly = _generate_star_shaped_polygon(n=n, radius=1.0, noise=0.35)
            np.random.set_state(state)
            self._polygons.append(poly)

        self._cache: Dict[int, HypergraphInstance] = {}

    def __len__(self) -> int:
        return self.num_samples

    def get_instance(self, idx: int) -> HypergraphInstance:
        if self.cache_instances and idx in self._cache:
            return self._cache[idx]

        pts = self._polygons[idx]
        triangles = triangulate_polygon(pts)
        vis = compute_visibility_graph(pts)
        M = build_hypergraph(triangles, vis)
        inst = HypergraphInstance(points=pts, triangles=triangles, vis_adj=vis, incidence=M)
        if self.cache_instances:
            self._cache[idx] = inst
        return inst

    def __getitem__(self, idx: int):
        inst = self.get_instance(idx)
        pts = inst.points
        n = pts.shape[0]
        deg = inst.vis_adj.sum(axis=1).astype(np.float32) / max(1.0, float(n))
        feats = np.concatenate([pts.astype(np.float32), deg[:, None]], axis=1)  # [n,3]
        return {
            "points": torch.tensor(pts, dtype=torch.float32),
            "feats": torch.tensor(feats, dtype=torch.float32),
            "incidence": torch.tensor(inst.incidence, dtype=torch.int8),
            "triangles": inst.triangles,
            "name": f"rand_{idx}",
        }


class AGPPolDataset(torch.utils.data.Dataset):
    """Dataset backed by existing AGP .pol files, using dataset.agp_read_samples.

    This keeps the SS training loop "organic" with the rest of the project: same parsing,
    same names, and easy drop-in replacement for synthetic polygons.
    """

    def __init__(
        self,
        pol_paths: Sequence[str],
        normalize: bool = True,
        cache_instances: bool = True,
    ):
        if _BaseDataset is None or agp_read_samples is None:
            raise ImportError("Could not import dataset.agp_read_samples / dataset.Dataset")

        self.normalize = bool(normalize)
        self.cache_instances = bool(cache_instances)
        samples = agp_read_samples(list(pol_paths), normalize=self.normalize)
        self._base = _BaseDataset(samples)
        self._cache: Dict[int, HypergraphInstance] = {}

    def __len__(self) -> int:
        return len(self._base)

    def get_instance(self, idx: int) -> HypergraphInstance:
        if self.cache_instances and idx in self._cache:
            return self._cache[idx]

        points_tensor, _label, _name = self._base[idx]
        pts = points_tensor.detach().cpu().numpy().astype(np.float64)
        triangles = triangulate_polygon(pts)
        vis = compute_visibility_graph(pts)
        M = build_hypergraph(triangles, vis)
        inst = HypergraphInstance(points=pts, triangles=triangles, vis_adj=vis, incidence=M)
        if self.cache_instances:
            self._cache[idx] = inst
        return inst

    def __getitem__(self, idx: int):
        inst = self.get_instance(idx)
        points_tensor, _label, name = self._base[idx]

        pts = inst.points
        n = pts.shape[0]
        deg = inst.vis_adj.sum(axis=1).astype(np.float32) / max(1.0, float(n))
        feats = np.concatenate([pts.astype(np.float32), deg[:, None]], axis=1)  # [n,3]
        return {
            "points": torch.tensor(pts, dtype=torch.float32),
            "feats": torch.tensor(feats, dtype=torch.float32),
            "incidence": torch.tensor(inst.incidence, dtype=torch.int8),
            "triangles": inst.triangles,
            "name": str(name),
        }


def collate_instances(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, List[int], List[torch.Tensor], List[str]]:
    feats = [b["feats"] for b in batch]
    points = [b["points"] for b in batch]
    incidences = [b["incidence"] for b in batch]
    names = [b["name"] for b in batch]

    lengths = [int(f.shape[0]) for f in feats]
    feats_pad = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True, padding_value=0.0)
    points_pad = torch.nn.utils.rnn.pad_sequence(points, batch_first=True, padding_value=0.0)
    return feats_pad, points_pad, lengths, incidences, names


def _get_pol_files(path: str) -> List[str]:
    """Mirror rl_agp.prepare_datasets behavior: accept dir of .pol or single .pol."""
    if path is None:
        return []
    if os.path.isdir(path):
        return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pol')]
    if os.path.isfile(path) and path.endswith('.pol'):
        return [path]
    raise ValueError(f"Provided path {path} is neither a .pol file nor a directory containing .pol files.")


def _infer_solution_dir(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    if os.path.isdir(path):
        return path
    if os.path.isfile(path):
        return os.path.dirname(path)
    return None


# ----------------------------
# Training
# ----------------------------


def policy_gradient_loss(
    log_probs: torch.Tensor,
    costs: torch.Tensor,
    neg_entropy: torch.Tensor,
    entropy_weight: float,
    baseline: float,
) -> torch.Tensor:
    # For minimization: grad E[cost] = E[(cost - b) * grad logp]
    adv = costs - float(baseline)
    return (log_probs * adv.detach()).mean() + float(entropy_weight) * neg_entropy.mean()


def _to_indicator(selected: List[int], n: int, device: torch.device) -> torch.Tensor:
    s = torch.zeros(n, device=device)
    if selected:
        s[torch.tensor(selected, dtype=torch.long, device=device)] = 1.0
    return s


def train_solution_sampler(
    model: SolutionSamplerPointerNet,
    dataset: AGPDataset,
    oracle: BilinearOracle,
    device: torch.device,
    epochs: int = 30,
    batch_size: int = 1,
    lr: float = 1e-3,
    beta_ema: float = 0.99,
    alpha_slow: float = 0.1,
    rho_slow: float = 0.1,
    entropy_weight: float = 0.05,
    entropy_decay_steps: int = 60,
    entropy_decay_rate: float = 0.5,
    lambda_card_start: float = 0.0,
    lambda_card_cov_threshold: float = 0.98,
    lambda_card_ramp_steps: int = 400,
    log_every: int = 50,
    use_greedy_baseline: bool = True,
):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_instances,
    )

    ema_baseline = 0.0

    # Curriculum for sparsity: keep lambda low until coverage is consistently high.
    lambda_final = float(oracle.lambda_card)
    lambda_start = float(lambda_card_start)
    lambda_cur = float(lambda_start)
    cov_ema = 0.0
    ramp_started = False
    ramp_step = 0

    global_step = 0

    for epoch in range(1, epochs + 1):
        epoch_costs: List[float] = []
        epoch_cov: List[float] = []
        epoch_size: List[float] = []

        t0 = time.time()
        running_cost = 0.0
        running_cov = 0.0
        running_size = 0.0
        running_batches = 0
        num_batches = len(loader) if hasattr(loader, "__len__") else None

        for batch_idx, (feats_pad, points_pad, lengths, incidences, _names) in enumerate(loader, start=1):
            feats_pad = feats_pad.to(device)

            # Update oracle lambda for this step (curriculum)
            oracle.lambda_card = float(lambda_cur)

            selections, logp, neg_ent = model(feats_pad, lengths=lengths, alpha_slow=alpha_slow, rho_slow=rho_slow)

            costs_list: List[torch.Tensor] = []
            cov_list: List[torch.Tensor] = []
            size_list: List[torch.Tensor] = []
            greedy_costs: List[float] = []

            for b, n in enumerate(lengths):
                M = incidences[b].to(device).float()  # [T,n]
                sel = selections[b]
                sel = [i for i in sel if 0 <= i < n]
                s = _to_indicator(sel, n=n, device=device)

                cov = oracle.discrete_coverage(s, M)
                # Use gated cost if coverage_gate > 0, otherwise standard oracle
                if oracle.coverage_gate > 0.0:
                    cost = oracle.gated_cost(s, M)
                else:
                    cost = oracle.oracle_value(s, M)

                if use_greedy_baseline:
                    greedy_sel = greedy_hitting_set(incidences[b].cpu().numpy())
                    sg = _to_indicator(greedy_sel, n=n, device=device)
                    # Use same cost function for greedy baseline
                    if oracle.coverage_gate > 0.0:
                        cg = float(oracle.gated_cost(sg, M).detach().cpu().item())
                    else:
                        cg = float(oracle.oracle_value(sg, M).detach().cpu().item())
                    greedy_costs.append(cg)
                else:
                    greedy_costs.append(0.0)

                costs_list.append(cost)
                cov_list.append(cov)
                size_list.append(s.sum() / max(1.0, float(n)))

            costs_t = torch.stack(costs_list)
            cov_t = torch.stack(cov_list)
            size_t = torch.stack(size_list)

            # Coverage EMA for curriculum
            mean_cov = float(cov_t.mean().detach().cpu().item())
            cov_ema = 0.9 * float(cov_ema) + 0.1 * float(mean_cov)
            if (
                (not ramp_started)
                and float(lambda_final) > float(lambda_start)
                and float(lambda_card_cov_threshold) > 0.0
                and float(cov_ema) >= float(lambda_card_cov_threshold)
            ):
                ramp_started = True
                ramp_step = 0
                print(
                    f"[curriculum] Start lambda ramp: cov_ema={cov_ema:.3f} >= {float(lambda_card_cov_threshold):.3f}",
                    flush=True,
                )

            if ramp_started and int(lambda_card_ramp_steps) > 0 and float(lambda_final) > float(lambda_start):
                ramp_step += 1
                frac = min(1.0, float(ramp_step) / float(lambda_card_ramp_steps))
                lambda_cur = float(lambda_start) + frac * (float(lambda_final) - float(lambda_start))
            else:
                lambda_cur = float(lambda_start)

            # Update EMA baseline on "cost after greedy subtraction".
            mean_cost = float(costs_t.mean().detach().cpu().item())
            mean_greedy = float(np.mean(greedy_costs)) if greedy_costs else 0.0
            centered = mean_cost - mean_greedy
            ema_baseline = beta_ema * ema_baseline + (1.0 - beta_ema) * centered

            baseline_total = ema_baseline + mean_greedy

            # Entropy regularization decay schedule (per instruction)
            if entropy_decay_steps is not None and int(entropy_decay_steps) > 0:
                decay_k = global_step // int(entropy_decay_steps)
            else:
                decay_k = 0
            lambda_t = float(entropy_weight) * (float(entropy_decay_rate) ** int(decay_k))

            loss = policy_gradient_loss(
                log_probs=logp,
                costs=costs_t,
                neg_entropy=neg_ent,
                entropy_weight=lambda_t,
                baseline=baseline_total,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            epoch_costs.append(float(costs_t.mean().detach().cpu().item()))
            epoch_cov.append(float(mean_cov))
            epoch_size.append(float(size_t.mean().detach().cpu().item()))

            running_cost += float(costs_t.mean().detach().cpu().item())
            running_cov += float(mean_cov)
            running_size += float(size_t.mean().detach().cpu().item())
            running_batches += 1

            if int(log_every) > 0 and (batch_idx % int(log_every) == 0):
                elapsed = time.time() - t0
                avg_cost = running_cost / max(1, running_batches)
                avg_cov = running_cov / max(1, running_batches)
                avg_size = running_size / max(1, running_batches)
                if num_batches is not None and num_batches > 0:
                    frac = float(batch_idx) / float(num_batches)
                    eta = (elapsed / max(1e-9, frac)) - elapsed
                    eta_s = f" ETA {eta:.0f}s"
                    denom_s = f"{batch_idx}/{num_batches}"
                else:
                    eta_s = ""
                    denom_s = f"{batch_idx}"

                print(
                    f"[train] epoch {epoch}/{epochs} batch {denom_s}"
                    f" | cost {avg_cost:.3f} | tri-cov {avg_cov:.3f} | |S|/n {avg_size:.3f}"
                    f" | lambda {lambda_cur:.4f} | ent-lambda {lambda_t:.4f}"
                    + eta_s,
                    flush=True,
                )

            global_step += 1

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"cost {np.mean(epoch_costs):.3f} | "
            f"tri-cov {np.mean(epoch_cov):.3f} | "
            f"|S|/n {np.mean(epoch_size):.3f} | "
            f"ema {ema_baseline:.3f} | "
            f"lambda {lambda_cur:.4f}"
        )


# ----------------------------
# Evaluation helpers
# ----------------------------


def evaluate_on_comb(
    model: SolutionSamplerPointerNet,
    oracle: BilinearOracle,
    device: torch.device,
    n_teeth_list: Sequence[int] = (4, 6, 8),
    alpha_slow: float = 0.0,
):
    model.eval()
    for n_teeth in n_teeth_list:
        pts = _generate_comb_polygon(n_teeth=n_teeth)
        triangles = triangulate_polygon(pts)
        vis = compute_visibility_graph(pts)
        M = build_hypergraph(triangles, vis)

        n = pts.shape[0]
        deg = vis.sum(axis=1).astype(np.float32) / max(1.0, float(n))
        feats = np.concatenate([pts.astype(np.float32), deg[:, None]], axis=1)

        feats_t = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
        chosen, logp, ent = model(feats_t, lengths=[n], alpha_slow=alpha_slow, rho_slow=0.1)
        sel = chosen[0]
        s = _to_indicator(sel, n=n, device=device)

        cov = float(BilinearOracle.discrete_coverage(s, torch.tensor(M, device=device).float()).cpu().item())
        rew = float(oracle.surrogate_reward(s, torch.tensor(M, device=device).float()).cpu().item())

        greedy_sel = greedy_hitting_set(M)
        s_g = _to_indicator(greedy_sel, n=n, device=device)
        cov_g = float(BilinearOracle.discrete_coverage(s_g, torch.tensor(M, device=device).float()).cpu().item())

        # Optional geometric coverage if skgeom is available
        geo_cov = None
        geo_cov_g = None
        if evaluate_polygon_visibility_numpy_wo_gt is not None:
            try:
                geo_cov = float(evaluate_polygon_visibility_numpy_wo_gt(pts.astype(np.float64), np.array(sel, dtype=int), f"comb_{n_teeth}"))
                geo_cov_g = float(evaluate_polygon_visibility_numpy_wo_gt(pts.astype(np.float64), np.array(greedy_sel, dtype=int), f"comb_{n_teeth}_greedy"))
            except Exception:
                geo_cov = None
                geo_cov_g = None

        print(
            f"Comb teeth={n_teeth} n={n:02d} | "
            f"SS: tri-cov={cov:.3f} |S|={len(sel)} rew={rew:.2f}"
            + (f" geo={geo_cov:.3f}" if geo_cov is not None else "")
            + f" || GreedyHS: tri-cov={cov_g:.3f} |S|={len(greedy_sel)}"
            + (f" geo={geo_cov_g:.3f}" if geo_cov_g is not None else "")
        )


def evaluate_on_dataset(
    model: SolutionSamplerPointerNet,
    oracle: BilinearOracle,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    k: int = -1,
    alpha_slow: float = 0.0,
    verbose: bool = False,
    sol_dir: Optional[str] = None,
):
    """Evaluate on k polygons from a dataset (AGPPolDataset or synthetic AGPDataset)."""
    model.eval()
    if k is None or int(k) <= 0:
        k = int(len(dataset))
    else:
        k = min(int(k), len(dataset))

    if k <= 0:
        print("No samples to evaluate.")
        return {
            "k": 0,
            "per_instance": [],
            "stats": {},
        }

    tri_covs: List[float] = []
    sizes: List[int] = []
    greedy_sizes: List[int] = []
    geo_covs: List[Optional[float]] = []
    geo_covs_greedy: List[Optional[float]] = []

    rows: List[Dict[str, str]] = []

    per_instance: List[Dict[str, object]] = []

    for i in range(k):
        item = dataset[i]
        pts = item["points"].detach().cpu().numpy().astype(np.float64)
        M_np = item["incidence"].detach().cpu().numpy().astype(np.int8)
        n = int(pts.shape[0])
        name = str(item.get("name", i))

        # Read optimal solution size for comparison (rl_agp.py convention: <name>.solution in sol_dir)
        opt_size: Optional[int] = None
        if sol_dir is not None:
            try:
                base_name = os.path.splitext(os.path.basename(name))[0]
                opt_sol_path = os.path.join(sol_dir, f"{base_name}.solution")
                with open(opt_sol_path, "r") as f:
                    lines = f.read().splitlines()
                if len(lines) >= 2:
                    true_indices = [int(x) for x in lines[1].split() if x.strip()]
                    if len(true_indices) > 0:
                        opt_size = int(len(true_indices))
            except Exception:
                opt_size = None

        # Features already computed by dataset
        feats = item["feats"].to(device).unsqueeze(0)

        chosen, _logp, _ent = model(feats, lengths=[n], alpha_slow=alpha_slow, rho_slow=0.1)
        sel = [j for j in chosen[0] if 0 <= j < n]
        s = _to_indicator(sel, n=n, device=device)
        cov = float(BilinearOracle.discrete_coverage(s, torch.tensor(M_np, device=device).float()).cpu().item())

        greedy_sel = greedy_hitting_set(M_np)

        geo_cov = None
        geo_cov_g = None
        if evaluate_polygon_visibility_numpy_wo_gt is not None:
            try:
                geo_cov = float(
                    evaluate_polygon_visibility_numpy_wo_gt(
                        pts.astype(np.float64), np.array(sel, dtype=int), f"{name}_ss"
                    )
                )
                geo_cov_g = float(
                    evaluate_polygon_visibility_numpy_wo_gt(
                        pts.astype(np.float64), np.array(greedy_sel, dtype=int), f"{name}_greedy"
                    )
                )
            except Exception:
                geo_cov = None
                geo_cov_g = None

        tri_covs.append(cov)
        sizes.append(len(sel))
        greedy_sizes.append(len(greedy_sel))
        geo_covs.append(geo_cov)
        geo_covs_greedy.append(geo_cov_g)

        per_instance.append(
            {
                "idx": int(i),
                "name": name,
                "n": int(n),
                "tri_cov": float(cov),
                "geo_cov": (None if geo_cov is None else float(geo_cov)),
                "S": int(len(sel)),
                "optS": (None if opt_size is None else int(opt_size)),
                "S_over_opt": (None if (opt_size is None or opt_size <= 0) else float(len(sel) / float(opt_size))),
                "greedyS": int(len(greedy_sel)),
                "greedy_geo_cov": (None if geo_cov_g is None else float(geo_cov_g)),
                "greedyS_over_opt": (None if (opt_size is None or opt_size <= 0) else float(len(greedy_sel) / float(opt_size))),
            }
        )

        if verbose:
            rows.append(
                {
                    "idx": f"{i}",
                    "name": name,
                    "n": f"{n}",
                    "tri_cov": f"{cov:.3f}",
                    "geo_cov": (f"{geo_cov:.3f}" if geo_cov is not None else "NA"),
                    "S": f"{len(sel)}",
                    "optS": (f"{opt_size}" if opt_size is not None else "NA"),
                    "Sopt": (f"{(len(sel)/opt_size):.2f}" if (opt_size is not None and opt_size > 0) else "NA"),
                    "greedyS": f"{len(greedy_sel)}",
                    "geo_cov_g": (f"{geo_cov_g:.3f}" if geo_cov_g is not None else "NA"),
                    "Gopt": (f"{(len(greedy_sel)/opt_size):.2f}" if (opt_size is not None and opt_size > 0) else "NA"),
                }
            )

    if verbose and rows:
        cols = [
            ("idx", "#"),
            ("name", "name"),
            ("n", "n"),
            ("tri_cov", "tri-cov"),
            ("geo_cov", "geo-cov"),
            ("S", "|S|"),
            ("optS", "opt|S|"),
            ("Sopt", "|S|/opt"),
            ("greedyS", "greedy|S|"),
            ("geo_cov_g", "greedy geo"),
            ("Gopt", "greedy/opt"),
        ]
        widths: Dict[str, int] = {}
        for key, title in cols:
            widths[key] = max(len(title), max(len(r[key]) for r in rows))

        header = " | ".join(title.ljust(widths[key]) for key, title in cols)
        sep = "-+-".join("-" * widths[key] for key, _title in cols)
        print(header)
        print(sep)
        for r in rows:
            line = " | ".join(r[key].ljust(widths[key]) for key, _title in cols)
            print(line)

    def _nanmean(xs: Sequence[Optional[float]]) -> Optional[float]:
        vals = [x for x in xs if x is not None]
        if not vals:
            return None
        return float(np.mean(vals))

    geo_mean = _nanmean(geo_covs)
    geo_mean_g = _nanmean(geo_covs_greedy)

    # OPT-size ratios (if .solution files exist)
    opt_ratios: List[float] = []
    greedy_opt_ratios: List[float] = []
    for rec in per_instance:
        opt_size = rec.get("optS")
        if isinstance(opt_size, int) and opt_size > 0:
            opt_ratios.append(float(rec.get("S", 0)) / float(opt_size))
            greedy_opt_ratios.append(float(rec.get("greedyS", 0)) / float(opt_size))

    opt_ratio_mean = float(np.mean(opt_ratios)) if opt_ratios else None
    greedy_opt_ratio_mean = float(np.mean(greedy_opt_ratios)) if greedy_opt_ratios else None

    msg = (
        f"Dataset eval k={k} | tri-cov mean={float(np.mean(tri_covs)):.3f}"
        + (f" | geo-cov mean={geo_mean:.3f}" if geo_mean is not None else "")
        + f" | |S| mean={float(np.mean(sizes)):.2f}"
        + (f" | |S|/opt mean={opt_ratio_mean:.2f}" if opt_ratio_mean is not None else "")
        + f" || greedy |S| mean={float(np.mean(greedy_sizes)):.2f}"
        + (f" | greedy geo mean={geo_mean_g:.3f}" if geo_mean_g is not None else "")
        + (f" | greedy |S|/opt mean={greedy_opt_ratio_mean:.2f}" if greedy_opt_ratio_mean is not None else "")
    )
    print(msg)

    report = {
        "k": int(k),
        "per_instance": per_instance,
        "stats": {
            "tri_cov_mean": float(np.mean(tri_covs)) if tri_covs else 0.0,
            "geo_cov_mean": geo_mean,
            "S_mean": float(np.mean(sizes)) if sizes else 0.0,
            "greedyS_mean": float(np.mean(greedy_sizes)) if greedy_sizes else 0.0,
            "greedy_geo_cov_mean": geo_mean_g,
            "S_over_opt_mean": opt_ratio_mean,
            "greedyS_over_opt_mean": greedy_opt_ratio_mean,
        },
    }
    return report


# ----------------------------
# CLI
# ----------------------------


def main():
    # Mirror rl_agp.py conventions: if DATASET_PATH is available, default to
    # DATASET_PATH/train and DATASET_PATH/dev.
    dataset_path = os.getenv("DATASET_PATH")
    if load_dotenv is not None:
        try:
            # python-dotenv's find_dotenv can be flaky in non-file entrypoints; pass explicit path.
            load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
            dataset_path = os.getenv("DATASET_PATH") or dataset_path
        except Exception:
            pass

    default_train = os.path.join(dataset_path, "train") if dataset_path else None
    default_val = os.path.join(dataset_path, "dev") if dataset_path else None
    if default_train is not None and not os.path.exists(default_train):
        default_train = None
    if default_val is not None and not os.path.exists(default_val):
        default_val = None

    parser = argparse.ArgumentParser()

    # Data selection
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Optional: path to a .pol file or a directory of .pol files to train on (uses dataset.agp_read_samples).",
    )
    parser.add_argument(
        "--agp_train_dir",
        type=str,
        default=default_train,
        help="Training path (dir of .pol or a single .pol). Defaults to DATASET_PATH/train if available.",
    )
    parser.add_argument(
        "--agp_val_dir",
        type=str,
        default=default_val,
        help="Validation path (dir of .pol or a single .pol). Defaults to DATASET_PATH/dev if available.",
    )
    # Default normalize ON to match other scripts in this repo.
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            "--normalize",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Normalize polygon coordinates when loading .pol files.",
        )
    else:  # pragma: no cover
        parser.add_argument(
            "--normalize",
            dest="normalize",
            action="store_true",
            default=True,
            help="Normalize polygon coordinates when loading .pol files.",
        )
        parser.add_argument(
            "--no-normalize",
            dest="normalize",
            action="store_false",
            help="Disable polygon normalization when loading .pol files.",
        )
    parser.add_argument(
        "--train-size",
        type=int,
        default=8000,
        help="Cap on number of polygons used for training (default: 8000; like rl_agp.py).",
    )

    # Evaluation
    parser.add_argument(
        "--eval-k",
        type=int,
        default=-1,
        help="How many polygons to evaluate on from the dataset (default: all).",
    )
    parser.add_argument(
        "--eval-verbose",
        action="store_true",
        help="Print per-instance evaluation lines (name, n, tri-cov, |S|, greedy |S|).",
    )
    parser.add_argument(
        "--eval-comb",
        action="store_true",
        help="Also run the synthetic comb-polygon evaluation (off by default).",
    )

    parser.add_argument("--num-train", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-n", type=int, default=12)
    parser.add_argument("--max-n", type=int, default=30)

    parser.add_argument("--embedding-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--lambda-card", type=float, default=0.05)
    parser.add_argument(
        "--lambda-card-start",
        type=float,
        default=0.0,
        help="Curriculum start value for lambda_card (coverage-first).",
    )
    parser.add_argument(
        "--lambda-card-cov-threshold",
        type=float,
        default=0.98,
        help="Start ramping lambda_card once EMA tri-cov >= threshold.",
    )
    parser.add_argument(
        "--lambda-card-ramp-steps",
        type=int,
        default=400,
        help="Number of optimizer steps to ramp lambda_card from start to final.",
    )
    parser.add_argument(
        "--penalty-weight",
        type=float,
        default=10.0,
        help="Weight for uncovered-triangle penalties in oracle cost (normalized by #triangles).",
    )
    parser.add_argument(
        "--penalty-power",
        type=float,
        default=1.0,
        help="Optional power on relu(1-Ms) penalties (e.g., 2.0 for squared hinge).",
    )
    parser.add_argument(
        "--coverage-gate",
        type=float,
        default=0.0,
        help="If > 0, use coverage-gated cost: below gate, primarily coverage penalty + small card term; "
        "above gate, primarily cardinality penalty. Prevents coverage/cardinality trade-off. "
        "Recommended: 0.90-0.95.",
    )
    parser.add_argument(
        "--alpha-slow",
        type=float,
        default=0.1,
        help="Fast/slow mixture weight beta (paper): p = (1-beta)*phi(W) + beta*phi(rho*W).",
    )
    parser.add_argument(
        "--beta-slow",
        dest="beta_slow",
        type=float,
        default=None,
        help="Alias for --alpha-slow (paper uses beta).",
    )
    parser.add_argument(
        "--rho-slow",
        type=float,
        default=0.03,
        help="Slow-component temperature scaling rho (paper): logits -> rho*logits (rho<<1 => more uniform-like).",
    )
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=0.05,
        help="Lambda for negative-entropy regularizer E[log pi] (minimization). Larger => smaller sets / earlier EOS.",
    )
    parser.add_argument(
        "--entropy-decay-steps",
        type=int,
        default=60,
        help="Halve-style entropy schedule: every N optimizer steps, multiply lambda by --entropy-decay-rate.",
    )
    parser.add_argument(
        "--entropy-decay-rate",
        type=float,
        default=0.5,
        help="Entropy schedule multiplier per --entropy-decay-steps (default: 0.5).",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta-ema", type=float, default=0.99)

    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Print a training progress line every N batches (0 disables).",
    )

    parser.add_argument("--no-greedy-baseline", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    beta_slow = getattr(args, "beta_slow", None)
    if beta_slow is not None:
        args.alpha_slow = float(beta_slow)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset selection: existing .pol files (project-native) or synthetic.
    train_path = args.agp_train_dir or args.train_path
    val_path = args.agp_val_dir

    # Keep --smoke fast: don't automatically pull from DATASET_PATH defaults.
    if args.smoke:
        if args.train_path is None and (default_train is not None) and args.agp_train_dir == default_train:
            train_path = None
        if (default_val is not None) and args.agp_val_dir == default_val:
            val_path = None

    if train_path is not None:
        pol_paths = _get_pol_files(train_path)
        if not pol_paths:
            raise ValueError(f"No .pol files found at {train_path}")
        dataset = AGPPolDataset(pol_paths=pol_paths, normalize=args.normalize, cache_instances=True)

        if args.train_size is not None:
            k = min(int(args.train_size), len(dataset))
            class _Slice(torch.utils.data.Dataset):
                def __init__(self, base, k: int):
                    self.base = base
                    self.k = min(k, len(base))
                def __len__(self):
                    return self.k
                def __getitem__(self, i):
                    return self.base[i]
            dataset = _Slice(dataset, k)

        if args.smoke:
            # Keep smoke fast: slice by wrapping indices.
            class _Slice(torch.utils.data.Dataset):
                def __init__(self, base, k: int):
                    self.base = base
                    self.k = min(k, len(base))
                def __len__(self):
                    return self.k
                def __getitem__(self, i):
                    return self.base[i]
            dataset = _Slice(dataset, 20)
    else:
        dataset = AGPDataset(
            num_samples=(20 if args.smoke else args.num_train),
            min_n=args.min_n,
            max_n=args.max_n,
            seed=args.seed,
            cache_instances=True,
        )

    # Optional val dataset (only for real .pol flows)
    val_dataset = None
    if val_path is not None:
        try:
            val_pol_paths = _get_pol_files(val_path)
            if val_pol_paths:
                val_dataset = AGPPolDataset(pol_paths=val_pol_paths, normalize=args.normalize, cache_instances=True)
        except Exception:
            val_dataset = None

    model = SolutionSamplerPointerNet(
        feature_dim=3,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        temperature=args.temperature,
    ).to(device)

    oracle = BilinearOracle(
        lambda_card=args.lambda_card,
        penalty_weight=args.penalty_weight,
        penalty_power=args.penalty_power,
        coverage_gate=args.coverage_gate,
    )

    train_solution_sampler(
        model=model,
        dataset=dataset,
        oracle=oracle,
        device=device,
        epochs=(2 if args.smoke else args.epochs),
        batch_size=(2 if args.smoke else args.batch_size),
        lr=args.lr,
        beta_ema=args.beta_ema,
        alpha_slow=args.alpha_slow,
        rho_slow=args.rho_slow,
        entropy_weight=args.entropy_weight,
        entropy_decay_steps=args.entropy_decay_steps,
        entropy_decay_rate=args.entropy_decay_rate,
        lambda_card_start=args.lambda_card_start,
        lambda_card_cov_threshold=args.lambda_card_cov_threshold,
        lambda_card_ramp_steps=args.lambda_card_ramp_steps,
        log_every=args.log_every,
        use_greedy_baseline=not args.no_greedy_baseline,
    )

    # Evaluate on dataset polygons by default.
    print("\n--- Eval on dataset polygons ---")
    eval_ds = val_dataset if val_dataset is not None else dataset
    sol_dir = _infer_solution_dir(args.agp_val_dir) or _infer_solution_dir(args.agp_train_dir)
    eval_report = evaluate_on_dataset(
        model,
        oracle,
        dataset=eval_ds,
        device=device,
        k=args.eval_k,
        alpha_slow=0.0,
        verbose=args.eval_verbose,
        sol_dir=sol_dir,
    )

    # Save a rl_agp-style evaluation report after training.
    os.makedirs("results", exist_ok=True)
    effective_train_size = int(len(dataset))
    out_path = os.path.join("results", f"ss_agp_evaluation_{effective_train_size}.json")
    results_summary = {
        "args": vars(args),
        "num_train_samples": effective_train_size,
        "num_val_samples": int(len(eval_ds)) if hasattr(eval_ds, "__len__") else None,
        "training_method": "solution_sampler_reinforce",
        "eval": eval_report,
    }
    with open(out_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Results summary saved to {out_path}")

    # Optional synthetic comb evaluation.
    if args.eval_comb:
        print("\n--- Eval on comb polygons ---")
        evaluate_on_comb(model, oracle, device=device, n_teeth_list=(4, 6, 8), alpha_slow=0.0)


if __name__ == "__main__":
    main()
