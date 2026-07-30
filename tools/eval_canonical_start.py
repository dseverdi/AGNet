#!/usr/bin/env python
"""eval_canonical_start.py -- does a CANONICAL START VERTEX buy exact
re-indexing invariance, and does it cost accuracy?

Motivation
----------
The policy encoder is a single-layer UNIDIRECTIONAL LSTM over the vertex
sequence, so a vertex's embedding summarises only the prefix before it and
vertices near index 0 are context-starved.  A cyclic roll of the input moves
which vertices are starved, which is why tools/eval_invariance.py measures a
non-trivial "reindex" degradation.

Proposed cheap fix: before the encoder sees a polygon, cyclically roll the
vertex sequence so that a deterministic, geometrically defined vertex sits at
index 0.  Because the canonical vertex is picked from the *set* of vertices
(not from positions), the canonical sequence is identical for every cyclic roll
of the same polygon, so re-indexing invariance becomes exact by construction.

Canonical rule
--------------
Canonical start = the vertex that is lexicographically smallest in (x, y)
after per-axis min-max normalisation.  Vertices of a simple polygon are
distinct points, so the argmin is unique and therefore roll-invariant.  A
cyclic roll preserves CCW orientation, so nothing else needs fixing.  We NEVER
break a tie by index -- an index tie-break would not be roll-invariant; any
tied polygon is reported instead.

What is measured
----------------
Q1  Exactness.  For each polygon, roll the input by several k, canonicalise,
    run the full pipeline, map the returned guard set back to ORIGINAL vertex
    identities, and require the set to be identical for every k.
Q2  Cost.  Full 362-polygon split at probe threshold t=0.20, exact CGAL
    coverage: (a) native file order (must reproduce 345/362 >= 0.95),
    (b) canonical start order.
Q3  Bonus.  Canonical start under geometric rot90 / rot180, against the
    rotation-only baselines (282/362, 223/362), which are re-measured here so
    the comparison is internally consistent.

Faithful full pipeline per condition: transform -> re-apply min-max ->
re-decode the frozen policy's greedy seed (never reuse the stored seed) ->
frozen SetPredictor at t -> exact CGAL coverage under a UNIQUE cache name.

Usage:
  set -a; . ./.env; set +a
  AGNET_DISC_VIS_CACHE_SIZE=14000 python tools/eval_canonical_start.py \
      --out results/canonical_start.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from po_agp import create_agp_model                                # noqa: E402
from set_predictor import SetPredictor, extract_pointer_embeddings  # noqa: E402
from utils import evaluate_polygon_visibility_numpy_wo_gt          # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pointer-checkpoint", type=str,
                   default="checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt")
    # set_predictor_FINAL.pt, not _best.pt -- _best.pt was selected on the
    # pooled dev pickle that contains this reporting split (test-set leak).
    p.add_argument("--probe-checkpoint", type=str,
                   default="checkpoints/set_predictor/standard/set_predictor_final.pt")
    p.add_argument("--traj", type=str,
                   default="data/ls_trajectories_dev_test_clean.pkl",
                   help="362-record post-dedup reporting split")
    p.add_argument("--embedding-size", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--n-glimpses", type=int, default=1)
    p.add_argument("--tanh-exploration", type=float, default=10.0)
    p.add_argument("--threshold", type=float, default=0.20)
    p.add_argument("--feasibility-gate", type=float, default=0.95)
    p.add_argument("--limit", type=int, default=0, help="0 = all polygons")
    p.add_argument("--skip-q1", action="store_true")
    p.add_argument("--out", type=str, default="results/canonical_start.json")
    return p.parse_args()


# ---- geometry helpers (same as tools/eval_invariance.py) -----------------
def _rot(pts: np.ndarray, deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]],
                 dtype=np.float64)
    c = pts.mean(0)
    return (pts - c) @ R.T + c


def _minmax(pts: np.ndarray) -> np.ndarray:
    mn = pts.min(0)
    mx = pts.max(0)
    den = mx - mn
    den[den == 0] = 1.0
    return (pts - mn) / den


def canonical_start_index(q: np.ndarray, mode: str = "min") -> tuple[int, bool]:
    """Index of the lexicographically smallest (mode='min') or largest
    (mode='max') (x, y) vertex of *q*.

    *q* must already be min-max normalised.  Returns (index, unique) where
    ``unique`` is False iff another vertex has exactly the same (x, y) -- in
    that case the argmin is NOT roll-invariant and the caller must report it
    rather than break the tie by index.
    """
    order = np.lexsort((q[:, 1], q[:, 0]))
    if mode == "max":
        order = order[::-1]
    j = int(order[0])
    unique = True
    if q.shape[0] > 1:
        j2 = int(order[1])
        unique = not (q[j, 0] == q[j2, 0] and q[j, 1] == q[j2, 1])
    return j, unique


def canonicalise(q: np.ndarray, mode: str = "min") -> tuple[np.ndarray, int, bool]:
    """Roll *q* (min-max normalised) so its canonical vertex is at index 0.

    A cyclic roll leaves the per-axis min/max untouched, so the result is
    still min-max normalised -- no renormalisation needed.  Returns
    (rolled, j, unique) with ``rolled[i] == q[(i + j) % n]``.
    """
    j, unique = canonical_start_index(q, mode=mode)
    return np.roll(q, -j, axis=0), j, unique


# ---- model plumbing ------------------------------------------------------
def _load_probe(path: str, device: str) -> SetPredictor:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or ckpt.get("args") or {}
    H = cfg.get("hidden", cfg.get("predictor_hidden", 128))
    L = cfg.get("n_attn_layers", cfg.get("predictor_attn_layers", 3))
    HD = cfg.get("heads", cfg.get("predictor_heads", 8))
    H_ptr = cfg.get("ptr_emb_dim", cfg.get("hidden_size", 128))
    model = SetPredictor(ptr_emb_dim=H_ptr, hidden=H, n_attn_layers=L,
                         heads=HD).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _decode_seed(pointer, pts_t: torch.Tensor, n: int, device: str) -> list[int]:
    """Greedy geo-free decode of the policy's seed.

    MASK CONVENTION -- the pointer's ``padding_mask`` is a VALIDITY mask
    ("True for real vertices, False for padding", dataset.collate_fn), the
    OPPOSITE of SetPredictor's ``pad`` (a true padding indicator).  Passing
    all-False here marks every vertex invalid, the decoder emits EOS at step 0
    and the seed comes back EMPTY -- silently the no-seed ablation.  With the
    all-True mask this reproduces the ``seed`` field stored in the trajectory
    pickle exactly (asserted below as a sanity check).
    """
    valid = torch.ones(1, n, dtype=torch.bool, device=device)
    lengths = torch.tensor([n], dtype=torch.long, device=device)
    det_idxs, _ = pointer(pts_t, padding_mask=valid, lengths=lengths,
                          deterministic=True, no_eos=False,
                          eos_cov_threshold=0.0)
    return [int(i) for i in det_idxs[0] if int(i) < n]


def _run_pipeline(pointer, probe, q: np.ndarray, n: int, threshold: float,
                  device: str) -> tuple[list[int], list[int], np.ndarray]:
    """Full pipeline on min-max-normalised coords *q*.

    Returns (seed, guards, probs) with indices in *q*'s own index space.
    """
    pt = torch.from_numpy(np.ascontiguousarray(q)).float().unsqueeze(0).to(device)
    seed = _decode_seed(pointer, pt, n, device)
    in_S = torch.zeros(1, n, dtype=torch.bool, device=device)
    if seed:
        in_S[0, seed] = True
    emb = extract_pointer_embeddings(pointer, pt, [n])
    pad = torch.zeros(1, n, dtype=torch.bool, device=device)
    logits = probe(emb, pt, in_S, pad)
    probs = torch.sigmoid(logits)[0, :n]
    guards = (probs >= threshold).nonzero(as_tuple=True)[0].cpu().tolist()
    return seed, guards, probs.cpu().numpy()


def _roll_ks(n: int) -> list[int]:
    """Several non-trivial cyclic roll amounts for the Q1 exactness check."""
    cands = [0, 1, 2, n // 4, n // 3, n // 2, (2 * n) // 3, n - 1]
    out = []
    for k in cands:
        k = int(k) % n
        if k not in out:
            out.append(k)
    return out


def _summarise(covs: list[float], ratios: list[float], gate: float) -> dict:
    a = np.asarray(covs, dtype=np.float64)
    r = np.asarray(ratios, dtype=np.float64)
    return {
        "n_polygons": int(a.size),
        "n_ge_gate": int((a >= gate).sum()),
        "feasibility_rate": round(float((a >= gate).mean()), 4),
        "mean_cov": round(float(a.mean()), 4),
        "min_cov": round(float(a.min()), 4),
        "mean_S_over_n": round(float(r.mean()), 4),
    }


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.time()
    print(f"[canon] device={device} t={args.threshold} traj={args.traj}")
    print(f"[canon] AGNET_DISC_VIS_CACHE_SIZE="
          f"{os.getenv('AGNET_DISC_VIS_CACHE_SIZE', '<unset>')}")

    pointer = create_agp_model(args.embedding_size, args.hidden_size,
                               args.n_glimpses, args.tanh_exploration,
                               use_tanh=True, temperature=1.0)
    pc = torch.load(os.path.join(REPO_ROOT, args.pointer_checkpoint),
                    map_location=device, weights_only=False)
    sd = pc["model_state_dict"] if isinstance(pc, dict) and "model_state_dict" in pc else pc
    pointer.load_state_dict(sd, strict=False)
    pointer.to(device).eval()
    probe = _load_probe(os.path.join(REPO_ROOT, args.probe_checkpoint), device)

    with open(os.path.join(REPO_ROOT, args.traj), "rb") as fh:
        blob = pickle.load(fh)
    records = blob.get("records", blob) if isinstance(blob, dict) else blob
    records = [r for r in records if r.get("points") is not None]
    if args.limit:
        records = records[: args.limit]
    print(f"[canon] {len(records)} polygons")

    # scoring conditions: q-builder from the RAW stored points
    def build_native(p0):
        return _minmax(p0), None

    def build_canon(p0):
        qc, j, uniq = canonicalise(_minmax(p0))
        return qc, (j, uniq)

    def build_rot(p0, deg):
        return _minmax(_rot(p0, deg)), None

    def build_canon_rot(p0, deg):
        qc, j, uniq = canonicalise(_minmax(_rot(p0, deg)))
        return qc, (j, uniq)

    def build_canon_max(p0):
        # CONTROL: same construction, opposite lexicographic extreme. Also
        # exactly roll-invariant. Separates "cost of pinning the start" from
        # "cost of this particular rule".
        qc, j, uniq = canonicalise(_minmax(p0), mode="max")
        return qc, (j, uniq)

    def build_reindex_n3(p0):
        # CONTROL: an arbitrary (non-canonical, non-invariant) cyclic roll --
        # the "reindex" row of tools/eval_invariance.py, recomputed here so the
        # canonical rows are compared against a same-run re-index baseline.
        n_ = p0.shape[0]
        return _minmax(np.roll(p0, max(1, n_ // 3), axis=0)), None

    conditions = {
        "native":        lambda p0: build_native(p0),
        "canon":         lambda p0: build_canon(p0),
        "canon_argmax":  lambda p0: build_canon_max(p0),
        "reindex_n3":    lambda p0: build_reindex_n3(p0),
        "rot90":         lambda p0: build_rot(p0, 90.0),
        "canon_rot90":   lambda p0: build_canon_rot(p0, 90.0),
        "rot180":        lambda p0: build_rot(p0, 180.0),
        "canon_rot180":  lambda p0: build_canon_rot(p0, 180.0),
    }

    # conditions also scored at a matched guard budget m = |S_native|
    mb_conditions = ["native", "canon", "canon_argmax", "reindex_n3"]
    mb_cov: dict[str, list[float]] = {c: [] for c in mb_conditions}
    mb_ratio: dict[str, list[float]] = {c: [] for c in mb_conditions}

    cov: dict[str, list[float]] = {c: [] for c in conditions}
    ratio: dict[str, list[float]] = {c: [] for c in conditions}
    nguards: dict[str, list[int]] = {c: [] for c in conditions}
    below: dict[str, list] = {c: [] for c in conditions}

    q1_ok = 0
    q1_total = 0
    q1_failures = []
    tie_polys = []            # canonical argmin not unique (must be empty)
    xtie_polys = []           # min-x shared, tie broken by y (still exact)
    seed_match = 0
    seed_mismatch = []
    canon_vs_native_same_set = 0
    canon_start_hist: dict[str, int] = {}

    with torch.no_grad():
        for ridx, rec in enumerate(records):
            n = int(rec["n"]) if "n" in rec else len(rec["points"])
            p0 = np.asarray(rec["points"], dtype=np.float64)[:n]
            name = rec.get("name", f"poly{ridx}")

            q_native = _minmax(p0)
            j0, uniq0 = canonical_start_index(q_native)
            if not uniq0:
                tie_polys.append({"name": name, "n": n, "argmin": j0})
            # informational: is the x-coordinate of the canonical vertex shared?
            if int((q_native[:, 0] == q_native[j0, 0]).sum()) > 1:
                xtie_polys.append(name)
            canon_start_hist[name] = j0

            # ---- Q1: exactness under arbitrary cyclic roll -------------
            if not args.skip_q1:
                ks = _roll_ks(n)
                sets_by_k = {}
                arrays_bit_identical = True
                ref_arr = None
                for k in ks:
                    p_k = np.roll(p0, k, axis=0)          # p_k[i] = p0[(i-k)%n]
                    q_k = _minmax(p_k)
                    qc, j_k, uniq_k = canonicalise(q_k)
                    if not uniq_k and {"name": name, "n": n, "argmin": j_k} not in tie_polys:
                        tie_polys.append({"name": name, "n": n, "argmin": j_k})
                    if ref_arr is None:
                        ref_arr = qc
                    elif not np.array_equal(ref_arr, qc):
                        arrays_bit_identical = False
                    _, g, _ = _run_pipeline(pointer, probe, qc, n,
                                            args.threshold, device)
                    # canonical index c -> position in p_k is (c + j_k) % n
                    # -> original index is (c + j_k - k) % n
                    orig = frozenset(int((c + j_k - k) % n) for c in g)
                    sets_by_k[k] = orig
                ref = sets_by_k[ks[0]]
                same = all(s == ref for s in sets_by_k.values())
                q1_total += 1
                if same and arrays_bit_identical:
                    q1_ok += 1
                else:
                    q1_failures.append({
                        "name": name, "n": n,
                        "arrays_bit_identical": bool(arrays_bit_identical),
                        "sets": {str(k): sorted(s) for k, s in sets_by_k.items()},
                    })

            # ---- Q2/Q3: scored conditions ------------------------------
            guards_by_cond = {}
            probs_by_cond = {}
            q_by_cond = {}
            for cname, builder in conditions.items():
                q, extra = builder(p0)
                seed, guards, probs = _run_pipeline(pointer, probe, q, n,
                                                    args.threshold, device)
                probs_by_cond[cname] = probs
                q_by_cond[cname] = q
                if cname == "native":
                    stored = rec.get("seed")
                    if stored is not None:
                        if list(seed) == [int(x) for x in stored]:
                            seed_match += 1
                        else:
                            seed_mismatch.append({
                                "name": name, "decoded": seed,
                                "stored": [int(x) for x in stored]})
                # unique cache name per (polygon, condition): the vis cache key
                # is name|n guarded by bbox, and a cyclic re-index preserves
                # both -> a shared name would return a stale coverage entry.
                uname = f"{name}__cstart_{cname}"
                c = evaluate_polygon_visibility_numpy_wo_gt(
                    q, np.array(guards, dtype=np.int64), uname) if guards else 0.0
                cov[cname].append(float(c))
                ratio[cname].append(len(guards) / float(n))
                nguards[cname].append(len(guards))
                if c < args.feasibility_gate:
                    below[cname].append({"name": name, "n": n,
                                         "cov": round(float(c), 4),
                                         "n_guards": len(guards)})
                # map guards back to original vertex identities
                if cname == "native":
                    guards_by_cond[cname] = frozenset(guards)
                elif cname == "canon":
                    guards_by_cond[cname] = frozenset(
                        int((g + j0) % n) for g in guards)

            if guards_by_cond.get("native") == guards_by_cond.get("canon"):
                canon_vs_native_same_set += 1

            # ---- matched-budget control --------------------------------
            # A fixed threshold t=0.20 lets each condition pick its own number
            # of guards, so the coverage columns above conflate ranking quality
            # with guard budget. Here every condition is forced to spend the
            # SAME budget m = |S_native| by taking its own top-m vertices by
            # predicted probability. Coverage is re-scored under the SAME cache
            # name as that condition (identical points array -> the per-guard
            # visibility cache is reused and only the new solution key is
            # unioned; the coverage_cache is keyed by the guard tuple, so this
            # is a correct hit, not a stale one).
            m = len(guards_by_cond["native"])
            for cname in mb_conditions:
                pr = probs_by_cond[cname]
                order = np.argsort(-pr, kind="stable")
                top = sorted(int(i) for i in order[:m])
                uname = f"{name}__cstart_{cname}"
                cmb = evaluate_polygon_visibility_numpy_wo_gt(
                    q_by_cond[cname], np.array(top, dtype=np.int64),
                    uname) if top else 0.0
                mb_cov[cname].append(float(cmb))
                mb_ratio[cname].append(len(top) / float(n))

            if (ridx + 1) % 25 == 0:
                el = time.time() - t_start
                print(f"  ...{ridx + 1}/{len(records)}  ({el:.0f}s)")

    gate = args.feasibility_gate
    summary = {c: _summarise(cov[c], ratio[c], gate) for c in conditions}
    for c in conditions:
        summary[c]["mean_n_guards"] = round(float(np.mean(nguards[c])), 3)
    for c in conditions:
        summary[c]["n_ge_gate_delta_vs_native"] = (
            summary[c]["n_ge_gate"] - summary["native"]["n_ge_gate"])

    mb_summary = {c: _summarise(mb_cov[c], mb_ratio[c], gate)
                  for c in mb_conditions}
    for c in mb_conditions:
        mb_summary[c]["n_ge_gate_delta_vs_native"] = (
            mb_summary[c]["n_ge_gate"] - mb_summary["native"]["n_ge_gate"])

    out = {
        "threshold": args.threshold,
        "gate": gate,
        "traj": args.traj,
        "pointer_checkpoint": args.pointer_checkpoint,
        "probe_checkpoint": args.probe_checkpoint,
        "canonical_rule": "argmin lexicographic (x,y) after per-axis min-max; "
                          "roll so that vertex is at index 0",
        "n_polygons": len(records),
        "q1": {
            "checked": q1_total,
            "exact": q1_ok,
            "fraction": (round(q1_ok / q1_total, 6) if q1_total else None),
            "roll_amounts": "k in {0,1,2,n//4,n//3,n//2,2n//3,n-1} (deduped)",
            "failures": q1_failures[:20],
        },
        "sanity": {
            "seed_reproduced_native": seed_match,
            "seed_mismatch_examples": seed_mismatch[:10],
            "canonical_argmin_ties": tie_polys,
            "n_polys_with_shared_min_x": len(xtie_polys),
            "canon_guardset_equals_native_guardset": canon_vs_native_same_set,
            "n_polys_already_canonical": int(
                sum(1 for v in canon_start_hist.values() if v == 0)),
        },
        "summary": summary,
        "matched_budget_summary": mb_summary,
        "matched_budget_note": "every condition forced to m=|S_native| guards "
                               "by taking its own top-m predicted vertices",
        "below_gate": {c: below[c] for c in conditions},
        "per_polygon_coverage": {c: [round(v, 6) for v in cov[c]]
                                for c in conditions},
        "names": [r.get("name", f"poly{i}") for i, r in enumerate(records)],
        "elapsed_sec": round(time.time() - t_start, 1),
    }

    out_path = os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n=== canonical start (t={args.threshold:.2f}, gate={gate:.2f}, "
          f"N={len(records)}) ===")
    print(f"Q1 exact under cyclic roll: {q1_ok}/{q1_total}")
    print(f"seed reproduced on native order: {seed_match}/{len(records)}")
    print(f"canonical argmin ties: {len(tie_polys)}")
    hdr = f"{'condition':<15}{'>=gate':>9}{'d_native':>10}{'mean_cov':>10}{'min_cov':>9}{'|S|/n':>8}"
    print(hdr)
    for c in conditions:
        s = summary[c]
        print(f"{c:<15}{s['n_ge_gate']:>5}/{s['n_polygons']:<3}"
              f"{s['n_ge_gate_delta_vs_native']:>10}{s['mean_cov']:>10.4f}"
              f"{s['min_cov']:>9.4f}{s['mean_S_over_n']:>8.4f}")
    print(f"\n--- matched budget (m = |S_native| per polygon) ---")
    print(hdr)
    for c in mb_conditions:
        s = mb_summary[c]
        print(f"{c:<15}{s['n_ge_gate']:>5}/{s['n_polygons']:<3}"
              f"{s['n_ge_gate_delta_vs_native']:>10}{s['mean_cov']:>10.4f}"
              f"{s['min_cov']:>9.4f}{s['mean_S_over_n']:>8.4f}")
    print(f"\n[canon] wrote {out_path}  ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
