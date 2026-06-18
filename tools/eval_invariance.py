#!/usr/bin/env python
"""eval_invariance.py — robustness of the geo-free policy+probe to rigid/index
transforms of the input polygon (reviewer Q1/Q8).

The encoder is a unidirectional LSTM over vertices in file order, and inputs are
per-axis MinMax-normalized to [0,1] (dataset.py). MinMax normalization already
makes the pipeline translation- and (axis-aligned) scale-invariant, so those are
not tested here. What is NOT invariant by construction — and is what we measure —
is rotation, mirroring, and cyclic re-indexing of the vertex order.

Faithful full-pipeline test: for each transform we
  1. apply the transform to the (normalized) polygon coordinates,
  2. RE-APPLY the pipeline's MinMax normalization (so the model sees a valid input),
  3. RE-DECODE the frozen policy's greedy seed on the transformed coords
     (we do NOT reuse the stored seed — that would only test the probe),
  4. run the frozen SetPredictor at t=0.20, and
  5. score exact CGAL coverage on the transformed polygon.

Coverage of a fixed guard set is affine-invariant, so step 5 is scored on the
same normalized coords fed to the model. Each (polygon, transform) is scored under
a UNIQUE name so the per-name visibility cache never returns a stale entry — the
documented coverage-cache gotcha (cache key = name|n, guarded by bbox; a cyclic
re-index keeps bbox+n and would otherwise hit a stale entry).

Reports, per transform: feasibility rate (Cov>=0.95), mean coverage, worst-polygon
coverage, and the delta vs. the identity baseline. A small delta = robust.

Usage:
  DATASET_PATH=/home/dseverdi/Radno/MLAG/dataset/AGPIL \
    python tools/eval_invariance.py --threshold 0.20 --out paper/data/invariance_test.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from po_agp import create_agp_model                               # noqa: E402
from set_predictor import SetPredictor, extract_pointer_embeddings  # noqa: E402
from utils import evaluate_polygon_visibility_numpy_wo_gt          # noqa: E402

try:
    import pickle
except ImportError:  # pragma: no cover
    raise


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pointer-checkpoint", type=str,
                   default="checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt")
    p.add_argument("--probe-checkpoint", type=str,
                   default="checkpoints/set_predictor/standard/set_predictor_best.pt")
    p.add_argument("--traj", type=str,
                   default="data/ls_trajectories_dev_test_clean.pkl",
                   help="test-split trajectory pkl; we use only points + name")
    p.add_argument("--embedding-size", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--n-glimpses", type=int, default=1)
    p.add_argument("--tanh-exploration", type=float, default=10.0)
    p.add_argument("--threshold", type=float, default=0.20)
    p.add_argument("--feasibility-gate", type=float, default=0.95)
    p.add_argument("--limit", type=int, default=0, help="0 = all polygons")
    p.add_argument("--out", type=str, default="paper/data/invariance_test.json")
    return p.parse_args()


# ── transforms on (n,2) coordinates ─────────────────────────────────────
def _rot(pts: np.ndarray, deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float64)
    c = pts.mean(0)
    return (pts - c) @ R.T + c


def _mirror_x(pts: np.ndarray) -> np.ndarray:
    # A reflection flips polygon orientation (CCW->CW); the visibility engine
    # requires CCW simple polygons, so reverse the vertex order to restore a
    # valid orientation of the mirrored shape. (Reversal is itself a re-index,
    # but it is the minimal way to obtain a valid mirrored polygon.)
    out = pts.copy()
    out[:, 0] = -out[:, 0]
    return out[::-1].copy()


def _reindex(pts: np.ndarray, k: int) -> np.ndarray:
    # cyclic roll of the vertex order: same polygon, different start index/order
    return np.roll(pts, k, axis=0)


def transforms_for(n: int) -> dict:
    return {
        "identity":    lambda p: p,
        "rot90":       lambda p: _rot(p, 90.0),
        "rot180":      lambda p: _rot(p, 180.0),
        "rot45":       lambda p: _rot(p, 45.0),
        "mirror_x":    _mirror_x,
        # Fixed key so every polygon aggregates into one "reindex" row; the roll
        # amount scales with n (a nontrivial cyclic re-index of the vertices).
        "reindex":     lambda p, k=max(1, n // 3): _reindex(p, k),
    }


def _minmax(pts: np.ndarray) -> np.ndarray:
    mn = pts.min(0)
    mx = pts.max(0)
    den = mx - mn
    den[den == 0] = 1.0
    return (pts - mn) / den


def _load_probe(path: str, device: str) -> SetPredictor:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or ckpt.get("args") or {}
    H = cfg.get("hidden", cfg.get("predictor_hidden", 128))
    L = cfg.get("n_attn_layers", cfg.get("predictor_attn_layers", 3))
    HD = cfg.get("heads", cfg.get("predictor_heads", 8))
    H_ptr = cfg.get("ptr_emb_dim", cfg.get("hidden_size", 128))
    model = SetPredictor(ptr_emb_dim=H_ptr, hidden=H, n_attn_layers=L, heads=HD).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _decode_seed(pointer, pts_t: torch.Tensor, n: int, device: str) -> list[int]:
    pad = torch.zeros(1, n, dtype=torch.bool, device=device)
    lengths = torch.tensor([n], dtype=torch.long, device=device)
    det_idxs, _ = pointer(pts_t, padding_mask=pad, lengths=lengths,
                          deterministic=True, no_eos=False, eos_cov_threshold=0.0)
    return [int(i) for i in det_idxs[0] if int(i) < n]


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[invariance] device={device} t={args.threshold} traj={args.traj}")

    pointer = create_agp_model(args.embedding_size, args.hidden_size,
                               args.n_glimpses, args.tanh_exploration,
                               use_tanh=True, temperature=1.0)
    pc = torch.load(args.pointer_checkpoint, map_location=device, weights_only=False)
    sd = pc["model_state_dict"] if isinstance(pc, dict) and "model_state_dict" in pc else pc
    pointer.load_state_dict(sd, strict=False)
    pointer.to(device).eval()
    probe = _load_probe(args.probe_checkpoint, device)

    with open(os.path.join(REPO_ROOT, args.traj), "rb") as fh:
        records = pickle.load(fh)
    if isinstance(records, dict):
        records = records.get("records", records)
    records = [r for r in records if r.get("points") is not None]
    if args.limit:
        records = records[: args.limit]
    print(f"[invariance] {len(records)} polygons")

    # accumulate per-transform coverage lists
    cov_by_t: dict[str, list[float]] = {}

    with torch.no_grad():
        for ridx, rec in enumerate(records):
            pts0 = np.asarray(rec["points"], dtype=np.float64)
            n = int(rec["n"]) if "n" in rec else pts0.shape[0]
            pts0 = pts0[:n]
            name = rec.get("name", f"poly{ridx}")
            tfs = transforms_for(n)

            for tname, tf in tfs.items():
                pts_t = _minmax(tf(pts0))
                pt = torch.from_numpy(pts_t).float().unsqueeze(0).to(device)  # (1,n,2)

                seed = _decode_seed(pointer, pt, n, device)
                in_S = torch.zeros(1, n, dtype=torch.bool, device=device)
                if seed:
                    in_S[0, seed] = True

                emb = extract_pointer_embeddings(pointer, pt, [n])
                pad = torch.zeros(1, n, dtype=torch.bool, device=device)
                logits = probe(emb, pt, in_S, pad)
                probs = torch.sigmoid(logits)[0, :n]
                guards = (probs >= args.threshold).nonzero(as_tuple=True)[0].cpu().tolist()

                # unique name per (polygon, transform) -> no stale vis-cache hit
                uname = f"{name}__inv_{tname}"
                cov = evaluate_polygon_visibility_numpy_wo_gt(
                    pts_t, np.array(guards, dtype=np.int64), uname,
                ) if guards else 0.0
                cov_by_t.setdefault(tname, []).append(float(cov))

            if (ridx + 1) % 50 == 0:
                print(f"  ...{ridx + 1}/{len(records)}")

    # ── summarize ──
    gate = args.feasibility_gate
    summary = {}
    base_feas = None
    for tname, covs in cov_by_t.items():
        arr = np.array(covs)
        feas = float((arr >= gate).mean())
        row = {
            "n_polygons": len(covs),
            "feasibility_rate": round(feas, 4),
            "n_below_gate": int((arr < gate).sum()),
            "mean_cov": round(float(arr.mean()), 4),
            "min_cov": round(float(arr.min()), 4),
        }
        summary[tname] = row
        if tname == "identity":
            base_feas = feas
    if base_feas is not None:
        for tname, row in summary.items():
            row["feasibility_delta_vs_identity"] = round(row["feasibility_rate"] - base_feas, 4)

    out_path = os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"threshold": args.threshold, "gate": gate, "summary": summary}, fh, indent=2)

    print("\n=== invariance summary (t={:.2f}, gate={:.2f}) ===".format(args.threshold, gate))
    print(f"{'transform':<14}{'feas_rate':>10}{'Δ vs id':>9}{'mean_cov':>10}{'min_cov':>9}")
    for tname, row in summary.items():
        print(f"{tname:<14}{row['feasibility_rate']:>10.4f}"
              f"{row.get('feasibility_delta_vs_identity', 0.0):>9.4f}"
              f"{row['mean_cov']:>10.4f}{row['min_cov']:>9.4f}")
    print(f"\n[invariance] wrote {out_path}")


if __name__ == "__main__":
    main()
