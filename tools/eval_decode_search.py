#!/usr/bin/env python
"""eval_decode_search.py — does decode-time search close the policy's coverage
tail without the probe? (reviewer Q4)

For each test polygon we compare three ways of reading the SAME frozen PO/BT
policy, all geo-free at decode time (eos_cov_threshold=0, no visibility oracle):

  greedy            argmax decode (the paper's "seed").
  best-of-K (LL)    K stochastic samples ranked by LENGTH-NORMALIZED
                    log-likelihood; pick the top-ranked. This is the fair
                    geo-free decode search the reviewer asks about (beam search
                    or stochastic sampling with length normalization).
  best-of-K (oracle) the same K samples, but pick the one with the highest
                    EXACT coverage (tie-break: fewer guards). This is NOT
                    geo-free — it consults the visibility oracle to choose, like
                    classical active search — and is reported only as an upper
                    bound on what the policy's own samples contain.

Coverage is scored post-hoc with exact CGAL (geo-free-legal: decode by
likelihood, score only for reporting). Greedy is included in each candidate pool
so neither selection can do worse than greedy on its own ranking metric.

Usage:
  DATASET_PATH=/home/dseverdi/Radno/MLAG/dataset/AGPIL \
    python tools/eval_decode_search.py --K 32 --out paper/data/decode_search.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from po_agp import create_agp_model                            # noqa: E402
from utils import evaluate_polygon_visibility_numpy_wo_gt      # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pointer-checkpoint", type=str,
                   default="checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt")
    p.add_argument("--traj", type=str,
                   default="data/ls_trajectories_dev_test_clean.pkl")
    p.add_argument("--opt-json", type=str,
                   default="paper/data/dist_dev_test.json",
                   help="source of per-polygon OPT (name->OPT), for |S|/OPT")
    p.add_argument("--embedding-size", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--n-glimpses", type=int, default=1)
    p.add_argument("--tanh-exploration", type=float, default=10.0)
    p.add_argument("--K", type=int, default=32, help="samples per polygon")
    p.add_argument("--feasibility-gate", type=float, default=0.95)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", type=str, default="paper/data/decode_search.json")
    return p.parse_args()


def _load_policy(args, device):
    pointer = create_agp_model(args.embedding_size, args.hidden_size,
                               args.n_glimpses, args.tanh_exploration,
                               use_tanh=True, temperature=1.0)
    path = os.path.join(REPO_ROOT, args.pointer_checkpoint)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    pointer.load_state_dict(sd, strict=False)
    pointer.to(device).eval()
    return pointer


def _decode(pointer, pts_t, n, device, deterministic, K=1):
    """Return list of (guards:list[int], norm_ll:float) for K rows (or 1)."""
    B = K
    inp = pts_t.unsqueeze(0).expand(B, -1, -1).contiguous()       # (B,n,2)
    pad = torch.ones(B, n, dtype=torch.bool, device=device)       # True=real
    lengths = torch.tensor([n] * B, dtype=torch.long, device=device)
    idxs, log_probs = pointer(inp, padding_mask=pad, lengths=lengths,
                              deterministic=deterministic, no_eos=False,
                              eos_cov_threshold=0.0)               # geo-free
    out = []
    for b in range(B):
        seq = [int(i) for i in idxs[b]]
        guards = [i for i in seq if i < n]
        steps = max(1, len(seq))
        norm_ll = float(log_probs[b]) / steps      # length-normalized log-lik
        out.append((guards, norm_ll))
    return out


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(f"[decode-search] device={device} K={args.K} traj={args.traj}")

    pointer = _load_policy(args, device)

    with open(os.path.join(REPO_ROOT, args.traj), "rb") as fh:
        records = pickle.load(fh)
    if isinstance(records, dict):
        records = records.get("records", records)
    records = [r for r in records if r.get("points") is not None]
    if args.limit:
        records = records[: args.limit]

    # OPT lookup by name (for |S|/OPT)
    opt_by_name = {}
    opt_path = os.path.join(REPO_ROOT, args.opt_json)
    if os.path.exists(opt_path):
        for r in json.load(open(opt_path))["polygons"]:
            if r.get("OPT"):
                opt_by_name[r["name"]] = r["OPT"]
    print(f"[decode-search] {len(records)} polygons; OPT for {len(opt_by_name)}")

    # per-method accumulators
    methods = ["greedy", "bestK_ll", "bestK_oracle"]
    cov = {m: [] for m in methods}
    son = {m: [] for m in methods}        # |S|/n
    sopt = {m: [] for m in methods}       # |S|/OPT (where available)

    def score(pts_np, guards, name):
        return float(evaluate_polygon_visibility_numpy_wo_gt(
            pts_np, np.array(guards, dtype=np.int64), name)) if guards else 0.0

    with torch.no_grad():
        for ridx, rec in enumerate(records):
            pts_np = np.asarray(rec["points"], dtype=np.float64)
            n = int(rec["n"]) if "n" in rec else pts_np.shape[0]
            pts_np = pts_np[:n]
            name = rec.get("name", f"poly{ridx}")
            pts_t = torch.from_numpy(pts_np).float().to(device)
            OPT = opt_by_name.get(name)

            # greedy candidate
            g_guards, g_ll = _decode(pointer, pts_t, n, device, deterministic=True, K=1)[0]
            # K stochastic candidates
            samples = _decode(pointer, pts_t, n, device, deterministic=False, K=args.K)

            # candidate pool = greedy + K samples, each scored for coverage
            pool = [(g_guards, g_ll)] + samples
            scored = []
            for guards, ll in pool:
                c = score(pts_np, guards, name)
                scored.append({"guards": guards, "ll": ll, "cov": c, "sz": len(guards)})

            # greedy (candidate 0)
            gsel = scored[0]
            # geo-free: argmax length-normalized log-likelihood
            llsel = max(scored, key=lambda d: d["ll"])
            # oracle: argmax coverage, tie-break fewer guards
            orsel = max(scored, key=lambda d: (d["cov"], -d["sz"]))

            for m, sel in (("greedy", gsel), ("bestK_ll", llsel), ("bestK_oracle", orsel)):
                cov[m].append(sel["cov"])
                son[m].append(sel["sz"] / n)
                if OPT:
                    sopt[m].append(sel["sz"] / OPT)

            if (ridx + 1) % 50 == 0:
                print(f"  ...{ridx + 1}/{len(records)}")

    gate = args.feasibility_gate
    summary = {}
    for m in methods:
        c = np.array(cov[m])
        summary[m] = {
            "n_polygons": len(c),
            "feasibility_rate": round(float((c >= gate).mean()), 4),
            "n_below_gate": int((c < gate).sum()),
            "n_below_099": int((c < 0.99).sum()),
            "mean_cov": round(float(c.mean()), 4),
            "min_cov": round(float(c.min()), 4),
            "mean_S_over_n": round(float(np.mean(son[m])), 4),
            "mean_S_over_OPT": round(float(np.mean(sopt[m])), 4) if sopt[m] else None,
        }

    out = {"K": args.K, "gate": gate, "geo_free": ["greedy", "bestK_ll"],
           "not_geo_free": ["bestK_oracle"], "summary": summary}
    out_path = os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)

    print(f"\n=== decode-search (test, K={args.K}) ===")
    print(f"{'method':<14}{'feas>=.95':>10}{'<0.99':>7}{'meanCov':>9}{'minCov':>8}{'|S|/n':>7}{'|S|/OPT':>9}")
    for m in methods:
        s = summary[m]
        so = "--" if s["mean_S_over_OPT"] is None else f"{s['mean_S_over_OPT']:.2f}"
        print(f"{m:<14}{s['feasibility_rate']:>10.4f}{s['n_below_099']:>7}"
              f"{s['mean_cov']:>9.4f}{s['min_cov']:>8.3f}{s['mean_S_over_n']:>7.3f}{so:>9}")
    print(f"\n[decode-search] wrote {out_path}")


if __name__ == "__main__":
    main()
