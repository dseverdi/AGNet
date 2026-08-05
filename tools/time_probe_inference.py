#!/usr/bin/env python
"""time_probe_inference.py — measure SetPredictor (probe) inference cost vs. n.

Answers reviewer Q7 ("SETPREDICTOR inference at large n, e.g. n~2000"). Times the
single geo-free forward pass that the deployed method runs: the frozen pointer
encode + one SetPredictor forward + threshold. Reports per-polygon latency at a
range of vertex counts on CPU and (if available) CUDA.

This is a timing harness only — coordinates are random (compute cost depends on n,
not on polygon validity), so it does NOT touch CGAL or coverage. Run it after the
no-seed batch or any time; it needs only the released checkpoints.

Usage:
  python tools/time_probe_inference.py
  python tools/time_probe_inference.py --ns 200 500 1000 2000 --reps 50 --out results/probe_timing.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from po_agp import create_agp_model                      # noqa: E402
from set_predictor import SetPredictor, extract_pointer_embeddings  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pointer-checkpoint", type=str,
                   default="checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt")
    p.add_argument("--probe-checkpoint", type=str,
                   default="checkpoints/set_predictor/standard/set_predictor_best.pt")
    p.add_argument("--embedding-size", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--n-glimpses", type=int, default=1)
    p.add_argument("--tanh-exploration", type=float, default=10.0)
    p.add_argument("--ns", nargs="+", type=int, default=[200, 500, 1000, 2000])
    p.add_argument("--reps", type=int, default=50, help="timed repetitions per n")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", type=str, default="results/probe_timing.json")
    return p.parse_args()


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


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def time_on(device: str, args: argparse.Namespace) -> list[dict]:
    pointer = create_agp_model(args.embedding_size, args.hidden_size,
                               args.n_glimpses, args.tanh_exploration,
                               use_tanh=True, temperature=1.0)
    pc = torch.load(args.pointer_checkpoint, map_location=device, weights_only=False)
    sd = pc["model_state_dict"] if isinstance(pc, dict) and "model_state_dict" in pc else pc
    pointer.load_state_dict(sd, strict=False)
    pointer.to(device).eval()
    probe = _load_probe(args.probe_checkpoint, device)

    rows = []
    for n in args.ns:
        # random single polygon batch (B=1) of n vertices in [0,1]^2
        pts = torch.rand(1, n, 2, device=device)
        lengths = torch.tensor([n], dtype=torch.long, device=device)
        in_S = torch.zeros(1, n, dtype=torch.bool, device=device)
        in_S[0, torch.randperm(n)[: max(1, n // 8)]] = True  # ~12.5% seeded
        pad = torch.zeros(1, n, dtype=torch.bool, device=device)

        with torch.no_grad():
            for _ in range(args.warmup):
                emb = extract_pointer_embeddings(pointer, pts, lengths)
                _ = probe(emb, pts, in_S, pad)
            _sync(device)

            # encode (pointer) timing
            t0 = time.perf_counter()
            for _ in range(args.reps):
                emb = extract_pointer_embeddings(pointer, pts, lengths)
            _sync(device)
            t_enc = (time.perf_counter() - t0) / args.reps

            # probe forward timing (uses a fixed precomputed emb)
            emb = extract_pointer_embeddings(pointer, pts, lengths)
            _sync(device)
            t0 = time.perf_counter()
            for _ in range(args.reps):
                _ = probe(emb, pts, in_S, pad)
            _sync(device)
            t_probe = (time.perf_counter() - t0) / args.reps

        rows.append({
            "device": device, "n": n,
            "encode_ms": round(t_enc * 1e3, 3),
            "probe_ms": round(t_probe * 1e3, 3),
            "total_ms": round((t_enc + t_probe) * 1e3, 3),
        })
        print(f"  [{device}] n={n:>5}  encode={t_enc*1e3:7.2f} ms  "
              f"probe={t_probe*1e3:7.2f} ms  total={(t_enc+t_probe)*1e3:7.2f} ms")
    return rows


def main() -> None:
    args = parse_args()
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    print(f"[time-probe] reps={args.reps} ns={args.ns} devices={devices}")
    all_rows = []
    for dev in devices:
        print(f"--- device={dev} ---")
        all_rows.extend(time_on(dev, args))

    os.makedirs(os.path.dirname(os.path.join(REPO_ROOT, args.out)), exist_ok=True)
    out_path = os.path.join(REPO_ROOT, args.out)
    with open(out_path, "w") as fh:
        json.dump({"reps": args.reps, "rows": all_rows}, fh, indent=2)
    print(f"[time-probe] wrote {out_path}")


if __name__ == "__main__":
    main()
