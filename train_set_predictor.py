"""Train the SetPredictor — single-shot per-vertex set-membership editor.

Inputs:
    --train-traj data/ls_trajectories_train.pkl       (any of the existing builds)
    --val-traj   data/ls_trajectories_dev.pkl

Each training example is (polygon, pointer_seed) → (LS_final_set). Target
is a per-vertex binary label: 1 if vertex was in LS_final, 0 otherwise.

This trainer is intentionally simple compared to train_editor.py:
    • no STOP head, no STOP supervision
    • no DAgger, no rollout drift
    • no auxiliary visibility loss (pointer encoder already encodes visibility
      implicitly; we let attention extract it from those embeddings)
    • single forward, single BCE per vertex
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
except ImportError:
    pass

from set_predictor import SetPredictor, set_loss, extract_pointer_embeddings
from utils import evaluate_polygon_visibility_numpy_wo_gt
from po_agp import (
    create_agp_model,
    prepare_datasets,
    _read_opt_solution,
    _load_json_config,
    _apply_config_to_args,
    _get_explicit_args,
)
from dataset import collate_fn


# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--train-traj", type=str, default=None)
    p.add_argument("--val-traj", type=str, default=None)
    p.add_argument("--pointer-checkpoint", type=str,
                   default="checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt")
    p.add_argument("--embedding-size", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--n-glimpses", type=int, default=1)
    p.add_argument("--tanh-exploration", type=float, default=10.0)

    p.add_argument("--predictor-hidden", type=int, default=128)
    p.add_argument("--predictor-attn-layers", type=int, default=3)
    p.add_argument("--predictor-heads", type=int, default=8)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--pos-weight", type=float, default=None,
                   help="BCE pos_weight. Auto-derived from training data "
                        "(ratio of kept/removed vertices) if not set.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--disable-ptr-emb", action="store_true",
                   help="Coordinate-only ablation: mask the frozen pointer "
                        "embedding to zeros so the SetPredictor sees only "
                        "(xy, in_S). Architecture and parameter count are "
                        "unchanged.")
    p.add_argument("--disable-seed", action="store_true",
                   help="No-seed ablation: mask the policy's seed indicator to "
                        "zeros (both the input feature and the seed-context "
                        "pool) so the SetPredictor cannot read the decoder's "
                        "guard set. Combine with --disable-ptr-emb for the "
                        "coords-only control. Architecture/param count "
                        "unchanged.")

    # Eval during training: per-epoch threshold sweep on dev.
    p.add_argument("--eval-thresholds", nargs="+", type=float,
                   default=[0.3, 0.5, 0.7, 0.9],
                   help="Probability thresholds to sweep at eval each epoch.")
    p.add_argument("--rollout-eval-k", type=int, default=100,
                   help="Number of dev polygons for the per-epoch rollout eval.")

    p.add_argument("--out-dir", type=str,
                   default="checkpoints/set_predictor")
    defaults = {a.dest: a.default for a in p._actions if a.dest != "help"}
    explicit = _get_explicit_args(p, sys.argv[1:])
    args = p.parse_args()
    if args.config:
        cfg = _load_json_config(args.config)
        _apply_config_to_args(args, cfg, defaults, list(defaults.keys()), explicit)
    if not args.train_traj:
        raise SystemExit("error: --train-traj required (CLI or config)")
    return args


# ──────────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────────
class SetPredDataset:
    """Each polygon is one training example. Records loaded from a trajectory
    pickle; we only use the polygon points, pointer's seed, and LS's final
    set (not the per-step trajectory).
    """

    def __init__(self, pickle_path: str):
        with open(pickle_path, "rb") as f:
            d = pickle.load(f)
        # Keep only records with non-empty seed and final.
        self.records = [
            r for r in d["records"]
            if r.get("seed") and r.get("final")
        ]
        print(f"  [SetData] {pickle_path}: {len(self.records)} polygons")

    def __len__(self):
        return len(self.records)

    def buckets_by_size(self) -> dict[int, list[int]]:
        b = defaultdict(list)
        for i, r in enumerate(self.records):
            b[r["n"]].append(i)
        return b


def make_batch(ds: SetPredDataset, indices: list[int], pointer,
               device: torch.device) -> tuple:
    """All indices must share polygon size n. Returns (ptr_emb, xy, in_S,
    pad, target, names)."""
    pts_list = []
    seed_list = []
    final_list = []
    names = []
    ns = []
    for i in indices:
        rec = ds.records[i]
        pts_list.append(torch.from_numpy(rec["points"]).float())
        seed_list.append(rec["seed"])
        final_list.append(rec["final"])
        names.append(rec["name"])
        ns.append(rec["n"])
    n = ns[0]
    pts = torch.stack(pts_list, dim=0).to(device)            # (B, n, 2)

    B = len(indices)
    lengths = torch.tensor([n] * B, dtype=torch.long, device=device)
    with torch.no_grad():
        ptr_emb = extract_pointer_embeddings(pointer, pts, lengths)   # (B, n, H)

    # in_S: pointer's seed mask. target: LS_final mask.
    in_S = torch.zeros(B, n, dtype=torch.bool, device=device)
    target = torch.zeros(B, n, dtype=torch.bool, device=device)
    for b, (seed, final) in enumerate(zip(seed_list, final_list)):
        in_S[b, list(seed)] = True
        target[b, list(final)] = True

    # No padding within a same-size bucket.
    pad = torch.zeros(B, n, dtype=torch.bool, device=device)
    return ptr_emb, pts, in_S, pad, target, names, ns


def epoch_batches(ds: SetPredDataset, batch_size: int, shuffle: bool = True):
    buckets = ds.buckets_by_size()
    if shuffle:
        for k in buckets:
            random.shuffle(buckets[k])
    bucket_sizes = list(buckets.keys())
    if shuffle:
        random.shuffle(bucket_sizes)
    for n in bucket_sizes:
        b = buckets[n]
        for i in range(0, len(b), batch_size):
            yield b[i: i + batch_size]


# ──────────────────────────────────────────────────────────────────────
#  Per-epoch eval: threshold sweep on dev
# ──────────────────────────────────────────────────────────────────────
def _coverage_exact(pts: np.ndarray, guards: list[int], name: str) -> float:
    if not guards:
        return 0.0
    try:
        return float(evaluate_polygon_visibility_numpy_wo_gt(
            pts, np.array(guards, dtype=np.int64), name,
        ))
    except Exception:
        return 0.0


def eval_threshold_sweep(model: SetPredictor, ds: SetPredDataset, pointer,
                         device, args, dev_sol_dir: str | None,
                         n_polygons: int = 100):
    """For each threshold, compute (cov, |S|/n, |S|/OPT) averaged over n_polygons
    dev polygons."""
    model.eval()
    pointer.eval()
    indices = list(range(min(n_polygons, len(ds))))
    if not indices:
        return {}
    # Group by polygon size for batched forward.
    buckets = defaultdict(list)
    for i in indices:
        buckets[ds.records[i]["n"]].append(i)

    per_t: dict[float, dict] = {t: {"cov": [], "chv": [], "opt": []}
                                for t in args.eval_thresholds}
    seed_cov_acc = []
    seed_chv_acc = []
    seed_opt_acc = []

    with torch.no_grad():
        for n, ids in buckets.items():
            for start in range(0, len(ids), args.batch_size):
                chunk = ids[start: start + args.batch_size]
                ptr_emb, pts, in_S, pad, target, names, ns = make_batch(
                    ds, chunk, pointer, device,
                )
                logits = model(ptr_emb, pts, in_S, pad)
                probs = torch.sigmoid(logits).cpu().numpy()

                for b, name in enumerate(names):
                    pts_np = pts[b].cpu().numpy()
                    seed_idx = ds.records[chunk[b]]["seed"]
                    seed_cov = _coverage_exact(pts_np, seed_idx, name)
                    seed_cov_acc.append(seed_cov)
                    seed_chv_acc.append(len(seed_idx) / max(1, n))
                    opt_sol = _read_opt_solution(dev_sol_dir, name) if dev_sol_dir else None
                    opt_size = len(opt_sol) if opt_sol else None
                    if opt_size:
                        seed_opt_acc.append(len(seed_idx) / opt_size)

                    for t in args.eval_thresholds:
                        keep = [v for v in range(n) if probs[b, v] >= t]
                        cov = _coverage_exact(pts_np, keep, name)
                        per_t[t]["cov"].append(cov)
                        per_t[t]["chv"].append(len(keep) / max(1, n))
                        if opt_size and keep:
                            per_t[t]["opt"].append(len(keep) / opt_size)

    seed_summary = {
        "cov": float(np.mean(seed_cov_acc)) if seed_cov_acc else None,
        "chv": float(np.mean(seed_chv_acc)) if seed_chv_acc else None,
        "opt": float(np.mean(seed_opt_acc)) if seed_opt_acc else None,
    }
    sweep_summary = {}
    for t in args.eval_thresholds:
        sweep_summary[t] = {
            "cov": float(np.mean(per_t[t]["cov"])) if per_t[t]["cov"] else None,
            "chv": float(np.mean(per_t[t]["chv"])) if per_t[t]["chv"] else None,
            "opt": float(np.mean(per_t[t]["opt"])) if per_t[t]["opt"] else None,
        }
    return {"seed": seed_summary, "thresholds": sweep_summary}


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[setpred] device={device}")

    train_ds = SetPredDataset(args.train_traj)
    val_ds = SetPredDataset(args.val_traj) if args.val_traj else None

    # Derive pos_weight from training data (ratio of removed/kept).
    if args.pos_weight is None:
        n_kept = 0; n_removed = 0
        for rec in train_ds.records:
            kept = set(rec["final"])
            for v in range(rec["n"]):
                if v in kept:
                    n_kept += 1
                else:
                    n_removed += 1
        pw = max(1.0, n_removed / max(1, n_kept))
        print(f"[setpred] pos_weight (auto) = {pw:.2f}  (kept={n_kept} removed={n_removed})")
    else:
        pw = args.pos_weight
        print(f"[setpred] pos_weight (manual) = {pw:.2f}")

    # Pointer (frozen).
    pointer = create_agp_model(
        args.embedding_size, args.hidden_size, args.n_glimpses,
        args.tanh_exploration, use_tanh=True, temperature=1.0,
    )
    ckpt_path = args.pointer_checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    pointer.load_state_dict(sd, strict=False)
    pointer.eval()
    for p in pointer.parameters():
        p.requires_grad_(False)

    # SetPredictor.
    model = SetPredictor(
        ptr_emb_dim=args.hidden_size,
        hidden=args.predictor_hidden,
        n_attn_layers=args.predictor_attn_layers,
        heads=args.predictor_heads,
        disable_ptr_emb=args.disable_ptr_emb,
        disable_seed=args.disable_seed,
    ).to(device)
    if args.disable_ptr_emb:
        print("[setpred] coordinate-only ablation: ptr_emb masked to zeros")
    if args.disable_seed:
        print("[setpred] no-seed ablation: seed indicator masked to zeros")
    print(f"[setpred] params = {model.num_params():,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # dev sol_dir for OPT lookups.
    DATASET_PATH = os.getenv("DATASET_PATH")
    dev_sol_dir = os.path.join(DATASET_PATH, "dev") if DATASET_PATH else None

    history = []
    best_metric = -float("inf")  # we'll use best cov-preserving |S|/n trade

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        loss_sum = 0.0
        n_examples = 0
        for batch_indices in epoch_batches(train_ds, args.batch_size, shuffle=True):
            ptr_emb, pts, in_S, pad, target, names, ns = make_batch(
                train_ds, batch_indices, pointer, device,
            )
            logits = model(ptr_emb, pts, in_S, pad)
            loss_d = set_loss(logits, target, pad=pad, pos_weight=pw)
            optimizer.zero_grad(set_to_none=True)
            loss_d["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            B = ptr_emb.shape[0]
            loss_sum += loss_d["loss"].item() * B
            n_examples += B
        train_loss = loss_sum / max(1, n_examples)
        elapsed = time.perf_counter() - t0

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "n_examples": n_examples,
            "epoch_time_s": elapsed,
        }
        print(f"[ep {epoch:3d}] L={train_loss:.4f}  n={n_examples}  {elapsed:.1f}s")

        # Per-epoch threshold sweep on dev.
        if val_ds is not None and args.rollout_eval_k > 0:
            sweep = eval_threshold_sweep(
                model, val_ds, pointer, device, args, dev_sol_dir,
                n_polygons=args.rollout_eval_k,
            )
            log["sweep"] = sweep
            seed = sweep["seed"]
            opt_s = f"{seed['opt']:.4f}" if seed['opt'] is not None else "  —  "
            print(f"           seed  cov={seed['cov']:.4f}  |S|/n={seed['chv']:.4f}  "
                  f"|S|/OPT={opt_s}")
            for t, m in sweep["thresholds"].items():
                cov_s = f"{m['cov']:.4f}" if m['cov'] is not None else "  —  "
                chv_s = f"{m['chv']:.4f}" if m['chv'] is not None else "  —  "
                opt_s = f"{m['opt']:.4f}" if m['opt'] is not None else "  —  "
                print(f"           t={t:.2f}  cov={cov_s}  |S|/n={chv_s}  |S|/OPT={opt_s}")

            # Promote checkpoint if any threshold yields cov ≥ seed_cov AND |S|/n < seed_chv.
            seed_cov = seed["cov"] or 0.0
            seed_chv = seed["chv"] or 1.0
            best_t = None
            best_dchv = 0.0  # negative is better (smaller |S|)
            for t, m in sweep["thresholds"].items():
                if m["cov"] is None or m["chv"] is None:
                    continue
                if m["cov"] + 1e-4 >= seed_cov and m["chv"] < seed_chv:
                    dchv = seed_chv - m["chv"]
                    if dchv > best_dchv:
                        best_dchv = dchv
                        best_t = t
            metric = best_dchv if best_t is not None else -1.0
            if best_t is not None and metric > best_metric:
                best_metric = metric
                save_path = os.path.join(args.out_dir, "set_predictor_best.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_threshold": best_t,
                    "best_dchv": best_dchv,
                }, save_path)
                print(f"           [save] best cov-preserving point: "
                      f"t={best_t}  Δ|S|/n={best_dchv:.4f}  -> {save_path}")

        history.append(log)
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2, default=str)

    save_path = os.path.join(args.out_dir, "set_predictor_final.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "args": vars(args),
    }, save_path)
    print(f"\n[setpred] saved final -> {save_path}")


if __name__ == "__main__":
    main()
