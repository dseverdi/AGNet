"""Probe capacity ladder (reviewer R3): does an intermediate-capacity probe
recover the guard-relevant signal, or does it need the full 464K Transformer?

We score four probe capacities on ONE held-out protocol — fit/train on the
train split, evaluate per-vertex ROC-AUC / PR-AUC on the 362 leak-free
dev_test polygons, against the same LS-target labels (rec['final']) used by
the linear probe in build_encoder_linear_probe:

  linear   logistic regression on frozen encoder features (held-out here;
           cf. the cross-validated 0.843 in encoder_linear_probe.json)
  mlp      SetPredictor with --predictor-attn-layers 0  (DeepSets-style MLP:
           per-vertex Linear + masked mean-pool context + MLP head, no attention)
  attn1    SetPredictor with one self-attention block
  full     the released 3-block, 8-head, 464K-param SetPredictor

The AUC is a representation-readout measure independent of the probability
threshold, so it sidesteps the calibration-mismatch worry (R2). Downstream
guard-count columns (#Cov<0.99, |S|/OPT) come separately from
build_per_polygon_all + build_ladder_table.py.

Output: paper/data/probe_ladder.json

Usage:
  python paper/scripts/eval_probe_ladder.py
  python paper/scripts/eval_probe_ladder.py --linear-fit-polys 600
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

# Reuse the exact loaders / dataset / batching the rest of the paper pipeline uses.
from build_paper_data import (
    REPO_ROOT, PAPER_DATA, DEV_TEST_TRAJ,
    _load_pointer, _load_setpredictor,
)
from set_predictor import SetPredictor, extract_pointer_embeddings
from train_set_predictor import SetPredDataset, make_batch

TRAIN_TRAJ = REPO_ROOT / "data/ls_trajectories_train.pkl"

# Capacity name -> list of checkpoint dirs (seed replicates). seed 1234 of the
# full probe is the released 'standard' run.
LADDER = {
    "mlp":   [f"checkpoints/set_predictor/ladder_mlp_seed{s}"   for s in (1234, 11, 22, 33)],
    "attn1": [f"checkpoints/set_predictor/ladder_attn1_seed{s}" for s in (1234, 11, 22, 33)],
    "full":  ["checkpoints/set_predictor/standard",
              "checkpoints/set_predictor/seed11",
              "checkpoints/set_predictor/seed22",
              "checkpoints/set_predictor/seed33"],
}


def _ckpt_file(d: Path) -> Path | None:
    """Prefer the dev-selected best checkpoint; fall back to final-epoch."""
    for name in ("set_predictor_best.pt", "set_predictor_final.pt"):
        p = d / name
        if p.exists():
            return p
    return None


def _labels(rec) -> np.ndarray:
    y = np.zeros(rec["n"], dtype=np.int32)
    for j in rec.get("final", []):
        if 0 <= int(j) < rec["n"]:
            y[int(j)] = 1
    return y


def _collect_vertex_features(pointer, records, device):
    """Per-vertex frozen encoder features (N, 128) and labels (N,)."""
    all_X, all_y = [], []
    for rec in records:
        pts = torch.tensor(np.asarray(rec["points"], dtype=np.float32),
                           device=device).unsqueeze(0)
        lengths = torch.tensor([rec["n"]], device=device)
        with torch.no_grad():
            emb = extract_pointer_embeddings(pointer, pts, lengths)[0]
        all_X.append(emb.cpu().numpy())
        all_y.append(_labels(rec))
    return np.concatenate(all_X, 0), np.concatenate(all_y, 0)


def _auc(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score
    return float(roc_auc_score(y, scores)), float(average_precision_score(y, scores))


def eval_linear_reference(pointer, dev_records, device, n_fit_polys: int) -> dict:
    """Held-out logistic regression: fit on train-split frozen features,
    evaluate on the dev_test vertices. Matched to the SetPredictor eval set."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    with open(TRAIN_TRAJ, "rb") as f:
        train_records = pickle.load(f)["records"]
    train_records = [r for r in train_records if r.get("final")][:n_fit_polys]
    print(f"  [linear] fitting on {len(train_records)} train polygons")
    Xtr, ytr = _collect_vertex_features(pointer, train_records, device)
    Xte, yte = _collect_vertex_features(pointer, dev_records, device)

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(scaler.transform(Xtr), ytr)
    scores = clf.predict_proba(scaler.transform(Xte))[:, 1]
    roc, prc = _auc(yte, scores)
    n_params = int(Xtr.shape[1] + 1)  # 128 weights + bias
    print(f"  [linear] held-out ROC-AUC={roc:.4f}  PR-AUC={prc:.4f}  params={n_params}")
    return {
        "name": "linear",
        "label": "Linear (logistic regression)",
        "n_params": n_params,
        "n_seeds": 1,
        "roc_auc_mean": roc, "roc_auc_std": 0.0,
        "pr_auc_mean": prc, "pr_auc_std": 0.0,
        "roc_auc_per_seed": [roc],
    }


def eval_setpred_capacity(name: str, dirs, pointer, ds, device) -> dict:
    """Per-vertex AUC of a SetPredictor capacity, aggregated over seed replicates.
    ``ds`` is a shared SetPredDataset so every rung is scored on identical records."""
    # make_batch requires all indices in a batch to share polygon size n.
    buckets = ds.buckets_by_size()

    rocs, prcs, n_params = [], [], None
    for d in dirs:
        ckpt = _ckpt_file(REPO_ROOT / d)
        if ckpt is None:
            print(f"  [{name}] MISSING checkpoint in {d} — skipping seed")
            continue
        model = _load_setpredictor(device, ckpt)
        if n_params is None:
            n_params = model.num_params()

        scores, labels = [], []
        with torch.no_grad():
            for _sz, ids in buckets.items():
                for start in range(0, len(ids), 32):
                    chunk = ids[start:start + 32]
                    ptr_emb, pts, in_S, pad, _, names, _ = make_batch(
                        ds, chunk, pointer, device)
                    probs = torch.sigmoid(model(ptr_emb, pts, in_S, pad))
                    for b, idx in enumerate(chunk):
                        rec = ds.records[idx]
                        nv = rec["n"]
                        scores.append(probs[b, :nv].cpu().numpy())
                        labels.append(_labels(rec))
        y = np.concatenate(labels, 0)
        s = np.concatenate(scores, 0)
        roc, prc = _auc(y, s)
        rocs.append(roc); prcs.append(prc)
        print(f"  [{name}] {Path(d).name}: ROC-AUC={roc:.4f}  PR-AUC={prc:.4f}")

    return {
        "name": name,
        "label": {"mlp": "Attention-free MLP (0 layers)",
                  "attn1": "1 self-attention layer",
                  "full": "Full SetPredictor (3 layers)"}.get(name, name),
        "n_params": int(n_params) if n_params else None,
        "n_seeds": len(rocs),
        "roc_auc_mean": float(np.mean(rocs)) if rocs else None,
        "roc_auc_std": float(np.std(rocs)) if rocs else None,
        "pr_auc_mean": float(np.mean(prcs)) if prcs else None,
        "pr_auc_std": float(np.std(prcs)) if prcs else None,
        "roc_auc_per_seed": rocs,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--linear-fit-polys", type=int, default=600)
    ap.add_argument("--out", type=str, default=str(PAPER_DATA / "probe_ladder.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[probe_ladder] device={device}")
    t0 = time.perf_counter()
    pointer = _load_pointer(device)

    # One shared eval set so every rung is scored on identical records (those
    # with a non-empty seed and LS target, as SetPredDataset filters).
    ds = SetPredDataset(str(DEV_TEST_TRAJ))
    dev_records = ds.records

    rows = [eval_linear_reference(pointer, dev_records, device, args.linear_fit_polys)]
    for name, dirs in LADDER.items():
        rows.append(eval_setpred_capacity(name, dirs, pointer, ds, device))

    out = {
        "protocol": "held-out: fit/train on train split, per-vertex AUC on "
                    "362 leak-free dev_test polygons (labels = rec['final'], "
                    "the LS target); matched to encoder_linear_probe.json labels",
        "eval_set": "ls_trajectories_dev_test_clean.pkl",
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[probe_ladder] wrote {args.out} ({time.perf_counter()-t0:.1f}s)")
    for r in rows:
        m = r["roc_auc_mean"]
        s = r["roc_auc_std"] or 0.0
        ms = f"{m:.4f} ± {s:.4f}" if m is not None else "n/a"
        print(f"  {r['label']:<34} params={str(r['n_params']):>7}  ROC-AUC={ms}")


if __name__ == "__main__":
    main()
