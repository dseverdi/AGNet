"""Generate the four data JSONs needed by the new figures in build_figures.py.

Outputs into paper/data/:
  - encoder_pca.json
  - worked_examples.json
  - setpred_test_OOD_per_polygon.json
  - (po_agp_training.json is skipped: per-epoch metric logs not preserved)

Each step is independent: a failure in one does not stop the others.

Usage:
  /home/dseverdi/.conda/envs/MLAG/bin/python paper/scripts/build_paper_data.py
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DATA = REPO_ROOT / "paper" / "data"
PAPER_DATA.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from po_agp import create_agp_model, _read_opt_solution
from set_predictor import SetPredictor, extract_pointer_embeddings
from train_set_predictor import SetPredDataset, make_batch
from utils import evaluate_polygon_visibility_numpy_wo_gt

PTR_CKPT = REPO_ROOT / "checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt"
SETPRED_CKPT = REPO_ROOT / "checkpoints/set_predictor/standard/set_predictor_best.pt"
DEV_TEST_TRAJ = REPO_ROOT / "data/ls_trajectories_dev_test.pkl"
TEST_TRAJ = REPO_ROOT / "data/ls_trajectories_test.pkl"


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_pointer(device: str):
    pointer = create_agp_model(128, 128, 1, 10.0, use_tanh=True, temperature=1.0)
    ckpt = torch.load(PTR_CKPT, map_location=device, weights_only=False)
    psd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    pointer.load_state_dict(psd, strict=False)
    pointer.to(device).eval()
    for p in pointer.parameters():
        p.requires_grad_(False)
    return pointer


def _load_setpredictor(device: str) -> SetPredictor:
    ckpt = torch.load(SETPRED_CKPT, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {}
    H = cfg.get("hidden", 128)
    L = cfg.get("n_attn_layers", 3)
    HD = cfg.get("heads", 8)
    H_ptr = cfg.get("ptr_emb_dim", 128)
    model = SetPredictor(ptr_emb_dim=H_ptr, hidden=H, n_attn_layers=L, heads=HD).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _coverage_exact(pts_np: np.ndarray, guards: list, name: str) -> float:
    if not guards:
        return 0.0
    try:
        return float(evaluate_polygon_visibility_numpy_wo_gt(
            pts_np, np.array(guards, dtype=np.int64), name,
        ))
    except Exception as exc:
        print(f"  warn: coverage failed for {name}: {exc}")
        return 0.0


# ──────────────────────────────────────────────────────────────────────
#  encoder_pca.json
# ──────────────────────────────────────────────────────────────────────
def build_encoder_pca(device: str, n_polygons: int = 80) -> None:
    """Run frozen pointer encoder on dev_test polygons; PCA-project the
    per-vertex 128-d embeddings down to 2D; label each vertex by whether
    it is in the LS-derived target set (rec['final']).
    """
    print("[encoder_pca] starting")
    t0 = time.perf_counter()
    pointer = _load_pointer(device)

    with open(DEV_TEST_TRAJ, "rb") as f:
        records = pickle.load(f)["records"]

    sample = records[:n_polygons]
    all_emb, all_label = [], []
    for rec in sample:
        pts = torch.tensor(np.asarray(rec["points"], dtype=np.float32),
                           device=device).unsqueeze(0)  # (1, n, 2)
        lengths = torch.tensor([rec["n"]], device=device)
        with torch.no_grad():
            emb = extract_pointer_embeddings(pointer, pts, lengths)[0]  # (n, 128)
        emb_np = emb.cpu().numpy()
        label = np.zeros(rec["n"], dtype=np.int32)
        for idx in rec["final"]:
            if 0 <= idx < rec["n"]:
                label[idx] = 1
        all_emb.append(emb_np)
        all_label.append(label)

    X = np.concatenate(all_emb, axis=0)  # (sum_n, 128)
    y = np.concatenate(all_label, axis=0)
    print(f"  collected {X.shape[0]} per-vertex embeddings "
          f"({y.sum()} guard / {len(y) - y.sum()} non-guard)")

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pts_2d = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_.tolist()
    print(f"  PCA: ev = {ev[0]:.3f}, {ev[1]:.3f}")

    out = {
        "points_2d": pts_2d.tolist(),
        "labels": y.tolist(),
        "method": "pca",
        "explained_variance": ev,
        "n_polygons": int(n_polygons),
        "n_vertices": int(X.shape[0]),
    }
    out_path = PAPER_DATA / "encoder_pca.json"
    out_path.write_text(json.dumps(out))
    print(f"  wrote {out_path} ({time.perf_counter() - t0:.1f}s)")


# ──────────────────────────────────────────────────────────────────────
#  worked_examples.json
# ──────────────────────────────────────────────────────────────────────
def build_worked_examples(device: str) -> None:
    """Pick two polygons (one in-distribution, one OOD), produce seed/probe/opt
    indices and coverages for each.
    """
    print("[worked_examples] starting")
    t0 = time.perf_counter()
    pointer = _load_pointer(device)
    setpred = _load_setpredictor(device)

    with open(DEV_TEST_TRAJ, "rb") as f:
        in_records = pickle.load(f)["records"]
    with open(TEST_TRAJ, "rb") as f:
        ood_records = pickle.load(f)["records"]

    DATASET_PATH = os.getenv("DATASET_PATH")
    dev_sol_dir = os.path.join(DATASET_PATH, "dev") if DATASET_PATH else None
    test_sol_dir = os.path.join(DATASET_PATH, "test") if DATASET_PATH else None

    # Choose one example from each split where the seed has a clear coverage gap
    # (so the figure visibly shows the probe contributing).
    def _pick(records, target_low: float, target_high: float):
        candidates = [r for r in records
                      if target_low <= r["seed_cov"] <= target_high
                      and r["n"] >= 12]
        if not candidates:
            candidates = records
        # pick median-sized in the eligible set
        candidates.sort(key=lambda r: r["n"])
        return candidates[len(candidates) // 2]

    in_pick = _pick(in_records, 0.85, 0.95)
    ood_pick = _pick(ood_records, 0.55, 0.85)

    threshold = 0.30

    def _probe_indices(rec) -> tuple[list[int], float]:
        pts = torch.tensor(np.asarray(rec["points"], dtype=np.float32),
                           device=device).unsqueeze(0)
        lengths = torch.tensor([rec["n"]], device=device)
        in_S_init = torch.zeros(1, rec["n"], dtype=torch.bool, device=device)
        for idx in rec["seed"]:
            if 0 <= idx < rec["n"]:
                in_S_init[0, idx] = True
        pad = torch.zeros(1, rec["n"], dtype=torch.bool, device=device)
        with torch.no_grad():
            ptr_emb = extract_pointer_embeddings(pointer, pts, lengths)
            logits = setpred(ptr_emb, pts, in_S_init, pad)
            probs = torch.sigmoid(logits)
            keep = (probs >= threshold) & (~pad)
        idxs = keep[0].nonzero(as_tuple=True)[0].cpu().tolist()
        cov = _coverage_exact(pts[0].cpu().numpy(), idxs, rec["name"])
        return idxs, cov

    examples = []
    for rec, split, sol_dir in [(in_pick, "dev_test", dev_sol_dir),
                                (ood_pick, "test", test_sol_dir)]:
        probe_idxs, probe_cov = _probe_indices(rec)
        opt_idxs = _read_opt_solution(sol_dir, rec["name"]) if sol_dir else None
        opt_cov = None
        if opt_idxs:
            opt_cov = _coverage_exact(np.asarray(rec["points"], dtype=np.float32),
                                      list(opt_idxs), rec["name"])
        examples.append({
            "name": rec["name"],
            "split": split,
            "n": int(rec["n"]),
            "points": np.asarray(rec["points"], dtype=float).tolist(),
            "seed_idxs": list(rec["seed"]),
            "probe_idxs": probe_idxs,
            "opt_idxs": list(opt_idxs) if opt_idxs else None,
            "seed_coverage": float(rec["seed_cov"]),
            "probe_coverage": float(probe_cov),
            "opt_coverage": float(opt_cov) if opt_cov is not None else None,
        })
        print(f"  picked {rec['name']} ({split}, n={rec['n']}): "
              f"seed_cov={rec['seed_cov']:.3f} "
              f"probe_cov={probe_cov:.3f} "
              f"opt_cov={opt_cov if opt_cov is None else f'{opt_cov:.3f}'}")

    out_path = PAPER_DATA / "worked_examples.json"
    out_path.write_text(json.dumps({"examples": examples}))
    print(f"  wrote {out_path} ({time.perf_counter() - t0:.1f}s)")


# ──────────────────────────────────────────────────────────────────────
#  setpred_test_OOD_per_polygon.json
# ──────────────────────────────────────────────────────────────────────
def build_per_polygon_ood(device: str, threshold: float = 0.20,
                           batch_size: int = 32, limit: int | None = None) -> None:
    """Run SetPredictor at t=0.20 on the OOD test split and dump per-polygon
    (name, n, seed_cov, probe_cov_t020).
    """
    print(f"[per_polygon_ood] starting (t={threshold}, batch_size={batch_size})")
    t0 = time.perf_counter()
    pointer = _load_pointer(device)
    setpred = _load_setpredictor(device)

    ds = SetPredDataset(str(TEST_TRAJ))
    n = len(ds.records)
    if limit is not None:
        n = min(n, limit)
    indices = list(range(n))

    from collections import defaultdict
    buckets = defaultdict(list)
    for i in indices:
        buckets[ds.records[i]["n"]].append(i)

    out_rows = []
    n_done = 0
    with torch.no_grad():
        for poly_n, ids in buckets.items():
            for start in range(0, len(ids), batch_size):
                chunk = ids[start: start + batch_size]
                ptr_emb, pts, in_S_init, pad, _, names, _ = make_batch(
                    ds, chunk, pointer, device,
                )
                logits = setpred(ptr_emb, pts, in_S_init, pad)
                probs = torch.sigmoid(logits)
                keep_mask = (probs >= threshold) & (~pad)

                for b, idx in enumerate(chunk):
                    rec = ds.records[idx]
                    pts_np = pts[b].cpu().numpy()[: rec["n"]]
                    keep = keep_mask[b].nonzero(as_tuple=True)[0].cpu().tolist()
                    probe_cov = _coverage_exact(pts_np, keep, names[b])
                    seed_cov = float(rec.get("seed_cov", 0.0))
                    out_rows.append({
                        "name": rec["name"],
                        "n": int(rec["n"]),
                        "seed_cov": seed_cov,
                        "probe_cov_t020": probe_cov,
                    })
                    n_done += 1
                    if n_done % 100 == 0:
                        dt = time.perf_counter() - t0
                        print(f"  {n_done}/{n} polygons, {dt:.1f}s elapsed")

    out_path = PAPER_DATA / "setpred_test_OOD_per_polygon.json"
    out_path.write_text(json.dumps({"polygons": out_rows}))
    print(f"  wrote {out_path} ({n_done} polygons, {time.perf_counter() - t0:.1f}s)")


# ──────────────────────────────────────────────────────────────────────
#  dist_dev_test.json + dist_test_OOD.json (per-polygon, for box plots)
# ──────────────────────────────────────────────────────────────────────
def build_per_polygon_all(device: str, batch_size: int = 32) -> None:
    """For every polygon in dev_test and test, record name, n, OPT, and
    (S_size, cov) for the policy seed and for the probe at three thresholds.
    Single shared inference pass per polygon; thresholding is local.
    """
    print(f"[per_polygon_all] starting (batch_size={batch_size})")
    t0 = time.perf_counter()
    pointer = _load_pointer(device)
    setpred = _load_setpredictor(device)

    DATASET_PATH = os.getenv("DATASET_PATH")
    if not DATASET_PATH:
        print("  warn: DATASET_PATH unset; OPT will be null")

    thresholds = [0.20, 0.25, 0.30]
    splits = [
        ("dev_test", DEV_TEST_TRAJ, "dev", "dist_dev_test.json"),
        ("test_OOD", TEST_TRAJ, "test", "dist_test_OOD.json"),
    ]

    from collections import defaultdict

    for split_name, traj_path, sol_subdir, out_name in splits:
        sol_dir = os.path.join(DATASET_PATH, sol_subdir) if DATASET_PATH else None
        print(f"  split={split_name}: traj={traj_path.name}, sol_dir={sol_dir}")
        ds = SetPredDataset(str(traj_path))
        n_total = len(ds.records)

        buckets = defaultdict(list)
        for i in range(n_total):
            buckets[ds.records[i]["n"]].append(i)

        out_rows = []
        n_done = 0
        with torch.no_grad():
            for poly_n, ids in buckets.items():
                for start in range(0, len(ids), batch_size):
                    chunk = ids[start: start + batch_size]
                    ptr_emb, pts, in_S_init, pad, _, names, _ = make_batch(
                        ds, chunk, pointer, device,
                    )
                    logits = setpred(ptr_emb, pts, in_S_init, pad)
                    probs = torch.sigmoid(logits)

                    for b, idx in enumerate(chunk):
                        rec = ds.records[idx]
                        n_v = rec["n"]
                        pts_np = pts[b].cpu().numpy()[:n_v]

                        opt_sol = (_read_opt_solution(sol_dir, rec["name"])
                                   if sol_dir else None)
                        opt_size = len(opt_sol) if opt_sol else None

                        seed_idxs = [int(i) for i in rec["seed"] if 0 <= int(i) < n_v]
                        row = {
                            "name": rec["name"],
                            "n": int(n_v),
                            "OPT": opt_size,
                            "seed": {
                                "S_size": len(seed_idxs),
                                "cov": float(rec.get("seed_cov", 0.0)),
                            },
                        }

                        for t in thresholds:
                            keep = (probs[b] >= t) & (~pad[b])
                            keep_idxs = keep[:n_v].nonzero(as_tuple=True)[0].cpu().tolist()
                            cov = _coverage_exact(pts_np, keep_idxs, rec["name"])
                            row[f"probe_t{int(round(t * 100)):03d}"] = {
                                "S_size": len(keep_idxs),
                                "cov": cov,
                            }

                        out_rows.append(row)
                        n_done += 1
                        if n_done % 100 == 0:
                            dt = time.perf_counter() - t0
                            print(f"    {n_done}/{n_total} polygons, {dt:.1f}s elapsed")

        out_path = PAPER_DATA / out_name
        out_path.write_text(json.dumps({"polygons": out_rows}))
        print(f"  wrote {out_path} ({len(out_rows)} polygons)")

    print(f"  total: {time.perf_counter() - t0:.1f}s")


# ──────────────────────────────────────────────────────────────────────
#  Dispatcher
# ──────────────────────────────────────────────────────────────────────
def _run(fn, name: str, *args, **kwargs) -> None:
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        import traceback
        print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    import sys as _sys
    device = _device()
    print(f"device: {device}")
    print(f"pointer ckpt: {PTR_CKPT}")
    print(f"setpred ckpt: {SETPRED_CKPT}")
    print()
    steps = _sys.argv[1:] or [
        "encoder_pca", "worked_examples", "per_polygon_ood", "per_polygon_all",
    ]
    if "encoder_pca" in steps:
        _run(build_encoder_pca, "encoder_pca", device, n_polygons=80)
        print()
    if "worked_examples" in steps:
        _run(build_worked_examples, "worked_examples", device)
        print()
    if "per_polygon_ood" in steps:
        _run(build_per_polygon_ood, "per_polygon_ood", device, threshold=0.20,
             batch_size=32, limit=None)
        print()
    if "per_polygon_all" in steps:
        _run(build_per_polygon_all, "per_polygon_all", device, batch_size=32)
