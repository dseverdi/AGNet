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


def _load_setpredictor(device: str, ckpt_path: Path | str | None = None) -> SetPredictor:
    """Load a SetPredictor checkpoint. Defaults to the standard probe; pass an
    alternate path for ablation/multi-seed checkpoints.
    """
    path = Path(ckpt_path) if ckpt_path is not None else SETPRED_CKPT
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or ckpt.get("args") or {}
    H = cfg.get("hidden", cfg.get("predictor_hidden", 128))
    L = cfg.get("n_attn_layers", cfg.get("predictor_attn_layers", 3))
    HD = cfg.get("heads", cfg.get("predictor_heads", 8))
    H_ptr = cfg.get("ptr_emb_dim", cfg.get("hidden_size", 128))
    disable_ptr_emb = bool(cfg.get("disable_ptr_emb", False))
    model = SetPredictor(ptr_emb_dim=H_ptr, hidden=H, n_attn_layers=L, heads=HD,
                         disable_ptr_emb=disable_ptr_emb).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    if disable_ptr_emb:
        print(f"  [setpred] loaded with disable_ptr_emb=True from {path.name}")
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

    def _pick_named(records, name: str, target_low: float, target_high: float):
        """Pin a specific instance by name for a reproducible, vetted figure;
        fall back to the heuristic _pick if that record is absent."""
        for r in records:
            if r["name"] == name:
                return r
        print(f"  warn: pinned example {name!r} not found; using heuristic pick")
        return _pick(records, target_low, target_high)

    # Pinned, reviewer-vetted examples (see paper/scripts/find_worked_example.py).
    # The OOD pick rand-350-13 replaces randsimple-300-4: it is unambiguously OOD
    # (n=350, ~1.8x the n=198 training max) yet at t=0.20 recovers above feasibility
    # (seed 0.942 -> probe 0.996) at |S|/OPT 2.41 and the lowest guard density of any
    # clean candidate (|S|/n 0.37) -- not the old 6x-OPT tail outlier that flooded
    # 229/300 vertices.
    in_pick = _pick_named(in_records, "rand-62-44", 0.85, 0.95)
    ood_pick = _pick_named(ood_records, "rand-350-13", 0.55, 0.85)

    threshold = 0.20  # matches tab_headline's headline operating point

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
def build_per_polygon_all(device: str, batch_size: int = 32,
                          ckpt_path: str | None = None,
                          out_suffix: str = "") -> None:
    """For every polygon in dev_test and test, record name, n, OPT, and
    (S_size, cov) for the policy seed and for the probe at three thresholds.
    Single shared inference pass per polygon; thresholding is local.

    ckpt_path: alternate SetPredictor checkpoint (ablation / multi-seed runs).
    out_suffix: appended to the output JSON stem, e.g. '_noenc' or '_seed11'.
    """
    print(f"[per_polygon_all] starting (batch_size={batch_size}, "
          f"ckpt={ckpt_path or 'default'}, suffix='{out_suffix}')")
    t0 = time.perf_counter()
    pointer = _load_pointer(device)
    setpred = _load_setpredictor(device, ckpt_path)

    DATASET_PATH = os.getenv("DATASET_PATH")
    if not DATASET_PATH:
        print("  warn: DATASET_PATH unset; OPT will be null")

    thresholds = [0.20, 0.25, 0.30]
    splits = [
        ("dev_test", DEV_TEST_TRAJ, "dev",
         f"dist_dev_test{out_suffix}.json"),
        ("test_OOD", TEST_TRAJ, "test",
         f"dist_test_OOD{out_suffix}.json"),
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
#  classical_baselines.json  (greedy + LS context for tab_headline)
# ──────────────────────────────────────────────────────────────────────
def build_classical_baselines(device: str | None = None) -> None:
    """Aggregate classical greedy and local-search baselines on dev_test and
    OOD test, using CGAL exact coverage uniformly so the numbers are directly
    comparable to the policy/probe rows in tab_headline.

    Greedy solutions are sourced from precomputed files:
      - dev_test (367): results/v3/greedy_prune/greedy_solutions.json
                       (reported as CGAL exact in greedy_prune_report.json)
      - OOD test (2107): data/greedy_baseline_test.pkl  (we recompute CGAL)

    LS solutions live in rec["final"] of data/ls_trajectories_*.pkl
    (cached coverage there is disc-vis-500; we recompute CGAL).

    Output: paper/data/baseline_classical.json
    """
    print("[classical_baselines] starting")
    t0 = time.perf_counter()

    DATASET_PATH = os.getenv("DATASET_PATH")
    if not DATASET_PATH:
        print("  warn: DATASET_PATH unset; OPT will be null")

    # Load greedy solutions for both splits (different sources).
    greedy_dev_path = REPO_ROOT / "results/v3/greedy_prune/greedy_solutions.json"
    greedy_dev = json.loads(greedy_dev_path.read_text()) if greedy_dev_path.exists() else {}
    print(f"  greedy_dev: {len(greedy_dev)} solutions from {greedy_dev_path.name}")

    import pickle
    greedy_ood_path = REPO_ROOT / "data/greedy_baseline_test.pkl"
    with open(greedy_ood_path, "rb") as f:
        greedy_ood = pickle.load(f)
    print(f"  greedy_ood: {len(greedy_ood)} solutions from {greedy_ood_path.name}")

    def _greedy_lookup(split: str, name: str) -> list[int] | None:
        if split == "dev_test":
            entry = greedy_dev.get(name) or {}
            return entry.get("guards") or entry.get("solution") or entry.get("vertices")
        elif split == "test_OOD":
            # greedy_solutions.json covers ~40 OOD polygons; pkl covers ~10
            entry = greedy_dev.get(name) or greedy_ood.get(name) or {}
            return entry.get("guards")
        return None

    splits = [
        ("dev_test", DEV_TEST_TRAJ, "dev", "dist_dev_test"),
        ("test_OOD", TEST_TRAJ, "test", "dist_test_OOD"),
    ]

    out = {"splits": {}, "schema_version": 1,
           "notes": "CGAL exact coverage used uniformly. dev_test greedy reuses "
                    "cached CGAL coverage from greedy_prune_report (all 367 polygons). "
                    "OOD greedy is partial: only polygons present in greedy_solutions.json "
                    "or greedy_baseline_test.pkl are included (~40-50/2107). "
                    "LS coverage recomputed from rec['final'] for all polygons."}

    for split_name, traj_path, sol_subdir, _ in splits:
        sol_dir = os.path.join(DATASET_PATH, sol_subdir) if DATASET_PATH else None
        with open(traj_path, "rb") as f:
            records = pickle.load(f)["records"]
        print(f"  split={split_name}: {len(records)} polygons")

        rows = {"greedy": [], "ls": []}
        n_missing_greedy = 0
        for i, rec in enumerate(records):
            pts_np = np.asarray(rec["points"], dtype=np.float32)
            n_v = rec["n"]
            opt_sol = _read_opt_solution(sol_dir, rec["name"]) if sol_dir else None
            opt_size = len(opt_sol) if opt_sol else None

            # LS solution: recompute CGAL coverage on rec["final"].
            ls_idxs = [int(j) for j in rec["final"] if 0 <= int(j) < n_v]
            ls_cov = _coverage_exact(pts_np, ls_idxs, rec["name"])
            rows["ls"].append({"name": rec["name"], "n": int(n_v), "OPT": opt_size,
                                "S_size": len(ls_idxs), "cov": float(ls_cov)})

            # Greedy: lookup precomputed guard set, recompute CGAL.
            g_idxs = _greedy_lookup(split_name, rec["name"])
            if g_idxs is None:
                n_missing_greedy += 1
                continue
            g_idxs = [int(j) for j in g_idxs if 0 <= int(j) < n_v]
            # greedy_solutions.json has CGAL-exact coverage; reuse if present.
            # For OOD entries only in pkl, recompute CGAL.
            cached = greedy_dev.get(rec["name"], {}).get("coverage")
            g_cov = float(cached) if cached is not None else _coverage_exact(
                pts_np, g_idxs, rec["name"])
            rows["greedy"].append({"name": rec["name"], "n": int(n_v), "OPT": opt_size,
                                    "S_size": len(g_idxs), "cov": float(g_cov)})

            if (i + 1) % 200 == 0:
                dt = time.perf_counter() - t0
                print(f"    {i+1}/{len(records)} polygons, {dt:.1f}s elapsed")

        if n_missing_greedy:
            print(f"  warn: {n_missing_greedy} polygons missing from greedy source")

        def _agg(rs: list[dict]) -> dict:
            if not rs:
                return {}
            covs = [r["cov"] for r in rs]
            sn = [r["S_size"] / r["n"] for r in rs]
            sopt = [r["S_size"] / r["OPT"] for r in rs if r.get("OPT")]
            return {
                "n_polygons": len(rs),
                "mean_cov": float(np.mean(covs)),
                "min_cov": float(np.min(covs)),
                "mean_S_over_n": float(np.mean(sn)),
                "mean_S_over_OPT": float(np.mean(sopt)) if sopt else None,
                "n_below_095": int(sum(1 for c in covs if c < 0.95)),
            }

        out["splits"][split_name] = {
            "greedy": _agg(rows["greedy"]),
            "ls": _agg(rows["ls"]),
            "per_polygon": rows,
        }

    out_path = PAPER_DATA / "baseline_classical.json"
    out_path.write_text(json.dumps(out, indent=2))
    dt = time.perf_counter() - t0
    print(f"  wrote {out_path} ({dt:.1f}s)")


# ──────────────────────────────────────────────────────────────────────
#  encoder_linear_probe.json  (quantitative C1 evidence)
# ──────────────────────────────────────────────────────────────────────
def build_encoder_linear_probe(device: str, n_polygons: int = 200) -> None:
    """Train logistic regression over frozen encoder embeddings to predict
    LS-target membership. Reports cross-validated ROC-AUC and PR-AUC with
    polygon-level grouping (no vertex leakage across folds).

    This replaces the soft "visible separation" claim in §7 with a number.
    """
    print(f"[encoder_linear_probe] starting (n_polygons={n_polygons})")
    t0 = time.perf_counter()
    pointer = _load_pointer(device)

    with open(DEV_TEST_TRAJ, "rb") as f:
        records = pickle.load(f)["records"]
    sample = records[:n_polygons]

    all_X, all_y, groups = [], [], []
    for poly_idx, rec in enumerate(sample):
        pts = torch.tensor(np.asarray(rec["points"], dtype=np.float32),
                           device=device).unsqueeze(0)
        lengths = torch.tensor([rec["n"]], device=device)
        with torch.no_grad():
            emb = extract_pointer_embeddings(pointer, pts, lengths)[0]  # (n, 128)
        emb_np = emb.cpu().numpy()
        labels = np.zeros(rec["n"], dtype=np.int32)
        for j in rec.get("final", []):
            if 0 <= int(j) < rec["n"]:
                labels[int(j)] = 1
        all_X.append(emb_np)
        all_y.append(labels)
        groups.append(np.full(rec["n"], poly_idx, dtype=np.int64))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    g = np.concatenate(groups, axis=0)
    print(f"  collected {X.shape[0]} per-vertex embeddings "
          f"({int(y.sum())} positive / {int((1-y).sum())} negative) "
          f"from {len(sample)} polygons")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import StandardScaler

    n_splits = 5
    gkf = GroupKFold(n_splits=n_splits)
    roc_aucs, pr_aucs = [], []
    for fold, (tr, te) in enumerate(gkf.split(X, y, g)):
        scaler = StandardScaler().fit(X[tr])
        Xtr = scaler.transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        clf.fit(Xtr, y[tr])
        scores = clf.predict_proba(Xte)[:, 1]
        roc = roc_auc_score(y[te], scores)
        prc = average_precision_score(y[te], scores)
        roc_aucs.append(float(roc)); pr_aucs.append(float(prc))
        print(f"  fold {fold+1}/{n_splits}: ROC-AUC={roc:.4f}  PR-AUC={prc:.4f}  "
              f"(test polygons={len(np.unique(g[te]))})")

    out = {
        "n_polygons": int(len(sample)),
        "n_vertices": int(X.shape[0]),
        "n_positive": int(y.sum()),
        "n_splits": n_splits,
        "split_strategy": "GroupKFold on polygon",
        "classifier": "LogisticRegression(C=1.0, class_weight=balanced)",
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std": float(np.std(roc_aucs)),
        "pr_auc_mean": float(np.mean(pr_aucs)),
        "pr_auc_std": float(np.std(pr_aucs)),
        "roc_auc_per_fold": roc_aucs,
        "pr_auc_per_fold": pr_aucs,
    }
    out_path = PAPER_DATA / "encoder_linear_probe.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  ROC-AUC = {out['roc_auc_mean']:.4f} ± {out['roc_auc_std']:.4f}  "
          f"PR-AUC = {out['pr_auc_mean']:.4f} ± {out['pr_auc_std']:.4f}")
    print(f"  wrote {out_path} ({time.perf_counter() - t0:.1f}s)")


# ──────────────────────────────────────────────────────────────────────
#  multi_seed_summary.json  (M1: probe-seed-variance error bars)
# ──────────────────────────────────────────────────────────────────────
def _aggregate_seed_files(files: list, label: str) -> tuple[list, dict]:
    """Aggregate a list of per-polygon dist_*.json files into (per_seed, summary)
    where summary holds mean/std across seeds for each metric. Shared by the full
    probe and the no-encoder ablation."""
    per_seed = []
    for fp in files:
        d = json.loads(fp.read_text())["polygons"]
        agg = {}
        for m in ("seed", "probe_t020", "probe_t025", "probe_t030"):
            covs = [r[m]["cov"] for r in d]
            sn = [r[m]["S_size"] / r["n"] for r in d]
            sopt = [r[m]["S_size"] / r["OPT"] for r in d if r.get("OPT")]
            agg[m] = {
                "mean_cov": float(np.mean(covs)),
                "mean_S_over_n": float(np.mean(sn)),
                "mean_S_over_OPT": float(np.mean(sopt)) if sopt else None,
                "n_below_095": int(sum(1 for c in covs if c < 0.95)),
            }
        per_seed.append({"file": fp.name, "agg": agg})
        print(f"  {label}: loaded {fp.name}")

    def _mean_std(method: str, key: str):
        vals = [s["agg"][method][key] for s in per_seed
                if s["agg"][method][key] is not None]
        if not vals:
            return (None, None)
        return (float(np.mean(vals)), float(np.std(vals)))

    summary = {}
    for m in ("seed", "probe_t020", "probe_t025", "probe_t030"):
        summary[m] = {}
        for key in ("mean_cov", "mean_S_over_n", "mean_S_over_OPT", "n_below_095"):
            mu, sd = _mean_std(m, key)
            summary[m][f"{key}_mean"] = mu
            summary[m][f"{key}_std"] = sd
    return per_seed, summary


def _collect_seed_files(base_name: str, glob_stem: str) -> list:
    """Files for one variant: the seed-1234 base (no suffix) followed by the
    seed{11,22,33} retrainings. Returns [] if none exist."""
    files = sorted(PAPER_DATA.glob(f"{glob_stem}*.json"))
    base_path = PAPER_DATA / base_name
    if base_path.exists():
        files = [base_path] + list(files)
    return files


def build_multi_seed_summary(device: str) -> None:
    """Aggregate multi-seed results from dist_*_seed{seed}.json files produced
    by re-running `per_polygon_all` with different SP_CKPT, AND the original
    single-seed runs in dist_{dev_test,test_OOD}{,_noenc}.json (treated as
    'seed1234', the standard/no_encoder checkpoint seed). Yields mean ± std over
    all available seeds per split, for BOTH the full probe (`summary`) and the
    no-encoder ablation (`no_encoder_summary`).
    """
    print("[multi_seed_summary] starting")
    t0 = time.perf_counter()
    out = {"splits": {}, "schema_version": 3,
           "notes": "Seed 1234 corresponds to the standard / no_encoder "
                    "checkpoints (dist_*{,_noenc}.json); the rest come from "
                    "seed{11,22,33} retrainings. `summary` is the full probe, "
                    "`no_encoder_summary` the matched-capacity ablation."}

    for split, base, glob_stem, ne_base, ne_glob in [
        ("dev_test", "dist_dev_test.json", "dist_dev_test_seed",
         "dist_dev_test_noenc.json", "dist_dev_test_noenc_seed"),
        ("test_OOD", "dist_test_OOD.json", "dist_test_OOD_seed",
         "dist_test_OOD_noenc.json", "dist_test_OOD_noenc_seed"),
    ]:
        files = _collect_seed_files(base, glob_stem)
        if not files:
            print(f"  {split}: no files matching {glob_stem}*.json — skipping")
            continue
        per_seed, summary = _aggregate_seed_files(files, split)
        split_out = {"per_seed": per_seed, "summary": summary,
                     "n_seeds": len(per_seed)}

        # No-encoder ablation (same structure, separate seed set).
        ne_files = _collect_seed_files(ne_base, ne_glob)
        if ne_files:
            ne_per_seed, ne_summary = _aggregate_seed_files(ne_files, f"{split}/noenc")
            split_out["no_encoder_per_seed"] = ne_per_seed
            split_out["no_encoder_summary"] = ne_summary
            split_out["n_seeds_no_encoder"] = len(ne_per_seed)

        out["splits"][split] = split_out

    out_path = PAPER_DATA / "multi_seed_summary.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  wrote {out_path} ({time.perf_counter() - t0:.1f}s)")


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
        # Optional env vars to evaluate alternate checkpoints (no-encoder
        # ablation or multi-seed probes) and tag the outputs.
        ckpt = os.getenv("SP_CKPT")  # absolute or repo-relative path
        suffix = os.getenv("SP_SUFFIX", "")
        _run(build_per_polygon_all, "per_polygon_all", device,
             batch_size=32, ckpt_path=ckpt, out_suffix=suffix)
        print()
    if "classical_baselines" in steps:
        _run(build_classical_baselines, "classical_baselines", device)
        print()
    if "encoder_linear_probe" in steps:
        n = int(os.getenv("LP_N_POLYGONS", "200"))
        _run(build_encoder_linear_probe, "encoder_linear_probe", device,
             n_polygons=n)
        print()
    if "multi_seed_summary" in steps:
        _run(build_multi_seed_summary, "multi_seed_summary", device)
