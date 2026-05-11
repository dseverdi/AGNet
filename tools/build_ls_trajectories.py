"""Build a supervised dataset for the learned LS-editor.

For each training polygon:
  1. Run the frozen pointer (greedy decode) -> seed S_0.
  2. Run a *deterministic best-improvement* local search starting from S_0.
  3. Record each (state, action) tuple along the trajectory.
  4. Persist to pickle for the editor trainer.

Why best-improvement (vs. first-improvement)?
  Each state has a unique "best edit" target -> unique supervision label.
  First-improvement depends on random vertex order and gives noisy labels.

Action encoding (per step):
    {
        "state":   list[int],          # current guard set BEFORE the action
        "kind":    "remove" | "swap" | "add" | "stop",
        "remove":  int | None,         # vertex removed (remove or swap)
        "add":     int | None,         # vertex added   (add or swap)
        "delta_r": float,              # reward delta this step
    }

The final trajectory ends with kind="stop" so the editor learns when to halt.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
except ImportError:
    pass

from dataset import collate_fn
from po_agp import create_agp_model, prepare_datasets
from utils import (
    evaluate_polygon_visibility_numpy_wo_gt,
    get_or_build_disc_vis,
    save_disc_vis_cache,
    load_disc_vis_cache,
)


# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str,
                   default="checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt")
    p.add_argument("--split", choices=["train", "dev"], default="train")
    p.add_argument("--n-samples", type=int, default=-1,
                   help="-1 = full split")
    p.add_argument("--embedding-size", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--n-glimpses", type=int, default=1)
    p.add_argument("--tanh-exploration", type=float, default=10.0)
    p.add_argument("--tau", type=float, default=0.99)
    p.add_argument("--tau-penalty", type=float, default=3.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--disc-vis-samples", type=int, default=500)
    p.add_argument("--ls-max-iter", type=int, default=50)
    p.add_argument("--min-dr", type=float, default=1e-3,
                   help="Minimum reward improvement to accept a move. "
                        "Filters sub-noise tiny moves so the editor learns "
                        "meaningful edits, matching first-improvement LS "
                        "convergence (~12 edits median).")
    p.add_argument("--out", type=str,
                   default="data/ls_trajectories_train.pkl")
    p.add_argument("--disc-vis-cache-path", type=str,
                   default="data/disc_vis_cache.pkl",
                   help="Persisted disc_vis cache. Loaded at start, "
                        "saved after the run so subsequent processes "
                        "(training, eval) don't have to recompute.")
    p.add_argument("--monotone-coverage", action="store_true", default=True,
                   help="Reject moves that drop coverage")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
#  Best-improvement LS that emits a trajectory
# ──────────────────────────────────────────────────────────────────────
def ls_best_improvement_trajectory(
    pts: np.ndarray,
    seed: list[int],
    name: str,
    n: int,
    *,
    lam: float,
    tau: float,
    tau_penalty: float,
    disc_vis_samples: int,
    max_iter: int,
    monotone_coverage: bool,
    min_dr: float = 1e-3,
) -> tuple[list[int], float, list[dict]]:
    """Replays local_search_improve_disc with best-improvement SWAP and
    records each step as a trajectory entry.

    Returns (final_set, final_reward, trajectory).
    The trajectory's last entry has kind="stop".
    """
    disc = get_or_build_disc_vis(pts, name, n_samples=disc_vis_samples)
    if not disc.get("valid"):
        # Degenerate polygon: emit a no-op stop trajectory.
        return sorted(seed), 0.0, [{
            "state": sorted(seed),
            "kind": "stop",
            "remove": None, "add": None, "delta_r": 0.0,
        }]

    vis = disc["vis_matrix"]                   # (n, M) bool
    M = int(disc["n_samples"])

    def _r_scalar(cov: float, k: int) -> float:
        return cov - lam * k / max(1, n) - tau_penalty * max(0.0, tau - cov)

    def _r_vec(cov: np.ndarray, k: int) -> np.ndarray:
        return cov - (lam * k / max(1, n)) - tau_penalty * np.maximum(0.0, tau - cov)

    current = set(int(v) for v in seed if 0 <= int(v) < n)
    valid_arr = np.array(sorted(current), dtype=np.int32)
    guard_count = (
        vis[valid_arr].astype(np.int32).sum(axis=0)
        if len(valid_arr) else np.zeros(M, dtype=np.int32)
    )
    covered = guard_count > 0
    cur_cov = float(covered.sum()) / M
    cur_r = _r_scalar(cur_cov, len(current))

    trajectory: list[dict] = []

    for _ in range(max_iter):
        k = len(current)
        guards_arr = np.array(sorted(current), dtype=np.int32)
        cands_arr = np.array(
            [v for v in range(n) if v not in current], dtype=np.int32,
        )

        # Track the best move across all three operations.
        # Threshold is min_dr (not 1e-9) so we only record meaningful edits;
        # this matches the convergence behaviour of first-improvement LS
        # and keeps trajectory lengths in the 8–20 range.
        best_kind: Optional[str] = None
        best_dr: float = float(min_dr)
        best_remove: Optional[int] = None
        best_add: Optional[int] = None
        best_state_after_cov: float = 0.0
        best_state_after_count: Optional[np.ndarray] = None
        best_state_after_covered: Optional[np.ndarray] = None

        # ── REMOVE ───────────────────────────────────────────────────
        if k > 1 and len(guards_arr) > 0:
            new_gc = guard_count[None, :] - vis[guards_arr].astype(np.int32)
            new_covs = (new_gc > 0).sum(axis=1).astype(np.float32) / M
            r_vals = _r_vec(new_covs, k - 1)
            if monotone_coverage:
                r_vals = np.where(new_covs >= cur_cov - 1e-9, r_vals, -np.inf)
            idx = int(np.argmax(r_vals))
            dr = float(r_vals[idx]) - cur_r
            if dr > best_dr:
                best_kind = "remove"
                best_dr = dr
                best_remove = int(guards_arr[idx])
                best_add = None
                best_state_after_cov = float(new_covs[idx])
                best_state_after_count = guard_count - vis[best_remove].astype(np.int32)
                best_state_after_covered = best_state_after_count > 0

        # ── ADD ──────────────────────────────────────────────────────
        if len(cands_arr) > 0:
            new_covered_mat = covered[None, :] | vis[cands_arr]
            new_covs = new_covered_mat.sum(axis=1).astype(np.float32) / M
            r_vals = _r_vec(new_covs, k + 1)
            idx = int(np.argmax(r_vals))
            dr = float(r_vals[idx]) - cur_r
            if dr > best_dr:
                best_kind = "add"
                best_dr = dr
                best_remove = None
                best_add = int(cands_arr[idx])
                best_state_after_cov = float(new_covs[idx])
                best_state_after_count = guard_count + vis[best_add].astype(np.int32)
                best_state_after_covered = new_covered_mat[idx]

        # ── SWAP (best-improvement over all (g, c) pairs) ────────────
        if len(guards_arr) > 0 and len(cands_arr) > 0:
            best_swap_g: Optional[int] = None
            best_swap_c: Optional[int] = None
            best_swap_cov: float = 0.0
            best_swap_dr: float = best_dr

            # For each guard g, vectorise over candidates c.
            for gi in range(len(guards_arr)):
                g = int(guards_arr[gi])
                after_remove_gc = guard_count - vis[g].astype(np.int32)
                after_remove_covered = after_remove_gc > 0
                new_covered_mat = (
                    after_remove_covered[None, :] | vis[cands_arr]
                )
                new_covs = new_covered_mat.sum(axis=1).astype(np.float32) / M
                r_vals = _r_vec(new_covs, k)
                if monotone_coverage:
                    r_vals = np.where(
                        new_covs >= cur_cov - 1e-9, r_vals, -np.inf,
                    )
                idx = int(np.argmax(r_vals))
                dr = float(r_vals[idx]) - cur_r
                if dr > best_swap_dr:
                    best_swap_dr = dr
                    best_swap_g = g
                    best_swap_c = int(cands_arr[idx])
                    best_swap_cov = float(new_covs[idx])
            if best_swap_g is not None:
                best_kind = "swap"
                best_dr = best_swap_dr
                best_remove = best_swap_g
                best_add = best_swap_c
                best_state_after_cov = best_swap_cov
                # Recompute count/covered for the chosen swap (cheap).
                tmp_gc = guard_count - vis[best_swap_g].astype(np.int32)
                tmp_gc = tmp_gc + vis[best_swap_c].astype(np.int32)
                best_state_after_count = tmp_gc
                best_state_after_covered = tmp_gc > 0

        if best_kind is None:
            # Local optimum reached.
            break

        # Record the chosen action.
        trajectory.append({
            "state": sorted(int(v) for v in current),
            "kind": best_kind,
            "remove": best_remove,
            "add": best_add,
            "delta_r": float(best_dr),
        })

        # Apply.
        if best_remove is not None:
            current.discard(best_remove)
        if best_add is not None:
            current.add(best_add)
        guard_count = best_state_after_count
        covered = best_state_after_covered
        cur_cov = best_state_after_cov
        cur_r = _r_scalar(cur_cov, len(current))

    # Terminal STOP step.
    trajectory.append({
        "state": sorted(int(v) for v in current),
        "kind": "stop",
        "remove": None, "add": None, "delta_r": 0.0,
    })

    return sorted(int(v) for v in current), cur_r, trajectory


# ──────────────────────────────────────────────────────────────────────
#  Model loading
# ──────────────────────────────────────────────────────────────────────
def load_model(args, device):
    model = create_agp_model(
        args.embedding_size, args.hidden_size, args.n_glimpses,
        args.tanh_exploration, use_tanh=True, temperature=1.0,
    )
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    missing, _ = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [load] missing keys: {len(missing)}")
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    DATASET_PATH = os.getenv("DATASET_PATH")
    if DATASET_PATH is None:
        raise EnvironmentError("DATASET_PATH not set")
    split_dir = os.path.join(DATASET_PATH, args.split)
    other_dir = os.path.join(DATASET_PATH, "dev" if args.split == "train" else "train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[traj] device={device}  checkpoint={args.checkpoint}")
    print(f"[traj] split={args.split}  dir={split_dir}")
    print(f"[traj] LS reward: lam={args.lam} tau={args.tau} tau_pen={args.tau_penalty}")

    # prepare_datasets needs both paths; we'll just take the right one.
    train_ds, val_ds = prepare_datasets(
        split_dir if args.split == "train" else other_dir,
        split_dir if args.split == "dev"   else other_dir,
        normalize=True,
    )
    ds = train_ds if args.split == "train" else val_ds
    if args.n_samples > 0:
        ds.samples = ds.samples[: args.n_samples]
    print(f"[traj] using {len(ds)} polygons")

    # Load persisted disc_vis cache if available — saves rebuilding on
    # subsequent runs (e.g. after reboot or for the dev split).
    cache_path = args.disc_vis_cache_path
    if cache_path and not os.path.isabs(cache_path):
        cache_path = os.path.join(REPO_ROOT, cache_path)
    if cache_path and os.path.exists(cache_path):
        load_disc_vis_cache(cache_path)

    model = load_model(args, device)
    loader = DataLoader(
        ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    out_records: list[dict] = []
    n_stop_only = 0
    n_total_steps = 0
    edits_dist = []
    t0 = time.perf_counter()

    with torch.no_grad():
        for i, (batch_data, pad_mask, lengths, names) in enumerate(loader):
            batch_data = batch_data.to(device)
            pad_mask = pad_mask.to(device)
            lens_t = torch.tensor(lengths, dtype=torch.long, device=device)
            n = int(lengths[0])
            name = names[0]
            pts = batch_data[0, :n].detach().cpu().numpy()

            get_or_build_disc_vis(pts, name, n_samples=args.disc_vis_samples)

            det_idxs, _ = model(
                batch_data, padding_mask=pad_mask, lengths=lens_t,
                deterministic=True, no_eos=False, eos_cov_threshold=0.0,
            )
            seed_sol = [int(idx) for idx in det_idxs[0] if int(idx) < n]

            try:
                seed_cov = float(evaluate_polygon_visibility_numpy_wo_gt(
                    pts, np.array(seed_sol, dtype=np.int64), name,
                )) if seed_sol else 0.0
            except Exception:
                seed_cov = 0.0

            final_sol, final_r, traj = ls_best_improvement_trajectory(
                pts, seed_sol, name, n,
                lam=args.lam, tau=args.tau, tau_penalty=args.tau_penalty,
                disc_vis_samples=args.disc_vis_samples,
                max_iter=args.ls_max_iter,
                monotone_coverage=args.monotone_coverage,
                min_dr=args.min_dr,
            )

            try:
                final_cov = float(evaluate_polygon_visibility_numpy_wo_gt(
                    pts, np.array(final_sol, dtype=np.int64), name,
                )) if final_sol else 0.0
            except Exception:
                final_cov = 0.0

            n_edits = sum(1 for s in traj if s["kind"] != "stop")
            edits_dist.append(n_edits)
            if n_edits == 0:
                n_stop_only += 1
            n_total_steps += len(traj)

            # Per-polygon record. Polygon points are stored as float32 to
            # keep the pickle compact; vertex indices are int.
            out_records.append({
                "name": name,
                "n": n,
                "points": pts.astype(np.float32),
                "seed": seed_sol,
                "seed_cov": seed_cov,
                "final": final_sol,
                "final_cov": final_cov,
                "final_r": final_r,
                "n_edits": n_edits,
                "trajectory": traj,
            })

            if (i + 1) % 100 == 0:
                arr = np.array(edits_dist, dtype=np.int64)
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / max(1e-6, elapsed)
                eta = (len(ds) - (i + 1)) / max(1e-6, rate)
                print(f"  [{i+1}/{len(ds)}]  edits median={np.median(arr):.0f} "
                      f"mean={arr.mean():.1f}  stop-only={n_stop_only}  "
                      f"rate={rate:.1f}/s  eta={eta:.0f}s")

    arr = np.array(edits_dist, dtype=np.int64)
    summary = {
        "n_polygons": len(out_records),
        "n_total_action_steps": int(n_total_steps),
        "n_total_edits": int(arr.sum()),
        "edits_mean": float(arr.mean()) if len(arr) else 0.0,
        "edits_median": float(np.median(arr)) if len(arr) else 0.0,
        "edits_p90": float(np.percentile(arr, 90)) if len(arr) else 0.0,
        "edits_max": int(arr.max()) if len(arr) else 0,
        "n_stop_only": int(n_stop_only),
        "checkpoint": args.checkpoint,
        "split": args.split,
        "ls_params": {
            "tau": args.tau, "lam": args.lam,
            "tau_penalty": args.tau_penalty,
            "disc_vis_samples": args.disc_vis_samples,
            "max_iter": args.ls_max_iter,
            "monotone_coverage": args.monotone_coverage,
        },
        "wall_time_s": time.perf_counter() - t0,
    }

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO_ROOT, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"summary": summary, "records": out_records}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)

    print()
    print("=" * 60)
    print(f"  Trajectory dataset built  ({len(out_records)} polygons)")
    print("=" * 60)
    print(f"  total edits = {summary['n_total_edits']}   "
          f"total action steps (incl. STOP) = {summary['n_total_action_steps']}")
    print(f"  edits mean={summary['edits_mean']:.2f}  "
          f"median={summary['edits_median']:.0f}  "
          f"p90={summary['edits_p90']:.0f}  max={summary['edits_max']}")
    print(f"  stop-only polygons = {summary['n_stop_only']}")
    print(f"  wall  {summary['wall_time_s']:.1f}s")
    print(f"  saved -> {out_path}")

    # Persist disc_vis cache so the next process (training, eval) starts
    # warm.
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            save_disc_vis_cache(cache_path)
        except Exception as e:
            print(f"  [warn] could not save disc_vis cache: {e}")


if __name__ == "__main__":
    main()
