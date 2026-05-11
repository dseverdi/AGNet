"""Supervised trainer for the EditHead.

Inputs:
    --train-traj data/ls_trajectories_train.pkl   (built by tools/build_ls_trajectories.py)
    --val-traj   data/ls_trajectories_dev.pkl     (optional; for end-to-end editor eval)

Each training example is a (state, action) tuple along an LS trajectory.
We unroll all polygons into a flat list and shuffle each epoch.

Two evaluation modes during training:
    1. Step-level accuracy (per-action CE loss + accuracy by action type).
    2. End-to-end rollout: pointer -> editor (greedy, multi-step) -> measure
       coverage, |S|, and recovery vs full LS on the dev set.

The pointer is loaded once and frozen.

A note on batching: we bucket states by polygon size so each batch has
similar L. This keeps padding low. Inside a batch every example is from
the SAME polygon-size bucket, so per-vertex features can be stacked.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
except ImportError:
    pass

from edit_head import EditHead, compute_vertex_features, edit_loss
from utils import (
    get_or_build_disc_vis, evaluate_polygon_visibility_numpy_wo_gt,
    save_disc_vis_cache, load_disc_vis_cache,
)
from po_agp import (
    create_agp_model,
    prepare_datasets,
    local_search_improve_disc,
    _read_opt_solution,
    _load_json_config,
    _apply_config_to_args,
    _get_explicit_args,
)
from dataset import collate_fn
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
from build_ls_trajectories import ls_best_improvement_trajectory  # noqa: E402


# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None,
                   help="JSON config file. Values are used as defaults; "
                        "CLI flags override.")
    p.add_argument("--train-traj", type=str, default=None)
    p.add_argument("--val-traj", type=str, default=None,
                   help="Optional dev trajectories for step-level eval")
    # Pointer (frozen) for end-to-end rollout eval.
    p.add_argument("--pointer-checkpoint", type=str,
                   default="checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt")
    p.add_argument("--embedding-size", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--n-glimpses", type=int, default=1)
    p.add_argument("--tanh-exploration", type=float, default=10.0)
    # Editor config.
    p.add_argument("--editor-hidden", type=int, default=64)
    p.add_argument("--editor-attn-layers", type=int, default=1)
    p.add_argument("--editor-heads", type=int, default=4)
    # Training.
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--w-stop", type=float, default=1.0)
    p.add_argument("--w-action", type=float, default=1.0)
    p.add_argument("--stop-pos-weight", type=float, default=None,
                   help="BCE pos_weight for STOP head; if None, derived "
                        "from train data (ratio of edit steps to stop "
                        "steps).")
    p.add_argument("--seed", type=int, default=1234)
    # End-to-end rollout eval.
    p.add_argument("--rollout-eval-k", type=int, default=100,
                   help="Sample size for end-to-end eval each epoch")
    p.add_argument("--rollout-max-steps", type=int, default=25)
    p.add_argument("--rollout-stop-threshold", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=0.99)
    p.add_argument("--tau-penalty", type=float, default=3.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--disc-vis-samples", type=int, default=500)
    # I/O.
    p.add_argument("--out-dir", type=str,
                   default="checkpoints/editor")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--disc-vis-cache-path", type=str,
                   default="data/disc_vis_cache.pkl",
                   help="Persisted disc_vis cache. Loaded at start, "
                        "warmed for any missing polygons, saved after warm-up.")
    # DAgger refresh.
    p.add_argument("--dagger-every", type=int, default=0,
                   help="If > 0, every K epochs roll out the current "
                        "editor on training polygons, query LS for the "
                        "correct action at each visited state, and add "
                        "those examples to the training set. 0 disables.")
    p.add_argument("--dagger-start", type=int, default=4,
                   help="First epoch to run DAgger refresh.")
    p.add_argument("--dagger-polygons", type=int, default=1000,
                   help="Number of training polygons to roll out per "
                        "DAgger refresh (subset, sampled w/o replacement).")
    p.add_argument("--dagger-accumulate", action="store_true", default=True,
                   help="Keep DAgger examples from previous refreshes "
                        "(default). Use --no-dagger-accumulate to replace.")
    p.add_argument("--no-dagger-accumulate", dest="dagger_accumulate",
                   action="store_false")

    # Capture defaults for config-merge.
    defaults = {a.dest: a.default for a in p._actions if a.dest != "help"}
    explicit = _get_explicit_args(p, sys.argv[1:])
    args = p.parse_args()

    if args.config:
        cfg = _load_json_config(args.config)
        _apply_config_to_args(args, cfg, defaults,
                              list(defaults.keys()), explicit)

    if not args.train_traj:
        raise SystemExit(
            "error: --train-traj is required (set via CLI or config file)")
    return args


# ──────────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────────
class TrajectoryStateDataset:
    """Flat list of (polygon_idx, step_idx) tuples. The actual features
    are computed lazily from the polygon record + disc_vis cache.

    Supports two sources of examples:
        1. Trajectory steps from the original LS run (loaded from pickle).
        2. DAgger examples: (state, LS-action) tuples collected by
           rolling out the partially-trained editor on training polygons.
           Stored in `self.dagger`; examples indexed at ex_idx >= len(self.index).
    """

    def __init__(self, traj_pickle_path: str, disc_vis_samples: int):
        with open(traj_pickle_path, "rb") as f:
            d = pickle.load(f)
        self.records = d["records"]
        self.summary = d["summary"]
        self.disc_vis_samples = disc_vis_samples
        # Index: (record_idx, step_idx) for every step in every trajectory.
        self.index: list[tuple[int, int]] = []
        for ri, rec in enumerate(self.records):
            for si in range(len(rec["trajectory"])):
                self.index.append((ri, si))
        # DAgger augmentation: (record_idx, step_dict) pairs.
        self.dagger: list[tuple[int, dict]] = []
        print(f"  [TrajData] {len(self.records)} polygons, "
              f"{len(self.index)} (state, action) examples")

    def __len__(self):
        return len(self.index) + len(self.dagger)

    def buckets_by_size(self) -> dict[int, list[int]]:
        """Returns {polygon_size n: [example_idx, ...]} over BOTH sources."""
        b = defaultdict(list)
        for i, (ri, _) in enumerate(self.index):
            b[self.records[ri]["n"]].append(i)
        N = len(self.index)
        for j, (ri, _) in enumerate(self.dagger):
            b[self.records[ri]["n"]].append(N + j)
        return b

    def fetch(self, ex_idx: int):
        """Return (rec, step_dict) for one training example, drawing
        from trajectory or DAgger pool by index range."""
        N = len(self.index)
        if ex_idx < N:
            ri, si = self.index[ex_idx]
            rec = self.records[ri]
            return rec, rec["trajectory"][si]
        ri, step = self.dagger[ex_idx - N]
        return self.records[ri], step

    def add_dagger_examples(self, examples: list[tuple[int, dict]]):
        self.dagger.extend(examples)

    def clear_dagger(self):
        self.dagger.clear()


# ──────────────────────────────────────────────────────────────────────
#  Batched feature stacking (same polygon size n)
# ──────────────────────────────────────────────────────────────────────
def make_batch(ds: TrajectoryStateDataset, indices: list[int],
               device: torch.device):
    """All examples in `indices` must share polygon size n."""
    feats_list = []
    in_S_list = []
    rm_idx_list = []
    ad_idx_list = []
    kinds = []

    for ex in indices:
        rec, step = ds.fetch(ex)
        pts = rec["points"]                       # (n, 2)
        n = rec["n"]
        # Pull the cached disc_vis (already populated during data gen).
        disc = get_or_build_disc_vis(pts, rec["name"],
                                     n_samples=ds.disc_vis_samples)
        if not disc.get("valid"):
            continue
        vis = disc["vis_matrix"]
        vf = compute_vertex_features(pts, step["state"], vis, device=device)
        feats_list.append(vf.feats)
        in_S_list.append(vf.in_S)
        kinds.append(step["kind"])
        rm_idx_list.append(step["remove"] if step["remove"] is not None else 0)
        ad_idx_list.append(step["add"] if step["add"] is not None else 0)

    if not feats_list:
        return None

    feats = torch.stack(feats_list, dim=0)         # (B, n, D)
    in_S = torch.stack(in_S_list, dim=0)           # (B, n) bool
    target = {
        "kind": kinds,
        "remove_idx": torch.tensor(rm_idx_list, dtype=torch.long, device=device),
        "add_idx":    torch.tensor(ad_idx_list, dtype=torch.long, device=device),
    }
    return feats, in_S, target


def epoch_batches(ds: TrajectoryStateDataset, batch_size: int,
                  shuffle: bool = True):
    """Yield batches: each batch is a list of example indices, all same n."""
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
#  Per-action accuracy
# ──────────────────────────────────────────────────────────────────────
def step_metrics(out: dict, target: dict) -> dict:
    """Step-level accuracy.
       stop_acc: P(stop_pred == stop_target)
       remove_top1: among kind=remove, P(argmax remove_logits == target)
       swap_out_top1, swap_in_top1: same for swap.
    """
    stop_logit = out["stop_logit"]
    p_stop = torch.sigmoid(stop_logit) >= 0.5
    is_stop = torch.tensor([k == "stop" for k in target["kind"]],
                           device=stop_logit.device)
    stop_acc = (p_stop == is_stop).float().mean().item()

    rm_pred = out["remove_logits"].argmax(dim=-1)
    sw_out_pred = out["swap_out_logits"].argmax(dim=-1)
    sw_in_pred = out["swap_in_logits"].argmax(dim=-1)

    rm_correct = []
    sw_out_correct = []
    sw_in_correct = []
    for b, k in enumerate(target["kind"]):
        if k == "remove":
            rm_correct.append(rm_pred[b].item() == target["remove_idx"][b].item())
        elif k == "swap":
            sw_out_correct.append(sw_out_pred[b].item() == target["remove_idx"][b].item())
            sw_in_correct.append(sw_in_pred[b].item() == target["add_idx"][b].item())

    return {
        "stop_acc": stop_acc,
        "remove_top1":   float(np.mean(rm_correct))   if rm_correct   else None,
        "swap_out_top1": float(np.mean(sw_out_correct)) if sw_out_correct else None,
        "swap_in_top1":  float(np.mean(sw_in_correct))  if sw_in_correct  else None,
        "n_remove": len(rm_correct),
        "n_swap":   len(sw_out_correct),
    }


# ──────────────────────────────────────────────────────────────────────
#  End-to-end rollout eval on a held-out polygon set
# ──────────────────────────────────────────────────────────────────────
def dagger_refresh(editor: "EditHead",
                   train_data: "TrajectoryStateDataset",
                   args, device) -> list[tuple[int, dict]]:
    """One DAgger pass.

    For ``args.dagger_polygons`` random training polygons:
        1. Start at the pointer's seed (already stored in the record).
        2. At each visited state ask LS for the locally-best action.
        3. Record (state, LS-action) as a new supervised example.
        4. Advance the rollout using the *editor*'s prediction (not LS's),
           so subsequent visited states reflect what the editor would do
           at inference. This is the core of DAgger: train on the
           distribution induced by the current policy, labelled by the
           expert.

    Halts the rollout when either LS or the editor says STOP, or when
    args.rollout_max_steps is hit.
    """
    editor.eval()
    n_polys = min(args.dagger_polygons, len(train_data.records))
    indices = random.sample(range(len(train_data.records)), n_polys)

    new_examples: list[tuple[int, dict]] = []
    n_stop = n_edit = 0
    t0 = time.perf_counter()

    with torch.no_grad():
        for k, ri in enumerate(indices, 1):
            rec = train_data.records[ri]
            pts = rec["points"]
            name = rec["name"]
            n = rec["n"]

            disc = get_or_build_disc_vis(pts, name,
                                         n_samples=args.disc_vis_samples)
            if not disc.get("valid"):
                continue
            vis = disc["vis_matrix"]

            S = list(rec["seed"])
            for _ in range(args.rollout_max_steps):
                # LS teacher: best action at the current state.
                _, _, mini = ls_best_improvement_trajectory(
                    pts, S, name, n,
                    lam=args.lam, tau=args.tau, tau_penalty=args.tau_penalty,
                    disc_vis_samples=args.disc_vis_samples,
                    max_iter=1, monotone_coverage=True,
                )
                teacher = mini[0]
                new_examples.append((ri, {
                    "state":   list(S),
                    "kind":    teacher["kind"],
                    "remove":  teacher["remove"],
                    "add":     teacher["add"],
                    "delta_r": float(teacher.get("delta_r", 0.0)),
                }))
                if teacher["kind"] == "stop":
                    n_stop += 1
                    break
                n_edit += 1

                # Editor decides where to go next (drives distribution).
                vf = compute_vertex_features(pts, S, vis, device=device)
                pred = editor.predict(
                    vf.feats, vf.in_S,
                    stop_threshold=args.rollout_stop_threshold,
                )
                ed_kind = pred["kind"][0]
                if ed_kind == "stop":
                    break
                rm = pred["remove_idx"][0].item()
                ad = pred["add_idx"][0].item()
                if ed_kind == "remove":
                    if rm in S:
                        S.remove(rm)
                elif ed_kind == "swap":
                    if rm in S:
                        S.remove(rm)
                    if 0 <= ad < n and ad not in S:
                        S.append(ad)
                else:
                    break
            if k % 200 == 0:
                rate = k / max(1e-6, time.perf_counter() - t0)
                print(f"  [dagger] {k}/{n_polys}  rate={rate:.1f}/s  "
                      f"+{len(new_examples)} examples")

    dt = time.perf_counter() - t0
    print(f"  [dagger] +{len(new_examples)} examples "
          f"({n_edit} edit, {n_stop} stop)  "
          f"from {n_polys} polygons  {dt:.0f}s")
    return new_examples


def prewarm_disc_vis_for_records(records, disc_vis_samples, label):
    """Build disc_vis for every polygon in `records`, with progress.
    Idempotent — already-cached polygons are no-ops in the helper."""
    n = len(records)
    t0 = time.perf_counter()
    last_log = t0
    for i, rec in enumerate(records, 1):
        get_or_build_disc_vis(rec["points"], rec["name"],
                              n_samples=disc_vis_samples)
        now = time.perf_counter()
        if now - last_log > 5.0 or i == n:
            elapsed = now - t0
            rate = i / max(1e-6, elapsed)
            eta = (n - i) / max(1e-6, rate)
            print(f"  [disc_vis] {label}: {i}/{n}  "
                  f"({rate:.1f}/s, eta {eta:.0f}s)")
            last_log = now


_ROLLOUT_REF_CACHE: dict | None = None  # populated once per process


def _build_rollout_reference(pointer, val_ds_torch, args, device,
                             sol_dir: str | None = None):
    """Compute (seed, LS-final, costs) for each eval polygon. Pointer is
    frozen so this is identical across training epochs — cache once.
    Also reads the exact AGP OPT size from sol_dir/<name>.solution.
    Returns a list of dicts."""
    pointer.eval()
    loader = DataLoader(val_ds_torch, batch_size=1, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    refs = []
    with torch.no_grad():
        for batch_data, pad_mask, lengths, names in loader:
            batch_data = batch_data.to(device)
            pad_mask = pad_mask.to(device)
            lens_t = torch.tensor(lengths, dtype=torch.long, device=device)
            n = int(lengths[0])
            name = names[0]
            pts = batch_data[0, :n].detach().cpu().numpy()

            disc = get_or_build_disc_vis(pts, name, n_samples=args.disc_vis_samples)
            if not disc.get("valid"):
                continue

            det_idxs, _ = pointer(
                batch_data, padding_mask=pad_mask, lengths=lens_t,
                deterministic=True, no_eos=False, eos_cov_threshold=0.0,
            )
            seed = [int(i) for i in det_idxs[0] if int(i) < n]
            if not seed:
                continue
            try:
                seed_cov = float(evaluate_polygon_visibility_numpy_wo_gt(
                    pts, np.array(seed, dtype=np.int64), name))
            except Exception:
                seed_cov = 0.0

            ls_sol, _, _ = local_search_improve_disc(
                pts, list(seed), name, n,
                max_iter=200, enable_swap=True,
                enable_remove=True, enable_add=True,
                lam=args.lam, tau=args.tau, tau_penalty=args.tau_penalty,
                cap_at_tau=False, n_samples=args.disc_vis_samples,
                reward_fn_fallback=None, monotone_coverage=True,
            )
            try:
                ls_cov = float(evaluate_polygon_visibility_numpy_wo_gt(
                    pts, np.array(ls_sol, dtype=np.int64), name)) if ls_sol else 0.0
            except Exception:
                ls_cov = 0.0

            opt_sol = _read_opt_solution(sol_dir, name) if sol_dir else None
            opt_size = len(opt_sol) if opt_sol else None

            refs.append({
                "name": name, "n": n, "pts": pts,
                "seed": seed, "seed_cov": seed_cov,
                "ls_sol": ls_sol, "ls_cov": ls_cov,
                "opt_size": opt_size,
            })
    return refs


def end_to_end_rollout(
    editor: EditHead,
    pointer: torch.nn.Module,
    val_ds_torch,
    args, device,
    n_eval: int = 100,
    sol_dir: str | None = None,
):
    """For n_eval validation polygons:
       1. Greedy-decode pointer -> seed.
       2. Apply editor up to args.rollout_max_steps.
       3. Compare with cached full LS reward.
       Reports recovery: (r_editor - r_seed) / (r_full_ls - r_seed).
    """
    global _ROLLOUT_REF_CACHE
    if _ROLLOUT_REF_CACHE is None:
        print("  [rollout] building reference (pointer + LS, one-time)...")
        _ROLLOUT_REF_CACHE = _build_rollout_reference(
            pointer, val_ds_torch, args, device, sol_dir=sol_dir,
        )
        print(f"  [rollout] reference cached: {len(_ROLLOUT_REF_CACHE)} polygons")
    refs = _ROLLOUT_REF_CACHE[: n_eval]

    editor.eval()

    def _r(cov, k, n):
        return cov - args.lam * k / max(1, n) - args.tau_penalty * max(0.0, args.tau - cov)

    recoveries = []
    seed_covs, ed_covs, full_covs = [], [], []
    seed_sizes, ed_sizes, full_sizes = [], [], []
    ns, opt_sizes = [], []
    stop_action_share = []
    n_done = 0

    with torch.no_grad():
        for ref in refs:
            n = ref["n"]
            name = ref["name"]
            pts = ref["pts"]
            seed = ref["seed"]
            seed_cov = ref["seed_cov"]
            ls_sol = ref["ls_sol"]
            full_cov = ref["ls_cov"]
            disc = get_or_build_disc_vis(pts, name, n_samples=args.disc_vis_samples)
            if not disc.get("valid"):
                continue
            vis = disc["vis_matrix"]

            r_seed = _r(seed_cov, len(seed), n)
            r_full = _r(full_cov, len(ls_sol), n)

            S = list(seed)
            stops_seen = 0
            for _ in range(args.rollout_max_steps):
                vf = compute_vertex_features(pts, S, vis, device=device)
                pred = editor.predict(vf.feats, vf.in_S,
                                      stop_threshold=args.rollout_stop_threshold)
                kind = pred["kind"][0]
                if kind == "stop":
                    stops_seen = 1
                    break
                rm = pred["remove_idx"][0].item()
                ad = pred["add_idx"][0].item()
                if kind == "remove":
                    if rm in S:
                        S.remove(rm)
                elif kind == "swap":
                    if rm in S:
                        S.remove(rm)
                    if ad >= 0 and ad < n and ad not in S:
                        S.append(ad)
                else:
                    break
            stop_action_share.append(stops_seen)
            try:
                ed_cov = float(evaluate_polygon_visibility_numpy_wo_gt(
                    pts, np.array(S, dtype=np.int64), name)) if S else 0.0
            except Exception:
                ed_cov = 0.0
            r_ed = _r(ed_cov, len(S), n)

            d_full = r_full - r_seed
            d_ed   = r_ed   - r_seed
            if d_full > 1e-6:
                recoveries.append(d_ed / d_full)
            seed_covs.append(seed_cov)
            ed_covs.append(ed_cov)
            full_covs.append(full_cov)
            seed_sizes.append(len(seed))
            ed_sizes.append(len(S))
            full_sizes.append(len(ls_sol))
            ns.append(n)
            if ref.get("opt_size"):
                opt_sizes.append(ref["opt_size"])
            n_done += 1

    rec_arr = np.array(recoveries) if recoveries else np.array([0.0])
    # Per-polygon ratios then averaged (more honest than mean(|S|)/mean(n)).
    ns_arr = np.array(ns, dtype=np.float64) if ns else np.array([1.0])
    seed_chv = float(np.mean(np.array(seed_sizes) / ns_arr)) if seed_sizes else 0.0
    ed_chv   = float(np.mean(np.array(ed_sizes)   / ns_arr)) if ed_sizes   else 0.0
    ls_chv   = float(np.mean(np.array(full_sizes) / ns_arr)) if full_sizes else 0.0

    seed_opt = ed_opt = ls_opt = None
    if opt_sizes and len(opt_sizes) == len(seed_sizes):
        opt_arr = np.array(opt_sizes, dtype=np.float64)
        seed_opt = float(np.mean(np.array(seed_sizes) / opt_arr))
        ed_opt   = float(np.mean(np.array(ed_sizes)   / opt_arr))
        ls_opt   = float(np.mean(np.array(full_sizes) / opt_arr))

    return {
        "n_eval": n_done,
        "recovery_mean":   float(rec_arr.mean()),
        "recovery_median": float(np.median(rec_arr)),
        "recovery_p10":    float(np.percentile(rec_arr, 10)) if len(rec_arr) > 1 else float(rec_arr[0]),
        "frac_recovery_ge_0.8": float((rec_arr >= 0.8).mean()),
        "frac_recovery_ge_1.0": float((rec_arr >= 1.0).mean()),
        "seed_cov_mean":   float(np.mean(seed_covs)) if seed_covs else 0.0,
        "ed_cov_mean":     float(np.mean(ed_covs))   if ed_covs   else 0.0,
        "full_cov_mean":   float(np.mean(full_covs)) if full_covs else 0.0,
        "seed_chv":        seed_chv,    # |S|/n averaged per polygon
        "ed_chv":          ed_chv,
        "ls_chv":          ls_chv,
        "seed_opt":        seed_opt,    # |S|/OPT averaged per polygon
        "ed_opt":          ed_opt,
        "ls_opt":          ls_opt,
        "seed_size_mean":  float(np.mean(seed_sizes)) if seed_sizes else 0.0,
        "ed_size_mean":    float(np.mean(ed_sizes))   if ed_sizes   else 0.0,
        "full_size_mean":  float(np.mean(full_sizes)) if full_sizes else 0.0,
        "stop_share":      float(np.mean(stop_action_share)) if stop_action_share else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[editor] device={device}")
    print(f"[editor] train_traj={args.train_traj}")

    train_data = TrajectoryStateDataset(args.train_traj, args.disc_vis_samples)
    val_data = (
        TrajectoryStateDataset(args.val_traj, args.disc_vis_samples)
        if args.val_traj else None
    )

    # Disc_vis cache: load any persisted entries, then warm up any
    # missing polygons up front so the first batch isn't doing silent
    # work that looks like a hang.
    cache_path = args.disc_vis_cache_path
    if cache_path and not os.path.isabs(cache_path):
        cache_path = os.path.join(REPO_ROOT, cache_path)
    if cache_path and os.path.exists(cache_path):
        load_disc_vis_cache(cache_path)
    print("[editor] pre-warming disc_vis cache (one-time after fresh process)...")
    prewarm_disc_vis_for_records(
        train_data.records, args.disc_vis_samples, label="train",
    )
    if val_data is not None:
        prewarm_disc_vis_for_records(
            val_data.records, args.disc_vis_samples, label="dev",
        )
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            save_disc_vis_cache(cache_path)
        except Exception as e:
            print(f"[editor] warning: could not save disc_vis cache: {e}")

    # Derive STOP pos_weight from class imbalance if not set.
    if args.stop_pos_weight is None:
        n_stop = sum(
            1 for ri, si in train_data.index
            if train_data.records[ri]["trajectory"][si]["kind"] == "stop"
        )
        n_edit = len(train_data.index) - n_stop
        stop_pw = max(1.0, n_edit / max(1, n_stop))
        print(f"[editor] STOP pos_weight (auto) = {stop_pw:.2f}  "
              f"(n_edit={n_edit} n_stop={n_stop})")
    else:
        stop_pw = args.stop_pos_weight
        print(f"[editor] STOP pos_weight (manual) = {stop_pw:.2f}")

    # Editor + pointer (frozen).
    editor = EditHead(hidden=args.editor_hidden,
                      n_attn_layers=args.editor_attn_layers,
                      heads=args.editor_heads).to(device)
    print(f"[editor] params = {editor.num_params():,}")
    optimizer = torch.optim.AdamW(editor.parameters(), lr=args.lr,
                                  weight_decay=1e-4)

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

    # Tiny dev dataset for end-to-end rollout (use the dev .pol files).
    DATASET_PATH = os.getenv("DATASET_PATH")
    if DATASET_PATH is not None:
        dev_sol_dir = os.path.join(DATASET_PATH, "dev")
        _, val_torch_ds = prepare_datasets(
            os.path.join(DATASET_PATH, "train"),
            dev_sol_dir,
            normalize=True,
        )
        # Subsample.
        val_torch_ds.samples = val_torch_ds.samples[: args.rollout_eval_k]
    else:
        dev_sol_dir = None
        val_torch_ds = None

    best_recovery = -float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        # ── Optional DAgger refresh before this epoch ────────────────
        if (args.dagger_every > 0
                and epoch >= args.dagger_start
                and (epoch - args.dagger_start) % args.dagger_every == 0):
            print(f"  [dagger] refresh at epoch {epoch} "
                  f"(rolling out current editor on {args.dagger_polygons} polygons)")
            if not args.dagger_accumulate:
                train_data.clear_dagger()
            new_ex = dagger_refresh(editor, train_data, args, device)
            train_data.add_dagger_examples(new_ex)
            print(f"  [dagger] training set now {len(train_data)} examples "
                  f"({len(train_data.index)} traj + {len(train_data.dagger)} dagger)")

        editor.train()
        t0 = time.perf_counter()
        n_examples = 0
        L_sum = 0.0
        L_stop_sum = 0.0
        L_action_sum = 0.0
        per_kind = defaultdict(int)

        for batch_indices in epoch_batches(train_data, args.batch_size,
                                           shuffle=True):
            batch = make_batch(train_data, batch_indices, device)
            if batch is None:
                continue
            feats, in_S, target = batch

            out = editor(feats, in_S)
            loss_d = edit_loss(out, target,
                               w_stop=args.w_stop, w_action=args.w_action,
                               stop_pos_weight=stop_pw)
            optimizer.zero_grad(set_to_none=True)
            loss_d["loss"].backward()
            torch.nn.utils.clip_grad_norm_(editor.parameters(), 1.0)
            optimizer.step()

            B = feats.shape[0]
            L_sum += loss_d["loss"].item() * B
            L_stop_sum += loss_d["L_stop"].item() * B
            L_action_sum += loss_d["L_action"].item() * B
            n_examples += B
            for k in target["kind"]:
                per_kind[k] += 1

        train_loss = L_sum / max(1, n_examples)
        train_stop = L_stop_sum / max(1, n_examples)
        train_action = L_action_sum / max(1, n_examples)
        epoch_time = time.perf_counter() - t0

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_loss_stop": train_stop,
            "train_loss_action": train_action,
            "n_examples": n_examples,
            "kind_counts": dict(per_kind),
            "epoch_time_s": epoch_time,
        }
        msg = (f"[ep {epoch:3d}] L={train_loss:.4f} "
               f"(stop={train_stop:.4f} act={train_action:.4f}) "
               f"n={n_examples}  {epoch_time:.1f}s")
        print(msg)

        # Step-level eval on dev trajectories.
        if val_data is not None:
            editor.eval()
            with torch.no_grad():
                acc_accum = defaultdict(list)
                count_remove = 0
                count_swap = 0
                stop_correct = 0
                stop_total = 0
                # Walk all dev (state, action) examples (cap at 5000 to keep this fast).
                cap = min(5000, len(val_data))
                idx_pool = list(range(len(val_data)))
                random.shuffle(idx_pool)
                idx_pool = idx_pool[:cap]
                ds_buckets = defaultdict(list)
                for i in idx_pool:
                    ri, _ = val_data.index[i]
                    ds_buckets[val_data.records[ri]["n"]].append(i)
                for n, ids in ds_buckets.items():
                    for i in range(0, len(ids), args.batch_size):
                        chunk = ids[i: i + args.batch_size]
                        b = make_batch(val_data, chunk, device)
                        if b is None:
                            continue
                        feats, in_S, target = b
                        out = editor(feats, in_S)
                        m = step_metrics(out, target)
                        # Aggregate.
                        stop_total += len(chunk)
                        stop_correct += int(m["stop_acc"] * len(chunk))
                        if m["remove_top1"] is not None:
                            acc_accum["remove"].append((m["remove_top1"], m["n_remove"]))
                            count_remove += m["n_remove"]
                        if m["swap_out_top1"] is not None:
                            acc_accum["swap_out"].append((m["swap_out_top1"], m["n_swap"]))
                            acc_accum["swap_in"].append((m["swap_in_top1"], m["n_swap"]))
                            count_swap += m["n_swap"]
                def _wmean(rows):
                    num = sum(a * w for a, w in rows)
                    den = sum(w for _, w in rows)
                    return num / max(1, den) if den else 0.0
                step_eval = {
                    "stop_acc":      stop_correct / max(1, stop_total),
                    "remove_top1":   _wmean(acc_accum["remove"]),
                    "swap_out_top1": _wmean(acc_accum["swap_out"]),
                    "swap_in_top1":  _wmean(acc_accum["swap_in"]),
                    "n_remove":      count_remove,
                    "n_swap":        count_swap,
                }
                log["step_eval"] = step_eval
                print(f"           step_eval: stop_acc={step_eval['stop_acc']:.3f}  "
                      f"remove_top1={step_eval['remove_top1']:.3f}  "
                      f"swap_out={step_eval['swap_out_top1']:.3f}  "
                      f"swap_in={step_eval['swap_in_top1']:.3f}")

        # End-to-end rollout eval.
        if val_torch_ds is not None and args.rollout_eval_k > 0:
            roll = end_to_end_rollout(editor, pointer, val_torch_ds,
                                      args, device, n_eval=args.rollout_eval_k,
                                      sol_dir=dev_sol_dir)
            log["rollout"] = roll
            # Primary metrics (cov, |S|/n, |S|/OPT) for seed / editor / LS.
            opt_line = ""
            if roll["seed_opt"] is not None:
                opt_line = (f"  |S|/OPT s/e/L={roll['seed_opt']:.3f}/"
                            f"{roll['ed_opt']:.3f}/{roll['ls_opt']:.3f}")
            print(f"           rollout (n={roll['n_eval']}): "
                  f"cov s/e/L={roll['seed_cov_mean']:.3f}/"
                  f"{roll['ed_cov_mean']:.3f}/{roll['full_cov_mean']:.3f}  "
                  f"|S|/n s/e/L={roll['seed_chv']:.3f}/"
                  f"{roll['ed_chv']:.3f}/{roll['ls_chv']:.3f}"
                  f"{opt_line}")
            # Diagnostics (with role labels).
            print(f"             recovery med={roll['recovery_median']:.3f} "
                  f"mean={roll['recovery_mean']:.3f}  "
                  f"P[>=0.8]={roll['frac_recovery_ge_0.8']:.3f}  "
                  f"P[>=1.0]={roll['frac_recovery_ge_1.0']:.3f}  "
                  f"[reward gain ed vs LS, fraction matching/beating LS]")
            print(f"             editor: stop_share={roll['stop_share']:.2f}  "
                  f"[fraction of rollouts where STOP head fired vs hit step cap]")
            metric = roll["recovery_median"]
            if metric > best_recovery:
                best_recovery = metric
                save_path = os.path.join(args.out_dir, "editor_best.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": editor.state_dict(),
                    "args": vars(args),
                    "rollout_metric": metric,
                }, save_path)
                print(f"           [save] new best recovery_median={metric:.3f} -> {save_path}")

        history.append(log)
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2, default=str)

    save_path = os.path.join(args.out_dir, "editor_final.pt")
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": editor.state_dict(),
        "args": vars(args),
    }, save_path)
    print(f"\n[editor] saved final -> {save_path}")


if __name__ == "__main__":
    main()
