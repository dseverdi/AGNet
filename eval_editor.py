"""Stand-alone evaluation for the EditHead.

Loads a frozen pointer + trained editor and reports, on the val set:
    • pointer-only (seed)
    • pointer + editor   (the learned method)
    • pointer + full LS  (the upper bound we are imitating)

For each: coverage, |S|/n, |S|/OPT, reward, wall time.
Plus the recovery fraction (editor_dr / ls_dr) per polygon.

This script does not touch po_agp.py or any training code. Editor on/off
is a CLI flag.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

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

from dataset import collate_fn
from edit_head import EditHead, compute_vertex_features, compute_vertex_features_geo_free
from po_agp import (
    create_agp_model,
    prepare_datasets,
    local_search_improve_disc,
    _read_opt_solution,
    _load_json_config,
    _apply_config_to_args,
    _get_explicit_args,
)
from utils import (
    evaluate_polygon_visibility_numpy_wo_gt,
    get_or_build_disc_vis,
    load_disc_vis_cache,
    save_disc_vis_cache,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None,
                   help="JSON config file. CLI flags override config values.")
    p.add_argument("--pointer-checkpoint", type=str,
                   default="checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt")
    p.add_argument("--editor-checkpoint", type=str, default=None)
    p.add_argument("--val-dir", type=str, default=None)
    p.add_argument("--n-samples", type=int, default=-1)
    # Pointer config (must match the pretrained checkpoint).
    p.add_argument("--embedding-size", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--n-glimpses", type=int, default=1)
    p.add_argument("--tanh-exploration", type=float, default=10.0)
    # Editor decoding.
    p.add_argument("--rollout-max-steps", type=int, default=25)
    p.add_argument("--stop-threshold", type=float, default=0.5)
    # LS reference (matches training reward).
    p.add_argument("--tau", type=float, default=0.99)
    p.add_argument("--tau-penalty", type=float, default=3.0)
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--disc-vis-samples", type=int, default=500)
    # Toggle full-LS reference comparison (slow on full val).
    p.add_argument("--include-ls-reference", action="store_true",
                   default=True)
    p.add_argument("--no-ls-reference", dest="include_ls_reference",
                   action="store_false")
    p.add_argument("--out", "--output-json", dest="out",
                   type=str, default="results/eval_editor.json")
    p.add_argument("--disc-vis-cache-path", type=str,
                   default="data/disc_vis_cache.pkl")
    p.add_argument("--cov-gate", type=float, default=None,
                   help="Absolute coverage floor. Refuse any edit that would "
                        "drop disc_vis coverage below this value.")
    p.add_argument("--cov-gate-relative", type=float, default=None,
                   metavar="EPS",
                   help="Adaptive (per-polygon) coverage floor. Refuse any "
                        "edit that would drop disc_vis coverage below "
                        "(seed_cov - EPS). E.g. --cov-gate-relative 0.005 "
                        "means each polygon's floor is its own seed coverage "
                        "minus 0.5 pp. Takes priority over --cov-gate.")
    p.add_argument("--geo-free-features", action="store_true", default=False,
                   help="Use only (x, y, in_S) as editor input — no disc_vis oracle "
                        "at inference time. Auto-detected from checkpoint saved args. "
                        "Disables coverage gate.")
    p.add_argument("--topology-features", action="store_true", default=False,
                   help="Geo-free + polygon topology (D_in=8). Auto-detected from "
                        "checkpoint.")
    defaults = {a.dest: a.default for a in p._actions if a.dest != "help"}
    explicit = _get_explicit_args(p, sys.argv[1:])
    args = p.parse_args()
    if args.config:
        cfg = _load_json_config(args.config)
        _apply_config_to_args(args, cfg, defaults,
                              list(defaults.keys()), explicit)
    if not args.editor_checkpoint:
        raise SystemExit(
            "error: --editor-checkpoint is required (set via CLI or config)")
    return args


def reward_scalar(cov: float, k: int, n: int, *, lam: float, tau: float,
                  tau_penalty: float) -> float:
    return cov - lam * k / max(1, n) - tau_penalty * max(0.0, tau - cov)


def coverage_exact(pts: np.ndarray, guards: list[int], name: str) -> float:
    if not guards:
        return 0.0
    try:
        return float(evaluate_polygon_visibility_numpy_wo_gt(
            pts, np.array(guards, dtype=np.int64), name,
        ))
    except Exception:
        return 0.0


def load_pointer(args, device):
    model = create_agp_model(
        args.embedding_size, args.hidden_size, args.n_glimpses,
        args.tanh_exploration, use_tanh=True, temperature=1.0,
    )
    p = args.pointer_checkpoint
    if not os.path.isabs(p):
        p = os.path.join(REPO_ROOT, p)
    ckpt = torch.load(p, map_location=device, weights_only=False)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, _ = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [pointer] missing keys: {len(missing)}")
    model.eval()
    return model


def load_editor(args, device):
    p = args.editor_checkpoint
    if not os.path.isabs(p):
        p = os.path.join(REPO_ROOT, p)
    ckpt = torch.load(p, map_location=device, weights_only=False)
    saved_args = ckpt.get("args", {})
    H = saved_args.get("editor_hidden", 64)
    L = saved_args.get("editor_attn_layers", 1)
    HD = saved_args.get("editor_heads", 4)
    # Auto-detect geo-free / topology / aux mode from checkpoint unless
    # already set by CLI.
    if not args.geo_free_features and saved_args.get("geo_free_features", False):
        print("  [editor] auto-detected geo_free_features=True from checkpoint")
        args.geo_free_features = True
    saved_topology = saved_args.get("topology_features", False)
    if saved_topology and not getattr(args, "topology_features", False):
        print("  [editor] auto-detected topology_features=True from checkpoint")
        args.topology_features = True
    saved_aux = saved_args.get("aux_visibility", False)
    # Determine d_in from saved flags.
    if args.geo_free_features:
        d_in = 8 if getattr(args, "topology_features", False) else 3
    else:
        d_in = None
    editor = EditHead(hidden=H, n_attn_layers=L, heads=HD, d_in=d_in,
                      aux_visibility=saved_aux).to(device)
    editor.load_state_dict(ckpt["model_state_dict"])
    editor.eval()
    return editor


def _disc_cov(vis: np.ndarray, S: list[int]) -> float:
    """Cheap coverage from cached disc_vis matrix."""
    if not S:
        return 0.0
    M = vis.shape[1]
    covered = np.zeros(M, dtype=np.bool_)
    for v in S:
        if 0 <= v < vis.shape[0]:
            np.bitwise_or(covered, vis[v], out=covered)
    return float(covered.sum()) / max(1, M)


def editor_rollout(editor, pts, vis, seed, n, *, max_steps, stop_threshold,
                   device, cov_gate: float | None = None,
                   geo_free: bool = False, topology: bool = False):
    """Apply editor greedily until STOP or step cap.

    geo_free=True: uses only (x,y,in_S) features; cov_gate is ignored.
    topology=True: only meaningful with geo_free; adds polygon-topology
    features (D_in=8 instead of 3)."""
    S = list(seed)
    n_steps = 0
    stopped = False
    n_rejected = 0
    for _ in range(max_steps):
        if geo_free:
            vf = compute_vertex_features_geo_free(
                pts, S, device=device, topology=topology,
            )
        else:
            vf = compute_vertex_features(pts, S, vis, device=device)
        pred = editor.predict(vf.feats, vf.in_S, stop_threshold=stop_threshold)
        kind = pred["kind"][0]
        if kind == "stop":
            stopped = True
            break
        rm = pred["remove_idx"][0].item()
        ad = pred["add_idx"][0].item()
        # Tentative apply.
        if kind == "remove":
            if rm not in S:
                break
            cand = [v for v in S if v != rm]
        elif kind == "swap":
            if rm not in S:
                break
            cand = [v for v in S if v != rm]
            if 0 <= ad < n and ad not in cand:
                cand.append(ad)
        else:
            break
        if cov_gate is not None and not geo_free:
            cand_cov = _disc_cov(vis, cand)
            if cand_cov < cov_gate:
                n_rejected += 1
                stopped = True  # treat as halt — don't degrade further
                break
        S = cand
        n_steps += 1
    return S, n_steps, stopped, n_rejected


def main() -> None:
    args = parse_args()
    DATASET_PATH = os.getenv("DATASET_PATH")
    if DATASET_PATH is None:
        raise EnvironmentError("DATASET_PATH not set")
    val_dir = args.val_dir or os.path.join(DATASET_PATH, "dev")
    train_dir = os.path.join(DATASET_PATH, "train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}")
    print(f"[eval] pointer={args.pointer_checkpoint}")
    print(f"[eval] editor={args.editor_checkpoint}")

    _, val_ds = prepare_datasets(train_dir, val_dir, normalize=True)
    if args.n_samples > 0:
        val_ds.samples = val_ds.samples[: args.n_samples]
    print(f"[eval] {len(val_ds)} polygons, include_ls={args.include_ls_reference}")

    pointer = load_pointer(args, device)
    # load_editor may set args.geo_free_features=True via auto-detect.
    editor = load_editor(args, device)
    geo_free = args.geo_free_features

    if geo_free:
        print("[eval] geo-free inference: features=(x,y,in_S) — no disc_vis oracle")
        if args.cov_gate is not None or args.cov_gate_relative is not None:
            print("[eval] WARNING: coverage gate ignored in geo-free mode")
    else:
        print(f"[eval] stop_threshold={args.stop_threshold}  "
              f"cov_gate={'relative(' + str(args.cov_gate_relative) + ')' if args.cov_gate_relative is not None else (str(args.cov_gate) if args.cov_gate is not None else 'none')}")

    # Load disc_vis cache only when needed (LS reference or oracle-mode features/gate).
    if not geo_free or args.include_ls_reference:
        cache_path = args.disc_vis_cache_path
        if cache_path and not os.path.isabs(cache_path):
            cache_path = os.path.join(REPO_ROOT, cache_path)
        if cache_path and os.path.exists(cache_path):
            load_disc_vis_cache(cache_path)

    print(f"[eval] editor params = {editor.num_params():,}  "
          f"features={'geo_free+topology(8)' if (geo_free and getattr(args, 'topology_features', False)) else ('geo_free(3)' if geo_free else 'oracle(6)')}")

    loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    records: list[dict] = []
    t_total = time.perf_counter()

    with torch.no_grad():
        for i, (batch_data, pad_mask, lengths, names) in enumerate(loader):
            batch_data = batch_data.to(device)
            pad_mask = pad_mask.to(device)
            lens_t = torch.tensor(lengths, dtype=torch.long, device=device)
            n = int(lengths[0])
            name = names[0]
            pts = batch_data[0, :n].detach().cpu().numpy()
            vis = None
            if not geo_free or args.include_ls_reference:
                disc = get_or_build_disc_vis(pts, name, n_samples=args.disc_vis_samples)
                if not disc.get("valid"):
                    continue
                vis = disc["vis_matrix"]

            # Pointer (seed).
            t0 = time.perf_counter()
            det_idxs, _ = pointer(
                batch_data, padding_mask=pad_mask, lengths=lens_t,
                deterministic=True, no_eos=False, eos_cov_threshold=0.0,
            )
            seed = [int(i) for i in det_idxs[0] if int(i) < n]
            t_seed = time.perf_counter() - t0
            seed_cov = coverage_exact(pts, seed, name)
            seed_r = reward_scalar(seed_cov, len(seed), n,
                                   lam=args.lam, tau=args.tau,
                                   tau_penalty=args.tau_penalty)

            # Coverage gate only applies in oracle mode (uses disc_vis).
            eff_cov_gate = None
            if not geo_free:
                if args.cov_gate_relative is not None:
                    eff_cov_gate = max(0.0, seed_cov - args.cov_gate_relative)
                else:
                    eff_cov_gate = args.cov_gate
            t0 = time.perf_counter()
            ed_S, ed_steps, ed_stopped, ed_rejected = editor_rollout(
                editor, pts, vis, seed, n,
                max_steps=args.rollout_max_steps,
                stop_threshold=args.stop_threshold,
                device=device,
                cov_gate=eff_cov_gate,
                geo_free=geo_free,
                topology=getattr(args, "topology_features", False),
            )
            t_editor = time.perf_counter() - t0
            ed_cov = coverage_exact(pts, ed_S, name)
            ed_r = reward_scalar(ed_cov, len(ed_S), n,
                                 lam=args.lam, tau=args.tau,
                                 tau_penalty=args.tau_penalty)

            # Full LS reference (optional).
            ls_cov = ls_size = ls_r = ls_steps = None
            t_ls = None
            recovery = None
            if args.include_ls_reference:
                t0 = time.perf_counter()
                ls_sol, _, ls_stats = local_search_improve_disc(
                    pts, list(seed), name, n,
                    max_iter=200, enable_swap=True,
                    enable_remove=True, enable_add=True,
                    lam=args.lam, tau=args.tau, tau_penalty=args.tau_penalty,
                    cap_at_tau=False, n_samples=args.disc_vis_samples,
                    reward_fn_fallback=None, monotone_coverage=True,
                )
                t_ls = time.perf_counter() - t0
                ls_cov = coverage_exact(pts, ls_sol, name)
                ls_size = len(ls_sol)
                ls_r = reward_scalar(ls_cov, ls_size, n,
                                     lam=args.lam, tau=args.tau,
                                     tau_penalty=args.tau_penalty)
                ls_steps = (ls_stats.get("n_remove", 0)
                            + ls_stats.get("n_add", 0)
                            + ls_stats.get("n_swap", 0))
                d_full = ls_r - seed_r
                d_ed = ed_r - seed_r
                if d_full > 1e-6:
                    recovery = d_ed / d_full

            opt_sol = _read_opt_solution(val_dir, name)
            opt_size = len(opt_sol) if opt_sol else None

            records.append({
                "name": name, "n": n,
                "seed_size": len(seed), "seed_cov": seed_cov, "seed_r": seed_r,
                "ed_size": len(ed_S), "ed_cov": ed_cov, "ed_r": ed_r,
                "ed_steps": ed_steps, "ed_stopped": ed_stopped,
                "ed_rejected": ed_rejected,
                "ls_size": ls_size, "ls_cov": ls_cov, "ls_r": ls_r,
                "ls_steps": ls_steps,
                "recovery": recovery,
                "opt_size": opt_size,
                "seed_chv": len(seed) / max(1, n),
                "ed_chv":   len(ed_S) / max(1, n),
                "ls_chv":   (ls_size / max(1, n)) if ls_size is not None else None,
                "seed_ratio_opt": (len(seed) / opt_size) if opt_size else None,
                "ed_ratio_opt":   (len(ed_S) / opt_size) if opt_size else None,
                "ls_ratio_opt":   (ls_size  / opt_size) if opt_size and ls_size else None,
                "t_seed_s": t_seed, "t_editor_s": t_editor, "t_ls_s": t_ls,
            })

            if (i + 1) % 100 == 0:
                arr = np.array([r["recovery"] for r in records
                                if r["recovery"] is not None]) if args.include_ls_reference else None
                if arr is not None and len(arr):
                    print(f"  [{i+1}/{len(val_ds)}] recovery med={np.median(arr):.3f} "
                          f"mean={arr.mean():.3f}")

    # ── Summary ──────────────────────────────────────────────────────
    seed_cov = np.array([r["seed_cov"] for r in records])
    seed_size = np.array([r["seed_size"] for r in records])
    ed_cov = np.array([r["ed_cov"] for r in records])
    ed_size = np.array([r["ed_size"] for r in records])
    ed_steps = np.array([r["ed_steps"] for r in records])
    ed_stopped_mask = np.array([r["ed_stopped"] for r in records])
    t_seed = np.array([r["t_seed_s"] for r in records])
    t_ed = np.array([r["t_editor_s"] for r in records])

    # |S|/n (Chvátal density) averaged per polygon, not |S|_mean / n_mean.
    seed_chv = float(np.mean([r["seed_chv"] for r in records]))
    ed_chv   = float(np.mean([r["ed_chv"]   for r in records]))
    # |S|/OPT averaged over polygons that have a .solution file.
    seed_opt = ed_opt = ls_opt = None
    sr = [r["seed_ratio_opt"] for r in records if r["seed_ratio_opt"] is not None]
    er = [r["ed_ratio_opt"]   for r in records if r["ed_ratio_opt"]   is not None]
    if sr:
        seed_opt = float(np.mean(sr))
    if er:
        ed_opt = float(np.mean(er))

    summary: dict = {
        "n_polygons": len(records),
        "pointer_checkpoint": args.pointer_checkpoint,
        "editor_checkpoint": args.editor_checkpoint,
        "stop_threshold": args.stop_threshold,
        "cov_gate_mode": (
            f"relative({args.cov_gate_relative})" if args.cov_gate_relative is not None
            else (f"absolute({args.cov_gate})" if args.cov_gate is not None else "none")
        ),
        "ed_rejected_total": int(sum(r["ed_rejected"] for r in records)),
        "seed_cov_mean": float(seed_cov.mean()),
        "seed_size_mean": float(seed_size.mean()),
        "seed_chv_mean": seed_chv,
        "seed_ratio_opt_mean": seed_opt,
        "ed_cov_mean": float(ed_cov.mean()),
        "ed_size_mean": float(ed_size.mean()),
        "ed_chv_mean": ed_chv,
        "ed_ratio_opt_mean": ed_opt,
        "ed_steps_mean": float(ed_steps.mean()),
        "ed_stopped_frac": float(ed_stopped_mask.mean()),
        "t_seed_mean_ms": float(t_seed.mean() * 1000),
        "t_editor_mean_ms": float(t_ed.mean() * 1000),
        "wall_s": time.perf_counter() - t_total,
    }

    if args.include_ls_reference:
        ls_cov = np.array([r["ls_cov"] for r in records])
        ls_size = np.array([r["ls_size"] for r in records])
        ls_steps = np.array([r["ls_steps"] for r in records])
        t_ls = np.array([r["t_ls_s"] for r in records])
        recoveries = [r["recovery"] for r in records if r["recovery"] is not None]
        ls_chv = float(np.mean([r["ls_chv"] for r in records if r["ls_chv"] is not None]))
        lr = [r["ls_ratio_opt"] for r in records if r["ls_ratio_opt"] is not None]
        ls_opt = float(np.mean(lr)) if lr else None

        summary.update({
            "ls_cov_mean": float(ls_cov.mean()),
            "ls_size_mean": float(ls_size.mean()),
            "ls_chv_mean": ls_chv,
            "ls_ratio_opt_mean": ls_opt,
            "ls_steps_mean": float(ls_steps.mean()),
            "t_ls_mean_ms": float(t_ls.mean() * 1000),
            "speedup_editor_vs_ls": float(t_ls.mean() / max(1e-9, t_ed.mean())),
            "recovery_mean":   float(np.mean(recoveries)) if recoveries else None,
            "recovery_median": float(np.median(recoveries)) if recoveries else None,
            "recovery_p10":    float(np.percentile(recoveries, 10)) if len(recoveries) > 1 else None,
            "recovery_p25":    float(np.percentile(recoveries, 25)) if len(recoveries) > 1 else None,
            "recovery_p75":    float(np.percentile(recoveries, 75)) if len(recoveries) > 1 else None,
            "recovery_p90":    float(np.percentile(recoveries, 90)) if len(recoveries) > 1 else None,
            "frac_recovery_ge_0.5":  float(np.mean(np.array(recoveries) >= 0.5)) if recoveries else None,
            "frac_recovery_ge_0.8":  float(np.mean(np.array(recoveries) >= 0.8)) if recoveries else None,
            "frac_recovery_ge_1.0":  float(np.mean(np.array(recoveries) >= 1.0)) if recoveries else None,
        })

    print()
    print("=" * 70)
    print(f"  Editor evaluation  ({len(records)} polygons)")
    print("=" * 70)
    # ── Primary metrics: one row per method ─────────────────────────
    #   cov     = coverage of the final guard set (CGAL exact)
    #   |S|/n   = Chvátal density (guards per polygon vertex)
    #   |S|/OPT = approximation ratio against exact AGP optimum
    #   time    = mean wall time per polygon
    def _fmt(x):
        return "  -  " if x is None else f"{x:.3f}"
    print(f"  {'':<12}{'cov':>8}{'|S|/n':>10}{'|S|/OPT':>10}{'time/poly':>14}")
    print(f"  {'pointer':<12}"
          f"{summary['seed_cov_mean']:8.4f}"
          f"{_fmt(summary['seed_chv_mean']):>10}"
          f"{_fmt(summary['seed_ratio_opt_mean']):>10}"
          f"{summary['t_seed_mean_ms']:>11.1f} ms")
    print(f"  {'+ editor':<12}"
          f"{summary['ed_cov_mean']:8.4f}"
          f"{_fmt(summary['ed_chv_mean']):>10}"
          f"{_fmt(summary['ed_ratio_opt_mean']):>10}"
          f"{summary['t_editor_mean_ms']:>11.1f} ms")
    if args.include_ls_reference:
        print(f"  {'+ full LS':<12}"
              f"{summary['ls_cov_mean']:8.4f}"
              f"{_fmt(summary['ls_chv_mean']):>10}"
              f"{_fmt(summary['ls_ratio_opt_mean']):>10}"
              f"{summary['t_ls_mean_ms']:>11.1f} ms")
        # ── Diagnostics (each metric labelled with its role) ────────
        print()
        print("  Diagnostics:")
        print(f"    recovery (editor's reward gain over seed, divided by LS's gain):")
        print(f"      median={summary['recovery_median']:.3f}  "
              f"mean={summary['recovery_mean']:.3f}  "
              f"p25={summary['recovery_p25']:.3f}  "
              f"p75={summary['recovery_p75']:.3f}  "
              f"p90={summary['recovery_p90']:.3f}")
        print(f"      P[recovery >= 0.5]={summary['frac_recovery_ge_0.5']:.3f}  "
              f"(editor at least halfway to LS)")
        print(f"      P[recovery >= 0.8]={summary['frac_recovery_ge_0.8']:.3f}  "
              f"(editor effectively matches LS)")
        print(f"      P[recovery >= 1.0]={summary['frac_recovery_ge_1.0']:.3f}  "
              f"(editor beats LS)")
        print(f"    editor stop_share={summary['ed_stopped_frac']:.2f}  "
              f"(fraction of rollouts where STOP head fired naturally; "
              f"the rest hit the step cap)")
        print(f"    editor steps mean={summary['ed_steps_mean']:.1f}  "
              f"LS steps mean={summary['ls_steps_mean']:.1f}")
        print(f"    speedup editor vs LS = {summary['speedup_editor_vs_ls']:.2f}x "
              f"(>1 = editor faster; at batch=1 NN overhead dominates)")
    print(f"  cov_gate  {summary['cov_gate_mode']}")
    print(f"  rejected  {summary['ed_rejected_total']} edits blocked by coverage gate")
    print(f"  wall  {summary['wall_s']:.1f}s")

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(REPO_ROOT, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"\n  saved -> {out_path}")


if __name__ == "__main__":
    main()
