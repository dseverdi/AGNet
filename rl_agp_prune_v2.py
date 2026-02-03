"""rl_agp_prune_v2.py

Minimal teacher->student training for one-pass pruning.
- Teacher: ss_agp_prune checkpoint (oracle-guided pruning)
- Student: one-pass scorer trained to predict teacher masks
- Optional eval/reporting (oracle only for coverage reporting)
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import pickle
import random
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from dataset import Dataset, agp_read_samples, collate_fn
from eval_reporting import make_report
from ss_agp_prune import (
    PrunePolicyNet,
    SkgeomVisibilityOracle,
    _list_pol_files,
    _read_opt_solution,
    run_policy_pruning,
)

_TEACHER_STATE: Dict[str, object] = {}


def make_oracle() -> SkgeomVisibilityOracle:
    return SkgeomVisibilityOracle(coverage_threshold=0.99, verbose=False)


@torch.no_grad()
def teacher_mask(teacher: PrunePolicyNet, oracle, pts: torch.Tensor, name: str, max_steps: Optional[int]) -> np.ndarray:
    active, _ = run_policy_pruning(teacher, oracle, pts, name, deterministic=True, max_steps=max_steps)
    return active


@torch.no_grad()
def select_onepass(student: PrunePolicyNet, points: torch.Tensor, select_threshold: float) -> np.ndarray:
    device = next(student.parameters()).device
    n = int(points.shape[0])
    pad_mask = torch.ones(1, n, device=device, dtype=torch.bool)
    pts = points.to(device).unsqueeze(0)
    s = torch.ones(1, n, device=device, dtype=torch.bool)
    blocked = torch.zeros(1, n, device=device, dtype=torch.bool)
    logits = student(pts, pad_mask, s, blocked).squeeze(0)

    keep = logits > float(select_threshold)
    if not torch.any(keep):
        active = torch.argmax(logits, dim=0, keepdim=True)
    else:
        active = torch.nonzero(keep, as_tuple=False).squeeze(-1)
    return active.detach().cpu().numpy().astype(np.int64)


@torch.no_grad()
def evaluate_onepass(
    student: PrunePolicyNet,
    val_dataset: Dataset,
    coverage_eval_oracle: SkgeomVisibilityOracle,
    eval_k: int,
    sol_dir: str,
    select_threshold: float,
    verbose: bool,
) -> List[Dict]:
    per_instance: List[Dict] = []
    student.eval()

    for j in range(eval_k):
        pts, _, name = val_dataset[j]
        if verbose:
            print(f"[eval] {j + 1}/{eval_k}: {name}")
        t0 = time.perf_counter()
        active = select_onepass(student, pts, select_threshold)
        dt = float(time.perf_counter() - t0)

        n = int(pts.shape[0])
        cov = None
        try:
            cov = coverage_eval_oracle.coverage(pts.detach().cpu().numpy(), active, name)
        except Exception:
            cov = None

        opt_sol = _read_opt_solution(sol_dir, name)
        opt_size = len(opt_sol) if opt_sol is not None else None
        approx_ratio = (float(active.size) / float(opt_size)) if opt_size else None

        per_instance.append(
            {
                "name": name,
                "n": n,
                "guards": int(active.size),
                "guard_ratio": float(active.size) / max(1.0, float(n)),
                "coverage": cov,
                "opt_size": opt_size,
                "approx_ratio": approx_ratio,
                "time_s": dt,
            }
        )

    return per_instance


def _init_teacher_worker(ckpt_path: str, hidden_size: int) -> None:
    global _TEACHER_STATE
    device = torch.device("cpu")
    teacher = PrunePolicyNet(hidden_size=int(hidden_size), use_coords=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    teacher.load_state_dict(state)
    teacher.eval().to(device)
    oracle = make_oracle()
    _TEACHER_STATE = {"teacher": teacher, "oracle": oracle}


def _compute_teacher_mask(idx: int, points: np.ndarray, name: str, max_steps: Optional[int]) -> Dict:
    teacher: PrunePolicyNet = _TEACHER_STATE["teacher"]
    oracle: SkgeomVisibilityOracle = _TEACHER_STATE["oracle"]
    pts = torch.tensor(points, dtype=torch.float32)
    active = teacher_mask(teacher, oracle, pts, name, max_steps)
    return {"idx": int(idx), "name": name, "active": active}


def main() -> None:
    load_dotenv()
    DATASET_PATH = os.getenv("DATASET_PATH")
    if not DATASET_PATH:
        raise EnvironmentError("DATASET_PATH environment variable must be set in .env file.")

    default_train = os.path.join(DATASET_PATH, "train")
    default_val = os.path.join(DATASET_PATH, "dev")

    parser = argparse.ArgumentParser(description="Minimal one-pass pruning training")
    parser.add_argument("--teacher-checkpoint", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for teacher mask precompute.")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-size", type=int, default=-1)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation after training.")
    parser.add_argument("--select-threshold", type=float, default=0.0)
    parser.add_argument("--agp_val_dir", type=str, default=default_val)
    parser.add_argument("--eval-k", type=int, default=-1)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--results-dir", type=str, default="results/v3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    def vprint(msg: str) -> None:
        if bool(args.verbose):
            print(msg)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_paths = _list_pol_files(default_train)
    if len(train_paths) == 0:
        raise SystemExit(f"No .pol files found under {default_train}")

    if int(args.train_size) > 0:
        train_paths = train_paths[: int(args.train_size)]
        vprint(f"[phase] using {len(train_paths)} train files (train-size cap)")

    vprint("[phase] loading samples")
    train_samples = agp_read_samples(train_paths, normalize=bool(args.normalize))
    train_dataset = Dataset(train_samples)

    train_loader = DataLoader(train_dataset, batch_size=int(args.batch_size), shuffle=True, collate_fn=collate_fn)

    oracle = make_oracle()

    teacher = PrunePolicyNet(hidden_size=int(args.hidden_size), use_coords=True)
    ckpt = torch.load(args.teacher_checkpoint, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    teacher.load_state_dict(state)
    teacher.eval().to(device)

    student = PrunePolicyNet(hidden_size=int(args.hidden_size), use_coords=True).to(device)

    opt = torch.optim.Adam(student.parameters(), lr=float(args.lr))
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    max_steps = None if int(args.max_steps) < 0 else int(args.max_steps)

    vprint("[phase] precomputing teacher masks")
    teacher_masks: Dict[str, np.ndarray] = {}
    cache_key = f"{args.teacher_checkpoint}|{len(train_paths)}|{bool(args.normalize)}|{max_steps}"
    cache_name = "teacher_masks_" + hashlib.md5(cache_key.encode("utf-8")).hexdigest() + ".pkl"
    cache_path = os.path.join(args.checkpoint_dir, cache_name)
    partial_cache_path = cache_path + ".partial"

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if isinstance(cached, dict):
                teacher_masks = cached
                vprint(f"[phase] loaded cached teacher masks from {cache_path}")
        except Exception:
            teacher_masks = {}

    if os.path.exists(partial_cache_path) and len(teacher_masks) < len(train_dataset):
        try:
            with open(partial_cache_path, "rb") as f:
                partial_cached = pickle.load(f)
            if isinstance(partial_cached, dict) and len(partial_cached) > len(teacher_masks):
                teacher_masks = partial_cached
                vprint(f"[phase] loaded partial cached teacher masks from {partial_cache_path}")
        except Exception:
            pass

    if len(teacher_masks) == len(train_dataset):
        vprint("[phase] cache complete, skipping precompute")
    elif int(args.workers) <= 1:
        for idx in range(len(train_dataset)):
            pts, _, name = train_dataset[idx]
            if name in teacher_masks:
                continue
            if bool(args.verbose) and (idx % 50 == 0):
                print(f"[teacher] {idx + 1}/{len(train_dataset)}: {name}")
            active = teacher_mask(teacher, oracle, pts.to(device), name, max_steps)
            teacher_masks[name] = active
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=int(args.workers),
            mp_context=ctx,
            initializer=_init_teacher_worker,
            initargs=(args.teacher_checkpoint, int(args.hidden_size)),
        ) as ex:
            jobs = []
            for idx in range(len(train_dataset)):
                pts, _, name = train_dataset[idx]
                if name in teacher_masks:
                    continue
                pts_np = pts.detach().cpu().numpy()
                jobs.append(ex.submit(_compute_teacher_mask, idx, pts_np, name, max_steps))

            completed = len(teacher_masks)
            save_interval = 50
            for fut in as_completed(jobs):
                try:
                    res = fut.result(timeout=300)
                except Exception as exc:
                    completed += 1
                    if bool(args.verbose):
                        print(f"[teacher] {completed}/{len(train_dataset)}: worker failed ({exc})", flush=True)
                    continue
                completed += 1
                if bool(args.verbose):
                    print(f"[teacher] {completed}/{len(train_dataset)}: {res['name']}", flush=True)
                teacher_masks[res["name"]] = res["active"]

                if completed % save_interval == 0:
                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                    try:
                        with open(partial_cache_path, "wb") as f:
                            pickle.dump(teacher_masks, f)
                        print(
                            f"[phase] saved incremental teacher mask cache ({len(teacher_masks)}/{len(train_dataset)}) to {partial_cache_path}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[error] failed to save incremental cache: {e}", flush=True)

        if len(teacher_masks) < len(train_dataset):
            vprint("[phase] recovering failed teacher masks sequentially")
            for idx in range(len(train_dataset)):
                pts, _, name = train_dataset[idx]
                if name in teacher_masks:
                    continue
                if bool(args.verbose):
                    print(f"[teacher][recover] {name}", flush=True)
                active = teacher_mask(teacher, oracle, pts.to(device), name, max_steps)
                teacher_masks[name] = active

    if len(teacher_masks) == len(train_dataset):
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(teacher_masks, f)
            vprint(f"[phase] saved teacher mask cache to {cache_path}")
            if os.path.exists(partial_cache_path):
                try:
                    os.remove(partial_cache_path)
                except Exception:
                    pass
        except Exception:
            pass

    vprint("[phase] training student")
    for epoch in range(1, int(args.epochs) + 1):
        student.train()
        if bool(args.verbose):
            print(f"[epoch] {epoch}/{int(args.epochs)}")
        epoch_losses: List[float] = []
        epoch_guard_ratios: List[float] = []

        for batch_idx, (points_pad, pad_mask, lengths, names) in enumerate(train_loader, start=1):
            points_pad = points_pad.to(device)
            pad_mask = pad_mask.to(device)

            targets = torch.zeros_like(pad_mask, dtype=torch.float32, device=device)
            batch_guard_ratios: List[float] = []

            with torch.no_grad():
                for i in range(points_pad.shape[0]):
                    n = int(lengths[i])
                    pts = points_pad[i, :n]
                    active = teacher_masks.get(names[i])
                    if active is None:
                        active = teacher_mask(teacher, oracle, pts, names[i], max_steps)
                    targets[i, active] = 1.0
                    batch_guard_ratios.append(float(active.size) / max(1.0, float(n)))

            b, n, _ = points_pad.shape
            s = torch.ones(b, n, device=device, dtype=torch.bool)
            blocked = torch.zeros(b, n, device=device, dtype=torch.bool)
            logits = student(points_pad, pad_mask, s, blocked)
            logits = logits.masked_fill(~pad_mask, 0.0)

            loss_raw = bce(logits, targets)
            loss = (loss_raw * pad_mask.float()).sum() / pad_mask.float().sum().clamp_min(1.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()

            epoch_losses.append(float(loss.item()))
            epoch_guard_ratios.extend(batch_guard_ratios)

            if bool(args.verbose) and (batch_idx % 20 == 0):
                print(
                    f"[train] epoch {epoch:02d}/{int(args.epochs)} batch {batch_idx:03d}/{len(train_loader)} "
                    f"| loss {float(np.mean(epoch_losses)):.4f} "
                    f"| |S|/n mean {float(np.mean(epoch_guard_ratios)):.3f}"
                )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(
        args.checkpoint_dir,
        f"rl_agp_prune_v2_h{args.hidden_size}_lr{args.lr}_epochs{args.epochs}.pt",
    )
    torch.save({"model": student.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    print("Training complete.")

    if args.evaluate:
        val_paths = _list_pol_files(args.agp_val_dir)
        if len(val_paths) == 0:
            raise SystemExit(f"No .pol files found under {args.agp_val_dir}")
        val_samples = agp_read_samples(val_paths, normalize=bool(args.normalize))
        val_dataset = Dataset(val_samples)

        coverage_eval_oracle = SkgeomVisibilityOracle(coverage_threshold=0.0, verbose=False)
        eval_k = len(val_dataset) if int(args.eval_k) < 0 else min(int(args.eval_k), len(val_dataset))

        per_instance = evaluate_onepass(
            student,
            val_dataset,
            coverage_eval_oracle,
            eval_k=eval_k,
            sol_dir=args.agp_val_dir,
            select_threshold=float(args.select_threshold),
            verbose=bool(args.verbose),
        )

        report = make_report(
            method="rl_agp_prune_v2",
            per_instance=per_instance,
            args=vars(args),
            dataset={
                "path": args.agp_val_dir,
                "eval_k": int(eval_k),
                "train_k": int(len(train_paths)),
            },
            oracle={
                "mode": "exact",
                "coverage_threshold": 0.99,
                "coverage_metric": "exact",
            },
            timing={},
        )
        report["checkpoint"] = ckpt_path

        os.makedirs(args.results_dir, exist_ok=True)
        out_path = os.path.join(args.results_dir, "rl_agp_prune_v2_report.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

        print("\n--- One-pass prune eval on dataset polygons ---")
        s = report["summary"]
        msg = f"Dataset eval k={eval_k} | |S| mean={s['guards']['mean']:.2f} | |S|/n mean={s['guard_ratio']['mean']:.3f}"
        if s["coverage"]["mean"] is not None:
            msg += f" | geo-cov mean={s['coverage']['mean']:.3f}"
        if s["approx_ratio"]["mean"] is not None:
            msg += f" | |S|/opt mean={s['approx_ratio']['mean']:.2f}"
        print(msg)
        print(f"Results summary saved to {out_path}")


if __name__ == "__main__":
    main()
