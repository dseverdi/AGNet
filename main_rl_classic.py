"""REINFORCE training for the classic (vertex-guard) Art Gallery Problem.

Uses PointerNetwork + Critic from models.py, trained with policy-gradient on
a coverage-based reward.  All parameters live in data/config/config_rl_classic.json.

Usage:
    python main_rl_classic.py              # single run from config
    python main_rl_classic.py grid         # grid search over hyperparameters
"""

import argparse
import csv
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models import PointerNetwork, Critic
from dataset import load_classic_datasets, agp_collate_fn
from losses import Reward, RewardHardCoverage, RewardOptimalRatio, RewardExpGuardPenalty
from main_classic_gt import compute_guard_coverage


# ── config ────────────────────────────────────────────────────────────────────

CONFIG_PATH = Path("data/config/config_rl_classic.json")
CONFIG_DOUBLE_PATH = Path("data/config/config_rl_classic_double.json")
CONFIG_HARD_PATH = Path("data/config/config_rl_classic_hard.json")
CONFIG_DOUBLE_HARD_PATH = Path("data/config/config_rl_classic_double_hard.json")
CONFIG_EXP_PATH = Path("data/config/config_rl_classic_exp.json")
CONFIG_DOUBLE_EXP_PATH = Path("data/config/config_rl_classic_double_exp.json")
CONFIG_OPTIMAL_RATIO_PATH = Path("data/config/config_rl_classic_optimal_ratio.json")
CONFIG_DOUBLE_OPTIMAL_RATIO_PATH = Path("data/config/config_rl_classic_double_optimal_ratio.json")

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


# ── coverage evaluation ──────────────────────────────────────────────────────

def _eval_one(args):
    """Evaluate a single room (runs in worker process)."""
    room, guards, n = args
    rel = len(guards) / n if n > 0 else 1.0
    if len(guards) == 0:
        return 0.0, rel
    try:
        cov = compute_guard_coverage(room, guards)
    except Exception:
        cov = 0.0
    return cov, rel


_worker_pool: ProcessPoolExecutor | None = None

def _get_pool(n_workers: int = 8) -> ProcessPoolExecutor:
    global _worker_pool
    if _worker_pool is None:
        _worker_pool = ProcessPoolExecutor(max_workers=n_workers)
    return _worker_pool


def compute_batch_coverage(vertices_padded, seq_lens, actions):
    """Compute guard coverage and relative guard count for a batch.

    Parallelises across rooms using a persistent process pool.
    Returns (coverages, rel_lengths) each of shape (batch,).
    """
    batch_size = vertices_padded.size(1)

    tasks = []
    for i in range(batch_size):
        n = seq_lens[i].item()
        room = vertices_padded[:n, i].cpu().numpy()
        acts = actions[i].cpu()
        guards = acts[(acts >= 0) & (acts < n)].unique().numpy()
        tasks.append((room, guards, n))

    results = list(_get_pool().map(_eval_one, tasks))

    coverages = torch.tensor([r[0] for r in results], dtype=torch.float)
    rel_lengths = torch.tensor([r[1] for r in results], dtype=torch.float)
    return coverages, rel_lengths


# ── training ──────────────────────────────────────────────────────────────────

def train(cfg, run_name: str | None = None, log_subdir: str = "rl_classic") -> dict:
    """Run one training session.

    Returns dict with best_coverage, best_guards, and per-epoch log rows.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── data ──────────────────────────────────────────────────────────
    full_ds = load_classic_datasets(cfg["data_dir"], cfg["chunk_size"])
    max_samples = cfg.get("max_samples")
    if max_samples and len(full_ds) > max_samples:
        full_ds, _ = random_split(full_ds, [max_samples, len(full_ds) - max_samples])
        print(f"Capped to {max_samples} samples")
    n_val = int(len(full_ds) * cfg["val_fraction"])
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val])
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    loader_kw = dict(batch_size=cfg["batch_size"], collate_fn=agp_collate_fn,
                     num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)

    # ── model + critic ────────────────────────────────────────────────
    hs = cfg["hidden_size"]
    bi = cfg["bidirectional"]
    dec_dim = hs * (2 if bi else 1)

    model = PointerNetwork(
        input_size=2, hidden_size=hs, output_size=dec_dim,
        n_categories=dec_dim, attention_dim=cfg["attention_dim"],
        bidirectional=bi,
    ).to(device)

    critic = Critic(input_size=2, hidden_size=hs, bidirectional=bi).to(device)

    reward_type = cfg.get("reward_type", "vanilla")
    if reward_type == "hard":
        reward_fn = RewardHardCoverage(cfg.get("coverage_threshold", 0.999))
    elif reward_type == "optimal_ratio":
        reward_fn = RewardOptimalRatio(cfg.get("coverage_threshold", 0.999),
                                       cfg.get("cov_penalty", 5.0))
    elif reward_type == "exp":
        reward_fn = RewardExpGuardPenalty(cfg.get("exp_scale", 3.0),
                                          cfg.get("coverage_weight", 2.0))
    else:  # "vanilla"
        reward_fn = Reward(cfg["coverage_weight"])
    optimizer = optim.Adam(
        list(model.parameters()) + list(critic.parameters()), lr=cfg["lr"])

    # ── directories ───────────────────────────────────────────────────
    if run_name:
        model_dir = Path(cfg["model_dir"]) / run_name
        log_dir = Path(f"data/logs/{log_subdir}")
    else:
        model_dir = Path(cfg["model_dir"])
        log_dir = Path(f"data/logs/{log_subdir}")
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model params:  {sum(p.numel() for p in model.parameters()):,}")
    print(f"Critic params: {sum(p.numel() for p in critic.parameters()):,}")

    # ── train loop ────────────────────────────────────────────────────
    best_coverage = 0.0
    best_guards = 1.0
    entropy_w = cfg["entropy_weight"]
    max_grad = cfg["max_grad_norm"]
    epoch_logs: list[dict] = []

    for epoch in range(cfg["num_epochs"]):
        model.train()
        critic.train()

        epoch_cov, epoch_guards, epoch_loss, n_batches = 0.0, 0.0, 0.0, 0

        label = run_name or "train"
        pbar = tqdm(train_loader, desc=f"[{label}] Epoch {epoch+1}/{cfg['num_epochs']}")
        for vertices, seq_lens, gt_guards, gt_n_guards in pbar:
            vertices, seq_lens = vertices.to(device), seq_lens.to(device)

            actions, log_probs, entropies = model.rl_forward(vertices, seq_lens)

            coverages, rel_lengths = compute_batch_coverage(
                vertices.cpu(), seq_lens.cpu(), actions.cpu())

            if reward_type == "optimal_ratio":
                n_pred = torch.tensor([
                    acts[(acts >= 0) & (acts < seq_lens[i].cpu().item())].unique().shape[0]
                    for i, acts in enumerate(actions.cpu())
                ], dtype=torch.float)
                rewards = reward_fn(coverages, rel_lengths, gt_n_guards.float(), n_pred).to(device)
            else:
                rewards = reward_fn(coverages, rel_lengths).to(device)

            values    = critic(vertices, seq_lens)
            advantage = (rewards - values).detach()

            valid = (actions >= 0).float().to(device)
            sample_lp  = (log_probs * valid).sum(dim=1)
            sample_ent = (entropies * valid).sum(dim=1)

            reinforce_loss = -(advantage * sample_lp).mean()
            critic_loss    = ((rewards - values) ** 2).mean()
            loss = reinforce_loss + critic_loss - entropy_w * sample_ent.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(critic.parameters()), max_grad)
            optimizer.step()

            epoch_cov += coverages.mean().item()
            epoch_guards += rel_lengths.mean().item()
            epoch_loss += loss.item()
            n_batches += 1

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                cov=f"{coverages.mean().item():.3f}",
                guards=f"{rel_lengths.mean().item():.3f}",
            )

        # ── epoch averages ────────────────────────────────────────────
        avg_cov = epoch_cov / n_batches
        avg_guards = epoch_guards / n_batches
        avg_loss = epoch_loss / n_batches

        # ── validation ────────────────────────────────────────────────
        val_cov, val_guards = evaluate(model, val_loader, device)

        row = dict(epoch=epoch + 1, train_cov=avg_cov, train_guards=avg_guards,
                   train_loss=avg_loss, val_cov=val_cov, val_guards=val_guards)
        epoch_logs.append(row)

        if val_cov > best_coverage:
            best_coverage = val_cov
            best_guards = val_guards
            torch.save(model.state_dict(), model_dir / "best_model.pt")
            torch.save(critic.state_dict(), model_dir / "best_critic.pt")
            tqdm.write(f"  → Best model @ epoch {epoch+1}, "
                       f"coverage {val_cov:.4f}, guards {val_guards:.4f}")

    # ── save per-epoch CSV ────────────────────────────────────────────
    csv_name = f"{run_name}.csv" if run_name else "single_run.csv"
    csv_path = log_dir / csv_name
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_logs[0].keys())
        writer.writeheader()
        writer.writerows(epoch_logs)
    print(f"Epoch log → {csv_path}")

    print(f"Done. Best val coverage: {best_coverage:.4f}, guards: {best_guards:.4f}")
    return dict(best_coverage=best_coverage, best_guards=best_guards,
                epoch_logs=epoch_logs)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_cov, total_guards, total_n = 0.0, 0.0, 0

    for vertices, seq_lens, _, _ in loader:
        vertices, seq_lens = vertices.to(device), seq_lens.to(device)
        actions, _, _ = model.rl_forward(vertices, seq_lens)
        coverages, rel_lengths = compute_batch_coverage(
            vertices.cpu(), seq_lens.cpu(), actions.cpu())
        bs = vertices.size(1)
        total_cov += coverages.sum().item()
        total_guards += rel_lengths.sum().item()
        total_n += bs

    avg_cov = total_cov / total_n
    avg_guards = total_guards / total_n
    tqdm.write(f"  Val coverage: {avg_cov:.4f}, guards: {avg_guards:.4f}")
    return avg_cov, avg_guards


# ── grid search ───────────────────────────────────────────────────────────────

GRID = {
    "lr":               [1e-3, 1e-4],
    "entropy_weight":   [0.01, 0.001],
    "coverage_weight":  [0.25, 0.5, 0.7],
}

def grid_search(config_path: Path = CONFIG_PATH, log_subdir: str = "rl_classic"):
    base_cfg = load_config(config_path)
    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))

    print(f"Grid search: {len(combos)} runs  (config: {config_path})")
    print(f"  {keys}")
    for c in combos:
        print(f"  {c}")
    print()

    summary_path = Path(f"data/logs/{log_subdir}/grid_summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for combo in combos:
        cfg = {**base_cfg}
        for k, v in zip(keys, combo):
            cfg[k] = v

        run_name = "_".join(f"{k}{v}" for k, v in zip(keys, combo))
        print(f"\n{'='*60}")
        print(f"Run: {run_name}")
        print(f"{'='*60}")

        result = train(cfg, run_name=run_name, log_subdir=log_subdir)
        row = {k: v for k, v in zip(keys, combo)}
        row["best_coverage"] = result["best_coverage"]
        row["best_guards"] = result["best_guards"]
        results.append(row)

        # Write summary after each run (crash-safe)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # ── print final ranking ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Grid search results (sorted by coverage):")
    print(f"{'='*60}")
    results.sort(key=lambda r: r["best_coverage"], reverse=True)
    for i, r in enumerate(results, 1):
        params = ", ".join(f"{k}={r[k]}" for k in keys)
        print(f"  {i:2d}. cov={r['best_coverage']:.4f}  "
              f"guards={r['best_guards']:.4f}  |  {params}")

    # Save winning config back to JSON
    best = results[0]
    for k in keys:
        base_cfg[k] = best[k]
    with open(config_path, "w") as f:
        json.dump(base_cfg, f, indent=4)
    print(f"\nBest config written to {config_path}")
    print(f"Summary → {summary_path}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REINFORCE for AGP")
    parser.add_argument("mode", nargs="?", default="single")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use (default: 0)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    mode = args.mode

    if mode == "grid":
        grid_search()
    elif mode == "double":
        train(load_config(CONFIG_DOUBLE_PATH), log_subdir="rl_classic_double")
    elif mode == "grid-double":
        grid_search(config_path=CONFIG_DOUBLE_PATH, log_subdir="rl_classic_double")
    elif mode == "hard":
        train(load_config(CONFIG_HARD_PATH), log_subdir="rl_classic_hard")
    elif mode == "grid-hard":
        grid_search(config_path=CONFIG_HARD_PATH, log_subdir="rl_classic_hard")
    elif mode == "double-hard":
        train(load_config(CONFIG_DOUBLE_HARD_PATH), log_subdir="rl_classic_double_hard")
    elif mode == "grid-double-hard":
        grid_search(config_path=CONFIG_DOUBLE_HARD_PATH, log_subdir="rl_classic_double_hard")
    elif mode == "exp":
        train(load_config(CONFIG_EXP_PATH), log_subdir="rl_classic_exp")
    elif mode == "grid-exp":
        grid_search(config_path=CONFIG_EXP_PATH, log_subdir="rl_classic_exp")
    elif mode == "double-exp":
        train(load_config(CONFIG_DOUBLE_EXP_PATH), log_subdir="rl_classic_double_exp")
    elif mode == "grid-double-exp":
        grid_search(config_path=CONFIG_DOUBLE_EXP_PATH, log_subdir="rl_classic_double_exp")
    elif mode == "optimal-ratio":
        train(load_config(CONFIG_OPTIMAL_RATIO_PATH), log_subdir="rl_classic_optimal_ratio")
    elif mode == "grid-optimal-ratio":
        grid_search(config_path=CONFIG_OPTIMAL_RATIO_PATH, log_subdir="rl_classic_optimal_ratio")
    elif mode == "double-optimal-ratio":
        train(load_config(CONFIG_DOUBLE_OPTIMAL_RATIO_PATH), log_subdir="rl_classic_double_optimal_ratio")
    elif mode == "grid-double-optimal-ratio":
        grid_search(config_path=CONFIG_DOUBLE_OPTIMAL_RATIO_PATH, log_subdir="rl_classic_double_optimal_ratio")
    else:
        train(load_config())
