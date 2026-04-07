"""Stage 3 fine-tuning: coverage-first RL starting from supervised-double checkpoint.

Loads the best supervised-double-exp model and fine-tunes with RewardOptimalThreshold,
which only penalises guards *above* the ILP-optimal count.  This lets the model
improve guard placement (and thus coverage) without being pushed back to selecting
too many guards.

Usage:
    python main_stage3_finetune.py single               # single run with defaults
    python main_stage3_finetune.py grid                 # grid search
    python main_stage3_finetune.py grid --gpu 1         # grid on GPU 1
    python main_stage3_finetune.py single --checkpoint checkpoints/supervised_double_exp/best_finetuned_model.pt
"""

import argparse
import csv
import itertools
import json
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models import PointerNetwork, Critic
from dataset import load_classic_datasets, agp_collate_fn
from losses import RewardOptimalThreshold
from main_classic_gt import compute_guard_coverage


# ── defaults ─────────────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT = "checkpoints/supervised_double_exp/lr0.0001_entropy_weight0.01_coverage_weight0.5/best_finetuned_model.pt"
DEFAULT_CRITIC_CKPT = "checkpoints/supervised_double_exp/lr0.0001_entropy_weight0.01_coverage_weight0.5/best_finetuned_critic.pt"

BASE_CFG = {
    "hidden_size":      256,
    "attention_dim":    256,
    "batch_size":       1024,
    "num_epochs":       40,
    "lr":               1e-4,
    "max_grad_norm":    1.0,
    "entropy_weight":   0.005,
    "excess_penalty":   2.0,
    "val_split":        0.1,
    "seed":             42,
}

GRID = {
    "lr":             [1e-4, 5e-5, 1e-5],
    "excess_penalty": [1.0, 2.0, 5.0, 10.0],
}

LOG_SUBDIR  = "stage3_double"
MODEL_DIR   = Path("checkpoints/stage3_double")


# ── coverage helper ───────────────────────────────────────────────────────────

def compute_batch_coverage(vertices, seq_lens, actions):
    results = []
    B = vertices.size(1)
    for i in range(B):
        n = seq_lens[i].item()
        verts = vertices[:n, i, :].numpy()
        acts  = actions[i]
        guards = acts[(acts >= 0) & (acts < n)].unique().tolist()
        if len(guards) == 0:
            results.append((0.0, 1.0))
            continue
        cov = compute_guard_coverage(verts, guards)
        rel = len(guards) / n
        results.append((cov, rel))
    coverages   = torch.tensor([r[0] for r in results], dtype=torch.float)
    rel_lengths = torch.tensor([r[1] for r in results], dtype=torch.float)
    return coverages, rel_lengths


# ── eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_cov, total_guards, total_ratio, total_n = 0.0, 0.0, 0.0, 0
    for vertices, seq_lens, _, gt_n_guards in loader:
        vertices, seq_lens = vertices.to(device), seq_lens.to(device)
        actions, _, _ = model.rl_forward(vertices, seq_lens)
        coverages, rel_lengths = compute_batch_coverage(
            vertices.cpu(), seq_lens.cpu(), actions.cpu())
        n_pred = rel_lengths * seq_lens.cpu().float()
        bs = vertices.size(1)
        total_cov    += coverages.sum().item()
        total_guards += rel_lengths.sum().item()
        total_ratio  += (n_pred / gt_n_guards.float()).sum().item()
        total_n      += bs
    avg_cov    = total_cov    / total_n
    avg_guards = total_guards / total_n
    avg_ratio  = total_ratio  / total_n
    tqdm.write(f"  Val  cov={avg_cov:.4f}  guards={avg_guards:.4f}  ratio_to_opt={avg_ratio:.4f}")
    return avg_cov, avg_guards, avg_ratio


# ── training loop ─────────────────────────────────────────────────────────────

def train(cfg: dict,
          checkpoint: str  = DEFAULT_CHECKPOINT,
          critic_ckpt: str = DEFAULT_CRITIC_CKPT,
          run_name: str    = None,
          log_subdir: str  = LOG_SUBDIR) -> dict:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg["seed"])

    # ── data ──
    dataset = load_classic_datasets()
    n_val   = max(1, int(len(dataset) * cfg["val_split"]))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg["seed"]))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  collate_fn=agp_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, collate_fn=agp_collate_fn)

    # ── model ──
    hs      = cfg["hidden_size"]
    dec_dim = hs  # bidirectional=False
    model = PointerNetwork(
        input_size=2, hidden_size=hs, output_size=dec_dim,
        n_categories=dec_dim, attention_dim=cfg["attention_dim"],
        bidirectional=False,
    ).to(device)
    critic = Critic(
        input_size=2, hidden_size=hs, bidirectional=False,
    ).to(device)

    ckpt_path = Path(checkpoint)
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded model from {ckpt_path}")
    else:
        print(f"WARNING: checkpoint not found at {ckpt_path}, training from scratch")

    critic_path = Path(critic_ckpt)
    if critic_path.exists():
        critic.load_state_dict(torch.load(critic_path, map_location=device))
        print(f"Loaded critic from {critic_path}")

    optimizer = optim.Adam(
        list(model.parameters()) + list(critic.parameters()),
        lr=cfg["lr"])

    reward_fn = RewardOptimalThreshold(excess_penalty=cfg["excess_penalty"])

    model_dir = MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir   = Path(f"data/logs/{log_subdir}")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model params:  {sum(p.numel() for p in model.parameters()):,}")
    print(f"excess_penalty={cfg['excess_penalty']}  lr={cfg['lr']}")

    best_coverage      = 0.0
    best_guards        = 1.0
    best_ratio_to_opt  = float("inf")
    entropy_w          = cfg["entropy_weight"]
    max_grad           = cfg["max_grad_norm"]
    epoch_logs: list[dict] = []

    for epoch in range(cfg["num_epochs"]):
        model.train()
        critic.train()
        epoch_cov, epoch_guards, epoch_loss, n_batches = 0.0, 0.0, 0.0, 0

        label = run_name or "stage3"
        pbar  = tqdm(train_loader,
                     desc=f"[{label}] Epoch {epoch+1}/{cfg['num_epochs']}")

        for vertices, seq_lens, gt_guards, gt_n_guards in pbar:
            vertices, seq_lens = vertices.to(device), seq_lens.to(device)

            actions, log_probs, entropies = model.rl_forward(vertices, seq_lens)

            coverages, rel_lengths = compute_batch_coverage(
                vertices.cpu(), seq_lens.cpu(), actions.cpu())

            n_pred = torch.tensor([
                actions.cpu()[i][
                    (actions.cpu()[i] >= 0) &
                    (actions.cpu()[i] < seq_lens[i].cpu().item())
                ].unique().shape[0]
                for i in range(actions.size(0))
            ], dtype=torch.float)

            rewards = reward_fn(
                coverages, rel_lengths,
                gt_n_guards.float(), n_pred
            ).to(device)

            values    = critic(vertices, seq_lens)
            advantage = (rewards - values).detach()

            valid      = (actions >= 0).float().to(device)
            sample_lp  = (log_probs * valid).sum(dim=1)
            sample_ent = (entropies * valid).sum(dim=1)

            reinforce_loss = -(advantage * sample_lp).mean()
            critic_loss    = ((rewards - values) ** 2).mean()
            loss           = reinforce_loss + critic_loss - entropy_w * sample_ent.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(critic.parameters()), max_grad)
            optimizer.step()

            epoch_cov    += coverages.mean().item()
            epoch_guards += rel_lengths.mean().item()
            epoch_loss   += loss.item()
            n_batches    += 1

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                cov=f"{coverages.mean().item():.3f}",
                guards=f"{rel_lengths.mean().item():.3f}",
            )

        avg_cov    = epoch_cov    / n_batches
        avg_guards = epoch_guards / n_batches
        avg_loss   = epoch_loss   / n_batches

        val_cov, val_guards, val_ratio = evaluate(model, val_loader, device)

        row = dict(epoch=epoch + 1,
                   train_cov=avg_cov, train_guards=avg_guards, train_loss=avg_loss,
                   val_cov=val_cov,   val_guards=val_guards,   val_ratio_to_opt=val_ratio)
        epoch_logs.append(row)

        if val_cov > best_coverage:
            best_coverage     = val_cov
            best_guards       = val_guards
            best_ratio_to_opt = val_ratio
            torch.save(model.state_dict(),  model_dir / "best_model.pt")
            torch.save(critic.state_dict(), model_dir / "best_critic.pt")
            tqdm.write(f"  -> Best @ epoch {epoch+1}  "
                       f"cov={val_cov:.4f}  guards={val_guards:.4f}  "
                       f"ratio_to_opt={val_ratio:.4f}")

    csv_name = f"{run_name}.csv" if run_name else "single_run.csv"
    csv_path = log_dir / csv_name
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_logs[0].keys())
        writer.writeheader()
        writer.writerows(epoch_logs)

    print(f"Done.  cov={best_coverage:.4f}  guards={best_guards:.4f}  "
          f"ratio_to_opt={best_ratio_to_opt:.4f}")
    return dict(best_coverage=best_coverage, best_guards=best_guards,
                best_ratio_to_opt=best_ratio_to_opt, epoch_logs=epoch_logs)


# ── grid search ───────────────────────────────────────────────────────────────

def grid_search(checkpoint: str = DEFAULT_CHECKPOINT,
                critic_ckpt: str = DEFAULT_CRITIC_CKPT,
                fixed_lr: float = None):
    """Run grid search over excess_penalty (and optionally lr).

    If fixed_lr is given, only runs combos with that lr value and writes to a
    per-lr partial CSV — intended for parallel execution across processes.
    Call grid-merge afterwards to combine all partial CSVs.
    """
    keys   = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))

    if fixed_lr is not None:
        combos = [c for c in combos if c[keys.index("lr")] == fixed_lr]
        lr_tag       = str(fixed_lr).replace("e-0", "e-").replace("e+0", "e")
        summary_name = f"grid_summary_lr{lr_tag}.csv"
    else:
        summary_name = "grid_summary.csv"

    print(f"Stage-3 grid search: {len(combos)} runs  (lr={fixed_lr or 'all'})")
    print(f"Checkpoint: {checkpoint}")

    summary_path = Path(f"data/logs/{LOG_SUBDIR}/{summary_name}")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for combo in combos:
        cfg = {**BASE_CFG}
        for k, v in zip(keys, combo):
            cfg[k] = v

        run_name = "_".join(f"{k}{v}" for k, v in zip(keys, combo))
        print(f"\n{'='*60}")
        print(f"Run: {run_name}")
        print(f"{'='*60}")

        result = train(cfg, checkpoint=checkpoint, critic_ckpt=critic_ckpt,
                       run_name=run_name, log_subdir=LOG_SUBDIR)
        row = {k: v for k, v in zip(keys, combo)}
        row["best_coverage"]     = result["best_coverage"]
        row["best_guards"]       = result["best_guards"]
        row["best_ratio_to_opt"] = result["best_ratio_to_opt"]
        results.append(row)

        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    _print_results(results, keys)
    print(f"\nSummary -> {summary_path}")


def merge_summaries():
    """Merge per-lr partial CSVs into a single grid_summary.csv."""
    log_dir = Path(f"data/logs/{LOG_SUBDIR}")
    partial_files = sorted(log_dir.glob("grid_summary_lr*.csv"))
    if not partial_files:
        print("No partial summary files found.")
        return

    all_rows = []
    for f in partial_files:
        with open(f) as fh:
            all_rows.extend(list(csv.DictReader(fh)))

    summary_path = log_dir / "grid_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)

    _print_results(all_rows, list(GRID.keys()))
    print(f"\nMerged {len(partial_files)} files -> {summary_path}")


def _print_results(results, keys):
    print(f"\n{'='*60}")
    print(f"Stage-3 results (sorted by coverage):")
    print(f"{'='*60}")
    results = sorted(results, key=lambda r: float(r["best_coverage"]), reverse=True)
    for i, r in enumerate(results, 1):
        params = ", ".join(f"{k}={r[k]}" for k in keys)
        print(f"  {i:2d}. cov={float(r['best_coverage']):.4f}  "
              f"guards={float(r['best_guards']):.4f}  "
              f"ratio_to_opt={float(r['best_ratio_to_opt']):.4f}  |  {params}")


# ── entry point ───────────────────────────────────────────────────────────────

MODES = ["single", "grid", "grid-lr1e-4", "grid-lr5e-5", "grid-lr1e-5", "grid-merge"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage-3 coverage-first fine-tuning")
    parser.add_argument("mode", nargs="?", default="grid", choices=MODES)
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index (default: 0)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to model checkpoint to start from")
    parser.add_argument("--critic-checkpoint", type=str, default=DEFAULT_CRITIC_CKPT,
                        help="Path to critic checkpoint to start from")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    ckpt  = args.checkpoint
    cckpt = args.critic_checkpoint

    if args.mode == "single":
        train(BASE_CFG, checkpoint=ckpt, critic_ckpt=cckpt)
    elif args.mode == "grid":
        grid_search(checkpoint=ckpt, critic_ckpt=cckpt)
    elif args.mode == "grid-lr1e-4":
        grid_search(checkpoint=ckpt, critic_ckpt=cckpt, fixed_lr=1e-4)
    elif args.mode == "grid-lr5e-5":
        grid_search(checkpoint=ckpt, critic_ckpt=cckpt, fixed_lr=5e-5)
    elif args.mode == "grid-lr1e-5":
        grid_search(checkpoint=ckpt, critic_ckpt=cckpt, fixed_lr=1e-5)
    elif args.mode == "grid-merge":
        merge_summaries()
