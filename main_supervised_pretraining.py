"""Supervised pretraining for the classic (vertex-guard) Art Gallery Problem.

Trains the PointerNetwork to imitate ILP-optimal guard selections via
cross-entropy on the autoregressive guard sequence, then fine-tunes with
REINFORCE using hard/exp/optimal_ratio rewards.

Usage:
    python main_supervised_pretraining.py pretrain             # supervised phase only
    python main_supervised_pretraining.py finetune-hard        # RL fine-tune with hard reward
    python main_supervised_pretraining.py grid-hard            # grid search: pretrain + RL hard
    python main_supervised_pretraining.py grid-double-hard     # double model size
    python main_supervised_pretraining.py grid-exp             # grid search: pretrain + RL exp
    python main_supervised_pretraining.py grid-double-exp
    python main_supervised_pretraining.py grid-optimal-ratio
    python main_supervised_pretraining.py grid-double-optimal-ratio
"""

import argparse
import csv
import itertools
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models import PointerNetwork, Critic
from dataset import load_classic_datasets, agp_collate_fn
from losses import RewardHardCoverage, RewardOptimalRatio, RewardExpGuardPenalty
from main_classic_gt import compute_guard_coverage


# ── config paths ─────────────────────────────────────────────────────────────

CONFIG_DIR = Path("data/config")

def _cfg_path(suffix: str) -> Path:
    return CONFIG_DIR / f"config_supervised_{suffix}.json"

def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

def _ensure_config(suffix: str, overrides: dict | None = None) -> Path:
    """Create config file if it doesn't exist, based on the RL classic config."""
    path = _cfg_path(suffix)
    if path.exists():
        return path
    base = {
        "data_dir": "data",
        "chunk_size": 100,
        "max_samples": 10000,
        "val_fraction": 0.1,
        "hidden_size": 128,
        "attention_dim": 128,
        "bidirectional": False,
        "batch_size": 1024,
        "lr": 1e-4,
        "num_epochs_pretrain": 30,
        "num_epochs_finetune": 30,
        "entropy_weight": 0.01,
        "max_grad_norm": 2.0,
        "coverage_weight": 0.6,
        "model_dir": f"checkpoints/supervised_{suffix}",
    }
    if overrides:
        base.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(base, f, indent=4)
    return path


# ── coverage evaluation (reused from main_rl_classic) ────────────────────────

from concurrent.futures import ProcessPoolExecutor

def _eval_one(args):
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


# ── supervised pretraining ────────────────────────────────────────────────────

def pretrain(cfg: dict, run_name: str | None = None,
             log_subdir: str = "supervised_pretrain") -> tuple[dict, Path]:
    """Supervised phase: train model to imitate ILP-optimal guard sequences.

    Returns (result_dict, best_model_path).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── data ──
    full_ds = load_classic_datasets(cfg["data_dir"], cfg["chunk_size"])
    max_samples = cfg.get("max_samples")
    if max_samples and len(full_ds) > max_samples:
        full_ds, _ = random_split(full_ds, [max_samples, len(full_ds) - max_samples])
    n_val = int(len(full_ds) * cfg["val_fraction"])
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val])
    print(f"Pretrain — Train: {len(train_ds)}, Val: {len(val_ds)}")

    loader_kw = dict(batch_size=cfg["batch_size"], collate_fn=agp_collate_fn,
                     num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    # ── model ──
    hs = cfg["hidden_size"]
    bi = cfg["bidirectional"]
    dec_dim = hs * (2 if bi else 1)

    model = PointerNetwork(
        input_size=2, hidden_size=hs, output_size=dec_dim,
        n_categories=dec_dim, attention_dim=cfg["attention_dim"],
        bidirectional=bi,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    model_dir = Path(cfg["model_dir"])
    if run_name:
        model_dir = model_dir / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(f"data/logs/{log_subdir}")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # ── train loop ──
    best_val_loss = float("inf")
    n_epochs = cfg.get("num_epochs_pretrain", 30)
    epoch_logs: list[dict] = []

    for epoch in range(n_epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0

        label = run_name or "pretrain"
        pbar = tqdm(train_loader,
                    desc=f"[{label}] Pretrain {epoch+1}/{n_epochs}")
        for vertices, seq_lens, gt_guards, gt_n_guards in pbar:
            vertices = vertices.to(device)
            seq_lens = seq_lens.to(device)
            gt_guards = gt_guards.to(device)
            gt_n_guards = gt_n_guards.to(device)

            # Build teacher-forced target sequence: guard indices + stop token.
            # For each room, the target is the guard indices followed by -1 (stop).
            # We need to feed the model and compute cross-entropy at each step.
            loss = _supervised_loss(model, vertices, seq_lens, gt_guards, gt_n_guards)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / n_batches

        # ── validation ──
        val_loss = _eval_pretrain(model, val_loader, device)
        row = dict(epoch=epoch + 1, train_loss=avg_loss, val_loss=val_loss)
        epoch_logs.append(row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = model_dir / "pretrained_model.pt"
            torch.save(model.state_dict(), best_path)
            tqdm.write(f"  -> Best pretrained model @ epoch {epoch+1}, "
                       f"val_loss {val_loss:.4f}")

    # ── save CSV ──
    csv_name = f"{run_name}_pretrain.csv" if run_name else "pretrain.csv"
    csv_path = log_dir / csv_name
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_logs[0].keys())
        writer.writeheader()
        writer.writerows(epoch_logs)

    print(f"Pretrain done. Best val loss: {best_val_loss:.4f}")
    return dict(best_val_loss=best_val_loss, epoch_logs=epoch_logs), best_path


def _supervised_loss(model, vertices, seq_lens, gt_guards, gt_n_guards):
    """Compute cross-entropy loss for imitating the ILP-optimal guard sequence.

    Uses rl_forward-style encoding with stop token, then teacher-forces the
    guard sequence and computes NLL at each step.
    """
    device = vertices.device
    batch_size = vertices.size(1)

    # Encode with stop token (position 0)
    stop = model.stop_token.expand(1, batch_size, -1)
    x_aug = torch.cat([stop, vertices], dim=0)
    seq_lens_aug = seq_lens + 1

    x_packed = nn.utils.rnn.pack_padded_sequence(
        x_aug, seq_lens_aug.cpu(), enforce_sorted=False)
    out_enc_packed, (h, c) = model.encoder(x_packed)
    out_enc, _ = nn.utils.rnn.pad_packed_sequence(
        out_enc_packed, batch_first=False, padding_value=model.padding_value)
    h, c = model._prepare_h_and_c_for_decoder(h, c)
    W_1_out = model.W_1(out_enc)

    enc_len = out_enc.size(0)
    batch_idx = torch.arange(batch_size, device=device)

    # Padding mask
    pad_mask = (torch.arange(enc_len, device=device).unsqueeze(1)
                >= seq_lens_aug.unsqueeze(0))

    inp = torch.zeros(batch_size, model.output_size, device=device)

    max_guards = gt_guards.size(1)
    total_loss = torch.zeros(1, device=device)
    n_steps = 0

    # Selection mask to avoid re-selecting
    sel_mask = pad_mask.clone()

    for step in range(max_guards + 1):  # +1 for stop
        h, c = model.decoder_cell(inp, (h, c))
        scores = model.attention(W_1_out, out_enc, h, seq_lens_aug)

        # Mask already-selected, keep stop (pos 0) open
        mask_step = sel_mask.clone()
        mask_step[0, :] = False
        scores = scores.masked_fill(mask_step, float('-inf'))

        log_probs = torch.log_softmax(scores.T, dim=1)  # (B, enc_len)

        # Target: for steps < n_guards, target is guard_idx + 1 (shifted by stop token)
        # For step == n_guards, target is 0 (stop)
        target = torch.zeros(batch_size, dtype=torch.long, device=device)
        active = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for i in range(batch_size):
            ng = gt_n_guards[i].item()
            if step < ng:
                target[i] = gt_guards[i, step].item() + 1  # +1 for stop token offset
                active[i] = True
            elif step == ng:
                target[i] = 0  # stop
                active[i] = True
            # else: already stopped, skip

        if not active.any():
            break

        step_loss = nn.functional.nll_loss(
            log_probs[active], target[active], reduction="sum")
        total_loss = total_loss + step_loss
        n_steps += active.sum().item()

        # Feed selected encoder state for next step
        # Use target (teacher forcing)
        selected_enc = out_enc[target, batch_idx]
        inp = model.enc_to_dec(selected_enc)
        inp = inp * active.float().unsqueeze(1)

        # Update selection mask
        non_stop = target > 0
        if non_stop.any():
            sel_mask = sel_mask.clone()
            sel_mask[target[non_stop & active], batch_idx[non_stop & active]] = True

    return total_loss / max(n_steps, 1)


@torch.no_grad()
def _eval_pretrain(model, loader, device):
    model.eval()
    total_loss, total_steps = 0.0, 0
    for vertices, seq_lens, gt_guards, gt_n_guards in loader:
        vertices = vertices.to(device)
        seq_lens = seq_lens.to(device)
        gt_guards = gt_guards.to(device)
        gt_n_guards = gt_n_guards.to(device)
        loss = _supervised_loss(model, vertices, seq_lens, gt_guards, gt_n_guards)
        total_loss += loss.item()
        total_steps += 1
    avg = total_loss / total_steps if total_steps > 0 else float("inf")
    tqdm.write(f"  Val pretrain loss: {avg:.4f}")
    return avg


# ── RL fine-tuning ────────────────────────────────────────────────────────────

def finetune(cfg: dict, pretrained_path: Path, reward_type: str = "hard",
             run_name: str | None = None,
             log_subdir: str = "supervised_finetune_hard") -> dict:
    """RL fine-tuning from a pretrained checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── data ──
    full_ds = load_classic_datasets(cfg["data_dir"], cfg["chunk_size"])
    max_samples = cfg.get("max_samples")
    if max_samples and len(full_ds) > max_samples:
        full_ds, _ = random_split(full_ds, [max_samples, len(full_ds) - max_samples])
    n_val = int(len(full_ds) * cfg["val_fraction"])
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val])
    print(f"Finetune — Train: {len(train_ds)}, Val: {len(val_ds)}")

    loader_kw = dict(batch_size=cfg["batch_size"], collate_fn=agp_collate_fn,
                     num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    # ── model + critic ──
    hs = cfg["hidden_size"]
    bi = cfg["bidirectional"]
    dec_dim = hs * (2 if bi else 1)

    model = PointerNetwork(
        input_size=2, hidden_size=hs, output_size=dec_dim,
        n_categories=dec_dim, attention_dim=cfg["attention_dim"],
        bidirectional=bi,
    ).to(device)

    # Load pretrained weights
    model.load_state_dict(torch.load(pretrained_path, map_location=device, weights_only=True))
    print(f"Loaded pretrained weights from {pretrained_path}")

    critic = Critic(input_size=2, hidden_size=hs, bidirectional=bi).to(device)

    # Reward function
    if reward_type == "hard":
        reward_fn = RewardHardCoverage(cfg.get("coverage_threshold", 0.999))
    elif reward_type == "exp":
        reward_fn = RewardExpGuardPenalty(cfg.get("exp_scale", 3.0),
                                          cfg.get("coverage_weight", 2.0))
    elif reward_type == "optimal_ratio":
        reward_fn = RewardOptimalRatio(cfg.get("coverage_threshold", 0.999),
                                       cfg.get("cov_penalty", 5.0))
    else:
        raise ValueError(f"Unknown reward_type: {reward_type}")

    # Use smaller LR for fine-tuning
    ft_lr = cfg.get("lr_finetune", cfg["lr"] * 0.1)
    optimizer = optim.Adam(
        list(model.parameters()) + list(critic.parameters()), lr=ft_lr)

    model_dir = Path(cfg["model_dir"])
    if run_name:
        model_dir = model_dir / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(f"data/logs/{log_subdir}")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Fine-tune LR: {ft_lr}")

    # ── train loop ──
    best_coverage = 0.0
    best_guards = 1.0
    best_ratio_to_opt = float("inf")
    entropy_w = cfg["entropy_weight"]
    max_grad = cfg["max_grad_norm"]
    n_epochs = cfg.get("num_epochs_finetune", 30)
    epoch_logs: list[dict] = []

    for epoch in range(n_epochs):
        model.train()
        critic.train()
        epoch_cov, epoch_guards, epoch_loss, n_batches = 0.0, 0.0, 0.0, 0

        label = run_name or "finetune"
        pbar = tqdm(train_loader,
                    desc=f"[{label}] FT {epoch+1}/{n_epochs}")
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
                rewards = reward_fn(coverages, rel_lengths,
                                    gt_n_guards.float(), n_pred).to(device)
            else:
                rewards = reward_fn(coverages, rel_lengths).to(device)

            values = critic(vertices, seq_lens)
            advantage = (rewards - values).detach()

            valid = (actions >= 0).float().to(device)
            sample_lp = (log_probs * valid).sum(dim=1)
            sample_ent = (entropies * valid).sum(dim=1)

            reinforce_loss = -(advantage * sample_lp).mean()
            critic_loss = ((rewards - values) ** 2).mean()
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

        avg_cov = epoch_cov / n_batches
        avg_guards = epoch_guards / n_batches
        avg_loss = epoch_loss / n_batches

        val_cov, val_guards, val_ratio = _eval_rl(model, val_loader, device)

        row = dict(epoch=epoch + 1, train_cov=avg_cov, train_guards=avg_guards,
                   train_loss=avg_loss, val_cov=val_cov, val_guards=val_guards,
                   val_ratio_to_opt=val_ratio)
        epoch_logs.append(row)

        if val_cov > best_coverage:
            best_coverage = val_cov
            best_guards = val_guards
            best_ratio_to_opt = val_ratio
            torch.save(model.state_dict(), model_dir / "best_finetuned_model.pt")
            torch.save(critic.state_dict(), model_dir / "best_finetuned_critic.pt")
            tqdm.write(f"  -> Best model @ epoch {epoch+1}, "
                       f"coverage {val_cov:.4f}, guards {val_guards:.4f}, "
                       f"ratio_to_opt {val_ratio:.4f}")

    # ── save CSV ──
    csv_name = f"{run_name}_finetune.csv" if run_name else "finetune.csv"
    csv_path = log_dir / csv_name
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_logs[0].keys())
        writer.writeheader()
        writer.writerows(epoch_logs)

    print(f"Finetune done. Best val coverage: {best_coverage:.4f}, guards: {best_guards:.4f}, "
          f"ratio_to_opt: {best_ratio_to_opt:.4f}")
    return dict(best_coverage=best_coverage, best_guards=best_guards,
                best_ratio_to_opt=best_ratio_to_opt, epoch_logs=epoch_logs)


@torch.no_grad()
def _eval_rl(model, loader, device):
    model.eval()
    total_cov, total_guards, total_ratio, total_n = 0.0, 0.0, 0.0, 0
    for vertices, seq_lens, _, gt_n_guards in loader:
        vertices, seq_lens = vertices.to(device), seq_lens.to(device)
        actions, _, _ = model.rl_forward(vertices, seq_lens)
        coverages, rel_lengths = compute_batch_coverage(
            vertices.cpu(), seq_lens.cpu(), actions.cpu())
        n_pred = rel_lengths * seq_lens.cpu().float()
        bs = vertices.size(1)
        total_cov += coverages.sum().item()
        total_guards += rel_lengths.sum().item()
        total_ratio += (n_pred / gt_n_guards.float()).sum().item()
        total_n += bs
    avg_cov = total_cov / total_n
    avg_guards = total_guards / total_n
    avg_ratio = total_ratio / total_n
    tqdm.write(f"  Val coverage: {avg_cov:.4f}, guards: {avg_guards:.4f}, ratio_to_opt: {avg_ratio:.4f}")
    return avg_cov, avg_guards, avg_ratio


# ── grid search ──────────────────────────────────────────────────────────────

GRID = {
    "lr":             [1e-3, 1e-4],
    "entropy_weight": [0.01, 0.001],
    "coverage_weight": [0.25, 0.5, 0.7],
}


def grid_search(reward_type: str, double: bool = False):
    """Run pretrain once, then grid-search RL fine-tuning hyperparams."""
    suffix = f"{'double_' if double else ''}{reward_type}"
    cfg_path = _ensure_config(suffix, overrides={
        "reward_type": reward_type,
        "hidden_size": 256 if double else 128,
        "attention_dim": 256 if double else 128,
        "model_dir": f"checkpoints/supervised_{suffix}",
    })
    base_cfg = load_config(cfg_path)

    log_sub_pretrain = f"supervised_{'double_' if double else ''}pretrain"
    log_sub_finetune = f"supervised_{'double_' if double else ''}finetune_{reward_type}"

    # Phase 1: Supervised pretraining (once)
    print(f"\n{'='*60}")
    print(f"Phase 1: Supervised Pretraining ({'double' if double else 'standard'})")
    print(f"{'='*60}")
    _, pretrained_path = pretrain(base_cfg, log_subdir=log_sub_pretrain)

    # Phase 2: Grid search over RL fine-tuning
    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))

    print(f"\n{'='*60}")
    print(f"Phase 2: RL Fine-tuning Grid Search ({reward_type})")
    print(f"  {len(combos)} configurations")
    print(f"{'='*60}")

    summary_path = Path(f"data/logs/{log_sub_finetune}/grid_summary.csv")
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

        result = finetune(cfg, pretrained_path, reward_type=reward_type,
                          run_name=run_name, log_subdir=log_sub_finetune)
        row = {k: v for k, v in zip(keys, combo)}
        row["best_coverage"] = result["best_coverage"]
        row["best_guards"] = result["best_guards"]
        row["best_ratio_to_opt"] = result["best_ratio_to_opt"]
        results.append(row)

        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Print ranking
    print(f"\n{'='*60}")
    print(f"Grid search results — supervised + {reward_type} (sorted by coverage):")
    print(f"{'='*60}")
    results.sort(key=lambda r: r["best_coverage"], reverse=True)
    for i, r in enumerate(results, 1):
        params = ", ".join(f"{k}={r[k]}" for k in keys)
        print(f"  {i:2d}. cov={r['best_coverage']:.4f}  "
              f"guards={r['best_guards']:.4f}  ratio_to_opt={r['best_ratio_to_opt']:.4f}  |  {params}")

    best = results[0]
    for k in keys:
        base_cfg[k] = best[k]
    with open(cfg_path, "w") as f:
        json.dump(base_cfg, f, indent=4)
    print(f"\nBest config written to {cfg_path}")
    print(f"Summary -> {summary_path}")


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised pretraining + RL fine-tune for AGP")
    parser.add_argument("mode", nargs="?", default="pretrain",
                        choices=["pretrain", "grid-hard", "grid-double-hard",
                                 "grid-exp", "grid-double-exp",
                                 "grid-optimal-ratio", "grid-double-optimal-ratio"])
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use (default: 0)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.mode == "pretrain":
        path = _ensure_config("hard")
        pretrain(load_config(path))
    elif args.mode == "grid-hard":
        grid_search("hard", double=False)
    elif args.mode == "grid-double-hard":
        grid_search("hard", double=True)
    elif args.mode == "grid-exp":
        grid_search("exp", double=False)
    elif args.mode == "grid-double-exp":
        grid_search("exp", double=True)
    elif args.mode == "grid-optimal-ratio":
        grid_search("optimal_ratio", double=False)
    elif args.mode == "grid-double-optimal-ratio":
        grid_search("optimal_ratio", double=True)
