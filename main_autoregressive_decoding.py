"""Autoregressive RL for the classic (vertex-guard) Art Gallery Problem.

Selects guards one at a time until the model decides to stop. Each step
masks already-selected vertices and provides incremental coverage feedback.
This naturally produces small guard sets because the model must actively
choose to continue adding guards vs stopping.

Usage:
    python main_autoregressive_decoding.py single              # single run
    python main_autoregressive_decoding.py grid-hard           # grid search with hard reward
    python main_autoregressive_decoding.py grid-double-hard    # double model size
    python main_autoregressive_decoding.py grid-exp
    python main_autoregressive_decoding.py grid-double-exp
    python main_autoregressive_decoding.py grid-optimal-ratio
    python main_autoregressive_decoding.py grid-double-optimal-ratio
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models import Critic
from dataset import load_classic_datasets, agp_collate_fn
from losses import RewardHardCoverage, RewardOptimalRatio, RewardExpGuardPenalty
from main_classic_gt import compute_guard_coverage
from concurrent.futures import ProcessPoolExecutor


# ── config ────────────────────────────────────────────────────────────────────

CONFIG_DIR = Path("data/config")

def _cfg_path(suffix: str) -> Path:
    return CONFIG_DIR / f"config_autoreg_{suffix}.json"

def load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)

def _ensure_config(suffix: str, overrides: dict | None = None) -> Path:
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
        "num_epochs": 50,
        "entropy_weight": 0.01,
        "max_grad_norm": 2.0,
        "coverage_weight": 0.6,
        "model_dir": f"checkpoints/autoreg_{suffix}",
    }
    if overrides:
        base.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(base, f, indent=4)
    return path


# ── Autoregressive Pointer Network ──────────────────────────────────────────

class AutoregressivePointerNet(nn.Module):
    """Pointer network that selects guards autoregressively with a stop action.

    Key difference from the base PointerNetwork rl_forward:
    - Uses incremental coverage as step-level reward signal
    - The stop token is always available (never masked)
    - Encoder runs once, decoder steps select one guard at a time
    """

    def __init__(self, input_size, hidden_size, attention_dim, bidirectional=False):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            bidirectional=bidirectional)

        enc_dim = hidden_size * (2 if bidirectional else 1)
        self.decoder_cell = nn.LSTMCell(enc_dim, enc_dim)

        # Attention
        self.W_1 = nn.Linear(enc_dim, attention_dim)
        self.W_2 = nn.Linear(enc_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)
        self.tanh = nn.Tanh()

        # Stop token (prepended at position 0)
        self.stop_token = nn.Parameter(torch.randn(1, 1, input_size))
        # Project encoder output → decoder input
        self.enc_to_dec = nn.Linear(enc_dim, enc_dim)

        self.hidden_size = hidden_size
        self.enc_dim = enc_dim
        self.bidirectional = bidirectional

    def _prepare_h_c(self, h, c):
        num_layers = self.encoder.num_layers
        num_dirs = 2 if self.bidirectional else 1
        bs = h.size(1)
        hs = h.size(2)
        h = h.view(num_layers, num_dirs, bs, hs)[-1].permute(1, 0, 2).reshape(bs, -1)
        c = c.view(num_layers, num_dirs, bs, hs)[-1].permute(1, 0, 2).reshape(bs, -1)
        return h, c

    def forward(self, x, seq_lens, max_steps=None):
        """Autoregressive decoding with sampling.

        Args:
            x:        (max_seq_len, batch, 2)
            seq_lens: (batch,)
            max_steps: max decode steps (default: max seq_len)

        Returns:
            actions:     (batch, n_steps) — vertex indices, -1 = stop/pad
            log_probs:   (batch, n_steps)
            entropies:   (batch, n_steps)
            step_counts: (batch,) — how many guards each sample selected
        """
        device = x.device
        batch_size = x.size(1)
        max_steps = max_steps or seq_lens.max().item()

        # Prepend stop token
        stop = self.stop_token.expand(1, batch_size, -1)
        x_aug = torch.cat([stop, x], dim=0)
        seq_lens_aug = seq_lens + 1

        # Encode
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x_aug, seq_lens_aug.cpu(), enforce_sorted=False)
        out_enc_packed, (h, c) = self.encoder(x_packed)
        out_enc, _ = nn.utils.rnn.pad_packed_sequence(out_enc_packed)
        h, c = self._prepare_h_c(h, c)
        W_1_out = self.W_1(out_enc)

        enc_len = out_enc.size(0)
        batch_idx = torch.arange(batch_size, device=device)

        # Padding mask
        pad_mask = (torch.arange(enc_len, device=device).unsqueeze(1)
                    >= seq_lens_aug.unsqueeze(0))
        sel_mask = pad_mask.clone()

        inp = torch.zeros(batch_size, self.enc_dim, device=device)
        active = torch.ones(batch_size, dtype=torch.bool, device=device)

        all_actions, all_log_probs, all_entropies = [], [], []

        for _ in range(max_steps):
            h, c = self.decoder_cell(inp, (h, c))

            # Attention scores
            W_2_out = self.W_2(h)
            scores = self.v(self.tanh(W_1_out + W_2_out)).squeeze(2)  # (enc_len, B)

            # Mask selected + padding, but keep stop (pos 0) open
            mask_step = sel_mask.clone()
            mask_step[0, :] = False
            scores = scores.masked_fill(mask_step, float('-inf'))

            probs = F.softmax(scores, dim=0).T  # (B, enc_len)
            probs = probs.clamp(min=1e-8)
            probs = probs / probs.sum(dim=1, keepdim=True)

            dist = torch.distributions.Categorical(probs)
            idx = dist.sample()

            all_actions.append(idx - 1)  # 0 → -1 (stop)
            all_log_probs.append(dist.log_prob(idx) * active.float())
            all_entropies.append(dist.entropy() * active.float())

            active = active & (idx != 0)
            if not active.any():
                break

            # Mask selected positions
            non_stop = idx > 0
            if non_stop.any():
                sel_mask = sel_mask.clone()
                sel_mask[idx[non_stop], batch_idx[non_stop]] = True

            selected_enc = out_enc[idx, batch_idx]
            inp = self.enc_to_dec(selected_enc)
            inp = inp * active.float().unsqueeze(1)

        actions = torch.stack(all_actions, dim=1)
        log_probs = torch.stack(all_log_probs, dim=1)
        entropies = torch.stack(all_entropies, dim=1)

        # Count guards per sample
        step_counts = (actions >= 0).sum(dim=1)

        return actions, log_probs, entropies, step_counts


# ── coverage evaluation ──────────────────────────────────────────────────────

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


# ── training ──────────────────────────────────────────────────────────────────

def train(cfg: dict, reward_type: str = "hard",
          run_name: str | None = None,
          log_subdir: str = "autoreg_hard") -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── data ──
    full_ds = load_classic_datasets(cfg["data_dir"], cfg["chunk_size"])
    max_samples = cfg.get("max_samples")
    if max_samples and len(full_ds) > max_samples:
        full_ds, _ = random_split(full_ds, [max_samples, len(full_ds) - max_samples])
    n_val = int(len(full_ds) * cfg["val_fraction"])
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - n_val, n_val])
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    loader_kw = dict(batch_size=cfg["batch_size"], collate_fn=agp_collate_fn,
                     num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    # ── model + critic ──
    hs = cfg["hidden_size"]
    bi = cfg["bidirectional"]

    model = AutoregressivePointerNet(
        input_size=2, hidden_size=hs,
        attention_dim=cfg["attention_dim"], bidirectional=bi,
    ).to(device)

    critic = Critic(input_size=2, hidden_size=hs, bidirectional=bi).to(device)

    # Reward
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

    optimizer = optim.Adam(
        list(model.parameters()) + list(critic.parameters()), lr=cfg["lr"])

    model_dir = Path(cfg["model_dir"])
    if run_name:
        model_dir = model_dir / run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(f"data/logs/{log_subdir}")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model params:  {sum(p.numel() for p in model.parameters()):,}")
    print(f"Critic params: {sum(p.numel() for p in critic.parameters()):,}")

    # ── train loop ──
    best_coverage = 0.0
    best_guards = 1.0
    best_ratio_to_opt = float("inf")
    entropy_w = cfg["entropy_weight"]
    max_grad = cfg["max_grad_norm"]
    epoch_logs: list[dict] = []

    for epoch in range(cfg["num_epochs"]):
        model.train()
        critic.train()
        epoch_cov, epoch_guards, epoch_loss, n_batches = 0.0, 0.0, 0.0, 0

        label = run_name or "train"
        pbar = tqdm(train_loader,
                    desc=f"[{label}] Epoch {epoch+1}/{cfg['num_epochs']}")
        for vertices, seq_lens, gt_guards, gt_n_guards in pbar:
            vertices, seq_lens = vertices.to(device), seq_lens.to(device)

            actions, log_probs, entropies, step_counts = model(vertices, seq_lens)

            coverages, rel_lengths = compute_batch_coverage(
                vertices.cpu(), seq_lens.cpu(), actions.cpu())

            if reward_type == "optimal_ratio":
                rewards = reward_fn(coverages, rel_lengths,
                                    gt_n_guards.float(),
                                    step_counts.cpu().float()).to(device)
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
                steps=f"{step_counts.float().mean().item():.1f}",
            )

        avg_cov = epoch_cov / n_batches
        avg_guards = epoch_guards / n_batches
        avg_loss = epoch_loss / n_batches

        val_cov, val_guards, val_ratio = evaluate(model, val_loader, device)

        row = dict(epoch=epoch + 1, train_cov=avg_cov, train_guards=avg_guards,
                   train_loss=avg_loss, val_cov=val_cov, val_guards=val_guards,
                   val_ratio_to_opt=val_ratio)
        epoch_logs.append(row)

        if val_cov > best_coverage:
            best_coverage = val_cov
            best_guards = val_guards
            best_ratio_to_opt = val_ratio
            torch.save(model.state_dict(), model_dir / "best_model.pt")
            torch.save(critic.state_dict(), model_dir / "best_critic.pt")
            tqdm.write(f"  -> Best model @ epoch {epoch+1}, "
                       f"coverage {val_cov:.4f}, guards {val_guards:.4f}, "
                       f"ratio_to_opt {val_ratio:.4f}")

    # ── save CSV ──
    csv_name = f"{run_name}.csv" if run_name else "single_run.csv"
    csv_path = log_dir / csv_name
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_logs[0].keys())
        writer.writeheader()
        writer.writerows(epoch_logs)

    print(f"Done. Best val coverage: {best_coverage:.4f}, guards: {best_guards:.4f}, "
          f"ratio_to_opt: {best_ratio_to_opt:.4f}")
    return dict(best_coverage=best_coverage, best_guards=best_guards,
                best_ratio_to_opt=best_ratio_to_opt, epoch_logs=epoch_logs)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_cov, total_guards, total_ratio, total_n = 0.0, 0.0, 0.0, 0
    for vertices, seq_lens, _, gt_n_guards in loader:
        vertices, seq_lens = vertices.to(device), seq_lens.to(device)
        actions, _, _, _ = model(vertices, seq_lens)
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
    suffix = f"{'double_' if double else ''}{reward_type}"
    cfg_path = _ensure_config(suffix, overrides={
        "reward_type": reward_type,
        "hidden_size": 256 if double else 128,
        "attention_dim": 256 if double else 128,
        "model_dir": f"checkpoints/autoreg_{suffix}",
    })
    base_cfg = load_config(cfg_path)

    log_subdir = f"autoreg_{'double_' if double else ''}{reward_type}"
    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))

    print(f"Grid search: {len(combos)} runs  (config: {cfg_path})")

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

        result = train(cfg, reward_type=reward_type,
                       run_name=run_name, log_subdir=log_subdir)
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
    print(f"Autoreg grid results — {reward_type} (sorted by coverage):")
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


# ── entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoregressive RL for AGP")
    parser.add_argument("mode", nargs="?", default="single",
                        choices=["single", "grid-hard", "grid-double-hard",
                                 "grid-exp", "grid-double-exp",
                                 "grid-optimal-ratio", "grid-double-optimal-ratio"])
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index to use (default: 0)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.mode == "single":
        path = _ensure_config("hard")
        train(load_config(path), reward_type="hard")
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
