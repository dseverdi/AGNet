#!/usr/bin/env python
"""profile_po_step.py — where does a PO/BT training step actually spend time?

Training shows ~17% GPU utilisation, i.e. the GPU is mostly idle and something
on the CPU/host side is the bottleneck. This times the three phases of one PO
step in isolation, with torch.cuda.synchronize() so GPU-async work is charged
correctly:

  1. decode   -- model() autoregressive rollout of B*K sequences (GPU, but a
                 Python per-step loop with host<->device syncs each step)
  2. reward   -- the B*K-iteration Python loop calling reward_fn (disc-vis,
                 pure CPU numpy over the cached matrix)
  3. backward -- Bradley-Terry loss + loss.backward() + optimizer step (GPU)

Prints per-phase mean ms and % of step, so the fix is obvious: if `decode`
dominates it is the sequential LSTM loop (needs vectorised/batched decode or a
faster per-step path); if `reward` dominates it is the serial CPU reward loop
(needs batching/parallelising reward_fn); if `backward` dominates the GPU is
the limit (a faster card helps).

USAGE
  python tools/profile_po_step.py                 # batch 10, K 8, 12 steps
  python tools/profile_po_step.py --batch 32 --steps 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dataset import Dataset, collate_fn                               # noqa: E402
from po_agp import (                                                  # noqa: E402
    create_agp_model, prepare_datasets, get_lengths_from_dataset,
    BucketBatchSampler,
)
from po_agp import po_reward_smooth_disc                              # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=10)
    p.add_argument("--rollouts", type=int, default=8)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--disc-vis-samples", type=int, default=500)
    args = p.parse_args()

    dp = os.getenv("DATASET_PATH")
    if not dp:
        sys.exit("DATASET_PATH must be set (see .env)")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr, _ = prepare_datasets(os.path.join(dp, "train"),
                             os.path.join(dp, "dev"), normalize=True)
    ds = Dataset(tr.samples[: max(2000, args.batch * (args.steps + args.warmup) * 2)])
    lengths = get_lengths_from_dataset(ds)
    sampler = BucketBatchSampler(lengths, args.batch, shuffle=True,
                                 bucket_size=args.batch)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler,
                                         collate_fn=collate_fn, num_workers=0)

    model = create_agp_model(128, 128, 1, 10.0, True, 1.0).to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    K = args.rollouts

    def reward_fn(points, solution, name, length=None):
        return po_reward_smooth_disc(points, solution, name, length=length,
                                     lam=1.0, tau=0.99, tau_penalty=3.0,
                                     cap_at_tau=True,
                                     n_samples=args.disc_vis_samples)

    t = {"decode": [], "reward": [], "backward": [], "step": []}
    n_done = 0
    for batch_data, pad_mask, lens, names in loader:
        bd = batch_data.to(dev); pm = pad_mask.to(dev)
        lt = torch.as_tensor(lens, device=dev)
        B = bd.shape[0]
        step0 = time.perf_counter()

        # -- 1. decode: B*K autoregressive rollouts --
        sync(); t0 = time.perf_counter()
        exp = bd.repeat_interleave(K, 0)
        em = pm.repeat_interleave(K, 0)
        el = lt.repeat_interleave(K)
        all_idxs, all_lp = model(exp, padding_mask=em, lengths=el,
                                 deterministic=False)
        sync(); t_dec = time.perf_counter() - t0

        # -- 2. reward: B*K serial CPU reward_fn calls --
        t0 = time.perf_counter()
        pts_cache = [bd[b, :int(lt[b].item())].detach().cpu().numpy()
                     for b in range(B)]
        rflat = []
        for i in range(B * K):
            b = i // K
            n = int(lt[b].item())
            sol = [int(idx) for idx in all_idxs[i] if int(idx) < n]
            rflat.append(float(reward_fn(pts_cache[b], sol, names[b], length=n)))
        rewards = torch.tensor(rflat, device=dev).view(B, K)
        t_rew = time.perf_counter() - t0

        # -- 3. backward: BT loss + step --
        sync(); t0 = time.perf_counter()
        if isinstance(all_lp, (list, tuple)):
            lp = torch.stack([l.sum() for l in all_lp]).view(B, K)
        else:
            lp = all_lp.view(B, K)
        pref = (rewards[:, :, None] > rewards[:, None, :]).float()
        diff = lp[:, :, None] - lp[:, None, :]
        loss = -(torch.nn.functional.logsigmoid(0.05 * diff) * pref).sum() / max(1, pref.sum())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sync(); t_bwd = time.perf_counter() - t0

        t_step = time.perf_counter() - step0
        if n_done >= args.warmup:
            t["decode"].append(t_dec); t["reward"].append(t_rew)
            t["backward"].append(t_bwd); t["step"].append(t_step)
        n_done += 1
        if n_done >= args.warmup + args.steps:
            break

    step = float(np.mean(t["step"]))
    util = None
    print("\n" + "=" * 60)
    print(f"  PO STEP PROFILE  (batch={args.batch}, K={K}, {args.steps} steps, "
          f"B*K={args.batch*K} seqs/step, device={dev})")
    print("=" * 60)
    for ph in ("decode", "reward", "backward"):
        ms = float(np.mean(t[ph])) * 1000
        print(f"  {ph:10} {ms:8.1f} ms   {ms/(step*1000)*100:5.1f}%")
    print(f"  {'STEP':10} {step*1000:8.1f} ms   (per optimizer step)")
    print("=" * 60)
    dom = max(("decode", "reward", "backward"), key=lambda k: np.mean(t[k]))
    print(f"  BOTTLENECK: {dom}  ({np.mean(t[dom])/step*100:.0f}% of the step)")
    if dom == "reward":
        print("  -> serial CPU reward loop. Batch/vectorise reward_fn or run it")
        print("     across CPU workers; a faster GPU will NOT help.")
    elif dom == "decode":
        print("  -> sequential autoregressive decode (host<->device sync per")
        print("     step). A faster GPU helps only a little; batching decode or")
        print("     cutting per-step syncs is the real lever.")
    else:
        print("  -> GPU compute-bound; a faster GPU helps proportionally.")
    print(f"\n  extrapolated: {step:.2f} s/step x {8000//args.batch} steps "
          f"= {step*(8000//args.batch)/60:.1f} min/epoch")


if __name__ == "__main__":
    main()
