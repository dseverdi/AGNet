#!/usr/bin/env python
"""smoke_fast_decode.py — prove the vectorised decode changes match the baseline.

Two gates:
  (A) unit: vectorised apply_mask_to_logits vs _apply_mask_to_logits_legacy on
      random inputs, including the degenerate all-masked / no-lengths cases.
  (B) end-to-end: the frozen released policy decodes N real polygons under the
      fast path (self._legacy_mask=False) and the legacy path
      (self._legacy_mask=True), with the SAME RNG seed each time; assert the
      greedy index sequences are identical and the stochastic log-probs match.

Exit non-zero (and print MISMATCH) if anything diverges, so it can gate CI /
a launch. Also prints a quick decode-time ratio as a sanity check on the win.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dataset import Dataset, collate_fn                               # noqa: E402
from po_agp import create_agp_model, prepare_datasets                 # noqa: E402

CKPT = os.path.join(REPO, "checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt")


def _eq_logits(a, b):
    """Equal treating -inf==-inf and NaN==NaN."""
    both_ninf = torch.isneginf(a) & torch.isneginf(b)
    both_nan = torch.isnan(a) & torch.isnan(b)
    finite = (~torch.isneginf(a) & ~torch.isneginf(b) & ~torch.isnan(a) & ~torch.isnan(b))
    return bool((both_ninf | both_nan | (finite & torch.isclose(a, b, atol=1e-6, rtol=1e-5)
                                         )).all())


def _actor(model):
    """The PointerNet that owns apply_mask_to_logits (model is the wrapper)."""
    return model.actor if hasattr(model, "actor") else model


def unit_test(model, device):
    print("== (A) apply_mask_to_logits: vectorised vs legacy ==")
    net = _actor(model)
    torch.manual_seed(0)
    ok = True
    cases = 0
    for trial in range(200):
        B = int(torch.randint(1, 12, (1,)))
        cols = int(torch.randint(3, 40, (1,)))
        logits = torch.randn(B, cols, device=device)
        mask = torch.rand(B, cols, device=device) < 0.5
        idxs = torch.randint(0, cols, (B,), device=device) if trial % 3 else None
        if trial % 2:
            lengths = torch.randint(1, cols, (B,), device=device)
        else:
            lengths = None
        # occasionally force an all-masked row to exercise the safety path
        if trial % 5 == 0:
            mask[0, :] = True
        net._legacy_mask = True
        lg, lm = net.apply_mask_to_logits(logits, mask, idxs, lengths)
        net._legacy_mask = False
        vg, vm = net.apply_mask_to_logits(logits, mask, idxs, lengths)
        if not _eq_logits(lg, vg) or not bool((lm == vm).all()):
            ok = False
            print(f"  MISMATCH trial {trial}: B={B} cols={cols} "
                  f"lengths={'None' if lengths is None else 'set'}")
            break
        cases += 1
    print(f"  {'OK' if ok else 'FAIL'} — {cases} random cases identical")
    return ok


def e2e_test(model, device):
    print("== (B) released policy: fast vs legacy decode on real polygons ==")
    dp = os.getenv("DATASET_PATH")
    tr, _ = prepare_datasets(os.path.join(dp, "train"),
                             os.path.join(dp, "dev"), normalize=True)
    ds = Dataset(tr.samples[:40])
    loader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=False,
                                         collate_fn=collate_fn)
    net = _actor(model)
    ok = True
    dec_fast, dec_leg = 0.0, 0.0
    for bd, pm, lens, names in loader:
        bd = bd.to(device); pm = pm.to(device)
        lt = torch.as_tensor(lens, device=device)

        # greedy (deterministic) — must be identical index sequences
        net._legacy_mask = True
        g_leg, _ = model(bd, padding_mask=pm, lengths=lt, deterministic=True)
        net._legacy_mask = False
        g_fast, _ = model(bd, padding_mask=pm, lengths=lt, deterministic=True)
        for a, b in zip(g_leg, g_fast):
            aa = [int(x) for x in a]; bb = [int(x) for x in b]
            if aa != bb:
                ok = False; print(f"  MISMATCH greedy on a polygon: {aa[:8]} vs {bb[:8]}")
                break

        # stochastic — same seed each side, log-probs must match
        for legacy in (True, False):
            net._legacy_mask = legacy
            torch.manual_seed(1234)
            t0 = time.perf_counter()
            _, lp = model(bd.repeat_interleave(8, 0),
                          padding_mask=pm.repeat_interleave(8, 0),
                          lengths=lt.repeat_interleave(8), deterministic=False)
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if legacy:
                dec_leg += dt; lp_leg = lp.detach().float().cpu()
            else:
                dec_fast += dt; lp_fast = lp.detach().float().cpu()
        if not torch.allclose(lp_leg, lp_fast, atol=1e-4, rtol=1e-3):
            ok = False
            d = (lp_leg - lp_fast).abs().max().item()
            print(f"  MISMATCH stochastic log-probs: max|Δ|={d:.2e}")
        if not ok:
            break
    print(f"  {'OK' if ok else 'FAIL'} — greedy identical, stochastic log-probs match")
    if dec_fast > 0:
        print(f"  decode wall: legacy {dec_leg:.2f}s  fast {dec_fast:.2f}s  "
              f"-> {dec_leg/dec_fast:.1f}x on the masking path")
    return ok


def reward_test(model, device, cap_at_tau=False):
    print(f"== (C) batched disc-vis reward vs per-call scalar "
          f"(cap_at_tau={cap_at_tau}) ==")
    from po_agp import po_reward_smooth_disc, po_rewards_smooth_disc_batch
    dp = os.getenv("DATASET_PATH")
    tr, _ = prepare_datasets(os.path.join(dp, "train"),
                             os.path.join(dp, "dev"), normalize=True)
    ds = Dataset(tr.samples[:40])
    loader = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=False,
                                         collate_fn=collate_fn)
    K = 8
    # cap_at_tau=False is what the RELEASED recipe uses
    # (configs/po_agp_transformer_bt.json -> "cap_coverage": false), so it is
    # the default here; True is checked too since other configs set it.
    rp = dict(lam=1.0, tau=0.99, tau_penalty=3.0, cap_at_tau=cap_at_tau,
              n_samples=500)
    ok = True
    t_loop = t_batch = 0.0
    max_abs = 0.0
    for bd, pm, lens, names in loader:
        bd = bd.to(device); pm = pm.to(device); lt = torch.as_tensor(lens, device=device)
        B = bd.shape[0]
        torch.manual_seed(7)
        all_idxs, _ = model(bd.repeat_interleave(K, 0),
                            padding_mask=pm.repeat_interleave(K, 0),
                            lengths=lt.repeat_interleave(K), deterministic=False)
        pts = [bd[b, :int(lt[b])].cpu().numpy() for b in range(B)]
        sols = [[int(v) for v in all_idxs[i] if int(v) < int(lt[i // K])]
                for i in range(B * K)]
        # scalar per-call
        t0 = time.perf_counter()
        r_loop = [po_reward_smooth_disc(pts[i // K], sols[i], names[i // K],
                                        length=int(lt[i // K]), **rp)
                  for i in range(B * K)]
        t_loop += time.perf_counter() - t0
        # batched
        t0 = time.perf_counter()
        r_batch = po_rewards_smooth_disc_batch(
            pts, list(names), [int(x) for x in lens], sols, K=K, device=device, **rp)
        if device == "cuda":
            torch.cuda.synchronize()
        t_batch += time.perf_counter() - t0
        d = max(abs(a - b) for a, b in zip(r_loop, r_batch))
        max_abs = max(max_abs, d)
        # orderings within each polygon must match (that's what BT consumes)
        for b in range(B):
            lo = [r_loop[b * K + k] for k in range(K)]
            ba = [r_batch[b * K + k] for k in range(K)]
            if sorted(range(K), key=lambda k: lo[k]) != sorted(range(K), key=lambda k: ba[k]):
                # only a problem if not a near-tie
                import itertools
                for i, j in itertools.combinations(range(K), 2):
                    if (lo[i] > lo[j]) != (ba[i] > ba[j]) and abs(lo[i] - lo[j]) > 1e-6:
                        ok = False
                        print(f"  ORDER FLIP poly {b}: Δloop={lo[i]-lo[j]:.2e}")
    print(f"  max |Δreward| = {max_abs:.2e}  (want < 1e-6)")
    print(f"  {'OK' if ok and max_abs < 1e-6 else 'FAIL'} — reward values + orderings match")
    print(f"  reward wall: per-call {t_loop:.2f}s  batched {t_batch:.2f}s  "
          f"-> {t_loop/max(t_batch,1e-9):.1f}x")
    return ok and max_abs < 1e-6


def edge_case_reward_test(device):
    """Degenerate solutions the random-rollout gate is unlikely to produce.

    Real rollouts are masked against re-selection and pre-filtered to < n, so
    empty / duplicate / out-of-range solutions almost never appear in gate (C).
    They CAN arise (a policy that emits EOS immediately yields an empty
    solution), and the batched path handles them on separate code paths from
    the scalar one, so they are checked explicitly.
    """
    print("== (D) degenerate solutions: empty / duplicate / out-of-range ==")
    from po_agp import po_reward_smooth_disc, po_rewards_smooth_disc_batch
    dp = os.getenv("DATASET_PATH")
    tr, _ = prepare_datasets(os.path.join(dp, "train"),
                             os.path.join(dp, "dev"), normalize=True)
    ds = Dataset(tr.samples[:4])
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False,
                                         collate_fn=collate_fn)
    bd, pm, lens, names = next(iter(loader))
    B = bd.shape[0]
    n0, n1 = int(lens[0]), int(lens[1])
    K = 4
    # per polygon: empty, single guard, duplicated guard, first-three guards
    sols = []
    for b, n in enumerate((n0, n1)):
        sols += [[], [0], [0, 0], list(range(min(3, n)))]
    pts = [bd[b, :int(lens[b])].cpu().numpy() for b in range(B)]
    ok = True
    for cap in (False, True):
        rp = dict(lam=1.0, tau=0.99, tau_penalty=3.0, cap_at_tau=cap, n_samples=500)
        r_loop = [po_reward_smooth_disc(pts[i // K], sols[i], names[i // K],
                                        length=int(lens[i // K]), **rp)
                  for i in range(B * K)]
        r_batch = po_rewards_smooth_disc_batch(
            pts, list(names), [int(x) for x in lens], sols, K=K, device=device, **rp)
        d = max(abs(a - b) for a, b in zip(r_loop, r_batch))
        tag = "OK " if d < 1e-6 else "FAIL"
        print(f"  {tag} cap_at_tau={cap!s:5} max|Δ|={d:.2e}")
        if d >= 1e-6:
            ok = False
            for i, (x, y) in enumerate(zip(r_loop, r_batch)):
                if abs(x - y) >= 1e-6:
                    print(f"    sol={sols[i]!r}: scalar={x:.6f} batched={y:.6f}")
    return ok


def main():
    if not os.getenv("DATASET_PATH"):
        sys.exit("DATASET_PATH must be set")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_agp_model(128, 128, 1, 10.0, True, 1.0).to(device)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.eval()
    a = unit_test(model, device)
    b = e2e_test(model, device)
    # Both reward regimes: cap_at_tau=False is the released recipe, True is
    # used by other configs. The batched path must match the scalar in both.
    c = reward_test(model, device, cap_at_tau=False)
    c2 = reward_test(model, device, cap_at_tau=True)
    d = edge_case_reward_test(device)
    allok = a and b and c and c2 and d
    print("\n" + ("ALL PASS — fast decode + batched reward match baseline" if allok
                  else "FAILED — do NOT use the optimised path"))
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
