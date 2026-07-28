#!/usr/bin/env python
"""compare_policy_checkpoints.py -- policy-only comparison on the paper's split.

WHY
    The paper's headline "Pretrained pointer (seed)" row (cov 0.9689,
    |S|/n 0.1664, |S|/OPT 1.0885) is the RELEASED policy's greedy decode on the
    362-polygon dev_test split, scored with exact CGAL. Retrained seeds came out
    at |S|/OPT ~2.8-2.9, and it was not clear whether that gap is
    (a) checkpoint SELECTION picking a bad epoch, or
    (b) the retrained policies genuinely being worse at every epoch.

    This decides it: decode N candidate checkpoints on the SAME 362 polygons
    with the SAME settings build_ls_trajectories.py uses (greedy, model's own
    EOS, no tau-prefix trimming: deterministic=True, no_eos=False,
    eos_cov_threshold=0.0), score with exact CGAL, and print them side by side.

    Nothing here trains or writes checkpoints; it is read-only.

USAGE
    python tools/compare_policy_checkpoints.py \
        --ckpt released=checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt \
        --ckpt rel_e110=checkpoints/v3/po_agp/lstm_bt/po_agp_epoch110.pt \
        --ckpt seed11_e9=/path/to/po_agp_best_greedy.pt
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

from dataset import collate_fn                                        # noqa: E402
from po_agp import create_agp_model, prepare_datasets                 # noqa: E402
from utils import (                                                   # noqa: E402
    evaluate_polygon_visibility_numpy_wo_gt, get_or_build_disc_vis,
    load_disc_vis_cache,
)
from torch.utils.data import DataLoader                               # noqa: E402


def load_reference_names(path: str) -> set:
    """Polygon names of the canonical dev_test split (the paper's 362)."""
    import pickle
    with open(path, "rb") as f:
        recs = pickle.load(f)
    if isinstance(recs, dict):
        recs = recs.get("records", recs.get("data", []))
    names = set()
    for r in recs:
        nm = r.get("name") if isinstance(r, dict) else None
        if nm:
            names.add(nm)
    return names


def load_opt(sol_dir: str, name: str) -> int | None:
    """OPT = guard count of an optimal solution, from the .solution file.

    FORMAT (easy to get wrong): the file is
        # solutions 26
        7 11 20 25 28
        1 7 11 20 25
        ...
    i.e. a header line then ONE LINE PER alternative optimal solution, all of
    the same length. OPT is therefore the LENGTH OF A LINE (5 above), not the
    count of all integers in the file. Flattening the whole file and counting
    tokens inflates OPT ~26x and drives |S|/OPT to nonsense (measured: 0.0285
    instead of the paper's 1.0885 for the released policy).
    """
    p = os.path.join(sol_dir, name + ".solution")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                vals = [t for t in line.split() if t.lstrip("-").isdigit()]
                if vals:
                    return len(vals)          # first solution line's length
    except Exception:
        return None
    return None


def build_model(ckpt_path: str, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cp = ck.get("checkpoint_params") or {}
    a = ck.get("args") or {}

    def g(k, default):
        if k in cp:
            return cp[k]
        if k in a:
            return a[k]
        return default

    model = create_agp_model(
        embedding_size=int(g("embedding_size", 128)),
        hidden_size=int(g("hidden_size", 128)),
        n_glimpses=int(g("n_glimpses", 1)),
        tanh_exploration=float(g("tanh_exploration", 10)),
        use_tanh=bool(g("use_tanh", True)),
        temperature=float(g("temperature", 1.0)),
    )
    sd = ck["model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"    [warn] state_dict mismatch: {len(missing)} missing, "
              f"{len(unexpected)} unexpected")
    model.to(device).eval()
    return model, ck


def evaluate(model, loader, names_keep, sol_dir, disc_vis_samples, device):
    covs, chvs, opts = [], [], []
    n_used = 0
    with torch.no_grad():
        for batch_data, pad_mask, lengths, names in loader:
            name = names[0]
            if names_keep and name not in names_keep:
                continue
            n = int(lengths[0])
            batch_data = batch_data.to(device)
            pad_mask = pad_mask.to(device)
            lens_t = torch.tensor(lengths, dtype=torch.long, device=device)
            pts = batch_data[0, :n].detach().cpu().numpy()

            # identical decode contract to build_ls_trajectories.py
            det_idxs, _ = model(
                batch_data, padding_mask=pad_mask, lengths=lens_t,
                deterministic=True, no_eos=False, eos_cov_threshold=0.0,
            )
            sol = [int(i) for i in det_idxs[0] if int(i) < n]
            try:
                cov = float(evaluate_polygon_visibility_numpy_wo_gt(
                    pts, np.array(sol, dtype=np.int64), name)) if sol else 0.0
            except Exception:
                cov = 0.0
            covs.append(cov)
            chvs.append(len(sol) / max(1, n))
            o = load_opt(sol_dir, name)
            if o:
                opts.append(len(sol) / o)
            n_used += 1
    return {
        "n": n_used,
        "cov": float(np.mean(covs)) if covs else None,
        "chv": float(np.mean(chvs)) if chvs else None,
        "opt": float(np.mean(opts)) if opts else None,
        "cov_ge_095": int(sum(c >= 0.95 for c in covs)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", default=[],
                   help="label=path (repeatable)")
    ap.add_argument("--ref-test", default="data/ls_trajectories_dev_test_clean.pkl",
                   help="canonical dev_test trajectory file; its polygon names "
                        "define the paper's 362-polygon split")
    ap.add_argument("--disc-vis-samples", type=int, default=500)
    ap.add_argument("--disc-vis-cache", default="data/disc_vis_cache.pkl")
    args = ap.parse_args()

    dp = os.getenv("DATASET_PATH")
    if not dp:
        sys.exit("DATASET_PATH must be set (see .env)")
    sol_dir = os.path.join(dp, "dev")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(args.disc_vis_cache):
        load_disc_vis_cache(args.disc_vis_cache, verbose=False)

    names_keep = set()
    if os.path.exists(args.ref_test):
        names_keep = load_reference_names(args.ref_test)
    print(f"[cmp] dev_test reference: {len(names_keep)} polygons "
          f"({args.ref_test})")

    _, val_ds = prepare_datasets(os.path.join(dp, "train"),
                                os.path.join(dp, "dev"), normalize=True)
    loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    print(f"[cmp] device={device}\n")
    hdr = f"{'label':<18}{'epoch':>6}  {'cov':>8}  {'|S|/n':>8}  {'|S|/OPT':>9}  {'cov>=.95':>9}  {'n':>5}"
    print(hdr)
    print("-" * len(hdr))
    for spec in args.ckpt:
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            label, path = os.path.basename(spec), spec
        if not os.path.exists(path):
            print(f"{label:<18}{'--':>6}  (missing: {path})")
            continue
        t0 = time.perf_counter()
        model, ck = build_model(path, device)
        r = evaluate(model, loader, names_keep, sol_dir,
                     args.disc_vis_samples, device)
        ep = ck.get("epoch")
        print(f"{label:<18}{str(ep):>6}  {r['cov']:8.4f}  {r['chv']:8.4f}  "
              f"{(r['opt'] if r['opt'] is not None else float('nan')):9.4f}  "
              f"{r['cov_ge_095']:>4}/{r['n']:<4}  {r['n']:>5}"
              f"   ({time.perf_counter()-t0:.0f}s)")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
