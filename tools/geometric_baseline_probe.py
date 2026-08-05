"""Does cheap analytic geometry explain what the frozen encoder gives us?

The capacity ladder answers "is the probe computing the geometry rather than
reading it?" by holding features fixed and shrinking the probe. The other half of
the standard probing control (Hewitt & Liang) is a *baseline representation* that
establishes the floor. The paper has one -- the coords-only ablation arm -- but it
is a LEARNED baseline: a 464K-parameter Transformer over raw (x, y). If a handful
of analytic, O(n), coordinate-only features beat it, then the coords-only arm
understates what coordinates carry, and the encoder's margin is smaller than the
2x2 ablation implies.

Reflexness is the classical AGP signal: guards go at reflex vertices. Nothing here
touches a visibility oracle, so every arm below is geo-free by the paper's own
definition (Section 2.3) -- these are functions of the vertex coordinates alone.

Protocol is copied exactly from build_paper_data.build_encoder_linear_probe so the
numbers are comparable cell-for-cell with the ladder's linear rung (0.843 +/- 0.009):
same 200 polygons, same LS labels, GroupKFold(5) on polygon, StandardScaler, then
LogisticRegression(C=1.0, class_weight="balanced").

Arm `embedding` re-runs the published rung as a harness check: if it does not come
back at 0.843 the comparison is not apples-to-apples and nothing else here counts.

Usage:  python tools/geometric_baseline_probe.py [--n-polygons 200] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
TRAJ = REPO / "data/ls_trajectories_dev_test_clean.pkl"
OUT = REPO / "paper/data/geometric_baseline_probe.json"


# ----------------------------------------------------------------- features
def polygon_features(pts: np.ndarray) -> dict[str, np.ndarray]:
    """Per-vertex analytic features from coordinates alone. O(n), no oracle.

    Orientation matters: the interior angle is only meaningful once we know which
    side is inside, so we take the signed area and flip the cross-product sign for
    clockwise polygons. Getting this backwards silently swaps reflex for convex,
    which would make the baseline look worse than it is.
    """
    n = len(pts)
    prv = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)

    e_in = pts - prv          # edge arriving at i
    e_out = nxt - pts         # edge leaving i
    len_in = np.linalg.norm(e_in, axis=1)
    len_out = np.linalg.norm(e_out, axis=1)
    eps = 1e-12

    # signed area (shoelace): >0 counter-clockwise
    signed_area = 0.5 * np.sum(pts[:, 0] * nxt[:, 1] - nxt[:, 0] * pts[:, 1])
    orient = 1.0 if signed_area > 0 else -1.0

    cross = e_in[:, 0] * e_out[:, 1] - e_in[:, 1] * e_out[:, 0]
    dot = np.sum(e_in * e_out, axis=1)
    # turning angle in (-pi, pi]; positive = left turn for a CCW polygon
    turn = np.arctan2(orient * cross, dot)
    interior = np.pi - turn                      # interior angle in (0, 2pi)
    reflex = (interior > np.pi).astype(np.float64)

    centroid = pts.mean(axis=0)
    d_cent = np.linalg.norm(pts - centroid, axis=1)

    # local visibility-ish proxies that still never call a visibility routine
    span = np.linalg.norm(nxt - prv, axis=1)     # chord across the corner
    return {
        "interior": interior,
        "sin_int": np.sin(interior),
        "cos_int": np.cos(interior),
        "reflex": reflex,
        "turn": turn,
        "len_in": len_in / (len_in.mean() + eps),
        "len_out": len_out / (len_out.mean() + eps),
        "span": span / (span.mean() + eps),
        "d_cent": d_cent / (d_cent.mean() + eps),
        "n_vert": np.full(n, float(n)),
    }


ARMS = {
    # the true floor: what the paper's coords-only arm is given
    "coords": lambda f, pts, emb: pts,
    # the classical AGP signal, one binary feature
    "reflex_only": lambda f, pts, emb: f["reflex"][:, None],
    # the interior angle, continuous
    "angle_only": lambda f, pts, emb: np.stack(
        [f["interior"], f["sin_int"], f["cos_int"]], axis=1),
    # everything analytic
    "geometry": lambda f, pts, emb: np.stack(
        [f[k] for k in ("interior", "sin_int", "cos_int", "reflex", "turn",
                        "len_in", "len_out", "span", "d_cent", "n_vert")], axis=1),
    "geometry+coords": lambda f, pts, emb: np.concatenate(
        [np.stack([f[k] for k in ("interior", "sin_int", "cos_int", "reflex", "turn",
                                  "len_in", "len_out", "span", "d_cent", "n_vert")],
                  axis=1), pts], axis=1),
    # the published rung, re-run here as a harness check
    "embedding": lambda f, pts, emb: emb,
    "embedding+geometry": lambda f, pts, emb: np.concatenate(
        [emb, np.stack([f[k] for k in ("interior", "sin_int", "cos_int", "reflex",
                                       "turn", "len_in", "len_out", "span",
                                       "d_cent", "n_vert")], axis=1)], axis=1),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-polygons", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-embedding", action="store_true",
                    help="analytic arms only; no torch/checkpoint needed")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.preprocessing import StandardScaler

    with open(TRAJ, "rb") as f:
        records = pickle.load(f)["records"]
    sample = records[: args.n_polygons]

    pointer = None
    if not args.skip_embedding:
        sys.path.insert(0, str(REPO / "paper/scripts"))
        from build_paper_data import _load_pointer, extract_pointer_embeddings  # noqa
        import torch
        pointer = _load_pointer(args.device)

    feats, coords, embs, ys, groups = [], [], [], [], []
    for gi, rec in enumerate(sample):
        pts = np.asarray(rec["points"], dtype=np.float64)
        n = int(rec["n"])
        feats.append(polygon_features(pts))
        coords.append(pts)
        y = np.zeros(n, dtype=np.int32)
        for j in rec.get("final", []):
            if 0 <= int(j) < n:
                y[int(j)] = 1
        ys.append(y)
        groups.append(np.full(n, gi, dtype=np.int64))
        if pointer is not None:
            import torch
            pt = torch.tensor(pts.astype(np.float32),
                              device=args.device).unsqueeze(0)
            lengths = torch.tensor([n], device=args.device)
            with torch.no_grad():
                e = extract_pointer_embeddings(pointer, pt, lengths)[0]
            embs.append(e.cpu().numpy())

    F = {k: np.concatenate([f[k] for f in feats]) for k in feats[0]}
    P = np.concatenate(coords, axis=0)
    E = np.concatenate(embs, axis=0) if embs else None
    y = np.concatenate(ys)
    g = np.concatenate(groups)
    print(f"{len(sample)} polygons, {len(y)} vertices, {int(y.sum())} positive "
          f"(prior {y.mean():.3f})")
    print(f"reflex vertices: {int(F['reflex'].sum())} "
          f"({F['reflex'].mean():.3f} of all)")
    print(f"P(guard | reflex) = {y[F['reflex']==1].mean():.3f}   "
          f"P(guard | convex) = {y[F['reflex']==0].mean():.3f}\n")

    results = {}
    for arm, build in ARMS.items():
        if E is None and "embedding" in arm:
            continue
        X = build(F, P, E)
        gkf = GroupKFold(n_splits=5)
        roc, prc = [], []
        for tr, te in gkf.split(X, y, g):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=1000, C=1.0,
                                     class_weight="balanced")
            clf.fit(sc.transform(X[tr]), y[tr])
            s = clf.predict_proba(sc.transform(X[te]))[:, 1]
            roc.append(float(roc_auc_score(y[te], s)))
            prc.append(float(average_precision_score(y[te], s)))
        results[arm] = {
            "n_features": int(X.shape[1]),
            "roc_auc_mean": float(np.mean(roc)), "roc_auc_std": float(np.std(roc)),
            "pr_auc_mean": float(np.mean(prc)), "pr_auc_std": float(np.std(prc)),
            "roc_auc_per_fold": roc, "pr_auc_per_fold": prc,
        }
        print(f"  {arm:<20} {X.shape[1]:>4} feat   "
              f"ROC-AUC {np.mean(roc):.4f} +/- {np.std(roc):.4f}   "
              f"PR-AUC {np.mean(prc):.4f} +/- {np.std(prc):.4f}")

    out = {
        "note": ("Analytic coordinate-only features vs the frozen encoder, same "
                 "protocol as encoder_linear_probe (200 polygons, GroupKFold(5) on "
                 "polygon, L2 logistic regression, LS-derived labels). No arm calls "
                 "a visibility oracle, so all are geo-free."),
        "n_polygons": len(sample), "n_vertices": int(len(y)),
        "n_positive": int(y.sum()), "positive_prior": float(y.mean()),
        "reflex_fraction": float(F["reflex"].mean()),
        "p_guard_given_reflex": float(y[F["reflex"] == 1].mean()),
        "p_guard_given_convex": float(y[F["reflex"] == 0].mean()),
        "arms": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print(f"({time.perf_counter()-t0:.1f}s)")
