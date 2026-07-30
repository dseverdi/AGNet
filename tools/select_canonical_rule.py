#!/usr/bin/env python
"""select_canonical_rule.py -- pick a canonical START-VERTEX rule OFF-SPLIT,
then report it on the reporting split.

Why this script exists
----------------------
``tools/eval_canonical_start.py`` established (results/canonical_start.json)
that pinning a deterministic, geometrically defined vertex at index 0 before
the unidirectional LSTM encoder sees the polygon makes cyclic re-indexing
invariance EXACT (362/362 bit-identical guard sets over 8 roll amounts).  It
also found two things that make "which rule?" a delicate question:

1. The start vertex is a GUARD-COUNT KNOB.  At the fixed probe threshold
   t=0.20 the mean |S| on dev_test was 13.20 (lex-min start) / 17.16 (native
   file order) / 20.69 (lex-max start) -- a 57 % swing produced by nothing but
   re-indexing.  A fixed-threshold comparison of start rules therefore compares
   different guard budgets, so every rule is ALSO scored at a matched budget
   (m = |S_native| per polygon, each rule spending it on its own top-m
   predicted vertices).
2. That run found lex-max ~ native and lex-min much worse -- but it measured
   this ON THE REPORTING SPLIT.  Choosing lex-max on that evidence would be
   selection-on-test, the same error as the ``set_predictor_best.pt`` leak this
   project already fixed.

Protocol (fixed in advance, no deviation)
-----------------------------------------
* Candidate set, a priori, six rules: lexmin_xy, lexmax_xy, lexmin_yx,
  lexmax_yx, far_centroid, near_centroid.  Every rule must give a UNIQUE
  argument on every polygon; a rule that ties anywhere is DISQUALIFIED, never
  index-tie-broken (an index tie-break is not roll-invariant and would destroy
  the very property under test).
* SELECTION SPLIT = dev_tune (857 records) minus the 17 polygons byte-identical
  to train => 840.  Identity by sha1 of the float64 coordinate bytes, the
  convention of tools/dedup_partitions.py.
* SELECTION CRITERION = number of polygons with exact-CGAL coverage >= 0.95 on
  dev_tune AT MATCHED BUDGET.  Matched budget, not fixed threshold, because of
  the guard-count confound above.
* Only the winner is then reported on dev_test (362) against the known native
  baseline 345/362, together with an exact roll-invariance check.
* dev_test numbers for all six rules are additionally reported, explicitly
  labelled as post-selection and NOT used for selection.

Coverage bookkeeping
--------------------
Every candidate rule is a pure CYCLIC RELABELLING of the same normalised point
set, so coverage is scored once per polygon on the NATIVE array with each
rule's guards mapped back to native vertex identities.  This is exact (the
guard set is a set of points; coverage does not depend on how they are
indexed), it collapses seven CGAL visibility builds per polygon into one, and
it removes any chance of a stale cross-condition cache hit because the points
array handed to the cache is literally the same object every time.  The
equality with the score-on-the-rolled-array convention used by
eval_canonical_start.py is verified numerically (``equivalence_check``).

Usage:
  set -a; . ./.env; set +a
  AGNET_DISC_VIS_CACHE_SIZE=14000 python tools/select_canonical_rule.py \
      --out results/canonical_rule_selection.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
import time

import numpy as np

# AGNET_VIS_WORKERS=12 crashes this workload with
#   CGAL ERROR: precondition violation! Cannot construct a degenerate segment
# in utils.py's parallel visibility path. Force the sequential path before
# utils is imported.
if os.getenv("AGNET_VIS_WORKERS", "0") not in ("", "0"):
    raise SystemExit("refusing to run with AGNET_VIS_WORKERS="
                     f"{os.environ['AGNET_VIS_WORKERS']!r}; must be 0/unset")
os.environ["AGNET_VIS_WORKERS"] = "0"
# One exact-visibility cache entry per polygon is enough (all rules share it),
# and we never revisit a polygon, so a small LRU keeps memory flat.
os.environ.setdefault("AGNET_VIS_CACHE_SIZE", "48")

import torch  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from eval_canonical_start import (_decode_seed, _load_probe, _minmax,  # noqa: E402
                                 _roll_ks, _summarise)
from po_agp import create_agp_model                                # noqa: E402
from set_predictor import extract_pointer_embeddings                # noqa: E402
from utils import evaluate_polygon_visibility_numpy_wo_gt          # noqa: E402

# ---------------------------------------------------------------- candidates
RULES = ["lexmin_xy", "lexmax_xy", "lexmin_yx", "lexmax_yx",
         "far_centroid", "near_centroid"]
CONDITIONS = ["native"] + RULES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pointer-checkpoint", type=str,
                   default="checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt")
    # _final.pt, not _best.pt (which was selected on a split containing the
    # reporting split -- test-set leak).
    p.add_argument("--probe-checkpoint", type=str,
                   default="checkpoints/set_predictor/standard/set_predictor_final.pt")
    p.add_argument("--tune-traj", type=str,
                   default="data/ls_trajectories_dev_tune.pkl")
    p.add_argument("--train-traj", type=str,
                   default="data/ls_trajectories_train.pkl")
    p.add_argument("--test-traj", type=str,
                   default="data/ls_trajectories_dev_test_clean.pkl")
    p.add_argument("--embedding-size", type=int, default=128)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--n-glimpses", type=int, default=1)
    p.add_argument("--tanh-exploration", type=float, default=10.0)
    p.add_argument("--threshold", type=float, default=0.20)
    p.add_argument("--feasibility-gate", type=float, default=0.95)
    p.add_argument("--limit-tune", type=int, default=0)
    p.add_argument("--limit-test", type=int, default=0)
    p.add_argument("--equiv-check-n", type=int, default=20,
                   help="polygons on which score-on-native == score-on-rolled "
                        "is verified numerically")
    p.add_argument("--out", type=str,
                   default="results/canonical_rule_selection.json")
    p.add_argument("--augment-only", action="store_true",
                   help="do not re-run; just append the dev_tune "
                        "size-sensitivity block to --out")
    return p.parse_args()


# ------------------------------------------------------------------ geometry
def _centroid(q: np.ndarray) -> np.ndarray:
    """Vertex centroid, computed PERMUTATION-INVARIANTLY.

    ``q.mean(0)`` sums in array order, so a cyclic roll can change the result
    in the last bits and, on a near-tie, could flip the centroid rules' argmax.
    Summing a lexicographically sorted copy with ``math.fsum`` (exact) makes
    the centroid bit-identical for every roll of the same polygon.
    """
    o = np.lexsort((q[:, 1], q[:, 0]))
    qs = q[o]
    n = float(q.shape[0])
    return np.array([math.fsum(qs[:, 0].tolist()) / n,
                     math.fsum(qs[:, 1].tolist()) / n], dtype=np.float64)


def start_index(q: np.ndarray, rule: str) -> tuple[int, bool, float]:
    """Index of *rule*'s canonical vertex of the min-max-normalised *q*.

    Returns (index, unique, gap).  ``unique`` is False iff the runner-up
    attains EXACTLY the same key, in which case the choice is not
    roll-invariant and the caller must disqualify the rule rather than break
    the tie by index.  ``gap`` is the (informational) key margin to the
    runner-up.
    """
    n = q.shape[0]
    if rule == "native":
        return 0, True, float("inf")
    if rule in ("lexmin_xy", "lexmax_xy", "lexmin_yx", "lexmax_yx"):
        if rule.endswith("_xy"):                    # primary x, secondary y
            order = np.lexsort((q[:, 1], q[:, 0]))
        else:                                        # primary y, secondary x
            order = np.lexsort((q[:, 0], q[:, 1]))
        if rule.startswith("lexmax"):
            order = order[::-1]
        j, j2 = int(order[0]), int(order[1 % n])
        same = bool(q[j, 0] == q[j2, 0] and q[j, 1] == q[j2, 1])
        gap = float(max(abs(q[j, 0] - q[j2, 0]), abs(q[j, 1] - q[j2, 1])))
        return j, (not same) or n == 1, gap
    if rule in ("far_centroid", "near_centroid"):
        c = _centroid(q)
        d = ((q - c) ** 2).sum(1)
        order = np.argsort(d, kind="stable")
        if rule == "far_centroid":
            order = order[::-1]
        j, j2 = int(order[0]), int(order[1 % n])
        same = bool(d[j] == d[j2])
        return j, (not same) or n == 1, float(abs(d[j] - d[j2]))
    raise ValueError(f"unknown rule {rule!r}")


def canonical_roll(q: np.ndarray, rule: str) -> tuple[np.ndarray, int, bool, float]:
    """Roll *q* so *rule*'s canonical vertex is at index 0.

    A cyclic roll leaves per-axis min/max untouched, so the result is still
    min-max normalised.  ``rolled[i] == q[(i + j) % n]``, hence a guard at
    canonical index c is native index ``(c + j) % n``.
    """
    j, uniq, gap = start_index(q, rule)
    if j == 0:
        return q, 0, uniq, gap
    return np.roll(q, -j, axis=0), j, uniq, gap


# ------------------------------------------------------------------- runtime
def _predict(pointer, probe, q: np.ndarray, n: int, threshold: float,
             device: str) -> tuple[list[int], list[int], np.ndarray]:
    """Full frozen pipeline on min-max-normalised coords *q*.

    Returns (seed, guards, probs) in *q*'s own index space.  The pointer's
    ``padding_mask`` is a VALIDITY mask (True = real vertex) -- see
    eval_canonical_start._decode_seed, which is reused verbatim.
    """
    pt = torch.from_numpy(np.ascontiguousarray(q)).float().unsqueeze(0).to(device)
    seed = _decode_seed(pointer, pt, n, device)
    in_S = torch.zeros(1, n, dtype=torch.bool, device=device)
    if seed:
        in_S[0, seed] = True
    emb = extract_pointer_embeddings(pointer, pt, [n])
    pad = torch.zeros(1, n, dtype=torch.bool, device=device)   # SetPredictor
    logits = probe(emb, pt, in_S, pad)                         # pad = TRUE padding
    probs = torch.sigmoid(logits)[0, :n]
    guards = (probs >= threshold).nonzero(as_tuple=True)[0].cpu().tolist()
    return seed, guards, probs.cpu().numpy()


def _sig(rec: dict) -> str:
    return hashlib.sha1(
        np.asarray(rec["points"], dtype=np.float64).tobytes()).hexdigest()


def _load_records(path: str) -> list[dict]:
    with open(os.path.join(REPO_ROOT, path), "rb") as fh:
        blob = pickle.load(fh)
    recs = blob.get("records", blob) if isinstance(blob, dict) else blob
    return [r for r in recs if r.get("points") is not None]


def _stats(cov: dict, ratio: dict, ng: dict, gate: float) -> dict:
    out = {}
    for c in cov:
        out[c] = _summarise(cov[c], ratio[c], gate)
        out[c]["mean_n_guards"] = round(float(np.mean(ng[c])), 3)
    for c in out:
        out[c]["n_ge_gate_delta_vs_native"] = (
            out[c]["n_ge_gate"] - out["native"]["n_ge_gate"])
    return out


def _print_table(title: str, summary: dict, order: list[str]) -> None:
    print(f"\n--- {title} ---")
    print(f"{'rule':<15}{'>=0.95':>10}{'d_nat':>8}{'mean_cov':>10}"
          f"{'min_cov':>9}{'|S|/n':>8}{'mean|S|':>9}")
    for c in order:
        if c not in summary:
            continue
        s = summary[c]
        print(f"{c:<15}{s['n_ge_gate']:>5}/{s['n_polygons']:<4}"
              f"{s['n_ge_gate_delta_vs_native']:>8}{s['mean_cov']:>10.4f}"
              f"{s['min_cov']:>9.4f}{s['mean_S_over_n']:>8.4f}"
              f"{s['mean_n_guards']:>9.2f}")


def evaluate_split(pointer, probe, records: list[dict], rules: list[str],
                   threshold: float, gate: float, device: str, tag: str,
                   verbose_every: int = 50) -> dict:
    """Score `native` + every rule in *rules* on *records*, fixed-t and matched
    budget.  Coverage is exact CGAL, scored on the native (unrolled) array with
    each rule's guards mapped back to native vertex identities."""
    conds = ["native"] + [r for r in rules]
    cov = {c: [] for c in conds}
    ratio = {c: [] for c in conds}
    ng = {c: [] for c in conds}
    below = {c: [] for c in conds}
    mb_cov = {c: [] for c in conds}
    mb_ratio = {c: [] for c in conds}
    mb_ng = {c: [] for c in conds}
    mb_below = {c: [] for c in conds}

    uniq = {c: {"n_ties": 0, "ties": [], "min_gap": float("inf")} for c in rules}
    start_hist = {c: [] for c in rules}
    seed_match = 0
    seed_mismatch = []
    guardset_eq_native = {c: 0 for c in rules}
    t0 = time.time()

    with torch.no_grad():
        for ridx, rec in enumerate(records):
            n = int(rec["n"]) if "n" in rec else len(rec["points"])
            p0 = np.asarray(rec["points"], dtype=np.float64)[:n]
            name = rec.get("name", f"poly{ridx}")
            q0 = _minmax(p0)
            # one exact-visibility cache entry per polygon, shared by all rules
            uname = f"{name}__canonrule_{tag}"

            guards_nat: dict[str, list[int]] = {}
            probs_j: dict[str, tuple[np.ndarray, int]] = {}
            for c in conds:
                qc, j, is_uniq, gap = canonical_roll(q0, c)
                if c in rules:
                    start_hist[c].append(j)
                    if not is_uniq:
                        uniq[c]["n_ties"] += 1
                        if len(uniq[c]["ties"]) < 10:
                            uniq[c]["ties"].append({"name": name, "n": n,
                                                    "index": j})
                    uniq[c]["min_gap"] = min(uniq[c]["min_gap"], gap)
                seed, guards, probs = _predict(pointer, probe, qc, n,
                                               threshold, device)
                if c == "native":
                    stored = rec.get("seed")
                    if stored is not None:
                        if list(seed) == [int(x) for x in stored]:
                            seed_match += 1
                        elif len(seed_mismatch) < 10:
                            seed_mismatch.append(
                                {"name": name, "decoded": seed,
                                 "stored": [int(x) for x in stored]})
                gn = sorted(int((g + j) % n) for g in guards)
                guards_nat[c] = gn
                probs_j[c] = (probs, j)
                cvg = evaluate_polygon_visibility_numpy_wo_gt(
                    q0, np.array(gn, dtype=np.int64), uname) if gn else 0.0
                cov[c].append(float(cvg))
                ratio[c].append(len(gn) / float(n))
                ng[c].append(len(gn))
                if cvg < gate:
                    below[c].append({"name": name, "n": n,
                                     "cov": round(float(cvg), 4),
                                     "n_guards": len(gn)})
                if c in rules and set(gn) == set(guards_nat["native"]):
                    guardset_eq_native[c] += 1

            # ---- matched budget: every rule spends m = |S_native| guards ---
            m = len(guards_nat["native"])
            for c in conds:
                pr, j = probs_j[c]
                order = np.argsort(-pr, kind="stable")
                top = sorted(int((int(i) + j) % n) for i in order[:m])
                cmb = evaluate_polygon_visibility_numpy_wo_gt(
                    q0, np.array(top, dtype=np.int64), uname) if top else 0.0
                mb_cov[c].append(float(cmb))
                mb_ratio[c].append(len(top) / float(n))
                mb_ng[c].append(len(top))
                if cmb < gate:
                    mb_below[c].append({"name": name, "n": n,
                                        "cov": round(float(cmb), 4),
                                        "n_guards": len(top)})

            if (ridx + 1) % verbose_every == 0:
                print(f"  [{tag}] {ridx + 1}/{len(records)}  "
                      f"({time.time() - t0:.0f}s)", flush=True)

    for c in rules:
        if uniq[c]["min_gap"] == float("inf"):
            uniq[c]["min_gap"] = None
        else:
            uniq[c]["min_gap"] = float(uniq[c]["min_gap"])
        uniq[c]["disqualified"] = uniq[c]["n_ties"] > 0
        uniq[c]["n_already_at_index0"] = int(
            sum(1 for v in start_hist[c] if v == 0))
        uniq[c]["mean_start_index"] = round(
            float(np.mean(start_hist[c])), 2) if start_hist[c] else None

    return {
        "n_polygons": len(records),
        "fixed_threshold": _stats(cov, ratio, ng, gate),
        "matched_budget": _stats(mb_cov, mb_ratio, mb_ng, gate),
        "uniqueness": uniq,
        "sanity": {
            "seed_reproduced_native": seed_match,
            "seed_mismatch_examples": seed_mismatch,
            "guardset_equals_native": guardset_eq_native,
        },
        "below_gate_fixed_t": {c: below[c] for c in conds},
        "below_gate_matched": {c: mb_below[c] for c in conds},
        "per_polygon_coverage_fixed_t": {c: [round(v, 6) for v in cov[c]]
                                         for c in conds},
        "per_polygon_coverage_matched": {c: [round(v, 6) for v in mb_cov[c]]
                                         for c in conds},
        "names": [r.get("name", f"poly{i}") for i, r in enumerate(records)],
        "elapsed_sec": round(time.time() - t0, 1),
    }


def check_roll_invariance(pointer, probe, records: list[dict], rule: str,
                          threshold: float, device: str) -> dict:
    """Exact-invariance check: canonicalise several cyclic rolls of each
    polygon, map the guard set back to ORIGINAL vertex identities and require
    it to be identical for every roll (and the canonical array bit-identical).
    """
    ok = 0
    failures = []
    t0 = time.time()
    with torch.no_grad():
        for ridx, rec in enumerate(records):
            n = int(rec["n"]) if "n" in rec else len(rec["points"])
            p0 = np.asarray(rec["points"], dtype=np.float64)[:n]
            name = rec.get("name", f"poly{ridx}")
            sets = {}
            ref_arr = None
            bit_identical = True
            for k in _roll_ks(n):
                p_k = np.roll(p0, k, axis=0)          # p_k[i] = p0[(i-k)%n]
                qc, j_k, _u, _g = canonical_roll(_minmax(p_k), rule)
                if ref_arr is None:
                    ref_arr = qc
                elif not np.array_equal(ref_arr, qc):
                    bit_identical = False
                _s, g, _p = _predict(pointer, probe, qc, n, threshold, device)
                sets[k] = frozenset(int((c + j_k - k) % n) for c in g)
            ref = sets[next(iter(sets))]
            if bit_identical and all(s == ref for s in sets.values()):
                ok += 1
            elif len(failures) < 20:
                failures.append({"name": name, "n": n,
                                 "arrays_bit_identical": bit_identical,
                                 "sets": {str(k): sorted(s)
                                          for k, s in sets.items()}})
    return {"rule": rule, "checked": len(records), "exact": ok,
            "fraction": round(ok / len(records), 6) if records else None,
            "roll_amounts": "k in {0,1,2,n//4,n//3,n//2,2n//3,n-1} (deduped)",
            "failures": failures,
            "elapsed_sec": round(time.time() - t0, 1)}


def check_score_equivalence(pointer, probe, records: list[dict], rules: list[str],
                            threshold: float, device: str, k: int) -> dict:
    """Verify that scoring coverage on the NATIVE array with guards mapped back
    to native identities equals scoring on the ROLLED array with the rule's own
    indices (the convention of eval_canonical_start.py)."""
    worst = 0.0
    pairs = 0
    with torch.no_grad():
        for ridx, rec in enumerate(records[:k]):
            n = int(rec["n"]) if "n" in rec else len(rec["points"])
            p0 = np.asarray(rec["points"], dtype=np.float64)[:n]
            name = rec.get("name", f"poly{ridx}")
            q0 = _minmax(p0)
            for r in rules:
                qc, j, _u, _g = canonical_roll(q0, r)
                _s, g, _p = _predict(pointer, probe, qc, n, threshold, device)
                if not g:
                    continue
                on_rolled = evaluate_polygon_visibility_numpy_wo_gt(
                    qc, np.array(sorted(g), dtype=np.int64),
                    f"{name}__equiv_rolled_{r}")
                gn = sorted(int((x + j) % n) for x in g)
                on_native = evaluate_polygon_visibility_numpy_wo_gt(
                    q0, np.array(gn, dtype=np.int64), f"{name}__equiv_native")
                worst = max(worst, abs(on_rolled - on_native))
                pairs += 1
    return {"n_polygons": min(k, len(records)), "n_comparisons": pairs,
            "max_abs_coverage_diff": worst,
            "note": "coverage is a property of the guard POINT SET, so a cyclic "
                    "relabelling cannot change it; this confirms it numerically"}


def size_sensitivity(res: dict, n_by_name: dict[str, int], rules: list[str],
                     gate: float, caps: tuple[int, ...]) -> dict:
    """Re-rank the candidate rules on SIZE-RESTRICTED subsets of the SELECTION
    split.

    Motivation: data/ls_trajectories_dev_tune.pkl turns out to be the
    alphabetical FIRST 857 of the name-sorted dev pool (its metadata carries no
    split_seed/split_method, i.e. it was not produced by the seeded shuffle that
    tools/split_dev_pickle.py now performs), and dev_test is the alphabetical
    LAST 367.  Because polygon names embed the vertex count as a STRING
    (rand-156-3 sorts before rand-28-10), the carve is a covariate shift in n:
    dev_tune spans 16-198 with median 156, dev_test spans 8-192 with median 72.
    Selection on dev_tune is therefore leak-free but OFF-DISTRIBUTION.  These
    subsets re-rank using only dev_tune polygons small enough to overlap
    dev_test's range -- still no reporting-split information whatsoever -- to
    test whether the winner is an artefact of the size shift.
    """
    names = res["names"]
    ns = np.array([n_by_name[nm] for nm in names])
    out = {"note": size_sensitivity.__doc__.strip(), "subsets": {}}
    for cap in caps:
        sel = ns <= cap
        if sel.sum() < 10:
            continue
        blk = {"n_polygons": int(sel.sum()), "cap_n": int(cap)}
        for key in ("matched", "fixed_t"):
            src = res[f"per_polygon_coverage_{key}"]
            rows = []
            for c in ["native"] + rules:
                a = np.asarray(src[c], dtype=np.float64)[sel]
                rows.append({"rule": c, "n_ge_gate": int((a >= gate).sum()),
                             "mean_cov": round(float(a.mean()), 4)})
            base = next(r["n_ge_gate"] for r in rows if r["rule"] == "native")
            for r in rows:
                r["delta_vs_native"] = r["n_ge_gate"] - base
            ranked = sorted([r for r in rows if r["rule"] in rules],
                            key=lambda r: (-r["n_ge_gate"], -r["mean_cov"],
                                           r["rule"]))
            blk[key] = {"rows": rows, "winner": ranked[0]["rule"],
                        "ranking": [(r["rule"], r["n_ge_gate"]) for r in ranked]}
        out["subsets"][f"n_le_{cap}"] = blk
    return out


def augment(args: argparse.Namespace) -> None:
    """Append the dev_tune size-sensitivity block to an existing result file."""
    out_path = os.path.join(REPO_ROOT, args.out)
    with open(out_path) as fh:
        res = json.load(fh)
    tune_all = _load_records(args.tune_traj)
    n_by_name = {r.get("name", f"poly{i}"):
                 (int(r["n"]) if "n" in r else len(r["points"]))
                 for i, r in enumerate(tune_all)}
    blk = size_sensitivity(res["dev_tune"], n_by_name, RULES,
                           args.feasibility_gate, (100, 110, 130, 156))
    res["dev_tune_size_sensitivity"] = blk
    res["protocol"]["selection_split_caveat"] = (
        "dev_tune is the alphabetical first 857 of the name-sorted dev pool "
        "(no split_seed in its metadata), dev_test the alphabetical last 367; "
        "since names embed n as a string this is a covariate shift in polygon "
        "size (dev_tune median n=156, dev_test median n=72). Selection is "
        "leak-free but off-distribution; see dev_tune_size_sensitivity.")
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"[canonrule] size-sensitivity on the SELECTION split "
          f"(dev_tune only, no dev_test information):")
    for k, b in blk["subsets"].items():
        print(f"  {k} (N={b['n_polygons']}): matched-budget winner="
              f"{b['matched']['winner']}  {b['matched']['ranking']}")
        print(f"      native={next(r['n_ge_gate'] for r in b['matched']['rows'] if r['rule']=='native')}")
    print(f"[canonrule] updated {out_path}")


def main() -> None:
    args = parse_args()
    if args.augment_only:
        augment(args)
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.time()
    print(f"[canonrule] device={device} t={args.threshold} gate={args.feasibility_gate}")
    print(f"[canonrule] AGNET_VIS_WORKERS={os.environ['AGNET_VIS_WORKERS']} "
          f"AGNET_VIS_CACHE_SIZE={os.environ['AGNET_VIS_CACHE_SIZE']} "
          f"AGNET_DISC_VIS_CACHE_SIZE={os.getenv('AGNET_DISC_VIS_CACHE_SIZE', '<unset>')}")

    pointer = create_agp_model(args.embedding_size, args.hidden_size,
                               args.n_glimpses, args.tanh_exploration,
                               use_tanh=True, temperature=1.0)
    pc = torch.load(os.path.join(REPO_ROOT, args.pointer_checkpoint),
                    map_location=device, weights_only=False)
    sd = pc["model_state_dict"] if isinstance(pc, dict) and "model_state_dict" in pc else pc
    pointer.load_state_dict(sd, strict=False)
    pointer.to(device).eval()
    probe = _load_probe(os.path.join(REPO_ROOT, args.probe_checkpoint), device)

    # ---- selection split: dev_tune minus byte-identical train duplicates ---
    tune_all = _load_records(args.tune_traj)
    train_sigs = {_sig(r) for r in _load_records(args.train_traj)}
    tune = [r for r in tune_all if _sig(r) not in train_sigs]
    n_dropped = len(tune_all) - len(tune)
    print(f"[canonrule] dev_tune {len(tune_all)} - {n_dropped} train-duplicates "
          f"= {len(tune)} selection polygons")
    test = _load_records(args.test_traj)
    print(f"[canonrule] dev_test (reporting) {len(test)} polygons")
    if args.limit_tune:
        tune = tune[: args.limit_tune]
    if args.limit_test:
        test = test[: args.limit_test]

    # ---- sanity: convention equivalence on a handful of test polygons -----
    equiv = check_score_equivalence(pointer, probe, test, RULES,
                                    args.threshold, device, args.equiv_check_n)
    print(f"[canonrule] score-convention equivalence: "
          f"max|diff|={equiv['max_abs_coverage_diff']:.3e} over "
          f"{equiv['n_comparisons']} comparisons")

    # ---- PHASE 1: selection on dev_tune ----------------------------------
    print(f"\n[canonrule] === PHASE 1: dev_tune ({len(tune)}) ===", flush=True)
    tune_res = evaluate_split(pointer, probe, tune, RULES, args.threshold,
                              args.feasibility_gate, device, tag="devtune")
    _print_table(f"dev_tune fixed t={args.threshold}", tune_res["fixed_threshold"],
                 CONDITIONS)
    _print_table("dev_tune MATCHED BUDGET (m=|S_native|)",
                 tune_res["matched_budget"], CONDITIONS)

    eligible = [r for r in RULES if not tune_res["uniqueness"][r]["disqualified"]]
    disqualified = [r for r in RULES if tune_res["uniqueness"][r]["disqualified"]]
    if disqualified:
        print(f"[canonrule] DISQUALIFIED (tied argument, index tie-break would "
              f"not be roll-invariant): {disqualified}")
    if not eligible:
        raise SystemExit("all candidate rules disqualified")

    # Primary key: dev_tune matched-budget gate count. Secondary key (fixed in
    # advance, also dev_tune-only): mean matched-budget coverage; final key:
    # rule name, so the outcome is deterministic and never index/order luck.
    ranking = sorted(
        ((r, tune_res["matched_budget"][r]["n_ge_gate"],
          tune_res["matched_budget"][r]["mean_cov"]) for r in eligible),
        key=lambda kv: (-kv[1], -kv[2], kv[0]))
    winner = ranking[0][0]
    margin = (ranking[0][1] - ranking[1][1]) if len(ranking) > 1 else None
    criterion = ("argmax over eligible rules of #{exact-CGAL coverage >= "
                 f"{args.feasibility_gate}"
                 "} on dev_tune AT MATCHED BUDGET "
                 "(m = |S_native| per polygon); fixed-threshold columns are "
                 "reported but NOT used, because the start vertex is itself a "
                 "guard-count knob. Ties broken (in advance) by dev_tune "
                 "matched-budget mean coverage, then rule name.")
    print(f"\n[canonrule] SELECTION CRITERION: {criterion}")
    print(f"[canonrule] ranking (dev_tune matched-budget >= gate): {ranking}")
    print(f"[canonrule] WINNER = {winner}  (margin over runner-up: {margin})")

    # ---- PHASE 2: report the winner on dev_test --------------------------
    print(f"\n[canonrule] === PHASE 2: dev_test ({len(test)}), all rules ===",
          flush=True)
    test_res = evaluate_split(pointer, probe, test, RULES, args.threshold,
                              args.feasibility_gate, device, tag="devtest")
    _print_table(f"dev_test fixed t={args.threshold} [POST-SELECTION]",
                 test_res["fixed_threshold"], CONDITIONS)
    _print_table("dev_test MATCHED BUDGET [POST-SELECTION]",
                 test_res["matched_budget"], CONDITIONS)

    print(f"\n[canonrule] roll-invariance for selected rule {winner} on dev_test",
          flush=True)
    roll = check_roll_invariance(pointer, probe, test, winner, args.threshold,
                                 device)
    print(f"[canonrule] exact under cyclic roll: {roll['exact']}/{roll['checked']}")

    nat_ft = test_res["fixed_threshold"]["native"]
    win_ft = test_res["fixed_threshold"][winner]
    nat_mb = test_res["matched_budget"]["native"]
    win_mb = test_res["matched_budget"][winner]

    out = {
        "protocol": {
            "question": "does a canonical start rule chosen OFF-SPLIT match "
                        "native accuracy on dev_test?",
            "candidate_rules": RULES,
            "candidates_fixed_a_priori": True,
            "selection_split": args.tune_traj,
            "selection_split_dedup": (
                f"{len(tune_all)} records minus {n_dropped} byte-identical "
                f"train duplicates (sha1 of float64 coordinate bytes) = "
                f"{len(tune)}"),
            "reporting_split": args.test_traj,
            "selection_criterion": criterion,
            "threshold": args.threshold,
            "gate": args.feasibility_gate,
            "coverage": "exact CGAL (utils.evaluate_polygon_visibility_numpy_wo_gt)",
            "pointer_checkpoint": args.pointer_checkpoint,
            "probe_checkpoint": args.probe_checkpoint,
            "tie_policy": "a rule whose argument ties on ANY polygon is "
                          "DISQUALIFIED; index tie-breaks are not roll-"
                          "invariant and would destroy the tested property",
        },
        "equivalence_check": equiv,
        "dev_tune": tune_res,
        "selection": {
            "criterion": criterion,
            "eligible_rules": eligible,
            "disqualified_rules": disqualified,
            "ranking_matched_budget": [{"rule": r, "n_ge_gate": c,
                                        "mean_cov": mc}
                                       for r, c, mc in ranking],
            "ranking_fixed_threshold": [
                {"rule": r, "n_ge_gate": tune_res["fixed_threshold"][r]["n_ge_gate"]}
                for r in sorted(eligible,
                                key=lambda x: -tune_res["fixed_threshold"][x]["n_ge_gate"])],
            "winner": winner,
            "margin_over_runner_up": margin,
        },
        "dev_test_selected_rule": {
            "rule": winner,
            "label": "PRIMARY RESULT -- rule chosen on dev_tune only",
            "native_baseline_fixed_t": nat_ft,
            "selected_fixed_t": win_ft,
            "native_baseline_matched": nat_mb,
            "selected_matched": win_mb,
            "delta_fixed_t": win_ft["n_ge_gate"] - nat_ft["n_ge_gate"],
            "delta_matched": win_mb["n_ge_gate"] - nat_mb["n_ge_gate"],
            "roll_invariance": roll,
        },
        "dev_test_all_rules_post_selection": {
            "label": "REPORTED AFTER SELECTION FOR TRANSPARENCY -- NOT USED TO "
                     "SELECT. The headline conclusion rests only on "
                     f"{winner}, chosen on dev_tune.",
            "fixed_threshold": test_res["fixed_threshold"],
            "matched_budget": test_res["matched_budget"],
            "uniqueness": test_res["uniqueness"],
            "sanity": test_res["sanity"],
        },
        "dev_test_full": test_res,
        "elapsed_sec": round(time.time() - t_start, 1),
    }

    out_path = os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    augment(args)          # appends the dev_tune size-sensitivity block

    print(f"\n=== HEADLINE ===")
    print(f"selected off-split (dev_tune, matched budget): {winner}")
    print(f"dev_test fixed t={args.threshold}: {winner} "
          f"{win_ft['n_ge_gate']}/{win_ft['n_polygons']} vs native "
          f"{nat_ft['n_ge_gate']}/{nat_ft['n_polygons']} "
          f"(mean|S| {win_ft['mean_n_guards']} vs {nat_ft['mean_n_guards']})")
    print(f"dev_test matched budget: {winner} "
          f"{win_mb['n_ge_gate']}/{win_mb['n_polygons']} vs native "
          f"{nat_mb['n_ge_gate']}/{nat_mb['n_polygons']}")
    print(f"roll-invariance exact: {roll['exact']}/{roll['checked']}")
    print(f"[canonrule] wrote {out_path}  ({out['elapsed_sec']}s)")


if __name__ == "__main__":
    main()
