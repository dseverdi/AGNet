"""Make the dataset partition proper: remove train/eval leakage from the
held-out evaluation artifacts, WITHOUT retraining.

Background. Splits are directory-defined ($DATASET_PATH/<split>/) and the source
directories overlapped, so identical polygons (byte-identical coordinates) leaked
between train and the held-out sets:
    train ∩ test = 5,  train ∩ dev = 17,  train ∩ ood = 16,
    test ∩ ood = 5,    dev ∩ ood = 5,     dev ∩ test = 0 (clean).
The frozen policy/probe are NOT retrained; instead we exclude the leaked
instances from the *evaluation* sets so reported metrics are on genuinely
held-out polygons. Identity is by byte-exact coordinate hash; within each eval
split polygon names are unique, so we drop by name.

What it does (idempotent):
  * recompute the drop-sets from the trajectory pickles (authoritative),
  * filter every per-polygon eval JSON (dist_dev_test*.json, dist_test_OOD*.json)
    and baseline_classical.json in place,
  * regenerate setpred_dev_test.json (pre-aggregated cells consumed by
    tab_dist_shift / tab_pareto) from the cleaned per-polygon data,
  * write cleaned trajectory pickles (*_clean.pkl) for reproducibility.

After running this, regenerate downstream artifacts (no model runs needed):
  python paper/scripts/build_paper_data.py multi_seed_summary
  python paper/scripts/build_tables.py
  python paper/scripts/build_figures.py
  python paper/scripts/compute_significance.py   # if present

Usage:
  python tools/dedup_partitions.py
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
PDATA = REPO / "paper" / "data"

TRAJ = {
    "train": DATA / "ls_trajectories_train.pkl",
    "pool":  DATA / "ls_trajectories_dev.pkl",        # dev+test combined (1224)
    "test":  DATA / "ls_trajectories_dev_test.pkl",   # in-distribution test (367)
    "ood":   DATA / "ls_trajectories_test.pkl",       # OOD eval (2107)
}


def _load(p: Path) -> list[dict]:
    with open(p, "rb") as f:
        d = pickle.load(f)
    return d["records"] if isinstance(d, dict) and "records" in d else d


def _sig(rec: dict) -> str:
    """Byte-exact coordinate signature."""
    return hashlib.sha1(np.asarray(rec["points"], dtype=np.float64).tobytes()).hexdigest()


def compute_drop_sets() -> tuple[set[str], set[str]]:
    """Return (test_drop_names, ood_drop_names) from the trajectory pickles."""
    train = _load(TRAJ["train"])
    pool = _load(TRAJ["pool"])
    test = _load(TRAJ["test"])
    ood = _load(TRAJ["ood"])

    train_sig = {_sig(r) for r in train}
    test_sig = {_sig(r) for r in test}
    test_sigset = set(test_sig)
    dev_sig = {_sig(r) for r in pool if _sig(r) not in test_sigset}  # true dev (857)

    test_drop = {r["name"] for r in test if _sig(r) in train_sig}
    ood_drop = {r["name"] for r in ood
                if _sig(r) in train_sig or _sig(r) in test_sig or _sig(r) in dev_sig}
    return test_drop, ood_drop


def _filter_json_polygons(path: Path, drop_names: set[str]) -> tuple[int, int]:
    """Drop polygons whose 'name' is in drop_names from a {'polygons': [...]} file."""
    if not path.exists():
        return (0, 0)
    d = json.loads(path.read_text())
    polys = d.get("polygons")
    if polys is None:
        return (0, 0)
    before = len(polys)
    d["polygons"] = [p for p in polys if p.get("name") not in drop_names]
    after = len(d["polygons"])
    path.write_text(json.dumps(d))
    return (before, after)


def _filter_baseline_classical(test_drop: set[str]) -> None:
    """Drop leaked test instances from baseline_classical.json dev_test per_polygon
    lists (greedy + ls). tab_headline re-aggregates these, so only the lists matter."""
    path = PDATA / "baseline_classical.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    pp = d.get("splits", {}).get("dev_test", {}).get("per_polygon", {})
    for key in ("greedy", "ls"):
        if key in pp:
            b = len(pp[key])
            pp[key] = [r for r in pp[key] if r.get("name") not in test_drop]
            print(f"  baseline_classical dev_test.{key}: {b} -> {len(pp[key])}")
    path.write_text(json.dumps(d, indent=2))


def _dist_block(records: list[dict], key: str) -> dict:
    covs = [r[key]["cov"] for r in records]
    return {
        "cov_mean": float(np.mean(covs)),
        "cov_min": float(np.min(covs)),
        "n_total": len(records),
        "n_cov_eq_1": sum(1 for c in covs if c >= 0.99995),
        "n_cov_ge_0999": sum(1 for c in covs if c >= 0.999),
        "n_cov_ge_099": sum(1 for c in covs if c >= 0.99),
        "n_cov_ge_095": sum(1 for c in covs if c >= 0.95),
    }


def _cell(records: list[dict], key: str) -> dict:
    cov = float(np.mean([r[key]["cov"] for r in records]))
    chv = float(np.mean([r[key]["S_size"] / r["n"] for r in records]))
    sopt = [r[key]["S_size"] / r["OPT"] for r in records if r.get("OPT")]
    opt = float(np.mean(sopt)) if sopt else None
    return {"cov": cov, "chv": chv, "opt": opt, "n": len(records),
            "dist": _dist_block(records, key)}


def regenerate_setpred_dev_test() -> None:
    """Rebuild paper/data/setpred_dev_test.json (cells consumed by tab_dist_shift /
    tab_pareto) from the cleaned per-polygon dist_dev_test.json."""
    src = PDATA / "dist_dev_test.json"
    polys = json.loads(src.read_text())["polygons"]
    out = {
        "n_polygons": len(polys),
        "seed": _cell(polys, "seed"),
        "cells": {
            "t=0.2|K=1": _cell(polys, "probe_t020"),
            "t=0.25|K=1": _cell(polys, "probe_t025"),
            "t=0.3|K=1": _cell(polys, "probe_t030"),
        },
        "note": "Regenerated by tools/dedup_partitions.py from cleaned "
                "dist_dev_test.json (train-leak instances removed).",
    }
    (PDATA / "setpred_dev_test.json").write_text(json.dumps(out))
    print(f"  regenerated setpred_dev_test.json (n={len(polys)})")


def write_clean_pickles(test_drop: set[str], ood_drop: set[str]) -> None:
    for label, path, drop in [("test", TRAJ["test"], test_drop),
                               ("ood", TRAJ["ood"], ood_drop)]:
        with open(path, "rb") as f:
            d = pickle.load(f)
        recs = d["records"] if isinstance(d, dict) else d
        kept = [r for r in recs if r["name"] not in drop]
        out = {"summary": (d.get("summary", {}) if isinstance(d, dict) else {}),
               "records": kept}
        out_path = path.with_name(path.stem + "_clean.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  wrote {out_path.name}: {len(recs)} -> {len(kept)}")


def main() -> None:
    test_drop, ood_drop = compute_drop_sets()
    print(f"[dedup] test drop: {len(test_drop)} | ood drop: {len(ood_drop)}")

    dev_test_files = ["dist_dev_test.json", "dist_dev_test_noenc.json",
                      "dist_dev_test_seed11.json", "dist_dev_test_seed22.json",
                      "dist_dev_test_seed33.json", "dist_dev_test_noenc_seed11.json",
                      "dist_dev_test_noenc_seed22.json", "dist_dev_test_noenc_seed33.json"]
    ood_files = ["dist_test_OOD.json", "dist_test_OOD_noenc.json",
                 "dist_test_OOD_seed11.json", "dist_test_OOD_seed22.json",
                 "dist_test_OOD_seed33.json", "dist_test_OOD_noenc_seed11.json",
                 "dist_test_OOD_noenc_seed22.json", "dist_test_OOD_noenc_seed33.json"]

    print("[dedup] filtering dev_test per-polygon JSONs (drop test-leaks):")
    for f in dev_test_files:
        b, a = _filter_json_polygons(PDATA / f, test_drop)
        if b:
            print(f"  {f}: {b} -> {a}")
    print("[dedup] filtering ood per-polygon JSONs (drop train/test/dev overlaps):")
    for f in ood_files:
        b, a = _filter_json_polygons(PDATA / f, ood_drop)
        if b:
            print(f"  {f}: {b} -> {a}")

    print("[dedup] baseline_classical.json:")
    _filter_baseline_classical(test_drop)
    print("[dedup] regenerate pre-aggregated setpred_dev_test.json:")
    regenerate_setpred_dev_test()
    print("[dedup] cleaned trajectory pickles:")
    write_clean_pickles(test_drop, ood_drop)
    print("[dedup] done. Now run: build_paper_data.py multi_seed_summary; "
          "build_tables.py; build_figures.py; compute_significance.py")


if __name__ == "__main__":
    main()
