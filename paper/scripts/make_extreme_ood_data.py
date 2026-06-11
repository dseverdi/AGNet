"""Curate paper/data/extreme_ood.json from the raw extreme-OOD eval results.

Reads results/v3/setpred_extreme_ood_{seed1234,seed11,seed22,seed33,no_encoder}.json
(produced by eval_extreme_ood.sh on the `large` split, 285 polygons, n=600-2250,
exact-CGAL coverage) and emits the aggregates consumed by build_tables.tab_large().

Honest-reporting choices baked in here:
  * Coverage, min-coverage, feasibility count, and |S|/n are over ALL 285 polygons.
  * |S|/OPT is deliberately omitted: only 206/285 polygons have an ILP optimum (the
    79 largest, n=800-2250, lack one), so a per-285 ratio would be biased toward the
    easier 600-vertex-dominated subset. This matches paper.tex's stated policy of
    reporting coverage only on `large`.
  * Probe rows are the four seeds {1234, 11, 22, 33} as mean +/- std (4-sample seed
    std, NOT a per-polygon CI). The probe's min-Cov is the mean +/- std of the four
    per-seed worst-polygon coverages.
  * The no-encoder ablation is a single run.

Run: python paper/scripts/make_extreme_ood_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "v3"
OUT = Path(__file__).resolve().parents[1] / "data" / "extreme_ood.json"

SEEDS = ["seed1234", "seed11", "seed22", "seed33"]
# No-encoder ablation seeds: "no_encoder" is the seed-1234 run; the rest are
# the no_encoder_seed{11,22,33} retrainings (only those present are aggregated).
NE_SEEDS = ["no_encoder", "no_encoder_seed11", "no_encoder_seed22", "no_encoder_seed33"]
T_KEY = "t=0.2|K=1"


def _load(variant: str) -> dict:
    return json.loads((RESULTS / f"setpred_extreme_ood_{variant}.json").read_text())


def _exists(variant: str) -> bool:
    return (RESULTS / f"setpred_extreme_ood_{variant}.json").exists()


def main() -> None:
    d0 = _load("seed1234")
    sd = d0["seed"]
    n_total = d0["n_polygons"]

    # Probe: four seeds at t=0.20.
    covs, sns, mins, belows = [], [], [], []
    for v in SEEDS:
        c = _load(v)["cells"][T_KEY]
        covs.append(c["cov"])
        sns.append(c["chv"])
        mins.append(c["dist"]["cov_min"])
        belows.append(c["dist"]["not_covered_ge_095"])

    # No-encoder ablation: aggregate over whatever seeds are present (mean±std),
    # mirroring the full-probe row. Falls back to a single value if only the
    # seed-1234 run exists.
    ne_present = [v for v in NE_SEEDS if _exists(v)]
    ne_covs, ne_sns, ne_mins, ne_belows = [], [], [], []
    for v in ne_present:
        c = _load(v)["cells"][T_KEY]
        ne_covs.append(c["cov"])
        ne_sns.append(c["chv"])
        ne_mins.append(c["dist"]["cov_min"])
        ne_belows.append(c["dist"]["not_covered_ge_095"])

    out = {
        "note": (
            "Extreme-OOD evaluation on the `large` split (285 polygons, "
            "n=600-2250; trained on n<=198). Coverage is exact CGAL over all 285; "
            "min-Cov, feasibility count, and |S|/n are over all 285. |S|/OPT is "
            "omitted because only 206/285 polygons have an ILP optimum (the 79 "
            "largest lack one), per the paper's policy of reporting coverage only "
            "on `large`. Probe rows are the four seeds {1234,11,22,33} as mean+/-std; "
            "the probe min-Cov is the mean+/-std of the four per-seed worst-polygon "
            "coverages. The no-encoder ablation is aggregated the same way over "
            "whatever no_encoder seeds are present. "
            "Sources: results/v3/setpred_extreme_ood_*.json."
        ),
        "n_polygons": n_total,
        "n_seeds": len(SEEDS),
        "t": 0.20,
        "seed": {
            "cov": sd["cov"],
            "S_n": sd["chv"],
            "cov_min": sd["dist"]["cov_min"],
            "n_below_095": sd["dist"]["not_covered_ge_095"],
        },
        "probe_t020": {
            "cov_mean": float(np.mean(covs)),
            "cov_std": float(np.std(covs)),
            "S_n_mean": float(np.mean(sns)),
            "S_n_std": float(np.std(sns)),
            "cov_min_mean": float(np.mean(mins)),
            "cov_min_std": float(np.std(mins)),
            "n_below_095_mean": float(np.mean(belows)),
            "n_below_095_std": float(np.std(belows)),
            "per_seed_n_below_095": dict(zip(SEEDS, belows)),
            "per_seed_cov_min": dict(zip(SEEDS, [float(m) for m in mins])),
        },
        "no_encoder_t020": {
            "n_seeds": len(ne_present),
            "cov_mean": float(np.mean(ne_covs)),
            "cov_std": float(np.std(ne_covs)),
            "S_n_mean": float(np.mean(ne_sns)),
            "S_n_std": float(np.std(ne_sns)),
            "cov_min_mean": float(np.mean(ne_mins)),
            "cov_min_std": float(np.std(ne_mins)),
            "n_below_095_mean": float(np.mean(ne_belows)),
            "n_below_095_std": float(np.std(ne_belows)),
            "per_seed_n_below_095": dict(zip(ne_present, ne_belows)),
            "per_seed_cov_min": dict(zip(ne_present, [float(m) for m in ne_mins])),
        },
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT.relative_to(REPO)}")
    print(f"  seed:       cov={out['seed']['cov']:.4f} min={out['seed']['cov_min']:.4f} "
          f"below={out['seed']['n_below_095']}/{n_total}")
    p = out["probe_t020"]
    print(f"  probe(4sd): cov={p['cov_mean']:.4f}±{p['cov_std']:.4f} "
          f"min={p['cov_min_mean']:.4f}±{p['cov_min_std']:.4f} "
          f"below={p['n_below_095_mean']:.1f}±{p['n_below_095_std']:.1f}")
    q = out["no_encoder_t020"]
    print(f"  no-enc({q['n_seeds']}sd): cov={q['cov_mean']:.4f}±{q['cov_std']:.4f} "
          f"min={q['cov_min_mean']:.4f}±{q['cov_min_std']:.4f} "
          f"below={q['n_below_095_mean']:.1f}±{q['n_below_095_std']:.1f}/{n_total}")


if __name__ == "__main__":
    main()
