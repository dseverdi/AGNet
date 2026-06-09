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
T_KEY = "t=0.2|K=1"


def _load(variant: str) -> dict:
    return json.loads((RESULTS / f"setpred_extreme_ood_{variant}.json").read_text())


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

    ne = _load("no_encoder")["cells"][T_KEY]

    out = {
        "note": (
            "Extreme-OOD evaluation on the `large` split (285 polygons, "
            "n=600-2250; trained on n<=198). Coverage is exact CGAL over all 285; "
            "min-Cov, feasibility count, and |S|/n are over all 285. |S|/OPT is "
            "omitted because only 206/285 polygons have an ILP optimum (the 79 "
            "largest lack one), per the paper's policy of reporting coverage only "
            "on `large`. Probe rows are the four seeds {1234,11,22,33} as mean+/-std; "
            "the probe min-Cov is the mean+/-std of the four per-seed worst-polygon "
            "coverages. No-encoder is a single run. "
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
            "cov": ne["cov"],
            "S_n": ne["chv"],
            "cov_min": ne["dist"]["cov_min"],
            "n_below_095": ne["dist"]["not_covered_ge_095"],
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
    print(f"  no-encoder: cov={q['cov']:.4f} min={q['cov_min']:.4f} below={q['n_below_095']}/{n_total}")


if __name__ == "__main__":
    main()
