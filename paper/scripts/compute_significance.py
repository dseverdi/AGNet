"""Significance statistics for the OOD feasibility improvement (policy seed vs probe).

Reads the stored per-polygon results (paper/data/dist_test_OOD*.json, one file per
probe seed; the policy seed column is identical across files) and computes, per probe
seed at t=0.20:

  * failure counts below the 0.95 feasibility gate (policy vs probe),
  * exact McNemar test on the paired per-polygon outcomes (the right test here:
    both methods are evaluated on the SAME 2107 polygons),
  * Wilson 95% CI for each feasibility proportion.

Writes paper/data/significance_ood.json. Every number in the paper's significance
sentence must come from this file.

Run: /home/dseverdi/.conda/envs/MLAG/bin/python paper/scripts/compute_significance.py
"""
from __future__ import annotations

import json
from pathlib import Path

import math

from scipy.stats import binomtest, norm


def proportion_confint(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half

PAPER_DATA = Path(__file__).resolve().parents[1] / "data"
GATE = 0.95
SEED_FILES = {
    "1234": "dist_test_OOD.json",
    "11": "dist_test_OOD_seed11.json",
    "22": "dist_test_OOD_seed22.json",
    "33": "dist_test_OOD_seed33.json",
}


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar p-value from the two discordant-cell counts."""
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue


def main() -> None:
    out = {"gate": GATE, "threshold": "t=0.20", "per_seed": {}}
    for seed, fname in SEED_FILES.items():
        polys = json.loads((PAPER_DATA / fname).read_text())["polygons"]
        n = len(polys)
        seed_fail = [p["seed"]["cov"] < GATE for p in polys]
        probe_fail = [p["probe_t020"]["cov"] < GATE for p in polys]
        n_seed_fail = sum(seed_fail)
        n_probe_fail = sum(probe_fail)
        # discordant cells: b = policy fails & probe passes; c = policy passes & probe fails
        b = sum(1 for s, p in zip(seed_fail, probe_fail) if s and not p)
        c = sum(1 for s, p in zip(seed_fail, probe_fail) if not s and p)
        p_mcnemar = mcnemar_exact(b, c)
        lo_s, hi_s = proportion_confint(n - n_seed_fail, n)
        lo_p, hi_p = proportion_confint(n - n_probe_fail, n)
        out["per_seed"][seed] = {
            "n": n,
            "policy_failures": n_seed_fail,
            "probe_failures": n_probe_fail,
            "discordant_policy_only": b,
            "discordant_probe_only": c,
            "mcnemar_exact_p": p_mcnemar,
            "policy_feasible_wilson95": [lo_s, hi_s],
            "probe_feasible_wilson95": [lo_p, hi_p],
        }
        print(f"seed {seed}: policy {n_seed_fail}/{n} vs probe {n_probe_fail}/{n} "
              f"(b={b}, c={c}) McNemar exact p={p_mcnemar:.3e} "
              f"probe Wilson95=[{lo_p:.4f},{hi_p:.4f}]")

    worst_p = max(v["mcnemar_exact_p"] for v in out["per_seed"].values())
    out["max_p_across_seeds"] = worst_p
    print(f"max p across seeds: {worst_p:.3e}")
    (PAPER_DATA / "significance_ood.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {PAPER_DATA / 'significance_ood.json'}")


if __name__ == "__main__":
    main()
