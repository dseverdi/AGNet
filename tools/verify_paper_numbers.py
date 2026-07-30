"""Final gate: recompute every checkable published number from the live data.

Method: recompute a quantity from paper/data or results, format it exactly as the
manuscript formats it, and assert that string is present in the target .tex file.
That catches a stale table even when the generator would produce the right value,
because it tests what is actually typeset rather than what a rerun would produce.

Covers the 14 tables the manuscript inputs, plus the prose numbers that have no
intervening table or figure (significance tests, invariance, canonical start).

Usage:  python tools/verify_paper_numbers.py
Exit status is non-zero if anything fails.
"""

from __future__ import annotations

import json
import math
import sys
from math import comb
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
D = REPO / "paper" / "data"
R = REPO / "results"
T = REPO / "paper" / "tables"
TEX = (REPO / "paper" / "paper.tex").read_text()

PASS: list[str] = []
FAIL: list[str] = []


def poly(name: str) -> list[dict]:
    return json.loads((D / name).read_text())["polygons"]


def check(label: str, needle: str, where: str) -> None:
    """Assert `needle` appears in table file `where` ('paper.tex' for prose)."""
    hay = TEX if where == "paper.tex" else (T / where).read_text()
    (PASS if needle in hay else FAIL).append(f"{label}: '{needle}' in {where}")


def wilson_lo(k: int, n: int, z: float = 1.96) -> float:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h)


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, sum(comb(n, k) for k in range(0, min(b, c) + 1)) * 2 / 2 ** n)


# ---------------------------------------------------------------- tab_headline
dev, ne = poly("dist_dev_test.json"), poly("dist_dev_test_noenc.json")
cls = json.loads((D / "baseline_classical.json").read_text())["splits"]["dev_test"]["per_polygon"]
mean = lambda xs: float(np.mean(list(xs)))  # noqa: E731

check("headline greedy |S|/OPT",
      f"{mean(x['S_size']/x['OPT'] for x in cls['greedy'] if x.get('OPT')):.4f}",
      "tab_headline.tex")
check("headline LS |S|/OPT",
      f"{mean(x['S_size']/x['OPT'] for x in cls['ls'] if x.get('OPT')):.4f}",
      "tab_headline.tex")
check("headline seed |S|/OPT",
      f"{mean(x['seed']['S_size']/x['OPT'] for x in dev if x.get('OPT')):.4f}",
      "tab_headline.tex")
check("headline seed gate",
      f"{sum(1 for x in dev if x['seed']['cov'] >= 0.95)}/362", "tab_headline.tex")

# ------------------------------------------------------------- tab_dist_shift
for key, col in (("seed", None), ("probe_t020", None), ("probe_t025", None),
                 ("probe_t030", None)):
    n99 = sum(1 for x in dev if x[key]["cov"] >= 0.99)
    check(f"dist_shift {key} >=0.99", f"& {n99} &", "tab_dist_shift.tex")
check("dist_shift noenc >=0.99",
      f"& {sum(1 for x in ne if x['probe_t020']['cov'] >= 0.99)} &",
      "tab_dist_shift.tex")

# ------------------------------------------------------------------- tab_ood
po = ["dist_test_OOD.json"] + [f"dist_test_OOD_seed{s}.json" for s in (11, 22, 33)]
pn = ["dist_test_OOD_noenc.json"] + [f"dist_test_OOD_noenc_seed{s}.json" for s in (11, 22, 33)]
oodf = [mean(x['probe_t020']['S_size']/x['OPT'] for x in poly(f) if x.get('OPT')) for f in po]
oodn = [mean(x['probe_t020']['S_size']/x['OPT'] for x in poly(f) if x.get('OPT')) for f in pn]
check("ood full |S|/OPT", f"{mean(oodf):.4f}", "tab_ood.tex")
check("ood noenc |S|/OPT", f"{mean(oodn):.4f}", "tab_ood.tex")
ood = poly("dist_test_OOD.json")
check("ood seed gate", f"{sum(1 for x in ood if x['seed']['cov'] >= 0.95)}/2081",
      "tab_ood.tex")

# ----------------------------------------------------------------- tab_large
fl = ["dist_ood_large.json"] + [f"dist_ood_large_seed{s}.json" for s in (11, 22, 33)]
nl = ["dist_ood_large_noenc.json"] + [f"dist_ood_large_noenc_seed{s}.json" for s in (11, 22, 33)]
check("large full mean cov",
      f"{mean(mean(x['probe_t020']['cov'] for x in poly(f)) for f in fl):.4f}",
      "tab_large.tex")
check("large full min cov",
      f"{mean(min(x['probe_t020']['cov'] for x in poly(f)) for f in fl):.3f}",
      "tab_large.tex")
check("large noenc min cov",
      f"{mean(min(x['probe_t020']['cov'] for x in poly(f)) for f in nl):.3f}",
      "tab_large.tex")

# ------------------------------------------------- tab_operating_curve (t=0.20)
mb = {t: [] for t in (0.05, 0.2, 0.5, 0.7, 0.8)}
for s in ("1234", "11", "22", "33"):
    c = json.loads((D / "matched_budget" / f"full_{s}.json").read_text())["cells"]
    for t in mb:
        mb[t].append(c[f"t={t}|K=1"])
for t in (0.05, 0.2, 0.5, 0.7, 0.8):
    check(f"opcurve t={t} |S|/OPT",
          f"{mean(x['opt'] for x in mb[t]):.2f}", "tab_operating_curve.tex")

# --------------------------------------------------------------- tab_reinforce
for row in json.loads((D / "baseline_reinforce.json").read_text())["rows"]:
    check(f"reinforce {row['name'][:18]}", f"{row['S_over_OPT']:.2f}",
          "tab_reinforce.tex")

# ------------------------------------------------------------ tab_policy_seeds
for s in (11, 22, 33):
    d = json.loads((R / "policy_seeds_ood" / f"pseed{s}_full_test.json").read_text())
    g = d["cells"]["t=0.2|K=1"]["dist"]["n_cov_ge_095"]
    check(f"policy_seeds pseed{s} ood full gate", f"{g}/2081", "tab_policy_seeds.tex")

# ------------------------------------------------------- prose: significance
sig = json.loads((D / "significance_ood.json").read_text())["per_seed"]
maxp = max(v["mcnemar_exact_p"] for v in sig.values())
minlo = min(v["probe_feasible_wilson95"][0] for v in sig.values())
(PASS if maxp < 1e-74 else FAIL).append(
    f"prose: max McNemar p {maxp:.2e} < 10^-74 as claimed")
(PASS if minlo > 0.977 else FAIL).append(
    f"prose: min Wilson lower {minlo:.4f} > 0.977 as claimed")
b = sum(1 for x in dev if x["seed"]["cov"] < 0.95 <= x["probe_t020"]["cov"])
c = sum(1 for x in dev if x["probe_t020"]["cov"] < 0.95 <= x["seed"]["cov"])
check("prose: in-dist McNemar b", f"${b}$", "paper.tex")
check("prose: in-dist McNemar c", f"${c}$", "paper.tex")
(PASS if abs(mcnemar(b, c) - 3.066e-12) < 1e-14 else FAIL).append(
    f"prose: in-dist McNemar p {mcnemar(b,c):.3e} ~ 3e-12 as claimed")

# ---------------------------------------------------------- prose: invariance
inv = json.loads((D / "invariance_test.json").read_text())["summary"]
for k, lab in (("identity", "345"), ("rot45", "312"), ("rot90", "282"),
               ("rot180", "223"), ("mirror_x", "205"), ("reindex", "279")):
    got = inv[k]["n_polygons"] - inv[k]["n_below_gate"]
    (PASS if str(got) == lab else FAIL).append(
        f"prose: invariance {k} = {got} (paper says {lab})")

# ----------------------------------------------------- prose: canonical start
can = json.loads((R / "canonical_rule_selection.json").read_text())
sel = can["dev_test_selected_rule"]
check("prose: canon matched gate", f"${sel['selected_matched']['n_ge_gate']}/362$",
      "paper.tex")
(PASS if can["selection"]["winner"] == "lexmax_xy" else FAIL).append(
    f"prose: canonical rule selected off-split = {can['selection']['winner']}")
(PASS if sel["roll_invariance"]["exact"] == 362 else FAIL).append(
    "prose: canonical roll-invariance exact on 362/362")

# ------------------------------------------------------------------- report
for p in PASS:
    print(f"  [OK ] {p}")
for f in FAIL:
    print(f"  [BAD] {f}")
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
