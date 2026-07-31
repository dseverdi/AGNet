"""Matched-BUDGET view of the encoder-seed ablation (reviewer R2, point 6).

R2 objected that the headline 2x2 (tab_ablation) compares all four conditions at
one common threshold t=0.20 despite their different calibration and guard counts.
That objection is correct, and the leak-free data proves it: at fixed t the
0.99-gate failure count is perfectly monotone in |S|/n, so the fixed-t table
measures set size rather than representation.

This table removes the confound. Each condition is swept over the whole
threshold range t in [0.05, 0.80] so that the four cost curves overlap, and the
conditions are then compared at a matched guard budget |S|/OPT instead of a
matched threshold.

Panel (a) reports the swept curves, so the overlap is visible.
Panel (b) reads each arm's own curve at four common cost anchors, interpolating
within the arm, and reports the resulting 0.99-gate tail.

Aggregation is the four-seed mean over probe seeds {1234, 11, 22, 33} on the
362-polygon leak-free dev_test split, coverage scored by exact CGAL. Source:
paper/data/matched_budget/<arm>_<seed>.json (16 sweeps).

Output: paper/tables/tab_ablation_thresholds.tex
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SWEEP = REPO / "paper" / "data" / "matched_budget"
TABLE = REPO / "paper" / "tables" / "tab_ablation_thresholds.tex"

SEEDS = ("1234", "11", "22", "33")
ARMS = [
    ("full", r"full"),
    ("noseed", r"no-seed"),
    ("noenc", r"no-encoder"),
    ("coords", r"coords-only"),
]

# Panel (a): a common grid spanning every arm's useful range. The arms spend very
# differently at any single t, which is the whole point of the table.
GRID = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80]
# Panel (b): cost anchors inside the band where the encoder-bearing arms overlap.
# 1.95 is the highest anchor all four `full` seeds reach (their curves top out at
# 1.97-2.04 at t=0.05), so every cell below is a four-seed mean, never three.
ANCHORS = [1.95, 1.7, 1.5, 1.3]


def curve(arm: str, seed: str) -> dict:
    """t -> (|S|/n, |S|/OPT, mean cov, #Cov<0.99, min cov) per arm and probe seed."""
    d = json.load(open(SWEEP / f"{arm}_{seed}.json"))
    out = {}
    for key, c in d["cells"].items():
        if not key.endswith("|K=1"):
            continue
        t = float(key.split("|")[0].split("=")[1])
        out[t] = (c["chv"], c["opt"], c["cov"],
                  float(c["dist"]["n_total"] - c["dist"]["n_cov_ge_099"]),
                  c["dist"]["cov_min"])
    return out


CURVES = {(a, s): curve(a, s) for a, _ in ARMS for s in SEEDS}


def mean_at(arm: str, t: float, idx: int) -> float:
    return float(np.mean([CURVES[(arm, s)][t][idx] for s in SEEDS]))


def std_at(arm: str, t: float, idx: int) -> float:
    """Population std over the four probe seeds, so panel (a) reports the same
    mean +- std as panel (b) rather than presenting a 4-seed mean as a point."""
    return float(np.std([CURVES[(arm, s)][t][idx] for s in SEEDS]))


def tail_at_cost(arm: str, seed: str, target: float):
    """#Cov<0.99 where this arm's OWN curve spends |S|/OPT = target.

    Interpolates within the arm, so no arm is ever read at another arm's
    threshold. Returns None when the arm's curve never reaches that cost.
    """
    pts = sorted(CURVES[(arm, seed)].values(), key=lambda v: v[1])
    xs = [v[1] for v in pts]
    ys = [v[3] for v in pts]
    if target < xs[0] or target > xs[-1]:
        return None
    for i in range(len(xs) - 1):
        if xs[i] <= target <= xs[i + 1]:
            if xs[i + 1] == xs[i]:
                return ys[i]
            w = (target - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + w * (ys[i + 1] - ys[i])
    return None


PSEEP = REPO / "paper" / "data" / "matched_budget_pseeds"
PSEEDS = ["11", "22", "33"]
TAILS = [30, 50, 76, 100]


def pseed_curve(arm: str, s: str) -> dict:
    d = json.load(open(PSEEP / f"{arm}_pseed{s}.json"))
    out = {}
    for k, c in d["cells"].items():
        if not k.endswith("|K=1"):
            continue
        out[float(k.split("|")[0].split("=")[1])] = (
            c["chv"], c["opt"], c["cov"],
            float(c["dist"]["n_total"] - c["dist"]["n_cov_ge_099"]))
    return out


def _cost_at_tail(pts, target):
    """|S|/OPT at which this arm's own curve leaves `target` polygons below 0.99."""
    p = sorted(pts, key=lambda v: v[3])
    ys, xs = [v[3] for v in p], [v[1] for v in p]
    if target < ys[0] or target > ys[-1]:
        return None
    for i in range(len(ys) - 1):
        if ys[i] <= target <= ys[i + 1]:
            if ys[i + 1] == ys[i]:
                return xs[i]
            return xs[i] + (target - ys[i]) / (ys[i + 1] - ys[i]) * (xs[i + 1] - xs[i])
    return None


def encoder_cost_multiple():
    """policy -> {tail: noenc/full guard-cost multiple at matched tail}.

    Policy 1234 averages its four probe seeds (the sweep in matched_budget/);
    policies 11/22/33 have one probe per arm (matched_budget_pseeds/).
    """
    out = {}
    row = {}
    for t in TAILS:
        f = [_cost_at_tail(CURVES[("full", s)].values(), t) for s in SEEDS]
        n = [_cost_at_tail(CURVES[("noenc", s)].values(), t) for s in SEEDS]
        row[t] = (float(np.mean(n)) / float(np.mean(f))
                  if all(x is not None for x in f + n) else None)
    out["1234"] = row
    for s in PSEEDS:
        fc, nc = pseed_curve("full", s).values(), pseed_curve("noenc", s).values()
        out[s] = {t: (lambda a, b: b / a if (a and b) else None)(
            _cost_at_tail(fc, t), _cost_at_tail(nc, t)) for t in TAILS}
    return out


def main() -> None:
    ncol = 1 + 2 * len(ARMS)
    L = ["% auto-generated by build_matched_threshold_table.py"]
    L.append(r"\begin{tabular}{l" + "rr" * len(ARMS) + "}")
    L.append(r"  \toprule")
    L.append("  & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{lab}}}" for _, lab in ARMS) + r" \\")
    L.append("  " + " ".join(
        rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(ARMS))))
    L.append(r"  $t$ & " + " & ".join(
        [r"$|S|/\OPT$ & $\#$fail"] * len(ARMS)) + r" \\")
    L.append(r"  \midrule")
    L.append(rf"  \multicolumn{{{ncol}}}{{l}}"
             r"{\emph{(a) swept operating curves}} \\")
    for t in GRID:
        cells = []
        for arm, _ in ARMS:
            cells.append(f"{mean_at(arm, t, 1):.2f}")
            cells.append(f"${mean_at(arm, t, 3):.0f} \\pm {std_at(arm, t, 3):.0f}$")
        L.append(f"  {t:.2f} & " + " & ".join(cells) + r" \\")
    L.append(r"  \midrule")
    L.append(rf"  \multicolumn{{{ncol}}}{{l}}"
             r"{\emph{(b) $\#$fail at matched guard budget} $|S|/\OPT$} \\")
    for a in ANCHORS:
        cells = []
        for arm, _ in ARMS:
            vals = [tail_at_cost(arm, s, a) for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if not vals:
                cells.append(r"\multicolumn{2}{c}{---}")
            else:
                cells.append(rf"\multicolumn{{2}}{{c}}{{${np.mean(vals):.0f}"
                             rf" \pm {np.std(vals):.0f}$}}")
        L.append(f"  {a:.2f} & " + " & ".join(cells) + r" \\")
    # Panel (c): does the matched-budget ordering survive a change of policy?
    mult = encoder_cost_multiple()
    if any(any(v is not None for v in r.values()) for r in mult.values()):
        L.append(r"  \midrule")
        # One number per policy, not a 4x4 grid: within a policy the multiple is
        # near-constant across tail levels (spread <= 0.09), so the grid spent 16
        # cells -- three of them empty -- to convey four values. Reported as the
        # range over the reachable tail levels {30, 50, 76, 100}.
        L.append(rf"  \multicolumn{{{ncol}}}{{l}}{{\emph{{(c) extra guards the"
                 r" no-encoder probe needs to leave the same tail as the full probe}} \\")
        L.append(r"  policy seed & " + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{{lab}}}" for lab in
            ("1234$^{\\ast}$", "11", "22", "33")) + r" \\")
        cells = []
        for p in ["1234"] + PSEEDS:
            vals = [mult[p][t] for t in TAILS if mult[p][t] is not None]
            cells.append(
                rf"\multicolumn{{2}}{{c}}{{$+{(min(vals) - 1) * 100:.0f}$--"
                rf"${(max(vals) - 1) * 100:.0f}\%$}}"
                if vals else r"\multicolumn{2}{c}{---}")
        L.append(r"  extra guards & " + " & ".join(cells) + r" \\")
    L.append(r"  \bottomrule")
    L.append(r"\end{tabular}")
    TABLE.write_text("\n".join(L) + "\n")
    print(f"wrote {TABLE}\n")
    print("\n".join(L))

    print("\n--- panel (c): encoder cost multiple by policy ---")
    allv = []
    for p in ["1234"] + PSEEDS:
        vs = [mult[p][t] for t in TAILS if mult[p][t] is not None]
        allv += vs
        print(f"  policy {p:>4}: " + "  ".join(
            f"{mult[p][t]:.2f}x" if mult[p][t] is not None else " n/a" for t in TAILS))
    if allv:
        print(f"  across all policies and tails: {min(allv):.2f}x - {max(allv):.2f}x")

    print("\n--- matched-cost tail, four-seed mean (for the prose) ---")
    for a in ANCHORS:
        row = {}
        for arm, _ in ARMS:
            v = [x for x in (tail_at_cost(arm, s, a) for s in SEEDS)
                 if x is not None]
            row[arm] = (float(np.mean(v)), len(v)) if v else (None, 0)
        f = row["full"][0]
        ne = row["noenc"][0]
        rat = f"{ne / f:.2f}x" if f and ne else "n/a"
        body = "  ".join(
            f"{k} {row[k][0]:.1f} (n={row[k][1]})" if row[k][0] is not None
            else f"{k} unreachable" for k in row)
        print(f"  |S|/OPT {a}: {body}   noenc/full {rat}")

    print("\n--- per-seed strict ordering full < noseed < noenc < coords ---")
    for a in ANCHORS:
        ok = tot = 0
        for s in SEEDS:
            v = [tail_at_cost(arm, s, a) for arm, _ in ARMS]
            if any(x is None for x in v):
                continue
            tot += 1
            ok += int(v[0] < v[1] < v[2] < v[3])
        print(f"  |S|/OPT {a}: {ok}/{tot} seeds")


if __name__ == "__main__":
    main()
