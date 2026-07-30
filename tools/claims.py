"""Registry of every quantitative claim in the manuscript.

WHY THIS EXISTS. Over the course of correcting this paper, every defect was found by a
different ad-hoc method: a stale table by `git diff`, a wrong p-value by asking what a
file supported, a conflated editor result by reading a sentence and tracing both numbers,
a broken measurement harness by noticing its no-op case disagreed with a published value.
Ad-hoc discovery has no completeness guarantee. This registry, together with
tools/verify_paper_consistency.py, replaces the judgement "did I check everything?" with a
computed property that fails closed.

TWO HARD RULES, both learned from actual failures:

  1. RECOMPUTE FROM RAW. Every claim is recomputed from per-polygon `dist_*.json` records
     or an equivalent primary source -- never from a pre-aggregated file such as
     multi_seed_summary.json, and never by calling build_tables.py's helpers. Reusing the
     paper's own aggregation would only demonstrate self-consistency. This is what caught
     significance_ood.json being six times off on a published p-value.

  2. EVERY CLAIM NAMES ITS POPULATION AND RUN. `n` and `run` are mandatory. The manuscript
     once paired a guard-count reduction from a 5-polygon smoke run with a recovery rate
     from a 300-polygon run and presented them as one result; claims in the same `group`
     are checked for agreement on both fields.

A claim is either
  * exact    -- `typeset` must appear verbatim in `target`, or
  * interval -- `value` must lie in `interval`, used for verbal magnitude claims
                ("nearly twice", "an order of magnitude") so that vague wording is checked
                rather than exempted.
"""

from __future__ import annotations

import json
import math
import pickle
import re
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass, field
from math import comb
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
D = REPO / "paper" / "data"
R = REPO / "results"

# The de-leak. Any source older than this must be on PROBE_INDEPENDENT below.
DELEAK = "2026-07-28"

# Sources predating the de-leak that cannot carry the leak, with the reason.
PROBE_INDEPENDENT = {
    "paper/data/po_agp_training.json": "policy checkpoints only, no probe",
    "paper/data/baseline_reinforce.json": "REINFORCE vs PO/BT policies, no probe",
    "paper/data/baseline_classical.json": "classical greedy and LS, no probe",
    "paper/data/encoder_embedding_views.json": "pointer encoder only (verified: loads no SetPredictor)",
    "paper/data/encoder_linear_probe.json": "logistic fit over frozen features, not the SetPredictor",
    "paper/data/decode_search.json": "decode of the frozen policy, no probe",
    "paper/data/probe_ladder.json": "ladder dirs contain only final.pt, no best.pt to leak",
    "results/discvis_greedy_timing.json": "timing only",
    "results/classical_timing.json": "timing only",
    "results/classical_timing_lazy.json": "timing only",
    "results/probe_timing.json": "timing only",
    "results/discvis_gap_correlation.json": "disc-vis vs exact geometry, no probe",
    "results/reward_estimator_agreement.json": "reward estimator vs exact CGAL, no probe",
    "results/eval_t08.json": "learned editor, predates and is independent of the probe",
    "results/editor_sweep/v2/st0p9_cgnone.json": "learned editor sweep",
    "results/v2/sl_agp_evaluation.json": "supervised pointer baseline, no probe",
    # coords-only arm: those checkpoint dirs hold only final.pt, so no
    # validation-based selection ever existed for them
    **{f"paper/data/dist_dev_test_noseednoenc_seed{s}.json":
       "coords-only arm; its checkpoint dir has no best.pt"
       for s in (1234, 11, 22, 33)},
    "data/ls_trajectories_dev_test_clean.pkl":
        "policy greedy decodes and LS targets; contains no probe output",
}


def norm(s: str) -> str:
    """Strip LaTeX presentation so a registered value can be sought in a .tex body.

    `\\textbf{0.9734}\\,$\\pm$\\,0.0064` and `0.9734 +- 0.0064` must compare equal;
    without this, a bolded table cell reads as a mismatch.
    """
    s = re.sub(r"\\(?:textbf|emph|quad|mathrm)\{([^}]*)\}", r"\1", s)
    s = s.replace("\\,", " ").replace("\\pm", "+-").replace("$", " ")
    return re.sub(r"\s+", " ", s).strip()


def fmt(v: float, nd: int) -> str:
    """Round half-up, matching how the hand-written prose values were rounded.

    Python's f-string uses the binary representation, so 0.9245 formats as '0.924'
    while the manuscript writes 0.925.
    """
    return str(Decimal(str(v)).quantize(Decimal("1." + "0" * nd), rounding=ROUND_HALF_UP))


@dataclass
class Claim:
    id: str
    target: str                     # "tab_headline.tex" or "paper.tex"
    sources: list[str]
    n: int | None
    run: str
    group: str
    typeset: str | None = None      # exact-match form
    value: float | None = None      # for interval claims
    interval: tuple | None = None
    phrase: str | None = None       # the verbal claim being justified
    note: str = ""


CLAIMS: list[Claim] = []


def add(**kw) -> None:
    CLAIMS.append(Claim(**kw))


# ---------------------------------------------------------------- loaders (raw only)
def poly(name: str) -> list[dict]:
    return json.loads((D / name).read_text())["polygons"]


def res(name: str) -> dict:
    return json.loads((R / name).read_text())


def mean(xs) -> float:
    xs = list(xs)
    return float(np.mean(xs))


def pstd(xs) -> float:
    return float(np.std(list(xs)))


def opt_ratio(recs, key):
    return mean(r[key]["S_size"] / r["OPT"] for r in recs if r.get("OPT"))


def chv(recs, key):
    return mean(r[key]["S_size"] / r["n"] for r in recs)


def gate(recs, key, g=0.95):
    return sum(1 for r in recs if r[key]["cov"] >= g)


def below(recs, key, g=0.95):
    return sum(1 for r in recs if r[key]["cov"] < g)


def wilson_lo(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h)


def mcnemar(b, c):
    n = b + c
    return min(1.0, sum(comb(n, k) for k in range(0, min(b, c) + 1)) * 2 / 2 ** n) if n else 1.0


SEEDS = (11, 22, 33)
DEV = ["dist_dev_test.json"] + [f"dist_dev_test_seed{s}.json" for s in SEEDS]
DEVN = ["dist_dev_test_noenc.json"] + [f"dist_dev_test_noenc_seed{s}.json" for s in SEEDS]
OOD = ["dist_test_OOD.json"] + [f"dist_test_OOD_seed{s}.json" for s in SEEDS]
OODN = ["dist_test_OOD_noenc.json"] + [f"dist_test_OOD_noenc_seed{s}.json" for s in SEEDS]
LRG = ["dist_ood_large.json"] + [f"dist_ood_large_seed{s}.json" for s in SEEDS]
LRGN = ["dist_ood_large_noenc.json"] + [f"dist_ood_large_noenc_seed{s}.json" for s in SEEDS]


# ============================================================ tab_headline / dist_shift
def _headline():
    dev = poly("dist_dev_test.json")
    cls = json.loads((D / "baseline_classical.json").read_text())
    cls = cls["splits"]["dev_test"]["per_polygon"]
    g, r1 = "tab_headline.tex", "policy1234"
    S = ["paper/data/dist_dev_test.json", "paper/data/baseline_classical.json"]
    add(id="headline.greedy_opt", target=g, sources=["paper/data/baseline_classical.json"],
        n=362, run="classical", group="headline",
        typeset=f"{mean(x['S_size']/x['OPT'] for x in cls['greedy'] if x.get('OPT')):.4f}")
    add(id="headline.ls_opt", target=g, sources=["paper/data/baseline_classical.json"],
        n=362, run="classical", group="headline",
        typeset=f"{mean(x['S_size']/x['OPT'] for x in cls['ls'] if x.get('OPT')):.4f}")
    add(id="headline.seed_opt", target=g, sources=S, n=362, run=r1, group="headline",
        typeset=f"{opt_ratio(dev,'seed'):.4f}")
    add(id="headline.seed_gate", target=g, sources=S, n=362, run=r1, group="headline",
        typeset=f"{gate(dev,'seed')}/362")
    # every remaining column of the two deterministic rows
    add(id="headline.greedy_cov", target=g, sources=["paper/data/baseline_classical.json"],
        n=362, run="classical", group="headline",
        typeset=f"{mean(x['cov'] for x in cls['greedy']):.4f}")
    add(id="headline.greedy_chv", target=g, sources=["paper/data/baseline_classical.json"],
        n=362, run="classical", group="headline",
        typeset=f"{mean(x['S_size']/x['n'] for x in cls['greedy']):.4f}")
    add(id="headline.ls_cov", target=g, sources=["paper/data/baseline_classical.json"],
        n=362, run="classical", group="headline",
        typeset=f"{mean(x['cov'] for x in cls['ls']):.4f}")
    add(id="headline.ls_chv", target=g, sources=["paper/data/baseline_classical.json"],
        n=362, run="classical", group="headline",
        typeset=f"{mean(x['S_size']/x['n'] for x in cls['ls']):.4f}")
    add(id="headline.seed_cov", target=g, sources=S, n=362, run=r1, group="headline",
        typeset=f"{mean(r['seed']['cov'] for r in dev):.4f}")
    add(id="headline.seed_chv", target=g, sources=S, n=362, run=r1, group="headline",
        typeset=f"{chv(dev,'seed'):.4f}")
    # four-probe-seed aggregates: all four columns, recomputed from the per-seed raw files
    for lab, files, key in (("full", DEV, "probe_t020"), ("noenc", DEVN, "probe_t020")):
        src = [f"paper/data/{f}" for f in files]
        o = [opt_ratio(poly(f), key) for f in files]
        gt = [gate(poly(f), key) for f in files]
        cv = [mean(r[key]["cov"] for r in poly(f)) for f in files]
        hv = [chv(poly(f), key) for f in files]
        for nm, vals, nd in (("opt", o, 4), ("cov", cv, 4), ("chv", hv, 4)):
            add(id=f"headline.{lab}_{nm}", target=g, sources=src, n=362,
                run="probe_seeds_1234_11_22_33", group="headline",
                typeset=f"{mean(vals):.{nd}f}\\,$\\pm$\\,{pstd(vals):.{nd}f}")
        add(id=f"headline.{lab}_gate", target=g, sources=src, n=362,
            run="probe_seeds_1234_11_22_33", group="headline",
            typeset=f"${mean(gt):.1f} \\pm {pstd(gt):.1f}$/362")


def _dist_shift():
    dev, ne = poly("dist_dev_test.json"), poly("dist_dev_test_noenc.json")
    g = "tab_dist_shift.tex"
    for key in ("seed", "probe_t020", "probe_t025", "probe_t030"):
        add(id=f"dist_shift.{key}.ge099", target=g,
            sources=["paper/data/dist_dev_test.json"], n=362, run="policy1234",
            group="dist_shift",
            typeset=f"& {sum(1 for r in dev if r[key]['cov']>=0.99)} &")
    add(id="dist_shift.noenc.ge099", target=g,
        sources=["paper/data/dist_dev_test_noenc.json"], n=362, run="policy1234",
        group="dist_shift",
        typeset=f"& {sum(1 for r in ne if r['probe_t020']['cov']>=0.99)} &")


def _pareto():
    """tab_pareto.tex is generated but NOT \input by paper.tex -- an orphan. Its
    values are therefore not published claims and are deliberately not registered;
    the PDF check flagged them as absent, which is how the orphan was confirmed."""
    return


# =============================================================== tab_ood / tab_large
def _ood():
    g = "tab_ood.tex"
    ood = poly("dist_test_OOD.json")
    add(id="ood.seed_gate", target=g, sources=["paper/data/dist_test_OOD.json"],
        n=2081, run="policy1234", group="ood", typeset=f"{gate(ood,'seed')}/2081")
    add(id="ood.seed_cov", target=g, sources=["paper/data/dist_test_OOD.json"],
        n=2081, run="policy1234", group="ood",
        typeset=f"{mean(r['seed']['cov'] for r in ood):.4f}")
    add(id="ood.seed_chv", target=g, sources=["paper/data/dist_test_OOD.json"],
        n=2081, run="policy1234", group="ood", typeset=f"{chv(ood,'seed'):.4f}")
    add(id="ood.seed_opt", target=g, sources=["paper/data/dist_test_OOD.json"],
        n=2081, run="policy1234", group="ood", typeset=f"{opt_ratio(ood,'seed'):.4f}")
    for lab, files in (("full", OOD), ("noenc", OODN)):
        src = [f"paper/data/{f}" for f in files]
        o = [opt_ratio(poly(f), "probe_t020") for f in files]
        cv = [mean(r["probe_t020"]["cov"] for r in poly(f)) for f in files]
        hv = [chv(poly(f), "probe_t020") for f in files]
        gt = [gate(poly(f), "probe_t020") for f in files]
        for nm, vals in (("opt", o), ("cov", cv), ("chv", hv)):
            add(id=f"ood.{lab}_{nm}", target=g, sources=src, n=2081,
                run="probe_seeds_1234_11_22_33", group="ood",
                typeset=f"{mean(vals):.4f}\\,$\\pm$\\,{pstd(vals):.4f}")
        add(id=f"ood.{lab}_gate", target=g, sources=src, n=2081,
            run="probe_seeds_1234_11_22_33", group="ood",
            typeset=f"${mean(gt):.1f} \\pm {pstd(gt):.1f}$/2081")


def _large():
    g = "tab_large.tex"
    lrg = poly("dist_ood_large.json")
    add(id="large.seed_cov", target=g, sources=["paper/data/dist_ood_large.json"],
        n=285, run="policy1234", group="large",
        typeset=f"{mean(r['seed']['cov'] for r in lrg):.4f}")
    add(id="large.seed_mincov", target=g, sources=["paper/data/dist_ood_large.json"],
        n=285, run="policy1234", group="large",
        typeset=f"{min(r['seed']['cov'] for r in lrg):.3f}")
    add(id="large.seed_gate", target=g, sources=["paper/data/dist_ood_large.json"],
        n=285, run="policy1234", group="large",
        typeset=f"{gate(lrg,'seed')}/285")
    add(id="large.seed_chv", target=g, sources=["paper/data/dist_ood_large.json"],
        n=285, run="policy1234", group="large", typeset=f"{chv(lrg,'seed'):.4f}")
    for lab, files in (("full", LRG), ("noenc", LRGN)):
        hv = [chv(poly(f), "probe_t020") for f in files]
        gt = [gate(poly(f), "probe_t020") for f in files]
        add(id=f"large.{lab}_chv", target=g, sources=[f"paper/data/{f}" for f in files],
            n=285, run="probe_seeds_1234_11_22_33", group="large",
            typeset=f"{mean(hv):.4f}\\,$\\pm$\\,{pstd(hv):.4f}")
        add(id=f"large.{lab}_gate", target=g, sources=[f"paper/data/{f}" for f in files],
            n=285, run="probe_seeds_1234_11_22_33", group="large",
            typeset=f"${mean(gt):.1f} \\pm {pstd(gt):.1f}$/285")
        c = [mean(r["probe_t020"]["cov"] for r in poly(f)) for f in files]
        m = [min(r["probe_t020"]["cov"] for r in poly(f)) for f in files]
        add(id=f"large.{lab}_cov", target=g, sources=[f"paper/data/{f}" for f in files],
            n=285, run="probe_seeds_1234_11_22_33", group="large",
            typeset=f"{mean(c):.4f}\\,$\\pm$\\,{pstd(c):.4f}")
        add(id=f"large.{lab}_mincov", target=g, sources=[f"paper/data/{f}" for f in files],
            n=285, run="probe_seeds_1234_11_22_33", group="large",
            typeset=f"{mean(m):.3f}\\,$\\pm$\\,{pstd(m):.3f}")


# ======================================================== matched budget (a,b,c) + curve
def _mb_curve(arm, seed, sub="matched_budget"):
    d = json.loads((D / sub / f"{arm}_{seed}.json").read_text())["cells"]
    out = {}
    for k, c in d.items():
        if not k.endswith("|K=1"):
            continue
        out[float(k.split("|")[0].split("=")[1])] = (c["chv"], c["opt"], c["cov"],
                                                    float(c["dist"]["n_total"] - c["dist"]["n_cov_ge_099"]))
    return out


def _interp(pts, x_i, y_i, target):
    p = sorted(pts, key=lambda v: v[x_i])
    xs, ys = [v[x_i] for v in p], [v[y_i] for v in p]
    if target < xs[0] or target > xs[-1]:
        return None
    for i in range(len(xs) - 1):
        if xs[i] <= target <= xs[i + 1]:
            if xs[i + 1] == xs[i]:
                return ys[i]
            w = (target - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + w * (ys[i + 1] - ys[i])
    return None


def _operating_curve():
    g = "tab_operating_curve.tex"
    srcs = [f"paper/data/matched_budget/full_{s}.json" for s in ("1234", 11, 22, 33)]
    cur = [_mb_curve("full", s) for s in ("1234", 11, 22, 33)]
    full = [json.loads((D / "matched_budget" / f"full_{s}.json").read_text())
            for s in ("1234", 11, 22, 33)]
    pol = full[0]["seed"]
    add(id="opcurve.policy.cov", target=g, sources=srcs, n=362, run="policy1234",
        group="opcurve_policy", typeset=f"${pol['cov']:.4f}$")
    add(id="opcurve.policy.chv", target=g, sources=srcs, n=362, run="policy1234",
        group="opcurve_policy", typeset=f"${pol['chv']:.3f}$")
    add(id="opcurve.policy.opt", target=g, sources=srcs, n=362, run="policy1234",
        group="opcurve_policy", typeset=f"${pol['opt']:.2f}$")
    add(id="opcurve.policy.gate", target=g, sources=srcs, n=362, run="policy1234",
        group="opcurve_policy", typeset=f"${pol['dist']['n_cov_ge_095']}$")
    for t in sorted(cur[0]):
        cells = [d["cells"][f"t={t}|K=1"] for d in full]
        add(id=f"opcurve.t{t}.opt", target=g, sources=srcs, n=362,
            run="probe_seeds_1234_11_22_33", group="opcurve",
            typeset=f"{mean(c[t][1] for c in cur):.2f}")
        for nm, vals, nd in (("cov", [c["cov"] for c in cells], 4),
                             ("chv", [c["chv"] for c in cells], 3),
                             ("gate", [c["dist"]["n_cov_ge_095"] for c in cells], 1)):
            mu, sd = mean(vals), pstd(vals)
            ts = (f"${mu:.{nd}f}$" if sd < 0.5 * 10 ** (-nd)
                  else f"${mu:.{nd}f} \\pm {sd:.{nd}f}$")
            add(id=f"opcurve.t{t}.{nm}", target=g, sources=srcs, n=362,
                run="probe_seeds_1234_11_22_33", group="opcurve", typeset=ts)


def _matched_budget():
    g = "tab_ablation_thresholds.tex"
    arms = ("full", "noseed", "noenc", "coords")
    srcs = [f"paper/data/matched_budget/{a}_{s}.json"
            for a in arms for s in ("1234", 11, 22, 33)]
    cur = {a: [_mb_curve(a, s) for s in ("1234", 11, 22, 33)] for a in arms}
    for t in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80):
        for a in arms:
            add(id=f"mb.a.{a}.t{t}.opt", target=g, sources=srcs, n=362,
                run="probe_seeds_1234_11_22_33", group="mb_panel_a",
                typeset=f"{mean(c[t][1] for c in cur[a]):.2f}")
            add(id=f"mb.a.{a}.t{t}.fail", target=g, sources=srcs, n=362,
                run="probe_seeds_1234_11_22_33", group="mb_panel_a",
                typeset=f"{mean(c[t][3] for c in cur[a]):.0f}")
    for anchor in (1.95, 1.7, 1.5, 1.3):
        for a in arms:
            v = [_interp(c.values(), 1, 3, anchor) for c in cur[a]]
            v = [x for x in v if x is not None]
            add(id=f"mb.b.{a}.{anchor}", target=g, sources=srcs, n=362,
                run="probe_seeds_1234_11_22_33", group="mb_panel_b",
                typeset=f"${mean(v):.0f} \\pm {pstd(v):.0f}$")


def _matched_budget_pseeds():
    """Panel (c): cost multiple per policy. Policy 1234 averages four probe seeds."""
    g = "tab_ablation_thresholds.tex"
    tails = (30, 50, 76, 100)
    base = {a: [_mb_curve(a, s) for s in ("1234", 11, 22, 33)] for a in ("full", "noenc")}
    for t in tails:
        f = [_interp(c.values(), 3, 1, t) for c in base["full"]]
        n_ = [_interp(c.values(), 3, 1, t) for c in base["noenc"]]
        if all(x is not None for x in f + n_):
            add(id=f"mb.c.1234.{t}", target=g,
                sources=[f"paper/data/matched_budget/{a}_{s}.json"
                         for a in ("full", "noenc") for s in ("1234", 11, 22, 33)],
                n=362, run="policy1234", group="mb_panel_c",
                typeset=f"{mean(n_)/mean(f):.2f}$\\times$")
    for s in SEEDS:
        fc = _mb_curve("full", f"pseed{s}", "matched_budget_pseeds").values()
        nc = _mb_curve("noenc", f"pseed{s}", "matched_budget_pseeds").values()
        for t in tails:
            a_, b_ = _interp(fc, 3, 1, t), _interp(nc, 3, 1, t)
            if a_ and b_:
                add(id=f"mb.c.p{s}.{t}", target=g,
                    sources=[f"paper/data/matched_budget_pseeds/full_pseed{s}.json",
                             f"paper/data/matched_budget_pseeds/noenc_pseed{s}.json"],
                    n=362, run=f"policy{s}", group="mb_panel_c",
                    typeset=f"{b_/a_:.2f}$\\times$")


# ============================================================== tab_policy_seeds
def _policy_seeds():
    g = "tab_policy_seeds.tex"
    for s in SEEDS:
        for split, tag, N in (("test", "ood", 2081), ("large", "ood_large", 285)):
            for arm in ("full", "noenc"):
                p = R / "policy_seeds_ood" / f"pseed{s}_{arm}_{split}.json"
                if not p.exists():
                    continue
                c = json.loads(p.read_text())["cells"]["t=0.2|K=1"]
                add(id=f"pseeds.{tag}.{s}.{arm}.gate", target=g,
                    sources=[f"results/policy_seeds_ood/pseed{s}_{arm}_{split}.json"],
                    n=N, run=f"policy{s}", group=f"pseeds_{tag}",
                    typeset=f"{c['dist']['n_cov_ge_095']}/{N}")
                add(id=f"pseeds.{tag}.{s}.{arm}.chv", target=g,
                    sources=[f"results/policy_seeds_ood/pseed{s}_{arm}_{split}.json"],
                    n=N, run=f"policy{s}", group=f"pseeds_{tag}",
                    typeset=f"{c['chv']:.3f}")
            sd = json.loads((R / "policy_seeds_ood" /
                             f"pseed{s}_full_{split}.json").read_text())["seed"]
            add(id=f"pseeds.{tag}.{s}.policy.gate", target=g,
                sources=[f"results/policy_seeds_ood/pseed{s}_full_{split}.json"],
                n=N, run=f"policy{s}", group=f"pseeds_{tag}",
                typeset=f"{sd['dist']['n_cov_ge_095']}/{N}")
            add(id=f"pseeds.{tag}.{s}.policy.chv", target=g,
                sources=[f"results/policy_seeds_ood/pseed{s}_full_{split}.json"],
                n=N, run=f"policy{s}", group=f"pseeds_{tag}",
                typeset=f"{sd['chv']:.3f}")


# ====================================================== hand-curated + prose sources
def _probe_ladder():
    d = json.loads((D / "probe_ladder.json").read_text())
    for row in d["rows"]:
        for nm, k, sd in (("roc", "roc_auc_mean", "roc_auc_std"),
                          ("pr", "pr_auc_mean", "pr_auc_std")):
            add(id=f"ladder.{row['name']}.{nm}", target="tab_probe_ladder.tex",
                sources=["paper/data/probe_ladder.json"], n=362, run="ladder",
                group="probe_ladder",
                typeset=f"{row[k]:.3f}\\,$\\pm$\\,{row[sd]:.3f}")
        add(id=f"ladder.{row['name']}.params", target="tab_probe_ladder.tex",
            sources=["paper/data/probe_ladder.json"], n=362, run="ladder",
            group="probe_ladder", typeset=f"{row['n_params']:,}")


def _decode_search():
    """tab_decode_search: three decode rows of the frozen policy, K=32."""
    d = json.loads((D / "decode_search.json").read_text())["summary"]
    for row, tag in (("greedy", "greedy"), ("bestK_ll", "ll"), ("bestK_oracle", "oracle")):
        r = d[row]
        src = ["paper/data/decode_search.json"]
        add(id=f"decode.{tag}.gate", target="tab_decode_search.tex", sources=src,
            n=362, run="decode_K32", group="decode_search",
            typeset=f"{r['n_polygons']-r['n_below_gate']}/362")
        add(id=f"decode.{tag}.below99", target="tab_decode_search.tex", sources=src,
            n=362, run="decode_K32", group="decode_search",
            typeset=f"& {r['n_below_099']} &")
        for nm, k, nd in (("cov", "mean_cov", 3), ("min", "min_cov", 3),
                          ("chv", "mean_S_over_n", 3), ("opt", "mean_S_over_OPT", 2)):
            add(id=f"decode.{tag}.{nm}", target="tab_decode_search.tex", sources=src,
                n=362, run="decode_K32", group="decode_search",
                typeset=f"{r[k]:.{nd}f}")


def _runtime():
    pt = res("probe_timing.json")["rows"]
    for r in pt:
        add(id=f"runtime.{r['device']}.n{r['n']}", target="tab_runtime.tex",
            sources=["results/probe_timing.json"], n=None, run="timing",
            group="runtime_learned", typeset=f"${round(r['total_ms']):d}$")


def _reinforce():
    for row in json.loads((D / "baseline_reinforce.json").read_text())["rows"]:
        add(id=f"reinforce.{row['name'][:12]}", target="tab_reinforce.tex",
            sources=["paper/data/baseline_reinforce.json"], n=1224,
            run="pre-dedup-protocol", group="reinforce",
            typeset=f"{row['S_over_OPT']:.2f}",
            note="earlier protocol on the pooled pool; caption says so")


def _significance():
    sig = json.loads((D / "significance_ood.json").read_text())["per_seed"]
    src = ["paper/data/significance_ood.json"] + [f"paper/data/{f}" for f in OOD]
    add(id="sig.maxp", target="paper.tex", sources=src, n=2081,
        run="probe_seeds_1234_11_22_33", group="significance",
        value=max(v["mcnemar_exact_p"] for v in sig.values()), interval=(0, 1e-74),
        phrase="$p < 10^{-74}$ in all four cases")
    add(id="sig.minwilson", target="paper.tex", sources=src, n=2081,
        run="probe_seeds_1234_11_22_33", group="significance",
        value=min(v["probe_feasible_wilson95"][0] for v in sig.values()),
        interval=(0.977, 1.0), phrase="all lie above $0.977$")
    dev = poly("dist_dev_test.json")
    b = sum(1 for r in dev if r["seed"]["cov"] < 0.95 <= r["probe_t020"]["cov"])
    c = sum(1 for r in dev if r["probe_t020"]["cov"] < 0.95 <= r["seed"]["cov"])
    for lab, v in (("b", b), ("c", c), ("disc", b + c)):
        add(id=f"sig.indist.{lab}", target="paper.tex",
            sources=["paper/data/dist_dev_test.json"], n=362, run="policy1234",
            group="significance_indist", typeset=f"${v}$")


def _invariance():
    s = json.loads((D / "invariance_test.json").read_text())["summary"]
    for k in ("identity", "rot45", "rot90", "rot180", "mirror_x", "reindex"):
        v = s[k]["n_polygons"] - s[k]["n_below_gate"]
        # the identity baseline is written with its denominator, the rest bare
        ts = f"${v}/362$" if k == "identity" else f"${v}$"
        add(id=f"inv.{k}", target="paper.tex",
            sources=["paper/data/invariance_test.json"], n=362, run="policy1234",
            group="invariance", typeset=ts)


def _canonical():
    c = res("canonical_rule_selection.json")
    sel = c["dev_test_selected_rule"]
    add(id="canon.matched_gate", target="paper.tex",
        sources=["results/canonical_rule_selection.json"], n=362, run="policy1234",
        group="canonical",
        typeset=f"${sel['selected_matched']['n_ge_gate']}/362$")
    add(id="canon.native_gate", target="paper.tex",
        sources=["results/canonical_rule_selection.json"], n=362, run="policy1234",
        group="canonical",
        typeset=f"${sel['native_baseline_matched']['n_ge_gate']}/362$")


def _editor():
    a = res("eval_t08.json")["summary"]
    b = res("editor_sweep/v2/st0p9_cgnone.json")["summary"]
    add(id="editor.cut13", target="paper.tex", sources=["results/eval_t08.json"],
        n=a["n_polygons"], run="editor_t08", group="editor",
        value=(1 - a["ed_size_mean"] / a["seed_size_mean"]) * 100, interval=(12.0, 13.9),
        phrase="cuts only $13\\%$ of the guards")
    add(id="editor.recovery", target="paper.tex", sources=["results/eval_t08.json"],
        n=a["n_polygons"], run="editor_t08", group="editor",
        typeset=f"${a['frac_recovery_ge_1.0']:.2f}$")
    add(id="editor.cut26", target="paper.tex",
        sources=["results/editor_sweep/v2/st0p9_cgnone.json"], n=b["n_polygons"],
        run="editor_st0p9", group="editor_aggressive",
        value=(1 - b["ed_size_mean"] / b["seed_size_mean"]) * 100, interval=(25.5, 26.9),
        phrase="up to $26\\%$")
    add(id="editor.cov925", target="paper.tex",
        sources=["results/editor_sweep/v2/st0p9_cgnone.json"], n=b["n_polygons"],
        run="editor_st0p9", group="editor_aggressive",
        value=b["ed_cov_mean"], interval=(0.9235, 0.9255),
        phrase="dropping the mean to $0.925$")


def _supervised():
    d = res("v2/sl_agp_evaluation.json")
    add(id="sup.cov", target="paper.tex", sources=["results/v2/sl_agp_evaluation.json"],
        n=None, run="supervised_pointer", group="supervised",
        value=d["polygon_coverage"]["mean"], interval=(0.45, 0.55),
        phrase="mean coverage $\\approx 0.50$")
    add(id="sup.ratio", target="paper.tex", sources=["results/v2/sl_agp_evaluation.json"],
        n=None, run="supervised_pointer", group="supervised",
        value=d["approx_ratio"]["median"], interval=(2.8, 3.3),
        phrase="roughly $3\\times$ the optimum")


def _linear_probe():
    lp = json.loads((D / "encoder_linear_probe.json").read_text())
    for k, key in (("roc", "roc_auc_mean"), ("rocsd", "roc_auc_std"),
                   ("pr", "pr_auc_mean"), ("prsd", "pr_auc_std")):
        nd = 3
        add(id=f"linprobe.{k}", target="paper.tex",
            sources=["paper/data/encoder_linear_probe.json"],
            n=lp["n_polygons"], run="linear_probe_cv", group="linprobe",
            typeset=f"{lp[key]:.{nd}f}")


def _reward_estimator():
    s = res("reward_estimator_agreement.json")["summary"]
    add(id="rew.bias", target="paper.tex",
        sources=["results/reward_estimator_agreement.json"], n=s["n_polygons"],
        run="reward_agreement", group="reward_est",
        typeset=f"{s['coverage_bias_disc_minus_exact']['mean']:+.4f}")
    add(id="rew.agree", target="paper.tex",
        sources=["results/reward_estimator_agreement.json"], n=s["n_polygons"],
        run="reward_agreement", group="reward_est",
        typeset=f"{s['pairwise_ordering_agreement']*100:.1f}\\%")


def _verbal():
    """Verbal magnitude claims, checked as intervals rather than exempted."""
    dev, ne = poly("dist_dev_test.json"), poly("dist_dev_test_noenc.json")
    cur = {a: [_mb_curve(a, s) for s in ("1234", 11, 22, 33)] for a in
           ("full", "noseed", "noenc", "coords")}
    m = lambda a, t, i: mean(c[t][i] for c in cur[a])  # noqa: E731
    S = [f"paper/data/matched_budget/{a}_{s}.json"
         for a in cur for s in ("1234", 11, 22, 33)]

    add(id="v.ablation_twice_guards", target="paper.tex",
        sources=["paper/data/dist_dev_test.json", "paper/data/dist_dev_test_noenc.json"],
        n=362, run="policy1234", group="verbal_ablation",
        value=chv(ne, "probe_t020") / chv(dev, "probe_t020"), interval=(1.7, 2.0),
        phrase="nearly twice the full probe's guard fraction")
    add(id="v.two_fifths_fewer", target="paper.tex", sources=S, n=362,
        run="probe_seeds_1234_11_22_33", group="verbal_mb",
        value=1 - m("full", 0.30, 3) / m("noenc", 0.30, 3), interval=(0.36, 0.45),
        phrase="about two-fifths fewer polygons below the $0.99$ gate")
    add(id="v.more_than_twice_cost", target="paper.tex", sources=S, n=362,
        run="probe_seeds_1234_11_22_33", group="verbal_mb",
        value=m("noenc", 0.05, 1) / m("full", 0.05, 1), interval=(2.0, 3.0),
        phrase="more than twice the full probe's guard cost")
    add(id="v.coords_three_quarters", target="paper.tex", sources=S, n=362,
        run="probe_seeds_1234_11_22_33", group="verbal_mb",
        value=m("coords", 0.20, 0), interval=(0.70, 0.78),
        phrase="nearly three-quarters of all vertices")
    add(id="v.coords_three_times", target="paper.tex", sources=S, n=362,
        run="probe_seeds_1234_11_22_33", group="verbal_mb",
        value=(_interp_mean(cur["coords"], 1.5) / _interp_mean(cur["full"], 1.5)),
        interval=(3.0, 3.6), phrase="more than three times as many polygons")
    # in-distribution tail reduction, per probe seed -- "at least four-fold"
    per = [below(poly(f), "probe_t020") for f in DEV]
    add(id="v.fourfold_min", target="paper.tex",
        sources=[f"paper/data/{f}" for f in DEV], n=362,
        run="probe_seeds_1234_11_22_33", group="verbal_reduction_indist",
        value=min(69 / p for p in per), interval=(4.0, 99),
        phrase="at least four-fold on every probe seed")
    # OOD reduction -- "about an order of magnitude"
    add(id="v.ood_order_mag", target="paper.tex",
        sources=[f"paper/data/{f}" for f in OOD], n=2081,
        run="probe_seeds_1234_11_22_33", group="verbal_reduction_ood",
        value=303 / mean(below(poly(f), "probe_t020") for f in OOD), interval=(8, 16),
        phrase="about an order of magnitude")
    # probe vs its own LS target cardinality -- "roughly 1.8x"
    cls = json.loads((D / "baseline_classical.json").read_text())
    ls = cls["splits"]["dev_test"]["per_polygon"]["ls"]
    coords_mult = []
    for t in (50, 76, 100):
        f = _interp_mean_x(cur["full"], t)
        c_ = _interp_mean_x(cur["coords"], t)
        if f and c_:
            coords_mult.append(c_ / f)
    add(id="v.coords_cost_multiple", target="paper.tex", sources=S, n=362,
        run="probe_seeds_1234_11_22_33", group="verbal_mb",
        value=mean(coords_mult), interval=(1.70, 1.80),
        phrase="coordinates alone about $1.75\\times$")
    add(id="v.probe_vs_ls", target="paper.tex",
        sources=["paper/data/baseline_classical.json"] + [f"paper/data/{f}" for f in DEV],
        n=362, run="probe_seeds_1234_11_22_33", group="verbal_ranking",
        value=mean(chv(poly(f), "probe_t020") for f in DEV)
              / mean(x["S_size"] / x["n"] for x in ls),
        interval=(1.72, 1.88), phrase="roughly $1.8\\times$ the cardinality")
    # LS edits -- "a handful"
    recs = pickle.load(open(REPO / "data/ls_trajectories_dev_test_clean.pkl", "rb"))["records"]
    add(id="v.ls_handful", target="paper.tex",
        sources=["data/ls_trajectories_dev_test_clean.pkl"], n=362, run="policy1234",
        group="verbal_ls", value=float(np.median([x["n_edits"] for x in recs])),
        interval=(1, 6), phrase="a handful of local edits")
    # ood-large size ratio -- "about eleven times"
    lrg = poly("dist_ood_large.json")
    add(id="v.eleven_times", target="paper.tex",
        sources=["paper/data/dist_ood_large.json"], n=285, run="dataset",
        group="verbal_scale", value=max(r["n"] for r in lrg) / 198,
        interval=(10.5, 11.9), phrase="about eleven times the largest training polygon")


def _interp_mean_x(curves, tail):
    """Guard cost at which each curve leaves `tail` polygons below the 0.99 gate."""
    v = [_interp(c.values(), 3, 1, tail) for c in curves]
    v = [x for x in v if x is not None]
    return mean(v) if v else None


def _interp_mean(curves, anchor):
    v = [_interp(c.values(), 1, 3, anchor) for c in curves]
    return mean([x for x in v if x is not None])


# ======================================== remaining tables, registered from raw sources
ARMS_2X2 = {
    "full":       ["dist_dev_test.json"] + [f"dist_dev_test_seed{s}.json" for s in SEEDS],
    "noseed":     [f"dist_dev_test_noseed_seed{s}.json" for s in (1234, 11, 22, 33)],
    "noencoder":  ["dist_dev_test_noenc.json"] + [f"dist_dev_test_noenc_seed{s}.json" for s in SEEDS],
    "coordsonly": [f"dist_dev_test_noseednoenc_seed{s}.json" for s in (1234, 11, 22, 33)],
}


def _ablation_2x2():
    """tab_ablation at t=0.20, four arms x four columns, from the per-polygon files."""
    g = "tab_ablation.tex"
    for arm, files in ARMS_2X2.items():
        src = [f"paper/data/{f}" for f in files]
        recs = [poly(f) for f in files]
        b99 = [sum(1 for r in x if r["probe_t020"]["cov"] < 0.99) for x in recs]
        mn = [min(r["probe_t020"]["cov"] for r in x) for x in recs]
        hv = [chv(x, "probe_t020") for x in recs]
        o = [opt_ratio(x, "probe_t020") for x in recs]
        add(id=f"abl.{arm}.below99", target=g, sources=src, n=362,
            run="probe_seeds_1234_11_22_33", group="ablation_2x2",
            typeset=f"${mean(b99):.1f} \\pm {pstd(b99):.1f}$")
        add(id=f"abl.{arm}.mincov", target=g, sources=src, n=362,
            run="probe_seeds_1234_11_22_33", group="ablation_2x2",
            typeset=f"{mean(mn):.3f}\\,$\\pm$\\,{pstd(mn):.3f}")
        add(id=f"abl.{arm}.chv", target=g, sources=src, n=362,
            run="probe_seeds_1234_11_22_33", group="ablation_2x2",
            typeset=f"{mean(hv):.2f}\\,$\\pm$\\,{pstd(hv):.2f}")
        add(id=f"abl.{arm}.opt", target=g, sources=src, n=362,
            run="probe_seeds_1234_11_22_33", group="ablation_2x2",
            typeset=f"{mean(o):.2f}\\,$\\pm$\\,{pstd(o):.2f}")


def _policy_seeds_indist():
    """tab_policy_seeds, the in-distribution block."""
    g = "tab_policy_seeds.tex"
    dev, ne = poly("dist_dev_test.json"), poly("dist_dev_test_noenc.json")
    S = ["paper/data/dist_dev_test.json", "paper/data/dist_dev_test_noenc.json"]
    for lab, recs, key in (("policy", dev, "seed"), ("full", dev, "probe_t020"),
                           ("noenc", ne, "probe_t020")):
        add(id=f"pseeds.test.1234.{lab}.chv", target=g, sources=S, n=362,
            run="policy1234", group="pseeds_test", typeset=f"{chv(recs,key):.3f}")
        add(id=f"pseeds.test.1234.{lab}.gate", target=g, sources=S, n=362,
            run="policy1234", group="pseeds_test", typeset=f"{gate(recs,key)}/362")
    for sd in SEEDS:
        src = [f"results/policy_seeds/pseed{sd}_{a}.json" for a in ("full", "noenc")]
        for lab, arm, key in (("policy", "full", "seed"), ("full", "full", "cell"),
                              ("noenc", "noenc", "cell")):
            d = res(f"policy_seeds/pseed{sd}_{arm}.json")
            src1 = d["seed"] if key == "seed" else d["cells"]["t=0.2|K=1"]
            add(id=f"pseeds.test.{sd}.{lab}.chv", target=g, sources=src, n=362,
                run=f"policy{sd}", group="pseeds_test",
                typeset=f"{src1['chv']:.3f}")
            add(id=f"pseeds.test.{sd}.{lab}.gate", target=g, sources=src, n=362,
                run=f"policy{sd}", group="pseeds_test",
                typeset=f"{src1['dist']['n_cov_ge_095']}/362")


def _discvis_quality():
    d = res("discvis_greedy_timing.json")
    for b in d["buckets"]:
        n = b["bucket"]
        add(id=f"dvq.{n}.guards", target="tab_discvis_quality.tex",
            sources=["results/discvis_greedy_timing.json"], n=b["n_polys"],
            run="discvis_timing", group=f"discvis_quality_{n}",
            typeset=f"{b['n_guards_mean']:.1f}")
        add(id=f"dvq.{n}.exact", target="tab_discvis_quality.tex",
            sources=["results/discvis_greedy_timing.json"], n=b["n_polys"],
            run="discvis_timing", group=f"discvis_quality_{n}",
            typeset=f"{b['exact_coverage_mean']:.3f}")


def _runtime_classical():
    cl_ = res("classical_timing_lazy.json")["classical"]
    dv = res("discvis_greedy_timing.json")["buckets"]
    for r in cl_:
        add(id=f"rt.exact.{r['bucket']}.pre", target="tab_runtime.tex",
            sources=["results/classical_timing_lazy.json"], n=r["n_polys"],
            run="timing", group=f"runtime_classical_{r['bucket']}",
            typeset=f"${round(r['vis_ms_mean']):d}$")
        add(id=f"rt.exact.{r['bucket']}.sel", target="tab_runtime.tex",
            sources=["results/classical_timing_lazy.json"], n=r["n_polys"],
            run="timing", group=f"runtime_classical_{r['bucket']}",
            typeset=f"${(r['greedy_ms_mean']-r['vis_ms_mean'])/1000:.1f}$")
    for b in dv:
        add(id=f"rt.disc.{b['bucket']}.pre", target="tab_runtime.tex",
            sources=["results/discvis_greedy_timing.json"], n=b["n_polys"],
            run="timing", group=f"runtime_disc_{b['bucket']}",
            typeset=f"${round(b['disc_vis_s_mean']*1000):d}$")
        add(id=f"rt.disc.{b['bucket']}.sel", target="tab_runtime.tex",
            sources=["results/discvis_greedy_timing.json"], n=b["n_polys"],
            run="timing", group=f"runtime_disc_{b['bucket']}",
            typeset=f"${b['selection_s_mean']*1000:.1f}$")


def _minor_cells():
    """min-Cov and Wilson cells that no other group covers."""
    dev, ne = poly("dist_dev_test.json"), poly("dist_dev_test_noenc.json")
    for lab, recs, key in (("seed", dev, "seed"), ("full", dev, "probe_t020"),
                           ("t025", dev, "probe_t025"), ("t030", dev, "probe_t030"),
                           ("noenc", ne, "probe_t020")):
        add(id=f"dist_shift.{lab}.mincov", target="tab_dist_shift.tex",
            sources=["paper/data/dist_dev_test.json",
                     "paper/data/dist_dev_test_noenc.json"],
            n=362, run="policy1234", group="dist_shift_min",
            typeset=f"{min(r[key]['cov'] for r in recs):.3f}")
    for lab, recs, key in (("seed", dev, "seed"), ("t025", dev, "probe_t025"),
                           ("t030", dev, "probe_t030")):
        add(id=f"dist_shift.{lab}.eq1", target="tab_dist_shift.tex",
            sources=["paper/data/dist_dev_test.json"], n=362, run="policy1234",
            group="dist_shift_min",
            typeset=f"& {sum(1 for r in recs if r[key]['cov']>=0.99995)} &")
    for lab, recs, key in (("t025", dev, "probe_t025"), ("t030", dev, "probe_t030")):
        add(id=f"dist_shift.{lab}.ge0999", target="tab_dist_shift.tex",
            sources=["paper/data/dist_dev_test.json"], n=362, run="policy1234",
            group="dist_shift_min",
            typeset=f"& {sum(1 for r in recs if r[key]['cov']>=0.999)} &")
    rf = json.loads((D / "baseline_reinforce.json").read_text())["rows"]
    for row in rf:
        add(id=f"reinforce.{row['name'][:12]}.cov", target="tab_reinforce.tex",
            sources=["paper/data/baseline_reinforce.json"], n=1224,
            run="pre-dedup-protocol", group="reinforce_cov",
            typeset=f"{row['cov']:.3f}")


def _remaining():
    """Cells and prose values not covered by any other group."""
    # tab_policy_seeds: the policy-1234 rows on ood / ood-large come from paper/data
    for tag, files, N in (("ood", ("dist_test_OOD.json", "dist_test_OOD_noenc.json"), 2081),
                          ("ood_large", ("dist_ood_large.json", "dist_ood_large_noenc.json"), 285)):
        f, fn = poly(files[0]), poly(files[1])
        src = [f"paper/data/{x}" for x in files]
        for lab, recs, key in (("policy", f, "seed"), ("full", f, "probe_t020"),
                               ("noenc", fn, "probe_t020")):
            add(id=f"pseeds.{tag}.1234.{lab}.chv", target="tab_policy_seeds.tex",
                sources=src, n=N, run="policy1234", group=f"pseeds_{tag}_1234",
                typeset=f"{chv(recs,key):.3f}")
            add(id=f"pseeds.{tag}.1234.{lab}.gate", target="tab_policy_seeds.tex",
                sources=src, n=N, run="policy1234", group=f"pseeds_{tag}_1234",
                typeset=f"{gate(recs,key)}/{N}")
    # disc-vis quality: the gap column and the guard-count spread
    for b in res("discvis_greedy_timing.json")["buckets"]:
        n = b["bucket"]
        add(id=f"dvq.{n}.gap", target="tab_discvis_quality.tex",
            sources=["results/discvis_greedy_timing.json"], n=b["n_polys"],
            run="discvis_timing", group=f"discvis_gap_{n}",
            typeset=f"{b['disc_vis_coverage_mean']-b['exact_coverage_mean']:.3f}")
        add(id=f"dvq.{n}.gstd", target="tab_discvis_quality.tex",
            sources=["results/discvis_greedy_timing.json"], n=b["n_polys"],
            run="discvis_timing", group=f"discvis_gap_{n}",
            typeset=f"{b['n_guards_std']:.1f}")
    # Wilson intervals printed alongside a gate count
    dev, ood = poly("dist_dev_test.json"), poly("dist_test_OOD.json")
    for tag, recs, N, tgt in (("headline", dev, 362, "tab_headline.tex"),
                              ("ood", ood, 2081, "tab_ood.tex")):
        k = gate(recs, "seed")
        lo = wilson_lo(k, N)
        add(id=f"{tag}.seed_wilson_lo", target=tgt,
            sources=[f"paper/data/{'dist_dev_test' if tag=='headline' else 'dist_test_OOD'}.json"],
            n=N, run="policy1234", group=f"{tag}_wilson",
            typeset=f"{lo:.3f}")
    # prose: worked example, invariance coverage range, operating curve, runtime, subset
    we = json.loads((D / "worked_examples.json").read_text())["examples"]
    for i, e in enumerate(we):
        add(id=f"fig.we{i}.seedcov", target="paper.tex",
            sources=["paper/data/worked_examples.json"], n=1, run=f"worked_{e['name']}",
            group=f"worked_example_{i}", typeset=f"{e['seed_coverage']:.3f}")
        add(id=f"fig.we{i}.probecov", target="paper.tex",
            sources=["paper/data/worked_examples.json"], n=1, run=f"worked_{e['name']}",
            group=f"worked_example_{i}", typeset=f"{e['probe_coverage']:.3f}")
        add(id=f"fig.we{i}.opt", target="paper.tex",
            sources=["paper/data/worked_examples.json"], n=1, run=f"worked_{e['name']}",
            group=f"worked_example_{i}",
            typeset=f"{len(e['probe_idxs'])/len(e['opt_idxs']):.2f}")
    inv = json.loads((D / "invariance_test.json").read_text())["summary"]
    covs = [v["mean_cov"] for k, v in inv.items() if k != "identity"]
    add(id="inv.identity_cov", target="paper.tex",
        sources=["paper/data/invariance_test.json"], n=362, run="policy1234",
        group="invariance_cov", typeset=f"{inv['identity']['mean_cov']:.4f}")
    add(id="inv.cov_lo", target="paper.tex",
        sources=["paper/data/invariance_test.json"], n=362, run="policy1234",
        group="invariance_cov", typeset=f"{min(covs):.3f}")
    add(id="inv.worst_min", target="paper.tex",
        sources=["paper/data/invariance_test.json"], n=362, run="policy1234",
        group="invariance_cov",
        typeset=f"{min(v['min_cov'] for k,v in inv.items() if k!='identity'):.3f}")
    full = [json.loads((D / "matched_budget" / f"full_{s}.json").read_text())
            for s in ("1234", 11, 22, 33)]
    c50 = [d["cells"]["t=0.5|K=1"] for d in full]
    add(id="opcurve.t050.cov_prose", target="paper.tex",
        sources=[f"paper/data/matched_budget/full_{s}.json" for s in ("1234", 11, 22, 33)],
        n=362, run="probe_seeds_1234_11_22_33", group="opcurve_prose",
        typeset=f"{mean(c['cov'] for c in c50):.3f}")
    add(id="opcurve.t050.gate_prose", target="paper.tex",
        sources=[f"paper/data/matched_budget/full_{s}.json" for s in ("1234", 11, 22, 33)],
        n=362, run="probe_seeds_1234_11_22_33", group="opcurve_prose",
        typeset=f"{mean(c['dist']['n_cov_ge_095'] for c in c50):.0f}")
    cl_ = res("classical_timing_lazy.json")["classical"]
    pt = {(r["device"], r["n"]): r for r in res("probe_timing.json")["rows"]}
    # quoted from the rounded table cells, so a reader can reproduce it from
    # tab_runtime itself; using unrounded values would give 1505 instead of 1495
    ratios = []
    for r in cl_:
        k = ("cuda", r["bucket"])
        if k not in pt:
            continue
        exact_total = round(r["vis_ms_mean"]) + round(
            (r["greedy_ms_mean"] - r["vis_ms_mean"]) / 1000, 1) * 1000
        ratios.append(exact_total / round(pt[k]["total_ms"]))
    add(id="runtime.exact_gpu_max", target="paper.tex",
        sources=["results/classical_timing_lazy.json", "results/probe_timing.json"],
        n=None, run="timing", group="runtime_ratio",
        typeset=f"{max(ratios):.0f}")
    big = [r for r in ood if r["n"] > 198]
    add(id="ood.n198_probe_pct", target="paper.tex",
        sources=[f"paper/data/{f}" for f in OOD], n=885,
        run="probe_seeds_1234_11_22_33", group="ood_subset",
        typeset=f"{mean(sum(1 for r in poly(f) if r['n']>198 and r['probe_t020']['cov']>=0.95)/len(big) for f in OOD)*100:.1f}")
    a = res("eval_t08.json")["summary"]
    add(id="editor.cov972", target="paper.tex", sources=["results/eval_t08.json"],
        n=a["n_polygons"], run="editor_t08", group="editor_cov",
        typeset=fmt(a["ed_cov_mean"], 3))
    add(id="editor.covseed", target="paper.tex", sources=["results/eval_t08.json"],
        n=a["n_polygons"], run="editor_t08", group="editor_cov",
        typeset=f"{a['seed_cov_mean']:.3f}")


def build() -> list[Claim]:
    CLAIMS.clear()
    for fn in (_headline, _dist_shift, _pareto, _ood, _large, _operating_curve,
               _matched_budget, _matched_budget_pseeds, _policy_seeds,
               _probe_ladder, _decode_search, _runtime, _reinforce,
               _ablation_2x2, _policy_seeds_indist, _discvis_quality,
               _runtime_classical, _minor_cells, _remaining,
               _significance, _invariance, _canonical, _editor, _supervised,
               _linear_probe, _reward_estimator, _verbal):
        fn()
    return CLAIMS


if __name__ == "__main__":
    cs = build()
    print(f"{len(cs)} claims registered")
    from collections import Counter
    for g, k in sorted(Counter(c.group for c in cs).items()):
        print(f"  {g:<24} {k}")
