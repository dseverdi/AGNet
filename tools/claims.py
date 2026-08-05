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
    # Probe TRAINING targets on the train split. The de-leak cut guards against
    # pre-dedup *evaluation* sources; these files hold no evaluation split, and
    # train/eval leakage is handled by deduplicating the eval splits against
    # train, not by rebuilding train. Used only for the BCE class weight.
    "data/ls_trajectories_train.pkl": "probe training targets, train split only",
    **{f"data/ls_trajectories_train_pseed{s}.pkl":
       "probe training targets, train split only" for s in (11, 22, 33)},
    "paper/data/po_agp_training.json": "policy checkpoints only, no probe",
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
    "results/eval_editor_test362_t08.json":
        "learned editor on the canonical dev_test 362; independent of the probe",
    "results/eval_editor_test362_st0p9.json":
        "learned editor (aggressive) on the canonical dev_test 362",
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
    for key in ("seed", "probe_t020"):
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
    """Retired: tab_operating_curve became fig:operating_curve.

    The 14x4 table is now a three-panel figure (t on x, mean +- std bands). Its 60
    cells were registered here against tab_operating_curve.tex; check 1 tests text
    presence in paper.tex or a table file and cannot read numbers rendered inside a
    PDF, so those registrations are gone. The figure is guarded instead by check 7
    (fig_operating_curve newer than its matched_budget/ sources) and the values the
    prose still quotes stay registered in _remaining()/_final_gaps().
    """
    return


def _matched_budget():
    g = "tab_ablation_thresholds.tex"
    arms = ("full", "noseed", "noenc", "coords")
    srcs = [f"paper/data/matched_budget/{a}_{s}.json"
            for a in arms for s in ("1234", 11, 22, 33)]
    cur = {a: [_mb_curve(a, s) for s in ("1234", 11, 22, 33)] for a in arms}
    # the full swept grid, matching GRID in build_matched_threshold_table.py:
    # panel (a) prints every threshold it sweeps, so every one is registered
    for t in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
              0.45, 0.50, 0.55, 0.60, 0.70, 0.80):
        for a in arms:
            add(id=f"mb.a.{a}.t{t}.opt", target=g, sources=srcs, n=362,
                run="probe_seeds_1234_11_22_33", group="mb_panel_a",
                typeset=f"{mean(c[t][1] for c in cur[a]):.2f}")
            # Register the full "$mean \pm std$" cell, as panel (b) does. Registering
            # the mean alone left every seed-spread in panel (a) unverified, which is
            # exactly the quantity the "strict and seed-stable" claim rests on.
            add(id=f"mb.a.{a}.t{t}.fail", target=g, sources=srcs, n=362,
                run="probe_seeds_1234_11_22_33", group="mb_panel_a",
                typeset=f"${mean(c[t][3] for c in cur[a]):.0f} \\pm "
                        f"{pstd([c[t][3] for c in cur[a]]):.0f}$")
    for anchor in (1.95, 1.7, 1.5, 1.3):
        for a in arms:
            v = [_interp(c.values(), 1, 3, anchor) for c in cur[a]]
            v = [x for x in v if x is not None]
            add(id=f"mb.b.{a}.{anchor}", target=g, sources=srcs, n=362,
                run="probe_seeds_1234_11_22_33", group="mb_panel_b",
                typeset=f"${mean(v):.0f} \\pm {pstd(v):.0f}$")


def _matched_budget_pseeds():
    """Panel (c): cost multiple per policy, reported as a range over tail levels.

    The panel used to be a 4x4 policy-by-tail grid, but within a policy the
    multiple is near-constant (spread <= 0.09) and three cells were unreachable,
    so it now shows one min--max range per policy. These claims therefore register
    the range endpoints, not each cell.
    """
    # Panel (c) was removed from the table (its columns were policies, not the
    # four conditions the header names); the multiples are now quoted in the
    # text of sec:res-headline-ablation, so these claims target paper.tex.
    g = "paper.tex"
    tails = (30, 50, 76, 100)
    base = {a: [_mb_curve(a, s) for s in ("1234", 11, 22, 33)] for a in ("full", "noenc")}
    vals = []
    for t in tails:
        f = [_interp(c.values(), 3, 1, t) for c in base["full"]]
        n_ = [_interp(c.values(), 3, 1, t) for c in base["noenc"]]
        if all(x is not None for x in f + n_):
            vals.append(mean(n_) / mean(f))
    src1234 = [f"paper/data/matched_budget/{a}_{s}.json"
               for a in ("full", "noenc") for s in ("1234", 11, 22, 33)]
    # The per-policy percentages are no longer printed: the prose states them as
    # magnitudes ("around a third", "close to four-fifths") because there is no
    # table for them to be read from. Both bounds stay verified here.
    allv = list(vals)
    for s_ in SEEDS:
        fc = _mb_curve("full", f"pseed{s_}", "matched_budget_pseeds").values()
        nc = _mb_curve("noenc", f"pseed{s_}", "matched_budget_pseeds").values()
        allv += [b_ / a_ for t in tails
                 for a_, b_ in [(_interp(fc, 3, 1, t), _interp(nc, 3, 1, t))] if a_ and b_]
    Sall = src1234 + [f"paper/data/matched_budget_pseeds/{a}_pseed{s_}.json"
                      for a in ("full", "noenc") for s_ in SEEDS]
    add(id="mb.c.min_extra", target="paper.tex", sources=Sall, n=362,
        run="policies_1234_11_22_33", group="mb_panel_c",
        value=(min(allv) - 1) * 100, interval=(28, 36),
        phrase="from about $+30\\%$ to $+81\\%$ depending on the policy")
    add(id="mb.c.max_extra", target="paper.tex", sources=Sall, n=362,
        run="policies_1234_11_22_33", group="mb_panel_c",
        value=(max(allv) - 1) * 100, interval=(72, 85),
        phrase="from about $+30\\%$ to $+81\\%$ depending on the policy")


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
                # |S|/OPT: on ood-large this averages the 206 polygons with an
                # optimum, not all 285; the caption says so.
                add(id=f"pseeds.{tag}.{s}.{arm}.cov", target=g,
                    sources=[f"results/policy_seeds_ood/pseed{s}_{arm}_{split}.json"],
                    n=N, run=f"policy{s}", group=f"pseeds_{tag}",
                    typeset=f"{c['cov']:.3f}")
                if c.get("opt"):
                    add(id=f"pseeds.{tag}.{s}.{arm}.opt", target=g,
                        sources=[f"results/policy_seeds_ood/pseed{s}_{arm}_{split}.json"],
                        n=N, run=f"policy{s}", group=f"pseeds_{tag}",
                        typeset=f"{c['opt']:.2f}")
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
            add(id=f"pseeds.{tag}.{s}.policy.cov", target=g,
                sources=[f"results/policy_seeds_ood/pseed{s}_full_{split}.json"],
                n=N, run=f"policy{s}", group=f"pseeds_{tag}",
                typeset=f"{sd['cov']:.3f}")
            if sd.get("opt"):
                add(id=f"pseeds.{tag}.{s}.policy.opt", target=g,
                    sources=[f"results/policy_seeds_ood/pseed{s}_full_{split}.json"],
                    n=N, run=f"policy{s}", group=f"pseeds_{tag}",
                    typeset=f"{sd['opt']:.2f}")


# ====================================================== hand-curated + prose sources
def _feature_baseline():
    """The feature-set control: coordinates / analytic geometry / encoder / both.

    Recomputed from the raw per-fold ROC and PR values, never from the printed
    table. The `embedding` arm must reproduce encoder_linear_probe's 0.8430 to four
    decimals -- if the harness ever drifts, that arm stops matching and the whole
    comparison is void, so it is registered against both files.
    """
    d = json.loads((D / "geometric_baseline_probe.json").read_text())
    src = ["paper/data/geometric_baseline_probe.json"]
    arms = {"coords": "coords", "geometry": "geometry",
            "embedding": "embedding", "embgeo": "embedding+geometry"}
    for tag, key in arms.items():
        a = d["arms"][key]
        for nm, mk, sk in (("roc", "roc_auc_mean", "roc_auc_std"),
                           ("pr", "pr_auc_mean", "pr_auc_std")):
            add(id=f"featbase.{tag}.{nm}", target="tab_feature_baseline.tex",
                sources=src, n=d["n_polygons"], run="feature_baseline_cv5",
                group="feature_baseline",
                typeset=f"{mean([a[mk]]):.3f}\\,$\\pm$\\,{a[sk]:.3f}")
    g, c, e = (d["arms"][k]["roc_auc_mean"] for k in ("geometry", "coords", "embedding"))
    both = d["arms"]["embedding+geometry"]["roc_auc_mean"]
    # The prose used to restate 0.537 / 0.753 / 0.872 immediately above the table that
    # shows them. It now states the relations instead, so only the DERIVED figures --
    # the 71% share and the every-fold and vs-attention comparisons -- are registered
    # against paper.tex; the levels themselves are checked in the table.
    add(id="featbase.pct_of_gain", target="paper.tex", sources=src,
        n=d["n_polygons"], run="feature_baseline_cv5", group="feature_baseline",
        value=(g - c) / (e - c) * 100, interval=(68, 74),
        phrase="about $71\\%$ of the encoder's gain")
    # "an improvement in every fold" -- the claim that makes +0.03 more than noise
    pf_e = d["arms"]["embedding"]["roc_auc_per_fold"]
    pf_b = d["arms"]["embedding+geometry"]["roc_auc_per_fold"]
    add(id="featbase.gain_every_fold", target="paper.tex", sources=src,
        n=d["n_polygons"], run="feature_baseline_cv5", group="feature_baseline",
        value=float(sum(1 for a_, b_ in zip(pf_e, pf_b) if b_ > a_)), interval=(5, 5),
        phrase="improves the readout in every fold")
    # The geometry gain vs what ATTENTION buys: mlp -> full, not linear -> full.
    # The first draft compared it to linear -> full (+0.086) and the gate caught that
    # the claim was false by -0.056 before it reached the PDF.
    _l = {r["name"]: r for r in json.loads((D / "probe_ladder.json").read_text())["rows"]}
    _pr = ((_l["mlp"]["pr_auc_mean"] - _l["linear"]["pr_auc_mean"])
           / (_l["full"]["pr_auc_mean"] - _l["linear"]["pr_auc_mean"])) * 100
    add(id="ladder.mlp_share_pr", target="paper.tex",
        sources=["paper/data/probe_ladder.json"], n=362, run="ladder",
        group="ladder_share", value=_pr, interval=(77, 83),
        phrase="the MLP recovers about $80\\%$ of the span")
    lad = json.loads((D / "probe_ladder.json").read_text())["rows"]
    r = {x["name"]: x["roc_auc_mean"] for x in lad}
    add(id="featbase.vs_attention_rungs", target="paper.tex", sources=src + [
        "paper/data/probe_ladder.json"], n=d["n_polygons"],
        run="feature_baseline_cv5", group="feature_baseline",
        value=(both - e) / (r["full"] - r["mlp"]), interval=(2.0, 2.6),
        phrase="more than twice what the two attention rungs")
    _e = d["arms"]["embedding"]["roc_auc_mean"]
    add(id="featbase.angle_share", target="paper.tex", sources=src,
        n=d["n_polygons"], run="feature_baseline_cv5", group="feature_baseline",
        value=(d["arms"]["embedding+angle_only"]["roc_auc_mean"] - _e)
              / (d["arms"]["embedding+geometry"]["roc_auc_mean"] - _e) * 100,
        interval=(45, 60), phrase="recovers about half of that gain")
    # The Discussion attributes the rotation deficit to a missing interior-angle
    # readout. That needs the angle-ALONE-on-top arm, not the all-ten arm -- the
    # first draft cited the wrong one and the arm was run to back the sentence.
    pf_ang = d["arms"]["embedding+angle_only"]["roc_auc_per_fold"]
    add(id="featbase.angle_on_top_every_fold", target="paper.tex", sources=src,
        n=d["n_polygons"], run="feature_baseline_cv5", group="feature_baseline",
        value=float(sum(1 for a_, b_ in zip(pf_e, pf_ang) if b_ > a_)),
        interval=(5, 5),
        phrase="appending the interior angle, a rigid-motion invariant, to the "
               "embedding improves the readout")
    # the dim column
    for tag, key in arms.items():
        add(id=f"featbase.{tag}.dim", target="tab_feature_baseline.tex", sources=src,
            n=d["n_polygons"], run="feature_baseline_cv5", group="feature_baseline",
            typeset=f"& {d['arms'][key]['n_features']} &")


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
    src = ["paper/data/invariance_test.json"]
    gate = {k: s[k]["n_polygons"] - s[k]["n_below_gate"]
            for k in ("identity", "rot45", "rot90", "rot180", "mirror_x", "reindex")}
    # The paper quotes the identity baseline and the RANGE over the five transforms
    # rather than listing each one. Registering only the two endpoints would leave the
    # three interior transforms unverified -- a later change to rot90 could break
    # "between 205 and 312" with nothing to catch it -- so the endpoints are registered
    # as typeset strings AND every transform is bounds-checked against that range.
    add(id="inv.identity", target="paper.tex", sources=src, n=362, run="policy1234",
        group="invariance", typeset=f"${gate['identity']}/362$")
    tr = {k: v for k, v in gate.items() if k != "identity"}
    add(id="inv.range_lo", target="paper.tex", sources=src, n=362, run="policy1234",
        group="invariance", typeset=f"${min(tr.values())}$")
    add(id="inv.range_hi", target="paper.tex", sources=src, n=362, run="policy1234",
        group="invariance", typeset=f"${max(tr.values())}$")
    for k, v in tr.items():
        add(id=f"inv.{k}_in_range", target="paper.tex", sources=src, n=362,
            run="policy1234", group="invariance",
            value=v, interval=(min(tr.values()), max(tr.values())),
            phrase="between $205$ and $312$")


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
    # Both arms re-run on the canonical 362-polygon dev_test split (2026-07-31).
    # The superseded runs (results/eval_t08.json, editor_sweep/v2/st0p9_cgnone.json)
    # were scored on an ad-hoc 300-polygon sample that straddled the dev/test
    # boundary (92 test polygons + 208 others), which flattered the coverage
    # figure: on the clean split the best config LOSES coverage rather than
    # preserving it. See eval_editor.py --restrict-to-names.
    a = res("eval_editor_test362_t08.json")["summary"]
    b = res("eval_editor_test362_st0p9.json")["summary"]
    add(id="editor.cut_best", target="paper.tex",
        sources=["results/eval_editor_test362_t08.json"],
        n=a["n_polygons"], run="editor_t08_test362", group="editor",
        value=(1 - a["ed_size_mean"] / a["seed_size_mean"]) * 100, interval=(15.0, 16.9),
        phrase="cuts $16\\%$ of the guards")
    # WHAT recovery IS, because the prose once got it wrong: eval_editor.py sets
    # recovery = (ed_r - seed_r) / (ls_r - seed_r) on the SCALAR reward
    # cov - lam*k/n - tau_penalty*max(0, tau-cov), and leaves it undefined where LS
    # did not improve the seed (291 of 362 polygons here). So >= 1.0 means "recovered
    # LS's reward gain", NOT "matched LS on coverage and guard count separately" --
    # only 54 of those 97 polygons do that. Do not restate this as a two-axis claim.
    add(id="editor.recovery", target="paper.tex",
        sources=["results/eval_editor_test362_t08.json"],
        n=a["n_polygons"], run="editor_t08_test362", group="editor",
        typeset=f"${a['frac_recovery_ge_1.0']:.2f}$")
    add(id="editor.cut_aggr", target="paper.tex",
        sources=["results/eval_editor_test362_st0p9.json"], n=b["n_polygons"],
        run="editor_st0p9_test362", group="editor_aggressive",
        value=(1 - b["ed_size_mean"] / b["seed_size_mean"]) * 100, interval=(21.5, 22.9),
        phrase="up to $22\\%$")
    add(id="editor.cov_aggr", target="paper.tex",
        sources=["results/eval_editor_test362_st0p9.json"], n=b["n_polygons"],
        run="editor_st0p9_test362", group="editor_aggressive",
        value=b["ed_cov_mean"], interval=(0.9325, 0.9345),
        phrase="dropping the mean to $0.933$")
    add(id="editor.recovery_med_aggr", target="paper.tex",
        sources=["results/eval_editor_test362_st0p9.json"], n=b["n_polygons"],
        run="editor_st0p9_test362", group="editor_aggressive",
        value=b["recovery_median"], interval=(-0.45, -0.30),
        phrase="median recovery $-0.38$")


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
    """The cross-validated linear probe.

    Its four figures used to be restated in a footnote and again in the Discussion.
    Both now point at the encoder row of tab_feature_baseline, which carries the same
    measurement, so registering them against paper.tex would only verify prose that
    no longer exists. The qualitative form the text does still make -- "reaches
    ROC-AUC approx 0.84" -- is `approx.roc084` below, which keeps this file live.

    Note the two runs are re-derivations, not the same numbers: this file has
    ROC 0.8430377, the feature-baseline harness 0.8429581, agreeing to three
    decimals. Do not assert equality beyond the precision printed.
    """


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
    # These three sat on the exemption list, so they were never checked against the
    # data at all -- the rollout count and, more importantly, the tau-gate asymmetry,
    # which is the one real defect this item reports. Registering them.
    add(id="rew.n_rollouts", target="paper.tex",
        sources=["results/reward_estimator_agreement.json"], n=s["n_polygons"],
        run="reward_agreement", group="reward_est_count",
        typeset=f"{s['n_rollouts_scored']}")
    TAU = s["reward"]["tau"]
    rr = res("reward_estimator_agreement.json")["rows"]
    for tag, cnt in (("tau_up", sum(1 for r in rr
                                    if r["cov_disc"] >= TAU > r["cov_exact"])),
                     ("tau_down", sum(1 for r in rr
                                      if r["cov_exact"] >= TAU > r["cov_disc"]))):
        add(id=f"rew.{tag}", target="paper.tex",
            sources=["results/reward_estimator_agreement.json"],
            n=s["n_rollouts_scored"], run="reward_agreement",
            group="reward_est_gate", typeset=f"{cnt}")


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
        phrase="coords-only about $1.75\\times$")
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
    # Policy 1234 is no longer a row in this table -- its numbers would be verbatim
    # copies of tab_headline/tab_ood/tab_large, which is why it was dropped. Those
    # tables register it; nothing to register here.
    for sd in SEEDS:
        src = [f"results/policy_seeds/pseed{sd}_{a}.json" for a in ("full", "noenc")]
        for lab, arm, key in (("policy", "full", "seed"), ("full", "full", "cell"),
                              ("noenc", "noenc", "cell")):
            d = res(f"policy_seeds/pseed{sd}_{arm}.json")
            src1 = d["seed"] if key == "seed" else d["cells"]["t=0.2|K=1"]
            add(id=f"pseeds.test.{sd}.{lab}.chv", target=g, sources=src, n=362,
                run=f"policy{sd}", group="pseeds_test",
                typeset=f"{src1['chv']:.3f}")
            add(id=f"pseeds.test.{sd}.{lab}.cov", target=g, sources=src, n=362,
                run=f"policy{sd}", group="pseeds_test",
                typeset=f"{src1['cov']:.3f}")
            if src1.get("opt"):
                add(id=f"pseeds.test.{sd}.{lab}.opt", target=g, sources=src, n=362,
                    run=f"policy{sd}", group="pseeds_test",
                    typeset=f"{src1['opt']:.2f}")
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
        # the std is typeset alongside the mean; register it so it is not left
        # riding on a coincidental collision with some other claim's value
        add(id=f"dvq.{n}.exact_sd", target="tab_discvis_quality.tex",
            sources=["results/discvis_greedy_timing.json"], n=b["n_polys"],
            run="discvis_timing", group=f"discvis_quality_{n}",
            typeset=f"{b['exact_coverage_std']:.3f}")
        add(id=f"dvq.{n}.gap_sd", target="tab_discvis_quality.tex",
            sources=["results/discvis_greedy_timing.json"], n=b["n_polys"],
            run="discvis_timing", group=f"discvis_quality_{n}",
            typeset=f"{b['gap_std']:.3f}")


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
    # t=0.25 / t=0.30 rows were dropped from tab_dist_shift once
    # fig:operating_curve took over the threshold sweep.
    for lab, recs, key in (("seed", dev, "seed"), ("full", dev, "probe_t020"),
                           ("noenc", ne, "probe_t020")):
        add(id=f"dist_shift.{lab}.mincov", target="tab_dist_shift.tex",
            sources=["paper/data/dist_dev_test.json",
                     "paper/data/dist_dev_test_noenc.json"],
            n=362, run="policy1234", group="dist_shift_min",
            typeset=f"{min(r[key]['cov'] for r in recs):.3f}")
    for lab, recs, key in (("seed", dev, "seed"),):
        add(id=f"dist_shift.{lab}.eq1", target="tab_dist_shift.tex",
            sources=["paper/data/dist_dev_test.json"], n=362, run="policy1234",
            group="dist_shift_min",
            typeset=f"& {sum(1 for r in recs if r[key]['cov']>=0.99995)} &")
    # tab_reinforce was cut: the PO/BT row's numbers matched no run on disk, and
    # no epoch-30 PO checkpoint exists to match the REINFORCE baselines' budget
    # (the one epoch-40 candidate is from a collapsed pre-fix run). The objective
    # choice is now argued from the saturation mechanism in sec:method-pointer,
    # with the development-run comparison stated qualitatively and no numbers.


def _remaining():
    """Cells and prose values not covered by any other group."""
    # The policy-1234 rows were dropped from tab_policy_seeds: with cov and |S|/OPT
    # present they restated tab_headline/tab_ood/tab_large verbatim. Those tables
    # register the released policy; nothing to register for it here.
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
    # The worked-example coverages and |S|/OPT are drawn ON the figure panels by
    # build_figures.py and are no longer restated in the caption, which would only
    # duplicate them. Check 1 tests presence in paper.tex or a table file, so it
    # cannot see numbers rendered inside a PDF; these values are instead guarded by
    # check 7, which requires fig_worked_example to be newer than
    # worked_examples.json. Registering them against paper.tex would fail, and
    # exempting them from the numeral audit would be wrong -- they are not in the
    # text to audit.
    inv = json.loads((D / "invariance_test.json").read_text())["summary"]
    # The identity worst-polygon coverage was exempt, so the "0.857 -> 0.382"
    # contrast had only its right-hand side checked. Both ends now registered.
    add(id="inv.identity_worst", target="paper.tex",
        sources=["paper/data/invariance_test.json"], n=362, run="policy1234",
        group="invariance_worst", typeset=f"{inv['identity']['min_cov']:.3f}")
    add(id="inv.worst_min", target="paper.tex",
        sources=["paper/data/invariance_test.json"], n=362, run="policy1234",
        group="invariance_cov",
        typeset=f"{min(v['min_cov'] for k,v in inv.items() if k!='identity'):.3f}")
    full = [json.loads((D / "matched_budget" / f"full_{s}.json").read_text())
            for s in ("1234", 11, 22, 33)]
    c50 = [d["cells"]["t=0.5|K=1"] for d in full]
    # the t=0.50 mean coverage is no longer restated in prose (it is in
    # tab_operating_curve, which registers it); only the gate count is quoted.
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
    # Was the point value 1495x. The prose now says "two to three orders of
    # magnitude", so bound the whole measured range inside [100, 10000) rather than
    # pinning one endpoint -- see prose.precompute_order for the same reasoning.
    for tag, v in (("lo", min(ratios)), ("hi", max(ratios))):
        add(id=f"runtime.exact_gpu_order_{tag}", target="paper.tex",
            sources=["results/classical_timing_lazy.json", "results/probe_timing.json"],
            n=None, run="timing", group="runtime_ratio",
            value=v, interval=(100.0, 10000.0),
            phrase="exact greedy runs two to three orders of magnitude above the learned pass")
    # The prose no longer quotes the n>198 rates (they duplicated the whole-split
    # ones two sentences earlier); it claims they are UNCHANGED. So verify that
    # relation directly: the largest gap between the subset rate and the
    # whole-split rate, over both the policy and the probe, must stay under a
    # percentage point. A bare numeral could not have caught the claim drifting.
    big = [r for r in ood if r["n"] > 198]
    gaps = []
    for which in ("seed", "probe_t020"):
        sub = mean(sum(1 for r in poly(f) if r["n"] > 198 and r[which]["cov"] >= 0.95)
                   / len(big) for f in OOD)
        allp = mean(sum(1 for r in poly(f) if r[which]["cov"] >= 0.95)
                    / len(poly(f)) for f in OOD)
        gaps.append(abs(sub - allp) * 100)
    add(id="ood.n198_gap", target="paper.tex",
        sources=[f"paper/data/{f}" for f in OOD], n=len(big),
        run="probe_seeds_1234_11_22_33", group="ood_subset",
        value=max(gaps), interval=(0.0, 1.0),
        phrase="both rates are essentially unchanged")
    # The subset rate itself now appears only once, in the C2 contribution
    # statement; sec:res-ood states the relation instead (ood.n198_gap above).
    add(id="ood.n198_probe_pct", target="paper.tex",
        sources=[f"paper/data/{f}" for f in OOD], n=len(big),
        run="probe_seeds_1234_11_22_33", group="ood_subset",
        typeset=f"{mean(sum(1 for r in poly(f) if r['n']>198 and r['probe_t020']['cov']>=0.95)/len(big) for f in OOD)*100:.1f}")
    a = res("eval_editor_test362_t08.json")["summary"]
    add(id="editor.cov_best", target="paper.tex",
        sources=["results/eval_editor_test362_t08.json"],
        n=a["n_polygons"], run="editor_t08_test362", group="editor_cov",
        typeset=fmt(a["ed_cov_mean"], 3))
    add(id="editor.covseed", target="paper.tex",
        sources=["results/eval_editor_test362_t08.json"],
        n=a["n_polygons"], run="editor_t08_test362", group="editor_cov",
        typeset=f"{a['seed_cov_mean']:.3f}")


def _pos_weight():
    """The probe's BCE positive-class weight, recomputed from the shipped targets.

    Registered because this number went stale once: the paper carried 5.66, which
    is the ratio in ls_trajectories_train_strict.pkl -- a superseded target set
    used by exactly one probe checkpoint. All 24 headline/ablation/ladder probes
    train on ls_trajectories_train.pkl (5.77), and each policy seed generates its
    own targets (5.77-5.90). train_set_predictor.py derives it as
    n_removed / n_kept over the training records.
    """
    def ratio(fn):
        with open(REPO / "data" / fn, "rb") as fh:
            recs = pickle.load(fh)
        if isinstance(recs, dict):
            recs = recs.get("records", recs)
        kept = sum(len(set(r["final"])) for r in recs)
        rem = sum(r["n"] - len(set(r["final"])) for r in recs)
        return rem / kept

    released = ratio("ls_trajectories_train.pkl")
    add(id="posw.released", target="paper.tex",
        sources=["data/ls_trajectories_train.pkl"], n=8867,
        run="probe_targets_policy1234", group="pos_weight",
        value=released, interval=(5.765, 5.785), phrase="$5.77$ for the released")
    pseeds = [ratio(f"ls_trajectories_train_pseed{s}.pkl") for s in (11, 22, 33)]
    add(id="posw.pseed_hi", target="paper.tex",
        sources=[f"data/ls_trajectories_train_pseed{s}.pkl" for s in (11, 22, 33)],
        n=8867, run="probe_targets_policy_seeds", group="pos_weight",
        value=max(pseeds), interval=(5.895, 5.905), phrase="$5.90$")


def _po_training():
    """The saturation figures quoted in sec:method-pointer.

    Provenance, which the paper does not lean on for quality claims: per-epoch
    eval on the first 50 polygons of the 8000 TRAINING prefix (epoch_eval_k=50,
    train_size=8000 in configs/po_agp_transformer_bt.json), disc-vis approximate
    coverage, greedy decode. Training data under an approximate metric -- fine
    for "this run stopped trading one axis against the other", not comparable to
    the results tables. The figure built on these curves was cut for that reason.
    """
    d = json.loads((D / "po_training_curves.json").read_text())
    src = ["paper/data/po_training_curves.json"]
    s = d["seeds"]
    # The saturation illustration in sec:method-pointer ("one seed settling at
    # coverage 1.000 while its guard cost stays near 3.8x the optimum") was cut from
    # the prose, so these two have nothing left to check. The underlying behaviour is
    # still reported, and still verified, in the seed-22 paragraph of the Limitations
    # (potrain.s22_gr_from_e2, potrain.*_endpoint_reward, potrain.s22_rawcov_*).
    # The seed-22 limitation paragraph previously described an under-guarding reward
    # exploit ("under 2% of vertices", "epoch 2"). Both numbers contradicted these
    # curves -- |S|/n never falls below 0.563 -- and neither was registered, so the
    # gate never saw them. Registering every figure the paragraph now quotes.
    ep, gr, cov = s["22"]["epochs"], s["22"]["guard_ratio_greedy"], s["22"]["coverage_greedy"]
    # The per-epoch progression (0.563/0.928/0.996 at epochs 1/2/13) was condensed to
    # "above 0.92 from the second epoch onwards"; that bound is what the claim now checks.
    add(id="potrain.s22_gr_from_e2", target="paper.tex", sources=src, n=50,
        run="policy_seed22_log_trainprefix", group="po_training_s22",
        value=min(gr[ep.index(2):]), interval=(0.92, 1.0),
        phrase="above $0.92$ from the second epoch")
    add(id="potrain.s22_last_epoch", target="paper.tex", sources=src, n=50,
        run="policy_seed22_log_trainprefix", group="po_training_s22",
        typeset=f"epoch ${ep[-1]}$")
    # The estimated coverage seed 22 collapses onto was exempt, so the sentence's
    # subject ("reaches an estimated coverage of 1.000") went unchecked.
    add(id="potrain.s22_maxcov", target="paper.tex", sources=src, n=50,
        run="policy_seed22_log_trainprefix", group="po_training_s22",
        typeset=f"{max(cov):.3f}")
    # Reward at each run's endpoint under Eq.(reward): min(cov,tau) - lambda*|S|/n
    # - rho*max(0, tau-cov), with tau=0.99, lambda=1.0, rho=3.0.
    def _u(c, g):
        return min(c, 0.99) - 1.0 * g - 3.0 * max(0.0, 0.99 - c)
    for sd, nd in (("22", 3), ("11", 3), ("33", 3)):
        c, g = s[sd]["coverage_greedy"][-1], s[sd]["guard_ratio_greedy"][-1]
        add(id=f"potrain.{sd}_endpoint_reward", target="paper.tex", sources=src, n=50,
            run=f"policy_seed{sd}_log_trainprefix", group="po_training_reward",
            typeset=f"${fmt(_u(c, g), nd)}$" if _u(c, g) >= 0 else f"$-{fmt(-_u(c, g), nd)}$")
    # raw-coverage argmax: the checkpoint a coverage-only rule would select
    _am = max(range(len(cov)), key=lambda i: cov[i])
    add(id="potrain.s22_rawcov_argmax", target="paper.tex", sources=src, n=50,
        run="policy_seed22_log_trainprefix", group="po_training_s22",
        typeset=f"epoch ${ep[_am]}$")
    add(id="potrain.s22_rawcov_gr", target="paper.tex", sources=src, n=50,
        run="policy_seed22_log_trainprefix", group="po_training_s22",
        value=gr[_am] * 100, interval=(95.5, 96.9), phrase="$96\\%$ of the vertices")


def _c4_drift():
    """The fixed-point drift bounds quoted in sec:res-iteration, from the K-sweep."""
    d = json.loads((D / "setpred_iter_sweep.json").read_text())["cells"]
    src = ["paper/data/setpred_iter_sweep.json"]
    g = lambda t, k, f: d[f"t={t}|K={k}"][f]  # noqa: E731
    # t=0.5's stationarity was quoted three times over (span, opt K1->K2, cov
    # K1->K5) for one fact; the prose now keeps only the span, so the two
    # per-step numerals are no longer typeset. c4.t05_span below still anchors it.
    # All drift figures are now quoted K=1 -> K=5, at the two ENDS of the swept
    # range. The 2026-08-05 extension to t in [0.2, 0.8] showed the drift is not
    # monotone in t: it is minimised near t=0.5 and reverses sign below it, so the
    # headline threshold has to be measured, not extrapolated from the high end.
    for t, f, nd, tag in ((0.2, "opt", 3, "t02_opt_k15"),
                          (0.8, "opt", 3, "t08_opt_k15")):
        add(id=f"c4.{tag}", target="paper.tex", sources=src, n=362,
            run="iter_sweep_policy1234", group="c4_drift",
            typeset=f"${fmt(abs(g(t,5,f)-g(t,1,f)), nd)}$")
    add(id="c4.t02_cov_k15", target="paper.tex", sources=src, n=362,
        run="iter_sweep_policy1234", group="c4_drift",
        typeset=f"${fmt(abs(g(0.2,5,'cov')-g(0.2,1,'cov')), 4)}$")
    add(id="c4.t08_cov_k15", target="paper.tex", sources=src, n=362,
        run="iter_sweep_policy1234", group="c4_drift",
        typeset=f"${fmt(g(0.8,1,'cov')-g(0.8,5,'cov'), 4)}$")
    # The sign reversal itself, as a relation rather than a numeral: below t=0.5
    # extra passes ADD guards, above it they SHED them. This is what replaced the
    # withdrawn "at every threshold tested, five passes end with fewer guards".
    lo = min(g(t, 5, "opt") - g(t, 1, "opt") for t in (0.2, 0.25, 0.3))
    hi = max(g(t, 5, "opt") - g(t, 1, "opt") for t in (0.6, 0.65, 0.75, 0.8))
    add(id="c4.sign_reversal", target="paper.tex", sources=src, n=362,
        run="iter_sweep_policy1234", group="c4_drift",
        value=min(lo, -hi), interval=(0.0, 1.0),
        phrase="the extra passes end with slightly \\emph{more} guards")
    # No threshold Pareto-improves under iteration. This is the affirmative case
    # for the single pass: not just that drift is small, but that spending more
    # passes never buys coverage and cost together. Counts the violations, so the
    # claim fails the moment one appears.
    ALLT = sorted({float(k.split("|")[0].split("=")[1]) for k in d if "|K=" in k})
    pareto = sum(1 for t in ALLT
                 if g(t, 5, "cov") >= g(t, 1, "cov") - 1e-12
                 and g(t, 5, "opt") <= g(t, 1, "opt") + 1e-12)
    add(id="c4.no_pareto_gain", target="paper.tex", sources=src, n=362,
        run="iter_sweep_policy1234", group="c4_drift",
        value=float(pareto), interval=(0.0, 0.0),
        phrase="At no threshold does iterating improve both axes at once")
    span = max(g(0.5, k, "opt") for k in (1, 2, 3, 5)) - min(
        g(0.5, k, "opt") for k in (1, 2, 3, 5))
    add(id="c4.t05_span", target="paper.tex", sources=src, n=362,
        run="iter_sweep_policy1234", group="c4_drift",
        typeset=f"${fmt(span, 3)}$")


def _discvis_ceiling():
    """The hard ceiling: disc-vis greedy's estimate saturates at 1.0 once all M=500
    sample points are covered, so past that no guard count can help. Measured by
    tools/analyze_discvis_ceiling.py. This is the CONSEQUENCE claim -- the earlier
    correlation work established only that the gap tracks rounds, which left open
    whether a few more guards would close it. They cannot."""
    d = res("discvis_ceiling.json")
    rows, big = d["rows"], d["n_ge_800"]
    src = ["results/discvis_ceiling.json"]
    # Every polygon at n >= 200 fails the gate at its own ceiling.
    ge200 = [r for r in rows if r["n"] >= 200]
    add(id="dvceil.n200_up_all_fail", target="paper.tex", sources=src, n=len(ge200),
        run="discvis_ceiling", group="discvis_ceiling_n200",
        value=float(sum(1 for r in ge200 if r["clears_gate"])), interval=(0.0, 0.0),
        phrase="every polygon we tried from $n = 200$ upward falls short of the "
               "$0.99$ gate on exact coverage")
    # The stated 0.77--0.91 band at n >= 800, and that the estimate reads exactly 1.
    add(id="dvceil.exact_lo", target="paper.tex", sources=src, n=big["count"],
        run="discvis_ceiling", group="discvis_ceiling_ge800",
        typeset=f"{big['exact_min']:.2f}")
    add(id="dvceil.exact_hi", target="paper.tex", sources=src, n=big["count"],
        run="discvis_ceiling", group="discvis_ceiling_ge800",
        typeset=f"{big['exact_max']:.2f}")
    add(id="dvceil.est_saturates", target="paper.tex", sources=src, n=len(rows),
        run="discvis_ceiling", group="discvis_ceiling_all",
        value=min(r["disc_vis_at_ceiling"] for r in rows), interval=(1.0, 1.0),
        phrase="the search halts and reports coverage of exactly $1$")


def _discvis_correlations():
    """Spearman and partial-Spearman in the disc-vis footnote."""
    from scipy.stats import spearmanr
    rows = res("discvis_gap_correlation.json")["rows"]
    gp = np.array([r["gap"] for r in rows])
    k = np.array([r["n_guards"] for r in rows])
    nn = np.array([r["n"] for r in rows])
    src = ["results/discvis_gap_correlation.json"]

    def partial(a, b, c):
        ra, rb = spearmanr(a, c).correlation, spearmanr(b, c).correlation
        rab = spearmanr(a, b).correlation
        return (rab - ra * rb) / np.sqrt((1 - ra ** 2) * (1 - rb ** 2))

    for tag, v in (("sp_guards", spearmanr(gp, k).correlation),
                   ("sp_n", spearmanr(gp, nn).correlation),
                   ("partial_guards", partial(gp, k, nn)),
                   ("partial_n", partial(gp, nn, k))):
        add(id=f"dvcorr.{tag}", target="paper.tex", sources=src, n=len(rows),
            run="discvis_gap", group="discvis_corr", typeset=f"${v:.2f}$")
    # The load-bearing fact for the SELECTION-effect reading, which the Spearman
    # correlations alone cannot establish: _sample_points_in_polygon draws uniformly
    # inside the polygon, so for a FIXED guard set the point-fraction estimate is an
    # unbiased estimator of the area fraction and its error should be two-sided.
    # Every measured gap being positive is therefore not explicable by finite sample
    # resolution; it requires that the set (and the stopping time) were chosen using
    # those same points. Registered as a count so one negative gap would fail it.
    npos = sum(1 for r in rows if r["gap"] > 0)
    add(id="dvcorr.all_gaps_positive", target="paper.tex", sources=src, n=len(rows),
        run="discvis_gap", group="discvis_gap_sign",
        value=float(len(rows) - npos), interval=(0.0, 0.0),
        phrase="the gap is positive on every polygon we measured")
    # The two same-n polygons the footnote contrasts. These were never actually
    # checked: the completeness pass had been absorbing "+0.004" as a prefix of an
    # unrelated C4 numeral ("0.0045"), so the pair only surfaced once that numeral
    # was cut. Registering them by (n, guard count) pins them to the right rows.
    for tag, nv, kv in (("gap_lo", 500, 3), ("gap_hi", 500, 56)):
        r = next(r for r in rows if r["n"] == nv and r["n_guards"] == kv)
        add(id=f"dvcorr.{tag}", target="paper.tex", sources=src, n=len(rows),
            run="discvis_gap", group="discvis_gap_pair",
            typeset=f"$+{r['gap']:.3f}$")


def _wilson_uppers():
    """The upper bound printed beside each Wilson lower bound."""
    def hi(k, n, z=1.96):
        p = k / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return min(1.0, c + h)
    for tag, f, N, tgt in (("headline", "dist_dev_test.json", 362, "tab_headline.tex"),
                           ("ood", "dist_test_OOD.json", 2081, "tab_ood.tex")):
        recs = poly(f)
        add(id=f"{tag}.seed_wilson_hi", target=tgt, sources=[f"paper/data/{f}"],
            n=N, run="policy1234", group=f"{tag}_wilson",
            typeset=f"{hi(gate(recs,'seed'), N):.3f}")


def _final_gaps():
    """Values found by auditing the exemption list -- each is a measurement, so it is
    registered rather than exempted."""
    dev, ne = poly("dist_dev_test.json"), poly("dist_dev_test_noenc.json")
    Sd = ["paper/data/dist_dev_test.json", "paper/data/dist_dev_test_noenc.json"]
    # exact-coverage counts the other groups missed
    add(id="dist_shift.probe_t020.eq1", target="tab_dist_shift.tex", sources=Sd,
        n=362, run="policy1234", group="dist_shift_min",
        typeset=f"& {sum(1 for r in dev if r['probe_t020']['cov']>=0.99995)} &")
    add(id="dist_shift.noenc.eq1", target="tab_dist_shift.tex", sources=Sd, n=362,
        run="policy1234", group="dist_shift_min",
        typeset=f"& {sum(1 for r in ne if r['probe_t020']['cov']>=0.99995)} &")
    add(id="dist_shift.seed.ge0999", target="tab_dist_shift.tex", sources=Sd, n=362,
        run="policy1234", group="dist_shift_min",
        typeset=f"& {sum(1 for r in dev if r['seed']['cov']>=0.999)} &")
    add(id="dist_shift.noenc.ge0999", target="tab_dist_shift.tex", sources=Sd, n=362,
        run="policy1234", group="dist_shift_min",
        typeset=f"& {sum(1 for r in ne if r['probe_t020']['cov']>=0.999)} &")
    # The full probe's own >=0.999 count was the one cell in this table with no
    # registration: its value (128) collided with the exempt embedding dimension, so
    # completeness never flagged it. Exemptions can shadow real measurements -- prefer
    # registering a cell over trusting that some other numeral covers it.
    add(id="dist_shift.probe_t020.ge0999", target="tab_dist_shift.tex", sources=Sd,
        n=362, run="policy1234", group="dist_shift_min",
        typeset=f"& {sum(1 for r in dev if r['probe_t020']['cov']>=0.999)} &")
    # The $\ge 0.95$ gate column was unregistered against this table: its values
    # (293/362, 345/362) are registered against tab_headline and against the
    # invariance prose, and completeness pools numerals across targets, so the cells
    # here were satisfied by a match somewhere else entirely.
    for lab, recs, key in (("seed", dev, "seed"), ("full", dev, "probe_t020"),
                           ("noenc", ne, "probe_t020")):
        add(id=f"dist_shift.{lab}.gate", target="tab_dist_shift.tex", sources=Sd,
            n=362, run="policy1234", group="dist_shift_gate",
            typeset=f"{sum(1 for r in recs if r[key]['cov']>=0.95)}/362")
    # The prose used to close the McNemar sentence with "a net $69 \to 17$", which
    # is just 362 minus the two gate counts the table already prints. Both marginals
    # stay verified as dist_shift.{seed,full}.gate; what the prose keeps is the part
    # the table cannot show -- the 57/5 discordant split behind the net figure.
    # The count came back in the caveats paragraph, where "a handful" was used for
    # seventeen polygons -- an understatement that flattered the full probe -- so it
    # is spelled out and registered.
    add(id="prose.full_below_gate_word", target="paper.tex", sources=Sd, n=362,
        run="policy1234", group="dist_shift_prose",
        value=float(below(dev, 'probe_t020')), interval=(17, 17),
        phrase="leaves seventeen below it")
    # The dagger footnote used to quote the LS oracle's mean exact coverage as
    # "$0.995$", which is just tab_headline's 0.9948 rounded. The footnote now says
    # "slightly undercovers" and lets the table's Mean cov column carry the value,
    # which `headline.ls_cov` already verifies against the same raw records.
    # C2 percentages
    ood = poly("dist_test_OOD.json")
    add(id="prose.ood_policy_pct", target="paper.tex",
        sources=["paper/data/dist_test_OOD.json"], n=2081, run="policy1234",
        group="c2_pct", typeset=f"${gate(ood,'seed')/len(ood)*100:.0f}\\%$")
    add(id="prose.ood_probe_pct", target="paper.tex",
        sources=[f"paper/data/{f}" for f in OOD], n=2081,
        run="probe_seeds_1234_11_22_33", group="c2_pct_probe",
        typeset=f"${mean(gate(poly(f),'probe_t020')/2081 for f in OOD)*100:.0f}\\%$")
    # capacity ladder: fraction of the linear-to-full gain the attention-free MLP recovers
    rows = {r["name"]: r for r in json.loads((D / "probe_ladder.json").read_text())["rows"]}
    lin, mlp, full = (rows["linear"]["roc_auc_mean"], rows["mlp"]["roc_auc_mean"],
                      rows["full"]["roc_auc_mean"])
    add(id="prose.ladder_mlp_share", target="paper.tex",
        sources=["paper/data/probe_ladder.json"], n=362, run="ladder",
        group="ladder_share",
        typeset=f"${(mlp-lin)/(full-lin)*100:.0f}\\%$")
    # cost multiples in percent form
    cur = {a: [_mb_curve(a, sd) for sd in ("1234", 11, 22, 33)]
           for a in ("full", "noenc", "coords")}
    S = [f"paper/data/matched_budget/{a}_{sd}.json"
         for a in cur for sd in ("1234", 11, 22, 33)]
    for arm, tag in (("noenc", "noenc"), ("coords", "coords")):
        r = mean([_interp_mean_x(cur[arm], t) / _interp_mean_x(cur["full"], t)
                  for t in (50, 76, 100)])
        want = {"noenc": (42, 48), "coords": (72, 78)}[tag]
        add(id=f"prose.pct_more_{tag}", target="paper.tex", sources=S, n=362,
            run="probe_seeds_1234_11_22_33", group=f"pct_more_{tag}",
            value=(r - 1) * 100, interval=want,
            # The paper states these as multiples, not percentages; the phrase must
            # match text that is actually in paper.tex or check 1 orphans the claim.
            phrase=("about $1.45\\times$ the full probe's guards" if tag == "noenc"
                    else "coords-only about $1.75\\times$"))
    # reward-estimator percentages
    rs = res("reward_estimator_agreement.json")
    sm = rs["summary"]
    add(id="prose.rew_optimistic_pct", target="paper.tex",
        sources=["results/reward_estimator_agreement.json"], n=sm["n_polygons"],
        run="reward_agreement", group="reward_pct",
        typeset=f"${sm['coverage_bias_disc_minus_exact']['frac_disc_optimistic']*100:.0f}\\%$")
    big = [r for r in rs["rows"] if abs(r["cov_disc"] - r["cov_exact"]) > 0.10]
    agree = 100.0 if not big else 0.0
    add(id="prose.rew_bigfap_agree", target="paper.tex",
        sources=["results/reward_estimator_agreement.json"], n=sm["n_polygons"],
        run="reward_agreement", group="reward_pct",
        typeset=f"${agree:.0f}\\%$")
    # tab_discvis_quality: the disc-vis estimate column
    for b in res("discvis_greedy_timing.json")["buckets"]:
        add(id=f"dvq.{b['bucket']}.disc", target="tab_discvis_quality.tex",
            sources=["results/discvis_greedy_timing.json"], n=b["n_polys"],
            run="discvis_timing", group=f"discvis_gap_{b['bucket']}",
            typeset=f"{b['disc_vis_coverage_mean']:.3f}")
    # precompute-only runtime ratios
    cl_ = res("classical_timing_lazy.json")["classical"]
    pt = {(r["device"], r["n"]): r for r in res("probe_timing.json")["rows"]}
    Sr = ["results/classical_timing_lazy.json", "results/probe_timing.json",
          "results/discvis_greedy_timing.json"]
    dv = {b["bucket"]: b for b in res("discvis_greedy_timing.json")["buckets"]}
    # sec:res-runtime used to quote every one of these ratios as a point range
    # (20--28x, 56--74x, 220--1495x, 16--28x, 57--75x). The section is a cost-profile
    # comparison, not a benchmark, so the prose now states orders of magnitude and
    # the table carries the times. Bound the ORDER instead of the value, so the
    # verbal claim is still checked: "one to two orders" means the whole measured
    # range stays inside [10, 100), and so on. A regression that moved a ratio
    # across a decade boundary would fail here rather than pass silently.
    ex = [r["vis_ms_mean"] / pt[("cuda", r["bucket"])]["total_ms"] for r in cl_]
    dvr = [dv[b]["disc_vis_s_mean"] * 1000 / pt[("cuda", b)]["total_ms"]
           for b in dv if ("cuda", b) in pt]
    add(id="prose.precompute_order", target="paper.tex", sources=Sr, n=None,
        run="timing", group="precompute_ratio",
        value=min(ex + dvr), interval=(10.0, 100.0),
        phrase="one to two orders of magnitude more than the entire learned pass")
    add(id="prose.precompute_order_hi", target="paper.tex", sources=Sr, n=None,
        run="timing", group="precompute_ratio",
        value=max(ex + dvr), interval=(10.0, 100.0),
        phrase="one to two orders of magnitude more than the entire learned pass")


def _approximations():
    """Rounded restatements in prose of values registered precisely elsewhere.
    Registered as intervals so the rounding is checked rather than exempted."""
    cur = {a: [_mb_curve(a, sd) for sd in ("1234", 11, 22, 33)]
           for a in ("full", "noenc", "coords", "noseed")}
    S = [f"paper/data/matched_budget/{a}_{sd}.json"
         for a in cur for sd in ("1234", 11, 22, 33)]
    m = lambda a, t, i: mean(c[t][i] for c in cur[a])  # noqa: E731
    add(id="approx.noseed_mult", target="paper.tex", sources=S, n=362,
        run="probe_seeds_1234_11_22_33", group="approx_mult",
        value=mean([_interp_mean_x(cur["noseed"], t) / _interp_mean_x(cur["full"], t)
                    for t in (50, 76, 100)]), interval=(1.05, 1.15),
        phrase="no-seed about $1.1\\times$")
    # The "economical regime ($|S|/\OPT$ 1.6--1.9)" sentence and the "$2.9\times$
    # (seed only) / $4.6\times$ (neither)" gloss were removed from the paper; the four
    # values they carried (1.62 / 1.94 / 2.94 / 4.64) are still verified against
    # tab_ablation, tab_ablation_thresholds and tab_headline, so nothing goes
    # unchecked here -- these four registrations were pointing at dead prose.
    # The "1.5 to 5.2x" range is over the THREE REPLICATION policies at all four
    # budgets (eleven reachable cells) -- not, as this claim previously computed,
    # over the released policy's probe seeds at the single budget 1.5. The old
    # version checked 2.578 against (1.9, 2.7) and passed while guarding nothing in
    # the sentence it was attached to; it also let a wrong upper figure (5.4 for a
    # true 5.19) stand. Both endpoints are now recomputed from the cells quoted.
    ratios = []
    for sd in SEEDS:
        fc = _mb_curve("full", f"pseed{sd}", "matched_budget_pseeds").values()
        nc = _mb_curve("noenc", f"pseed{sd}", "matched_budget_pseeds").values()
        for anc in (1.95, 1.70, 1.50, 1.30):
            a_, b_ = _interp(fc, 1, 3, anc), _interp(nc, 1, 3, anc)
            if a_ and b_:
                ratios.append(b_ / a_)
    Sp = [f"paper/data/matched_budget_pseeds/{a}_pseed{sd}.json"
          for a in ("full", "noenc") for sd in SEEDS]
    add(id="approx.tail_ratio_lo", target="paper.tex", sources=Sp, n=362,
        run="policy_seeds_11_22_33", group="approx_tail_ratio",
        value=min(ratios), interval=(1.45, 1.55),
        phrase="compared at the same guard budget, the full probe leaves the shorter tail")
    add(id="approx.tail_ratio_hi", target="paper.tex", sources=Sp, n=362,
        run="policy_seeds_11_22_33", group="approx_tail_ratio",
        value=max(ratios), interval=(5.15, 5.25),
        phrase="the ablation leaves several times as many polygons")
    add(id="approx.tail_ratio_cells", target="paper.tex", sources=Sp, n=362,
        run="policy_seeds_11_22_33", group="approx_tail_ratio",
        value=float(len(ratios)), interval=(11, 11),
        phrase="compared at the same guard budget, the full probe leaves the shorter tail")
    # 0.83M total parameters = pointer + probe
    lad = {r["name"]: r for r in json.loads((D / "probe_ladder.json").read_text())["rows"]}
    add(id="approx.total_params", target="paper.tex",
        sources=["paper/data/probe_ladder.json"], n=None, run="architecture",
        group="params", value=(lad["full"]["n_params"] + 364162) / 1e6,
        interval=(0.82, 0.84), phrase="${\\approx}\\,0.83$M parameters")
    # The single-precision footprint (3.3 MB) is no longer restated: it is a
    # mechanical 4x of the parameter count above, and sec:res-runtime is a cost
    # comparison rather than a hardware spec.
    # rounded ROC-AUC and the t=0.70 coverage
    lp = json.loads((D / "encoder_linear_probe.json").read_text())
    add(id="approx.roc084", target="paper.tex",
        sources=["paper/data/encoder_linear_probe.json"], n=lp["n_polygons"],
        run="linear_probe_cv", group="approx_roc", value=lp["roc_auc_mean"],
        interval=(0.835, 0.845), phrase="$\\mathrm{ROC\\text{-}AUC} \\approx 0.84$")
    full = [json.loads((D / "matched_budget" / f"full_{sd}.json").read_text())
            for sd in ("1234", 11, 22, 33)]
    add(id="approx.t070_cov", target="paper.tex",
        sources=[f"paper/data/matched_budget/full_{sd}.json" for sd in ("1234", 11, 22, 33)],
        n=362, run="probe_seeds_1234_11_22_33", group="approx_t070",
        value=mean(d["cells"]["t=0.7|K=1"]["cov"] for d in full), interval=(0.9655, 0.9670),
        # The paper now makes this claim qualitatively; the interval is what keeps the
        # word "converged" honest (0.9664 probe against the seed's 0.9689).
        phrase="the probe has converged to the policy seed on both axes")


def build() -> list[Claim]:
    CLAIMS.clear()
    for fn in (_headline, _dist_shift, _pareto, _ood, _large, _operating_curve,
               _matched_budget, _matched_budget_pseeds, _policy_seeds,
               _probe_ladder, _feature_baseline, _decode_search, _runtime,
               _ablation_2x2, _policy_seeds_indist, _discvis_quality,
               _runtime_classical, _minor_cells, _remaining,
               _c4_drift, _po_training, _pos_weight, _discvis_ceiling, _discvis_correlations, _wilson_uppers, _final_gaps, _approximations,
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
