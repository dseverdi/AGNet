"""Build LaTeX tables in paper/tables/ from JSON results in paper/data/.

Run: python paper/scripts/build_tables.py
Output: paper/tables/tab_*.tex
Each table is a `tabular` environment ready to \\input{} from paper.tex.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABLES = ROOT / "tables"
TABLES.mkdir(exist_ok=True)


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text())


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def write(name: str, body: str) -> None:
    out = TABLES / name
    out.write_text(body)
    print(f"wrote {out.relative_to(ROOT.parent)}")


# ---------- helpers for per-polygon aggregation ----------
def _agg_per_poly(records: list[dict], method_key: str | None = None) -> dict:
    """Aggregate a list of per-polygon records into headline statistics.

    If method_key is given (e.g. "seed", "probe_t020"), pulls S_size / cov
    from records[i][method_key]. Otherwise records are flat dicts with
    S_size / cov directly (e.g. classical greedy / LS rows).
    """
    if not records:
        return {"n": 0, "mean_cov": 0.0, "n_below_095": 0,
                "mean_S_n": 0.0, "mean_S_opt": None}

    def pick(r):
        return r[method_key] if method_key else r

    covs = [pick(r)["cov"] for r in records]
    sn = [pick(r)["S_size"] / r["n"] for r in records]
    sopt = [pick(r)["S_size"] / r["OPT"]
            for r in records if r.get("OPT")]
    return {
        "n": len(records),
        "mean_cov": sum(covs) / len(covs),
        "n_below_095": sum(1 for c in covs if c < 0.95),
        "mean_S_n": sum(sn) / len(sn),
        "mean_S_opt": (sum(sopt) / len(sopt)) if sopt else None,
    }


def _row(name: str, agg: dict, n_total: int, *,
         bold_cov: bool = False, sopt_dagger: bool = False) -> str:
    """Render one row of the headline table from an _agg_per_poly dict.

    Feasibility proportion is reported as count/N. Wilson CI is shown only
    when n_below_095 > 0 (i.e. the proportion is not at the upper degenerate
    boundary); for deterministic rows that reach N/N the CI collapses to
    (.990, 1.000) and adds no information.
    """
    n_ok = n_total - agg["n_below_095"]
    cov_s = f"{agg['mean_cov']:.4f}"
    if bold_cov:
        cov_s = f"\\textbf{{{cov_s}}}"
    sn_s = f"{agg['mean_S_n']:.4f}"
    if agg["mean_S_opt"] is None:
        sopt_s = "--"
    else:
        sopt_s = f"{agg['mean_S_opt']:.4f}"
        if sopt_dagger:
            sopt_s += "$^{\\dagger}$"
    if agg["n_below_095"] > 0:
        lo, hi = wilson_ci(n_ok, n_total)
        feas_s = f"{n_ok}/{n_total}\\,({lo:.3f},{hi:.3f})"
    else:
        feas_s = f"{n_ok}/{n_total}"
    return f"  {name} & {cov_s} & {feas_s} & {sn_s} & {sopt_s} \\\\"


# ---------- tab_headline: dev_test consolidated comparison ----------
def _multiseed_row(name: str, summary: dict, method: str, n_total: int,
                   *, bold_cov: bool = False) -> str:
    """Render a row from multi_seed_summary.json: mean ± std across seeds."""
    sm = summary[method]
    cov = sm["mean_cov_mean"]
    cov_sd = sm["mean_cov_std"]
    sn = sm["mean_S_over_n_mean"]
    sn_sd = sm["mean_S_over_n_std"]
    sopt = sm["mean_S_over_OPT_mean"]
    sopt_sd = sm["mean_S_over_OPT_std"]
    nbelow = sm["n_below_095_mean"]
    nbelow_sd = sm["n_below_095_std"]
    n_ok = n_total - nbelow
    cov_s = f"{cov:.4f}\\,$\\pm$\\,{cov_sd:.4f}"
    if bold_cov:
        cov_s = f"\\textbf{{{cov:.4f}}}\\,$\\pm$\\,{cov_sd:.4f}"
    feas_s = f"${n_ok:.1f} \\pm {nbelow_sd:.1f}$/{n_total}"
    sn_s = f"{sn:.4f}\\,$\\pm$\\,{sn_sd:.4f}"
    sopt_s = "--" if sopt is None else f"{sopt:.4f}\\,$\\pm$\\,{sopt_sd:.4f}"
    return f"  {name} & {cov_s} & {feas_s} & {sn_s} & {sopt_s} \\\\"


def _noenc_row(name: str, ms_split: dict, n_total: int, fallback_file: str) -> str:
    """No-encoder row: 4-seed mean ± std if the multi-seed summary has it,
    otherwise the single-seed value from `fallback_file` (graceful fallback so
    the table still builds before the extra seeds are run)."""
    ne = ms_split.get("no_encoder_summary")
    if ne and ms_split.get("n_seeds_no_encoder", 0) >= 2:
        return _multiseed_row(name, ne, "probe_t020", n_total)
    agg = _agg_per_poly(load(fallback_file)["polygons"], "probe_t020")
    return _row(name, agg, n_total)


def tab_headline() -> None:
    """Six-row headline: classical greedy / LS, policy seed, full probe
    (4-seed mean ± std), no-encoder probe. All on dev_test (367 polygons).
    Classical/seed/no-encoder rows are deterministic at the chosen seed;
    the full-probe row averages over seeds {1234, 11, 22, 33}.
    """
    dev = load("dist_dev_test.json")["polygons"]
    cls = load("baseline_classical.json")["splits"]["dev_test"]["per_polygon"]
    ms = load("multi_seed_summary.json")["splits"]["dev_test"]
    n = len(dev)
    assert n == 367, f"expected 367 dev_test polygons, got {n}"

    seed_agg   = _agg_per_poly(dev,   "seed")
    greedy_agg = _agg_per_poly(cls["greedy"])
    ls_agg     = _agg_per_poly(cls["ls"])
    assert ms["n_seeds"] == 4, f"expected 4 probe seeds, got {ms['n_seeds']}"

    lines = [
        "% auto-generated by build_tables.py from dist_dev_test{,_noenc}.json"
        " + baseline_classical.json + multi_seed_summary.json",
        "\\begin{tabular}{lrrrr}",
        "  \\toprule",
        "  Method & Mean cov & $\\#\\{\\mathrm{Cov}\\ge .95\\}/N$"
        " & Mean $|S|/n$ & Mean $|S|/\\OPT$ \\\\",
        "  \\midrule",
        "  \\multicolumn{5}{l}{\\emph{Classical baselines}} \\\\",
        _row("\\quad Greedy", greedy_agg, n),
        _row("\\quad Local search", ls_agg, n, sopt_dagger=True),
        "  \\midrule",
        "  \\multicolumn{5}{l}{\\emph{Learned policy / probe}} \\\\",
        _row("\\quad Pretrained pointer (seed)", seed_agg, n),
        _multiseed_row(
            "\\quad SetPredictor full ($t=0.20$)",
            ms["summary"], "probe_t020", n, bold_cov=True),
        _noenc_row("\\quad SetPredictor, no-encoder ($t=0.20$)",
                   ms, n, "dist_dev_test_noenc.json"),
        "  \\bottomrule",
        "\\end{tabular}",
        "",
    ]
    write("tab_headline.tex", "\n".join(lines))


# ---------- tab_dist_shift: per-polygon coverage distribution ----------
def _dist_from_per_poly(records: list[dict], method_key: str) -> dict:
    """Coverage-threshold counts from per-polygon records, using the same
    thresholds as eval_set_predictor.py._dist() (Cov=1 is cov>=0.99995) so a
    row built here is methodologically identical to the setpred_*.json rows."""
    covs = [r[method_key]["cov"] for r in records]
    return {
        "n_cov_eq_1":    sum(1 for c in covs if c >= 0.99995),
        "n_cov_ge_0999": sum(1 for c in covs if c >= 0.999),
        "n_cov_ge_099":  sum(1 for c in covs if c >= 0.99),
        "n_cov_ge_095":  sum(1 for c in covs if c >= 0.95),
        "cov_min":       min(covs),
    }


def tab_dist_shift() -> None:
    d = load("setpred_dev_test.json")
    n = d["n_polygons"]
    # No-encoder ablation row (single seed) from the per-polygon dump, counted
    # with the identical threshold convention as the setpred_*.json rows so the
    # encoder's contribution is readable at the 0.99 gate -- where it lives --
    # rather than only at the saturated 0.95 gate of Table~\ref{tab:headline}.
    noenc = load("dist_dev_test_noenc.json")["polygons"]
    noenc_dist = _dist_from_per_poly(noenc, "probe_t020")
    rows = [("Pretrained pointer (seed)", d["seed"]["dist"]),
            ("SetPredictor $t=0.20$ (full)", d["cells"]["t=0.2|K=1"]["dist"]),
            ("SetPredictor $t=0.20$, no-encoder", noenc_dist),
            ("SetPredictor $t=0.25$ (full)", d["cells"]["t=0.25|K=1"]["dist"]),
            ("SetPredictor $t=0.30$ (full)", d["cells"]["t=0.3|K=1"]["dist"])]
    lines = [
        "% auto-generated by build_tables.py from data/setpred_dev_test.json"
        " + dist_dev_test_noenc.json",
        "\\begin{tabular}{lrrrrr}",
        "  \\toprule",
        "  Method & $\\#\\{\\mathrm{Cov}=1\\}$ & $\\#\\{\\mathrm{Cov}\\ge .999\\}$ & "
        "$\\#\\{\\mathrm{Cov}\\ge .99\\}$ & $\\#\\{\\mathrm{Cov}\\ge .95\\}$ & min $\\mathrm{Cov}$ \\\\",
        "  \\midrule",
    ]
    for name, dist in rows:
        lines.append(
            f"  {name} & {dist['n_cov_eq_1']} & {dist['n_cov_ge_0999']} & "
            f"{dist['n_cov_ge_099']} & {dist['n_cov_ge_095']}/{n} & "
            f"{dist['cov_min']:.3f} \\\\"
        )
    lines += ["  \\bottomrule", "\\end{tabular}", ""]
    write("tab_dist_shift.tex", "\n".join(lines))


# ---------- tab_pareto: threshold sweep on dev_test ----------
def tab_pareto() -> None:
    d = load("setpred_dev_test.json")
    n = d["n_polygons"]
    cells = d["cells"]
    lines = [
        "% auto-generated by build_tables.py from data/setpred_dev_test.json",
        "\\begin{tabular}{lrrrr}",
        "  \\toprule",
        "  Threshold $t$ & Mean $\\mathrm{Cov}$ & Mean $|S|/n$ & Mean $|S|/\\OPT$ & "
        "$\\#\\{\\mathrm{Cov}\\ge .95\\}/N$ \\\\",
        "  \\midrule",
    ]
    order = ["t=0.2|K=1", "t=0.25|K=1", "t=0.3|K=1"]
    label = {"t=0.2|K=1": "0.20", "t=0.25|K=1": "0.25", "t=0.3|K=1": "0.30"}
    for k in order:
        c = cells[k]
        lines.append(
            f"  {label[k]} & {c['cov']:.4f} & {c['chv']:.4f} & "
            f"{c['opt']:.4f} & {c['dist']['n_cov_ge_095']}/{n} \\\\"
        )
    lines += ["  \\bottomrule", "\\end{tabular}", ""]
    write("tab_pareto.tex", "\n".join(lines))


# ---------- tab_fixed_point: K=1,2,3,5 from iter sweep ----------
def tab_fixed_point() -> None:
    d = load("setpred_iter_sweep.json")
    cells = d["cells"]
    t_show = "0.65"
    lines = [
        "% auto-generated by build_tables.py from data/setpred_iter_sweep.json",
        "\\begin{tabular}{lrrr}",
        "  \\toprule",
        "  Inference passes $K$ & Mean $\\mathrm{Cov}$ & Mean $|S|/n$ & Mean $|S|/\\OPT$ \\\\",
        "  \\midrule",
    ]
    for K in [1, 2, 3, 5]:
        key = f"t={t_show}|K={K}"
        if key in cells:
            c = cells[key]
            lines.append(
                f"  $K={K}$ & {c['cov']:.4f} & {c['chv']:.4f} & {c['opt']:.4f} \\\\"
            )
    lines += ["  \\bottomrule", "\\end{tabular}", ""]
    write("tab_fixed_point.tex", "\n".join(lines))


# ---------- tab_ood_test: OOD on test/ split ----------
def tab_ood_test() -> None:
    d = load("setpred_test_OOD.json")
    n = d["n_polygons"]
    seed = d["seed"]
    cell_t02 = d["cells"]["t=0.2|K=1"]
    cell_t025 = d["cells"]["t=0.25|K=1"]
    cell_t03 = d["cells"]["t=0.3|K=1"]
    lines = [
        "% auto-generated by build_tables.py from data/setpred_test_OOD.json",
        "\\begin{tabular}{lrrrr}",
        "  \\toprule",
        "  Method & Mean $\\mathrm{Cov}$ & $\\#\\{\\mathrm{Cov}\\ge .95\\}/N$ & "
        "min $\\mathrm{Cov}$ & Mean $|S|/\\OPT$ \\\\",
        "  \\midrule",
    ]
    rows = [
        ("Pretrained pointer (seed)", seed["cov"], seed["dist"], seed["opt"]),
        ("SetPredictor $t=0.20$", cell_t02["cov"], cell_t02["dist"], cell_t02["opt"]),
        ("SetPredictor $t=0.25$", cell_t025["cov"], cell_t025["dist"], cell_t025["opt"]),
        ("SetPredictor $t=0.30$", cell_t03["cov"], cell_t03["dist"], cell_t03["opt"]),
    ]
    for name, cov, dist, opt in rows:
        opt_s = f"{opt:.4f}" if opt is not None else "--"
        lines.append(
            f"  {name} & {cov:.4f} & {dist['n_cov_ge_095']}/{n} & "
            f"{dist['cov_min']:.3f} & {opt_s} \\\\"
        )
    lines += ["  \\bottomrule", "\\end{tabular}", ""]
    write("tab_ood_test.tex", "\n".join(lines))


# ---------- tab_reinforce: training-objective comparison (PO/BT vs REINFORCE) ----------
def tab_reinforce() -> None:
    """Compact REINFORCE-vs-BT comparison from curated baseline_reinforce.json.
    Geometric coverage (polygon_coverage_stats.mean) and |S|/OPT.
    """
    d = load("baseline_reinforce.json")
    lines = [
        "% auto-generated by build_tables.py from data/baseline_reinforce.json",
        "\\begin{tabular}{lrr}",
        "  \\toprule",
        "  Training objective & Mean cov & Mean $|S|/\\OPT$ \\\\",
        "  \\midrule",
    ]
    for r in d["rows"]:
        cov = f"{r['cov']:.3f}"
        sopt = f"{r['S_over_OPT']:.2f}"
        if r.get("bold"):
            cov = f"\\textbf{{{cov}}}"
            sopt = f"\\textbf{{{sopt}}}"
        lines.append(f"  {r['name']} & {cov} & {sopt} \\\\")
    lines += ["  \\bottomrule", "\\end{tabular}", ""]
    write("tab_reinforce.tex", "\n".join(lines))


# ---------- tab_ood: headline-style OOD comparison (test split, 2107) ----------
def tab_ood() -> None:
    """OOD analogue of tab_headline: policy seed, full probe (4-seed mean),
    no-encoder probe, on the test split (2107 polygons)."""
    dev = load("dist_test_OOD.json")["polygons"]
    ms = load("multi_seed_summary.json")["splits"]["test_OOD"]
    n = len(dev)
    assert n == 2107, f"expected 2107 OOD polygons, got {n}"
    assert ms["n_seeds"] == 4, f"expected 4 probe seeds, got {ms['n_seeds']}"
    seed_agg = _agg_per_poly(dev, "seed")
    lines = [
        "% auto-generated by build_tables.py from dist_test_OOD{,_noenc}.json"
        " + multi_seed_summary.json",
        "\\begin{tabular}{lrrrr}",
        "  \\toprule",
        "  Method & Mean cov & $\\#\\{\\mathrm{Cov}\\ge .95\\}/N$"
        " & Mean $|S|/n$ & Mean $|S|/\\OPT$ \\\\",
        "  \\midrule",
        _row("Pretrained pointer (seed)", seed_agg, n),
        _multiseed_row("SetPredictor full ($t=0.20$)",
                       ms["summary"], "probe_t020", n, bold_cov=True),
        _noenc_row("SetPredictor, no-encoder ($t=0.20$)",
                   ms, n, "dist_test_OOD_noenc.json"),
        "  \\bottomrule",
        "\\end{tabular}",
        "",
    ]
    write("tab_ood.tex", "\n".join(lines))


def _large_noenc_row(ne: dict, n: int) -> str:
    """No-encoder row for the extreme-OOD table: 4-seed mean ± std when present,
    single value otherwise. Reads the *_mean keys written by make_extreme_ood_data."""
    ok = n - ne["n_below_095_mean"]
    if ne.get("n_seeds", 1) >= 2:
        return (
            "  SetPredictor, no-encoder ($t=0.20$) & "
            f"{ne['cov_mean']:.4f}\\,$\\pm$\\,{ne['cov_std']:.4f} & "
            f"{ne['cov_min_mean']:.3f}\\,$\\pm$\\,{ne['cov_min_std']:.3f} & "
            f"${ok:.1f} \\pm {ne['n_below_095_std']:.1f}$/{n} & "
            f"{ne['S_n_mean']:.4f}\\,$\\pm$\\,{ne['S_n_std']:.4f} \\\\"
        )
    return (
        f"  SetPredictor, no-encoder ($t=0.20$) & {ne['cov_mean']:.4f} & "
        f"{ne['cov_min_mean']:.3f} & {ok:.0f}/{n} & {ne['S_n_mean']:.4f} \\\\"
    )


# ---------- tab_large: extreme-OOD on the large split (285, n=600-2250) ----------
def tab_large() -> None:
    """Extreme-OOD comparison on the `large` split. Leads with the worst case
    (min Cov) and the feasibility count -- the metrics that separate the frozen
    encoder from the no-encoder ablation. |S|/OPT is omitted (only 206/285 have
    an ILP optimum); coverage, min-Cov, feasibility, and |S|/n are over all 285.
    """
    d = load("extreme_ood.json")
    n = d["n_polygons"]
    sd = d["seed"]
    pr = d["probe_t020"]
    ne = d["no_encoder_t020"]
    n_seeds = d["n_seeds"]

    def feas(below: float) -> str:
        return f"{n - below:.0f}/{n}"

    # Probe (multi-seed) feasibility carries the across-seed std on the count.
    pr_ok = n - pr["n_below_095_mean"]

    lines = [
        "% auto-generated by build_tables.py from data/extreme_ood.json",
        "\\begin{tabular}{lrrrr}",
        "  \\toprule",
        "  Method & Mean $\\mathrm{Cov}$ & $\\min \\mathrm{Cov}$ & "
        "$\\#\\{\\mathrm{Cov}\\ge .95\\}/N$ & Mean $|S|/n$ \\\\",
        "  \\midrule",
        f"  Pretrained pointer (seed) & {sd['cov']:.4f} & {sd['cov_min']:.3f} & "
        f"{feas(sd['n_below_095'])} & {sd['S_n']:.4f} \\\\",
        f"  SetPredictor full ($t=0.20$) & "
        f"\\textbf{{{pr['cov_mean']:.4f}}}\\,$\\pm$\\,{pr['cov_std']:.4f} & "
        f"{pr['cov_min_mean']:.3f}\\,$\\pm$\\,{pr['cov_min_std']:.3f} & "
        f"${pr_ok:.1f} \\pm {pr['n_below_095_std']:.1f}$/{n} & "
        f"{pr['S_n_mean']:.4f}\\,$\\pm$\\,{pr['S_n_std']:.4f} \\\\",
        _large_noenc_row(ne, n),
        "  \\bottomrule",
        "\\end{tabular}",
        "",
    ]
    write("tab_large.tex", "\n".join(lines))


# ---------- tab_dataset_partition: hand-curated partition statistics ----------
def tab_dataset_partition() -> None:
    # Numbers come from paper.md §5 (dataset size distribution).
    body = (
        "% hand-curated dataset partition table\n"
        "\\begin{tabular}{lrrr}\n"
        "  \\toprule\n"
        "  Split & \\# polygons & $n$ range & Use \\\\\n"
        "  \\midrule\n"
        "  train      & 9{,}791 & 8--150  & Pointer + SetPredictor training \\\\\n"
        "  dev\\_tune & 857     & 8--150  & SetPredictor checkpoint selection \\\\\n"
        "  dev\\_test & 367     & 8--150  & Held-out evaluation (in-distribution) \\\\\n"
        "  test       & 2{,}107 & 12--1{,}000 & Out-of-distribution evaluation \\\\\n"
        "  large      & 285     & 600--2{,}250 & Extreme-OOD evaluation \\\\\n"
        "  \\bottomrule\n"
        "\\end{tabular}\n"
    )
    write("tab_dataset_partition.tex", body)


if __name__ == "__main__":
    tab_headline()
    tab_dist_shift()
    tab_fixed_point()
    tab_dataset_partition()
    tab_reinforce()   # PO/BT vs REINFORCE (Results §6.1)
    tab_ood()         # headline-style OOD comparison (Results §6.5)
    tab_large()       # extreme-OOD on the large split (Results §6.5)
    tab_pareto()      # cost-of-feasibility threshold sweep (Results §6.4)
    # tab_ood_test() retained above for reproducibility (superseded by tab_ood()).
