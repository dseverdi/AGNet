"""Build PDF figures in paper/gfx/setpred/ from JSON results in paper/data/.

Run: python paper/scripts/build_figures.py
Output: paper/gfx/setpred/fig_*.pdf

Figures:
- fig_coverage_cdf.pdf   : stair-step CDF of per-polygon coverage, two panels (in-dist + OOD)
- fig_pareto.pdf         : Pareto curve (mean cov vs mean |S|/OPT), two panels (in-dist + OOD)
- fig_kinvariance.pdf    : K-invariance of iterative inference (mean cov vs K, one line per t)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "gfx" / "setpred"
OUT.mkdir(parents=True, exist_ok=True)

# Palette from tools/plot_value_net_proxy.py for visual consistency.
COLORS = {
    "seed":  "#577590",  # blue-grey -- pretrained pointer
    "t020":  "#f3722c",  # orange   -- aggressive threshold
    "t025":  "#43aa8b",  # green    -- middle threshold
    "t030":  "#277da1",  # blue     -- conservative threshold
}

# Common matplotlib defaults for journal-quality output.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text())


def save(fig, name: str) -> None:
    out = OUT / name
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT.parent)}")


def empirical_cdf_points(dist: dict, n: int) -> tuple[list[float], list[float]]:
    """Construct a coarse empirical complementary CDF from the summary stats
    available in each `dist` block. Returns (x, y) where y is the fraction of
    polygons with coverage >= x. Stair-step rendering is left to the caller.
    """
    cov_min = dist["cov_min"]
    cov_p01 = dist["cov_p01"]
    cov_p05 = dist["cov_p05"]
    cov_p10 = dist["cov_p10"]
    n_ge_095 = dist["n_cov_ge_095"]
    n_ge_099 = dist["n_cov_ge_099"]
    n_ge_0999 = dist["n_cov_ge_0999"]
    n_eq_1 = dist["n_cov_eq_1"]
    # (x_coverage, y_fraction_with_cov_at_least_x), sorted by x ascending
    pts = [
        (cov_min, 1.0),
        (cov_p01, 0.99),
        (cov_p05, 0.95),
        (cov_p10, 0.90),
        (0.95,  n_ge_095 / n),
        (0.99,  n_ge_099 / n),
        (0.999, n_ge_0999 / n),
        (1.0,   n_eq_1 / n),
    ]
    pts.sort(key=lambda p: p[0])
    # Enforce monotone non-increasing y as x increases (it must be by definition)
    cleaned = []
    last_y = 1.0
    for x, y in pts:
        y = min(y, last_y)
        cleaned.append((x, y))
        last_y = y
    xs = [p[0] for p in cleaned]
    ys = [p[1] for p in cleaned]
    return xs, ys


def fig_coverage_cdf() -> None:
    """Two-panel coverage CDF: dev_test (367) and OOD test (2107)."""
    d_in = load("setpred_dev_test.json")
    d_ood = load("setpred_test_OOD.json")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    panels = [
        ("In-distribution (dev$\\_$test, 367 polygons)", d_in, axes[0]),
        ("Out-of-distribution (test, 2107 polygons)", d_ood, axes[1]),
    ]
    series = [
        ("Pretrained pointer (seed)", "seed",  d_in["seed"]["dist"],  d_ood["seed"]["dist"]),
        ("SetPredictor $t=0.30$",      "t030", d_in["cells"]["t=0.3|K=1"]["dist"],  d_ood["cells"]["t=0.3|K=1"]["dist"]),
        ("SetPredictor $t=0.25$",      "t025", d_in["cells"]["t=0.25|K=1"]["dist"], d_ood["cells"]["t=0.25|K=1"]["dist"]),
        ("SetPredictor $t=0.20$",      "t020", d_in["cells"]["t=0.2|K=1"]["dist"],  d_ood["cells"]["t=0.2|K=1"]["dist"]),
    ]
    n_in = d_in["n_polygons"]
    n_ood = d_ood["n_polygons"]

    for label, ckey, dist_in, dist_ood in series:
        xs_in,  ys_in  = empirical_cdf_points(dist_in,  n_in)
        xs_ood, ys_ood = empirical_cdf_points(dist_ood, n_ood)
        axes[0].step(xs_in,  ys_in,  where="post", color=COLORS[ckey], label=label, linewidth=1.6)
        axes[1].step(xs_ood, ys_ood, where="post", color=COLORS[ckey], label=label, linewidth=1.6)

    for title, _, ax in panels:
        ax.set_xlabel("Per-polygon coverage $\\mathrm{Cov}$")
        ax.set_xlim(0.6, 1.005)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.axvline(0.95, color="gray", linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("Fraction of polygons with $\\mathrm{Cov}\\geq x$")
    axes[1].legend(loc="lower left", frameon=False)
    save(fig, "fig_coverage_cdf.pdf")


def fig_pareto() -> None:
    """Two-panel Pareto: in-distribution and OOD. X = mean |S|/OPT, Y = mean cov."""
    d_tune = load("setpred_dev_tune.json")   # broader in-dist threshold sweep
    d_test = load("setpred_dev_test.json")   # canonical in-dist eval (3 thresholds)
    d_ood  = load("setpred_test_OOD.json")   # OOD eval (3 thresholds)

    def collect(d, t_keys):
        xs, ys, labels = [], [], []
        for tkey in t_keys:
            if tkey not in d["cells"]:
                continue
            c = d["cells"][tkey]
            if c.get("opt") is None:
                continue
            xs.append(c["opt"])
            ys.append(c["cov"])
            # extract threshold value from "t=0.X|K=1"
            t_str = tkey.split("|")[0].replace("t=", "")
            labels.append(t_str)
        return xs, ys, labels

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    # ---- panel 1: in-distribution ----
    ax = axes[0]
    # broader sweep from dev_tune (857 polys); plot as a faint curve
    xs_tune, ys_tune, labels_tune = collect(
        d_tune,
        ["t=0.2|K=1", "t=0.25|K=1", "t=0.3|K=1", "t=0.35|K=1",
         "t=0.4|K=1", "t=0.5|K=1", "t=0.65|K=1"],
    )
    ax.plot(xs_tune, ys_tune, "-", color=COLORS["t030"], alpha=0.45,
            label="SetPredictor (dev$\\_$tune, t sweep)", linewidth=1.4)
    ax.scatter(xs_tune, ys_tune, color=COLORS["t030"], s=22, alpha=0.7, zorder=3)

    # canonical 3 thresholds on dev_test
    xs_test, ys_test, labels_test = collect(d_test, ["t=0.2|K=1", "t=0.25|K=1", "t=0.3|K=1"])
    ax.plot(xs_test, ys_test, "o-", color=COLORS["t020"], markersize=6,
            label="SetPredictor (dev$\\_$test)", linewidth=1.8, zorder=4)

    # pretrained pointer (seed) on dev_test
    seed = d_test["seed"]
    ax.scatter(seed["opt"], seed["cov"], marker="^", s=70, color=COLORS["seed"],
               label="Pretrained pointer", zorder=5, edgecolor="black", linewidth=0.6)

    # annotate threshold values next to the dev_test points
    for x, y, lab in zip(xs_test, ys_test, labels_test):
        ax.annotate(f"$t={lab}$", (x, y), xytext=(6, -10), textcoords="offset points",
                    fontsize=7, color="dimgray")

    ax.set_xlabel("Mean $|S|/\\mathrm{OPT}$ (lower is better)")
    ax.set_ylabel("Mean coverage $\\mathrm{Cov}$ (higher is better)")
    ax.set_title("In-distribution (dev$\\_$test / dev$\\_$tune)")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(loc="lower right", frameon=False)

    # ---- panel 2: OOD ----
    ax = axes[1]
    xs_ood, ys_ood, labels_ood = collect(d_ood, ["t=0.2|K=1", "t=0.25|K=1", "t=0.3|K=1"])
    ax.plot(xs_ood, ys_ood, "o-", color=COLORS["t020"], markersize=6,
            label="SetPredictor (test)", linewidth=1.8, zorder=4)
    seed_ood = d_ood["seed"]
    ax.scatter(seed_ood["opt"], seed_ood["cov"], marker="^", s=70,
               color=COLORS["seed"], label="Pretrained pointer", zorder=5,
               edgecolor="black", linewidth=0.6)
    for x, y, lab in zip(xs_ood, ys_ood, labels_ood):
        ax.annotate(f"$t={lab}$", (x, y), xytext=(6, -10), textcoords="offset points",
                    fontsize=7, color="dimgray")

    ax.set_xlabel("Mean $|S|/\\mathrm{OPT}$ (lower is better)")
    ax.set_title("Out-of-distribution (test, 2107 polygons)")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(loc="lower right", frameon=False)

    save(fig, "fig_pareto.pdf")


def fig_kinvariance() -> None:
    """K-invariance: cov vs K for several thresholds. Lines visually flat = fixed point."""
    d = load("setpred_iter_sweep.json")
    cells = d["cells"]

    thresholds = ["0.5", "0.6", "0.65", "0.7", "0.75", "0.8"]
    Ks = [1, 2, 3, 5]

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    palette = ["#577590", "#43aa8b", "#90be6d", "#f9c74f", "#f8961e", "#f3722c"]
    for t, col in zip(thresholds, palette):
        covs = []
        for K in Ks:
            key = f"t={t}|K={K}"
            if key in cells:
                covs.append(cells[key]["cov"])
            else:
                covs.append(None)
        ax.plot(Ks, covs, "o-", color=col, label=f"$t={t}$", linewidth=1.5, markersize=5)

    ax.set_xlabel("Number of inference passes $K$")
    ax.set_ylabel("Mean coverage $\\mathrm{Cov}$")
    ax.set_title("Iterative inference is a fixed point (full dev, 1224 polygons)")
    ax.set_xticks(Ks)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(loc="lower right", frameon=False, ncol=2, title="Threshold")
    # tighten the y-range so the (small) variations are visible but not exaggerated
    all_covs = [cells[f"t={t}|K={K}"]["cov"] for t in thresholds for K in Ks
                if f"t={t}|K={K}" in cells]
    ymin = min(all_covs) - 0.003
    ymax = max(all_covs) + 0.003
    ax.set_ylim(ymin, ymax)

    save(fig, "fig_kinvariance.pdf")


if __name__ == "__main__":
    fig_coverage_cdf()
    fig_pareto()
    fig_kinvariance()
