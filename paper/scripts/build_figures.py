"""Build PDF figures in paper/gfx/setpred/ from JSON results in paper/data/.

Run: python paper/scripts/build_figures.py
Output: paper/gfx/setpred/fig_*.pdf

Figures in the current paper layout (5):
- fig_po_training.pdf    : PO/BT training dynamics (held-out dev)
- fig_worked_example.pdf : two polygons, seed vs probe vs optimum
- fig_distributions.pdf  : 2x3 grid, {dev_test, OOD} x {coverage CCDF, |S|/n, |S|/OPT}
- fig_cov_vs_n.pdf       : per-polygon OOD coverage vs polygon size n
- fig_mechanism.pdf      : (a) K-invariance fixed point, (b) encoder PCA separation

Deprecated single-panel builders (fig_coverage_cdf, fig_pareto, fig_kinvariance,
fig_encoder_pca) remain defined for reproducibility but are not dispatched; their
content now lives in fig_distributions and fig_mechanism.
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


# ──────────────────────────────────────────────────────────────────────
#  New figures: training curve, worked example, stretch figures
#
#  Each new figure reads a paper/data/*.json file with a documented schema
#  (see BUILD_INFO.md) and skips gracefully if the data is not on disk.
# ──────────────────────────────────────────────────────────────────────

def _maybe_load(name: str) -> dict | None:
    path = DATA / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"warn: failed to parse {path}: {exc}")
        return None


def fig_po_training() -> None:
    """PO/BT training curve on the held-out dev split.

    Reads paper/data/po_agp_training.json. Expected schema:
        {"epochs": [int, ...],
         "coverage_greedy_mean": [float, ...],
         "guard_ratio_greedy_mean": [float, ...],
         "size_over_opt_greedy_mean": [float, ...]   # optional
        }
    """
    d = _maybe_load("po_agp_training.json")
    if d is None:
        print("skip fig_po_training: paper/data/po_agp_training.json not found")
        return

    epochs = d.get("epochs")
    cov_g = d.get("coverage_greedy_mean")
    if not epochs or not cov_g:
        print("skip fig_po_training: required fields (epochs, coverage_greedy_mean) missing")
        return

    fig, ax_cov = plt.subplots(figsize=(5.4, 3.2))
    ax_cov.plot(epochs, cov_g, "-o", color=COLORS["seed"],
                markersize=3, linewidth=1.4, label="coverage (greedy decode)")
    ax_cov.set_xlabel("Training epoch")
    ax_cov.set_ylabel("Mean coverage", color=COLORS["seed"])
    ax_cov.tick_params(axis="y", labelcolor=COLORS["seed"])
    ax_cov.grid(alpha=0.3, linewidth=0.5)
    ax_cov.set_ylim(0.0, 1.02)

    sopt = d.get("size_over_opt_greedy_mean")
    gr_g = d.get("guard_ratio_greedy_mean")
    second = sopt if sopt else gr_g
    second_label = ("$|S|/\\mathrm{OPT}$" if sopt
                    else ("$|S|/n$" if gr_g else None))
    if second is not None:
        ax_sopt = ax_cov.twinx()
        ax_sopt.plot(epochs, second, "-s", color=COLORS["t020"],
                     markersize=3, linewidth=1.4, label=second_label)
        ax_sopt.set_ylabel("Mean " + second_label, color=COLORS["t020"])
        ax_sopt.tick_params(axis="y", labelcolor=COLORS["t020"])
        h1, l1 = ax_cov.get_legend_handles_labels()
        h2, l2 = ax_sopt.get_legend_handles_labels()
        ax_cov.legend(h1 + h2, l1 + l2, loc="lower right", frameon=False)

    ax_cov.set_title("PO/BT training dynamics (held-out dev)")
    save(fig, "fig_po_training.pdf")


def fig_worked_example() -> None:
    """One or two polygons drawn with seed / probe / optimum overlays.

    Reads paper/data/worked_examples.json. Expected schema:
        {"examples": [
            {"name": str,
             "split": str,
             "n": int,
             "points": [[x, y], ...],
             "seed_idxs": [int, ...],
             "probe_idxs": [int, ...],
             "opt_idxs": [int, ...] | null,
             "seed_coverage": float,
             "probe_coverage": float,
             "opt_coverage": float | null},
            ...
        ]}
    """
    d = _maybe_load("worked_examples.json")
    if d is None:
        print("skip fig_worked_example: paper/data/worked_examples.json not found")
        return

    examples = d.get("examples") or []
    if not examples:
        print("skip fig_worked_example: no examples in JSON")
        return

    import numpy as np  # local: only needed when this figure runs

    n_examples = len(examples)
    n_panels_per = 3
    fig, axes = plt.subplots(
        n_examples, n_panels_per,
        figsize=(3.0 * n_panels_per, 3.0 * n_examples),
        squeeze=False,
    )

    panel_specs = [
        (0, "seed_idxs",  "seed_coverage",  "Pretrained policy"),
        (1, "probe_idxs", "probe_coverage", "Policy + probe ($t{=}0.30$)"),
        (2, "opt_idxs",   "opt_coverage",   "Optimum"),
    ]

    for row, ex in enumerate(examples):
        pts = np.array(ex["points"], dtype=float)
        for col, idx_key, cov_key, title_root in panel_specs:
            ax = axes[row, col]
            ax.plot(np.append(pts[:, 0], pts[0, 0]),
                    np.append(pts[:, 1], pts[0, 1]),
                    color="black", linewidth=1.4)
            idxs = ex.get(idx_key)
            if idxs is not None:
                guards = pts[np.asarray(idxs, dtype=int)]
                ax.scatter(guards[:, 0], guards[:, 1],
                           color="red", s=36, zorder=5,
                           edgecolor="black", linewidth=0.4)
                cov = ex.get(cov_key)
                size_str = f"$|S|{{=}}{len(idxs)}$"
                cov_str = (f"$\\mathrm{{Cov}}{{=}}{cov:.3f}$"
                           if cov is not None else "")
                subtitle = ", ".join(s for s in [size_str, cov_str] if s)
                title = (f"{title_root}\n{subtitle}" if row == 0
                         else subtitle)
            else:
                title = (f"{title_root}\n(unavailable)" if row == 0
                         else "(unavailable)")
            ax.set_title(title, fontsize=8)
            ax.set_aspect("equal")
            ax.axis("off")

        axes[row, 0].text(-0.06, 0.5, ex.get("name", ""),
                          transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="right",
                          fontsize=8, family="monospace")

    plt.tight_layout()
    save(fig, "fig_worked_example.pdf")


def fig_encoder_pca() -> None:
    """2D projection of frozen PO/BT encoder embeddings, colored by guard label.

    Reads paper/data/encoder_pca.json. Expected schema:
        {"points_2d": [[x, y], ...],
         "labels": [0/1, ...],
         "method": "pca" | "umap" | "tsne",
         "explained_variance": [float, float] | null
        }
    """
    d = _maybe_load("encoder_pca.json")
    if d is None:
        print("skip fig_encoder_pca: paper/data/encoder_pca.json not found")
        return

    import numpy as np

    xy = np.asarray(d.get("points_2d") or [], dtype=float)
    labels = np.asarray(d.get("labels") or [], dtype=int)
    if xy.size == 0 or labels.size != len(xy):
        print("skip fig_encoder_pca: invalid points_2d / labels shape")
        return

    method = (d.get("method") or "pca").upper()
    ev = d.get("explained_variance")

    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    non_guard = labels == 0
    guard = labels == 1
    ax.scatter(xy[non_guard, 0], xy[non_guard, 1], s=8,
               color=COLORS["seed"], alpha=0.35, label="non-guard")
    ax.scatter(xy[guard, 0], xy[guard, 1], s=10,
               color=COLORS["t020"], alpha=0.75, label="guard (LS target)")

    if method == "PCA" and ev and len(ev) >= 2:
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}\\% var)")
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}\\% var)")
    else:
        ax.set_xlabel(f"{method} dim 1")
        ax.set_ylabel(f"{method} dim 2")
    ax.set_title("Frozen PO/BT encoder embeddings")
    ax.legend(loc="best", frameon=False)
    ax.grid(alpha=0.3, linewidth=0.5)
    save(fig, "fig_encoder_pca.pdf")


def fig_cov_vs_n() -> None:
    """Per-polygon coverage vs polygon size n on the OOD test split.

    Reads paper/data/setpred_test_OOD_per_polygon.json. Expected schema:
        {"polygons": [
            {"name": str,
             "n": int,
             "seed_cov": float,
             "probe_cov_t020": float | null},
            ...
        ]}
    """
    d = _maybe_load("setpred_test_OOD_per_polygon.json")
    if d is None:
        print("skip fig_cov_vs_n: paper/data/setpred_test_OOD_per_polygon.json not found")
        return

    polys = d.get("polygons") or []
    if not polys:
        print("skip fig_cov_vs_n: no polygons in JSON")
        return

    import numpy as np

    ns = np.array([p["n"] for p in polys])
    seed_cov = np.array([p.get("seed_cov", np.nan) for p in polys])
    probe_cov = np.array([p.get("probe_cov_t020", np.nan) for p in polys])

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.scatter(ns, seed_cov, s=6, alpha=0.45,
               color=COLORS["seed"], label="Pretrained pointer")
    ax.scatter(ns, probe_cov, s=6, alpha=0.45,
               color=COLORS["t020"], label="SetPredictor $t{=}0.20$")
    ax.axhline(0.95, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Polygon vertex count $n$")
    ax.set_ylabel("Per-polygon coverage $\\mathrm{Cov}$")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Out-of-distribution coverage vs polygon size (test, 2107 polygons)")
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.legend(loc="lower left", frameon=False)
    save(fig, "fig_cov_vs_n.pdf")


# ──────────────────────────────────────────────────────────────────────
#  Grouped figures (current paper layout): fig_distributions, fig_mechanism
# ──────────────────────────────────────────────────────────────────────

def _ccdf_xy(cov):
    """Exact complementary CDF (fraction with coverage >= x) as a step curve."""
    import numpy as np
    cov_sorted = np.sort(np.asarray(cov, dtype=float))
    n = len(cov_sorted)
    if n == 0:
        return [], []
    ys = (n - np.arange(n)) / n          # fraction with cov >= cov_sorted[i]
    xs_plot = np.concatenate([[cov_sorted[0]], cov_sorted])
    ys_plot = np.concatenate([[1.0], ys])
    return xs_plot, ys_plot


def fig_distributions() -> None:
    """2x3 grid: rows = {dev_test, OOD test}, cols = {coverage CCDF, |S|/n, |S|/OPT}.

    Reads paper/data/dist_dev_test.json and dist_test_OOD.json. Per-polygon schema:
        {"name", "n", "OPT",
         "seed":       {"S_size", "cov"},
         "probe_t020": {"S_size", "cov"},
         "probe_t025": {"S_size", "cov"},
         "probe_t030": {"S_size", "cov"}}
    """
    d_in = _maybe_load("dist_dev_test.json")
    d_ood = _maybe_load("dist_test_OOD.json")
    if d_in is None or d_ood is None:
        print("skip fig_distributions: dist_dev_test.json / dist_test_OOD.json not found")
        return

    import numpy as np

    # methods in display order: policy seed, then conservative -> aggressive
    methods = [("seed", "seed", "Seed"),
               ("probe_t030", "t030", "$t{=}0.30$"),
               ("probe_t025", "t025", "$t{=}0.25$"),
               ("probe_t020", "t020", "$t{=}0.20$")]
    box_colors = [COLORS[c] for _, c, _ in methods]
    box_labels = [lab for _, _, lab in methods]

    def cov_arr(polys, mkey):
        return [p[mkey]["cov"] for p in polys if mkey in p]

    def ratio_arr(polys, mkey, denom):
        out = []
        for p in polys:
            if mkey not in p:
                continue
            d = p["n"] if denom == "n" else p.get("OPT")
            if not d:
                continue
            out.append(p[mkey]["S_size"] / d)
        return out

    def style_box(ax, bp):
        for patch, col in zip(bp["boxes"], box_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.65)
        for med in bp["medians"]:
            med.set_color("black")

    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    rows = [("In-distribution (dev$\\_$test, 367 polygons)", d_in["polygons"], axes[0]),
            ("Out-of-distribution (test, 2107 polygons)", d_ood["polygons"], axes[1])]

    for title, polys, axrow in rows:
        # col 0: coverage complementary CDF
        ax = axrow[0]
        for mkey, ckey, lab in methods:
            xs, ys = _ccdf_xy(cov_arr(polys, mkey))
            ax.step(xs, ys, where="post", color=COLORS[ckey], linewidth=1.5,
                    label=("Seed" if mkey == "seed" else f"SetPredictor {lab}"))
        ax.axvline(0.95, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlim(0.6, 1.005)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Per-polygon coverage $x$")
        ax.set_ylabel("Fraction with $\\mathrm{Cov}\\geq x$")
        ax.grid(alpha=0.3, linewidth=0.5)

        # col 1: |S|/n box plot
        ax = axrow[1]
        data = [ratio_arr(polys, mkey, "n") for mkey, _, _ in methods]
        bp = ax.boxplot(data, patch_artist=True, showmeans=True, showfliers=True,
                        widths=0.6,
                        flierprops=dict(marker=".", markersize=2, alpha=0.3))
        style_box(ax, bp)
        ax.set_xticklabels(box_labels)
        ax.set_ylabel("$|S|/n$")
        ax.set_title(title)
        ax.grid(alpha=0.3, linewidth=0.5, axis="y")

        # col 2: |S|/OPT box plot (robust y-limit so outliers don't dominate)
        ax = axrow[2]
        data = [ratio_arr(polys, mkey, "OPT") for mkey, _, _ in methods]
        bp = ax.boxplot(data, patch_artist=True, showmeans=True, showfliers=True,
                        widths=0.6,
                        flierprops=dict(marker=".", markersize=2, alpha=0.3))
        style_box(ax, bp)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xticklabels(box_labels)
        ax.set_ylabel("$|S|/\\mathrm{OPT}$")
        ax.grid(alpha=0.3, linewidth=0.5, axis="y")
        all_vals = [v for di in data for v in di]
        if all_vals:
            ax.set_ylim(0, float(np.percentile(all_vals, 99)) * 1.1)

    axes[0][0].legend(loc="lower left", frameon=False, fontsize=7)
    plt.tight_layout()
    save(fig, "fig_distributions.pdf")


def fig_mechanism() -> None:
    """1x2: (a) K-invariance fixed point, (b) encoder PCA embedding separation.

    Reuses paper/data/setpred_iter_sweep.json and encoder_pca.json.
    """
    d_iter = _maybe_load("setpred_iter_sweep.json")
    d_pca = _maybe_load("encoder_pca.json")
    if d_iter is None and d_pca is None:
        print("skip fig_mechanism: neither setpred_iter_sweep.json nor encoder_pca.json found")
        return

    import numpy as np
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(9.5, 3.8))

    # panel (a): K-invariance
    if d_iter is not None:
        cells = d_iter["cells"]
        thresholds = ["0.5", "0.6", "0.65", "0.7", "0.75", "0.8"]
        Ks = [1, 2, 3, 5]
        palette = ["#577590", "#43aa8b", "#90be6d", "#f9c74f", "#f8961e", "#f3722c"]
        for t, col in zip(thresholds, palette):
            covs = [cells[f"t={t}|K={K}"]["cov"] if f"t={t}|K={K}" in cells else None
                    for K in Ks]
            axa.plot(Ks, covs, "o-", color=col, label=f"$t={t}$",
                     linewidth=1.5, markersize=5)
        axa.set_xlabel("Number of inference passes $K$")
        axa.set_ylabel("Mean coverage $\\mathrm{Cov}$")
        axa.set_title("(a) Iterative inference is a fixed point")
        axa.set_xticks(Ks)
        axa.grid(alpha=0.3, linewidth=0.5)
        axa.legend(loc="lower right", frameon=False, ncol=2, title="Threshold",
                   fontsize=7)
        all_covs = [cells[f"t={t}|K={K}"]["cov"] for t in thresholds for K in Ks
                    if f"t={t}|K={K}" in cells]
        if all_covs:
            axa.set_ylim(min(all_covs) - 0.003, max(all_covs) + 0.003)
    else:
        axa.axis("off")

    # panel (b): encoder PCA
    xy = np.asarray((d_pca or {}).get("points_2d") or [], dtype=float)
    labels = np.asarray((d_pca or {}).get("labels") or [], dtype=int)
    if xy.size and labels.size == len(xy):
        ev = d_pca.get("explained_variance")
        non_guard = labels == 0
        guard = labels == 1
        axb.scatter(xy[non_guard, 0], xy[non_guard, 1], s=8,
                    color=COLORS["seed"], alpha=0.35, label="non-guard")
        axb.scatter(xy[guard, 0], xy[guard, 1], s=10,
                    color=COLORS["t020"], alpha=0.75, label="guard (LS target)")
        if ev and len(ev) >= 2:
            axb.set_xlabel(f"PC1 ({ev[0]*100:.1f}\\% var)")
            axb.set_ylabel(f"PC2 ({ev[1]*100:.1f}\\% var)")
        axb.set_title("(b) Frozen encoder embeddings")
        axb.legend(loc="best", frameon=False, fontsize=7)
        axb.grid(alpha=0.3, linewidth=0.5)
    else:
        axb.axis("off")

    plt.tight_layout()
    save(fig, "fig_mechanism.pdf")


# ──────────────────────────────────────────────────────────────────────
#  Dispatcher: attempts every figure, reports skips/failures cleanly.
#  fig_coverage_cdf / fig_pareto / fig_kinvariance / fig_encoder_pca are
#  retained above as deprecated helpers (their content now lives in
#  fig_distributions and fig_mechanism); they are not dispatched.
# ──────────────────────────────────────────────────────────────────────

def _run(fn, name: str) -> None:
    try:
        fn()
    except FileNotFoundError as exc:
        print(f"skip {name}: {exc}")
    except Exception as exc:
        print(f"fail {name}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    _run(fig_po_training,    "fig_po_training")
    _run(fig_worked_example, "fig_worked_example")
    _run(fig_distributions,  "fig_distributions")
    _run(fig_cov_vs_n,       "fig_cov_vs_n")
    _run(fig_mechanism,      "fig_mechanism")
