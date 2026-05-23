# Paper Build Information

## Latest Compilation
- **Date**: 2026-05-22
- **Output**: `paper.pdf` (15 pages, 311 KB)
- **Status**: ✓ Clean (no errors, only cosmetic hbox warnings)

## Key Changes from Previous Version
- **Title**: "Neural Combinatorial Optimization for the Vertex-Guard Art Gallery Problem"
- **Structure**: 9 sections (Intro, Problem, Related Work, Method, Experiments, Results, Discussion, Limitations, Conclusion)
- **Methods removed**: Tabular Q-learning and learned reward proxy (value_net) subsections
- **Methods added**: Diagnostic of editor pathologies; SetPredictor architecture with geo-free inference
- **Results**: 6 tables auto-generated from JSON data files

## Data and Table Generation
Tables are **automatically generated** from JSON results files pooled in `paper/data/`:

```bash
python paper/scripts/build_tables.py
```

This script reads:
- `paper/data/setpred_dev_tune.json` → (used for checkpoint selection, not directly tabled)
- `paper/data/setpred_dev_test.json` → headlines, distribution shift, Pareto curve
- `paper/data/setpred_test_OOD.json` → OOD evaluation
- `paper/data/setpred_iter_sweep.json` → fixed-point analysis
- `paper/data/setpred_coverage_dist.json` → (referenced in build script, not directly tabled)
- `paper/data/setpred_large_smoke.json` → (smoke test, not directly tabled)

Output tables:
- `paper/tables/tab_headline.tex` — held-out headline (dev_test)
- `paper/tables/tab_dist_shift.tex` — coverage distribution shift
- `paper/tables/tab_pareto.tex` — threshold sweep (Pareto curve)
- `paper/tables/tab_fixed_point.tex` — fixed-point analysis (K=1,2,3,5)
- `paper/tables/tab_ood_test.tex` — OOD evaluation (test split)
- `paper/tables/tab_dataset_partition.tex` — dataset partition statistics

All tables are included in `paper.tex` via `\input{tables/tab_*.tex}`.

## How to Rebuild

### Quick rebuild (if only content changed, not table data):
```bash
cd paper/
latexmk -pdf paper.tex
```

### Full rebuild (if JSON results or data changed):
```bash
cd ../
source .venv/bin/activate            # matplotlib lives here
python paper/scripts/build_tables.py   # regenerate tables
python paper/scripts/build_figures.py  # regenerate figures
cd paper/
latexmk -pdf paper.tex
```

### Figure generation
`paper/scripts/build_figures.py` produces PDFs into `paper/gfx/setpred/` from JSON files in `paper/data/`. Each figure-builder function is independent and skips gracefully if its JSON input is missing; the script always exits cleanly and prints which figures were produced and which were skipped.

Figures and their data files:

| Figure PDF | JSON input | Section in paper.tex |
|---|---|---|
| `fig_po_training.pdf` | `po_agp_training.json` | §4.1 (RL method) |
| `fig_worked_example.pdf` | `worked_examples.json` | §6.1 (headline) |
| `fig_coverage_cdf.pdf` | `setpred_dev_test.json`, `setpred_test_OOD.json` | §6.2 (distribution) |
| `fig_pareto.pdf` | `setpred_dev_tune.json`, `setpred_dev_test.json`, `setpred_test_OOD.json` | §6.3 (Pareto) |
| `fig_kinvariance.pdf` | `setpred_iter_sweep.json` | §6.4 (fixed point) |
| `fig_cov_vs_n.pdf` | `setpred_test_OOD_per_polygon.json` | §6.5 (OOD) |
| `fig_encoder_pca.pdf` | `encoder_pca.json` | §7 (Discussion) |

The paper compiles either way: `paper.tex` uses a `\figincl{path}{caption}{label}` macro that renders an `\includegraphics` if the file is present and a small "Figure pending" placeholder otherwise, while keeping the `\label` defined so body `\ref{}` statements resolve in both cases. To suppress the placeholder at submission time, edit the macro definition near the top of `paper.tex` (remove the `\fbox{...}` branch).

JSON schemas (see `paper/scripts/build_figures.py` for authoritative docs):

```text
po_agp_training.json
  { "epochs": [int, ...],
    "coverage_greedy_mean": [float, ...],
    "guard_ratio_greedy_mean": [float, ...],       # optional
    "size_over_opt_greedy_mean": [float, ...] }    # optional

worked_examples.json
  { "examples": [
      { "name": str, "split": str, "n": int,
        "points": [[x, y], ...],
        "seed_idxs":  [int, ...],
        "probe_idxs": [int, ...],
        "opt_idxs":   [int, ...] | null,
        "seed_coverage":  float,
        "probe_coverage": float,
        "opt_coverage":   float | null }, ... ] }

encoder_pca.json
  { "points_2d": [[x, y], ...],
    "labels": [0|1, ...],
    "method": "pca" | "umap" | "tsne",
    "explained_variance": [float, float] | null }

setpred_test_OOD_per_polygon.json
  { "polygons": [
      { "name": str, "n": int,
        "seed_cov": float,
        "probe_cov_t020": float | null }, ... ] }
```

The `setpred_*.json` files (for the already-existing CDF / Pareto / K-invariance figures) follow the schema produced by `eval_set_predictor.py`; see that script for the authoritative shape.

Requires `matplotlib` (install via `pip install matplotlib` inside `.venv`).

### Regenerating figure data

The figure data files in `paper/data/` are not committed to the repo (they exceed git-friendly sizes for the per-polygon variant). Regenerate as follows from the AGNet repo root:

```bash
# Already-coded figures (1-3): threshold sweep on each split.
python eval_set_predictor.py \
  --checkpoint checkpoints/set_predictor/standard/set_predictor_best.pt \
  --val-traj   data/ls_trajectories_dev_tune.pkl \
  --thresholds 0.20 0.25 0.30 0.35 0.40 0.50 0.65 \
  --iter-passes 1 \
  --out paper/data/setpred_dev_tune.json

python eval_set_predictor.py \
  --checkpoint checkpoints/set_predictor/standard/set_predictor_best.pt \
  --val-traj   data/ls_trajectories_dev_test.pkl \
  --thresholds 0.20 0.25 0.30 \
  --iter-passes 1 \
  --out paper/data/setpred_dev_test.json

python eval_set_predictor.py \
  --checkpoint checkpoints/set_predictor/standard/set_predictor_best.pt \
  --val-traj   data/ls_trajectories_test.pkl \
  --thresholds 0.20 0.25 0.30 \
  --iter-passes 1 \
  --out paper/data/setpred_test_OOD.json

python eval_set_predictor.py \
  --checkpoint checkpoints/set_predictor/standard/set_predictor_best.pt \
  --val-traj   data/ls_trajectories_dev.pkl \
  --thresholds 0.5 0.6 0.65 0.7 0.75 0.8 \
  --iter-passes-sweep 1 2 3 5 \
  --out paper/data/setpred_iter_sweep.json
```

For the new figures (4-7), the regeneration paths are:

- **`po_agp_training.json`** (training curve): `po_agp.py` prints per-epoch metrics to stdout but does not write JSON by default. Either capture stdout from a training run and parse the per-epoch lines into the schema above, or add a minimal `--metrics-log` JSON sink to `po_agp.py` and re-run training for ~30 epochs on the same config that produced `lstm_bt`.
- **`worked_examples.json`** (worked example): pick polygon names from `dev_test` and `test`, run the pretrained policy to get `seed_idxs`, run the SetPredictor at `t=0.30` to get `probe_idxs`, read `opt_idxs` from the `.opt` files in the AGPVG library, and compute CGAL coverage via `evaluate_polygon_visibility_numpy_wo_gt` in `utils.py`. A small helper script `tools/build_worked_examples.py` (not yet written) is the natural home for this.
- **`encoder_pca.json`** (encoder PCA): run `extract_pointer_embeddings` from `set_predictor.py` on a sample of held-out polygons, stack per-vertex embeddings, run `sklearn.decomposition.PCA(n_components=2)` (or UMAP), serialize `points_2d` + `labels` (`y_v` from LS-derived targets in the trajectory pickles).
- **`setpred_test_OOD_per_polygon.json`** (cov-vs-n): the `eval_set_predictor.py` machinery already evaluates per polygon; if its current JSON output collapses to aggregate stats only, add a per-polygon dump option, or post-process the eval loop to emit one row per polygon with name, n, seed_cov, probe_cov at each threshold.

Then build figures:

```bash
python paper/scripts/build_figures.py
```

The script prints `wrote paper/gfx/setpred/fig_*.pdf` for each figure produced and `skip <name>: <reason>` for each one whose data is missing.

### Clean and rebuild:
```bash
cd paper/
latexmk -C  # removes all generated files
latexmk -pdf paper.tex
```

## Bibliography
- **File**: `paper/paper.bib` (updated with new entries: Pan et al. 2025 PO/NCO, POMO, Kool 2019, Lee–Lin 1986, Bradley–Terry, AdamW, CGAL, DAgger)
- **Style**: `plain`
- **References**: 17 distinct bibkeys, all resolved

## File Structure

```
paper/
├── paper.tex                          # Main LaTeX source (fully rewritten)
├── paper.bib                          # Bibliography (13 new entries added)
├── paper.pdf                          # Compiled output (15 pages)
├── data/                              # Pooled result JSONs
│   ├── setpred_dev_tune.json
│   ├── setpred_dev_test.json
│   ├── setpred_test_OOD.json
│   ├── setpred_iter_sweep.json
│   ├── setpred_coverage_dist.json
│   └── setpred_large_smoke.json
├── scripts/                           # Build scripts
│   └── build_tables.py                # Generates .tex tables from JSON
├── tables/                            # Auto-generated LaTeX tables
│   ├── tab_headline.tex
│   ├── tab_dist_shift.tex
│   ├── tab_pareto.tex
│   ├── tab_fixed_point.tex
│   ├── tab_ood_test.tex
│   └── tab_dataset_partition.tex
├── gfx/                               # Figures (unchanged)
│   └── (historical figures, not currently referenced)
└── BUILD_INFO.md                      # This file
```

## Validation Checklist ✓

- [x] All 17 `\cite{}` keys resolve in `paper.bib`
- [x] All 6 `\input{tables/...}` files exist
- [x] All `\ref{}` have corresponding `\label{}`
- [x] Numerical claims in Abstract/Intro/Results cross-checked against tables
- [x] Citations sorted alphabetically in body (checked against plain style)
- [x] No undefined control sequences
- [x] No missing `$...$` math delimiters
- [x] Compiles to valid PDF with no errors

## Notes

- **Unused historical files** in `paper/tables/`: `summary_table.tex`, `performance_comparison.tex` (old structure predates SetPredictor work)
- **Unused preamble commands**: `\sinst` (harmless, legacy)
- **Cosmetic warnings**: Overfull hbox on a few lines (minor typography, does not affect readability)
- **Orphaned figures** in `paper/gfx/`: unused after removal of Q-learning / value-net sections (safe to ignore or delete)

## Contact

For questions about table regeneration or compilation, see `paper/scripts/build_tables.py` documentation.
