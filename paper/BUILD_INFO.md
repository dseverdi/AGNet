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
python paper/scripts/build_tables.py  # regenerate tables
cd paper/
latexmk -pdf paper.tex
```

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
