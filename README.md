# AGNet — Learning to Place Guards by Reinforcement

> **Paper:** *Learning to Place Guards by Reinforcement: A Geo-Free Neural Policy for the Vertex-Guard Art Gallery Problem*
> Domagoj Ševerdija, Jurica Maltar, Nathan Chappel, Domagoj Matijević

This repository contains the code and data for the paper. We study whether an LSTM
pointer-network policy trained by reinforcement learning on the
**vertex-guard Art Gallery Problem (AGPVG)** learns a representation that encodes the
underlying geometry — independent of what its decoder expresses. The key findings:

- A pointer-network policy trained **geo-free** (only vertex coordinates at inference,
  no visibility oracle) places guards at near-greedy cardinality but leaves a tail of
  under-covered polygons.
- Freezing the encoder and probing it with a small single-shot **SetPredictor** classifier
  closes most of the feasibility gap, in and out of distribution (up to 5× the training range),
  cutting infeasible polygons by roughly an order of magnitude.
- A no-encoder ablation and a linear probe (ROC-AUC = 0.842) confirm that the representation —
  not the probe's own capacity — carries the guard-relevant geometry.

---

## Repository structure

```
AGNet/
├── po_agp.py                   # PO/BT policy training (main entry point)
├── train_set_predictor.py      # SetPredictor probe training
├── eval_set_predictor.py       # Evaluation: threshold sweep + iterative refinement
├── eval_checkpoint.py          # Evaluate policy checkpoint (CGAL visibility)
├── models.py                   # PointerNet encoder/decoder architecture
├── set_predictor.py            # SetPredictor architecture
├── rewards.py                  # Coverage + guard-count reward definitions
├── dataset.py                  # AGPVG dataset loader
├── greedy_agp.py               # Classical greedy baseline
├── tools/
│   └── build_ls_trajectories.py  # Generate LS-editor training targets
├── configs/                    # JSON configs for all experiments
│   ├── po_agp_lstm.json
│   └── set_predictor_train_standard.json
└── data/                       # Precomputed LS trajectories and greedy baselines
    ├── ls_trajectories_train.pkl
    ├── ls_trajectories_dev.pkl
    ├── ls_trajectories_test.pkl
    └── ...
```

---

## Installation

**Requirements:** Python 3.10+, PyTorch 2.0+, CGAL (for exact visibility at evaluation).

```bash
git clone git@github.com:dseverdi/AGNet.git
cd AGNet
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### CGAL / scikit-geometry

Exact visibility (used only at evaluation, not training) requires
[scikit-geometry](https://github.com/scikit-geometry/scikit-geometry):

```bash
# Recommended: via conda
conda install -c conda-forge scikit-geometry

# Alternative: build from source (see scikit-geometry docs)
```

---

## Dataset (AGPVG)

The polygon instances come from the **Art Gallery Problem Vertex-Guard (AGPVG)** benchmark:

> Couto et al., *Art Gallery Problem Vertex Guard instances*
> https://www.ic.unicamp.br/~cid/Problem-instances/Art-Gallery/AGPVG/

Download and extract the dataset, then configure the path:

```bash
cp .env.example .env
# Edit .env and set DATASET_PATH to the extracted AGPVG directory
```

The `data/` directory in this repo already contains **precomputed LS-editor trajectories**
(training targets for the SetPredictor) and greedy baseline results — you do not need to
regenerate these to run evaluation.

---

## Quick start: run the trained probe

Download the pretrained checkpoints from the
[GitHub Releases page](https://github.com/dseverdi/AGNet/releases):

| File | Description |
|---|---|
| `po_agp_best_greedy.pt` | PO/BT LSTM pointer-network policy |
| `set_predictor_best.pt` | SetPredictor probe (standard τ=0.99 targets) |

Place them at `checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt` and
`checkpoints/set_predictor/standard/set_predictor_best.pt`, then:

```bash
# Threshold sweep on the dev_test split (reproduces Table 3 in the paper)
python eval_set_predictor.py \
    --checkpoint checkpoints/set_predictor/standard/set_predictor_best.pt \
    --split dev_test

# Full Pareto sweep across thresholds and inference passes (Tables 4–5)
python eval_set_predictor.py \
    --checkpoint checkpoints/set_predictor/standard/set_predictor_best.pt \
    --split test \
    --iter-passes-sweep 1 2 3 5
```

---

## Paper source and claim verification

`paper/` holds the LaTeX source, the generated tables and figures, and the scripts
that produce them. The manuscript rebuilds as-is:

```bash
latexmk -pdf -cd paper/paper.tex        # 39 pages, no external data needed
```

Every numeric claim in the paper is registered in `tools/claims.py` and checked
against the files under `results/` by a consistency gate:

```bash
python tools/verify_paper_consistency.py   # numerals, captions, populations, PDF text
python tools/mutation_test_gate.py         # verifies the gate itself still catches errors
```

### What is and is not shipped here

| shipped | not shipped |
|---|---|
| paper source, tables, the four figures it inputs | `data/ls_trajectories_*.pkl` (163 MB) |
| `results/*.json` — the measurements behind every claim | `paper/data/*.json` (17 MB per-polygon dumps) |
| all generators under `paper/scripts/` and `tools/` | model checkpoints |

The large files are regenerable: trajectories via step 2 above, and
`paper/data/*.json` from `results/` via `paper/scripts/build_paper_data.py`. Without
them `verify_paper_consistency.py` stops at its source-readability check, since the
registry cites the trajectory pickles; the paper build itself is unaffected.

## Reproducing paper results from scratch

### 1. Train the PO/BT policy

```bash
python po_agp.py configs/po_agp_lstm.json
# Checkpoint saved to checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt
```

### 2. Build LS-editor training targets

The `data/ls_trajectories_*.pkl` targets (163 MB) are **not** tracked on this branch.
Regenerate them from the trained policy:

```bash
python tools/build_ls_trajectories.py \
    --split train \
    --checkpoint checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt \
    --out data/ls_trajectories_train.pkl
```

### 3. Train the SetPredictor probe

```bash
# Four seeds as used in the paper: 1234, 11, 22, 33
for SEED in 1234 11 22 33; do
    python train_set_predictor.py configs/set_predictor_train_standard.json \
        --seed $SEED \
        --out-dir checkpoints/set_predictor/standard_seed${SEED}
done
```

### 4. Evaluate

```bash
python eval_set_predictor.py \
    --checkpoint checkpoints/set_predictor/standard_seed1234/set_predictor_best.pt \
    --split dev_test
```

---

## Configuration

All experiments are driven by JSON configs in `configs/`. Key files:

| Config | Purpose |
|---|---|
| `configs/po_agp_lstm.json` | PO/BT LSTM policy training |
| `configs/set_predictor_train_standard.json` | SetPredictor probe (τ=0.99 targets) |

---

## Citation

```bibtex
@article{severdija2026agnet,
  title   = {Learning to Place Guards by Reinforcement: A Geo-Free Neural Policy
             for the Vertex-Guard Art Gallery Problem},
  author  = {Ševerdija, Domagoj and Maltar, Jurica and Chappel, Nathan
             and Matijević, Domagoj},
  journal = {Preprint},
  year    = {2026}
}
```

---

## License

Code: MIT. Dataset: see [AGPVG page](https://www.ic.unicamp.br/~cid/Problem-instances/Art-Gallery/AGPVG/) for terms.
