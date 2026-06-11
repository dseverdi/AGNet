#!/usr/bin/env bash
# Runbook for the reviewer-driven revision pass (M1/M2/M4/m2/m3 + cleanups).
#
# Run from the repo root:  bash paper/scripts/run_revisions.sh [step]
# With no argument, every step runs in order (~14 hours GPU sequentially).
# With a step name, only that step runs. Step names:
#   ablation_train       — Step 1: coordinate-only ablation training (M2)
#   multiseed_train      — Step 2: three multi-seed probe trainings (M1)
#   ablation_eval        — Step 3: per-polygon eval of the ablated probe
#   multiseed_eval       — Step 4: per-polygon eval of each multi-seed probe
#   classical_baselines  — Step 5: aggregate greedy + LS baselines (M4)
#   linear_probe         — Step 6: encoder linear-probe AUC (m2)
#   multiseed_summary    — Step 7: aggregate multi-seed JSONs
#   worked_examples      — Step 8: regenerate worked-example figure at t=0.20
#
# Steps 1–4 require GPU. Steps 5–8 are CPU-light (5 needs CGAL).
# Steps 5–8 do not depend on Steps 1–4 and can be run in parallel to them.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

PY=/home/dseverdi/.conda/envs/MLAG/bin/python   # PyTorch + CGAL env
SP_DIR=checkpoints/set_predictor
CFG=configs/set_predictor_train_standard.json   # supplies train/val traj + pointer ckpt
LOG_DIR="paper/logs"
mkdir -p "$LOG_DIR"

banner() { echo; echo "============================================================"; echo "  $*"; echo "============================================================"; }

# Run a step, tee'ing all output to paper/logs/<step>_<timestamp>.log AND the
# console. Exit status from the step is preserved (set -o pipefail).
run_step() {
  local step="$1"; shift
  local ts; ts="$(date +%Y%m%d_%H%M%S)"
  local log="$LOG_DIR/${step}_${ts}.log"
  echo "[run_step] $step  ->  $log"
  ( "$step" ) 2>&1 | tee "$log"
  echo "[run_step] $step done  (log: $log)"
}

ablation_train() {
  banner "Step 1: coordinate-only ablation training (~3 hours GPU)"
  $PY train_set_predictor.py --config "$CFG" --disable-ptr-emb --seed 1234 \
      --out-dir "$SP_DIR/no_encoder"
}

multiseed_train() {
  banner "Step 2: multi-seed probe training (~3 hours each, 3 seeds)"
  for s in 11 22 33; do
    echo "[seed=$s]"
    $PY train_set_predictor.py --config "$CFG" --seed "$s" \
        --out-dir "$SP_DIR/seed$s"
  done
}

ablation_eval() {
  banner "Step 3: per-polygon eval of ablated probe (~50 min GPU)"
  SP_CKPT="$SP_DIR/no_encoder/set_predictor_best.pt" \
  SP_SUFFIX="_noenc" \
    $PY paper/scripts/build_paper_data.py per_polygon_all
}

multiseed_eval() {
  banner "Step 4: per-polygon eval of each multi-seed probe (~50 min × 3)"
  for s in 11 22 33; do
    echo "[seed=$s]"
    SP_CKPT="$SP_DIR/seed$s/set_predictor_best.pt" \
    SP_SUFFIX="_seed$s" \
      $PY paper/scripts/build_paper_data.py per_polygon_all
  done
}

classical_baselines() {
  banner "Step 5: aggregate greedy + LS baselines (~10 min CPU if cached, ~3h if recomputing CGAL)"
  $PY paper/scripts/build_paper_data.py classical_baselines
}

linear_probe() {
  banner "Step 6: encoder linear-probe AUC (~5-10 min GPU + CPU)"
  $PY paper/scripts/build_paper_data.py encoder_linear_probe
}

multiseed_summary() {
  banner "Step 7: aggregate multi-seed JSONs (instant; needs Step 4 outputs)"
  $PY paper/scripts/build_paper_data.py multi_seed_summary
}

worked_examples() {
  banner "Step 8: regenerate worked-example figure at t=0.20 (~30 sec GPU)"
  $PY paper/scripts/build_paper_data.py worked_examples
}

if [ $# -eq 0 ]; then
  for s in ablation_train multiseed_train ablation_eval multiseed_eval \
           classical_baselines linear_probe multiseed_summary worked_examples; do
    run_step "$s"
  done
  banner "All steps done. Hand off to Claude for Phase C integration."
else
  run_step "$1"
fi
