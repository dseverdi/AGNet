#!/usr/bin/env bash
# run_probe_ladder.sh — train the intermediate-capacity probes for the
# capacity ladder requested by Reviewer 3 (probe-expressiveness concern).
#
# The released full probe is a 3-layer, 8-head, ~464K-param Transformer. R3
# asks whether an intermediate-capacity probe already recovers the signal, or
# whether the full Transformer's nonlinear reasoning is doing the work. We add
# two rungs below "full":
#
#   mlp    --predictor-attn-layers 0  -> DeepSets-style MLP, NO self-attention
#          (per-vertex Linear + masked mean-pool context + MLP head)
#   attn1  --predictor-attn-layers 1  -> a single self-attention block
#
# 4 seeds each {1234,11,22,33}, same train/val/pointer/epochs as the released
# full probe (exact saved args from checkpoints/set_predictor/standard; the
# released config file set_predictor_train_standard.json is gone).
#
# The full-probe rung reuses the EXISTING 4 seeds (standard, seed11/22/33).
# The linear rung is computed held-out inside eval_probe_ladder.py.
#
# IDEMPOTENT: a step is skipped if its output already exists.
#
# Usage:
#   bash run_probe_ladder.sh              # train 8 probes + dev_test eval (~1 h GPU)
#   bash run_probe_ladder.sh --dry-run    # print commands only
#   SMOKE=1 bash run_probe_ladder.sh      # 2-epoch wiring check, seed 1234 only
#
# When it prints LADDER_TRAIN_DONE, the AUC + table steps are wiring-only (no
# GPU): eval_probe_ladder.py then build_ladder_table.py.

set -euo pipefail

PYTHON=/home/dseverdi/.conda/envs/MLAG/bin/python
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

POINTER=checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt
TRAIN_TRAJ=data/ls_trajectories_train.pkl
VAL_TRAJ=data/ls_trajectories_dev.pkl
DATASET_ROOT=/home/dseverdi/Radno/MLAG/dataset/AGPIL
EPOCHS=60
SEEDS=(1234 11 22 33)

# condition name -> attn layers
declare -A COND_LAYERS=( [ladder_mlp]=0 [ladder_attn1]=1 )
COND_NAMES=(ladder_mlp ladder_attn1)

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && { DRY_RUN=1; echo "[dry-run] no commands will execute"; }

if [[ "${SMOKE:-0}" == "1" ]]; then
    echo "[smoke] 2-epoch wiring check, seed 1234 only"
    EPOCHS=2
    SEEDS=(1234)
fi

run() { echo "+ $*"; [[ $DRY_RUN -eq 0 ]] && "$@"; }
export DATASET_PATH="$DATASET_ROOT"

for COND in "${COND_NAMES[@]}"; do
    LAYERS=${COND_LAYERS[$COND]}
    for S in "${SEEDS[@]}"; do
        CKPT_DIR="checkpoints/set_predictor/${COND}_seed${S}"
        BEST="${CKPT_DIR}/set_predictor_best.pt"
        FINAL="${CKPT_DIR}/set_predictor_final.pt"
        SUFFIX="_${COND}_seed${S}"

        echo ""
        echo "############################################################"
        echo "#  ${COND}  seed=${S}   attn-layers=${LAYERS}"
        echo "############################################################"

        if [[ -e "$BEST" || -e "$FINAL" ]]; then
            echo "[skip] checkpoint exists"
        else
            run "$PYTHON" train_set_predictor.py \
                --train-traj "$TRAIN_TRAJ" \
                --val-traj "$VAL_TRAJ" \
                --pointer-checkpoint "$POINTER" \
                --epochs "$EPOCHS" \
                --batch-size 32 \
                --lr 3e-4 \
                --predictor-hidden 128 \
                --predictor-attn-layers "$LAYERS" \
                --predictor-heads 8 \
                --rollout-eval-k 0 \
                --seed "$S" \
                --out-dir "$CKPT_DIR"
        fi

        # Use the final-epoch checkpoint for every ladder rung (per-epoch dev
        # eval is disabled above: it only selects a checkpoint, not the weights,
        # and its CGAL cost dominated the run). final.pt = 60-epoch weights.
        CKPT="$FINAL"; [[ -e "$FINAL" ]] || CKPT="$BEST"

        # Downstream metrics on dev_test only (fast, 362 polys). OOD not needed
        # for the ladder — the AUC is the headline; guard-count columns are
        # in-distribution to compare against tab_ablation.
        DEV_OUT="paper/data/dist_dev_test${SUFFIX}.json"
        if [[ -e "$DEV_OUT" ]]; then
            echo "[skip] $DEV_OUT exists"
        else
            echo "+ SP_CKPT=$CKPT SP_SUFFIX=$SUFFIX SP_SPLITS=dev_test build_paper_data.py per_polygon_all"
            if [[ $DRY_RUN -eq 0 ]]; then
                SP_CKPT="$CKPT" SP_SUFFIX="$SUFFIX" SP_SPLITS=dev_test \
                    "$PYTHON" paper/scripts/build_paper_data.py per_polygon_all
            fi
        fi
    done
done

echo ""
echo "============================================================"
echo "LADDER_TRAIN_DONE"
echo "Next (no GPU):"
echo "  $PYTHON paper/scripts/eval_probe_ladder.py     # AUC per capacity"
echo "  $PYTHON paper/scripts/build_ladder_table.py    # tab_probe_ladder.tex"
echo "============================================================"
