#!/usr/bin/env bash
# run_noseed_ablation.sh — train + evaluate the NO-SEED SetPredictor ablations
# requested by the reviewer (Q2): disentangle the frozen encoder's contribution
# from the policy's seed indicator. Completes the 2x2 design
#
#                     seed ON              seed OFF (this script)
#   encoder ON    full probe (have)     no-seed              -> condition "noseed"
#   encoder OFF   no-encoder (have)     coords-only          -> condition "noseednoenc"
#
# For each condition we train 4 seeds {1234,11,22,33} so the rows can be reported
# as a 4-seed mean +/- std, symmetric with the existing full / no-encoder rows.
#
# The no-seed mask is applied INSIDE SetPredictor (set_predictor.py: --disable-seed),
# zeroing the seed indicator BOTH as an input feature and in the seed-context pool.
# eval_set_predictor.py and paper/scripts/build_paper_data.py both read
# disable_seed back from the checkpoint args, so eval reconstructs the mask.
#
# IDEMPOTENT: a step is skipped if its output already exists; safe to re-run.
#
# Produces, for COND in {noseed, noseednoenc} and S in {1234,11,22,33}:
#   checkpoints/set_predictor/${COND}_seed${S}/set_predictor_best.pt
#   paper/data/dist_dev_test_${COND}_seed${S}.json      (362 in-dist "test")
#   paper/data/dist_test_OOD_${COND}_seed${S}.json      (2081 OOD)
#   results/v3/setpred_extreme_ood_${COND}_seed${S}.json (285 ood-large; ONLY if RUN_LARGE=1)
#
# After it finishes the aggregation + tables + prose are wiring-only (no GPU);
# tell Claude "no-seed ablation done" and it will add the 2x2 rows and recompile.
#
# Usage:
#   bash run_noseed_ablation.sh                # train + dev/OOD eval, 8 runs (~1 h GPU)
#   bash run_noseed_ablation.sh --dry-run      # print commands only
#   RUN_LARGE=1 bash run_noseed_ablation.sh    # also run the slow ood-large eval
#   SMOKE=1 bash run_noseed_ablation.sh        # 2-epoch wiring check, seed 1234 only
#
# Wall-clock: ~5 min train + ~2-4 min dev/OOD eval per run; 8 runs ~= 1 h.
# The optional `large` evals (exact CGAL up to n=2250) are the slow part.

set -euo pipefail

PYTHON=/home/dseverdi/.conda/envs/MLAG/bin/python
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ---- training config (the released full-probe config file is gone; these are
#      the exact saved args from checkpoints/set_predictor/standard). ----
POINTER=checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt
TRAIN_TRAJ=data/ls_trajectories_train.pkl
VAL_TRAJ=data/ls_trajectories_dev.pkl
LARGE_TRAJ=data/ls_trajectories_large.pkl
DATASET_ROOT=/home/dseverdi/Radno/MLAG/dataset/AGPIL   # has dev/ test/ large/
EPOCHS=60
SEEDS=(1234 11 22 33)

# Conditions: name -> extra train flags
COND_NAMES=(noseed noseednoenc)
declare -A COND_FLAGS=(
    [noseed]="--disable-seed"
    [noseednoenc]="--disable-seed --disable-ptr-emb"
)

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && { DRY_RUN=1; echo "[dry-run] no commands will execute"; }

if [[ "${SMOKE:-0}" == "1" ]]; then
    echo "[smoke] 2-epoch wiring check, seed 1234 only"
    EPOCHS=2
    SEEDS=(1234)
fi

run() {  # echo + (optionally) execute
    echo "+ $*"
    [[ $DRY_RUN -eq 0 ]] && "$@"
}

export DATASET_PATH="$DATASET_ROOT"   # build_paper_data.py reads this for OPT

for COND in "${COND_NAMES[@]}"; do
    FLAGS=${COND_FLAGS[$COND]}
    for S in "${SEEDS[@]}"; do
        CKPT_DIR="checkpoints/set_predictor/${COND}_seed${S}"
        BEST="${CKPT_DIR}/set_predictor_best.pt"
        FINAL="${CKPT_DIR}/set_predictor_final.pt"
        SUFFIX="_${COND}_seed${S}"

        echo ""
        echo "############################################################"
        echo "#  condition=${COND}  seed=${S}   flags: ${FLAGS}"
        echo "############################################################"

        # ---- 1. train (skip if a checkpoint already exists). The coords-only
        #         (no-seed + no-encoder) probe degenerates and never beats the
        #         seed on the cov/|S| trade, so train_set_predictor.py writes
        #         NO set_predictor_best.pt for it — only _final.pt. Treat either
        #         as "trained" so we don't needlessly retrain. ----
        if [[ -e "$BEST" || -e "$FINAL" ]]; then
            echo "[skip] checkpoint exists ($([ -e "$BEST" ] && echo best || echo final))"
        else
            run "$PYTHON" train_set_predictor.py \
                --train-traj "$TRAIN_TRAJ" \
                --val-traj "$VAL_TRAJ" \
                --pointer-checkpoint "$POINTER" \
                --epochs "$EPOCHS" \
                --batch-size 32 \
                --lr 3e-4 \
                --predictor-hidden 128 \
                --predictor-attn-layers 3 \
                --predictor-heads 8 \
                --eval-thresholds 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
                --rollout-eval-k 200 \
                --seed "$S" \
                $FLAGS \
                --out-dir "$CKPT_DIR"
        fi

        # Evaluate the dev-selected best checkpoint; fall back to final-epoch for
        # the degenerate coords-only cell (no best was ever promoted). Note which.
        if [[ -e "$BEST" ]]; then
            CKPT="$BEST"
        else
            CKPT="$FINAL"
            echo "[note] no best checkpoint for ${COND}_seed${S}; evaluating final-epoch (expected for coords-only)"
        fi

        # ---- 2. dev_test (362) + OOD (2081) per-polygon eval (one pass) ----
        DEV_OUT="paper/data/dist_dev_test${SUFFIX}.json"
        OOD_OUT="paper/data/dist_test_OOD${SUFFIX}.json"
        if [[ -e "$DEV_OUT" && -e "$OOD_OUT" ]]; then
            echo "[skip] $DEV_OUT and $OOD_OUT exist"
        else
            echo "+ SP_CKPT=$CKPT SP_SUFFIX=$SUFFIX $PYTHON paper/scripts/build_paper_data.py per_polygon_all"
            if [[ $DRY_RUN -eq 0 ]]; then
                SP_CKPT="$CKPT" SP_SUFFIX="$SUFFIX" \
                    "$PYTHON" paper/scripts/build_paper_data.py per_polygon_all
            fi
        fi

        # ---- 3. extreme-OOD (ood-large) eval — OPTIONAL (slow) ----
        if [[ "${RUN_LARGE:-0}" == "1" ]]; then
            LARGE_OUT="results/v3/setpred_extreme_ood_${COND}_seed${S}.json"
            if [[ -e "$LARGE_OUT" ]]; then
                echo "[skip] $LARGE_OUT exists"
            else
                run "$PYTHON" eval_set_predictor.py \
                    --checkpoint "$CKPT" \
                    --val-traj "$LARGE_TRAJ" \
                    --pointer-checkpoint "$POINTER" \
                    --sol-dir "${DATASET_ROOT}/large" \
                    --thresholds 0.20 0.25 0.30 \
                    --batch-size 4 \
                    --out "$LARGE_OUT"
            fi
        fi
    done
done

echo ""
echo "============================================================"
echo "ALL_NOSEED_RUNS_DONE"
echo "Per-polygon JSONs are in paper/data/dist_{dev_test,test_OOD}_{noseed,noseednoenc}_seed*.json"
echo "Next (no GPU): tell Claude 'no-seed ablation done' to aggregate + build the 2x2 table + prose."
echo "============================================================"
