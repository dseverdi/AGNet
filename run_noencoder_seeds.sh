#!/usr/bin/env bash
# run_noencoder_seeds.sh — train + evaluate the no-encoder SetPredictor ablation
# for seeds {11, 22, 33}, so the no-encoder row can be reported as a 4-seed
# mean ± std (symmetric with the full probe) instead of a single run.
#
# Seed 1234 already exists (checkpoints/set_predictor/no_encoder + the
# dist_*_noenc.json / setpred_extreme_ood_no_encoder.json results); this script
# adds the other three seeds. It is IDEMPOTENT: a step is skipped if its output
# already exists, so it is safe to re-run after an interruption.
#
# Produces:
#   checkpoints/set_predictor/no_encoder_seed{11,22,33}/set_predictor_best.pt
#   paper/data/dist_dev_test_noenc_seed{11,22,33}.json
#   paper/data/dist_test_OOD_noenc_seed{11,22,33}.json
#   results/v3/setpred_extreme_ood_no_encoder_seed{11,22,33}.json
#
# After it finishes, the aggregation + table + prose updates are wiring-only
# (no GPU); tell Claude "no-encoder seeds done" and it will flip the
# tab_headline / tab_ood / tab_large rows to mean ± std and recompile.
#
# Usage:
#   bash run_noencoder_seeds.sh            # train + eval all three splits
#   bash run_noencoder_seeds.sh --dry-run  # print the commands only
#
# Wall-clock estimate: ~3 min train + ~2-4 min dev/OOD eval + a few min large
# eval, per seed (GPU). The `large` evals (exact CGAL on n up to 2250) are the
# slow part.

set -euo pipefail

PYTHON=/home/dseverdi/.conda/envs/MLAG/bin/python
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CONFIG=configs/set_predictor_train_standard.json
POINTER=checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt
DATASET_ROOT=/home/dseverdi/Radno/MLAG/dataset/AGPIL   # has dev/ test/ large/
LARGE_TRAJ=data/ls_trajectories_large.pkl
SEEDS=(11 22 33)

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "[dry-run] no commands will execute"
fi

run() {  # echo + (optionally) execute a command
    echo "+ $*"
    if [[ $DRY_RUN -eq 0 ]]; then
        "$@"
    fi
}

export DATASET_PATH="$DATASET_ROOT"   # build_paper_data.py reads this for OPT

for S in "${SEEDS[@]}"; do
    CKPT_DIR="checkpoints/set_predictor/no_encoder_seed${S}"
    CKPT="${CKPT_DIR}/set_predictor_best.pt"

    echo ""
    echo "############################################################"
    echo "#  no-encoder ablation, seed ${S}"
    echo "############################################################"

    # ---- 1. train (skip if a complete checkpoint already exists) ----
    if [[ -e "$CKPT" ]]; then
        echo "[skip] $CKPT exists"
    else
        run "$PYTHON" train_set_predictor.py \
            --config "$CONFIG" \
            --seed "$S" \
            --disable-ptr-emb \
            --out-dir "$CKPT_DIR"
    fi

    # ---- 2. dev_test + OOD test eval (one pass writes both files) ----
    DEV_OUT="paper/data/dist_dev_test_noenc_seed${S}.json"
    OOD_OUT="paper/data/dist_test_OOD_noenc_seed${S}.json"
    if [[ -e "$DEV_OUT" && -e "$OOD_OUT" ]]; then
        echo "[skip] $DEV_OUT and $OOD_OUT exist"
    else
        echo "+ SP_CKPT=$CKPT SP_SUFFIX=_noenc_seed${S} $PYTHON paper/scripts/build_paper_data.py per_polygon_all"
        if [[ $DRY_RUN -eq 0 ]]; then
            SP_CKPT="$CKPT" SP_SUFFIX="_noenc_seed${S}" \
                "$PYTHON" paper/scripts/build_paper_data.py per_polygon_all
        fi
    fi

    # ---- 3. extreme-OOD (large split) eval ----
    LARGE_OUT="results/v3/setpred_extreme_ood_no_encoder_seed${S}.json"
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
done

echo ""
echo "============================================================"
echo "ALL_NOENCODER_SEEDS_DONE"
echo "Next (no GPU): Claude wires the aggregation + tables to mean ± std:"
echo "  $PYTHON paper/scripts/build_paper_data.py multi_seed_summary"
echo "  $PYTHON paper/scripts/make_extreme_ood_data.py"
echo "  $PYTHON paper/scripts/build_tables.py"
echo "============================================================"
