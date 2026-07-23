#!/usr/bin/env bash
# run_policy_seeds.sh — policy-seed replication for the encoder-probing result.
#
# WHY THIS EXISTS
#   Every conclusion in the paper currently rests on ONE PO/BT policy run
#   (seed 1234). The four seeds reported throughout are SetPredictor *probe*
#   seeds, not policy seeds. This script trains additional PO/BT policies from
#   scratch and re-runs the encoder-vs-no-encoder comparison on each, so the
#   central claim can be stated as replicating across policies rather than as a
#   single-run observation.
#
#   The claim under test is a WITHIN-policy contrast:
#       does the full probe beat the no-encoder ablation on the coverage tail,
#       for every policy, regardless of that policy's absolute quality?
#   Absolute coverage varying across seeds does NOT threaten the claim. Only the
#   gap disappearing would, and that is exactly what we want to know before a
#   reviewer asks.
#
# WHAT IT PRODUCES (per policy seed S)
#   checkpoints/v3/po_agp/lstm_bt_seed<S>/po_agp_best_greedy.pt   trained policy
#   data/ls_trajectories_{train,dev}_pseed<S>.pkl                 probe targets
#   checkpoints/set_predictor/pseed<S>_{full,noenc}/…             two probes
#   results/policy_seeds/pseed<S>_{full,noenc}.json               eval output
#   results/policy_seeds/manifest_pseed<S>.json                   timings + paths
#
# USAGE
#   bash run_policy_seeds.sh --smoke            # ~10 min end-to-end pipeline test
#   bash run_policy_seeds.sh                    # real run, seeds 11 22 33
#   bash run_policy_seeds.sh 11 22              # pick seeds
#   bash run_policy_seeds.sh --dry-run          # print commands only
#
#   Recommended on the compute server:
#     tmux new -s MLAG
#     bash run_policy_seeds.sh 2>&1 | tee logs/policy_seeds_$(date +%Y%m%d).log
#     # detach with Ctrl-b d
#
# RUN --smoke FIRST. It exercises all five phases with 2 epochs so a broken path
# or missing dependency surfaces in minutes rather than after hours of training.
#
# ETA
#   po_agp.py does not timestamp its own epochs, so this script timestamps them
#   and prints a projected finish after every epoch. Wall-clock per policy is
#   unknown up front (the original run's per-epoch log was not preserved); the
#   dominant cost is 8000 polygons x K=8 sequential autoregressive rollouts per
#   epoch, x 200 epochs. The first projection appears after epoch 1.
#
# RESUMABLE
#   Every phase is skipped if its output exists. Safe to re-run after an
#   interruption; safe to add seeds later.

set -euo pipefail

# ----------------------------------------------------------------- environment
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Override on the compute server if the interpreter lives elsewhere, e.g.
#   PYTHON=/opt/conda/envs/MLAG/bin/python bash run_policy_seeds.sh
PYTHON="${PYTHON:-python}"

# DATASET_PATH must point at the AGPIL root (containing train/ dev/ test/ large/).
if [[ -z "${DATASET_PATH:-}" && -f .env ]]; then
    set -a; source .env; set +a
fi
if [[ -z "${DATASET_PATH:-}" ]]; then
    echo "ERROR: DATASET_PATH is not set and .env did not provide it." >&2
    echo "       export DATASET_PATH=/path/to/AGPIL   (needs train/ dev/ test/)" >&2
    exit 1
fi
export DATASET_PATH
DATASET_ROOT="${DATASET_PATH%/}"

# ----------------------------------------------------------------------- flags
DRY_RUN=0
SMOKE=0
SEEDS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --smoke)   SMOKE=1 ;;
        --help|-h) sed -n '2,50p' "$0"; exit 0 ;;
        *)         SEEDS+=("$arg") ;;
    esac
done
[[ ${#SEEDS[@]} -eq 0 ]] && SEEDS=(11 22 33)

# Released-policy hyperparameters, recovered from
# checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt -> checkpoint_params.
# Do not change these: the whole point is that the new policies differ from the
# released one ONLY in random seed.
PO_EPOCHS=200
PO_ROLLOUTS=8
PO_ALPHA=0.05
PO_LOSS=bt
PO_LAMBDA=1.0
PO_TAU=0.99
PO_TAU_PEN=3.0
PO_TRAIN_SIZE=8000
PO_BATCH=64
PO_EVAL_K=200
PO_DISC_VIS=500

# Probe settings (paper Section 5.2): 60 epochs, batch 32, lr 3e-4, 3 layers,
# 8 heads, width 128. Per-epoch rollout eval is DISABLED (--rollout-eval-k 0):
# it dominates probe runtime and we only need the final checkpoint here.
PROBE_EPOCHS=60
PROBE_SEED=1234
PROBE_ROLLOUT_EVAL=0

if [[ $SMOKE -eq 1 ]]; then
    echo "### SMOKE MODE: 2-epoch pipeline validation, results are NOT usable ###"
    PO_EPOCHS=2
    PROBE_EPOCHS=2
    PO_TRAIN_SIZE=256
    PO_EVAL_K=30
    SEEDS=(99)
fi

mkdir -p results/policy_seeds logs

run() {
    echo "+ $*"
    # NB: an "&&" one-liner here would return 1 under --dry-run and, with
    # `set -e`, silently abort the whole script. Keep the explicit if.
    if [[ $DRY_RUN -eq 0 ]]; then
        "$@"
    fi
}

# --------------------------------------------------------------- ETA reporting
# po_agp.py prints "[epoch N] ..." with no timing. We prefix each line with
# elapsed seconds, then project the finish from the mean inter-epoch interval.
train_policy_with_eta() {
    local seed="$1" ckpt_dir="$2" log="$3"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "+ (policy training, seed=$seed, ${PO_EPOCHS} epochs) -> $ckpt_dir"
        return 0
    fi

    stdbuf -oL -eL "$PYTHON" po_agp.py \
        --epochs "$PO_EPOCHS" \
        --seed "$seed" \
        --num-rollouts "$PO_ROLLOUTS" \
        --alpha "$PO_ALPHA" \
        --preference-loss "$PO_LOSS" \
        --reward-lambda "$PO_LAMBDA" \
        --coverage-threshold "$PO_TAU" \
        --tau-penalty "$PO_TAU_PEN" \
        --fast-reward --disc-vis-samples "$PO_DISC_VIS" \
        --train-size "$PO_TRAIN_SIZE" \
        --batch-size "$PO_BATCH" \
        --epoch-eval-k "$PO_EVAL_K" \
        --skip-finetune \
        --checkpoint-dir "$ckpt_dir" 2>&1 \
    | stdbuf -oL awk -v total="$PO_EPOCHS" -v start="$(date +%s)" '
        {
            now = systime()
            printf "[%6ds] %s\n", now - start, $0
            fflush()
        }
        /^\[epoch [0-9]+\]/ {
            ep = $2; gsub(/[^0-9]/, "", ep)
            if (ep + 0 > 0) {
                el  = now - start
                per = el / (ep + 0)
                rem = per * (total - ep)
                printf "  >> ETA  epoch %d/%d | %.1f min/epoch | elapsed %.2f h | remaining %.2f h | finish ~%s\n",
                       ep, total, per/60, el/3600, rem/3600, strftime("%a %H:%M", now + rem)
                fflush()
            }
        }
    ' | tee -a "$log"
}

# ------------------------------------------------------------------------ main
echo "============================================================"
echo "  policy-seed replication"
echo "  seeds        : ${SEEDS[*]}"
echo "  policy epochs: $PO_EPOCHS   probe epochs: $PROBE_EPOCHS"
echo "  python       : $PYTHON"
echo "  dataset      : $DATASET_ROOT"
echo "============================================================"

for S in "${SEEDS[@]}"; do
    echo ""
    echo "############################################################"
    echo "#  POLICY SEED ${S}"
    echo "############################################################"

    POL_DIR="checkpoints/v3/po_agp/lstm_bt_seed${S}"
    POL_CKPT="${POL_DIR}/po_agp_best_greedy.pt"
    TRAJ_TRAIN="data/ls_trajectories_train_pseed${S}.pkl"
    TRAJ_DEV="data/ls_trajectories_dev_pseed${S}.pkl"
    FULL_DIR="checkpoints/set_predictor/pseed${S}_full"
    NOENC_DIR="checkpoints/set_predictor/pseed${S}_noenc"
    LOG="logs/policy_seed${S}.log"
    T_START=$(date +%s)

    # -- phase 1: train the policy (dominant cost) ---------------------------
    if [[ -e "$POL_CKPT" ]]; then
        echo "[skip] phase 1 — $POL_CKPT exists"
    else
        echo "--- phase 1/5: PO/BT policy training (${PO_EPOCHS} epochs) ---"
        mkdir -p "$POL_DIR"
        train_policy_with_eta "$S" "$POL_DIR" "$LOG"
        if [[ $DRY_RUN -eq 0 && ! -e "$POL_CKPT" ]]; then
            echo "ERROR: training finished but $POL_CKPT was not written." >&2
            echo "       Check $LOG — best-checkpoint saving may not have triggered." >&2
            exit 1
        fi
    fi
    T_POLICY=$(( $(date +%s) - T_START ))

    # -- phase 2: rebuild LS targets FROM THIS POLICY ------------------------
    # The probe's supervision is the policy's own greedy seed refined by local
    # search, so a new policy REQUIRES new targets. Reusing seed-1234 targets
    # here would silently invalidate the comparison.
    for SPLIT in train dev; do
        OUT_VAR="TRAJ_$(echo $SPLIT | tr a-z A-Z)"
        OUT="${!OUT_VAR}"
        if [[ -e "$OUT" ]]; then
            echo "[skip] phase 2 ($SPLIT) — $OUT exists"
        else
            echo "--- phase 2/5: LS trajectories ($SPLIT) from policy seed ${S} ---"
            run "$PYTHON" tools/build_ls_trajectories.py \
                --checkpoint "$POL_CKPT" \
                --split "$SPLIT" \
                --tau "$PO_TAU" --tau-penalty "$PO_TAU_PEN" --lam "$PO_LAMBDA" \
                --disc-vis-samples "$PO_DISC_VIS" \
                --out "$OUT"
        fi
    done

    # -- phases 3+4: the two probes that constitute the ablation -------------
    for VARIANT in full noenc; do
        if [[ "$VARIANT" == "full" ]]; then
            VDIR="$FULL_DIR"; EXTRA=()
        else
            VDIR="$NOENC_DIR"; EXTRA=(--disable-ptr-emb)
        fi
        if [[ -e "${VDIR}/set_predictor_best.pt" ]]; then
            echo "[skip] phase 3/4 ($VARIANT) — ${VDIR}/set_predictor_best.pt exists"
        else
            echo "--- phase 3/5: probe '${VARIANT}' on policy seed ${S} ---"
            run "$PYTHON" train_set_predictor.py \
                --train-traj "$TRAJ_TRAIN" \
                --val-traj "$TRAJ_DEV" \
                --pointer-checkpoint "$POL_CKPT" \
                --epochs "$PROBE_EPOCHS" \
                --batch-size 32 --lr 3e-4 \
                --predictor-hidden 128 --predictor-attn-layers 3 --predictor-heads 8 \
                --rollout-eval-k "$PROBE_ROLLOUT_EVAL" \
                --seed "$PROBE_SEED" \
                "${EXTRA[@]}" \
                --out-dir "$VDIR"
        fi
    done

    # -- phase 5: evaluate both probes against the SAME policy ---------------
    for VARIANT in full noenc; do
        if [[ "$VARIANT" == "full" ]]; then VDIR="$FULL_DIR"; else VDIR="$NOENC_DIR"; fi
        OUT="results/policy_seeds/pseed${S}_${VARIANT}.json"
        if [[ -e "$OUT" ]]; then
            echo "[skip] phase 5 ($VARIANT) — $OUT exists"
        else
            echo "--- phase 5/5: eval '${VARIANT}' (policy seed ${S}) ---"
            run "$PYTHON" eval_set_predictor.py \
                --checkpoint "${VDIR}/set_predictor_best.pt" \
                --val-traj "$TRAJ_DEV" \
                --pointer-checkpoint "$POL_CKPT" \
                --sol-dir "${DATASET_ROOT}/dev" \
                --thresholds 0.20 0.25 0.30 \
                --out "$OUT"
        fi
    done

    # -- manifest ------------------------------------------------------------
    T_TOTAL=$(( $(date +%s) - T_START ))
    if [[ $DRY_RUN -eq 0 ]]; then
        cat > "results/policy_seeds/manifest_pseed${S}.json" <<EOF
{
  "policy_seed": ${S},
  "probe_seed": ${PROBE_SEED},
  "smoke": $([[ $SMOKE -eq 1 ]] && echo true || echo false),
  "policy_epochs": ${PO_EPOCHS},
  "probe_epochs": ${PROBE_EPOCHS},
  "seconds_policy_training": ${T_POLICY},
  "seconds_total": ${T_TOTAL},
  "hyperparams": {
    "num_rollouts": ${PO_ROLLOUTS}, "alpha": ${PO_ALPHA},
    "preference_loss": "${PO_LOSS}", "reward_lambda": ${PO_LAMBDA},
    "tau": ${PO_TAU}, "tau_penalty": ${PO_TAU_PEN},
    "train_size": ${PO_TRAIN_SIZE}, "batch_size": ${PO_BATCH}
  },
  "artifacts": {
    "policy": "${POL_CKPT}",
    "traj_train": "${TRAJ_TRAIN}",
    "traj_dev": "${TRAJ_DEV}",
    "probe_full": "${FULL_DIR}/set_predictor_best.pt",
    "probe_noenc": "${NOENC_DIR}/set_predictor_best.pt",
    "eval_full": "results/policy_seeds/pseed${S}_full.json",
    "eval_noenc": "results/policy_seeds/pseed${S}_noenc.json"
  }
}
EOF
        echo "[manifest] results/policy_seeds/manifest_pseed${S}.json"
        printf "[done] policy seed %s in %.2f h (policy training %.2f h)\n" \
               "$S" "$(echo "$T_TOTAL/3600" | bc -l)" "$(echo "$T_POLICY/3600" | bc -l)"
    fi
done

echo ""
echo "============================================================"
echo "ALL_POLICY_SEEDS_DONE  (${SEEDS[*]})"
echo ""
echo "Retrieve these back to the paper repo:"
echo "  results/policy_seeds/          (eval JSON + manifests, small)"
echo "  logs/policy_seed*.log          (timings, for the ETA/cost note)"
echo "Checkpoints can stay on the server unless you want to re-evaluate."
echo ""
echo "The number that matters per seed: tail/gate metrics of _full vs _noenc."
echo "If full beats noenc on every policy seed, the encoder claim replicates"
echo "and the paper can drop its single-run caveat."
echo "============================================================"
