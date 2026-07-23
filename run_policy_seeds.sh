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
# FASTEST OPTION -- ONE SEED PER GPU, IN PARALLEL
#   Policy training dominates the cost and the seeds are fully independent, so
#   with multiple GPUs run them concurrently. This needs no extra flags: every
#   artefact path is already per-seed, and the shared disc-vis cache is now
#   read-only in training and only rewritten by the trajectory builder when it
#   actually computed something new, so there is no write race.
#     tmux new -s MLAG
#     CUDA_VISIBLE_DEVICES=0 bash run_policy_seeds.sh 11 2>&1 | tee logs/seed11.log &
#     CUDA_VISIBLE_DEVICES=1 bash run_policy_seeds.sh 22 2>&1 | tee logs/seed22.log &
#     CUDA_VISIBLE_DEVICES=2 bash run_policy_seeds.sh 33 2>&1 | tee logs/seed33.log &
#     wait
#   Three seeds then cost about the wall-clock of one. On a single GPU, run the
#   default sequential form instead -- concurrent seeds on one GPU will contend
#   for memory and be slower than running them one after another.
#
# OTHER SPEED LEVERS (in rough order of value)
#   * PO_BATCH   -- rollout decoding is latency-bound, so a larger batch is
#                   close to free until the GPU saturates. Default 128 here.
#                   Try PO_BATCH=256 if memory allows; drop to 64 on OOM.
#   * PO_EPOCHS  -- the released policy's best checkpoint was epoch 114 of 200,
#                   so 200 is likely more than needed. Shortening changes the
#                   recipe, but since the claim is within-policy that is
#                   defensible if every seed uses the same value.
#   * AMP        -- already on by default on CUDA (po_agp.py resolves use_amp
#                   to True when device.type == "cuda"); nothing to enable.
#   * PO_EVAL_K  -- per-epoch eval costs PO_EVAL_K*K extra decodes per epoch
#                   (200*8 against 8000*8 of training, so only ~2%). Lowering
#                   it is a marginal win; left at 200 for comparability.
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

# With `set -e` the script aborts at the failing command, so the per-phase
# guards further down never get to print. On an unattended multi-day run that
# leaves only a bare traceback in the log, so surface the context here.
trap 'rc=$?; {
  echo ""
  echo "############################################################"
  echo "# FAILED (exit $rc) at line $LINENO: ${BASH_COMMAND%% *}"
  echo "#"
  echo "# If this was CUDA OOM: lower PO_BATCH (and AGNET_BUCKET_SIZE"
  echo "#   follows it automatically), e.g. PO_BATCH=16. Also check no"
  echo "#   orphaned python process is holding the GPU:"
  echo "#     nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
  echo "# If it was ModuleNotFoundError: torch, set PYTHON explicitly."
  echo "#"
  echo "# Every phase is idempotent, so fix and re-run the same command."
  echo "# BUT: po_agp_best_greedy.pt is written DURING training, so if a"
  echo "#   policy run died part-way, DELETE its checkpoint dir first or"
  echo "#   phase 1 will be skipped and a half-trained policy used:"
  echo "#     rm -rf checkpoints/v3/po_agp/lstm_bt_seed<S>"
  echo "############################################################"
} >&2' ERR

# ----------------------------------------------------------------- environment
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# Override on the compute server if the interpreter lives elsewhere, e.g.
#   PYTHON=/path/to/envs/MLAG/bin/python bash run_policy_seeds.sh
#
# Prefer an explicit PYTHON; otherwise, if a conda env is active, use its
# python (avoids a leftover `source .venv` shadowing conda — a common cause of
# ModuleNotFoundError: torch when the prompt shows both (MLAG) and (.venv)).
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
        PYTHON="${CONDA_PREFIX}/bin/python"
    else
        PYTHON="python"
    fi
fi

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

# Fail fast if this interpreter has no torch (wrong venv / wrong conda env).
if ! "$PYTHON" -c 'import torch' 2>/dev/null; then
    echo "ERROR: '$PYTHON' cannot import torch." >&2
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        echo "       A virtualenv is still active: VIRTUAL_ENV=$VIRTUAL_ENV" >&2
        echo "       Run:  deactivate && conda activate MLAG" >&2
    else
        echo "       Activate the MLAG conda env (or set PYTHON=.../envs/MLAG/bin/python)." >&2
        echo "       Example:  conda activate MLAG && which python && python -c 'import torch'" >&2
    fi
    exit 1
fi

# ----------------------------------------------------------------------- flags
DRY_RUN=0
SMOKE=0
SEEDS=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --smoke)   SMOKE=1 ;;
        --help|-h) sed -n '2,50p' "$0"; exit 0 ;;
        *)
            if [[ ! "$arg" =~ ^[0-9]+$ ]]; then
                echo "ERROR: seed arguments must be integers, got: '$arg'" >&2
                echo "       usage: bash run_policy_seeds.sh [11 22 33] [--smoke] [--dry-run]" >&2
                exit 1
            fi
            SEEDS+=("$arg")
            ;;
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
# Batch size is NOT recorded in the released checkpoint (checkpoint_params has
# only the model/reward hyperparameters), so it cannot be matched exactly in any
# case. Raising it is safe here because the claim under test is a WITHIN-policy
# contrast -- each policy is compared against its own no-encoder ablation -- so
# hyperparameter drift across policies does not threaten it, provided all seeds
# use the same value. Rollout decoding is latency-bound (K=8 sequential LSTM
# decodes), so a bigger batch buys parallelism per sequential step almost for
# free until the GPU saturates. Override with PO_BATCH=... if you hit OOM.
PO_BATCH="${PO_BATCH:-10}"
PO_LR="${PO_LR:-2e-4}"

# po_agp.py batches via BucketBatchSampler, which chunks length-sorted indices
# into buckets and only then splits each bucket by batch_size. With bucket_size
# = 10 any larger --batch-size is INERT: every bucket yields one batch of 10.
#
# That cap is a MEMORY constraint, not an oversight. A PO step decodes
# batch*K sequences autoregressively and holds the graph across all decode
# steps, so peak memory grows steeply with polygon size. Measured at the
# worst-case training polygon (n=198, K=8) on a 7.57 GiB RTX 2080 SUPER:
#     batch  8 -> 3.82 GiB      batch 12 -> 5.70 GiB
#     batch 10 -> 4.61 GiB      batch 16 -> OOM
# So 10 is about the ceiling on an 8 GiB card, and the ~73 min/epoch
# (~10 days per 200-epoch seed) that implies is inherent to that hardware.
#
# The defaults below therefore reproduce the released recipe and are SAFE on
# 8 GiB. On a larger card raise BOTH together -- bucket_size must be >=
# batch_size or the batch is silently capped. Size it for the target machine:
#     python tools/probe_max_batch.py
# Roughly batch ~= 1.8 * (GPU GiB). Note the speedup is sub-linear: decoding is
# latency-bound, so k times fewer steps is not k times faster. Changing the
# batch also changes optimisation (fewer, larger gradient steps); that is
# acceptable for a within-policy contrast provided every seed uses one value.
export AGNET_BUCKET_SIZE="${AGNET_BUCKET_SIZE:-$PO_BATCH}"
PO_EVAL_K=200
PO_DISC_VIS=500

# Reward is discretised visibility (--fast-reward), matching the released policy:
# the lstm_bt config was never committed, but its same-era sibling
# configs/po_agp_transformer_bt.json matches the released checkpoint on every
# recorded parameter (K=8, alpha=0.05, lambda=1.0, tau=0.99, 200 epochs) and sets
# fast_reward=true / disc_vis_samples=500. Without --fast-reward the reward path
# is exact CGAL (po_reward_smooth), which would be a second difference from the
# released run on top of the seed -- do not drop the flag.
#
# Reusing the prebuilt disc-vis cache matters for two reasons, not just speed:
#   cost        -- otherwise ~8k per-polygon visibility matrices are rebuilt per run;
#   consistency -- the M sample points are seeded by hash(name), and Python salts
#                  string hashes per process, so without a shared cache every run
#                  would draw DIFFERENT points. tools/build_ls_trajectories.py
#                  loads this same cache by default, so the policy and the probe
#                  targets must come from it too or they are scored on different
#                  point sets, and each seed would differ by an uncontrolled
#                  nuisance variable on top of its seed.
# Unset DISC_VIS_CACHE to force recomputation, or point it elsewhere.
DISC_VIS_CACHE="${DISC_VIS_CACHE-data/disc_vis_cache.pkl}"
if [[ -n "$DISC_VIS_CACHE" && ! -e "$DISC_VIS_CACHE" ]]; then
    echo "[warn] disc-vis cache '$DISC_VIS_CACHE' not found. It will be built"
    echo "       from scratch (slow first epoch) and, because hash(name) is"
    echo "       salted per process, the sample points will differ between runs."
    echo "       Prefer copying the cache to the compute server."
fi

# Make hash(name) reproducible so that any polygon NOT covered by the cache
# still gets the same M sample points in every run (see above).
export PYTHONHASHSEED=0

# The in-process disc-vis cache defaults to 10000 entries, but train+dev is
# 8867+1224 = 10091, so loading the full persisted cache would evict at the
# margin and rebuild those matrices on demand. Raise the ceiling (~1 GB RAM).
export AGNET_DISC_VIS_CACHE_SIZE="${AGNET_DISC_VIS_CACHE_SIZE:-12000}"

# Probe settings (paper Section 5.2): 60 epochs, batch 32, lr 3e-4, 3 layers,
# 8 heads, width 128. Per-epoch rollout eval is DISABLED (--rollout-eval-k 0):
# it dominates probe runtime and we only need the final checkpoint here.
PROBE_EPOCHS=60
PROBE_SEED=1234
# Per-epoch rollout eval is OFF, and we evaluate the FINAL checkpoint rather
# than set_predictor_best.pt. Two reasons:
#   1. train_set_predictor.py only writes set_predictor_best.pt when some
#      threshold beats the policy seed on coverage AND guard count. That fires
#      for some conditions and not others (the coords-only probe never triggers
#      it), so "best" would give the ablation arms checkpoints chosen by
#      different logic at different epochs -- a confound in a controlled
#      full-vs-no-encoder comparison.
#   2. "best" is a selection, and selection is exactly what leaked in the
#      published probe runs (their val-traj was the POOLED dev pickle, so the
#      selection metric saw 64 of the 362 held-out test polygons). Taking the
#      final epoch removes the selection step altogether, so no leak is
#      possible here regardless of which pickle is passed.
# Setting PROBE_ROLLOUT_EVAL>0 re-enables the sweep; it would then select on the
# TUNE partition only, which is legitimate but reintroduces the asymmetry in 1.
PROBE_ROLLOUT_EVAL=0
PROBE_CKPT_NAME="set_predictor_final.pt"

if [[ $SMOKE -eq 1 ]]; then
    echo "### SMOKE MODE: 2-epoch pipeline validation, results are NOT usable ###"
    # Phase 2 dominates a smoke run if left unbounded: the real trajectory
    # builds took 67.5 min (train, 8867) and 9.6 min (dev, 1224) as recorded in
    # their summaries. Cap them so the smoke test exercises every phase in
    # minutes. The carve must then tolerate missing reference polygons.
    TRAJ_N=200
    CARVE_STRICT=0
    PO_EPOCHS=2
    PROBE_EPOCHS=2
    PO_TRAIN_SIZE=256
    PO_EVAL_K=30
    SEEDS=(99)
fi

# Full splits and strict carve unless --smoke overrides them.
TRAJ_N="${TRAJ_N:-}"
CARVE_STRICT="${CARVE_STRICT:-1}"

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
        ${DISC_VIS_CACHE:+--disc-vis-cache-path "$DISC_VIS_CACHE"} \
        --train-size "$PO_TRAIN_SIZE" \
        --batch-size "$PO_BATCH" \
        --lr "$PO_LR" \
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
echo "  python       : $PYTHON  ($("$PYTHON" -c 'import sys; print(sys.executable)'))"
echo "  torch/cuda   : $("$PYTHON" -c 'import torch; print(torch.__version__, "cuda="+str(torch.cuda.is_available()))')"
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
    # Pooled dev directory (1224 = dev+test). NOT usable directly: it must be
    # carved into the canonical tune/test partitions, see phase 2b.
    TRAJ_DEV="data/ls_trajectories_dev_pseed${S}.pkl"
    TRAJ_TUNE="data/ls_trajectories_dev_tune_pseed${S}.pkl"   # 857, selection
    TRAJ_TEST="data/ls_trajectories_dev_test_pseed${S}.pkl"   # 362, held out
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
                ${TRAJ_N:+--n-samples "$TRAJ_N"} \
                --out "$OUT"
        fi
    done

    # -- phase 2b: carve the pooled dev trajectories into the paper's splits --
    # DATASET_PATH/dev holds 1224 polygons: dev and test POOLED. The paper's
    # test split is the 362 that survive the 70/30 carve plus train-leak dedup.
    # Training or evaluating on the raw 1224 would mix the tuning and held-out
    # partitions and produce numbers not comparable with the paper. We filter by
    # polygon name against the canonical pickles so the partition is identical.
    if [[ -e "$TRAJ_TUNE" && -e "$TRAJ_TEST" ]]; then
        echo "[skip] phase 2b -- $TRAJ_TUNE and $TRAJ_TEST exist"
    else
        echo "--- phase 2b/5: carve dev into tune(857)/test(362) ---"
        run "$PYTHON" tools/carve_traj_by_reference.py \
            --input "$TRAJ_DEV" \
            --ref-tune data/ls_trajectories_dev_tune.pkl \
            --ref-test data/ls_trajectories_dev_test_clean.pkl \
            --out-tune "$TRAJ_TUNE" \
            --out-test "$TRAJ_TEST" \
            $([[ "$CARVE_STRICT" == "0" ]] && echo --no-strict)
    fi

    # -- phases 3+4: the two probes that constitute the ablation -------------
    for VARIANT in full noenc; do
        if [[ "$VARIANT" == "full" ]]; then
            VDIR="$FULL_DIR"; EXTRA=()
        else
            VDIR="$NOENC_DIR"; EXTRA=(--disable-ptr-emb)
        fi
        if [[ -e "${VDIR}/${PROBE_CKPT_NAME}" ]]; then
            echo "[skip] phase 3/4 ($VARIANT) — ${VDIR}/${PROBE_CKPT_NAME} exists"
        else
            echo "--- phase 3/5: probe '${VARIANT}' on policy seed ${S} ---"
            # val-traj is the TUNE partition: any checkpoint/threshold selection
            # must not see the held-out 362.
            run "$PYTHON" train_set_predictor.py \
                --train-traj "$TRAJ_TRAIN" \
                --val-traj "$TRAJ_TUNE" \
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
            # Evaluate on the held-out 362, the same split as Table tab:headline.
            run "$PYTHON" eval_set_predictor.py \
                --checkpoint "${VDIR}/${PROBE_CKPT_NAME}" \
                --val-traj "$TRAJ_TEST" \
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
    "traj_dev_pooled": "${TRAJ_DEV}",
    "traj_tune": "${TRAJ_TUNE}",
    "traj_test": "${TRAJ_TEST}",
    "probe_full": "${FULL_DIR}/${PROBE_CKPT_NAME}",
    "probe_noenc": "${NOENC_DIR}/${PROBE_CKPT_NAME}",
    "eval_full": "results/policy_seeds/pseed${S}_full.json",
    "eval_noenc": "results/policy_seeds/pseed${S}_noenc.json"
  }
}
EOF
        echo "[manifest] results/policy_seeds/manifest_pseed${S}.json"
        # awk, not bc: bc is not installed on every box, and an empty $(... | bc)
        # made printf %.2f fail under `set -e`, aborting AFTER all real work was
        # done. awk is guaranteed present wherever this script's awk ETA runs.
        awk -v t="$T_TOTAL" -v p="$T_POLICY" -v s="$S" 'BEGIN{
            printf "[done] policy seed %s in %.2f h (policy training %.2f h)\n", s, t/3600, p/3600 }'
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
