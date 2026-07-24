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
# THE RECIPE COMES FROM --config, NOT FROM THIS SCRIPT
#   All hyperparameters are read from configs/po_agp_transformer_bt.json, the
#   file the released run actually used (its checkpoint records the path). Do
#   not re-specify them here. To depart from the recipe on purpose, set a
#   PO_*_OVERRIDE env var, which is passed on the command line and so wins over
#   the config:
#     PO_EPOCHS_OVERRIDE  PO_BATCH_OVERRIDE  PO_LR_OVERRIDE  PO_PATIENCE_OVERRIDE
#
# SPEED / MEMORY
#   * The decode is vectorised (see models.py apply_mask_to_logits and the
#     use_fast path): ~4.4x end-to-end vs the old per-sample .item() loops,
#     verified bit-identical by tools/smoke_fast_decode.py. AGNET_LEGACY_DECODE=1
#     and AGNET_LEGACY_REWARD=1 revert the two optimisations independently.
#   * The recipe's nominal batch_size is 32 but its EFFECTIVE batch was 10,
#     because bucket_size was hardcoded to 10 when it was trained (see the
#     bucket note further down). An 8 GiB card is therefore sufficient; raising
#     AGNET_BUCKET_SIZE to make the batch really 32 is a DEPARTURE from the
#     published run, not a free speed knob.
#   * AMP is OFF in this recipe. po_agp.py defaults use_amp to True on CUDA,
#     but only when it is left unset -- the released config sets
#     "use_amp": false explicitly, so the seeds train without it. Turning it on
#     would depart from the recipe, and would buy little anyway: the step is
#     CPU/decode-latency-bound, with only ~7% real GPU compute.
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
  echo "# If this was CUDA OOM: lower the batch with PO_BATCH_OVERRIDE"
  echo "#   (AGNET_BUCKET_SIZE follows it automatically), e.g."
  echo "#   PO_BATCH_OVERRIDE=10 on an 8 GiB card. Also check no"
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

# A CPU-only torch imports fine and then trains ~20x slower, silently, for
# days -- the single most expensive way for this script to "work". Auto-picking
# $CONDA_PREFIX can easily land on a base env whose torch has no CUDA, so
# require a visible GPU up front. ALLOW_CPU=1 to override deliberately.
if [[ "${ALLOW_CPU:-0}" != "1" ]]; then
    if ! "$PYTHON" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
        {
            echo "ERROR: '$PYTHON' has torch but NO CUDA device."
            "$PYTHON" - <<'PY' 2>/dev/null || true
import torch
print(f"       torch {torch.__version__}, cuda.is_available()=False, "
      f"device_count={torch.cuda.device_count()}")
print(f"       built with CUDA: {torch.version.cuda}")
PY
            echo "       Training on CPU takes ~20x longer. Fix the interpreter"
            echo "       (PYTHON=/path/to/envs/MLAG/bin/python) or check the driver:"
            echo "         nvidia-smi"
            echo "       If NVML fails while nvidia-smi lists a healthy GPU, the"
            echo "       driver state needs a reset/reboot (root)."
            echo "       To train on CPU anyway: ALLOW_CPU=1"
        } >&2
        exit 1
    fi
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

# ---------------------------------------------------------------------------
# Released-policy recipe: ONE source of truth.
#
# The released run recorded the config file it used --
# checkpoints/v3/po_agp/lstm_bt/po_agp_final_epoch200.pt -> args['config'] ==
# 'configs/po_agp_transformer_bt.json' -- and its saved args match that file.
# So we pass --config and override ONLY --seed and --checkpoint-dir. Any
# hyperparameter re-specified here as a shell default is a chance to drift from
# the published recipe; earlier versions of this script did exactly that and
# silently differed on SIX values (batch 10 vs 32, lr 2e-4 vs 3e-4, schedule
# none vs cosine, cap_coverage true vs false -- which changes the REWARD
# FUNCTION -- patience 20 vs 40, epoch_eval_k 200 vs 50).
#
# NOTE the config's "model_type": "transformer" key is inert: model_type no
# longer exists anywhere in po_agp.py, which always builds the LSTM PointerNet
# (create_agp_model). There is no --model-type flag. So --config alone gives
# the released LSTM policy.
PO_CONFIG="${PO_CONFIG:-configs/po_agp_transformer_bt.json}"
if [[ ! -f "$PO_CONFIG" ]]; then
    echo "[fatal] recipe config '$PO_CONFIG' not found" >&2
    exit 1
fi

# Pull the few values the SCRIPT itself needs (ETA display, bucket sizing,
# manifest) straight out of the config, so they can never disagree with what
# po_agp.py actually trains with.
cfg_get() { "$PYTHON" -c "import json,sys;print(json.load(open('$PO_CONFIG')).get(sys.argv[1],''))" "$1"; }
# Epoch budget: DELIBERATELY 150, not the config's 200. The released seed's
# best checkpoint was epoch 114/200 (po_agp_best_greedy.pt -> epoch 114), so
# 115-200 improved nothing in that run, and early_stop_patience=40 means a
# similarly-behaved seed halts itself near ~154 regardless of this cap. 150
# keeps that safety margin while skipping ~13h/seed of the released run's
# unproductive tail. This is a recorded DEPARTURE from the published 200 (see
# the manifest's "epochs"/"overrides" fields) -- not a free lunch. If a seed's
# training_dynamics.jsonl shows its best epoch landing at/near the cap, that
# seed was truncated and should be resumed with PO_EPOCHS_OVERRIDE=200.
# Override per-seed with PO_EPOCHS_OVERRIDE=<n> (e.g. =200 for the exact
# released recipe); override this default itself with PO_EPOCHS=<n>.
PO_EPOCHS="${PO_EPOCHS:-150}"
PO_BATCH="${PO_BATCH:-$(cfg_get batch_size)}"
PO_TRAIN_SIZE="${PO_TRAIN_SIZE:-$(cfg_get train_size)}"
PO_DISC_VIS="${PO_DISC_VIS:-$(cfg_get disc_vis_samples)}"
# Reward/loss settings. po_agp.py receives these via --config; they are read
# here only so the LS-trajectory phase (which must score with the SAME reward
# as training) and the run manifest stay in lockstep with the recipe.
PO_ROLLOUTS="$(cfg_get num_rollouts)"
PO_ALPHA="$(cfg_get alpha)"
PO_LOSS="$(cfg_get preference_loss)"
PO_LAMBDA="$(cfg_get reward_lambda)"
PO_TAU="$(cfg_get coverage_threshold)"
PO_TAU_PEN="$(cfg_get tau_penalty)"
PO_LR="$(cfg_get lr)"
PO_PATIENCE="$(cfg_get early_stop_patience)"
PO_EVAL_K="$(cfg_get epoch_eval_k)"

# Optional overrides. These are EMPTY by default so the config wins; set the
# env var only if you deliberately want to depart from the published recipe.
# (po_agp.py's config loader applies a config value only when the flag was not
# given on the command line, so anything passed here takes precedence.)
PO_OVERRIDES=()
[[ -n "${PO_EPOCHS_OVERRIDE:-}" ]] && PO_OVERRIDES+=(--epochs "$PO_EPOCHS_OVERRIDE")
[[ -n "${PO_BATCH_OVERRIDE:-}"  ]] && PO_OVERRIDES+=(--batch-size "$PO_BATCH_OVERRIDE")
[[ -n "${PO_LR_OVERRIDE:-}"     ]] && PO_OVERRIDES+=(--lr "$PO_LR_OVERRIDE")
[[ -n "${PO_PATIENCE_OVERRIDE:-}" ]] && PO_OVERRIDES+=(--early-stop-patience "$PO_PATIENCE_OVERRIDE")
# Keep the display/bucket values consistent with an explicit override.
PO_EPOCHS="${PO_EPOCHS_OVERRIDE:-$PO_EPOCHS}"
PO_BATCH="${PO_BATCH_OVERRIDE:-$PO_BATCH}"

# THE RELEASED RUN'S EFFECTIVE BATCH WAS 10, NOT 32.
#
# po_agp.py batches via BucketBatchSampler, which chunks length-sorted indices
# into buckets of bucket_size and only THEN splits each bucket by batch_size.
# So batch_size > bucket_size is INERT: a bucket of 10 split by 32 yields one
# batch of 10.
#
# When the released policy was trained, bucket_size was HARDCODED to 10 (see
# `git show 8592243^:po_agp.py`, the commit that introduced AGNET_BUCKET_SIZE).
# Its config therefore says batch_size 32 while the run actually stepped on
# batches of 10. Config value != effective value.
#
# Consequence: leaving AGNET_BUCKET_SIZE at its default of 10 reproduces the
# released data pipeline exactly, and RAISING it to 32 would depart from the
# published run rather than match it. That is why we do NOT tie the bucket to
# batch_size here. It also means the replication fits comfortably on an 8 GiB
# card -- no >=24 GB GPU is required.
: "${AGNET_BUCKET_SIZE:=10}"
export AGNET_BUCKET_SIZE
if [[ "$AGNET_BUCKET_SIZE" != "10" ]]; then
    echo "[warn] AGNET_BUCKET_SIZE=$AGNET_BUCKET_SIZE (released run used 10)."
    echo "       Effective batch becomes min(batch_size, $AGNET_BUCKET_SIZE);"
    echo "       this DEPARTS from the published recipe."
fi

# MEMORY at the effective batch: a PO step decodes batch*K sequences
# autoregressively and holds the graph across all decode steps, so peak grows
# steeply with polygon size. Measured on a 7.57 GiB RTX 2080 SUPER:
# batch 8 -> 3.82 GiB, 10 -> 4.61 GiB, 12 -> 5.70 GiB, 16 -> OOM (and an
# effective batch of 32 or 64 OOMs at ~7.4 GiB). Size a new machine with:
#   python tools/probe_max_batch.py

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
    PROBE_EPOCHS=2
    SEEDS=(99)
    # These must go through PO_OVERRIDES: the recipe now comes from --config,
    # so plain shell assignments would change only the ETA display while
    # po_agp.py still trained the full 200-epoch recipe.
    PO_EPOCHS=2
    PO_TRAIN_SIZE=256
    PO_EVAL_K=30
    PO_OVERRIDES+=(--epochs 2 --train-size 256 --epoch-eval-k 30)
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
# Return the path of the highest-epoch periodic checkpoint in $1, or empty.
# po_agp.py writes po_agp_epoch<N>.pt every 5 epochs WITH optimizer state, so
# any of them is a valid --resume-from point.
#
# Pure bash, no pipeline: an `ls *.pt | ...` here would exit non-zero on an
# empty dir, and under `set -euo pipefail` that abort propagates through
# resume="$(latest_periodic_ckpt ...)" and kills the whole script BEFORE
# training starts. The loop below simply prints nothing for an empty dir and
# always returns 0.
latest_periodic_ckpt() {
    local dir="$1" f n best=-1 latest=""
    shopt -s nullglob
    for f in "$dir"/po_agp_epoch*.pt; do
        n="${f##*po_agp_epoch}"; n="${n%.pt}"
        [[ "$n" =~ ^[0-9]+$ ]] || continue
        if (( n > best )); then best="$n"; latest="$f"; fi
    done
    shopt -u nullglob
    printf '%s' "$latest"
}

train_policy_with_eta() {
    local seed="$1" ckpt_dir="$2" log="$3"

    # Resume from the newest periodic checkpoint if one exists. This makes an
    # interrupted multi-day run cost at most the last <5 epochs rather than the
    # whole seed -- essential on a rented box (99% reliability over days => a
    # real chance of one eviction).
    local resume="" resume_flag=()
    resume="$(latest_periodic_ckpt "$ckpt_dir")"
    if [[ -n "$resume" ]]; then
        resume_flag=(--resume-from "$resume")
        echo "  [resume] found $resume -> continuing that run"
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "+ (policy training, seed=$seed, ${PO_EPOCHS} epochs${resume:+, resume from $resume}) -> $ckpt_dir"
        return 0
    fi

    # Everything except the seed, the output dir and the cache comes from
    # --config, so this run cannot drift from the released recipe. Deliberate
    # departures go through the PO_*_OVERRIDE env vars (see the recipe block).
    stdbuf -oL -eL "$PYTHON" po_agp.py \
        --config "$PO_CONFIG" \
        --seed "$seed" \
        ${DISC_VIS_CACHE:+--disc-vis-cache-path "$DISC_VIS_CACHE"} \
        "${PO_OVERRIDES[@]}" \
        "${resume_flag[@]}" \
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
cfg_epochs="$(cfg_get epochs)"
epoch_note=""
[[ "$PO_EPOCHS" != "$cfg_epochs" ]] && epoch_note="  (config recipe: ${cfg_epochs}; this is a recorded departure)"
echo "  policy epochs: $PO_EPOCHS   probe epochs: $PROBE_EPOCHS${epoch_note}"
echo "  batch        : ${PO_BATCH} nominal / bucket ${AGNET_BUCKET_SIZE}"\
" -> effective $(( PO_BATCH < AGNET_BUCKET_SIZE ? PO_BATCH : AGNET_BUCKET_SIZE ))"
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
    # DONE is keyed on the FINAL-epoch checkpoint, not po_agp_best_greedy.pt.
    # best_greedy.pt is written mid-run (best-so-far), so keying the skip on it
    # would treat a half-trained/interrupted policy as complete. The final
    # checkpoint is written only when training reaches PO_EPOCHS.
    POL_DONE="${POL_DIR}/po_agp_final_epoch${PO_EPOCHS}.pt"
    if [[ -e "$POL_DONE" ]]; then
        echo "[skip] phase 1 — completed policy exists ($POL_DONE)"
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
    "config": "${PO_CONFIG}",
    "overrides": "${PO_OVERRIDES[*]-}",
    "num_rollouts": ${PO_ROLLOUTS}, "alpha": ${PO_ALPHA},
    "preference_loss": "${PO_LOSS}", "reward_lambda": ${PO_LAMBDA},
    "tau": ${PO_TAU}, "tau_penalty": ${PO_TAU_PEN},
    "lr": ${PO_LR}, "epochs": ${PO_EPOCHS},
    "early_stop_patience": ${PO_PATIENCE}, "epoch_eval_k": ${PO_EVAL_K},
    "train_size": ${PO_TRAIN_SIZE},
    "batch_size_nominal": ${PO_BATCH},
    "bucket_size": ${AGNET_BUCKET_SIZE},
    "batch_size_effective": $(( PO_BATCH < AGNET_BUCKET_SIZE ? PO_BATCH : AGNET_BUCKET_SIZE ))
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
