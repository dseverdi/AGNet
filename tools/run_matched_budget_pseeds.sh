#!/usr/bin/env bash
# run_matched_budget_pseeds.sh — does C1 hold across independently trained policies?
#
# WHY THIS EXISTS
#   C1 ("the frozen encoder, not the seed or the coordinates, is what makes
#   economical coverage reachable") is supported by ONE experiment: the wide
#   threshold sweep in paper/data/matched_budget/, which compares arms at matched
#   guard budget instead of matched threshold. That sweep varies the PROBE seed
#   but is entirely on policy 1234.
#
#   We now know the fixed-threshold version of the same comparison holds on only
#   2 of 4 policies out of distribution. So the obvious question is whether the
#   matched-budget result is also a property of that one policy. This script
#   answers it: same sweep, same thresholds, same split, on policies 11/22/33.
#
#   Only `full` and `noenc` probes exist per policy (run_policy_seeds.sh trains
#   those two), so this tests encoder-present vs encoder-absent -- which is the
#   comparison C1 turns on. The no-seed and coords-only arms remain policy-1234
#   only.
#
# PROVENANCE
#   The original 16 sweeps were run ad hoc and their JSONs record only
#   n_polygons and wall_s -- no checkpoint or input paths. This script writes a
#   sidecar .prov.json next to each output so the inputs are recoverable.
#
# OUTPUT
#   paper/data/matched_budget_pseeds/{full,noenc}_pseed<S>.json
#   paper/data/matched_budget_pseeds/{full,noenc}_pseed<S>.prov.json
#
# USAGE
#   bash tools/run_matched_budget_pseeds.sh              # policies 11 22 33
#   bash tools/run_matched_budget_pseeds.sh 11           # one policy
#   DRY_RUN=1 bash tools/run_matched_budget_pseeds.sh    # print commands only
set -uo pipefail
cd "$(dirname "$0")/.."
[[ -f .env ]] && { set -a; . ./.env; set +a; }

PYTHON="${PYTHON:-/home/dseverdi/.conda/envs/MLAG/bin/python}"
read -r -a SEEDS <<< "${*:-11 22 33}"

# Identical to the policy-1234 sweep in paper/data/matched_budget/.
THRESHOLDS="0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60 0.70 0.80"
OUTDIR=paper/data/matched_budget_pseeds
mkdir -p "$OUTDIR" logs

# A miss adds an entry and evicts down to AGNET_DISC_VIS_CACHE_SIZE (default
# 10000) while the master cache holds ~12.1k; raise it so a miss cannot truncate.
export AGNET_DISC_VIS_CACHE_SIZE="${AGNET_DISC_VIS_CACHE_SIZE:-14000}"

for S in "${SEEDS[@]}"; do
    POL="checkpoints/v3/po_agp/lstm_bt_seed${S}/po_agp_best_greedy.pt"
    TRAJ="data/ls_trajectories_dev_test_pseed${S}.pkl"
    [[ -f "$POL"  ]] || { echo "[skip] policy $S: no $POL";  continue; }
    [[ -f "$TRAJ" ]] || { echo "[skip] policy $S: no $TRAJ"; continue; }

    for ARM in full noenc; do
        PROBE="checkpoints/set_predictor/pseed${S}_${ARM}/set_predictor_final.pt"
        OUT="$OUTDIR/${ARM}_pseed${S}.json"
        [[ -f "$PROBE" ]] || { echo "[skip] no probe $PROBE"; continue; }
        if [[ -s "$OUT" ]]; then echo "[skip] $OUT exists"; continue; fi

        echo "--- sweep: policy $S, arm $ARM ---"
        if [[ -n "${DRY_RUN:-}" ]]; then
            echo "+ $PYTHON eval_set_predictor.py --checkpoint $PROBE --val-traj $TRAJ ..."
            continue
        fi
        "$PYTHON" eval_set_predictor.py \
            --checkpoint "$PROBE" \
            --val-traj "$TRAJ" \
            --pointer-checkpoint "$POL" \
            --thresholds $THRESHOLDS \
            --batch-size 32 \
            ${DATASET_PATH:+--sol-dir "${DATASET_PATH}/dev"} \
            --out "$OUT" || { echo "[FAIL] $OUT"; continue; }

        "$PYTHON" - "$OUT" "$PROBE" "$TRAJ" "$POL" "$THRESHOLDS" <<'PY'
import json, sys
out, probe, traj, pol, thr = sys.argv[1:6]
json.dump({"probe_checkpoint": probe, "val_traj": traj,
           "pointer_checkpoint": pol, "thresholds": thr.split(),
           "split": "dev_test (362, post-dedup)",
           "note": "policy-seed companion to paper/data/matched_budget/ "
                   "(which is policy 1234, varying the probe seed)"},
          open(out.replace(".json", ".prov.json"), "w"), indent=1)
print(f"  wrote {out} + provenance")
PY
    done
done
echo "MATCHED_BUDGET_PSEEDS_DONE"
