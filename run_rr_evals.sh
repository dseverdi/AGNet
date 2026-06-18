#!/usr/bin/env bash
# run_rr_evals.sh — finish the remaining R&R eval work, fast parts first.
#
# The no-seed (encoder-only) ablation is already fully done. What remains:
#   Phase 1  coords-only (no-seed + no-encoder) dev_test eval  -> completes the
#            2x2 ablation table on the in-distribution test split. FAST (362 polys).
#   Phase 2  invariance robustness + probe inference timing.    FAST (eval-only).
#   Phase 3  coords-only OOD eval -> the OOD 2x2.  SLOW and OPTIONAL: the
#            degenerate coords-only probe selects ~80% of vertices, so the exact
#            CGAL coverage over 2081 OOD polygons (n up to 1000) is expensive.
#            Skipped unless RUN_OOD=1.
#
# IDEMPOTENT: each step is skipped if its output exists. Safe to re-run.
#
# Usage (in tmux):
#   bash run_rr_evals.sh              # phases 1 + 2  (minutes) -> enough for the paper
#   RUN_OOD=1 bash run_rr_evals.sh    # also phase 3 (slow; for the OOD 2x2)
#
# When it prints RR_CORE_DONE, tell Claude "rr evals done" and paste the tail.

set -euo pipefail

PY=/home/dseverdi/.conda/envs/MLAG/bin/python
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export DATASET_PATH=/home/dseverdi/Radno/MLAG/dataset/AGPIL

SEEDS=(1234 11 22 33)
POINTER=checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt

echo "############ Phase 1: coords-only dev_test (2x2 table) ############"
for S in "${SEEDS[@]}"; do
    CK="checkpoints/set_predictor/noseednoenc_seed${S}/set_predictor_final.pt"
    OUT="paper/data/dist_dev_test_noseednoenc_seed${S}.json"
    if [[ -e "$OUT" ]]; then
        echo "[skip] $OUT"
    elif [[ ! -e "$CK" ]]; then
        echo "[WARN] missing checkpoint $CK — run run_noseed_ablation.sh first"
    else
        echo "+ coords-only dev_test seed ${S}"
        SP_CKPT="$CK" SP_SUFFIX="_noseednoenc_seed${S}" SP_SPLITS=dev_test \
            "$PY" paper/scripts/build_paper_data.py per_polygon_all
    fi
done

echo ""
echo "############ Phase 2: invariance + timing (eval-only) ############"
if [[ -e paper/data/invariance_test.json ]]; then
    echo "[skip] paper/data/invariance_test.json"
else
    "$PY" tools/eval_invariance.py --threshold 0.20 --out paper/data/invariance_test.json
fi
if [[ -e results/probe_timing.json ]]; then
    echo "[skip] results/probe_timing.json"
else
    "$PY" tools/time_probe_inference.py --out results/probe_timing.json
fi

echo ""
echo "=================== RR_CORE_DONE ==================="
echo "2x2 table + invariance + timing data are ready. Tell Claude 'rr evals done'."

if [[ "${RUN_OOD:-0}" == "1" ]]; then
    echo ""
    echo "############ Phase 3: coords-only OOD (SLOW, optional) ############"
    for S in "${SEEDS[@]}"; do
        CK="checkpoints/set_predictor/noseednoenc_seed${S}/set_predictor_final.pt"
        OUT="paper/data/dist_test_OOD_noseednoenc_seed${S}.json"
        if [[ -e "$OUT" ]]; then
            echo "[skip] $OUT"
        elif [[ ! -e "$CK" ]]; then
            echo "[WARN] missing checkpoint $CK"
        else
            echo "+ coords-only OOD seed ${S}"
            SP_CKPT="$CK" SP_SUFFIX="_noseednoenc_seed${S}" SP_SPLITS=test_OOD \
                "$PY" paper/scripts/build_paper_data.py per_polygon_all
        fi
    done
    echo "=================== RR_OOD_DONE ==================="
fi
