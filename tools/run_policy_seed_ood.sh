#!/usr/bin/env bash
# run_policy_seed_ood.sh — policy-seed variance for the OOD claim (C2).
#
# WHY THIS EXISTS
#   C2 ("the probe recovers gate coverage out of distribution") currently rests
#   on ONE policy seed (1234). The three-seed replication in the paper is
#   IN-DISTRIBUTION only, because each policy seed needs its own OOD seed sets
#   before its probes can be evaluated there. This script produces them, so C2
#   can be stated across policies instead of carrying a limitation.
#
#   The claim under test is again a WITHIN-policy contrast: does the full probe
#   beat the no-encoder ablation on the OOD coverage tail, for every policy?
#   Absolute coverage varying across policy seeds does not threaten C2; the gap
#   vanishing would.
#
# WHY --ls-max-iter 0 (this is the whole reason it is ~4 h and not ~20 h)
#   The LS refinement is what makes trajectory building expensive, and it exists
#   only to produce probe TRAINING targets. The probes already exist (trained on
#   in-distribution targets by run_policy_seeds.sh); here we only EVALUATE them,
#   and eval_set_predictor.py reads just rec["seed"] -- it rescores coverage
#   itself with exact CGAL and discards the target. So we run the same vetted
#   greedy-decode path with the refinement loop disabled, which makes
#   final == seed. SetPredDataset requires `final` to be non-empty (it filters
#   on it), which that satisfies.
#
#   Consequence: these pickles carry NO usable LS target. The `_seedonly` suffix
#   is deliberate -- do NOT train a probe on them.
#
# DISC-VIS CACHE SAFETY
#   utils.get_or_build_disc_vis evicts down to AGNET_DISC_VIS_CACHE_SIZE
#   (default 10000) whenever a MISS adds an entry, and the master cache holds
#   ~12.1k -- so a single miss under the default would silently drop ~2k entries
#   and write the truncated cache back. We raise the cap well above the loaded
#   count. build_ls_trajectories.py additionally skips the writeback entirely
#   when nothing changed, which it reports.
#
# WHAT IT PRODUCES (per policy seed S, per split SP in {test,large})
#   data/ls_trajectories_${SP}_pseed${S}_seedonly.pkl      policy greedy seeds
#   results/policy_seeds_ood/pseed${S}_${ARM}_${SP}.json   eval, ARM in full/noenc
#
# USAGE
#   bash tools/run_policy_seed_ood.sh                # seeds 11 22 33, both splits
#   bash tools/run_policy_seed_ood.sh 11             # one seed
#   SKIP_BUILD=1 bash tools/run_policy_seed_ood.sh   # eval only (pickles exist)
#   DRY_RUN=1    bash tools/run_policy_seed_ood.sh   # print commands only
#
# Restartable: every step skips when its output already exists.
set -uo pipefail

cd "$(dirname "$0")/.."
[[ -f .env ]] && { set -a; . ./.env; set +a; }

PYTHON="${PYTHON:-/home/dseverdi/.conda/envs/MLAG/bin/python}"
SEEDS=("${@:-11 22 33}")
read -r -a SEEDS <<< "${SEEDS[*]}"
SPLITS=(test large)

# Reward hyperparameters -- must match run_policy_seeds.sh or the greedy seeds
# would come from a different objective than the probes were trained against.
PO_TAU=0.99; PO_TAU_PEN=3.0; PO_LAMBDA=1.0; PO_DISC_VIS=500

# t=0.20 only -- the headline operating point, and the only threshold
# tab_policy_seeds reads. Each extra threshold costs another exact-CGAL coverage
# pass over every polygon (seed metrics are 1 pass, then 1 per threshold at K=1),
# so sweeping {0.20,0.25,0.30} doubles a 2081-polygon eval for numbers no table
# uses. Measured: >25 min per eval at three thresholds. The operating curve is
# already reported for policy 1234 in tab_operating_curve.
THRESHOLDS="${THRESHOLDS:-0.20}"

export AGNET_DISC_VIS_CACHE_SIZE="${AGNET_DISC_VIS_CACHE_SIZE:-14000}"

OUTDIR=results/policy_seeds_ood
mkdir -p "$OUTDIR" logs

run() {
    echo "+ $*"
    [[ -n "${DRY_RUN:-}" ]] && return 0
    "$@"
}

# Subset a freshly built pickle to the paper's reporting population.
#
# Splits are directory-defined and the source directories overlapped, so the raw
# `test` split contains 2107 polygons of which 26 are byte-identical duplicates
# of train/dev instances. tools/dedup_partitions.py excludes those from every
# reported artifact, leaving 2081 -- so a pickle built straight from the split
# directory is the PRE-dedup superset and its aggregates would not be comparable
# with tab_ood. We therefore filter by name against the authoritative leak-free
# per-polygon dump and assert the exact expected count.
#
# `large` needs no filtering (raw 285 == reported 285); the guard below asserts
# that rather than assuming it.
dedup_to_paper_population() {
    local traj="$1" split="$2"
    "$PYTHON" - "$traj" "$split" <<'PY'
import json, pickle, sys
traj, split = sys.argv[1], sys.argv[2]
REF = {"test": ("paper/data/dist_test_OOD.json", 2081),
       "large": ("paper/data/dist_ood_large.json", 285)}[split]
ref_path, expect = REF
names = {r["name"] for r in json.load(open(ref_path))["polygons"]}
assert len(names) == expect, f"reference {ref_path} has {len(names)}, expected {expect}"

with open(traj, "rb") as f:
    d = pickle.load(f)
recs = d["records"]
if len(recs) == expect:
    print(f"  [dedup] {traj}: already {expect} records, unchanged")
    sys.exit(0)
kept = [r for r in recs if r["name"] in names]
missing = names - {r["name"] for r in kept}
if missing:
    sys.exit(f"  [dedup] FAIL: {len(missing)} reference polygons absent from "
             f"{traj}, e.g. {sorted(missing)[:3]}")
assert len(kept) == expect, f"filtered to {len(kept)}, expected {expect}"
d["records"] = kept
d.setdefault("provenance", {})["deduped_to"] = ref_path
with open(traj, "wb") as f:
    pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"  [dedup] {traj}: {len(recs)} -> {expect} records "
      f"(dropped {len(recs)-expect} train-overlapping)")
PY
}

cache_guard() {
    "$PYTHON" - <<'PY'
import pickle, sys
try:
    n = len(pickle.load(open('data/disc_vis_cache.pkl','rb')))
except Exception as e:
    print(f"  [cache] unreadable: {e}"); sys.exit(0)
print(f"  [cache] {n} entries")
if n < 12132:
    print(f"  [cache] *** SHRANK below 12132 -- restore from "
          f"data/disc_vis_cache.pkl.bak_pseedood ***")
    sys.exit(1)
PY
}

echo "=== policy-seed OOD: seeds ${SEEDS[*]}, splits ${SPLITS[*]} ==="
cache_guard || exit 1

for S in "${SEEDS[@]}"; do
    POL_CKPT="checkpoints/v3/po_agp/lstm_bt_seed${S}/po_agp_best_greedy.pt"
    [[ -f "$POL_CKPT" ]] || { echo "[skip] seed $S: no $POL_CKPT"; continue; }

    for SP in "${SPLITS[@]}"; do
        TRAJ="data/ls_trajectories_${SP}_pseed${S}_seedonly.pkl"

        if [[ -n "${SKIP_BUILD:-}" || -s "$TRAJ" ]]; then
            echo "[skip] build $TRAJ (exists)"
        else
            echo "--- build seeds: policy $S, split $SP ---"
            run "$PYTHON" tools/build_ls_trajectories.py \
                --checkpoint "$POL_CKPT" \
                --split "$SP" \
                --ls-max-iter 0 \
                --tau "$PO_TAU" --tau-penalty "$PO_TAU_PEN" --lam "$PO_LAMBDA" \
                --disc-vis-samples "$PO_DISC_VIS" \
                --out "$TRAJ" || { echo "[FAIL] build $TRAJ"; continue; }
            cache_guard || exit 1
        fi

        # Always run, including on the skip path: a pickle built before this
        # step existed is the pre-dedup superset.
        if [[ -z "${DRY_RUN:-}" ]]; then
            dedup_to_paper_population "$TRAJ" "$SP" || {
                echo "[FAIL] dedup $TRAJ"; continue; }
        fi

        for ARM in full noenc; do
            PROBE="checkpoints/set_predictor/pseed${S}_${ARM}/set_predictor_final.pt"
            [[ -f "$PROBE" ]] || { echo "[skip] no probe $PROBE"; continue; }
            OUT="$OUTDIR/pseed${S}_${ARM}_${SP}.json"
            if [[ -s "$OUT" ]]; then echo "[skip] eval $OUT (exists)"; continue; fi

            # --sol-dir gives |S|/OPT where an optimum exists. On `large` it is
            # missing for 79 polygons (n>=800), so that split reports coverage
            # and |S|/n only -- same convention as tab_large.
            SOL_ARG=()
            [[ -n "${DATASET_PATH:-}" ]] && SOL_ARG=(--sol-dir "${DATASET_PATH}/${SP}")

            echo "--- eval: policy $S, $ARM, split $SP ---"
            run "$PYTHON" eval_set_predictor.py \
                --checkpoint "$PROBE" \
                --val-traj "$TRAJ" \
                --pointer-checkpoint "$POL_CKPT" \
                --thresholds $THRESHOLDS \
                --batch-size 32 \
                "${SOL_ARG[@]}" \
                --out "$OUT" || echo "[FAIL] eval $OUT"
        done
    done
done

echo
echo "=== done. cache check: ==="
cache_guard
ls -la "$OUTDIR" 2>/dev/null
echo "POLICY_SEED_OOD_DONE"
