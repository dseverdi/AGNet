#!/usr/bin/env bash
# eval_extreme_ood.sh — evaluate SetPredictor on ls_trajectories_large.pkl
#
# "Extreme OOD": polygons with n=600–2250 vertices, trained only on n≤198.
# Runs 4 probe seeds + no-encoder ablation; each eval is a threshold sweep
# at t∈{0.20, 0.25, 0.30}.
#
# Usage:
#   bash eval_extreme_ood.sh [--dry-run]
#
# Output: results/v3/setpred_extreme_ood_{variant}.json
#         results/v3/setpred_extreme_ood_summary.json  (aggregate across seeds)

set -euo pipefail

PYTHON=/home/dseverdi/.conda/envs/MLAG/bin/python
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

TRAJ=data/ls_trajectories_large.pkl
POINTER=checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt
SOL_DIR=/home/dseverdi/Radno/MLAG/dataset/AGPIL/large
OUTDIR=results/v3
THRESHOLDS="0.20 0.25 0.30"

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    echo "[dry-run] no commands will execute"
fi

mkdir -p "$OUTDIR"

run_eval() {
    local variant="$1"
    local ckpt="$2"
    local out="$OUTDIR/setpred_extreme_ood_${variant}.json"

    echo ""
    echo "========================================================"
    echo "  variant : $variant"
    echo "  ckpt    : $ckpt"
    echo "  out     : $out"
    echo "========================================================"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry-run] would run: $PYTHON eval_set_predictor.py \\"
        echo "    --checkpoint $ckpt \\"
        echo "    --val-traj $TRAJ \\"
        echo "    --pointer-checkpoint $POINTER \\"
        echo "    --sol-dir $SOL_DIR \\"
        echo "    --thresholds $THRESHOLDS \\"
        echo "    --batch-size 4 \\"
        echo "    --out $out"
        return
    fi

    $PYTHON eval_set_predictor.py \
        --checkpoint "$ckpt" \
        --val-traj "$TRAJ" \
        --pointer-checkpoint "$POINTER" \
        --sol-dir "$SOL_DIR" \
        --thresholds $THRESHOLDS \
        --batch-size 4 \
        --out "$out"

    echo "[done] saved -> $out"
}

# 4 probe seeds (seed1234 = "standard")
run_eval "seed1234"  "checkpoints/set_predictor/standard/set_predictor_best.pt"
run_eval "seed11"    "checkpoints/set_predictor/seed11/set_predictor_best.pt"
run_eval "seed22"    "checkpoints/set_predictor/seed22/set_predictor_best.pt"
run_eval "seed33"    "checkpoints/set_predictor/seed33/set_predictor_best.pt"

# No-encoder ablation (disable_ptr_emb=True is read from checkpoint)
run_eval "no_encoder" "checkpoints/set_predictor/no_encoder/set_predictor_best.pt"

# ──────────────────────────────────────────────────────────────────────
# Aggregate summary across 4 seeds (at each threshold)
# ──────────────────────────────────────────────────────────────────────
if [[ $DRY_RUN -eq 0 ]]; then
    echo ""
    echo "Computing aggregate summary..."
    $PYTHON - <<'PYEOF'
import json, numpy as np, os, sys

OUTDIR = "results/v3"
SEEDS  = ["seed1234", "seed11", "seed22", "seed33"]
THRESHOLDS = [0.20, 0.25, 0.30]

seed_data = []
for s in SEEDS:
    path = os.path.join(OUTDIR, f"setpred_extreme_ood_{s}.json")
    if not os.path.exists(path):
        print(f"  [warn] missing {path}, skipping seed {s}")
        continue
    with open(path) as f:
        seed_data.append(json.load(f))

if not seed_data:
    print("No seed results found, aborting summary.")
    sys.exit(1)

noenc_path = os.path.join(OUTDIR, "setpred_extreme_ood_no_encoder.json")
noenc_data = None
if os.path.exists(noenc_path):
    with open(noenc_path) as f:
        noenc_data = json.load(f)

def aggregate_cells(data_list, t, K=1):
    key = f"t={t}|K={K}"
    covs, chvs, opts = [], [], []
    for d in data_list:
        cell = d["cells"].get(key)
        if cell and cell["cov"] is not None:
            covs.append(cell["cov"])
        if cell and cell["chv"] is not None:
            chvs.append(cell["chv"])
        if cell and cell["opt"] is not None:
            opts.append(cell["opt"])
    return {
        "cov_mean": float(np.mean(covs)) if covs else None,
        "cov_std":  float(np.std(covs))  if covs else None,
        "chv_mean": float(np.mean(chvs)) if chvs else None,
        "chv_std":  float(np.std(chvs))  if chvs else None,
        "opt_mean": float(np.mean(opts)) if opts else None,
        "opt_std":  float(np.std(opts))  if opts else None,
        "n_seeds":  len(covs),
    }

def aggregate_seed_metric(data_list):
    covs = [d["seed"]["cov"] for d in data_list if d["seed"].get("cov") is not None]
    chvs = [d["seed"]["chv"] for d in data_list if d["seed"].get("chv") is not None]
    return {"cov_mean": float(np.mean(covs)) if covs else None,
            "chv_mean": float(np.mean(chvs)) if chvs else None}

summary = {
    "split":      "extreme_ood_large",
    "n_polygons": seed_data[0]["n_polygons"] if seed_data else 0,
    "n_seeds":    len(seed_data),
    "seed_policy": aggregate_seed_metric(seed_data),
    "probe_4seed": {
        f"t={t}": aggregate_cells(seed_data, t) for t in THRESHOLDS
    },
}
if noenc_data:
    summary["no_encoder"] = {
        f"t={t}": {
            "cov_mean": noenc_data["cells"].get(f"t={t}|K=1", {}).get("cov"),
            "chv_mean": noenc_data["cells"].get(f"t={t}|K=1", {}).get("chv"),
            "opt_mean": noenc_data["cells"].get(f"t={t}|K=1", {}).get("opt"),
        } for t in THRESHOLDS
    }

out = os.path.join(OUTDIR, "setpred_extreme_ood_summary.json")
with open(out, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  saved -> {out}")

# Print table
print()
print(f"{'Variant':<22} {'t':>5}  {'cov':>8}  {'|S|/n':>8}  {'|S|/OPT':>9}")
print("-" * 60)
for t in THRESHOLDS:
    c = summary["probe_4seed"][f"t={t}"]
    opt_s = f"{c['opt_mean']:.4f}" if c["opt_mean"] is not None else "   —   "
    std_s = f"±{c['cov_std']:.3f}" if c["cov_std"] is not None else ""
    print(f"  probe-4seed{std_s:<11}  {t:.2f}  {c['cov_mean'] or 0:>8.4f}  {c['chv_mean'] or 0:>8.4f}  {opt_s:>9}")
if noenc_data:
    for t in THRESHOLDS:
        c = summary["no_encoder"][f"t={t}"]
        opt_s = f"{c['opt_mean']:.4f}" if c["opt_mean"] is not None else "   —   "
        print(f"  no-encoder           {t:.2f}  {c['cov_mean'] or 0:>8.4f}  {c['chv_mean'] or 0:>8.4f}  {opt_s:>9}")
p = summary["seed_policy"]
print(f"\n  seed (policy only)         cov={p['cov_mean']:.4f}  |S|/n={p['chv_mean']:.4f}")
PYEOF
    echo ""
    echo "All done. Results in $OUTDIR/setpred_extreme_ood_*.json"
fi
