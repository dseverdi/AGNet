#!/usr/bin/env bash
# setup_vast.sh -- provision a rented box (vast.ai etc.) for the AGNet
# policy-seed replication, then verify it BEFORE committing to multi-day runs.
#
# Run ON the rented instance, from the repo root, after the code and data are
# in place (see deploy/DEPLOY.md for the rsync commands).
#
#   bash deploy/setup_vast.sh
#
# It is idempotent: re-running skips the conda install and env creation if they
# already exist.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME="${ENV_NAME:-MLAG}"

echo "=============================================================="
echo " AGNet remote setup"
echo "=============================================================="

# --- 1. host GPU sanity -----------------------------------------------------
# Do this first: it is the cheapest check and the one most likely to reveal a
# broken rental (a box whose driver is wedged looks fine until torch loads).
echo "--- host GPU ---"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader || {
    echo "[fatal] nvidia-smi failed; this box has no usable GPU." >&2; exit 1; }

vram_mib="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
if (( vram_mib < 7000 )); then
    echo ""
    echo "[WARNING] Only ${vram_mib} MiB VRAM."
    echo "  The replication trains at an EFFECTIVE batch of 10 (the recipe's"
    echo "  batch_size 32 is inert against bucket_size 10 -- see DEPLOY.md),"
    echo "  which peaks around 4.6 GiB. Below ~7 GiB you will likely OOM."
    echo ""
fi

# --- 2. miniconda -----------------------------------------------------------
# scikit-geometry is conda-only (see deploy/environment.yml), so a pip-based
# image is not enough no matter what CUDA it ships.
if [[ ! -x "$CONDA_DIR/bin/conda" ]]; then
    echo "--- installing miniconda into $CONDA_DIR ---"
    tmp_installer="$(mktemp /tmp/miniconda.XXXXXX.sh)"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o "$tmp_installer"
    bash "$tmp_installer" -b -p "$CONDA_DIR"
    rm -f "$tmp_installer"
else
    echo "--- miniconda already present at $CONDA_DIR ---"
fi
export PATH="$CONDA_DIR/bin:$PATH"

# --- 3. environment ---------------------------------------------------------
if ! "$CONDA_DIR/bin/conda" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "--- creating env '$ENV_NAME' (this takes several minutes) ---"
    # libmamba resolves this env far faster than the classic solver.
    "$CONDA_DIR/bin/conda" install -y -n base conda-libmamba-solver >/dev/null 2>&1 || true
    "$CONDA_DIR/bin/conda" env create -f deploy/environment.yml -n "$ENV_NAME" \
        --solver=libmamba || \
    "$CONDA_DIR/bin/conda" env create -f deploy/environment.yml -n "$ENV_NAME"
else
    echo "--- env '$ENV_NAME' already exists ---"
fi

PYTHON="$CONDA_DIR/envs/$ENV_NAME/bin/python"
[[ -x "$PYTHON" ]] || { echo "[fatal] $PYTHON missing after env create" >&2; exit 1; }

# --- 4. verify the two imports that actually gate the run -------------------
echo "--- verifying interpreter ---"
"$PYTHON" - <<'PY'
import sys
print(f"  python {sys.version.split()[0]}")
import torch
print(f"  torch  {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device {torch.cuda.get_device_name(0)}  "
          f"cap sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
    # A card newer than the wheel's arch list silently falls back or errors on
    # first kernel launch, so force one real allocation + matmul here.
    x = torch.randn(2048, 2048, device="cuda")
    assert float((x @ x).sum()) == float((x @ x).sum())
    print("  cuda matmul OK")
else:
    print("  [FATAL] torch cannot see a GPU")
    sys.exit(1)
import skgeom
print(f"  skgeom OK ({skgeom.__file__.rsplit('/', 3)[-3]})")
PY

# --- 5. data presence -------------------------------------------------------
echo "--- data ---"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi
: "${DATASET_PATH:?DATASET_PATH not set (create .env with DATASET_PATH=/path/to/AGPIL)}"
for split in train dev; do
    n=$(ls "$DATASET_PATH/$split"/*.pol 2>/dev/null | wc -l)
    printf "  %-6s %s .pol files\n" "$split" "$n"
    (( n > 0 )) || { echo "[fatal] no polygons in $DATASET_PATH/$split" >&2; exit 1; }
done
if [[ -f data/disc_vis_cache.pkl ]]; then
    echo "  disc_vis_cache.pkl $(du -h data/disc_vis_cache.pkl | cut -f1)"
else
    echo "  [WARNING] data/disc_vis_cache.pkl missing -- it will be rebuilt on"
    echo "            the fly, which makes the first epoch far slower."
fi
for f in data/ls_trajectories_dev_tune.pkl data/ls_trajectories_dev_test_clean.pkl; do
    [[ -f "$f" ]] || echo "  [WARNING] $f missing -- phase 2b carve will fail."
done

# --- 6. correctness gate ----------------------------------------------------
# The optimised decode/reward must be re-verified on every new machine: it is
# the one thing that could silently change results, and a wrong policy trained
# for two days is far more expensive than this two-minute check.
echo "--- correctness gate (tools/smoke_fast_decode.py) ---"
GATE_CKPT="checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt"
if [[ ! -f "$GATE_CKPT" ]]; then
    echo "[fatal] $GATE_CKPT missing -- the gate decodes with the released" >&2
    echo "        policy to prove the optimised path matches the baseline." >&2
    echo "        The usual repo rsync excludes checkpoints/; send it with:" >&2
    echo "          rsync -avzR $GATE_CKPT \$VAST:~/AGNet/" >&2
    exit 1
fi
if DATASET_PATH="$DATASET_PATH" "$PYTHON" tools/smoke_fast_decode.py; then
    echo "  gate PASSED"
else
    echo "[fatal] smoke_fast_decode.py FAILED -- do not train on this box." >&2
    echo "        Fall back with AGNET_LEGACY_DECODE=1 AGNET_LEGACY_REWARD=1." >&2
    exit 1
fi

cat <<EOF

==============================================================
 Setup complete.

 Next, in this order:

 1. Pipeline smoke (minutes, results NOT usable):
      PYTHON=$PYTHON bash run_policy_seeds.sh --smoke

 2. Measure ONE real epoch to get a true ETA before committing:
      PYTHON=$PYTHON PO_EPOCHS_OVERRIDE=1 bash run_policy_seeds.sh 11
    Read the per-epoch seconds off the log, multiply by 200.

 3. Full seeds (use tmux; these run for days):
      tmux new -s seed11 -d "PYTHON=$PYTHON bash run_policy_seeds.sh 11 2>&1 | tee logs/seed11.log"

 Interruptions are safe: every phase resumes from the newest periodic
 checkpoint, so an eviction costs at most the last few epochs.
==============================================================
EOF
