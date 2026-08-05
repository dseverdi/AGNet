#!/bin/bash
# sync_vast_checkpoints.sh -- pull policy-seed checkpoints off the vast.ai box.
#
# This instance has NO persistent storage (workspace_is_volume: false) --
# recycle/destroy wipes everything, so periodic checkpoints only protect
# against an in-run interruption, not against losing the instance itself.
# Run this on a schedule (cron / manual) while seeds train.
#
# ACTIVE RUNS as of 2026-07-25: seed11 (GPU0) and seed22 (GPU1), both batch 32,
# isolated in /workspace/AGNet_batch32/ (a separate symlinked workspace so
# they can't collide with any batch-10 run of the same seed number elsewhere,
# e.g. locally). The original /workspace/AGNet/checkpoints/.../lstm_bt_seed22
# (collapsed at epoch 5) and lstm_bt_seed33 (healthy but superseded) are DEAD
# -- their processes were killed, nothing writes to them anymore, so syncing
# them just re-fetches the same stale epoch-42 snapshot forever. Update the
# SEEDS/REMOTE_BASE table below if the active runs change again.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# CONNECTION IS PER-INSTANCE AND DRIFTS -- re-check whenever it fails.
#   vastai show instances
#   vastai show instance <ID> --raw | python3 -c "import json,sys; d=json.load(sys.stdin); \
#     print(d['ssh_host'], d['ssh_port'], d['public_ipaddr'], d['ports'].get('22/tcp'))"
# Observed failure modes, both real:
#   * the proxy route can be dead while the direct IP works. On instance
#     45927934 (current) `ssh9.vast.ai:17934` fails with
#     kex_exchange_identification while 142.115.32.84:41664 (the 22/tcp
#     mapping) works -- so prefer the DIRECT ip:22-mapping here.
#   * the CLI's ssh_port can lag reality (on the previous box it reported
#     13624 when 13625 was the working port), and public_ipaddr can change
#     while the container itself never restarts.
# Always confirm empirically before trusting either value.
HOST="root@142.115.32.84"     # instance 45927934, 22/tcp -> 41664
PORT=41664
KEY="$HOME/.ssh/id_ed25519_dseverdi"
RS="ssh -p $PORT -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ServerAliveCountMax=4"

# seed -> remote base dir (workspace root containing checkpoints/ and logs/)
declare -A REMOTE_BASE=(
    [11]="/workspace/AGNet"
    [22]="/workspace/AGNet"
)
DEST_ROOT="results/policy_seeds_vast_backup"
mkdir -p "$DEST_ROOT"

echo "[sync] $(date -u +%FT%TZ) pulling checkpoints + dynamics + logs ..."
for s in "${!REMOTE_BASE[@]}"; do
    base="${REMOTE_BASE[$s]}"
    dest="$DEST_ROOT/lstm_bt_seed${s}"
    mkdir -p "$dest"
    # --delete: without it, a seed number that gets reused for a NEW run (as
    # seed22 and seed33 both were here) leaves the OLD run's checkpoint/jsonl
    # sitting locally forever once REMOTE_BASE points elsewhere -- bit us
    # twice (stale epoch-42 snapshots silently surviving a re-point).
    rsync -avz --delete --partial --append-verify -e "$RS" \
        "$HOST:${base}/checkpoints/v3/po_agp/lstm_bt_seed${s}/" \
        "$dest/"
    rsync -avz --partial --append-verify -e "$RS" \
        "$HOST:${base}/logs/policy_seed${s}*.log" \
        "$DEST_ROOT/" 2>/dev/null || true
done

echo "[sync] done. Local state:"
for s in "${!REMOTE_BASE[@]}"; do
    f="$DEST_ROOT/lstm_bt_seed${s}/training_dynamics.jsonl"
    if [[ -f "$f" ]]; then
        echo "  seed${s}: $(tail -n1 "$f")"
    else
        echo "  seed${s}: no dynamics file yet"
    fi
done
