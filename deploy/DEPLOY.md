# Deploying the policy-seed replication on a rented GPU (vast.ai)

Goal: train 3 policy seeds under the **published recipe** so the paper can drop
its single-run caveat, then run the probe pipeline on each.

## 1. What to rent

| requirement | value | why |
|---|---|---|
| **VRAM** | **8 GB is enough**; 12 GB comfortable | The released run's *effective* batch was 10 (see the box below), which peaks at ~4.6 GiB. You do **not** need a 24 GB card. |
| **GPU model** | **RTX 3090 / 4090 / 3080** | sm_86 / sm_89 are covered by the torch 2.4.1 + cu118 build that `scikit-geometry 0.1.2` is verified against here. |
| **NOT the RTX 5090** | — | sm_120 needs torch >= 2.7 / cu128, forcing a different python + skgeom build and full revalidation. The earlier 5090 template is the wrong choice for this repo. |
| **CPU** | high single-thread clock, >= 8 cores | The decode is a sequential Python loop; profiling put the workload at CPU/latency-bound with only ~7% real GPU compute. A faster GPU buys much less than a faster core. |
| **Disk** | >= 20 GB | ~2.1 GB payload plus checkpoints. |
| **Reliability** | on-demand, not interruptible | Runs last days. Resume exists (see §5) but evictions still cost time. |

An A100/H100 is **wasted money** on this workload — it is not GPU-bound.

> **The recipe's `batch_size: 32` is not the batch it trained with.**
> `BucketBatchSampler` chunks length-sorted indices into buckets of
> `bucket_size` and only then splits each bucket by `batch_size`, so a
> `batch_size` above `bucket_size` is inert. When the released policy was
> trained, `bucket_size` was hardcoded to 10 (`git show 8592243^:po_agp.py`),
> so its effective batch was **10**. `run_policy_seeds.sh` therefore leaves
> `AGNET_BUCKET_SIZE` at 10; raising it would *depart* from the published run,
> not match it. The runner prints
> `batch: 32 nominal / bucket 10 -> effective 10` at startup — check that line.
>
> This is also why the whole job fits on a modest card. Renting is about
> running seeds **in parallel** and not tying up a local workstation for a
> week — not about VRAM.

## 2. The one hard dependency

`utils.py` does an unconditional `import skgeom` (scikit-geometry), a CGAL
binding shipped **only as conda-forge binaries**. There are no PyPI wheels, so
a pip-based image cannot run this code at all, regardless of its CUDA version.
`deploy/setup_vast.sh` therefore installs miniconda and builds the env from
`deploy/environment.yml`.

**The disc-vis cache is precomputed locally for the same reason**, and to
avoid re-paying its CPU cost on a metered box. `data/disc_vis_cache.pkl` now
has **full train+dev coverage** (12132 entries, 0 missing across 8867 train +
1224 dev polygons; built/verified 2026-07-24 with
`tools/build_disc_vis_cache.py`, ~1.5 min on 20 workers). Ship the finished
`.pkl` and the rented box never touches CGAL for this step at all. If it is
ever regenerated, `tools/build_disc_vis_cache.py` is a safe top-up (loads the
existing cache, builds only what's missing, saves back) — but **diff its
output against a backup before trusting it**; a wrong `AGNET_DISC_VIS_CACHE_SIZE`
silently evicts already-good entries rather than erroring (see
[[agnet-disc-vis-cache-eviction-trap]] in memory).

## 3. Getting code + data onto the box

~2.2 GB total: dataset 1.1 GB, disc-vis cache 919 MB, code 174 MB.

**Always exclude `.env`.** It holds the LOCAL `DATASET_PATH` (wrong on the box,
so runs fail) and `VAST_AI_API_KEY` / `VAST_SSH_KEY` — syncing it both breaks
the run and copies your credentials onto a rented machine. Write the box's own
one-line `.env` instead:
`printf 'DATASET_PATH=/workspace/dataset/AGPIL/\n' > /workspace/AGNet/.env`
(bit us 2026-07-25: a code sync clobbered the remote `.env` and pushed the API
key; caught only by reading the file back afterwards.)

```bash
# from the local repo root; $VAST is e.g. root@ssh5.vast.ai -p 12345
rsync -avz --exclude .git --exclude .env --exclude checkpoints --exclude results \
      --exclude data --exclude logs --exclude __pycache__ \
      ./ "$VAST:~/AGNet/"

# the released policy checkpoint (3.9 MB). REQUIRED: the correctness gate in
# setup_vast.sh decodes with it to prove the optimised path still matches the
# baseline, and the rsync above excludes checkpoints/.
rsync -avzR checkpoints/v3/po_agp/lstm_bt/po_agp_best_greedy.pt "$VAST:~/AGNet/"

# dataset (AGPIL root containing train/ dev/ test/)
rsync -avz /home/dseverdi/Radno/MLAG/dataset/AGPIL "$VAST:~/dataset/"

# the prebuilt disc-vis cache is worth transferring: without it the first
# epoch rebuilds every per-polygon visibility matrix from CGAL
rsync -avz --progress data/disc_vis_cache.pkl "$VAST:~/AGNet/data/"

# canonical split references, required by the phase-2b carve
rsync -avz data/ls_trajectories_dev_tune.pkl \
           data/ls_trajectories_dev_test_clean.pkl "$VAST:~/AGNet/data/"
```

Then on the box, create `.env`:

```bash
echo "DATASET_PATH=$HOME/dataset/AGPIL/" > ~/AGNet/.env
```

## 4. Provision and verify

```bash
cd ~/AGNet && bash deploy/setup_vast.sh
```

This installs miniconda + the env, then runs three gates in order: a real CUDA
matmul (catches an arch mismatch that otherwise only shows up at the first
kernel launch), a data-presence check, and `tools/smoke_fast_decode.py`.

That last gate matters: the decode and reward were optimised (~4.4x) and are
only trustworthy while they stay bit-identical to the baseline. **Re-run it on
every new machine.** If it ever fails, `AGNET_LEGACY_DECODE=1` and
`AGNET_LEGACY_REWARD=1` restore the original code paths independently.

Then, before committing days of compute:

```bash
bash run_policy_seeds.sh --smoke                              # minutes
PO_EPOCHS_OVERRIDE=1 bash run_policy_seeds.sh 11               # one real epoch
```

`setup_vast.sh` prints the exact `PYTHON=...` path for this box at the end of
its run — some images put conda envs somewhere other than
`~/miniconda3/envs/<name>` (e.g. this vast.ai base image's `.condarc` sets
`envs_dirs: [/venv]`, landing the env at `/venv/MLAG`), so don't assume the
path; copy it from that output.

Multiply the measured epoch time by 200 for the true per-seed ETA. Do not trust
the estimate in §6 over a measurement on the actual box.

## 5. Running the seeds

```bash
tmux new -s seed11 -d "PYTHON=/path/from/setup_vast.sh/output \
    bash run_policy_seeds.sh 11 2>&1 | tee logs/seed11.log"
```

One seed per GPU. Concurrent seeds on a *single* GPU contend for memory and
finish slower than running them sequentially.

**Do not run two seeds concurrently on one filesystem.** `po_agp.py` writes its
final report to a shared, non-per-seed path
(`results/v3/po_agp/po_agp_report.json`), so parallel seeds overwrite each
other's copy. The artifacts the claim actually rests on
(`results/policy_seeds/pseed<N>_{full,noenc}.json` and the manifests) *are*
per-seed and unaffected — but separate instances, or one seed at a time, avoid
the ambiguity entirely.

Every phase is skipped if its output exists, and policy training resumes from
the newest periodic checkpoint, so an interruption costs at most the last few
epochs rather than the whole seed.

**Restart trap:** `po_agp_best_greedy.pt` is written *during* training
(best-so-far), so a seed that died part-way leaves a checkpoint that makes
phase 1 SKIP on re-run, silently using a half-trained policy. Delete that
seed's checkpoint dir if you want a clean restart rather than a resume.

## 6. Time and cost estimate

Because the effective batch is 10, this is a **measured** figure rather than an
extrapolation: 1.25 s/step on an RTX 2080 SUPER (post-optimisation, warm
disc-vis cache).

| | value |
|---|---|
| steps/epoch | 8000 / 10 = 800 |
| epoch | ~17 min |
| 200 epochs (1 seed) | **~56 h, i.e. ~2.3 days** |

A 3090 will be somewhat but not dramatically faster — the step is CPU-bound, so
the card is not the limiter.

Add per seed, on top of training:

* **`po_agp.py`'s own final evaluation.** After the last epoch it calls
  `prewarm_vis_cache` on the validation set, which builds **exact CGAL**
  visibility — not the disc-vis cache. Because `small_val` is capped at
  `train_size` (8000) and dev has only 1224 polygons, this runs over the
  *entire* dev split. In the 2-epoch smoke it took longer than the training
  itself. Budget tens of minutes.
* LS trajectories ~77 min, two probes at 60 epochs, and the probe evals.

Call the non-training overhead ~3-4 h per seed.

So **~2.5 days per seed** all-in, ~7.5 days for three sequentially on one GPU,
or **~2.5 days wall-clock on three instances in parallel** — which is the main
reason to rent at all.

At RTX 3090 pricing (~$0.20-0.35/h): **~$12-21 per seed, ~$36-63 for three**,
plus a few dollars of storage. Cheap relative to the risk of shipping a
single-seed claim.

## 7. Retrieving results

Only small files matter:

```bash
rsync -avz "$VAST:~/AGNet/results/policy_seeds/" results/policy_seeds/
rsync -avz "$VAST:~/AGNet/logs/policy_seed*.log" logs/
```

Checkpoints can stay on the box unless you want to re-evaluate. The number that
decides the claim is `_full` vs `_noenc` per seed.
