"""Separate the data the paper depends on from everything else, and archive the rest.

The keep-set is the union of three things, because a generator-read audit alone is
not sufficient:

  1. GENERATOR READS -- ground truth from tools/audit_paper_data_deps.py, which
     patches builtins.open AND the pathlib.Path read methods and records every
     file the table/figure generators actually open.

  2. HAND-CURATED TABLE SOURCES -- tab_decode_search, tab_discvis_quality and
     tab_runtime are written by hand from a JSON that no generator opens, so the
     audit cannot see them. Their provenance comment names the file.

  3. PROSE-QUOTED SOURCES -- numbers cited directly in the manuscript with no
     table or figure in between: the invariance measurement, the linear-probe
     ROC-AUC, and the canonical-start experiment. Dropping these would leave
     published numbers with no traceable origin.

Archiving is a MOVE, never a delete, into paper/data/ARCHIVE/ and results/ARCHIVE/
with the original relative path preserved, so anything mistakenly archived can be
put back. Known-leaky artifacts are archived too: they are already refused by the
deny-list in build_tables.py, and moving them makes that belt-and-braces.

Usage:
  python tools/isolate_paper_data.py --dry-run     # report only (default)
  python tools/isolate_paper_data.py --apply       # perform the moves
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEPS = REPO / "results" / "paper_data_deps.json"

# (2) Hand-curated tables: the JSON is named in the table's header comment but
# never opened programmatically.
HAND_CURATED = [
    "paper/data/decode_search.json",        # tab_decode_search
    "results/discvis_greedy_timing.json",   # tab_discvis_quality + tab_runtime
    "results/classical_timing_lazy.json",   # tab_runtime (the lazy-greedy timings)
    # tab_runtime's learned CPU/GPU columns come from here (11.5 -> 12, 33.4 -> 33,
    # 89.0 -> 89 ms). Its header comment names the file; no generator opens it, so
    # the read-audit could not see it and a first pass archived it by mistake.
    "results/probe_timing.json",
]

# Raw provenance for a published aggregate. Nothing reads these, but each is the
# only way to re-derive a number the manuscript prints, so archiving them would
# leave a published value with no checkable origin.
PROVENANCE = [
    # tab_probe_ladder prints per-rung mean +- std; probe_ladder.json holds the
    # aggregate, these hold the per-polygon evaluations behind it.
    *[f"paper/data/dist_dev_test_ladder_{a}_seed{s}.json"
      for a in ("attn1", "mlp") for s in (1234, 11, 22, 33)],
    # Which checkpoint and which trajectory pickle produced each policy-seed
    # matched-budget sweep. The original 16 sweeps have no such record, which is
    # why these were written.
    *[f"paper/data/matched_budget_pseeds/{a}_pseed{s}.prov.json"
      for a in ("full", "noenc") for s in (11, 22, 33)],
    # Per-seed run manifests (timings, paths) for the policy-seed replication.
    *[f"results/policy_seeds/manifest_pseed{s}.json" for s in (11, 22, 33)],
]

# (3) Quoted in prose with no intervening artifact.
PROSE_QUOTED = [
    "paper/data/invariance_test.json",        # the rotation/reflection limitation
    "paper/data/encoder_linear_probe.json",   # ROC-AUC 0.843 +- 0.009
    "results/canonical_start.json",           # canonical-start limitation
    "results/canonical_rule_selection.json",  # its off-split rule selection
    # McNemar p-values and Wilson intervals in sec:res-ood. Written by
    # paper/scripts/compute_significance.py, which the audit does not run because
    # it both reads and writes paper/data. Its four inputs are listed below --
    # they also carry the per-seed spread behind tab_ood, which build_tables
    # reads only in pre-aggregated form via multi_seed_summary.json, so without
    # them the published +- would have no re-checkable origin.
    "paper/data/significance_ood.json",
    "paper/data/dist_test_OOD_seed11.json",
    "paper/data/dist_test_OOD_seed22.json",
    "paper/data/dist_test_OOD_seed33.json",
    "paper/data/dist_test_OOD_noenc_seed11.json",
    "paper/data/dist_test_OOD_noenc_seed22.json",
    "paper/data/dist_test_OOD_noenc_seed33.json",
    # Footnote to the disc-vis limitation: Spearman 0.95 (guards) vs 0.88 (n),
    # partial 0.78 vs 0.29, over 40 polygons spanning n in [20, 1750].
    "results/discvis_gap_correlation.json",
    # Same limitation: coverage bias +0.0003, optimistic on 52% of rollouts,
    # 91.9% pairwise-ordering agreement over 1600 rescored rollouts, and the
    # 94-against-22 gate-crossing asymmetry.
    "results/reward_estimator_agreement.json",
    # sec:res-iteration, the learned-editor result. eval_t08 is the
    # coverage-preserving end (13% guard cut, 0.27 recovery); st0p9_cgnone is the
    # aggressive end (26% cut, mean coverage 0.969 -> 0.925). Both are N=300; the
    # manuscript previously paired the 0.27 with a 24% cut taken from an N=5 smoke
    # run, which is why both endpoints are now pinned here.
    "results/eval_t08.json",
    "results/editor_sweep/v2/st0p9_cgnone.json",
    # The supervised-pointer collapse quoted in the same section (mean coverage
    # 0.496, median approximation ratio 3.10).
    "results/v2/sl_agp_evaluation.json",
]

# Directories to sweep. Everything under them that is not in the keep-set moves.
SWEEP = ["paper/data", "results"]

# Never move these, whatever the audit says.
PROTECT_NAMES = {"paper_data_deps.json"}
PROTECT_DIRS = {"ARCHIVE"}


def keep_set() -> tuple[set[str], dict[str, list[str]]]:
    if not DEPS.exists():
        raise SystemExit(
            f"{DEPS} missing -- run tools/audit_paper_data_deps.py first")
    audit = json.loads(DEPS.read_text())
    if audit.get("failures"):
        raise SystemExit(
            f"audit recorded failures, dependency list is incomplete: "
            f"{audit['failures']}")
    gen = set(audit["union"])
    keep = set(gen) | set(HAND_CURATED) | set(PROSE_QUOTED) | set(PROVENANCE)
    return keep, {"generator_reads": sorted(gen),
                  "hand_curated": HAND_CURATED,
                  "prose_quoted": PROSE_QUOTED,
                  "provenance": PROVENANCE}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="perform the moves (default is a dry run)")
    args = ap.parse_args()

    keep, prov = keep_set()
    missing = [f for f in keep if not (REPO / f).exists()]

    moves: list[tuple[str, str]] = []
    for root in SWEEP:
        for p in sorted((REPO / root).rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(REPO).as_posix()
            parts = set(p.relative_to(REPO).parts)
            if rel in keep or p.name in PROTECT_NAMES or parts & PROTECT_DIRS:
                continue
            dest = f"{root}/ARCHIVE/{p.relative_to(REPO / root).as_posix()}"
            moves.append((rel, dest))

    kept_bytes = sum((REPO / f).stat().st_size for f in keep if (REPO / f).exists())
    moved_bytes = sum((REPO / s).stat().st_size for s, _ in moves)

    print(f"KEEP   {len(keep)} files, {kept_bytes/1e6:.1f} MB")
    print(f"         generator reads {len(prov['generator_reads'])}"
          f" + hand-curated {len(prov['hand_curated'])}"
          f" + prose-quoted {len(prov['prose_quoted'])}")
    print(f"ARCHIVE {len(moves)} files, {moved_bytes/1e6:.1f} MB")
    if missing:
        print(f"\n!! {len(missing)} keep-set files do not exist:")
        for f in sorted(missing):
            print(f"     {f}")

    print("\nsample of what would move:")
    for s, d in moves[:12]:
        print(f"    {s}")
    if len(moves) > 12:
        print(f"    ... and {len(moves)-12} more")

    if not args.apply:
        print("\nDRY RUN -- nothing moved. Re-run with --apply.")
        return

    for s, d in moves:
        dst = REPO / d
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(REPO / s), str(dst))
    manifest = REPO / "results" / "paper_data_isolation.json"
    manifest.write_text(json.dumps(
        {"keep": sorted(keep), "provenance": prov,
         "archived": [{"from": s, "to": d} for s, d in moves]}, indent=1))
    print(f"\nmoved {len(moves)} files. manifest: {manifest.relative_to(REPO)}")


if __name__ == "__main__":
    main()
