#!/usr/bin/env python
"""select_best_checkpoint.py -- pick a seed's best policy by EXACT CGAL.

WHY THIS EXISTS
    Every in-training signal we tried picks the wrong epoch:

      raw coverage      -> picks post-collapse epochs (guarding every vertex
                           trivially maxes coverage). Froze seed11 at epoch 17.
      cov - guard_ratio -> picks barely-trained early epochs. On seed22 it
                           selected epoch 4 (|S|/OPT 2.32) over epoch 80
                           (|S|/OPT 1.12) because both had identical
                           cov 0.9927 / gr 0.0129, so the composite saw no
                           improvement -- while real quality kept improving.
      disc-vis approx   -> a 500-sample PROXY. Read ~1.03 for a policy that
                           scored 2.79 on exact CGAL. Worst of the three.

    So selection is done AFTER training, by decoding each retained checkpoint
    and scoring with exact CGAL. That measurement is validated:
    compare_policy_checkpoints.py reproduces the paper's released-policy row
    exactly (cov 0.9689, |S|/n 0.1664, |S|/OPT 1.0885, 293/362) when pointed at
    dev_test.

    TEST-SET LEAK, FIXED 2026-07-27: the first version of this script selected
    AND would then have been reported on the SAME 362-polygon dev_test split --
    selection-on-test, the same class of bug as the already-documented
    checkpoint-selection leak on the probe side. Selection and reporting now
    use DIFFERENT, disjoint splits by default:
        --select-on data/ls_trajectories_dev_tune.pkl   (857 polygons, default)
        --report-on data/ls_trajectories_dev_test_clean.pkl (362, default)
    Never pass the same file to both. If you must (e.g. no tune split exists
    for a dataset), pass --allow-same-split explicitly so it is a deliberate,
    visible choice rather than a silent default.

    Requires the retention ladder (AGNET_KEEP_EVERY, default 20) -- otherwise
    only one periodic checkpoint survives and there is nothing to choose from.

RANKING
    Primary: |S|/OPT ascending (fewer guards relative to the baseline), among
    checkpoints meeting --min-cov. Coverage is a GATE, not a term to trade
    against guard count -- that trade is exactly what the cov-gr composite got
    wrong. Default gate 0.95 sits below the released policy's 0.9689 so a
    released-quality policy always qualifies.

USAGE
    python tools/select_best_checkpoint.py \
        --ckpt-dir checkpoints/v3/po_agp/lstm_bt_seed22
    python tools/select_best_checkpoint.py --ckpt-dir <dir> --apply
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_checkpoints(d: str) -> list[tuple[int | None, str]]:
    """Retained periodic checkpoints + best_greedy/final, as (epoch, path)."""
    out: list[tuple[int | None, str]] = []
    for p in sorted(glob.glob(os.path.join(d, "po_agp_epoch*.pt"))):
        m = re.search(r"epoch(\d+)\.pt$", p)
        out.append((int(m.group(1)) if m else None, p))
    for extra in ("po_agp_best_greedy.pt", "po_agp_best_stoch.pt"):
        p = os.path.join(d, extra)
        if os.path.exists(p):
            out.append((None, p))
    for p in sorted(glob.glob(os.path.join(d, "po_agp_final_epoch*.pt"))):
        out.append((None, p))
    out.sort(key=lambda t: (t[0] is None, t[0] if t[0] is not None else 0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--min-cov", type=float, default=0.95,
                    help="coverage GATE; candidates below this are rejected "
                         "outright rather than traded off against guards")
    ap.add_argument("--apply", action="store_true",
                    help="copy the winner over po_agp_best_greedy.pt "
                         "(the original is saved as *.preselect.bak)")
    ap.add_argument("--select-on", default="data/ls_trajectories_dev_tune.pkl",
                    help="split used to CHOOSE the winner (default: dev_tune, "
                         "857 polygons)")
    ap.add_argument("--report-on", default="data/ls_trajectories_dev_test_clean.pkl",
                    help="split used only to report the winner's headline "
                         "numbers, after selection is already decided "
                         "(default: dev_test, 362 polygons)")
    ap.add_argument("--allow-same-split", action="store_true",
                    help="permit --select-on == --report-on (selection-on-test) "
                         "-- only for datasets with no separate tune split; "
                         "must be passed explicitly, there is no silent default")
    args = ap.parse_args()

    if not os.path.isdir(args.ckpt_dir):
        sys.exit(f"no such dir: {args.ckpt_dir}")
    if (os.path.abspath(args.select_on) == os.path.abspath(args.report_on)
            and not args.allow_same_split):
        sys.exit(
            "[select] --select-on and --report-on are the SAME file "
            f"({args.select_on}). That is selection-on-test -- pass "
            "--allow-same-split if this is deliberate.")

    cands = find_checkpoints(args.ckpt_dir)
    if not cands:
        sys.exit(f"no checkpoints found in {args.ckpt_dir}")

    n_periodic = sum(1 for e, _ in cands if e is not None)
    print(f"[select] {len(cands)} candidates ({n_periodic} periodic) in "
          f"{args.ckpt_dir}")
    if n_periodic <= 1:
        print("[select] WARNING: <=1 periodic checkpoint retained. This run "
              "predates the AGNET_KEEP_EVERY retention ladder, so the best "
              "epoch may simply not exist on disk.")

    def _score(ref_split: str, ckpts: list[tuple[int | None, str]]):
        cmd = [sys.executable, os.path.join(REPO, "tools",
                                            "compare_policy_checkpoints.py"),
               "--ref-test", ref_split]
        for ep, p in ckpts:
            label = f"ep{ep}" if ep is not None else os.path.basename(p)[7:-3]
            cmd += ["--ckpt", f"{label}={p}"]
        return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)

    print(f"[select] SELECTING on {args.select_on} (~15s/checkpoint each, "
          f"~{len(cands)*15//60 + 1} min total) ...\n")
    res = _score(args.select_on, cands)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr[-2000:], file=sys.stderr)
        return res.returncode

    def _parse(stdout: str):
        out = []
        for line in stdout.splitlines():
            m = re.match(
                r"^(\S+)\s+(\d+|None)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)/(\d+)",
                line.strip())
            if m:
                out.append({
                    "label": m.group(1), "epoch": m.group(2),
                    "cov": float(m.group(3)), "chv": float(m.group(4)),
                    "opt": float(m.group(5)),
                    "ge95": int(m.group(6)), "n": int(m.group(7)),
                })
        return out

    rows = _parse(res.stdout)
    if not rows:
        print("[select] could not parse any scored rows", file=sys.stderr)
        return 1

    ok = [r for r in rows if r["cov"] >= args.min_cov]
    print(f"[select] {len(ok)}/{len(rows)} pass the cov >= {args.min_cov} gate "
          f"(selected on {args.select_on})")
    if not ok:
        best = max(rows, key=lambda r: r["cov"])
        print(f"[select] NO candidate meets the gate. Best coverage is "
              f"{best['cov']:.4f} ({best['label']}). This policy is not "
              f"usable; do not silently take the least-bad one.")
        return 2

    best = min(ok, key=lambda r: r["opt"])
    print(f"\n[select] WINNER (on select-on split): {best['label']}  "
          f"cov={best['cov']:.4f}  |S|/n={best['chv']:.4f}  "
          f"|S|/OPT={best['opt']:.4f}  cov>=.95={best['ge95']}/{best['n']}")

    sel = next((p for ep, p in cands
                if (f"ep{ep}" if ep is not None
                    else os.path.basename(p)[7:-3]) == best["label"]), None)
    if sel is None:
        print("[select] could not map winner back to a file", file=sys.stderr)
        return 1
    print(f"[select] file: {sel}")

    # Second pass: report the WINNER ONLY on the held-out reporting split.
    # This is the number that goes in the paper -- never the select-on score.
    if os.path.abspath(args.select_on) != os.path.abspath(args.report_on):
        print(f"\n[select] REPORTING {best['label']} on {args.report_on} "
              f"(held out from selection) ...\n")
        rep_ep = None if best["epoch"] == "None" else int(best["epoch"])
        rep_res = _score(args.report_on, [(rep_ep, sel)])
        print(rep_res.stdout)
        rep_rows = _parse(rep_res.stdout)
        if rep_rows:
            r = rep_rows[0]
            print(f"[select] HEADLINE (report-on): cov={r['cov']:.4f}  "
                  f"|S|/n={r['chv']:.4f}  |S|/OPT={r['opt']:.4f}  "
                  f"cov>=.95={r['ge95']}/{r['n']}")

    if args.apply:
        dst = os.path.join(args.ckpt_dir, "po_agp_best_greedy.pt")
        if os.path.abspath(sel) == os.path.abspath(dst):
            print("[select] winner is already best_greedy.pt; nothing to do.")
        else:
            bak = dst + ".preselect.bak"
            if os.path.exists(dst) and not os.path.exists(bak):
                shutil.copy2(dst, bak)
                print(f"[select] backed up old best_greedy -> {bak}")
            shutil.copy2(sel, dst)
            print(f"[select] APPLIED: {os.path.basename(sel)} -> "
                  f"po_agp_best_greedy.pt")
            print("[select] re-run phases 2-5 to rebuild trajectories/probe "
                  "against this policy.")
    else:
        print("\n[select] dry run. Re-run with --apply to install the winner "
              "as po_agp_best_greedy.pt.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
