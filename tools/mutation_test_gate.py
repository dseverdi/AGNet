"""Mutation test for the consistency gate: does it actually catch what went wrong?

A checker that passes proves nothing unless it can fail. This re-introduces each defect
that genuinely occurred in this project and asserts the corresponding check fires, then
restores the tree. Every mutation is applied to a real file and reverted in a finally
block; nothing is committed.

The claim this supports is not "I checked carefully" but "the gate demonstrably catches
every failure mode that actually occurred here" -- which is falsifiable, and fails loudly
if someone later weakens a check.

Usage:  python tools/mutation_test_gate.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
GATE = REPO / "tools" / "verify_paper_consistency.py"


def run_gate() -> tuple[int, str]:
    p = subprocess.run([PY, str(GATE)], capture_output=True, text=True, cwd=REPO)
    return p.returncode, p.stdout + p.stderr


class _AnchorOK(Exception):
    """Raised by edit() under DRY_RUN once an anchor has been validated, to
    short-circuit the rest of the mutation before it touches anything."""


DRY_RUN = False


def edit(path: Path, old: str, new: str) -> None:
    t = path.read_text()
    if old not in t:
        raise RuntimeError(f"mutation anchor not found in {path.name}: {old[:60]!r}")
    if DRY_RUN:
        raise _AnchorOK
    path.write_text(t.replace(old, new, 1))


def preflight() -> list[str]:
    """Validate every anchor BEFORE mutating anything.

    Editing the paper silently invalidates anchors -- it has happened three times
    (`$302$ of $362$`, `$312$ under a $45^\\circ$`, and an editor-group line in
    claims.py). Previously the first stale anchor raised mid-run and killed the whole
    suite, so one reworded sentence made every other mutation untested while still
    looking like a hard failure. Validating up front turns that into a precise list.
    Anchors are checked, never repaired: a stale one means the prose moved and the
    mutation must be repointed deliberately.
    """
    global DRY_RUN
    DRY_RUN = True
    stale = []
    try:
        for label, mutate, _ in MUTATIONS:
            try:
                mutate()
            except _AnchorOK:
                pass                      # anchor found, nothing written
            except RuntimeError as e:
                stale.append(f"{label}: {e}")
            except Exception as e:        # noqa: BLE001 - non-edit mutation, report it
                stale.append(f"{label}: preflight raised {type(e).__name__}: {e}")
    finally:
        DRY_RUN = False
    return stale


# Each entry: label, how to mutate, which check must fire.
MUTATIONS = [
    ("stale p-value restored in prose",
     lambda: edit(REPO / "paper/paper.tex", r"$p < 10^{-74}$", r"$p < 1.2 \times 10^{-75}$"),
     ["[1]", "[8]"]),
    ("a table numeral edited",
     lambda: edit(REPO / "paper/tables/tab_headline.tex", "1.6208", "1.6308"),
     ["[1]"]),
    # Anchor on a numeral that lives ONLY in prose (no table carries it), so this
    # exercises the tex side of check 1 rather than a table's. The previous anchor
    # ("$302$ of $362$") went stale when tab_dist_shift was refactored in dd53f14,
    # which silently made the whole suite unrunnable -- if this raises "anchor not
    # found", repoint it, do not delete the mutation.
    ("a prose numeral edited",
     lambda: edit(REPO / "paper/paper.tex", r"$91.9\%$ of pairs", r"$92.9\%$ of pairs"),
     ["[1]", "[2]"]),
    # The invariance item quotes a range over five transforms instead of listing each.
    # Widening it must fail, or the three interior transforms go unverified.
    ("a quoted range no longer bounds its measurements",
     lambda: edit(REPO / "paper/paper.tex", r"between $205$ and $312$",
                  r"between $240$ and $312$"),
     ["[1]"]),
    ("retracted phrase reinserted",
     lambda: edit(REPO / "paper/paper.tex",
                  r"\subsection{Reading the encoder shortens the tail}",
                  r"\subsection{Reading the encoder closes the tail}"),
     ["[8]"]),
    ("verbal claim inflated beyond its interval",
     lambda: edit(REPO / "tools/claims.py",
                  'interval=(1.7, 2.0),\n        phrase="nearly twice',
                  'interval=(2.5, 3.0),\n        phrase="nearly twice'),
     ["[1]"]),
    ("aggregate file diverges from raw data",
     lambda: edit(REPO / "paper/data/significance_ood.json",
                  '"probe_failures": 26', '"probe_failures": 3'),
     ["[6]"]),
    ("a source archived while still cited",
     lambda: (REPO / "results/ARCHIVE").mkdir(exist_ok=True) or
             shutil.move(str(REPO / "results/probe_timing.json"),
                         str(REPO / "results/ARCHIVE/probe_timing.json")),
     ["[5]"]),
    ("figure older than its data",
     lambda: (REPO / "paper/data/setpred_iter_sweep.json").touch(),
     ["[7]"]),
    ("an input changed without rebuilding the pdf",
     lambda: edit(REPO / "paper/tables/tab_probe_ladder.tex", "0.923", "0.933"),
     ["[9]"]),
    # A verbal claim can be deleted from the paper while its interval still passes,
    # leaving the gate verifying a sentence no reader sees. Seven claims were orphaned
    # this way before check 1 asserted phrase presence.
    ("a verbal claim's phrase deleted from the paper",
     lambda: edit(REPO / "paper/paper.tex",
                  r"about $1.45\times$ the full probe's guards",
                  r"appreciably more of the full probe's guards"),
     ["[1]"]),
    ("claims in one group drawn from different populations",
     lambda: edit(REPO / "tools/claims.py",
                  'n=a["n_polygons"], run="editor_t08_test362", group="editor",\n'
                  '        value=(1 - a["ed_size_mean"]',
                  'n=5, run="editor_smoke", group="editor",\n'
                  '        value=(1 - a["ed_size_mean"]'),
     ["[4]"]),
]


def main() -> None:
    backup = Path(tempfile.mkdtemp(prefix="gate_mutation_"))
    touched = [
        "paper/paper.tex", "paper/tables/tab_headline.tex",
        "paper/tables/tab_probe_ladder.tex", "tools/claims.py",
        "paper/data/significance_ood.json", "results/probe_timing.json",
        "paper/data/setpred_iter_sweep.json",
    ]
    # every file any mutation touches must be listed above, or the finally block
    # cannot restore it -- this bit once, leaving a mutated table behind
    stamps = {}
    for rel in touched:
        src = REPO / rel
        dst = backup / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        stamps[rel] = (src.stat().st_atime, src.stat().st_mtime)
    pdf_stamp = ((REPO / "paper/paper.pdf").stat().st_atime,
                 (REPO / "paper/paper.pdf").stat().st_mtime)

    # Anchors first: a stale one is a repointing job, not a gate failure, and finding
    # them all at once beats discovering them one crash at a time. Preflight runs after
    # the backup because the two non-edit mutations (file move, touch) genuinely execute.
    def _restore() -> None:
        import os
        for rel in touched:
            shutil.copy2(backup / rel, REPO / rel)
            os.utime(REPO / rel, stamps[rel])
        os.utime(REPO / "paper/paper.pdf", pdf_stamp)

    stale = preflight()
    _restore()
    if stale:
        print(f"{len(stale)} stale mutation anchor(s) -- the prose moved; repoint them:\n")
        for s in stale:
            print(f"  - {s}")
        print("\nNot a gate failure. Do NOT delete the mutation; find where the text went.")
        shutil.rmtree(backup, ignore_errors=True)
        sys.exit(3)
    print(f"preflight: all {len(MUTATIONS)} anchors resolve")

    rc, _ = run_gate()
    if rc != 0:
        print("baseline gate does not pass; fix that before mutation testing")
        shutil.rmtree(backup, ignore_errors=True)
        sys.exit(2)
    print("baseline: gate passes\n")

    results = []
    try:
        for label, mutate, expect in MUTATIONS:
            try:
                mutate()
                rc, out = run_gate()
                fired = sorted({m for m in re.findall(r"\[(\d)\]",
                                "\n".join(l for l in out.splitlines()
                                          if l.strip().startswith("[BAD]")))})
                fired = [f"[{f}]" for f in fired]
                caught = rc != 0 and any(e in fired for e in expect)
                results.append((label, caught, expect, fired))
            finally:
                for rel in touched:
                    shutil.copy2(backup / rel, REPO / rel)
                    import os
                    os.utime(REPO / rel, stamps[rel])
                import os
                os.utime(REPO / "paper/paper.pdf", pdf_stamp)
    finally:
        shutil.rmtree(backup, ignore_errors=True)

    print(f"  {'mutation':<52} {'caught':>7}  expected / fired")
    for label, caught, expect, fired in results:
        print(f"  {label:<52} {'YES' if caught else 'NO ':>7}  "
              f"{','.join(expect)} / {','.join(fired) or '-'}")
    n_ok = sum(1 for _, c, _, _ in results if c)
    print(f"\n{n_ok}/{len(results)} mutations caught")

    rc, _ = run_gate()
    print("tree restored, gate passes again" if rc == 0
          else "!! TREE NOT RESTORED -- gate now failing")
    sys.exit(0 if n_ok == len(results) and rc == 0 else 1)


if __name__ == "__main__":
    main()
