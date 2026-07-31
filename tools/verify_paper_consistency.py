"""The consistency gate. Nine checks, fails closed, non-zero exit on any failure.

Each check exists because a specific defect got through without it. The mapping is in the
docstrings below, so the gate is justified by observed failures rather than invented.

Supersedes tools/verify_paper_numbers.py, whose 40 spot-checks are subsumed by check 1.

Usage:  python tools/verify_paper_consistency.py [-v]
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claims as C  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TEX = REPO / "paper" / "paper.tex"
PDF = REPO / "paper" / "paper.pdf"
TABDIR = REPO / "paper" / "tables"

PASS: list[str] = []
FAIL: list[str] = []


def ok(m):
    PASS.append(m)


def bad(m):
    FAIL.append(m)


def tex_body() -> str:
    return "\n".join(l for l in TEX.read_text().split("\n")
                     if not l.strip().startswith("%"))


def denumber(t: str) -> str:
    """Normalise LaTeX numeric constructs so token extraction sees whole numbers.

    Without this, `12{,}424` yields the fragments 12 and 424, `10^{-74}` yields 10
    and 74, and `45^\circ` yields 45 -- fragments that can never be registered and
    so would pollute the unclassified list indefinitely.
    """
    t = t.replace("{,}", "")                          # LaTeX thousands separator
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)           # literal 464,001 -> 464001
    t = re.sub(r"\\figincl\[[^\]]*\]", " ", t)        # \linewidth layout arguments
    t = re.sub(r"\^\{?-?\d+\}?", " ", t)              # exponents, degree powers
    t = t.replace(r"\circ", " ")
    return t


def used_tables() -> list[str]:
    return sorted(set(re.findall(r"tables/(tab_[a-z_]*\.tex)", TEX.read_text())))


# ---------------------------------------------------------------------- 1
def check_registry(cl):
    """Every registered claim recomputes and its typeset form appears in its target.
    Catches: leak-era 2.8x/2.6x in the Discussion; the stale p-value."""
    body = tex_body()
    for c in cl:
        hay = body if c.target == "paper.tex" else (TABDIR / c.target).read_text()
        if c.typeset is not None:
            # compare after stripping LaTeX presentation, so a bolded or
            # \quad-indented cell is not read as a mismatch
            found = c.typeset in hay or C.norm(c.typeset) in C.norm(hay)
            (ok if found else bad)(f"[1] {c.id}: '{c.typeset}' in {c.target}")
        elif c.interval is not None:
            lo, hi = c.interval
            good = lo <= c.value <= hi
            (ok if good else bad)(
                f"[1] {c.id}: {c.value:.4g} in [{lo},{hi}] -- \"{c.phrase}\"")
        else:
            bad(f"[1] {c.id}: neither typeset nor interval")


# ---------------------------------------------------------------------- 2
EXEMPT_RE = [
    # architecture / hyperparameters / protocol constants -- design choices, not measurements
    r"^(?:0\.05|1\.0|3\.0|0\.99|0\.95|0\.999|500|8|32|60|200|128|464|364|5\.66|9\.9|1600|12|16|2|3|4|5|6|7|10|20|40|50|100|1224|362|2081|285|840|857|8867|313|198|192|600|700|800|1000|1750|2250|350|70|30|110|114|160|95|0\.16|0\.5|1\.5|0\.20|0\.25|0\.30|0\.35|0\.40|0\.45|0\.50|0\.55|0\.60|0\.65|0\.70|0\.75|0\.80|12424|79|206|885|1\.2|0\.9|0\.1)$",
]
EXEMPT_EXPLICIT = {
    # Structural or not derivable from any stored artifact. Everything measurable is
    # registered instead -- exempting a measurement would make the completeness
    # property hollow, which an audit of this list caught for the C4 drift bounds,
    # the partial correlations and the Wilson upper bounds.
    "0.582", "0.027", "0.843", "0.009",                        # linear probe (registered)
    "0.52", "91.9", "0.0003",                                  # reward estimator (registered)
    "1.15", "0.97",                                            # Transformer trial, dev-time
    "0.9994", "0.1524", "0.9948", "0.1369", "0.8854",          # classical rows (registered)
    "1.0092", "1.0885", "0.9689", "0.1664",
    # structural: seed label, polygon-size row headers, a chosen matched-budget anchor,
    # and the dataset-partition upper bound
    "1234", "2000", "900", "480", "193", "250", "1.95",
    # the source library's own vertex range, a dataset description rather than a
    # measurement of our results, and not derivable from any stored artifact
    "2500",
}


def check_completeness(cl):
    """Every numeral in paper.tex and in the input tables is registered or exempt.
    Fails on anything unclassified -- this is the property that makes 'did I miss
    something?' computable. Catches: whatever has not yet been imagined."""
    # Exact token sets, not substring containment. `"0.029" in registered` was true
    # merely because some other claim's string contained those characters, which let a
    # real measurement hide behind a coincidental match.
    registered = set()
    for c in cl:
        for blob in (c.typeset, c.phrase):
            if blob:
                # normalise the registered side identically to the haystack, or
                # "464,001" registers as 464 and 001 and never matches 464001
                registered |= set(re.findall(
                    r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", denumber(blob)))
    body = tex_body()
    nums = set()
    for m in re.finditer(r"\$([^$]{1,120})\$", denumber(body)):
        nums |= {v for v in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", m.group(1))}
    for t in used_tables():
        # normalise tables identically to the tex body; without it "66,561" splits
        # into 66 and 561 and the fragment can never be matched
        nums |= set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])",
                               denumber((TABDIR / t).read_text())))
    # A token that is a substring of a longer extracted token is an artefact of
    # LaTeX splitting (8,867 -> 867; 2081 -> 081), not a distinct claim.
    nums = {v for v in nums
            if not any(v != w and v in w for w in nums)}
    unclassified = []
    for v in sorted(nums):
        if v in registered or v in EXEMPT_EXPLICIT:
            continue
        if any(re.match(p, v) for p in EXEMPT_RE):
            continue
        unclassified.append(v)
    if unclassified:
        bad(f"[2] {len(unclassified)} unclassified numerals: {unclassified[:25]}")
    else:
        ok(f"[2] all {len(nums)} numerals registered or classified")


# ---------------------------------------------------------------------- 3 (in check 1)
# Verbal magnitude claims are Claim objects with an interval; check 1 evaluates them.


# ---------------------------------------------------------------------- 4
def check_population(cl):
    """Claims in one group share population N and run identity.
    Catches: the 24%-from-N=5 paired with 0.27-from-N=300."""
    groups: dict[str, set] = {}
    for c in cl:
        groups.setdefault(c.group, set()).add((c.n, c.run))
    for g, s in sorted(groups.items()):
        if len(s) == 1:
            ok(f"[4] group {g}: single population {sorted(s)[0]}")
        else:
            ns = {n for n, _ in s}
            if len(ns) > 1:
                bad(f"[4] group {g} MIXES populations: {sorted(s)}")
            else:
                ok(f"[4] group {g}: one N, runs {sorted(r for _, r in s)}")


# ---------------------------------------------------------------------- 5
def check_sources(cl):
    """Every source is live (not archived) and post-de-leak, or declared
    probe-independent. Catches: the reverted leaky tables; probe_timing.json archived
    while feeding tab_runtime; the stale significance file."""
    cut = datetime.strptime(C.DELEAK, "%Y-%m-%d").date()
    srcs = sorted({s for c in cl for s in c.sources})
    for s in srcs:
        p = REPO / s
        if not p.exists():
            bad(f"[5] source missing: {s}")
            continue
        if "ARCHIVE" in p.parts:
            bad(f"[5] source is archived: {s}")
            continue
        d = date.fromtimestamp(p.stat().st_mtime)
        if d < cut and s not in C.PROBE_INDEPENDENT:
            bad(f"[5] source predates de-leak with no exemption: {s} ({d})")
        else:
            ok(f"[5] {s} live, {d}")


# ---------------------------------------------------------------------- 6
def check_aggregates():
    """Pre-aggregated files must agree with recomputation from their raw inputs.
    Catches the class of defect where a summary file is never regenerated."""
    ms = json.loads((C.D / "multi_seed_summary.json").read_text())["splits"]
    for split, files in (("dev_test", C.DEV), ("test_OOD", C.OOD)):
        per = ms[split]["per_seed"]
        for entry, f in zip(per, files):
            if entry["file"] != f:
                bad(f"[6] multi_seed_summary {split} order/file mismatch: "
                    f"{entry['file']} vs {f}")
                continue
            got = entry["agg"]["probe_t020"]["mean_S_over_OPT"]
            want = C.opt_ratio(C.poly(f), "probe_t020")
            (ok if abs(got - want) < 5e-9 else bad)(
                f"[6] multi_seed_summary {f}: {got:.6f} vs raw {want:.6f}")
    sig = json.loads((C.D / "significance_ood.json").read_text())["per_seed"]
    for s, f in zip(("1234", "11", "22", "33"), C.OOD):
        recs = C.poly(f)
        want = C.below(recs, "probe_t020")
        got = sig[s]["probe_failures"]
        (ok if got == want else bad)(
            f"[6] significance_ood seed {s}: probe_failures {got} vs raw {want}")


# ---------------------------------------------------------------------- 7
def check_figures():
    """Figures newer than their data; sources live. Catches a figure rebuilt from
    stale data, and (with the visual review) the phantom legend entry."""
    figs = {
        "fig_po_training": ["paper/data/po_agp_training.json"],
        "fig_worked_example": ["paper/data/worked_examples.json"],
        "fig_embedding": ["paper/data/encoder_embedding_views.json"],
        "fig_distributions": ["paper/data/dist_dev_test.json",
                              "paper/data/dist_test_OOD.json",
                              "paper/data/dist_ood_large.json"],
        "fig_mechanism": ["paper/data/setpred_iter_sweep.json"],
    }
    for f, srcs in figs.items():
        pdf = REPO / "paper" / "gfx" / "setpred" / f"{f}.pdf"
        if not pdf.exists():
            bad(f"[7] figure missing: {f}")
            continue
        fm = pdf.stat().st_mtime
        stale = [s for s in srcs if (REPO / s).stat().st_mtime > fm]
        (bad if stale else ok)(
            f"[7] {f}: " + (f"STALE vs {stale}" if stale else "newer than its data"))


# ---------------------------------------------------------------------- 8
RETRACTED = {
    r"closes the tail": "the tail is shortened, not closed (17/362 remain)",
    r"eliminates that tail": "same",
    r"linearly separable(?!:)": "AUC 0.84 with substantial overlap is not separability",
    r"settles the diagnosis": "the decoder-calibration inference two reviewers objected to",
    r"2\.6\\times": "leak-era guard-overhead figure",
    r"2\.8\\times": "leak-era probe-vs-LS cardinality figure",
    r"1\.2 \\times 10\^\{-75\}": "leak-era p-value bound; true max is 7.4e-75",
    r"two to three times": "leak-era guard overhead; leak-free is ~1.6x",
    r"fourteen-fold": "leak-era",
    r"2\.556|3\.109|2\.334": "leak-era table values",
}


def check_retracted():
    """Phrases removed as overclaims must not reappear. Catches the Method's
    surviving decoder-calibration inference and the 'closes the tail' section title."""
    body = tex_body()
    for pat, why in RETRACTED.items():
        hits = re.findall(pat, body)
        # "linearly separable" is allowed when explicitly negated
        if pat.startswith("linearly separable") and hits:
            neg = len(re.findall(r"\\emph\{not\} linearly separable", body))
            if neg >= len(hits):
                ok(f"[8] '{pat}' only in negated form")
                continue
        (bad if hits else ok)(
            f"[8] '{pat}': {len(hits)} occurrence(s)" + (f" -- {why}" if hits else ""))


# ---------------------------------------------------------------------- 9
def check_pdf(cl):
    """The rendered PDF contains every registered numeral, and is newer than its
    inputs. The only check that validates the artifact a reader actually sees."""
    if not PDF.exists():
        bad("[9] paper.pdf missing")
        return
    # Content hashes, not mtimes. Regenerating a table byte-identically updates its
    # mtime and latexmk then correctly skips the rebuild, so an mtime comparison
    # reports staleness that does not exist. What matters is whether the inputs have
    # *changed* since the PDF was built.
    import hashlib
    inputs = [TEX] + [TABDIR / t for t in used_tables()] + sorted(
        (REPO / "paper" / "gfx" / "setpred").glob("*.pdf"))
    digest = {p.relative_to(REPO).as_posix():
              hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in inputs}
    man = REPO / "paper" / ".build_manifest.json"
    if "--record" in sys.argv:
        man.write_text(json.dumps(digest, indent=1))
        print(f"  recorded build manifest for {len(digest)} inputs")
        return
    if not man.exists():
        bad("[9] no build manifest; run with --record after a clean compile")
    else:
        old = json.loads(man.read_text())
        changed = [k for k in digest if old.get(k) != digest[k]]
        (bad if changed else ok)(
            "[9] pdf built from the current inputs" if not changed
            else f"[9] inputs changed since the pdf was built: {changed}")
    try:
        txt = subprocess.run(["pdftotext", str(PDF), "-"], capture_output=True,
                             text=True, timeout=120).stdout
    except Exception as e:  # noqa: BLE001
        bad(f"[9] pdftotext failed: {e}")
        return
    flat = re.sub(r"\s+", " ", txt)
    missing = []
    for c in cl:
        if not c.typeset:
            continue
        # strip LaTeX so the numeral can be sought in rendered text
        plain = re.sub(r"\\[a-zA-Z]+|[${}\\,]", " ", c.typeset)
        for tok in re.findall(r"\d+(?:\.\d+)?(?:/\d+)?", plain):
            if len(tok) >= 3 and tok not in flat:
                missing.append((c.id, tok))
    if missing:
        bad(f"[9] {len(missing)} registered numerals absent from the PDF: {missing[:10]}")
    else:
        ok("[9] every registered numeral appears in the rendered PDF")


# ---------------------------------------------------------------------- main
def main() -> None:
    verbose = "-v" in sys.argv
    try:
        cl = C.build()
    except (FileNotFoundError, OSError) as e:
        # A cited source going missing (archived, moved, deleted) makes the registry
        # unbuildable. Report it as the check-5 failure it is rather than dying with a
        # traceback -- mutation testing showed the crash was indistinguishable from a
        # gate bug.
        print(f"  [BAD] [5] a cited source is unreadable, registry cannot be built: {e}")
        print("\n  check    pass   fail\n  5          0      1")
        sys.exit(1)
    check_registry(cl)
    check_completeness(cl)
    check_population(cl)
    check_sources(cl)
    check_aggregates()
    check_figures()
    check_retracted()
    check_pdf(cl)

    if verbose:
        for m in PASS:
            print(f"  [OK ] {m}")
    for m in FAIL:
        print(f"  [BAD] {m}")
    by = {}
    for m in PASS + FAIL:
        k = m.split("]")[0].strip("[")
        by.setdefault(k, [0, 0])
        by[k][0 if m in PASS else 1] += 1
    print(f"\n  {'check':<6} {'pass':>6} {'fail':>6}")
    for k in sorted(by, key=lambda x: int(x)):
        print(f"  {k:<6} {by[k][0]:>6} {by[k][1]:>6}")
    print(f"\n{len(cl)} claims registered. {len(PASS)} passed, {len(FAIL)} failed.")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
