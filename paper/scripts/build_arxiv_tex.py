"""Generate paper/paper-arxiv.tex: a single, self-contained arXiv submission file.

paper.tex builds two targets from one source, selected by \\ifdefined\\arxiv:
elsarticle (journal, default) or article (preprint). A tiny \\input-based wrapper
was tried first but rejected: arXiv's own toplevel-detection heuristic picks
whichever file contains \\documentclass, which is paper.tex, not the wrapper --
so uploading the wrapper alongside paper.tex silently built the wrong target
unless a 00README.XXX override also shipped. Flattening removes the ambiguity
instead of routing around it: the generated file is the ONLY .tex in the arXiv
bundle with a \\documentclass, so there is nothing for arXiv to disambiguate.

paper-arxiv.tex is committed (like paper.pdf) so it always matches the current
paper.tex; regenerate after every edit and before repackaging the arXiv bundle.

Output: paper/paper-arxiv.tex

Usage:
  python paper/scripts/build_arxiv_tex.py
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "paper" / "paper.tex"
DST = REPO / "paper" / "paper-arxiv.tex"


def main() -> None:
    body = SRC.read_text()
    DST.write_text("\\def\\arxiv{}\n" + body)
    print(f"wrote {DST.relative_to(REPO)} ({len(body.splitlines())+1} lines)")


if __name__ == "__main__":
    main()
