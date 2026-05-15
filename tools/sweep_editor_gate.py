#!/usr/bin/env python3
"""Sweep (stop_threshold, cov_gate_relative) on a trained EditHead checkpoint.

For each combination the script calls eval_editor.py as a subprocess (with
--no-ls-reference for speed), saves the result JSON, then reads all results
back and prints a Markdown table.

Typical usage:
    python tools/sweep_editor_gate.py \
        --editor-checkpoint checkpoints/editor_dagger/ep-21.pt \
        --output-dir results/editor_sweep/ep21

The table is also written to <output-dir>/sweep_table.md.
"""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


# Grid axes -------------------------------------------------------------------

STOP_THRESHOLDS: list[float] = [0.5, 0.6, 0.7, 0.8, 0.9]
COV_GATE_RELATIVE: list[float | None] = [None, 0.005, 0.01, 0.02]

# Markdown table columns ------------------------------------------------------

TABLE_HEADER = (
    "| stop_thresh | cov_gate_rel | cov_mean | sopt_mean | "
    "stopped_frac | rejected | n |\n"
    "| --- | --- | --- | --- | --- | --- | --- |"
)


# -----------------------------------------------------------------------------

def _slug(value: float | None) -> str:
    if value is None:
        return "none"
    return f"{value}".replace(".", "p")


def _run_eval(
    *,
    python: str,
    editor_checkpoint: str,
    stop_threshold: float,
    cov_gate_relative: float | None,
    output_json: Path,
    extra_args: list[str],
) -> dict:
    """Call eval_editor.py and return the parsed JSON result."""
    cmd = [
        python, "eval_editor.py",
        "--editor-checkpoint", editor_checkpoint,
        "--stop-threshold", str(stop_threshold),
        "--output-json", str(output_json),
        "--no-ls-reference",
    ]
    if cov_gate_relative is not None:
        cmd += ["--cov-gate-relative", str(cov_gate_relative)]
    cmd.extend(extra_args)

    label = f"st={stop_threshold}  cg={cov_gate_relative}"
    print(f"[sweep] running  {label} …")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[sweep] FAILED  {label}")
        print(result.stderr[-2000:] if result.stderr else "")
        return {}

    if not output_json.exists():
        print(f"[sweep] no output JSON for  {label}")
        return {}

    with open(output_json) as f:
        return json.load(f)


def _fmt(value: object, fmt: str = ".4f") -> str:
    if value is None:
        return "—"
    try:
        return format(float(value), fmt)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


def _build_table_row(stop_t: float, cov_gate_rel: float | None, data: dict) -> str:
    cov = _fmt(data.get("cov_mean"))
    sopt = _fmt(data.get("sopt_mean"))
    stopped = _fmt(data.get("ed_stopped_frac"))
    rejected = str(data.get("ed_rejected_total", "—"))
    n = str(data.get("n_polygons", "—"))
    cg_label = str(cov_gate_rel) if cov_gate_rel is not None else "none"
    return f"| {stop_t} | {cg_label} | {cov} | {sopt} | {stopped} | {rejected} | {n} |"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--editor-checkpoint", required=True,
                        help="Path to the EditHead checkpoint (.pt).")
    parser.add_argument("--output-dir", default="results/editor_sweep",
                        help="Directory to write per-run JSONs and the summary table.")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter to use (default: current interpreter).")
    parser.add_argument("--stop-thresholds", nargs="+", type=float,
                        default=STOP_THRESHOLDS,
                        metavar="T",
                        help="Stop-threshold values to sweep (default: 0.5 0.6 0.7 0.8 0.9).")
    parser.add_argument("--cov-gate-relatives", nargs="+", type=float,
                        default=None,
                        metavar="EPS",
                        help="cov_gate_relative values to sweep. Pass '0' to include "
                             "no-gate baseline. Default includes None plus "
                             "0.005 0.01 0.02.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands that would be run, but don't execute them.")
    # Extra args forwarded verbatim to eval_editor.py (e.g. --val-split, --device).
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="Extra arguments forwarded to eval_editor.py.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build cov_gate_relative axis.
    if args.cov_gate_relatives is None:
        cov_gate_axis: list[float | None] = COV_GATE_RELATIVE
    else:
        cov_gate_axis = [None if v == 0.0 else v for v in args.cov_gate_relatives]

    extra_args: list[str] = [a for a in args.extra if a != "--"]

    grid = list(itertools.product(args.stop_thresholds, cov_gate_axis))
    print(f"[sweep] {len(grid)} combinations  "
          f"({len(args.stop_thresholds)} stop × {len(cov_gate_axis)} cov_gate)")
    print(f"[sweep] output → {out_dir}/")

    rows: list[str] = [TABLE_HEADER]
    results: list[tuple[float, float | None, dict]] = []

    for stop_t, cov_gate_rel in grid:
        json_name = f"st{_slug(stop_t)}_cg{_slug(cov_gate_rel)}.json"
        output_json = out_dir / json_name

        if args.dry_run:
            cg_label = f"--cov-gate-relative {cov_gate_rel}" if cov_gate_rel is not None else "(no gate)"
            print(f"  [dry] st={stop_t}  {cg_label}  → {output_json}")
            continue

        data = _run_eval(
            python=args.python,
            editor_checkpoint=args.editor_checkpoint,
            stop_threshold=stop_t,
            cov_gate_relative=cov_gate_rel,
            output_json=output_json,
            extra_args=extra_args,
        )
        results.append((stop_t, cov_gate_rel, data))
        rows.append(_build_table_row(stop_t, cov_gate_rel, data))

    if args.dry_run:
        return

    table_text = "\n".join(rows) + "\n"
    table_path = out_dir / "sweep_table.md"
    table_path.write_text(table_text)

    print("\n--- Sweep results ---")
    print(table_text)
    print(f"[sweep] table written to {table_path}")

    # Highlight operating points: cov_mean >= 0.970, sopt in [0.95, 1.00].
    good = [
        (st, cg, d) for st, cg, d in results
        if float(d.get("cov_mean", 0)) >= 0.970
        and 0.95 <= float(d.get("sopt_mean", 0)) <= 1.00
    ]
    if good:
        print("\n[sweep] Operating points (cov>=0.970, 0.95<=|S|/OPT<=1.00):")
        for st, cg, d in good:
            print(f"  stop={st}  cov_gate_rel={cg}  "
                  f"cov={d.get('cov_mean', '?'):.4f}  "
                  f"|S|/OPT={d.get('sopt_mean', '?'):.4f}")
    else:
        print("\n[sweep] No operating point found meeting (cov>=0.970, 0.95<=|S|/OPT<=1.00).")


if __name__ == "__main__":
    main()
