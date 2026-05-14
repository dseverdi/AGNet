#!/usr/bin/env python3
"""Sweep (stop_threshold, cov_gate_relative) on a trained EditHead checkpoint.

For each combination the script calls eval_editor.py as a subprocess (with
--no-ls-reference for speed), saves the result JSON, then reads all results
back and prints a Markdown table.

Goal: locate the operating point that maximises "cov ≥ baseline AND smallest
|S|". The grid sweeps:
  • stop_threshold ∈ {0.5, 0.7, 0.9}          — when STOP head fires
  • cov_gate_relative ∈ {-0.005, 0.0, 0.005, 0.01, 0.02, None}
        Negative value forces every committed edit to INCREASE coverage by
        at least that amount. 0.0 forbids any cov drop. Positive values
        allow degradation up to the magnitude.

Typical usage:
    python tools/sweep_editor_gate.py \
        --editor-checkpoint checkpoints/editor_dagger_geo_free_v2/editor_best.pt \
        --output-dir results/editor_sweep/v2 \
        --n-samples 300

The table is also written to <output-dir>/sweep_table.md.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


# Grid axes -------------------------------------------------------------------

STOP_THRESHOLDS: list[float] = [0.5, 0.7, 0.9]
COV_GATE_RELATIVE: list[float | None] = [None, -0.005, 0.0, 0.005, 0.01, 0.02]

# Markdown table columns ------------------------------------------------------
# Primary metrics: cov, |S|/n, |S|/OPT. Diagnostics: stopped, rejected.

TABLE_HEADER = (
    "| stop_thresh | cov_gate_rel | cov     | |S|/n   | |S|/OPT | "
    "stopped | rejected | n |\n"
    "| ---:        | ---:         | ---:    | ---:    | ---:    | "
    "---:    | ---:     | ---: |"
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
    """Call eval_editor.py and return the *summary* dict from its result JSON.

    eval_editor.py writes {"summary": {...}, "records": [...]}. We unwrap
    and return the summary so downstream code can read keys directly.
    Also stamps the operating point onto the dict for later formatting.
    """
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
        full = json.load(f)
    summary = full.get("summary", {}) if isinstance(full, dict) else {}

    # Compute rejection total from per-polygon records (eval_editor saves
    # ed_rejected per record but doesn't aggregate it into the summary).
    records = full.get("records", []) if isinstance(full, dict) else []
    rejected_total = sum(int(r.get("ed_rejected", 0)) for r in records)
    summary["ed_rejected_total"] = rejected_total
    return summary


def _fmt(value: object, fmt: str = ".4f") -> str:
    if value is None:
        return "—"
    try:
        return format(float(value), fmt)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


def _build_table_row(stop_t: float, cov_gate_rel: float | None, data: dict) -> str:
    """Primary metrics (cov, |S|/n, |S|/OPT) plus diagnostics."""
    cov  = _fmt(data.get("ed_cov_mean"))
    chv  = _fmt(data.get("ed_chv_mean"))
    sopt = _fmt(data.get("ed_ratio_opt_mean"))
    stopped = _fmt(data.get("ed_stopped_frac"), fmt=".2f")
    rejected = str(data.get("ed_rejected_total", "—"))
    n = str(data.get("n_polygons", "—"))
    cg_label = str(cov_gate_rel) if cov_gate_rel is not None else "none"
    return f"| {stop_t} | {cg_label} | {cov} | {chv} | {sopt} | {stopped} | {rejected} | {n} |"


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
                        help=f"Stop-threshold values to sweep (default: {STOP_THRESHOLDS}).")
    parser.add_argument("--cov-gate-relatives", nargs="+", type=float,
                        default=None,
                        metavar="EPS",
                        help="cov_gate_relative values to sweep. Each value is "
                             "passed literally — 0.0 means 'no degradation', "
                             "negative means 'require improvement of that magnitude'. "
                             "Pass nothing to use default grid: "
                             f"{[v for v in COV_GATE_RELATIVE if v is not None]}.")
    parser.add_argument("--include-no-gate", action="store_true",
                        default=True,
                        help="Include the no-gate baseline (no --cov-gate-relative "
                             "argument) in addition to the values above.")
    parser.add_argument("--no-include-no-gate", dest="include_no_gate",
                        action="store_false")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Forward to eval_editor.py --n-samples. -1 = full val.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands that would be run, but don't execute them.")
    # Extra args forwarded verbatim to eval_editor.py (e.g. --val-split, --device).
    parser.add_argument("extra", nargs=argparse.REMAINDER,
                        help="Extra arguments forwarded to eval_editor.py.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build cov_gate_relative axis. None encodes "no gate at all"; numerical
    # values are passed literally (incl. 0.0 and negatives).
    if args.cov_gate_relatives is None:
        cov_gate_axis: list[float | None] = list(COV_GATE_RELATIVE)
    else:
        cov_gate_axis = list(args.cov_gate_relatives)
        if args.include_no_gate and None not in cov_gate_axis:
            cov_gate_axis = [None] + cov_gate_axis

    extra_args: list[str] = [a for a in args.extra if a != "--"]
    if args.n_samples is not None:
        extra_args.extend(["--n-samples", str(args.n_samples)])

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

    # Pareto frontier on (cov, |S|/OPT): keep points where no other point
    # dominates (no other has cov ≥ this AND |S|/OPT ≤ this strictly).
    def _kv(d, key, default=None):
        v = d.get(key)
        try:
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    points = [(st, cg, _kv(d, "ed_cov_mean"), _kv(d, "ed_ratio_opt_mean"), d)
              for st, cg, d in results
              if _kv(d, "ed_cov_mean") is not None
              and _kv(d, "ed_ratio_opt_mean") is not None]

    pareto = []
    for i, (st, cg, c, s, d) in enumerate(points):
        dominated = False
        for j, (_, _, c2, s2, _) in enumerate(points):
            if j == i:
                continue
            if c2 >= c and s2 <= s and (c2 > c or s2 < s):
                dominated = True
                break
        if not dominated:
            pareto.append((st, cg, c, s, d))
    pareto.sort(key=lambda x: x[3])  # by |S|/OPT ascending

    if pareto:
        print("\n[sweep] Pareto frontier on (cov, |S|/OPT):")
        print(f"  {'stop':>6}  {'cov_gate':>10}  {'cov':>8}  {'|S|/n':>8}  {'|S|/OPT':>10}")
        for st, cg, c, s, d in pareto:
            chv = _kv(d, "ed_chv_mean", 0.0)
            cg_label = str(cg) if cg is not None else "none"
            print(f"  {st:>6}  {cg_label:>10}  {c:8.4f}  {chv:8.4f}  {s:10.4f}")
    else:
        print("\n[sweep] No valid points to build Pareto frontier.")

    # Also surface the "cov-preserving" operating points: ed_cov >= seed_cov.
    seed_cov_lookup = [d.get("seed_cov_mean") for st, cg, d in results
                       if d.get("seed_cov_mean") is not None]
    seed_cov = max(seed_cov_lookup) if seed_cov_lookup else 0.97
    cov_preserving = [(st, cg, d) for st, cg, c, s, d in points
                      if c >= seed_cov - 1e-4]
    if cov_preserving:
        cov_preserving.sort(key=lambda x: _kv(x[2], "ed_ratio_opt_mean", 1e9))
        st, cg, d = cov_preserving[0]
        cg_label = str(cg) if cg is not None else "none"
        print(f"\n[sweep] Best cov-preserving point (cov >= {seed_cov:.4f}):")
        print(f"  stop={st}  cov_gate_rel={cg_label}  "
              f"cov={d.get('ed_cov_mean'):.4f}  "
              f"|S|/n={d.get('ed_chv_mean'):.4f}  "
              f"|S|/OPT={d.get('ed_ratio_opt_mean'):.4f}")


if __name__ == "__main__":
    main()
