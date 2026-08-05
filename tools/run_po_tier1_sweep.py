#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path


def _slug_float(value: float) -> str:
    text = f"{value}".replace(".", "p")
    return text.replace("-", "m")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and optionally run Tier-1 PO sweep configs."
    )
    parser.add_argument(
        "--sweep-config",
        type=str,
        default="configs/tier1/po_agp_tier1_sweep.json",
        help="Path to sweep definition JSON.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="If set, run all generated configs sequentially.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter to use when --run is enabled.",
    )
    args = parser.parse_args()

    with open(args.sweep_config, "r") as f:
        sweep = json.load(f)

    base_config_path = Path(sweep["base_config"])
    with open(base_config_path, "r") as f:
        base_config = json.load(f)

    alphas = [float(a) for a in sweep["alphas"]]
    rollouts = [int(k) for k in sweep["num_rollouts"]]
    out_dir = Path(sweep.get("output_config_dir", "configs/tier1/generated"))
    checkpoint_root = sweep.get("checkpoint_root", "checkpoints/v3/po_agp/tier1")
    verbose = bool(sweep.get("verbose", False))

    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for alpha, k in itertools.product(alphas, rollouts):
        cfg = dict(base_config)
        cfg["alpha"] = alpha
        cfg["num_rollouts"] = k
        run_id = f"a{_slug_float(alpha)}_k{k}"
        cfg["checkpoint_dir"] = f"{checkpoint_root}/{run_id}"

        out_path = out_dir / f"po_agp_tier1_{run_id}.json"
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=4)
            f.write("\n")
        generated.append((run_id, out_path))

    print(f"Generated {len(generated)} Tier-1 configs in {out_dir}")
    for run_id, out_path in generated:
        print(f"  - {run_id}: {out_path}")

    if not args.run:
        return

    for run_id, out_path in generated:
        cmd = [args.python, "po_agp.py", "--config", str(out_path)]
        if verbose:
            cmd.append("--verbose")
        print(f"\n[run] {run_id}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
