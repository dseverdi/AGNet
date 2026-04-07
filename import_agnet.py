"""Import AGNet .pol + .solution/.clusters files as dataset_3 splits in our .pt chunk format.

Usage:
    python import_agnet.py [--src PATH] [--dst data] [--chunk-size 100]

Imports each subdirectory (dev, train, test, large) as a separate dataset:
    data/dataset_3_dev_classic_100/chunk_XXXX.pt
    data/dataset_3_train_classic_100/chunk_XXXX.pt
    data/dataset_3_test_classic_100/chunk_XXXX.pt
    data/dataset_3_large_classic_100/chunk_XXXX.pt

Each datum matches the format used by dataset_1 and dataset_2:
    {"vertices": FloatTensor(N, 2), "guards": LongTensor(K,),
     "n_vertices": int, "n_guards": int, "solve_time": 0.0, "name": str}
"""

import argparse
from fractions import Fraction
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm


def _read_pol(path: Path) -> np.ndarray:
    """Read a .pol file → (N, 2) float64 array."""
    tokens = path.read_text().split()
    n = int(tokens[0])
    coords = []
    for i in range(1, 2 * n, 2):
        x = float(Fraction(tokens[i]))
        y = float(Fraction(tokens[i + 1]))
        coords.append((x, y))
    return np.array(coords, dtype=np.float64)


def _read_guards(path: Path) -> list[np.ndarray]:
    """Read a .solution or .clusters file → list of guard index arrays."""
    lines = path.read_text().strip().splitlines()
    n_sols = int(lines[0].split()[-1])
    solutions = []
    for line in lines[1: 1 + n_sols]:
        indices = np.array(list(map(int, line.split())), dtype=np.int64)
        solutions.append(indices)
    return solutions


def _import_split(
    src_dir: Path,
    split_name: str,
    dst: Path,
    chunk_size: int,
) -> Path:
    """Import a single split (dev/train/test/large) into chunks."""
    out_dir = dst / f"dataset_3_{split_name}_classic_{chunk_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pol_files = sorted(src_dir.glob("*.pol"))
    if not pol_files:
        print(f"  {split_name}: no .pol files found, skipping")
        return out_dir

    chunk: list[dict] = []
    chunk_idx = 0
    total = 0

    for pol_path in tqdm(pol_files, desc=f"dataset_3_{split_name}", unit="room"):
        stem = pol_path.stem
        clusters_path = pol_path.with_suffix(".clusters")
        solution_path = pol_path.with_suffix(".solution")

        if clusters_path.exists():
            sol_path = clusters_path
        elif solution_path.exists():
            sol_path = solution_path
        else:
            continue

        vertices = _read_pol(pol_path)
        solutions = _read_guards(sol_path)
        if not solutions:
            continue

        guards = solutions[0]

        chunk.append({
            "vertices": torch.from_numpy(vertices).float(),
            "guards": torch.from_numpy(guards).long(),
            "n_vertices": vertices.shape[0],
            "n_guards": len(guards),
            "solve_time": 0.0,
            "name": stem,
        })
        total += 1

        if len(chunk) >= chunk_size:
            cp = out_dir / f"chunk_{chunk_idx:04d}.pt"
            torch.save(chunk, cp)
            chunk = []
            chunk_idx += 1

    if chunk:
        cp = out_dir / f"chunk_{chunk_idx:04d}.pt"
        torch.save(chunk, cp)

    n_chunks = chunk_idx + (1 if chunk else 0)
    print(f"  → {total} rooms → {out_dir}  ({n_chunks} chunks)")
    return out_dir


SPLITS = ["dev", "train", "test", "large"]


def import_agnet(
    src: str | Path = "/home/jurica/Desktop/AGNet/dataset/AGPIL2/development",
    dst: str | Path = "data",
    chunk_size: int = 100,
) -> dict[str, Path]:
    src = Path(src)
    dst = Path(dst)
    results: dict[str, Path] = {}

    for split in SPLITS:
        split_dir = src / split
        if not split_dir.is_dir():
            print(f"  {split}: directory not found at {split_dir}, skipping")
            continue
        results[split] = _import_split(split_dir, split, dst, chunk_size)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import AGNet dataset as dataset_3 splits")
    parser.add_argument("--src", type=str,
                        default="/home/jurica/Desktop/AGNet/dataset/AGPIL2/development")
    parser.add_argument("--dst", type=str, default="data")
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()
    import_agnet(src=args.src, dst=args.dst, chunk_size=args.chunk_size)
