"""Art Gallery Problem — dataset creation (ILP ground truth) and visualization."""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, PULP_CBC_CMD

import visilibity as vis

import colors
from room_dataset import generate_room_dataset

# ── colour palette for plotting ──────────────────────────────────────────────
GUARD_COLORS = [
    (colors.green,   "#c3f5e0"),
    (colors.purple,  "#e2d6f5"),
    (colors.orange,  "#ffe0c2"),
    (colors.cyan,    "#d4f5f7"),
    (colors.magenta, "#fcd4e8"),
    (colors.yellow,  "#fff9d4"),
    (colors.red,     "#fcd4d4"),
    (colors.blue,    "#d4e4fc"),
]

# ── geometry helpers ─────────────────────────────────────────────────────────

def _ensure_ccw(vertices: np.ndarray) -> np.ndarray:
    area2 = np.sum(
        vertices[:, 0] * np.roll(vertices[:, 1], -1)
        - vertices[:, 1] * np.roll(vertices[:, 0], -1)
    )
    if area2 < 0:
        return vertices[::-1].copy()
    return vertices


def _point_in_polygon(point: np.ndarray, poly: np.ndarray) -> bool:
    x, y = point
    n = poly.shape[0]
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _nudge_inward(room: np.ndarray, idx: int, amount: float = 0.01) -> np.ndarray:
    n = room.shape[0]
    prev = room[(idx - 1) % n]
    curr = room[idx]
    nxt = room[(idx + 1) % n]
    e1 = prev - curr
    e2 = nxt - curr
    n1 = np.linalg.norm(e1)
    n2 = np.linalg.norm(e2)
    if n1 > 0 and n2 > 0:
        bisector = e1 / n1 + e2 / n2
    else:
        bisector = np.array([0.0, 0.0])
    bn = np.linalg.norm(bisector)
    if bn > 0:
        candidate = curr + bisector / bn * amount
        if _point_in_polygon(candidate, room):
            return candidate
        candidate = curr - bisector / bn * amount
        if _point_in_polygon(candidate, room):
            return candidate
    return curr.copy()


def compute_visibility_polygon(
    room: np.ndarray, origin: np.ndarray, eps: float = 1e-7,
) -> np.ndarray:
    ccw_room = _ensure_ccw(room)
    wall = vis.Polygon([vis.Point(float(x), float(y)) for x, y in ccw_room])
    env = vis.Environment([wall])
    observer = vis.Point(float(origin[0]), float(origin[1]))
    vp = vis.Visibility_Polygon(observer, env, eps)
    return np.array([[vp[i].x(), vp[i].y()] for i in range(vp.n())])


# ── dataset creation ─────────────────────────────────────────────────────────

def _get_interior_witnesses(room: np.ndarray, spacing: float = 0.5) -> np.ndarray:
    from shapely.prepared import prep as _prep
    xs = room[:, 0]
    ys = room[:, 1]
    gx = np.arange(xs.min() + spacing / 2, xs.max(), spacing)
    gy = np.arange(ys.min() + spacing / 2, ys.max(), spacing)
    grid = np.array(np.meshgrid(gx, gy)).reshape(2, -1).T
    room_prep = _prep(ShapelyPolygon(room))
    pts = [ShapelyPoint(p) for p in grid]
    inside = np.array([room_prep.contains(p) for p in pts])
    return grid[inside]


def _coverage_row(room, origin, witnesses, eps=1e-7):
    """Return a boolean array: which witnesses are visible from *origin*."""
    from shapely.prepared import prep as _prep
    vp = compute_visibility_polygon(room, origin, eps=eps)
    if vp.shape[0] < 3:
        return np.zeros(len(witnesses), dtype=int)
    vp_prep = _prep(ShapelyPolygon(vp))
    pts = [ShapelyPoint(w) for w in witnesses]
    return np.array([int(vp_prep.contains(p)) for p in pts], dtype=int)


def build_coverage_matrix(room: np.ndarray, eps: float = 1e-7) -> tuple[np.ndarray, np.ndarray]:
    n_verts = room.shape[0]
    interior = _get_interior_witnesses(room, spacing=0.5)
    witnesses = np.vstack([room, interior])

    C = np.zeros((n_verts, len(witnesses)), dtype=int)
    for i in range(n_verts):
        origin = _nudge_inward(room, i)
        C[i] = _coverage_row(room, origin, witnesses, eps)
        C[i, i] = 1  # vertex always sees itself
    return C, witnesses


def compute_guard_coverage(
    room: np.ndarray,
    guard_indices: np.ndarray,
    spacing: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """Fast coverage check — only computes visibility for selected guards."""
    if len(guard_indices) == 0:
        return 0.0
    interior = _get_interior_witnesses(room, spacing=spacing)
    witnesses = np.vstack([room, interior])
    covered = np.zeros(len(witnesses), dtype=int)
    for i in guard_indices:
        origin = _nudge_inward(room, int(i))
        covered |= _coverage_row(room, origin, witnesses, eps)
    return covered.mean()


def solve_agp_ilp(coverage: np.ndarray) -> np.ndarray:
    n_guards, n_witnesses = coverage.shape
    prob = LpProblem("AGP", LpMinimize)
    x = [LpVariable(f"x{i}", cat="Binary") for i in range(n_guards)]
    prob += lpSum(x)

    for j in range(n_witnesses):
        guards_seeing_j = [i for i in range(n_guards) if coverage[i, j]]
        if guards_seeing_j:
            prob += lpSum(x[i] for i in guards_seeing_j) >= 1

    prob.solve(PULP_CBC_CMD(msg=0))
    return np.array([i for i in range(n_guards) if x[i].varValue > 0.5])


DATASET_CONFIGS = {
    "dataset_1": dict(
        min_vertices=4, max_vertices=12, seed=7,
        hard_mode=True, min_reflex_vertices=4, min_passage_width=1.0,
        center_at_zero=True,
    ),
    "dataset_2": dict(
        bounds=(-15.0, 15.0, -15.0, 15.0), seed=42,
        rectangle_intersection_mode=True, center_at_zero=True,
    ),
}


def create_dataset(
    output_dir: str | Path = "data",
    n_rooms: int = 10,
    chunk_size: int = 1000,
) -> dict[str, Path]:
    """Generate rooms, solve ILP, save as .pt chunk files.

    Saves chunks of `chunk_size` rooms to data/<ds_name>_<chunk_size>/chunk_0000.pt, etc.
    Resumes from existing chunks if the folder already contains data.

    Each datum dict:
        {"vertices": FloatTensor(N, 2), "guards": LongTensor(K,),
         "n_vertices": int, "n_guards": int, "solve_time": float}
    """
    output_dir = Path(output_dir)
    saved: dict[str, Path] = {}

    for ds_name, cfg in DATASET_CONFIGS.items():
        ds_dir = output_dir / f"{ds_name}_classic_{chunk_size}"
        ds_dir.mkdir(parents=True, exist_ok=True)

        # Detect existing chunks and resume from where we left off.
        existing_chunks = sorted(ds_dir.glob("chunk_*.pt"))
        n_existing_chunks = len(existing_chunks)
        n_existing_rooms = n_existing_chunks * chunk_size

        if n_existing_rooms >= n_rooms:
            print(f"{ds_name}: already have {n_existing_rooms} rooms "
                  f"({n_existing_chunks} chunks), nothing to do.")
            saved[ds_name] = ds_dir
            continue

        n_remaining = n_rooms - n_existing_rooms
        print(f"{ds_name}: found {n_existing_rooms} existing rooms "
              f"({n_existing_chunks} chunks), generating {n_remaining} more …")

        # Generate all rooms (deterministic via seed), then skip existing ones.
        kwargs = {**cfg, "n_rooms": n_rooms}
        rooms = generate_room_dataset(**kwargs)
        rooms_to_process = rooms[n_existing_rooms:]

        chunk: list[dict] = []
        chunk_idx = n_existing_chunks

        pbar = tqdm(enumerate(rooms_to_process), total=len(rooms_to_process),
                    desc=ds_name, unit="room")
        for i, room in pbar:
            t0 = time.perf_counter()
            C, _ = build_coverage_matrix(room)
            guards = solve_agp_ilp(C)
            solve_time = time.perf_counter() - t0

            chunk.append({
                "vertices": torch.from_numpy(room).float(),
                "guards": torch.from_numpy(guards).long(),
                "n_vertices": room.shape[0],
                "n_guards": len(guards),
                "solve_time": solve_time,
            })
            pbar.set_postfix(verts=room.shape[0], guards=len(guards),
                             t=f"{solve_time:.2f}s")

            if len(chunk) >= chunk_size:
                chunk_path = ds_dir / f"chunk_{chunk_idx:04d}.pt"
                torch.save(chunk, chunk_path)
                tqdm.write(f"  → Saved {chunk_path} ({len(chunk)} rooms)")
                chunk = []
                chunk_idx += 1

        if chunk:
            chunk_path = ds_dir / f"chunk_{chunk_idx:04d}.pt"
            torch.save(chunk, chunk_path)
            tqdm.write(f"  → Saved {chunk_path} ({len(chunk)} rooms)")

        saved[ds_name] = ds_dir

    return saved


def concat_dataset(data_dir: str | Path = "data", chunk_size: int = 1000) -> dict[str, Path]:
    """Concatenate chunk files into a single .pt file per dataset."""
    data_dir = Path(data_dir)
    saved: dict[str, Path] = {}

    for ds_name in DATASET_CONFIGS:
        ds_dir = data_dir / f"{ds_name}_classic_{chunk_size}"
        chunks = sorted(ds_dir.glob("chunk_*.pt"))
        if not chunks:
            print(f"Skipping {ds_name}: no chunks found in {ds_dir}")
            continue

        dataset: list[dict] = []
        for cp in chunks:
            dataset.extend(torch.load(cp, weights_only=False))

        out_path = data_dir / f"{ds_name}_classic_{chunk_size}.pt"
        torch.save(dataset, out_path)
        saved[ds_name] = out_path
        print(f"→ {out_path}: {len(dataset)} rooms from {len(chunks)} chunks")

    return saved


# ── plotting from saved dataset ──────────────────────────────────────────────

def plot_dataset(
    data_dir: str | Path = "data",
    image_dir: str | Path = "data/images/optimal_classic",
    bounds: tuple[float, float, float, float] = (-15.0, 15.0, -15.0, 15.0),
    max_images: int | None = None,
    chunk_size: int = 1000,
) -> list[Path]:
    """Load .pt dataset files and render guard visualizations."""
    data_dir = Path(data_dir)
    image_dir = Path(image_dir)
    paths: list[Path] = []

    for ds_name in DATASET_CONFIGS:
        folder_name = f"{ds_name}_classic_{chunk_size}"
        pt_path = data_dir / f"{folder_name}.pt"
        ds_chunks = data_dir / folder_name
        if pt_path.exists():
            dataset = torch.load(pt_path, weights_only=False)
        elif ds_chunks.exists():
            dataset = []
            for cp in sorted(ds_chunks.glob("chunk_*.pt")):
                dataset.extend(torch.load(cp, weights_only=False))
        else:
            print(f"Skipping {ds_name}: no data found in {ds_chunks}")
            continue
        ds_dir = image_dir / folder_name
        ds_dir.mkdir(parents=True, exist_ok=True)

        n = len(dataset) if max_images is None else min(max_images, len(dataset))
        for i, datum in enumerate(dataset[:n], start=1):
            room = datum["vertices"].numpy()
            guard_indices = datum["guards"].numpy()

            out = ds_dir / f"{i}.png"
            _render_one(room, guard_indices, out, bounds)
            paths.append(out)
            print(f"Plotted {out}")

    return paths


def _render_one(
    room: np.ndarray,
    guard_indices: np.ndarray,
    output_path: Path,
    bounds: tuple[float, float, float, float] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if bounds is not None:
        l, r, b, t = bounds
    else:
        xspan = room[:, 0].max() - room[:, 0].min()
        yspan = room[:, 1].max() - room[:, 1].min()
        margin = max(xspan, yspan) * 0.05 + 1.0
        l, r = room[:, 0].min() - margin, room[:, 0].max() + margin
        b, t = room[:, 1].min() - margin, room[:, 1].max() + margin
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=170)

    room_shapely = ShapelyPolygon(room)

    for gi, idx in enumerate(guard_indices):
        strong, pale = GUARD_COLORS[gi % len(GUARD_COLORS)]
        origin = _nudge_inward(room, idx)
        vp = compute_visibility_polygon(room, origin)
        if vp.shape[0] >= 3:
            vp_shapely = ShapelyPolygon(vp)
            clipped = room_shapely.intersection(vp_shapely)
            if not clipped.is_empty:
                geoms = [clipped] if clipped.geom_type == "Polygon" else list(clipped.geoms)
                for geom in geoms:
                    if geom.geom_type == "Polygon" and not geom.is_empty:
                        coords = np.array(geom.exterior.coords)
                        patch = MplPolygon(
                            coords, closed=True,
                            facecolor=pale, edgecolor=strong,
                            linewidth=0.4, alpha=0.45, zorder=1,
                        )
                        ax.add_patch(patch)
        ax.scatter([room[idx, 0]], [room[idx, 1]], s=50, c=strong, zorder=5, marker="*")

    # Room boundary.
    closed_x = np.append(room[:, 0], room[0, 0])
    closed_y = np.append(room[:, 1], room[0, 1])
    ax.plot(closed_x, closed_y, color=colors.blue, linewidth=0.8, linestyle="--", zorder=2)

    # Non-guard vertices.
    non_guard = np.array([i for i in range(room.shape[0]) if i not in guard_indices])
    if non_guard.size > 0:
        ax.scatter(room[non_guard, 0], room[non_guard, 1], s=14, c=colors.blue, zorder=3)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(l, r)
    ax.set_ylim(b, t)
    span = max(r - l, t - b)
    if span > 100:
        step = 50
    elif span > 40:
        step = 10
    else:
        step = 5
    ax.set_xticks(np.arange(np.ceil(l / step) * step, r + 1, step))
    ax.set_yticks(np.arange(np.ceil(b / step) * step, t + 1, step))
    ax.set_axisbelow(True)
    ax.grid(True, which="both", color=colors.lightgray, linewidth=0.7, alpha=0.7, zorder=-10)

    ax.set_title(f"Guards: {len(guard_indices)}", fontsize=9)
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classic AGP dataset creation & plotting")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Generate rooms + solve ILP → chunk .pt files")
    p_create.add_argument("-n", "--n-rooms", type=int, default=10)
    p_create.add_argument("-o", "--output-dir", type=str, default="data")
    p_create.add_argument("-c", "--chunk-size", type=int, default=1000)

    p_concat = sub.add_parser("concat", help="Concatenate chunk files into single .pt")
    p_concat.add_argument("-d", "--data-dir", type=str, default="data")
    p_concat.add_argument("-c", "--chunk-size", type=int, default=1000)

    p_plot = sub.add_parser("plot", help="Render images from saved .pt files")
    p_plot.add_argument("-d", "--data-dir", type=str, default="data")
    p_plot.add_argument("-o", "--image-dir", type=str, default="data/images/optimal_classic")
    p_plot.add_argument("-m", "--max-images", type=int, default=None)
    p_plot.add_argument("-c", "--chunk-size", type=int, default=1000)

    args = parser.parse_args()

    if args.command == "create":
        create_dataset(output_dir=args.output_dir, n_rooms=args.n_rooms,
                       chunk_size=args.chunk_size)
    elif args.command == "concat":
        concat_dataset(data_dir=args.data_dir, chunk_size=args.chunk_size)
    elif args.command == "plot":
        plot_dataset(data_dir=args.data_dir, image_dir=args.image_dir,
                     max_images=args.max_images, chunk_size=args.chunk_size)
