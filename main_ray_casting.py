from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import colors
from ray_casting import ray_casting, ray_casting_init
from room_dataset import (
    discretize_room_to_grid,
    generate_room_dataset,
    polygon_to_edges,
)


def compute_visible_cells(
    room: np.ndarray,
    origin: np.ndarray,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute visible cells from one origin using 360 radial rays and ray-casting."""
    north, south, east, west, centers = discretize_room_to_grid(room, dx=dx, dy=dy)
    if centers.shape[0] == 0:
        return centers, np.zeros((0,), dtype=bool), np.zeros((0, 2, 2), dtype=float)

    n_cells = centers.shape[0]
    grid_edges = np.concatenate([north, south, east, west], axis=0)

    # 360 evenly distributed rays from the selected cell center.
    n_rays = 360
    angles = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
    z_max = 60.0

    rays = np.zeros((n_rays, 2, 2), dtype=float)
    rays[:, 0, 0] = origin[0]
    rays[:, 0, 1] = origin[1]
    rays[:, 1, 0] = origin[0] + z_max * np.cos(angles)
    rays[:, 1, 1] = origin[1] + z_max * np.sin(angles)

    room_edges = polygon_to_edges(room)
    clipped_rays = ray_casting(room_edges, rays)

    ray_casting_init(clipped_rays.shape[0], grid_edges.shape[0])
    whether_intersect = ray_casting(
        clipped_rays,
        grid_edges,
        initialized_indices=True,
        return_whether_intersect=True,
    )
    # Reshape from (n_rays, 4*n_cells) → (n_rays, n_cells, 4) since
    # grid_edges are ordered [north, south, east, west].
    whether_intersect = whether_intersect.reshape(n_rays, 4, n_cells).transpose(0, 2, 1)

    intersect_at_least_one_side = np.sum(whether_intersect, axis=2) > 0
    visible_cells = np.sum(intersect_at_least_one_side.astype(int), axis=0) > 0

    return centers, visible_cells, clipped_rays


def render_visibility_image(
    room: np.ndarray,
    centers: np.ndarray,
    visible_cells: np.ndarray,
    origin: np.ndarray,
    rays: np.ndarray,
    output_path: Path,
    bounds: tuple[float, float, float, float],
    dx: float,
    dy: float,
) -> None:
    """Render one visibility image with visible discretized cells and dashed room border."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    l, r, b, t = bounds
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=170)

    # Draw all inside cells faintly, visible cells highlighted.
    for c in centers:
        x0 = c[0] - dx / 2.0
        y0 = c[1] - dy / 2.0
        ax.add_patch(
            Rectangle(
                (x0, y0),
                dx,
                dy,
                facecolor="#ffffff",
                edgecolor=colors.lightgray,
                linewidth=0.2,
                zorder=0,
            )
        )

    for c in centers[visible_cells]:
        x0 = c[0] - dx / 2.0
        y0 = c[1] - dy / 2.0
        ax.add_patch(
            Rectangle(
                (x0, y0),
                dx,
                dy,
                facecolor=colors.paleblue,
                edgecolor=colors.orange,
                linewidth=0.25,
                zorder=1,
            )
        )

    # Draw a sparse subset of clipped rays for readability.
    for ray in rays[::6]:
        ax.plot(ray[:, 0], ray[:, 1], color=colors.darkgray, linewidth=0.4, alpha=0.35, zorder=2)

    # Dashed room border from polygon points.
    ax.plot(
        np.append(room[:, 0], room[0, 0]),
        np.append(room[:, 1], room[0, 1]),
        color=colors.blue,
        linewidth=0.8,
        linestyle="--",
        zorder=3,
    )

    # Origin cell center.
    ax.scatter([origin[0]], [origin[1]], s=22, c=colors.red, zorder=4)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(l, r)
    ax.set_ylim(b, t)
    ticks = np.arange(l, r + 1, 3)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_axisbelow(True)
    ax.grid(True, which="both", color=colors.lightgray, linewidth=0.7, alpha=0.7, zorder=-10)

    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_visibility_images(
    output_dir: str | Path = "data/images/visibility_demo",
    n_images: int = 10,
    bounds: tuple[float, float, float, float] = (-15.0, 15.0, -15.0, 15.0),
    dx: float = 0.5,
    dy: float = 0.5,
    seed: int = 2026,
) -> list[Path]:
    """Generate and save visibility demo images from random origin cells."""
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the second dataset style (union-of-rectangles rooms) for demonstration.
    rooms = generate_room_dataset(
        n_rooms=n_images,
        bounds=bounds,
        seed=seed,
        rectangle_intersection_mode=True,
        center_at_zero=True,
    )

    paths: list[Path] = []
    for i, room in enumerate(rooms, start=1):
        _, _, _, _, centers = discretize_room_to_grid(room, dx=dx, dy=dy)
        if centers.shape[0] == 0:
            continue
        origin = centers[int(rng.integers(0, centers.shape[0]))]

        centers2, visible_cells, clipped_rays = compute_visible_cells(room, origin, dx=dx, dy=dy)

        out = output_dir / f"{i}.png"
        render_visibility_image(
            room=room,
            centers=centers2,
            visible_cells=visible_cells,
            origin=origin,
            rays=clipped_rays,
            output_path=out,
            bounds=bounds,
            dx=dx,
            dy=dy,
        )
        paths.append(out)

    return paths


if __name__ == "__main__":
    generated = save_visibility_images()
    print(f"Saved {len(generated)} visibility images to {(Path('data/images/visibility_demo')).resolve()}")
