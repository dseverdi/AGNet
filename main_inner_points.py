from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import colors
from room_dataset import (
    discretize_room_to_grid,
    generate_room_dataset,
)


def render_inner_points_image(
    room: np.ndarray,
    output_path: Path,
    bounds: tuple[float, float, float, float],
    dx: float = 1.0,
    dy: float = 1.0,
) -> None:
    """Render room with inner grid points spaced dx/dy apart."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _, _, _, _, centers = discretize_room_to_grid(room, dx=dx, dy=dy)

    l, r, b, t = bounds
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=170)

    # Room boundary — blue dashed line with vertex markers.
    closed_x = np.append(room[:, 0], room[0, 0])
    closed_y = np.append(room[:, 1], room[0, 1])
    ax.plot(closed_x, closed_y, color=colors.blue, linewidth=0.8, linestyle="--", zorder=2)
    ax.scatter(room[:, 0], room[:, 1], s=12, c=colors.blue, zorder=3)

    # Inner points.
    if centers.shape[0] > 0:
        ax.scatter(centers[:, 0], centers[:, 1], s=0.05, c=colors.darkgray, zorder=4)

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


def save_inner_points_images(
    output_dir: str | Path = "data/images/inner_points",
    bounds: tuple[float, float, float, float] = (-15.0, 15.0, -15.0, 15.0),
    seed: int = 2026,
) -> list[Path]:
    """Generate one inner-points image per dataset type."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("dataset_1", dict(
            n_rooms=1, min_vertices=4, max_vertices=12, seed=7,
            hard_mode=True, min_reflex_vertices=4, min_passage_width=1.0,
            center_at_zero=True,
        )),
        ("dataset_2", dict(
            n_rooms=1, bounds=bounds, seed=42,
            rectangle_intersection_mode=True, center_at_zero=True,
        )),
    ]

    paths: list[Path] = []
    for name, kwargs in configs:
        rooms = generate_room_dataset(**kwargs)
        room = rooms[0]

        out = output_dir / f"{name}.png"
        render_inner_points_image(room, out, bounds=bounds, dx=1.0, dy=1.0)
        paths.append(out)
        print(f"Saved {out}")

    return paths


if __name__ == "__main__":
    save_inner_points_images()
