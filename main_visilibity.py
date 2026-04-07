from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

import visilibity as vis

import colors
from room_dataset import (
    discretize_room_to_grid,
    generate_room_dataset,
)


def _ensure_ccw(vertices: np.ndarray) -> np.ndarray:
    """Return vertices in counter-clockwise order (required by VisiLibity for outer boundary)."""
    area2 = np.sum(
        vertices[:, 0] * np.roll(vertices[:, 1], -1)
        - vertices[:, 1] * np.roll(vertices[:, 0], -1)
    )
    if area2 < 0:  # CW → reverse to CCW
        return vertices[::-1].copy()
    return vertices


def compute_visibility_polygon(
    room: np.ndarray,
    origin: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Compute the visibility polygon from *origin* inside *room* using VisiLibity."""
    ccw_room = _ensure_ccw(room)
    wall = vis.Polygon([vis.Point(float(x), float(y)) for x, y in ccw_room])
    env = vis.Environment([wall])

    observer = vis.Point(float(origin[0]), float(origin[1]))
    vp = vis.Visibility_Polygon(observer, env, eps)

    vis_poly = np.array([[vp[i].x(), vp[i].y()] for i in range(vp.n())])
    return vis_poly


def render_visibility_image(
    room: np.ndarray,
    vis_poly: np.ndarray,
    origin: np.ndarray,
    output_path: Path,
    bounds: tuple[float, float, float, float],
) -> None:
    """Render continuous visibility polygon inside the room boundary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    l, r, b, t = bounds
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=170)

    # Filled visibility polygon.
    if vis_poly.shape[0] >= 3:
        vis_patch = MplPolygon(
            vis_poly, closed=True,
            facecolor=colors.paleblue, edgecolor=colors.orange,
            linewidth=0.6, alpha=0.7, zorder=1,
        )
        ax.add_patch(vis_patch)

    # Room boundary — blue dashed line with vertex markers.
    closed_x = np.append(room[:, 0], room[0, 0])
    closed_y = np.append(room[:, 1], room[0, 1])
    ax.plot(closed_x, closed_y, color=colors.blue, linewidth=0.8, linestyle="--", zorder=2)
    ax.scatter(room[:, 0], room[:, 1], s=12, c=colors.blue, zorder=3)

    # Observer point.
    ax.scatter([origin[0]], [origin[1]], s=30, c=colors.red, zorder=4)

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
    output_dir: str | Path = "data/images/visibility_demo_visilibity1",
    bounds: tuple[float, float, float, float] = (-15.0, 15.0, -15.0, 15.0),
    seed: int = 2026,
) -> list[Path]:
    """Generate one visibility image per dataset type."""
    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []

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

    for name, kwargs in configs:
        rooms = generate_room_dataset(**kwargs)
        room = rooms[0]

        # Pick a random interior cell center as observer (guaranteed inside).
        _, _, _, _, centers = discretize_room_to_grid(room, dx=0.5, dy=0.5)
        if centers.shape[0] == 0:
            continue
        origin = centers[int(rng.integers(0, centers.shape[0]))]

        vis_poly = compute_visibility_polygon(room, origin)

        out = output_dir / f"{name}.png"
        render_visibility_image(
            room=room, vis_poly=vis_poly, origin=origin,
            output_path=out, bounds=bounds,
        )
        paths.append(out)
        print(f"Saved {out}")

    return paths


if __name__ == "__main__":
    save_visibility_images()
