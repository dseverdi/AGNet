import json
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

import colors


Point = np.ndarray


def polygon_to_edges(vertices: np.ndarray) -> np.ndarray:
    """Convert polygon vertices [N, 2] to wall segments [N, 2, 2]."""
    n = vertices.shape[0]
    edges = np.zeros((n, 2, 2), dtype=float)
    edges[:, 0, :] = vertices
    edges[:, 1, :] = np.roll(vertices, shift=-1, axis=0)
    return edges


def _cross(a: Point, b: Point, c: Point) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a: Point, b: Point, p: Point, eps: float = 1e-9) -> bool:
    return (
        min(a[0], b[0]) - eps <= p[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= p[1] <= max(a[1], b[1]) + eps
    )


def _point_segment_distance(p: Point, a: Point, b: Point) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _segment_segment_distance(a: Point, b: Point, c: Point, d: Point) -> float:
    if segments_intersect(a, b, c, d):
        return 0.0
    return min(
        _point_segment_distance(a, c, d),
        _point_segment_distance(b, c, d),
        _point_segment_distance(c, a, b),
        _point_segment_distance(d, a, b),
    )


def segments_intersect(a: Point, b: Point, c: Point, d: Point, eps: float = 1e-9) -> bool:
    """Return True if closed segments AB and CD intersect."""
    o1 = _cross(a, b, c)
    o2 = _cross(a, b, d)
    o3 = _cross(c, d, a)
    o4 = _cross(c, d, b)

    if (o1 * o2 < -eps) and (o3 * o4 < -eps):
        return True

    if abs(o1) <= eps and _on_segment(a, b, c, eps):
        return True
    if abs(o2) <= eps and _on_segment(a, b, d, eps):
        return True
    if abs(o3) <= eps and _on_segment(c, d, a, eps):
        return True
    if abs(o4) <= eps and _on_segment(c, d, b, eps):
        return True

    return False


def is_simple_polygon(vertices: np.ndarray, eps: float = 1e-9) -> bool:
    """Check that polygon has no self-intersections between non-adjacent walls."""
    n = vertices.shape[0]
    if n < 3:
        return False

    edges = polygon_to_edges(vertices)

    for i in range(n):
        a, b = edges[i, 0], edges[i, 1]
        for j in range(i + 1, n):
            if j == i:
                continue
            if j == (i + 1) % n:
                continue
            if i == (j + 1) % n:
                continue

            c, d = edges[j, 0], edges[j, 1]
            if segments_intersect(a, b, c, d, eps=eps):
                return False

    return True


def min_non_adjacent_wall_distance(vertices: np.ndarray) -> float:
    """Return minimum distance between non-adjacent polygon edges."""
    edges = polygon_to_edges(vertices)
    n = edges.shape[0]
    min_d = np.inf

    for i in range(n):
        a, b = edges[i, 0], edges[i, 1]
        for j in range(i + 1, n):
            if j == (i + 1) % n or i == (j + 1) % n:
                continue
            c, d = edges[j, 0], edges[j, 1]
            d_ij = _segment_segment_distance(a, b, c, d)
            if d_ij < min_d:
                min_d = d_ij

    return float(min_d)


def _polygon_area(vertices: np.ndarray) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def count_reflex_vertices(vertices: np.ndarray, eps: float = 1e-9) -> int:
    """Count reflex (concave) vertices of a simple polygon."""
    n = vertices.shape[0]
    if n < 3:
        return 0

    area = _polygon_area(vertices)
    orientation = 1.0 if area >= 0 else -1.0
    reflex = 0

    for i in range(n):
        prev = vertices[i - 1]
        curr = vertices[i]
        nxt = vertices[(i + 1) % n]

        v1 = curr - prev
        v2 = nxt - curr
        turn = v1[0] * v2[1] - v1[1] * v2[0]

        if orientation * turn < -eps:
            reflex += 1

    return reflex


def _dedupe_consecutive(points: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    out = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(p - out[-1]) > eps:
            out.append(p)
    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= eps:
        out.pop()
    return np.array(out, dtype=float)


def _simplify_polygon_vertices(vertices: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Remove duplicate/collinear vertices while preserving polygon shape."""
    if vertices.shape[0] < 4:
        return _dedupe_consecutive(vertices, eps=eps)

    v = _dedupe_consecutive(vertices, eps=eps)
    changed = True
    while changed and v.shape[0] >= 4:
        changed = False
        keep = np.ones((v.shape[0],), dtype=bool)
        n = v.shape[0]
        for i in range(n):
            prev = v[(i - 1) % n]
            curr = v[i]
            nxt = v[(i + 1) % n]
            # Remove nearly-collinear middle points.
            if abs(_cross(prev, curr, nxt)) <= eps:
                keep[i] = False
                changed = True
        if np.any(keep):
            v = v[keep]
        else:
            break

    return _dedupe_consecutive(v, eps=eps)


def _ensure_ccw(vertices: np.ndarray) -> np.ndarray:
    """Return vertices in CCW order."""
    return vertices if _polygon_area(vertices) >= 0 else vertices[::-1].copy()


def _line_intersection(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Intersection of infinite lines p1-p2 and q1-q2 (assumes non-parallel)."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return p2.copy()
    det1 = x1 * y2 - y1 * x2
    det2 = x3 * y4 - y3 * x4
    px = (det1 * (x3 - x4) - (x1 - x2) * det2) / den
    py = (det1 * (y3 - y4) - (y1 - y2) * det2) / den
    return np.array([px, py], dtype=float)


def _convex_polygon_intersection(subject: np.ndarray, clip_poly: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clipping of a convex subject polygon by a convex clip polygon."""
    out = _ensure_ccw(subject)
    clip = _ensure_ccw(clip_poly)

    def inside(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
        return _cross(a, b, p) >= -1e-9

    for i in range(clip.shape[0]):
        cp1 = clip[i]
        cp2 = clip[(i + 1) % clip.shape[0]]
        if out.shape[0] == 0:
            break

        inp = out
        new_out = []
        s = inp[-1]
        for e in inp:
            if inside(e, cp1, cp2):
                if not inside(s, cp1, cp2):
                    new_out.append(_line_intersection(s, e, cp1, cp2))
                new_out.append(e)
            elif inside(s, cp1, cp2):
                new_out.append(_line_intersection(s, e, cp1, cp2))
            s = e

        if len(new_out) == 0:
            out = np.zeros((0, 2), dtype=float)
        else:
            out = _dedupe_consecutive(np.array(new_out, dtype=float))

    return out


def _rotated_rectangle_vertices(center: np.ndarray, width: float, height: float, angle: float) -> np.ndarray:
    """Create CCW vertices of a rotated rectangle."""
    hw, hh = 0.5 * width, 0.5 * height
    local = np.array([
        [-hw, -hh],
        [hw, -hh],
        [hw, hh],
        [-hw, hh],
    ], dtype=float)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]], dtype=float)
    verts = local @ R.T + center
    return _ensure_ccw(verts)


def generate_rotated_rectangle_intersection_room(
    bounds: Tuple[float, float, float, float] = (-15.0, 15.0, -15.0, 15.0),
    rng: np.random.Generator | None = None,
    n_rectangles: int = 4,
    max_tries: int = 250,
) -> np.ndarray:
    """Generate one room as overlap/union of multiple connected rotated rectangles."""
    if n_rectangles < 2:
        raise ValueError("n_rectangles must be >= 2")
    if rng is None:
        rng = np.random.default_rng()

    l, r, b, t = bounds
    span_x = r - l
    span_y = t - b
    margin_x = 0.02 * span_x
    margin_y = 0.02 * span_y

    # Raster grid used to build overlap-union mask and extract a boundary contour.
    step = 0.12
    xs = np.arange(l, r + step, step)
    ys = np.arange(b, t + step, step)
    X, Y = np.meshgrid(xs, ys)

    for _ in range(max_tries):
        # Build a chain of connected rectangle centers so overlaps are likely connected.
        centers = []
        c0 = np.array(
            [
                rng.uniform(l + 0.3 * span_x, r - 0.3 * span_x),
                rng.uniform(b + 0.3 * span_y, t - 0.3 * span_y),
            ],
            dtype=float,
        )
        centers.append(c0)

        for _k in range(1, n_rectangles):
            prev = centers[-1]
            ang_step = rng.uniform(0.0, 2.0 * np.pi)
            step_len = rng.uniform(0.08 * min(span_x, span_y), 0.18 * min(span_x, span_y))
            cand = prev + step_len * np.array([np.cos(ang_step), np.sin(ang_step)], dtype=float)
            cand[0] = np.clip(cand[0], l + 0.2 * span_x, r - 0.2 * span_x)
            cand[1] = np.clip(cand[1], b + 0.2 * span_y, t - 0.2 * span_y)
            centers.append(cand)

        mask = np.zeros_like(X, dtype=bool)
        for c in centers:
            w = rng.uniform(0.22 * span_x, 0.46 * span_x)
            h = rng.uniform(0.22 * span_y, 0.46 * span_y)
            ang = rng.uniform(0.0, np.pi)
            ca, sa = np.cos(ang), np.sin(ang)

            dx = X - c[0]
            dy = Y - c[1]
            x_local = ca * dx + sa * dy
            y_local = -sa * dx + ca * dy
            in_rect = (np.abs(x_local) <= 0.5 * w) & (np.abs(y_local) <= 0.5 * h)
            mask |= in_rect

        if np.count_nonzero(mask) < 50:
            continue

        fig, ax = plt.subplots(figsize=(3, 3), dpi=60)
        contour = ax.contour(X, Y, mask.astype(float), levels=[0.5])
        segs = contour.allsegs[0] if len(contour.allsegs) > 0 else []
        plt.close(fig)

        if len(segs) == 0:
            continue

        best_poly = None
        best_area = -np.inf
        for seg in segs:
            v = _dedupe_consecutive(np.array(seg, dtype=float))
            if v.shape[0] < 3:
                continue
            if np.linalg.norm(v[0] - v[-1]) <= 1e-3:
                v = v[:-1]
            if v.shape[0] < 3:
                continue
            a = abs(_polygon_area(v))
            if a > best_area:
                best_area = a
                best_poly = v

        if best_poly is None or best_area < 1e-2:
            continue

        poly = _simplify_polygon_vertices(_ensure_ccw(best_poly), eps=2e-3)
        if not is_simple_polygon(poly):
            continue

        return _ensure_ccw(poly)

    raise RuntimeError("Could not generate rotated-rectangle overlap room")


def _point_on_boundary(point: np.ndarray, vertices: np.ndarray, eps: float = 1e-9) -> bool:
    """Return True if a point lies on any polygon edge."""
    edges = polygon_to_edges(vertices)
    p = np.asarray(point, dtype=float)
    for a, b in edges:
        if abs(_cross(a, b, p)) <= eps and _on_segment(a, b, p, eps=eps):
            return True
    return False


def point_in_polygon(point: np.ndarray, vertices: np.ndarray, include_boundary: bool = True) -> bool:
    """Ray-casting point-in-polygon test for a simple polygon."""
    p = np.asarray(point, dtype=float)
    if include_boundary and _point_on_boundary(p, vertices):
        return True

    x, y = float(p[0]), float(p[1])
    inside = False
    n = vertices.shape[0]
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]

        # Edge crosses horizontal ray to the right of point.
        intersects = ((y1 > y) != (y2 > y))
        if intersects:
            x_inter = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x_inter > x:
                inside = not inside

    return inside


def discretize_room_to_grid(
    room: np.ndarray,
    dx: float = 0.1,
    dy: float = 0.1,
    padding: float = 0.0,
    include_boundary: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Discretize one room polygon into grid cells selected by polygon boundaries.

    Returns north/south/east/west cell sides in the same format used in
    robotics_mapping, plus centers for selected cells.
    """
    if dx <= 0 or dy <= 0:
        raise ValueError("dx and dy must be > 0")
    if room.ndim != 2 or room.shape[1] != 2 or room.shape[0] < 3:
        raise ValueError("room must be an array of shape [N, 2] with N >= 3")

    l = float(np.min(room[:, 0])) - padding
    r = float(np.max(room[:, 0])) + padding
    b = float(np.min(room[:, 1])) - padding
    t = float(np.max(room[:, 1])) + padding

    points_x_from = np.arange(l, r, dx)
    points_x_to = points_x_from + dx
    points_y_from = np.arange(b, t, dy)
    points_y_to = points_y_from + dy

    n_x = points_x_from.shape[0]
    n_y = points_y_from.shape[0]
    n_cells = n_x * n_y

    centers = np.zeros((n_cells, 2), dtype=float)
    centers[:, 0] = np.tile((points_x_from + points_x_to) / 2.0, n_y)
    centers[:, 1] = np.repeat((points_y_from + points_y_to) / 2.0, n_x)

    inside = np.array(
        [point_in_polygon(c, room, include_boundary=include_boundary) for c in centers],
        dtype=bool,
    )

    grid_north = np.zeros((n_cells, 2, 2), dtype=float)
    grid_north[:, 0, 0] = np.tile(points_x_from, n_y)
    grid_north[:, 0, 1] = np.repeat(points_y_to, n_x)
    grid_north[:, 1, 0] = np.tile(points_x_to, n_y)
    grid_north[:, 1, 1] = np.repeat(points_y_to, n_x)

    grid_south = np.zeros((n_cells, 2, 2), dtype=float)
    grid_south[:, 0, 0] = np.tile(points_x_from, n_y)
    grid_south[:, 0, 1] = np.repeat(points_y_from, n_x)
    grid_south[:, 1, 0] = np.tile(points_x_to, n_y)
    grid_south[:, 1, 1] = np.repeat(points_y_from, n_x)

    grid_east = np.zeros((n_cells, 2, 2), dtype=float)
    grid_east[:, 0, 0] = np.tile(points_x_to, n_y)
    grid_east[:, 0, 1] = np.repeat(points_y_from, n_x)
    grid_east[:, 1, 0] = np.tile(points_x_to, n_y)
    grid_east[:, 1, 1] = np.repeat(points_y_to, n_x)

    grid_west = np.zeros((n_cells, 2, 2), dtype=float)
    grid_west[:, 0, 0] = np.tile(points_x_from, n_y)
    grid_west[:, 0, 1] = np.repeat(points_y_from, n_x)
    grid_west[:, 1, 0] = np.tile(points_x_from, n_y)
    grid_west[:, 1, 1] = np.repeat(points_y_to, n_x)

    return (
        grid_north[inside],
        grid_south[inside],
        grid_east[inside],
        grid_west[inside],
        centers[inside],
    )


def generate_agp_hard_room_vertices(
    bounds: Tuple[float, float, float, float] = (-3.0, 3.0, -3.0, 3.0),
    rng: np.random.Generator | None = None,
    n_pockets: int = 4,
    min_reflex_vertices: int = 4,
    min_passage_width: float = 1.0,
    max_tries: int = 120,
) -> np.ndarray:
    """
    Generate a pocketed non-convex room polygon designed to be harder for AGP.
    The shape remains simple (no wall intersections) and has multiple reflex vertices.
    """
    if n_pockets < 2:
        raise ValueError("n_pockets must be >= 2")
    if min_passage_width <= 0:
        raise ValueError("min_passage_width must be > 0")

    if rng is None:
        rng = np.random.default_rng()

    l, r, b, t = bounds
    width = r - l
    height = t - b

    if width < 2.0 * min_passage_width or height < 2.0 * min_passage_width:
        raise ValueError("Bounds are too small for the requested min_passage_width")

    for _ in range(max_tries):
        # Controlled template-based construction for reliable geometric feasibility.
        margin_x = 0.12 * width
        margin_y = 0.12 * height

        x_left = l + margin_x
        x_right = r - margin_x
        y_bottom = b + margin_y
        y_top = t - margin_y

        span_x = x_right - x_left
        y_span = y_top - y_bottom
        if span_x <= min_passage_width or y_span <= min_passage_width:
            continue

        xs = np.linspace(x_left + 0.15 * span_x, x_right - 0.15 * span_x, n_pockets)
        spacing = (xs[1] - xs[0]) if n_pockets > 1 else 0.3 * span_x

        # Choose notch width so neighboring notches leave at least min_passage_width.
        notch_width = min(0.38 * spacing, spacing - min_passage_width)
        if notch_width <= 1e-9:
            continue
        half = 0.5 * notch_width

        desired_gap = min_passage_width + 0.25 * min_passage_width
        if desired_gap >= y_span:
            continue

        y_center = 0.5 * (y_bottom + y_top)
        y_bot_notch = y_center - 0.5 * desired_gap
        y_top_notch = y_center + 0.5 * desired_gap

        if y_bot_notch <= y_bottom + 0.08 * y_span:
            continue
        if y_top_notch >= y_top - 0.08 * y_span:
            continue

        pts = []

        # Bottom chain left -> right with upward pockets on odd indices.
        pts.append([x_left, y_bottom])
        for i, x in enumerate(xs):
            x1 = max(x_left, x - half)
            x2 = min(x_right, x + half)
            pts.append([x1, y_bottom])
            if i % 2 == 1:
                pts.extend([[x1, y_bot_notch], [x2, y_bot_notch]])
            pts.append([x2, y_bottom])
        pts.append([x_right, y_bottom])

        # Right side up.
        pts.append([x_right, y_top])

        # Top chain right -> left with downward pockets on even indices.
        for i in range(n_pockets - 1, -1, -1):
            x = xs[i]
            x1 = max(x_left, x - half)
            x2 = min(x_right, x + half)
            pts.append([x2, y_top])
            if i % 2 == 0:
                pts.extend([[x2, y_top_notch], [x1, y_top_notch]])
            pts.append([x1, y_top])
        pts.append([x_left, y_top])

        vertices = _simplify_polygon_vertices(np.array(pts, dtype=float))

        if vertices.shape[0] < 8:
            continue
        if not is_simple_polygon(vertices):
            continue

        reflex = count_reflex_vertices(vertices)
        if reflex < min_reflex_vertices:
            continue

        # Hall widths are enforced by construction via notch spacing and center-gap.
        return vertices

    raise RuntimeError("Could not generate AGP-hard room within max_tries")


def generate_room_vertices(
    n_vertices: int,
    bounds: Tuple[float, float, float, float] = (-3.0, 3.0, -3.0, 3.0),
    rng: np.random.Generator | None = None,
    max_tries: int = 200,
) -> np.ndarray:
    """
    Generate one room boundary polygon (single closed polygon, no inner objects).
    Ensures non-self-intersecting walls.
    """
    if n_vertices < 3:
        raise ValueError("n_vertices must be >= 3")

    if rng is None:
        rng = np.random.default_rng()

    l, r, b, t = bounds
    width = r - l
    height = t - b
    radius_scale = 0.35 * min(width, height)

    for _ in range(max_tries):
        center = np.array(
            [
                rng.uniform(l + 0.2 * width, r - 0.2 * width),
                rng.uniform(b + 0.2 * height, t - 0.2 * height),
            ],
            dtype=float,
        )

        angles = np.sort(rng.uniform(0.0, 2.0 * np.pi, n_vertices))
        radii = rng.uniform(0.5 * radius_scale, 1.0 * radius_scale, n_vertices)

        vertices = np.stack(
            [center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)],
            axis=1,
        )

        if abs(_polygon_area(vertices)) < 1e-4:
            continue

        if np.any(vertices[:, 0] < l) or np.any(vertices[:, 0] > r):
            continue
        if np.any(vertices[:, 1] < b) or np.any(vertices[:, 1] > t):
            continue

        if is_simple_polygon(vertices):
            return vertices

    raise RuntimeError("Could not generate a simple polygon room within max_tries")


def generate_room_dataset(
    n_rooms: int,
    min_vertices: int = 4,
    max_vertices: int = 10,
    bounds: Tuple[float, float, float, float] = (-15.0, 15.0, -15.0, 15.0),
    seed: int | None = 0,
    hard_mode: bool = False,
    rectangle_intersection_mode: bool = False,
    narrow_passages: bool = False,
    min_reflex_vertices: int = 4,
    min_passage_width: float = 1.0,
    center_at_zero: bool = True,
) -> List[np.ndarray]:
    """
    Generate a dataset of simple room polygons (no holes/inner objects).

    If hard_mode=True, rooms are pocketed non-convex polygons intended to be
    harder AGP instances that typically require multiple guards.

    In hard mode, min_passage_width enforces a minimum corridor/opening width.
    If center_at_zero=True, each generated room is translated so its centroid is at (0, 0).

    If rectangle_intersection_mode=True, each room is generated as a polygon
    produced from the union of multiple connected rectangles.
    If *narrow_passages* is also True, rooms are connected by narrow corridors
    (AGP-hard: typically needs one guard per room).
    """
    if min_vertices < 3 or max_vertices < min_vertices:
        raise ValueError("Invalid vertex range")
    if hard_mode and rectangle_intersection_mode:
        raise ValueError("hard_mode and rectangle_intersection_mode are mutually exclusive")

    rng = np.random.default_rng(seed)
    rooms: List[np.ndarray] = []

    l, r, b, t = bounds
    cx, cy = 0.5 * (l + r), 0.5 * (b + t)
    half_w, half_h = 0.5 * (r - l), 0.5 * (t - b)

    for _ in range(n_rooms):
        if rectangle_intersection_mode:
            n_rectangles = int(rng.integers(3, 5))
            rooms.append(
                generate_rectangle_union_room(
                    bounds=bounds,
                    rng=rng,
                    n_rectangles=n_rectangles,
                    max_tries=120,
                    narrow_passages=narrow_passages,
                )
            )
        elif hard_mode:
            # Randomise room size independently per axis.
            # Clamp minimum scale so the effective span is always >= 4 * min_passage_width.
            sw_min = min(0.9, max(0.4, (4.0 * min_passage_width) / (2.0 * half_w)))
            sh_min = min(0.9, max(0.4, (4.0 * min_passage_width) / (2.0 * half_h)))
            # Triangular distribution: mode near 0.85, max 1.0 — biased large but with
            # genuine spread (rooms can range from ~40% to 100% of the available bounds).
            mode_w = max(sw_min, 0.85)
            mode_h = max(sh_min, 0.85)
            sw = float(rng.triangular(sw_min, mode_w, 1.0))
            sh = float(rng.triangular(sh_min, mode_h, 1.0))
            room_bounds = (
                cx - half_w * sw, cx + half_w * sw,
                cy - half_h * sh, cy + half_h * sh,
            )
            rl, rr, _, _ = room_bounds
            usable_span = 0.76 * (rr - rl)
            max_pockets = int(np.floor(usable_span / min_passage_width) + 1)
            max_pockets = max(3, min(max_pockets, 4))
            min_pockets = 3 if max_pockets >= 3 else 2
            created = False
            for _attempt in range(25):
                n_pockets = int(rng.integers(min_pockets, max_pockets + 1))
                try:
                    room = generate_agp_hard_room_vertices(
                        bounds=room_bounds,
                        rng=rng,
                        n_pockets=n_pockets,
                        min_reflex_vertices=min_reflex_vertices,
                        min_passage_width=min_passage_width,
                        max_tries=120,
                    )
                    rooms.append(room)
                    created = True
                    break
                except RuntimeError:
                    continue
            if not created:
                raise RuntimeError("Could not generate hard-mode room with given constraints")
        else:
            n_vertices = int(rng.integers(min_vertices, max_vertices + 1))
            rooms.append(generate_room_vertices(n_vertices=n_vertices, bounds=bounds, rng=rng))

    if center_at_zero:
        centered_rooms: List[np.ndarray] = []
        for room in rooms:
            centroid = np.mean(room, axis=0, keepdims=True)
            centered = room - centroid
            centered_rooms.append(centered)
        rooms = centered_rooms

    # Apply a random rotation around the origin to each room
    # (skip for rectangle_intersection_mode to keep rooms axis-aligned).
    if not rectangle_intersection_mode:
        rotated_rooms: List[np.ndarray] = []
        for room in rooms:
            angle = rng.uniform(0.0, 2.0 * np.pi)
            c, s = np.cos(angle), np.sin(angle)
            R = np.array([[c, -s], [s, c]])
            rotated_rooms.append(room @ R.T)
        rooms = rotated_rooms

    return rooms


def save_room_dataset(path: str | Path, rooms: Sequence[np.ndarray]) -> None:
    """Save rooms as JSON list of vertex lists."""
    payload = {"rooms": [room.tolist() for room in rooms]}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_room_dataset(path: str | Path) -> List[np.ndarray]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [np.array(room, dtype=float) for room in payload["rooms"]]


def rooms_to_world_segments(rooms: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Convert each room polygon [N,2] to world-style walls [N,2,2]."""
    return [polygon_to_edges(room) for room in rooms]
def render_room_image(
    room: np.ndarray,
    output_path: str | Path,
    bounds: Tuple[float, float, float, float] = (-15.0, 15.0, -15.0, 15.0),
    dpi: int = 160,
) -> None:
    """Render one room polygon image with blue walls and translucent blue interior."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    l, r, b, t = bounds
    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    patch = Polygon(
        room,
        closed=True,
        fill=True,
        facecolor=colors.paleblue,
        edgecolor=colors.blue,
        linewidth=2.5,
        alpha=0.3,
    )
    ax.add_patch(patch)
    ax.plot(
        np.append(room[:, 0], room[0, 0]),
        np.append(room[:, 1], room[0, 1]),
        color=colors.blue,
        linewidth=2.5,
    )
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


def render_discretized_room_image(
    room: np.ndarray,
    output_path: str | Path,
    dx: float = 0.5,
    dy: float = 0.5,
    bounds: Tuple[float, float, float, float] = (-15.0, 15.0, -15.0, 15.0),
    dpi: int = 160,
) -> None:
    """Render a room with selected discretized cells overlaid."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _, _, _, _, centers = discretize_room_to_grid(room, dx=dx, dy=dy)

    l, r, b, t = bounds
    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)

    for c in centers:
        x0 = c[0] - dx / 2.0
        y0 = c[1] - dy / 2.0
        ax.add_patch(Rectangle(
            (x0, y0), dx, dy,
            facecolor=colors.paleblue, edgecolor=colors.orange,
            linewidth=0.25, alpha=1.0, zorder=0,
        ))

    ax.plot(
        np.append(room[:, 0], room[0, 0]),
        np.append(room[:, 1], room[0, 1]),
        color=colors.blue, linewidth=0.7, linestyle="--", zorder=2,
    )
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




def _subtract_intervals(lo: float, hi: float, removes: list) -> list:
    """Subtract a list of (a, b) intervals from [lo, hi], returning remaining sub-intervals."""
    if not removes:
        return [(lo, hi)] if hi > lo + 1e-12 else []
    removes_s = sorted(removes)
    merged: list = []
    for a, b in removes_s:
        if merged and a <= merged[-1][1] + 1e-12:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    result = []
    cur = lo
    for a, b in merged:
        if a > cur + 1e-12:
            result.append((cur, min(a, hi)))
        cur = max(cur, b)
        if cur >= hi - 1e-12:
            break
    if cur < hi - 1e-12:
        result.append((cur, hi))
    return [(a, b) for a, b in result if b - a > 1e-12]


def _rect_union_polygon(rects: np.ndarray) -> np.ndarray:
    """
    Compute the exact union polygon of axis-aligned rectangles.

    rects: [N, 4] array of [x, y, width, height].
    Returns: CCW polygon [M, 2] with only true corner vertices
             (no collinear intermediate points).
    """
    n = rects.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=float)

    x0s = rects[:, 0]
    y0s = rects[:, 1]
    x1s = rects[:, 0] + rects[:, 2]
    y1s = rects[:, 1] + rects[:, 3]

    # Coordinate compression: collect all unique x and y from rectangle edges.
    xs = np.unique(np.concatenate([x0s, x1s]))
    ys = np.unique(np.concatenate([y0s, y1s]))
    nx = len(xs) - 1
    ny = len(ys) - 1
    if nx <= 0 or ny <= 0:
        return np.zeros((0, 2), dtype=float)

    # Mark grid cells occupied by at least one rectangle.
    grid = np.zeros((nx, ny), dtype=bool)
    for i in range(n):
        ix0 = int(np.searchsorted(xs, x0s[i]))
        ix1 = int(np.searchsorted(xs, x1s[i]))
        iy0 = int(np.searchsorted(ys, y0s[i]))
        iy1 = int(np.searchsorted(ys, y1s[i]))
        grid[ix0:ix1, iy0:iy1] = True

    # Extract directed boundary edges (CCW around occupied region).
    # Each edge goes from start to end along an axis-aligned direction.
    edges: list = []
    for i in range(nx):
        for j in range(ny):
            if not grid[i, j]:
                continue
            if j == 0 or not grid[i, j - 1]:          # bottom
                edges.append((xs[i], ys[j], xs[i + 1], ys[j]))
            if j == ny - 1 or not grid[i, j + 1]:     # top
                edges.append((xs[i + 1], ys[j + 1], xs[i], ys[j + 1]))
            if i == 0 or not grid[i - 1, j]:           # left
                edges.append((xs[i], ys[j + 1], xs[i], ys[j]))
            if i == nx - 1 or not grid[i + 1, j]:      # right
                edges.append((xs[i + 1], ys[j], xs[i + 1], ys[j + 1]))

    if not edges:
        return np.zeros((0, 2), dtype=float)

    # Build start-point → edge lookup and chain into closed loops.
    prec = 12
    start_map: dict = {}
    for e in edges:
        k = (round(e[0], prec), round(e[1], prec))
        start_map.setdefault(k, []).append(e)

    polygons: list = []
    used_edges: set = set()

    for e0 in edges:
        eid0 = id(e0)
        if eid0 in used_edges:
            continue
        k0 = (round(e0[0], prec), round(e0[1], prec))
        pts: list = []
        e = e0
        for _ in range(len(edges) + 2):
            eid = id(e)
            if eid in used_edges:
                break
            used_edges.add(eid)
            pts.append((e[0], e[1]))
            ek = (round(e[2], prec), round(e[3], prec))
            if ek == k0 and len(pts) >= 3:
                break
            nxt = None
            for c in start_map.get(ek, []):
                if id(c) not in used_edges:
                    nxt = c
                    break
            if nxt is None:
                break
            e = nxt
        if len(pts) >= 4:
            polygons.append(np.array(pts, dtype=float))

    if not polygons:
        return np.zeros((0, 2), dtype=float)
    best = max(polygons, key=lambda p: abs(_polygon_area(p)))
    return _simplify_polygon_vertices(_ensure_ccw(best))


def _rescale_rects_to_bounds(
    arr: np.ndarray,
    bounds: Tuple[float, float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Rescale rectangles [N,4] into a random sub-region of *bounds*."""
    l, r, b, t = bounds
    span_x, span_y = r - l, t - b

    all_x0 = arr[:, 0].min()
    all_y0 = arr[:, 1].min()
    all_x1 = (arr[:, 0] + arr[:, 2]).max()
    all_y1 = (arr[:, 1] + arr[:, 3]).max()

    fill_x = float(rng.uniform(0.50, 1.0))
    fill_y = float(rng.uniform(0.50, 1.0))
    target_w = span_x * fill_x
    target_h = span_y * fill_y

    ox = l + float(rng.uniform(0.0, span_x - target_w))
    oy = b + float(rng.uniform(0.0, span_y - target_h))

    sx = target_w / max(all_x1 - all_x0, 1e-9)
    sy = target_h / max(all_y1 - all_y0, 1e-9)

    arr[:, 0] = ox + (arr[:, 0] - all_x0) * sx
    arr[:, 1] = oy + (arr[:, 1] - all_y0) * sy
    arr[:, 2] *= sx
    arr[:, 3] *= sy
    return arr


def generate_overlapping_rectangles_scene(
    bounds: Tuple[float, float, float, float] = (-15.0, 15.0, -15.0, 15.0),
    n_rectangles: int = 4,
    rng: np.random.Generator | None = None,
    overlap_fraction: float = 0.10,
    min_bump_fraction: float = 0.15,
    narrow_passages: bool = False,
) -> np.ndarray:
    """
    Generate connected axis-aligned rectangles.

    If *narrow_passages* is False (default), rectangles overlap broadly
    (easy for AGP).  If True, large rooms are connected by narrow corridors,
    making the polygon harder for AGP — typically one guard per room.

    Returns: [M, 4] array of [x, y, width, height].
    """
    if rng is None:
        rng = np.random.default_rng()

    if narrow_passages:
        return _generate_rooms_with_corridors(
            bounds=bounds, n_rooms=n_rectangles, rng=rng,
        )

    # --- Easy mode: corner-based attachment ---
    # Build in normalised [0, 1] space, rescale at the end.
    # Each rectangle has 4 corners (TR, TL, BL, BR).  Once a corner is used
    # for attachment it cannot be reused, preventing rectangles from piling
    # onto the same spot.
    w0 = rng.uniform(0.35, 0.65)
    h0 = rng.uniform(0.35, 0.65)
    x0 = rng.uniform(0.0, 1.0 - w0)
    y0 = rng.uniform(0.0, 1.0 - h0)

    rects = [[x0, y0, w0, h0]]
    # Available corners per rectangle: 0=TR, 1=TL, 2=BL, 3=BR
    available_corners: list[list[int]] = [[0, 1, 2, 3]]

    for _ in range(1, n_rectangles):
        # Collect all (rect_index, corner) pairs still available.
        candidates = [
            (ri, c) for ri, corners in enumerate(available_corners) for c in corners
        ]
        if not candidates:
            break
        ri, corner = candidates[int(rng.integers(0, len(candidates)))]
        rx, ry, rw, rh = rects[ri]

        nw = rng.uniform(0.25, 0.60)
        nh = rng.uniform(0.25, 0.60)

        # Sample independent random overlap fractions for x and y.
        of_x = rng.uniform(overlap_fraction, 2.0 * overlap_fraction)
        of_y = rng.uniform(overlap_fraction, 2.0 * overlap_fraction)
        if corner == 0:    # top-right  → new rect extends right & up
            nx = rx + rw - of_x * nw
            ny = ry + rh - of_y * nh
        elif corner == 1:  # top-left   → new rect extends left & up
            nx = rx - nw + of_x * nw
            ny = ry + rh - of_y * nh
        elif corner == 2:  # bottom-left → new rect extends left & down
            nx = rx - nw + of_x * nw
            ny = ry - nh + of_y * nh
        else:              # bottom-right → new rect extends right & down
            nx = rx + rw - of_x * nw
            ny = ry - nh + of_y * nh

        # Mark the parent corner as used.
        available_corners[ri].remove(corner)

        # The new rect's opposite corner is also effectively used (it's where
        # the overlap is), so remove it from the new rect's available set.
        opposite = {0: 2, 1: 3, 2: 0, 3: 1}[corner]
        new_corners = [c for c in [0, 1, 2, 3] if c != opposite]

        rects.append([nx, ny, nw, nh])
        available_corners.append(new_corners)

    arr = np.array(rects, dtype=float)
    return _rescale_rects_to_bounds(arr, bounds, rng)


def _generate_rooms_with_corridors(
    bounds: Tuple[float, float, float, float],
    n_rooms: int,
    rng: np.random.Generator,
    corridor_width: float = 0.04,
) -> np.ndarray:
    """Generate large rooms connected by narrow corridors in normalised space,
    then rescale into *bounds*.  Each room needs its own guard."""

    # First room.
    w0 = rng.uniform(0.30, 0.50)
    h0 = rng.uniform(0.30, 0.50)
    rooms: list = [[0.0, 0.0, w0, h0]]
    all_rects: list = [[0.0, 0.0, w0, h0]]

    for _ in range(1, n_rooms):
        # Attach to a random existing room.
        idx = int(rng.integers(0, len(rooms)))
        rx, ry, rw, rh = rooms[idx]

        nw = rng.uniform(0.25, 0.45)
        nh = rng.uniform(0.25, 0.45)

        side = int(rng.integers(0, 4))
        gap = rng.uniform(0.06, 0.16)
        cw = corridor_width

        if side == 0:  # corridor goes right
            corr_y = ry + rng.uniform(0.15 * rh, rh - 0.15 * rh - cw)
            all_rects.append([rx + rw, corr_y, gap, cw])
            nx = rx + rw + gap
            ny = corr_y - rng.uniform(0.2 * nh, 0.8 * nh - cw)
        elif side == 1:  # corridor goes up
            corr_x = rx + rng.uniform(0.15 * rw, rw - 0.15 * rw - cw)
            all_rects.append([corr_x, ry + rh, cw, gap])
            ny = ry + rh + gap
            nx = corr_x - rng.uniform(0.2 * nw, 0.8 * nw - cw)
        elif side == 2:  # corridor goes left
            corr_y = ry + rng.uniform(0.15 * rh, rh - 0.15 * rh - cw)
            all_rects.append([rx - gap, corr_y, gap, cw])
            nx = rx - gap - nw
            ny = corr_y - rng.uniform(0.2 * nh, 0.8 * nh - cw)
        else:  # corridor goes down
            corr_x = rx + rng.uniform(0.15 * rw, rw - 0.15 * rw - cw)
            all_rects.append([corr_x, ry - gap, cw, gap])
            ny = ry - gap - nh
            nx = corr_x - rng.uniform(0.2 * nw, 0.8 * nw - cw)

        rooms.append([nx, ny, nw, nh])
        all_rects.append([nx, ny, nw, nh])

    arr = np.array(all_rects, dtype=float)
    return _rescale_rects_to_bounds(arr, bounds, rng)


def generate_rectangle_union_room(
    bounds: Tuple[float, float, float, float] = (-15.0, 15.0, -15.0, 15.0),
    rng: np.random.Generator | None = None,
    n_rectangles: int = 4,
    max_tries: int = 120,
    narrow_passages: bool = False,
) -> np.ndarray:
    """Generate one room polygon as the exact union of connected axis-aligned rectangles."""
    if n_rectangles < 2:
        raise ValueError("n_rectangles must be >= 2")
    if rng is None:
        rng = np.random.default_rng()

    l, r, b, t = bounds
    for _ in range(max_tries):
        rects = generate_overlapping_rectangles_scene(
            bounds=bounds, n_rectangles=n_rectangles, rng=rng,
            narrow_passages=narrow_passages,
        )
        poly = _rect_union_polygon(rects)
        if poly.shape[0] < 4:
            continue
        if (np.any(poly[:, 0] < l) or np.any(poly[:, 0] > r) or
                np.any(poly[:, 1] < b) or np.any(poly[:, 1] > t)):
            continue
        if not is_simple_polygon(poly):
            continue
        return poly

    raise RuntimeError("Could not generate rectangle-union room")
def save_discretization_images(
    rooms: Sequence[np.ndarray],
    output_dir: str | Path = "data/images/dataset_1",
    prefix: str | None = None,
    dx: float = 0.5,
    dy: float = 0.5,
    bounds: Tuple[float, float, float, float] = (-15.0, 15.0, -15.0, 15.0),
) -> List[Path]:
    """Render and save discretized-room overlays as PNG images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for i, room in enumerate(rooms, start=1):
        out = output_dir / (f"{i}.png" if not prefix else f"{prefix}_{i:02d}.png")
        render_discretized_room_image(room, out, dx=dx, dy=dy, bounds=bounds)
        paths.append(out)
    return paths


_RECT_COLORS = [
    colors.paleblue, colors.cyan, colors.yellow,
    colors.magenta, colors.green, colors.purple, colors.orange,
]


def render_overlapping_rectangles_scene(
    rectangles: np.ndarray,
    output_path: str | Path,
    bounds: Tuple[float, float, float, float] = (-15.0, 15.0, -15.0, 15.0),
    dx: float = 0.5,
    dy: float = 0.5,
    dpi: int = 160,
) -> None:
    """Render one overlapping-rectangles scene with exact polygon boundary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    poly = _rect_union_polygon(rectangles)

    l, r, b, t = bounds
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=dpi)

    # Inner grid cells with paleblue fill and orange edges.
    north, south, east, west, centers = discretize_room_to_grid(poly, dx=dx, dy=dy)
    for c in centers:
        x0 = c[0] - dx / 2.0
        y0 = c[1] - dy / 2.0
        ax.add_patch(Rectangle(
            (x0, y0), dx, dy,
            facecolor=colors.paleblue, edgecolor=colors.orange,
            linewidth=0.25, alpha=1.0, zorder=0,
        ))

    # Dashed blue room border.
    if poly.shape[0] >= 3:
        px = np.append(poly[:, 0], poly[0, 0])
        py = np.append(poly[:, 1], poly[0, 1])
        ax.plot(px, py, color=colors.blue, linewidth=0.7, linestyle="--", zorder=2)

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


def save_overlapping_rectangles_images(
    output_dir: str | Path = "data/images/dataset_2",
    n_images: int = 10,
    bounds: Tuple[float, float, float, float] = (-15.0, 15.0, -15.0, 15.0),
    seed: int | None = 0,
    prefix: str | None = None,
) -> List[Path]:
    """Generate and save multiple Medium-style overlapping rectangle scenes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    paths: List[Path] = []
    for i in range(1, n_images + 1):
        n_rectangles = int(rng.integers(2, 5))
        rects = generate_overlapping_rectangles_scene(
            bounds=bounds,
            n_rectangles=n_rectangles,
            rng=rng,
        )

        out = output_dir / (f"{i}.png" if not prefix else f"{prefix}_{i:02d}.png")
        render_overlapping_rectangles_scene(rects, out, bounds=bounds)
        paths.append(out)

    return paths


def save_room_images(
    rooms: Sequence[np.ndarray],
    output_dir: str | Path,
    prefix: str = "room",
    bounds: Tuple[float, float, float, float] = (-15.0, 15.0, -15.0, 15.0),
) -> List[Path]:
    """Render and save all rooms as PNG images."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for i, room in enumerate(rooms, start=1):
        out = output_dir / f"{prefix}_{i:02d}.png"
        render_room_image(room, out, bounds=bounds)
        paths.append(out)
    return paths


if __name__ == "__main__":
    rooms = generate_room_dataset(n_rooms=50, min_vertices=4, max_vertices=12, seed=42)
    save_room_dataset("./data/rooms_no_inner_objects.json", rooms)
    print(f"Saved {len(rooms)} rooms to ./data/rooms_no_inner_objects.json")

    demo_rooms = generate_room_dataset(
        n_rooms=10,
        min_vertices=4,
        max_vertices=12,
        seed=7,
        hard_mode=True,
        min_reflex_vertices=4,
        min_passage_width=1.0,
        center_at_zero=True,
    )
    image_paths = save_room_images(demo_rooms, ".", prefix="room")
    print(f"Saved {len(image_paths)} images to {Path('.').resolve()}")
