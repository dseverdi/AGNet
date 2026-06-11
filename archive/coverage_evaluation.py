import sys
import faulthandler
faulthandler.enable()

import numpy as np
import skgeom
from concurrent.futures import ThreadPoolExecutor

# Compute visibility polygon for a single guard index
def compute_visibility(vs, arr, poly, eps, edges, i):
    v_prev = edges[(i - 1) % len(edges)].source()
    v = edges[i % len(edges)].source()
    v_next = edges[i % len(edges)].target()

    p = skgeom.Vector2(v, v_prev)
    p = p / np.sqrt(p.squared_length())
    r = skgeom.Vector2(v, v_next)
    r = r / np.sqrt(r.squared_length())

    q = skgeom.Point2(v.x() + eps * (p.x() + r.x()), v.y() + eps * (p.y() + r.y()))
    if poly.oriented_side(q) != skgeom.Sign.POSITIVE:
        q = skgeom.Point2(v.x() - eps * (p.x() + r.x()), v.y() - eps * (p.y() + r.y()))

    face = arr.find(q)
    if face is None or face.is_unbounded():
        return None, i, q
    try:
        vx = vs.compute_visibility(q, face)
        visibility_polygon = skgeom.Polygon([vertex.point() for vertex in vx.vertices])
        return visibility_polygon, None, None
    except RuntimeError:
        return None, i, q

# Merge a list of PolygonSet into one
def merge_polygon_sets(polygon_sets):
    merged = skgeom.PolygonSet()
    for ps in polygon_sets:
        merged = merged.union(ps)
    return merged

# Evaluate coverage without ground-truth (numpy-based)
def evaluate_polygon_visibility_numpy_wo_gt(points: np.ndarray, solution: np.ndarray, name: str) -> float:
    eps = 1e-8
    poly = skgeom.Polygon(points)

    # Build arrangement
    arr = skgeom.arrangement.Arrangement()
    for edge in poly.edges:
        arr.insert(edge)
    vs = skgeom.TriangularExpansionVisibility(arr)

    # Compute visibility polygons in parallel
    edges = list(poly.edges)
    views = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_visibility, vs, arr, poly, eps, edges, idx) for idx in solution]
        for future in futures:
            vis_poly, err_idx, err_q = future.result()
            if vis_poly:
                views.append(skgeom.PolygonSet([vis_poly]))
            else:
                print(f"Warning: visibility failed at guard {err_idx}", file=sys.stderr)

    # Merge partial regions
    merged_parts = []
    with ThreadPoolExecutor() as executor:
        chunk_size = max(1, len(views) // executor._max_workers)
        chunks = [views[i:i + chunk_size] for i in range(0, len(views), chunk_size)]
        merged_parts = [executor.submit(merge_polygon_sets, chunk).result() for chunk in chunks]
    region = merge_polygon_sets(merged_parts)

    # Calculate total visible area
    total_area = 0.0
    poly_area = abs(float(poly.area()))
    for vis in region.polygons:
        outer = abs(float(vis.outer_boundary().area()))
        holes = sum(abs(float(h.area())) for h in vis.holes)
        total_area += outer - holes

    return total_area / poly_area if poly_area > 0 else 0.0
