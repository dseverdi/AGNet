import os
import torch
import numpy as np
from torch.autograd import Variable

import sys
import faulthandler
faulthandler.enable()

import numpy as np
import skgeom
from concurrent.futures import ProcessPoolExecutor, as_completed
import math  # for sqrt
from collections import OrderedDict
import threading
import atexit
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import Dataset, agp_read_samples, collate_fn



# Global configuration
USE_CUDA = torch.cuda.is_available()



# Compute visibility polygon for a single guard index
def compute_visibility(vs, arr, poly, eps, edges, i):
    v_prev = edges[(i - 1) % len(edges)].source()
    v = edges[i % len(edges)].source()
    v_next = edges[i % len(edges)].target()

    p = skgeom.Vector2(v, v_prev)
    # normalize vector p
    p = p / math.sqrt(float(p.squared_length()))
    r = skgeom.Vector2(v, v_next)
    # normalize vector r
    r = r / math.sqrt(float(r.squared_length()))

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


# ===================================================================
#  Multiprocessing pool for parallel CGAL visibility
# ===================================================================

_PROC_POOL = None
_PROC_POOL_LOCK = threading.Lock()


def _get_proc_pool():
    """Lazy-initialise a persistent ProcessPoolExecutor.

    Controlled by ``AGNET_VIS_WORKERS`` env-var (default 0 = sequential).
    """
    global _PROC_POOL
    n = int(os.getenv("AGNET_VIS_WORKERS", "0"))
    if n <= 1:
        return None
    with _PROC_POOL_LOCK:
        if _PROC_POOL is None:
            _PROC_POOL = ProcessPoolExecutor(max_workers=n)
            atexit.register(_PROC_POOL.shutdown, wait=False)
        return _PROC_POOL


def _compute_guards_vis_vertices(points_arr, guard_indices):
    """Worker (runs in a subprocess): build CGAL arrangement from scratch
    and compute visibility polygons for *guard_indices*.

    Returns a list of ``(guard_idx, verts_or_None)`` where *verts* is an
    ``(M, 2)`` float64 ndarray of visibility-polygon vertices, or ``None``
    if the computation failed for that guard.
    """
    poly = createPolygon(points_arr)
    if poly is None or poly is False:
        return [(gi, None) for gi in guard_indices]

    arr = skgeom.arrangement.Arrangement()
    try:
        for edge in poly.edges:
            arr.insert(edge)
    except RuntimeError:
        return [(gi, None) for gi in guard_indices]

    vs = skgeom.TriangularExpansionVisibility(arr)
    edges = list(poly.edges)
    eps = 1e-8

    results = []
    for gi in guard_indices:
        vis_poly, _, _ = compute_visibility(vs, arr, poly, eps, edges, gi)
        if vis_poly:
            verts = np.array([[float(v.x()), float(v.y())]
                              for v in vis_poly.vertices])
            results.append((gi, verts))
        else:
            results.append((gi, None))

    del vs, arr, poly, edges
    return results


def _compute_guards_disc_vis(points_arr, guard_indices, sample_pts):
    """Worker (runs in a subprocess): build CGAL arrangement from scratch
    and compute a boolean visibility row for each guard in *guard_indices*.

    *sample_pts* is ``(S, 2)`` float64 ndarray of sample points inside the
    polygon.  Returns ``[(guard_idx, bool_row), ...]`` where *bool_row* is
    a 1-D bool ndarray of length ``S``.
    """
    n_samples = len(sample_pts)
    poly = createPolygon(points_arr)
    if poly is None or poly is False:
        return [(gi, np.zeros(n_samples, dtype=np.bool_))
                for gi in guard_indices]

    arr = skgeom.arrangement.Arrangement()
    try:
        for edge in poly.edges:
            arr.insert(edge)
    except RuntimeError:
        return [(gi, np.zeros(n_samples, dtype=np.bool_))
                for gi in guard_indices]

    vs = skgeom.TriangularExpansionVisibility(arr)
    edges = list(poly.edges)
    eps = 1e-8

    results = []
    for gi in guard_indices:
        vis_poly, _, _ = compute_visibility(vs, arr, poly, eps, edges, gi)
        if vis_poly:
            row = np.array([
                _point_in_visibility_polygon(vis_poly,
                                             sample_pts[s, 0],
                                             sample_pts[s, 1])
                for s in range(n_samples)
            ], dtype=np.bool_)
        else:
            row = np.zeros(n_samples, dtype=np.bool_)
        results.append((gi, row))

    del vs, arr, poly, edges
    return results


def _build_full_vis_cache_worker(points_arr):
    """Worker (runs in a subprocess): build visibility polygons for ALL
    guards of one polygon.

    Returns ``(poly_area_or_None, all_verts, offsets)`` where:
    - *all_verts* is ``(total_verts, 2)`` float64 ndarray with concatenated
      visibility polygon vertices for all guards.
    - *offsets* is ``(n+1,)`` int32 ndarray: guard *i*'s vertices are
      ``all_verts[offsets[i]:offsets[i+1]]`` (empty range = guard failed).
    """
    poly = createPolygon(points_arr)
    if poly is None or poly is False:
        return (None, None, None)

    poly_area = abs(float(poly.area()))
    arr = skgeom.arrangement.Arrangement()
    try:
        for edge in poly.edges:
            arr.insert(edge)
    except RuntimeError:
        return (None, None, None)

    vs = skgeom.TriangularExpansionVisibility(arr)
    edges = list(poly.edges)
    n = int(points_arr.shape[0])
    eps = 1e-8

    parts = []
    offsets = [0]
    total = 0
    for gi in range(n):
        vis_poly, _, _ = compute_visibility(vs, arr, poly, eps, edges, gi)
        if vis_poly:
            verts = np.array([[float(v.x()), float(v.y())]
                              for v in vis_poly.vertices])
            parts.append(verts)
            total += len(verts)
        offsets.append(total)

    del vs, arr, poly, edges

    if parts:
        all_verts = np.concatenate(parts)
    else:
        all_verts = np.empty((0, 2), dtype=np.float64)
    offsets = np.array(offsets, dtype=np.int32)
    return (poly_area, all_verts, offsets)


def _build_batch_vis_cache_worker(batch):
    """Worker: process a batch of polygons.

    *batch* is a list of ``(points_arr, name)`` tuples.
    Returns a list of ``(name, poly_area, all_verts, offsets)`` in the
    same order.
    """
    results = []
    for points_arr, name in batch:
        poly_area, all_verts, offsets = _build_full_vis_cache_worker(
            points_arr)
        results.append((name, poly_area, all_verts, offsets))
    return results


def _reconstruct_guard_vis_cache(all_verts, offsets, n):
    """Reconstruct guard_visibility_cache dict from packed arrays."""
    guard_visibility_cache = {}
    for gi in range(n):
        start, end = int(offsets[gi]), int(offsets[gi + 1])
        if start < end:
            verts = all_verts[start:end]
            vis_poly = skgeom.Polygon(
                [skgeom.Point2(float(x), float(y))
                 for x, y in verts])
            guard_visibility_cache[gi] = skgeom.PolygonSet([vis_poly])
        else:
            guard_visibility_cache[gi] = skgeom.PolygonSet()
    return guard_visibility_cache


def prewarm_vis_cache(dataset, n_workers=None, verbose=True):
    """Pre-build visibility caches for all polygons in *dataset* using
    a process pool (one polygon per task).  Skips polygons already cached.

    Uses ProcessPoolExecutor to bypass the GIL entirely — each subprocess
    builds its own CGAL objects, serialises results as packed numpy arrays,
    and the main process reconstructs the skgeom PolygonSet objects.

    Call before training / evaluation with exact CGAL to amortise the
    heavy per-polygon visibility work across all CPU cores.
    """
    if n_workers is None:
        n_workers = int(os.getenv("AGNET_VIS_WORKERS", "0"))
        if n_workers <= 1:
            n_workers = max(1, min(os.cpu_count() or 1, 16))

    # Collect polygons not yet cached
    items = []  # (points_array, name)
    for i in range(len(dataset)):
        data, _, name = dataset[i]
        pts = np.ascontiguousarray(data.numpy(), dtype=np.float64)
        key = _cache_key(pts, name)
        with _VIS_CACHE_LOCK:
            if key in _VIS_CACHE:
                continue
        items.append((pts, name))

    if not items:
        if verbose:
            print("[prewarm] All polygons already cached.")
        return

    if verbose:
        print(f"[prewarm] Building vis cache for {len(items)} polygons "
              f"with {n_workers} workers ...")

    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_build_full_vis_cache_worker, pts): (pts, name)
            for pts, name in items
        }
        for future in as_completed(futures):
            pts, name = futures[future]
            poly_area, all_verts, offsets = future.result()
            done += 1

            n = int(pts.shape[0])
            bbox = _points_bbox(pts)
            key = _cache_key(pts, name)

            if poly_area is None:
                cache = {"invalid": True, "n": n, "bbox": bbox}
            else:
                guard_vis = _reconstruct_guard_vis_cache(
                    all_verts, offsets, n)
                cache = {
                    "invalid": False,
                    "n": n,
                    "bbox": bbox,
                    "poly_area": poly_area,
                    "guard_visibility_cache": guard_vis,
                    "coverage_cache": {},
                }

            with _VIS_CACHE_LOCK:
                _VIS_CACHE[key] = cache
                max_cache = int(os.getenv("AGNET_VIS_CACHE_SIZE", "10000"))
                while len(_VIS_CACHE) > max_cache:
                    _VIS_CACHE.popitem(last=False)

            if verbose and (done % 100 == 0 or done == len(items)):
                print(f"[prewarm] {done}/{len(items)} done")

    if verbose:
        print(f"[prewarm] Complete. Cache size: {len(_VIS_CACHE)}")


# ===================================================================
#  Discretised visibility matrix (for fast training reward)
# ===================================================================

_DISC_VIS_CACHE_LOCK = threading.Lock()
_DISC_VIS_CACHE: "OrderedDict[str, dict]" = OrderedDict()


def _sample_points_in_polygon(poly, n_samples: int, rng=None):
    """Sample n_samples points uniformly inside a skgeom.Polygon via rejection."""
    if rng is None:
        rng = np.random.default_rng(42)
    verts = np.array([[float(v.x()), float(v.y())] for v in poly.vertices])
    xmin, xmax = verts[:, 0].min(), verts[:, 0].max()
    ymin, ymax = verts[:, 1].min(), verts[:, 1].max()
    pts = []
    batch = max(n_samples * 4, 1024)
    while len(pts) < n_samples:
        cands = np.column_stack([
            rng.uniform(xmin, xmax, batch),
            rng.uniform(ymin, ymax, batch),
        ])
        for x, y in cands:
            p = skgeom.Point2(float(x), float(y))
            if poly.oriented_side(p) == skgeom.Sign.POSITIVE:
                pts.append((float(x), float(y)))
                if len(pts) >= n_samples:
                    break
    return np.array(pts[:n_samples], dtype=np.float64)


def _point_in_visibility_polygon(vis_poly, px, py):
    """Check if point (px,py) is inside a visibility polygon."""
    p = skgeom.Point2(float(px), float(py))
    side = vis_poly.oriented_side(p)
    return side != skgeom.Sign.NEGATIVE  # POSITIVE or ON_BOUNDARY


def get_or_build_disc_vis(points: np.ndarray, name: str,
                          n_samples: int = 500) -> dict:
    """Build or retrieve a discretised visibility matrix.

    Returns dict with:
        vis_matrix: np.ndarray shape (n_guards, n_samples) bool
            vis_matrix[g, s] = True iff guard g sees sample point s
        n: int (number of guards/vertices)
        valid: bool
    """
    key = _cache_key(points, name)
    max_cache = int(os.getenv("AGNET_DISC_VIS_CACHE_SIZE", "10000"))

    with _DISC_VIS_CACHE_LOCK:
        cache = _DISC_VIS_CACHE.get(key)
        if cache is not None and cache.get("n") == int(points.shape[0]):
            _DISC_VIS_CACHE.move_to_end(key)
            return cache

    n = int(points.shape[0])
    poly = createPolygon(points)
    if poly is None or poly is False:
        cache = {"valid": False, "n": n}
        with _DISC_VIS_CACHE_LOCK:
            _DISC_VIS_CACHE[key] = cache
            _DISC_VIS_CACHE.move_to_end(key)
        return cache

    # Sample points inside the polygon (deterministic seed per polygon)
    seed = hash(name) % (2**31)
    rng = np.random.default_rng(seed)
    sample_pts = _sample_points_in_polygon(poly, n_samples, rng)
    if len(sample_pts) < n_samples:
        cache = {"valid": False, "n": n}
        with _DISC_VIS_CACHE_LOCK:
            _DISC_VIS_CACHE[key] = cache
            _DISC_VIS_CACHE.move_to_end(key)
        return cache

    # Build arrangement & visibility
    vis_matrix = np.zeros((n, n_samples), dtype=np.bool_)
    pool = _get_proc_pool()

    if pool is None:
        # Sequential path
        arr = skgeom.arrangement.Arrangement()
        try:
            for edge in poly.edges:
                arr.insert(edge)
        except RuntimeError:
            cache = {"valid": False, "n": n}
            with _DISC_VIS_CACHE_LOCK:
                _DISC_VIS_CACHE[key] = cache
                _DISC_VIS_CACHE.move_to_end(key)
            return cache

        vs = skgeom.TriangularExpansionVisibility(arr)
        edges = list(poly.edges)
        eps = 1e-8

        for guard_idx in range(n):
            vis_poly, err_idx, _ = compute_visibility(vs, arr, poly, eps, edges, guard_idx)
            if vis_poly:
                for s in range(n_samples):
                    vis_matrix[guard_idx, s] = _point_in_visibility_polygon(
                        vis_poly, sample_pts[s, 0], sample_pts[s, 1]
                    )

        del vs, arr, edges
    else:
        # Parallel path — each worker builds its own CGAL objects
        n_workers = int(os.getenv("AGNET_VIS_WORKERS", "0"))
        chunk_size = max(1, (n + n_workers - 1) // n_workers)
        chunks = [list(range(i, min(i + chunk_size, n)))
                  for i in range(0, n, chunk_size)]

        points_arr = np.ascontiguousarray(points, dtype=np.float64)
        sample_pts_arr = np.ascontiguousarray(sample_pts, dtype=np.float64)
        futures = [pool.submit(_compute_guards_disc_vis,
                               points_arr, chunk, sample_pts_arr)
                   for chunk in chunks]

        for future in futures:
            for gi, row in future.result():
                vis_matrix[gi] = row

    del poly

    cache = {
        "valid": True,
        "n": n,
        "vis_matrix": vis_matrix,
        "n_samples": n_samples,
    }
    with _DISC_VIS_CACHE_LOCK:
        _DISC_VIS_CACHE[key] = cache
        _DISC_VIS_CACHE.move_to_end(key)
        while len(_DISC_VIS_CACHE) > max_cache:
            _DISC_VIS_CACHE.popitem(last=False)
    return cache


def _build_full_disc_vis_worker(points_arr: np.ndarray,
                                n_samples: int,
                                seed: int) -> "tuple[np.ndarray | None, np.ndarray | None]":
    """Subprocess worker: build the full disc_vis matrix for one polygon.

    Returns ``(vis_matrix, sample_pts)`` where *vis_matrix* is shape
    ``(n, n_samples)`` bool, or ``(None, None)`` if the polygon is invalid.
    Builds the CGAL arrangement once and sweeps all guard vertices.
    """
    n = int(points_arr.shape[0])
    poly = createPolygon(points_arr)
    if poly is None or poly is False:
        return None, None

    rng = np.random.default_rng(seed)
    sample_pts = _sample_points_in_polygon(poly, n_samples, rng)
    if len(sample_pts) < n_samples:
        return None, None

    arr = skgeom.arrangement.Arrangement()
    try:
        for edge in poly.edges:
            arr.insert(edge)
    except RuntimeError:
        return None, None

    vs = skgeom.TriangularExpansionVisibility(arr)
    edges = list(poly.edges)
    eps = 1e-8

    vis_matrix = np.zeros((n, n_samples), dtype=np.bool_)
    for guard_idx in range(n):
        vis_poly, _, _ = compute_visibility(vs, arr, poly, eps, edges, guard_idx)
        if vis_poly:
            for s in range(n_samples):
                vis_matrix[guard_idx, s] = _point_in_visibility_polygon(
                    vis_poly, float(sample_pts[s, 0]), float(sample_pts[s, 1])
                )

    del vs, arr, edges, poly
    return vis_matrix, sample_pts


def prewarm_disc_vis_cache(dataset, n_samples: int = 500,
                           n_workers=None,
                           verbose: bool = True) -> None:
    """Pre-build discretised visibility caches for all polygons in *dataset*.

    Uses ProcessPoolExecutor — one polygon per task.  Each worker builds the
    CGAL arrangement once and sweeps all n guards, returning the full
    (n × n_samples) boolean visibility matrix.  Skips polygons already cached.

    Call this before LS fine-tuning to amortise the per-polygon CGAL cost
    across all CPU cores instead of hitting it lazily during training.
    """
    if n_workers is None:
        n_workers = int(os.getenv("AGNET_VIS_WORKERS", "0"))
        if n_workers <= 1:
            n_workers = max(1, min(os.cpu_count() or 1, 16))

    items: list[tuple[np.ndarray, str, int]] = []
    for i in range(len(dataset)):
        data, _, name = dataset[i]
        pts = np.ascontiguousarray(data.numpy(), dtype=np.float64)
        key = _cache_key(pts, name)
        with _DISC_VIS_CACHE_LOCK:
            if key in _DISC_VIS_CACHE:
                continue
        seed = hash(name) % (2 ** 31)
        items.append((pts, name, seed))

    if not items:
        if verbose:
            print("[prewarm-disc] All disc_vis caches already built.")
        return

    if verbose:
        print(f"[prewarm-disc] Building disc_vis for {len(items)} polygons "
              f"with {n_workers} workers (n_samples={n_samples}) ...")

    done = 0
    max_cache = int(os.getenv("AGNET_DISC_VIS_CACHE_SIZE", "10000"))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_build_full_disc_vis_worker,
                        pts, n_samples, seed): (pts, name)
            for pts, name, seed in items
        }
        for future in as_completed(futures):
            pts, name = futures[future]
            vis_matrix, sample_pts = future.result()
            done += 1

            n = int(pts.shape[0])
            key = _cache_key(pts, name)
            if vis_matrix is None:
                cache: dict = {"valid": False, "n": n}
            else:
                cache = {
                    "valid": True,
                    "n": n,
                    "vis_matrix": vis_matrix,
                    "n_samples": n_samples,
                }

            with _DISC_VIS_CACHE_LOCK:
                _DISC_VIS_CACHE[key] = cache
                _DISC_VIS_CACHE.move_to_end(key)
                while len(_DISC_VIS_CACHE) > max_cache:
                    _DISC_VIS_CACHE.popitem(last=False)

            if verbose and (done % 100 == 0 or done == len(items)):
                print(f"[prewarm-disc] {done}/{len(items)} done")

    if verbose:
        print(f"[prewarm-disc] Complete. Disc_vis cache size: {len(_DISC_VIS_CACHE)}")


# Merge a list of PolygonSet into one
def merge_polygon_sets(polygon_sets):
    merged = skgeom.PolygonSet()
    for ps in polygon_sets:
        merged = merged.union(ps)
    return merged

def createPolygon(points: np.ndarray) -> bool:
    """
    Quick validity check: at least 3 distinct points and non-zero area.
    """
    # Construct polygon from Point2 list
    verts = [skgeom.Point2(float(x), float(y)) for x, y in points]
    try:
        poly = skgeom.Polygon(verts)        
    except Exception:
        return None
    # need at least 3 vertices
    verts = list(poly.vertices)
    if len(verts) < 3:
        return False
    # area should be non-zero
    if abs(float(poly.area())) < 1e-8:
        return False
    return poly

# ---------------------------
# Visibility cache (per polygon)
# ---------------------------

_VIS_CACHE_LOCK = threading.Lock()
_VIS_CACHE: "OrderedDict[str, dict]" = OrderedDict()


def _cache_key(points: np.ndarray, name: str) -> str:
    n = int(points.shape[0])
    if name:
        return f"{name}|{n}"
    return f"anon|{n}|{id(points)}"


def _points_bbox(points: np.ndarray) -> tuple:
    xs = points[:, 0]
    ys = points[:, 1]
    return (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))


def _get_or_build_vis_cache(points: np.ndarray, name: str) -> dict:
    key = _cache_key(points, name)
    bbox = _points_bbox(points)
    max_cache = int(os.getenv("AGNET_VIS_CACHE_SIZE", "10000"))
    verbose = bool(int(os.getenv("AGNET_VIS_CACHE_VERBOSE", "0")))

    with _VIS_CACHE_LOCK:
        cache = _VIS_CACHE.get(key)
        if cache is not None and cache.get("n") == int(points.shape[0]) and cache.get("bbox") == bbox:
            _VIS_CACHE.move_to_end(key)
            return cache

    # Build cache (outside lock)
    poly = createPolygon(points)
    if poly is None or poly is False:
        cache = {"invalid": True, "n": int(points.shape[0]), "bbox": bbox}
        with _VIS_CACHE_LOCK:
            _VIS_CACHE[key] = cache
            _VIS_CACHE.move_to_end(key)
        return cache

    n = int(points.shape[0])
    poly_area = abs(float(poly.area()))

    guard_visibility_cache = {}
    pool = _get_proc_pool()

    if pool is None:
        # Sequential path (AGNET_VIS_WORKERS <= 1)
        arr = skgeom.arrangement.Arrangement()
        try:
            for edge in poly.edges:
                arr.insert(edge)
        except RuntimeError as e:
            if verbose:
                print(f"[vis-cache] arrangement build failed for {name}: {e}", file=sys.stderr)
            cache = {"invalid": True, "n": int(points.shape[0]), "bbox": bbox}
            with _VIS_CACHE_LOCK:
                _VIS_CACHE[key] = cache
                _VIS_CACHE.move_to_end(key)
            return cache

        vs = skgeom.TriangularExpansionVisibility(arr)
        edges = list(poly.edges)
        eps = 1e-8

        for guard_idx in range(n):
            vis_poly, err_idx, _ = compute_visibility(vs, arr, poly, eps, edges, guard_idx)
            if vis_poly:
                guard_visibility_cache[guard_idx] = skgeom.PolygonSet([vis_poly])
            else:
                if verbose:
                    print(f"[vis-cache] visibility failed at guard {err_idx} for {name}", file=sys.stderr)
                guard_visibility_cache[guard_idx] = skgeom.PolygonSet()

        del vs, arr, edges
    else:
        # Parallel path — each worker builds its own CGAL objects (safe)
        n_workers = int(os.getenv("AGNET_VIS_WORKERS", "0"))
        chunk_size = max(1, (n + n_workers - 1) // n_workers)
        chunks = [list(range(i, min(i + chunk_size, n)))
                  for i in range(0, n, chunk_size)]

        points_arr = np.ascontiguousarray(points, dtype=np.float64)
        futures = [pool.submit(_compute_guards_vis_vertices,
                               points_arr, chunk)
                   for chunk in chunks]

        for future in futures:
            for gi, verts in future.result():
                if verts is not None:
                    vis_poly = skgeom.Polygon(
                        [skgeom.Point2(float(x), float(y))
                         for x, y in verts])
                    guard_visibility_cache[gi] = skgeom.PolygonSet([vis_poly])
                else:
                    if verbose:
                        print(f"[vis-cache] visibility failed at guard {gi} for {name}",
                              file=sys.stderr)
                    guard_visibility_cache[gi] = skgeom.PolygonSet()

    del poly

    cache = {
        "invalid": False,
        "n": n,
        "bbox": bbox,
        "poly_area": poly_area,
        "guard_visibility_cache": guard_visibility_cache,
        "coverage_cache": {},
    }

    with _VIS_CACHE_LOCK:
        _VIS_CACHE[key] = cache
        _VIS_CACHE.move_to_end(key)
        # LRU eviction
        while len(_VIS_CACHE) > max_cache:
            _VIS_CACHE.popitem(last=False)
    return cache


# Evaluate coverage without ground-truth (numpy-based)
def evaluate_polygon_visibility_numpy_wo_gt(points: np.ndarray, solution: np.ndarray, name: str) -> float:
    use_cache = bool(int(os.getenv("AGNET_VIS_CACHE", "1")))
    if use_cache:
        cache = _get_or_build_vis_cache(points, name)
        if cache.get("invalid"):
            print(f"Skipping invalid polygon in {name}: less than 3 vertices or zero area", file=sys.stderr)
            return 0.0

        solution_key = tuple(sorted(int(x) for x in solution))
        cov_cache = cache["coverage_cache"]
        if solution_key in cov_cache:
            return cov_cache[solution_key]

        if len(solution_key) == 0:
            coverage = 0.0
        else:
            try:
                union_region = skgeom.PolygonSet()
                guard_visibility_cache = cache["guard_visibility_cache"]
                for guard_idx in solution_key:
                    if guard_idx in guard_visibility_cache:
                        union_region = union_region.union(guard_visibility_cache[guard_idx])

                total_area = 0.0
                for vis in union_region.polygons:
                    outer = abs(float(vis.outer_boundary().area()))
                    holes = sum(abs(float(h.area())) for h in vis.holes)
                    total_area += outer - holes

                poly_area = cache["poly_area"]
                coverage = total_area / poly_area if poly_area > 0 else 0.0
            except Exception as e:
                print(f"Fast coverage computation failed for {name}: {e}", file=sys.stderr)
                coverage = 0.0

        cov_cache[solution_key] = coverage
        return coverage

    # Validate polygon definition
    eps = 1e-8
    # Construct polygon from Point2 list
    poly = createPolygon(points)
    if poly is None or poly is False:
        print(f"Skipping invalid polygon in {name}: less than 3 vertices or zero area", file=sys.stderr)
        return 0.0

    # Compute visibility polygons (optionally parallel via multiprocessing)
    views = []
    pool = _get_proc_pool()

    if pool is None:
        # Sequential path
        arr = skgeom.arrangement.Arrangement()
        arrangement_ok = True
        for edge in poly.edges:
            try:
                arr.insert(edge)
            except RuntimeError as e:
                print(f"Skipping polygon {name} due to CGAL precondition violation during arrangement construction.", file=sys.stderr)
                plot_problematic_polygon(points, name, edge)
                arrangement_ok = False
                break
        if not arrangement_ok:
            return 0.0
        vs = skgeom.TriangularExpansionVisibility(arr)
        edges = list(poly.edges)
        for idx in solution:
            vis_poly, err_idx, err_q = compute_visibility(vs, arr, poly, eps, edges, idx)
            if vis_poly:
                views.append(skgeom.PolygonSet([vis_poly]))
            else:
                print(f"Warning: visibility failed at guard {err_idx}", file=sys.stderr)
        del vs, arr, edges
    else:
        # Parallel path — workers build own CGAL objects
        sol_list = [int(idx) for idx in solution]
        n_workers = int(os.getenv("AGNET_VIS_WORKERS", "0"))
        chunk_size = max(1, (len(sol_list) + n_workers - 1) // n_workers)
        chunks = [sol_list[i:i + chunk_size]
                  for i in range(0, len(sol_list), chunk_size)]

        points_arr = np.ascontiguousarray(points, dtype=np.float64)
        futures = [pool.submit(_compute_guards_vis_vertices,
                               points_arr, chunk)
                   for chunk in chunks]
        for future in futures:
            for gi, verts in future.result():
                if verts is not None:
                    vp = skgeom.Polygon(
                        [skgeom.Point2(float(x), float(y))
                         for x, y in verts])
                    views.append(skgeom.PolygonSet([vp]))
                else:
                    print(f"Warning: visibility failed at guard {gi}", file=sys.stderr)

    # Merge partial regions
    region = merge_polygon_sets(views)

    # Calculate total visible area
    total_area = 0.0
    poly_area = abs(float(poly.area()))
    for vis in region.polygons:
        outer = abs(float(vis.outer_boundary().area()))
        holes = sum(abs(float(h.area())) for h in vis.holes)
        total_area += outer - holes

    return total_area / poly_area if poly_area > 0 else 0.0

# --- Utility ---
def get_checkpoint_path(folder, model_name, params, n_epochs):
    """Generate a checkpoint path based on model name, parameters, and epoch count."""
    param_str = "_".join([f"{k}{v}" for k, v in sorted(params.items())])
    filename = f"{model_name}_{param_str}_epochs{n_epochs}.pt"
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, filename)

# --- Data Preparation ---
def prepare_datasets(train_dir, val_dir, normalize=True):
    agp_train_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith('.pol')]
    agp_val_paths = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.pol')]
    print(f"Found {len(agp_train_paths)} training and {len(agp_val_paths)} validation AGP .pol files.")
    train_samples = agp_read_samples(agp_train_paths, normalize=normalize)
    val_samples = agp_read_samples(agp_val_paths, normalize=normalize)
    return Dataset(train_samples), Dataset(val_samples)

# --- Model Creation ---
def create_agp_model(embedding_size, hidden_size, n_glimpses, tanh_exploration, use_tanh, reward, temperature):
    return create_model(
        embedding_size, hidden_size, None, n_glimpses,
        tanh_exploration, use_tanh, "Bahdanau", reward, temperature=temperature
    )

# --- Test Forward Pass ---
def test_model_on_sample(model, dataset):
    print("\n--- Forward pass test with a single validation sample ---")
    if len(dataset) == 0:
        print("No validation samples available for forward pass test.")
        return
    sample_data, _, sample_name = dataset[0]
    sample_data = sample_data.unsqueeze(0)
    device = next(model.parameters()).device
    sample_data = sample_data.to(device)
    model.eval()
    with torch.no_grad():
        result = model(sample_data)
    print(f"Sample name: {sample_name}")
    print(f"Model output: {result}")

# --- Test Batch Forward Pass ---
def test_model_on_batch(model, dataset, batch_size=2):
    """Run a forward pass on a single batch of specified size"""
    print(f"\n--- Forward pass test with a batch of size {batch_size} ---")
    if len(dataset) < batch_size:
        print(f"Not enough samples to form a batch of size {batch_size}")
        return
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    batch_data, mask, batch_names = next(iter(loader))  # mask for padded vertices
    device = next(model.parameters()).device
    batch_data = batch_data.to(device)
    model.eval()
    with torch.no_grad():
        result = model(batch_data)
    print(f"Batch sample names: {batch_names}")
    print(f"Model outputs: {result}")

# --- Simple Training Loop for Debugging ---
def train_on_small_sample(model, dataset, reward_fn, epochs=2, batch_size=1, lr=1e-3):
    print(f"\n--- Training on {len(dataset)} samples for {epochs} epochs (batch size {batch_size}) ---")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    device = next(model.parameters()).device
    for epoch in range(epochs):
        total_loss = 0
        for batch_data, mask, batch_names in loader:  # mask for padded vertices
            # Forward pass: model should return (selected_idxs, log_probs)
            selected_idxs, log_probs = model(batch_data)
            rewards_list = []
            for data_tensor, idxs, name in zip(batch_data.cpu(), selected_idxs, batch_names):
                points = data_tensor.numpy()
                sol = np.array(idxs)
                r = reward_fn(points, sol, name)
                rewards_list.append(r)
            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
            # REINFORCE loss: negative expected reward weighted by log-probabilities
            # log_probs should be shape (batch_size,)
            loss = -(log_probs * rewards).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_data.size(0)
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Avg loss: {avg_loss:.4f}")
    print("Training done.")

# --- Test Coverage ---
def test_coverage_on_sample(dataset, sol_dir, index=0, regime="opt", n_random_guards=None):
    """Load the optimal solution or a random guard set and evaluate coverage on one sample
    n_random_guards: if set and regime=='random', use this many guards (default: match optimal solution)
    """
    import random
    print(f"\n--- Coverage test on a single sample (regime: {regime}) ---")
    if len(dataset) == 0:
        print("No validation samples available for coverage test.")
        return
    # Get raw polygon points and sample name
    sample_data, _, sample_name = dataset[index]
    points = sample_data.numpy()
    n_points = len(points)
    true_idxs = []
    if regime == "opt":
        # Read optimal guard indices from .solution file (second line)
        sol_path = os.path.join(sol_dir, f"{sample_name}.solution")
        try:
            with open(sol_path, 'r') as f:
                lines = f.read().splitlines()
                if len(lines) >= 2:
                    true_idxs = [int(x) for x in lines[1].split()]
        except Exception as e:
            print(f"Could not read solution file {sol_path}: {e}", file=sys.stderr)
            return
        if not true_idxs:
            print(f"No guards found in solution file for sample {sample_name}")
            return
        guard_idxs = np.array(true_idxs)
        label = "True (optimal)"
    elif regime == "random":
        # Try to match the number of guards in the optimal solution if possible, unless overridden
        if n_random_guards is not None:
            n_guards = min(max(1, int(n_random_guards)), n_points)
        else:
            sol_path = os.path.join(sol_dir, f"{sample_name}.solution")
            n_guards = 1
            try:
                with open(sol_path, 'r') as f:
                    lines = f.read().splitlines()
                    if len(lines) >= 2:
                        n_guards = max(1, len([int(x) for x in lines[1].split()]))
            except Exception:
                n_guards = max(1, n_points // 10)  # fallback: 10% of vertices
        guard_idxs = np.array(sorted(random.sample(range(n_points), min(n_guards, n_points))))
        label = f"Random ({len(guard_idxs)} guards)"
    else:
        print(f"Unknown regime: {regime}")
        return
    # Evaluate coverage
    coverage = evaluate_polygon_visibility_numpy_wo_gt(points, guard_idxs, sample_name)
    print(f"Sample: {sample_name}  {label} coverage: {coverage:.4f}")

    # --- Visualization with visibility regions ---
    # Compute visibility polygons for each guard
    import skgeom
    eps = 1e-8
    poly_obj = createPolygon(points)
    if poly_obj is None:
        print(f"Invalid polygon in {sample_name}: less than 3 vertices or zero area", file=sys.stderr)
        return
    arr = skgeom.arrangement.Arrangement()
    for edge in poly_obj.edges:
        arr.insert(edge)
    vs = skgeom.TriangularExpansionVisibility(arr)
    edges = list(poly_obj.edges)
    vis_polys = []
    for idx in guard_idxs:
        vis_poly, err_idx, err_q = compute_visibility(vs, arr, poly_obj, eps, edges, idx)
        vis_polys.append(vis_poly if vis_poly else None)

    fig, ax = plt.subplots()
    poly = np.array(points)
    if poly.shape[0] > 2:
        ax.plot(np.append(poly[:,0], poly[0,0]), np.append(poly[:,1], poly[0,1]), 'k-', lw=1, label='Polygon')
    # Plot each guard's visibility region
    for i, vis_poly in enumerate(vis_polys):
        if vis_poly is not None:
            vis_pts = np.array([[p.x(), p.y()] for p in vis_poly.vertices])
            ax.fill(vis_pts[:,0], vis_pts[:,1], alpha=0.25, label=f'Guard {i} vis' if i==0 else None)
    # Guards
    guards = poly[guard_idxs]
    ax.scatter(guards[:,0], guards[:,1], c='red', s=60, marker='*', label='Guards')
    ax.set_aspect('equal')
    ax.set_title(f"{sample_name} ({label})\nCoverage: {coverage:.2f}")
    ax.legend()
    out_dir = os.path.join(os.path.dirname(__file__), 'gfx')
    os.makedirs(out_dir, exist_ok=True)
    # Add number of guards to the filename
    n_guards_str = f"{len(guard_idxs)}_guards"
    out_path = os.path.join(out_dir, f"{sample_name}_{regime}_{n_guards_str}_coverage.png")
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved coverage plot to {out_path}")
    plt.close(fig)


def plot_problematic_polygon(points, name, edge=None):
    poly = np.array(points)
    fig, ax = plt.subplots()
    if poly.shape[0] > 2:
        # Close the polygon by connecting last point to the first
        closed_poly = np.vstack([poly, poly[0]])
        ax.plot(closed_poly[:,0], closed_poly[:,1], 'k-', lw=2, label='Polygon')
    ax.scatter(poly[:,0], poly[:,1], c='blue', s=40, label='Vertices')
    if edge is not None:
        ex = [edge.source().x(), edge.target().x()]
        ey = [edge.source().y(), edge.target().y()]
        ax.plot(ex, ey, 'r-', lw=3, label='Problem Edge')
    ax.set_aspect('equal')
    ax.set_title(f"Problematic polygon: {name}")
    ax.legend()
    import os
    out_dir = os.path.join(os.path.dirname(__file__), 'gfx')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{name}_error.png')
    plt.savefig(out_path, bbox_inches='tight')
    print(f"Saved problematic polygon plot to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    # Flexible: test coverage on a .pol file or all .pol files in a folder, using agp_read_samples and test_coverage_on_sample
    import os
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Test AGP coverage for .pol file(s) using all vertices as guards or optimal guards.")
    parser.add_argument("path", type=str, help="Path to a .pol file or a directory containing .pol files.")
    parser.add_argument("--regime", type=str, default="opt", choices=["opt", "random"], help="Guard selection regime: 'opt' for optimal guards, 'random' for all vertices as guards.")
    args = parser.parse_args()
    path = args.path
    regime = args.regime
    if os.path.isdir(path):
        pol_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pol')]
        if not pol_files:
            print(f"No .pol files found in directory {path}")
            sys.exit(0)
    elif os.path.isfile(path) and path.endswith('.pol'):
        pol_files = [path]
    else:
        print(f"Provided path {path} is neither a .pol file nor a directory containing .pol files.")
        sys.exit(1)
    # Use agp_read_samples for robust loading
    samples = agp_read_samples(pol_files, normalize=True)
    dataset = Dataset(samples)
    for idx, pol_path in enumerate(pol_files):
        sol_dir = os.path.dirname(pol_path)
        if regime == "opt":
            test_coverage_on_sample(dataset, sol_dir=sol_dir, index=idx, regime="opt")
        else:
            n_vertices = len(samples[idx].data)
            test_coverage_on_sample(dataset, sol_dir=sol_dir, index=idx, regime="random", n_random_guards=n_vertices)
