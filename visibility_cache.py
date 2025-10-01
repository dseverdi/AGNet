"""
Visibility Region Cache Manager for AGP.

Precomputes and caches per-vertex visibility regions to enable fast coverage evaluation
via cached polygon set unions, avoiding repeated expensive CGAL visibility computations.
"""

import numpy as np
import skgeom
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import math


class VisibilityCache:
    """
    Global cache for precomputed visibility regions and coverage computations.
    
    Architecture:
    - Per-instance cache: polygon_name -> {guard_idx -> PolygonSet}
    - Coverage cache: (polygon_name, tuple(sorted_guards)) -> coverage
    - Thread-safe for parallel data loading
    """
    
    def __init__(self):
        # Instance cache: polygon_name -> dict of {guard_idx: PolygonSet}
        self.guard_visibility_cache = {}
        
        # Coverage cache: (polygon_name, tuple(sorted_guards)) -> coverage
        self.coverage_cache = {}
        
        # Polygon metadata: polygon_name -> {'poly': Polygon, 'area': float, 'arr': Arrangement}
        self.polygon_metadata = {}
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        self.precompute_times = {}
    
    def _create_polygon(self, points: np.ndarray):
        """Create and validate polygon from points."""
        verts = [skgeom.Point2(float(x), float(y)) for x, y in points]
        try:
            poly = skgeom.Polygon(verts)
        except Exception:
            return None
        
        verts = list(poly.vertices)
        if len(verts) < 3:
            return None
        if abs(float(poly.area())) < 1e-8:
            return None
        return poly
    
    def _compute_single_visibility(self, vs, arr, poly, eps, edges, guard_idx):
        """Compute visibility polygon for a single guard."""
        v_prev = edges[(guard_idx - 1) % len(edges)].source()
        v = edges[guard_idx % len(edges)].source()
        v_next = edges[guard_idx % len(edges)].target()
        
        p = skgeom.Vector2(v, v_prev)
        p = p / math.sqrt(float(p.squared_length()))
        r = skgeom.Vector2(v, v_next)
        r = r / math.sqrt(float(r.squared_length()))
        
        q = skgeom.Point2(v.x() + eps * (p.x() + r.x()), v.y() + eps * (p.y() + r.y()))
        if poly.oriented_side(q) != skgeom.Sign.POSITIVE:
            q = skgeom.Point2(v.x() - eps * (p.x() + r.x()), v.y() - eps * (p.y() + r.y()))
        
        face = arr.find(q)
        if face is None or face.is_unbounded():
            return None
        
        try:
            vx = vs.compute_visibility(q, face)
            visibility_polygon = skgeom.Polygon([vertex.point() for vertex in vx.vertices])
            return visibility_polygon
        except RuntimeError:
            return None
    
    def precompute_instance(self, points: np.ndarray, polygon_name: str, force=False):
        """
        Precompute visibility regions for all vertices of a polygon instance.
        
        Args:
            points: Polygon vertices as numpy array [n, 2]
            polygon_name: Unique identifier for this polygon
            force: Force recomputation even if already cached
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Skip if already cached
        if not force and polygon_name in self.guard_visibility_cache:
            return True
        
        start_time = time.time()
        eps = 1e-8
        
        # Create polygon
        poly = self._create_polygon(points)
        if poly is None:
            print(f"Warning: Invalid polygon {polygon_name}", file=sys.stderr)
            return False
        
        # Build arrangement
        arr = skgeom.arrangement.Arrangement()
        try:
            for edge in poly.edges:
                arr.insert(edge)
        except RuntimeError as e:
            print(f"Warning: CGAL arrangement failed for {polygon_name}: {e}", file=sys.stderr)
            return False
        
        # Store metadata
        poly_area = abs(float(poly.area()))
        self.polygon_metadata[polygon_name] = {
            'poly': poly,
            'area': poly_area,
            'arr': arr,
            'n_vertices': len(points)
        }
        
        # Compute visibility regions for all vertices in parallel
        vs = skgeom.TriangularExpansionVisibility(arr)
        edges = list(poly.edges)
        n_vertices = len(points)
        
        visibility_regions = {}
        
        with ThreadPoolExecutor() as executor:
            futures = {
                guard_idx: executor.submit(
                    self._compute_single_visibility,
                    vs, arr, poly, eps, edges, guard_idx
                )
                for guard_idx in range(n_vertices)
            }
            
            for guard_idx, future in futures.items():
                vis_poly = future.result()
                if vis_poly:
                    # Store as PolygonSet for fast union operations
                    visibility_regions[guard_idx] = skgeom.PolygonSet([vis_poly])
                else:
                    # Empty visibility region
                    visibility_regions[guard_idx] = skgeom.PolygonSet()
        
        self.guard_visibility_cache[polygon_name] = visibility_regions
        
        elapsed = time.time() - start_time
        self.precompute_times[polygon_name] = elapsed
        
        return True
    
    def get_coverage_fast(self, points: np.ndarray, solution: np.ndarray, polygon_name: str) -> float:
        """
        Compute coverage using cached visibility regions.
        
        Args:
            points: Polygon vertices [n, 2] (used if not cached)
            solution: Guard indices as numpy array
            polygon_name: Unique identifier for polygon
        
        Returns:
            float: Coverage ratio (0.0 to 1.0)
        """
        # Ensure instance is precomputed
        if polygon_name not in self.guard_visibility_cache:
            success = self.precompute_instance(points, polygon_name)
            if not success:
                self.cache_misses += 1
                return 0.0
        
        # Check coverage cache
        solution_key = (polygon_name, tuple(sorted(solution)))
        if solution_key in self.coverage_cache:
            self.cache_hits += 1
            return self.coverage_cache[solution_key]
        
        self.cache_misses += 1
        
        # Compute coverage using cached visibility regions
        if len(solution) == 0:
            coverage = 0.0
        else:
            try:
                visibility_regions = self.guard_visibility_cache[polygon_name]
                poly_area = self.polygon_metadata[polygon_name]['area']
                
                # Union all visibility regions for guards in solution
                union_region = skgeom.PolygonSet()
                for guard_idx in solution:
                    guard_idx = int(guard_idx)  # Ensure int type
                    if guard_idx in visibility_regions:
                        union_region = union_region.union(visibility_regions[guard_idx])
                
                # Calculate total visible area
                total_area = 0.0
                for vis in union_region.polygons:
                    outer = abs(float(vis.outer_boundary().area()))
                    holes = sum(abs(float(h.area())) for h in vis.holes)
                    total_area += outer - holes
                
                coverage = total_area / poly_area if poly_area > 0 else 0.0
                
            except Exception as e:
                print(f"Warning: Fast coverage computation failed for {polygon_name}: {e}", file=sys.stderr)
                coverage = 0.0
        
        # Cache the result
        self.coverage_cache[solution_key] = coverage
        return coverage
    
    def clear_coverage_cache(self):
        """Clear coverage cache but keep precomputed visibility regions."""
        self.coverage_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def clear_all(self):
        """Clear all caches."""
        self.guard_visibility_cache.clear()
        self.coverage_cache.clear()
        self.polygon_metadata.clear()
        self.precompute_times.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_stats(self):
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'instances_cached': len(self.guard_visibility_cache),
            'coverage_cache_size': len(self.coverage_cache),
            'total_precompute_time': sum(self.precompute_times.values()),
            'avg_precompute_time': np.mean(list(self.precompute_times.values())) if self.precompute_times else 0.0
        }
    
    def precompute_dataset(self, dataset, max_instances=None, show_progress=True):
        """
        Precompute visibility regions for an entire dataset.
        
        Args:
            dataset: Dataset object with samples
            max_instances: Maximum number of instances to precompute (None = all)
            show_progress: Show progress bar
        """
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(len(dataset)), desc="Precomputing visibility regions")
            except ImportError:
                iterator = range(len(dataset))
                print(f"Precomputing visibility regions for {len(dataset)} instances...")
        else:
            iterator = range(len(dataset))
        
        success_count = 0
        for idx in iterator:
            if max_instances is not None and idx >= max_instances:
                break
            
            polygon_tensor, _, polygon_name = dataset[idx]
            points = polygon_tensor.numpy()
            
            if self.precompute_instance(points, polygon_name):
                success_count += 1
        
        if show_progress:
            print(f"Successfully precomputed {success_count}/{min(len(dataset), max_instances or len(dataset))} instances")
        
        return success_count


# Global singleton instance
_global_cache = None

def get_global_cache():
    """Get or create the global visibility cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = VisibilityCache()
    return _global_cache


def clear_global_cache():
    """Clear the global visibility cache."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear_all()
        _global_cache = None
