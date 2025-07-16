import torch
import numpy as np
from torch.autograd import Variable

import sys
import faulthandler
faulthandler.enable()

import numpy as np
import skgeom
from concurrent.futures import ThreadPoolExecutor
import math  # for sqrt
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import Dataset, agp_read_samples, collate_fn



# Global configuration
USE_CUDA = True



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

# Evaluate coverage without ground-truth (numpy-based)
def evaluate_polygon_visibility_numpy_wo_gt(points: np.ndarray, solution: np.ndarray, name: str) -> float:
    # Validate polygon definition
    eps = 1e-8
    # Construct polygon from Point2 list
    poly = createPolygon(points)
    if poly is None or poly is False:
        print(f"Skipping invalid polygon in {name}: less than 3 vertices or zero area", file=sys.stderr)
        return 0.0
    

    # Build arrangement
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
    from concurrent.futures import ThreadPoolExecutor
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
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_visibility, vs, arr, poly_obj, eps, edges, idx) for idx in guard_idxs]
        for future in futures:
            vis_poly, err_idx, err_q = future.result()
            if vis_poly:
                vis_polys.append(vis_poly)
            else:
                vis_polys.append(None)

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
