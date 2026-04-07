import numpy as np


indices1_g = None
indices2_g = None


def ray_casting_init(m: int, n: int) -> None:
    """Precompute pair indices for repeated ray casting with fixed sizes."""
    global indices1_g, indices2_g
    indices1_g = np.arange(m).repeat(n)
    indices2_g = np.tile(np.arange(n), m)


def ray_casting(
    lines1: np.ndarray,
    lines2: np.ndarray,
    initialized_indices: bool = False,
    overwrite: bool = False,
    return_whether_intersect: bool = False,
):
    """Compute closest intersections from lines2 rays to line segments in lines1."""
    if not initialized_indices:
        indices1 = np.arange(lines1.shape[0]).repeat(lines2.shape[0])
        indices2 = np.tile(np.arange(lines2.shape[0]), lines1.shape[0])
    else:
        indices1 = indices1_g
        indices2 = indices2_g

    x1 = lines1[indices1, 0, 0]
    y1 = lines1[indices1, 0, 1]
    x2 = lines1[indices1, 1, 0]
    y2 = lines1[indices1, 1, 1]

    x3 = lines2[indices2, 0, 0]
    y3 = lines2[indices2, 0, 1]
    x4 = lines2[indices2, 1, 0]
    y4 = lines2[indices2, 1, 1]

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    non_parallel = np.abs(den) > np.finfo(float).eps

    t = np.full_like(den, np.inf, dtype=float)
    u = np.full_like(den, np.inf, dtype=float)
    t[non_parallel] = (
        ((x1[non_parallel] - x3[non_parallel]) * (y3[non_parallel] - y4[non_parallel])
         - (y1[non_parallel] - y3[non_parallel]) * (x3[non_parallel] - x4[non_parallel]))
        / den[non_parallel]
    )
    u[non_parallel] = (
        ((x2[non_parallel] - x1[non_parallel]) * (y1[non_parallel] - y3[non_parallel])
         - (y2[non_parallel] - y1[non_parallel]) * (x1[non_parallel] - x3[non_parallel]))
        / den[non_parallel]
    )

    px = np.full_like(t, np.nan, dtype=float)
    py = np.full_like(t, np.nan, dtype=float)
    px[non_parallel] = x1[non_parallel] + t[non_parallel] * (x2[non_parallel] - x1[non_parallel])
    py[non_parallel] = y1[non_parallel] + t[non_parallel] * (y2[non_parallel] - y1[non_parallel])
    p = np.vstack([px, py]).T

    p = p.reshape(lines1.shape[0], lines2.shape[0], 2)
    origins = np.tile(lines2[:, 0], (lines1.shape[0], 1, 1))
    distances = np.sqrt((p[:, :, 0] - origins[:, :, 0]) ** 2 + (p[:, :, 1] - origins[:, :, 1]) ** 2)

    valid_indices = (non_parallel & (0 <= t) & (t <= 1) & (0 <= u) & (u <= 1)).reshape(lines1.shape[0], lines2.shape[0])

    if return_whether_intersect:
        return valid_indices

    distances[~valid_indices] = np.inf

    min_values = np.min(distances, axis=0)
    min_indices = np.argmin(distances, axis=0)
    min_pair_indices = np.vstack([min_indices, np.arange(lines2.shape[0])]).T
    mpi_valid = min_pair_indices[~np.isinf(min_values)]

    if overwrite:
        lines2[~np.isinf(min_values), 1, :] = p[mpi_valid[:, 0], mpi_valid[:, 1]]
        return lines2

    output = lines2.copy()
    output[~np.isinf(min_values), 1, :] = p[mpi_valid[:, 0], mpi_valid[:, 1]]
    return output
