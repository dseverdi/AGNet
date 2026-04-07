"""PyTorch Dataset for AGP .pt chunk datasets (classic vertex-guard)."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import colors


class AGPClassicDataset(Dataset):
    """Loads all .pt chunk files from a single dataset directory into memory."""

    def __init__(self, ds_dir: str | Path):
        ds_dir = Path(ds_dir)
        self.data: list[dict] = []
        for cp in sorted(ds_dir.glob("chunk_*.pt")):
            self.data.extend(torch.load(cp, weights_only=False))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return d["vertices"], d["guards"], d["n_vertices"], d["n_guards"]


def agp_collate_fn(batch):
    """Collate variable-length rooms into padded tensors.

    Returns
    -------
    vertices : (max_verts, batch, 2)   padded vertex coordinates
    seq_lens : (batch,)                actual vertex count per room
    guards   : (batch, max_guards)     ground-truth guard indices (padded with -1)
    n_guards : (batch,)                number of ground-truth guards
    """
    vertices_list, guards_list, n_verts, n_guards = zip(*batch)

    seq_lens = torch.tensor(n_verts, dtype=torch.long)
    vertices_padded = pad_sequence(
        list(vertices_list), batch_first=False, padding_value=0.0)

    max_g = max(g.shape[0] for g in guards_list)
    guards_padded = torch.full((len(batch), max_g), -1, dtype=torch.long)
    for i, g in enumerate(guards_list):
        guards_padded[i, : g.shape[0]] = g

    n_guards_t = torch.tensor(n_guards, dtype=torch.long)
    return vertices_padded, seq_lens, guards_padded, n_guards_t


def load_classic_datasets(
    data_dir: str | Path = "data",
    chunk_size: int = 100,
    ds_names: tuple[str, ...] = ("dataset_1", "dataset_2"),
) -> ConcatDataset:
    """Load and blend classic AGP chunk datasets.

    Looks for ``data/<ds_name>_classic_<chunk_size>/chunk_*.pt``.
    """
    data_dir = Path(data_dir)
    datasets: list[AGPClassicDataset] = []

    for name in ds_names:
        p = data_dir / f"{name}_classic_{chunk_size}"
        if p.exists():
            ds = AGPClassicDataset(p)
            print(f"Loaded {p.name}: {len(ds)} rooms")
            datasets.append(ds)
        else:
            print(f"Warning: {p} not found, skipping")

    if not datasets:
        raise FileNotFoundError(
            f"No classic datasets found in {data_dir} with chunk_size={chunk_size}")
    return ConcatDataset(datasets)


# ── batch plotting ────────────────────────────────────────────────────────────

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


def plot_batch(
    vertices_padded,
    seq_lens,
    guards_padded,
    output_path: str | Path = "data/images/batch_plot/batch.png",
):
    """Plot a batch of rooms with their guards as a grid image.

    Args:
        vertices_padded: (max_verts, batch, 2)
        seq_lens:        (batch,)
        guards_padded:   (batch, max_guards)  — padded with -1
        output_path:     where to save the figure
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    batch_size = vertices_padded.size(1)
    ncols = min(batch_size, 4)
    nrows = (batch_size + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=150)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for i in range(batch_size):
        ax = axes[i]
        n = seq_lens[i].item()
        room = vertices_padded[:n, i].cpu().numpy()
        guards = guards_padded[i]
        guards = guards[guards >= 0].cpu().numpy()

        # Room boundary
        closed = np.vstack([room, room[:1]])
        ax.plot(closed[:, 0], closed[:, 1],
                color=colors.blue, linewidth=1.0, linestyle="--", zorder=2)

        from main_classic_gt import _nudge_inward, compute_visibility_polygon
        from shapely.geometry import Polygon as ShapelyPolygon
        room_shapely = ShapelyPolygon(room)

        for gi, g_idx in enumerate(guards):
            strong, pale = GUARD_COLORS[gi % len(GUARD_COLORS)]
            origin = _nudge_inward(room, int(g_idx))
            vp = compute_visibility_polygon(room, origin)
            if vp.shape[0] >= 3:
                vp_shapely = ShapelyPolygon(vp)
                clipped = room_shapely.intersection(vp_shapely)
                if not clipped.is_empty:
                    geoms = [clipped] if clipped.geom_type == "Polygon" else list(clipped.geoms)
                    for geom in geoms:
                        if geom.geom_type == "Polygon" and not geom.is_empty:
                            patch = MplPolygon(
                                np.array(geom.exterior.coords), closed=True,
                                facecolor=pale, edgecolor=strong,
                                linewidth=0.4, alpha=0.45, zorder=1)
                            ax.add_patch(patch)
            ax.scatter([room[g_idx, 0]], [room[g_idx, 1]],
                       s=60, c=strong, zorder=5, marker="*")

        # Non-guard vertices
        non_guard = [v for v in range(n) if v not in guards]
        if non_guard:
            ax.scatter(room[non_guard, 0], room[non_guard, 1],
                       s=14, c=colors.blue, zorder=3)

        ax.set_aspect("equal", adjustable="box")
        margin = 1.0
        ax.set_xlim(room[:, 0].min() - margin, room[:, 0].max() + margin)
        ax.set_ylim(room[:, 1].min() - margin, room[:, 1].max() + margin)
        ax.set_axisbelow(True)
        ax.grid(True, color=colors.lightgray, linewidth=0.5, alpha=0.6)
        ax.set_title(f"Room {i+1} — {len(guards)} guards", fontsize=9)

    for j in range(batch_size, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout(pad=0.5)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Saved batch plot → {output_path}")


# ── example ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ds = load_classic_datasets("data", chunk_size=100)
    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=agp_collate_fn)
    vertices, seq_lens, guards, n_guards = next(iter(loader))
    plot_batch(vertices, seq_lens, guards)
