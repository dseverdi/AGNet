import torch
from torch.utils.data import Dataset as TorchDataset
from typing import List, Optional, Any
import os

class Sample:
    """
    Generic sample class for any problem (AGP, etc).
    """
    def __init__(self, data: Any, label: Any = None, name: str = ""):
        self.data = data  # e.g., polygon points, etc.
        self.label = label  # e.g., guards, etc.
        self.name = name



class Dataset(TorchDataset):
    """
    Dataset class for any problem.
    """
    def __init__(self, samples: Optional[List[Sample]] = None):
        self.samples = samples if samples else []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample.data, sample.label, sample.name

    def extend(self, other):
        if isinstance(other, list):
            for l in other:
                self.samples.extend(l.samples)

    @property
    def data(self):
        """Return a list of all sample.data tensors."""
        return [sample.data for sample in self.samples]

# AGP-specific sample loading (placeholder, adapt as needed)
def agp_read_samples(paths: List[str], normalize: bool = False) -> List[Sample]:
    """
    Read AGP samples from a list of .pol file paths.
    Format: First line may contain a comment (starts with #) or the number of points.
    Then N consecutive lines, each with "x y" coordinates.
    Each sample contains only polygon points and the instance name.
    """
    samples = []
    for path in paths:
        with open(path, 'r') as f:
            tokens = f.read().split()
            num_points = int(tokens[0])
            points = []
            for i in range(1, 2 * num_points, 2):
                x_token = tokens[i]
                y_token = tokens[i + 1]
                # Parse x coordinate (handle x/1 format)
                if '/' in x_token:
                    x_num, x_denom = map(float, x_token.split('/'))
                    x = x_num / x_denom if x_denom != 0 else 0.0
                else:
                    x = float(x_token)
                # Parse y coordinate (handle y/1 format)
                if '/' in y_token:
                    y_num, y_denom = map(float, y_token.split('/'))
                    y = y_num / y_denom if y_denom != 0 else 0.0
                else:
                    y = float(y_token)
                points.append((x, y))
            points_tensor = torch.tensor(points, dtype=torch.float32)
            if normalize:
                min_xy = points_tensor.min(dim=0)[0]
                max_xy = points_tensor.max(dim=0)[0]
                denom = (max_xy - min_xy)
                denom[denom == 0] = 1.0  # avoid division by zero
                points_tensor = (points_tensor - min_xy) / denom
            name = os.path.splitext(os.path.basename(path))[0]
            samples.append(Sample(data=points_tensor, label=None, name=name))
    return samples

def collate_fn(batch):
    """
    Collate function for batching variable-length polygons.
    Pads all polygons in the batch to the same number of vertices and returns a mask.
    """
    datas, labels, names = zip(*batch)
    lengths = [d.shape[0] for d in datas]
    # Pad variable-length polygons (each data is [num_points, 2])
    datas_padded = torch.nn.utils.rnn.pad_sequence(datas, batch_first=True, padding_value=0.0)
    # Create mask: True for real vertices, False for padding
    max_len = datas_padded.size(1)
    mask = torch.zeros(len(datas), max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
    # Also return true lengths of each polygon for per-sample EOS
    return datas_padded, mask, lengths, names



if __name__ == "__main__":
    # Test AGP sample reading from a folder or file
    agp_folder = "/home/dseverdi/Radno/MLAG/dataset/AGPIL/train/rand-116-103.pol"
    # Allow agp_folder to be a file or a directory
    if os.path.isfile(agp_folder):
        agp_paths = [agp_folder]
    else:
        agp_paths = [os.path.join(agp_folder, f) for f in os.listdir(agp_folder) if f.endswith('.pol')]
    print(f"Found {len(agp_paths)} AGP .pol files.")
    samples = agp_read_samples(agp_paths, normalize=True)
    dataset = Dataset(samples)
    print(f"Loaded {len(dataset)} samples.")
   
   
    # Test DataLoader batching
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
    for batch_idx, (batch_data, batch_names) in enumerate(loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  batch_data shape: {batch_data.shape}")
        print(f"  batch_names: {batch_names}")
        print(f"  batch_data[0]:\n{batch_data[0]}")
        break

