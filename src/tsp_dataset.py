# code based in part on
# http://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
# and from
# https://github.com/devsisters/neural-combinatorial-rl-tensorflow/blob/master/data_loader.py
import requests
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch
import os
import numpy as np
import re
import zipfile
import itertools
from collections import namedtuple
import gdown


#######################################
# Reward Fn
#######################################
def reward(sample_solution, USE_CUDA=False):
    """
    Args:
        List of length sourceL of [batch_size] Tensors
    Returns:
        Tensor of shape [batch_size] containins rewards
    """
    batch_size = sample_solution[0].size(0)
    n = len(sample_solution)
    tour_len = Variable(torch.zeros([batch_size]))
    
    if USE_CUDA:
        tour_len = tour_len.cuda()

    for i in range(n-1):
        tour_len += torch.norm(sample_solution[i] - sample_solution[i+1], dim=1)
    
    tour_len += torch.norm(sample_solution[n-1] - sample_solution[0], dim=1)

    return tour_len


#######################################
# Functions for downloading dataset
#######################################
TSP = namedtuple('TSP', ['x', 'y', 'name'])

GOOGLE_DRIVE_IDS = {
    'tsp5_train.zip': '0B2fg8yPGn2TCSW1pNTJMXzFPYTg',
    'tsp10_train.zip': '0B2fg8yPGn2TCbHowM0hfOTJCNkU',
    'tsp5-20_train.zip': '0B2fg8yPGn2TCTWNxX21jTDBGeXc',
    'tsp50_train.zip': '0B2fg8yPGn2TCaVQxSl9ab29QajA',
    'tsp20_test.txt': '0B2fg8yPGn2TCdF9TUU5DZVNCNjQ',
    'tsp40_test.txt': '0B2fg8yPGn2TCcjFrYk85SGFVNlU',
    'tsp50_test.txt.zip': '0B2fg8yPGn2TCUVlCQmQtelpZTTQ',
}

def download_file_from_google_drive(file_id, destination):
    gdown.download(f'https://drive.google.com/uc?id={file_id}', destination, quiet=False)
    return True


def get_confirm_token(html):
    # Regex to extract confirmation token from the warning page
    match = re.search(r"confirm=([0-9A-Za-z_]+)", html)
    if match:
        return match.group(1)
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE)):
            if chunk: # filter out keep-alive new chunks
                 f.write(chunk)

def download_google_drive_file(data_dir, task, min_length, max_length):
    paths = {}
    for mode in ['train', 'test']:
        candidates = []
        candidates.append(
            '{}{}_{}'.format(task, max_length, mode))
        candidates.append(
            '{}{}-{}_{}'.format(task, min_length, max_length, mode))

        for key in candidates:
            print(key)
            for search_key in GOOGLE_DRIVE_IDS.keys():
                if search_key.startswith(key):
                    path = os.path.join(data_dir, search_key)
                    print("Download dataset of the paper to {}".format(path))

                    if not os.path.exists(path):
                        download_file_from_google_drive(GOOGLE_DRIVE_IDS[search_key], path)
                    if path.endswith('zip'):
                        with zipfile.ZipFile(path, 'r') as z:
                            z.extractall(data_dir)
                    paths[mode] = path

    return paths

def read_paper_dataset(paths):
    x, y = [], []
    for path in paths:
        print("Read dataset {} which is used in the paper..".format(path))
        #length = max(re.findall('\d+', path))
        with open(path) as f:
            for l in tqdm(f):
                inputs, outputs = l.split(' output ')
                x.append(np.array(inputs.split(), dtype=np.float32).reshape([-1, 2]))
                y.append(np.array(outputs.split(), dtype=np.int32)[:-1]) # skip the last one

    return x, y


def save_numpy_dataset(dataset_name, dataset_directory):
    dataset_numpy_directory = "./dataset/tsp/numpy/"
    os.system(f"mkdir -p {dataset_numpy_directory}")
    
    p = f"{dataset_directory}{dataset_name}.txt"

    x, y = read_paper_dataset([p])
    x, y = np.stack(x), np.stack(y)
    
    np.save(f"{dataset_numpy_directory}/{dataset_name}_x.npy", x)
    np.save(f"{dataset_numpy_directory}/{dataset_name}_y.npy", y)    
    
def load_numpy_dataset(dataset_name):
    dataset_numpy_directory = "./dataset/tsp/numpy/"    
    try:  
        x = np.load(f"{dataset_numpy_directory}/{dataset_name}_x.npy")
        y = np.load(f"{dataset_numpy_directory}/{dataset_name}_y.npy")
        
    except:
        dataset_directory = "./dataset/tsp/"
        save_numpy_dataset(dataset_name, dataset_directory)
        x = np.load(f"{dataset_numpy_directory}/{dataset_name}_x.npy")
        y = np.load(f"{dataset_numpy_directory}/{dataset_name}_y.npy")
    
    return x, y

def modify_for_ptrnet(x, y):
    x_with_third_zero = np.zeros((x.shape[0], x.shape[1], 3), dtype=x.dtype)
    x_with_third_zero[:, :, :2] = x.copy()
    y -= 1
    return x_with_third_zero, y 


class TSPDataset(torch.utils.data.Dataset):
    def __init__(self, coord_tensors, tour_tensors):
        self.samples = []

        for coords, tours in zip(coord_tensors, tour_tensors):
            for c, t in zip(coords, tours):
                self.samples.append({
                    'coords': torch.tensor(c, dtype=torch.float),  # shape [n, 2]
                    'tour': torch.tensor(t, dtype=torch.long)      # shape [n]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'coords': sample['coords'],
            'tour': sample['tour'],
            'length': sample['coords'].size(0)
        }

eos = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)

def collate_fn(batch):
    coords = [item['coords'] for item in batch]
    tours = [item['tour'] for item in batch]
    
    # Normalize per sample
    for i in range(len(coords)):
        max_val = coords[i][:, :2].abs().max()
        coords[i][:, :2] /= max_val
    
    # Prepend EOS token to coords
    coords = tuple(torch.cat((eos.unsqueeze(0), coord)) for coord in coords)
    
    # Shift tour indices by 1 and append EOS (index 0)
    tours = tuple(torch.cat((pos + 1, torch.tensor([0]))) for coord, pos in zip(coords, tours))
    
    # Compute lengths (optional)
    lengths = torch.tensor([coord.shape[0] for coord in coords], dtype=torch.long)
    
    # Pad sequences
    coords_padded = torch.nn.utils.rnn.pad_sequence(coords, batch_first=False, padding_value=-1.0)
    tours_padded = torch.nn.utils.rnn.pad_sequence(tours, batch_first=False, padding_value=-1)
    
    return coords_padded, tours_padded, lengths

def tsp_examples():
    a_sample = True
    num_samples_each_dataset = 1000

    dataset_names = ["tsp_all_len5"]#, "tsp_all_len7", "tsp_all_len9"]
    
    coord_tensors, tour_tensors = [], []

    for dataset_name in dataset_names:
        coords, tours = modify_for_ptrnet(*load_numpy_dataset(dataset_name))
        if a_sample:
            coords, tours = coords[:num_samples_each_dataset], tours[:num_samples_each_dataset]

        coord_tensors.append(coords)
        tour_tensors.append(tours)

    return coord_tensors, tour_tensors



if __name__ == '__main__':
    
    dataset_directory = "./dataset/tsp/"
    os.system(f"mkdir -p {dataset_directory}")
    paths = download_google_drive_file(dataset_directory[:-1], 'tsp', '', '5')
    paths = download_google_drive_file(dataset_directory[:-1], 'tsp', '5', '20')

    coord_tensors, tour_tensors = tsp_examples()

    dataset = TSPDataset(
        coord_tensors=coord_tensors,
        tour_tensors=tour_tensors
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    for batch in dataloader:
        coords, tours, lengths = batch
        
        seq = coords
        seq_lens = lengths
        positions = tours

        
    
