# code based in part on
# http://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
# and from
# https://github.com/devsisters/neural-combinatorial-rl-tensorflow/blob/master/data_loader.py
import requests
from tqdm import tqdm
from torch.utils.data import Dataset
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

    # For TSP_20 - map to a number between 0 and 1
    # min_len = 3.5
    # max_len = 10.
    # TODO: generalize this for any TSP size
    #tour_len = -0.1538*tour_len + 1.538 
    #tour_len[tour_len < 0.] = 0.
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

def maybe_generate_and_save(self, except_list=[]):
    data = {}

    for name, num in self.data_num.items():
        if name in except_list:
            print("Skip creating {} because of given except_list {}".format(name, except_list))
            continue
        path = self.get_path(name)

        print("Skip creating {} for [{}]".format(path, self.task))
        tmp = np.load(path)
        self.data[name] = TSP(x=tmp['x'], y=tmp['y'], name=name)

def get_path(self, name):
    return os.path.join(
        self.data_dir, "{}_{}={}.npz".format(
            self.task_name, name, self.data_num[name]))

def read_zip_and_update_data(self, path, name):
    if path.endswith('zip'):
        filenames = zipfile.ZipFile(path).namelist()
        paths = [os.path.join(self.data_dir, filename) for filename in filenames]
    else:
        paths = [path]

    x_list, y_list = read_paper_dataset(paths, self.max_length)

    x = np.zeros([len(x_list), self.max_length, 2], dtype=np.float32)
    y = np.zeros([len(y_list), self.max_length], dtype=np.int32)

    for idx, (nodes, res) in enumerate(tqdm(zip(x_list, y_list))):
        x[idx,:len(nodes)] = nodes
        y[idx,:len(res)] = res

    if self.data is None:
        self.data = {}

    print("Update [{}] data with {} used in the paper".format(name, path))
    self.data[name] = TSP(x=x, y=y, name=name)


def create_dataset(
    problem_size, 
    data_dir):

    def find_or_return_empty(data_dir, problem_size):
        #train_fname1 = os.path.join(data_dir, 'tsp{}.txt'.format(problem_size))
        val_fname1 = os.path.join(data_dir, 'tsp{}_test.txt'.format(problem_size))
        #train_fname2 = os.path.join(data_dir, 'tsp-{}.txt'.format(problem_size))
        val_fname2 = os.path.join(data_dir, 'tsp-{}_test.txt'.format(problem_size))
        
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        else:
    #         if os.path.exists(train_fname1) and os.path.exists(val_fname1):
    #             return train_fname1, val_fname1
    #         if os.path.exists(train_fname2) and os.path.exists(val_fname2):
    #             return train_fname2, val_fname2
    #     return None, None

    # train, val = find_or_return_empty(data_dir, problem_size)
    # if train is None and val is None:
    #     download_google_drive_file(data_dir,
    #         'tsp', '', problem_size) 
    #     train, val = find_or_return_empty(data_dir, problem_size)

    # return train, val
            if os.path.exists(val_fname1):
                return val_fname1
            if os.path.exists(val_fname2):
                return val_fname2
        return None

    val = find_or_return_empty(data_dir, problem_size)
    if val is None:
        download_google_drive_file(data_dir, 'tsp', '', problem_size)
        val = find_or_return_empty(data_dir, problem_size)

    return val


#######################################
# Dataset
#######################################
class TSPDataset(Dataset):
    
    def __init__(self, dataset_fname=None, train=False, size=50, num_samples=1000000, random_seed=1111):
        super(TSPDataset, self).__init__()
        #start = torch.FloatTensor([[-1], [-1]]) 
        
        torch.manual_seed(random_seed)

        self.data_set = []
        if not train:
            with open(dataset_fname, 'r') as dset:
                for l in tqdm(dset):
                    inputs, outputs = l.split(' output ')
                    sample = torch.zeros(1, )
                    x = np.array(inputs.split(), dtype=np.float32).reshape([-1, 2]).T
                    #y.append(np.array(outputs.split(), dtype=np.int32)[:-1]) # skip the last one
                    self.data_set.append(x)
        else:
            # randomly sample points uniformly from [0, 1]
            for l in tqdm(range(num_samples)):
                x = torch.FloatTensor(2, size).uniform_(0, 1)
                #x = torch.cat([start, x], 1)
                self.data_set.append(x)

        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]

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
    x = np.load(f"{dataset_numpy_directory}/{dataset_name}_x.npy")
    y = np.load(f"{dataset_numpy_directory}/{dataset_name}_y.npy")
    
    return x, y

if __name__ == '__main__':
    
    dataset_directory = "./dataset/tsp/"
    #os.system(f"mkdir -p {dataset_directory}")
    #paths = download_google_drive_file(dataset_directory[:-1], 'tsp', '', '5')
    #paths = download_google_drive_file(dataset_directory[:-1], 'tsp', '5', '20')
    
    dataset_name = "tsp5"
        

    #save_numpy_dataset(dataset_name, dataset_directory)
    x, y = load_numpy_dataset(dataset_name)
    
    print(x.shape, y.shape)
    print(x[0], y[0])    