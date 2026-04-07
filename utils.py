import json
import torch
import math
import warnings
import pickle

def save_json(filepath, d):
    j = json.dumps(d, indent=4)
    with open(filepath, 'w') as f:
        f.write(j)

def load_json(filepath):
    with open(filepath) as f:
        d = json.load(f)
    return d

def save_pickle(filepath, o):
    with open(filepath, "wb") as f:
        pickle.dump(o, f)
        
def load_pickle(filepath):
    return pickle.load(open(filepath, "rb"))

class BatchIterator:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.n = len(data)
        self.n_batches = int(math.ceil(self.n / self.batch_size))
        self.batch_size_last = self.n - (self.n_batches - 1) * self.batch_size

    def __iter__(self):
        self.batch_i = 0
        return self

    def __next__(self):
        if self.batch_i >= self.n_batches:
            raise StopIteration
        start = self.batch_i * self.batch_size
        end = start + (self.batch_size if self.batch_i < self.n_batches - 1 else self.batch_size_last)
        self.batch_i += 1
        return self.data[start:end]

class MultiBatchIterator:
    def __init__(self, batch_size, *data):
        self.data = data
        self.batch_size = batch_size
        self.n = len(data[0])
        self.n_batches = int(math.ceil(self.n / self.batch_size))
        self.batch_size_last = self.n - (self.n_batches - 1) * self.batch_size

    def __iter__(self):
        self.batch_i = 0
        return self

    def __next__(self):
        if self.batch_i >= self.n_batches:
            raise StopIteration
        start = self.batch_i * self.batch_size
        end = start + (self.batch_size if self.batch_i < self.n_batches - 1 else self.batch_size_last)
        self.batch_i += 1
        return tuple(d[start:end] for d in self.data)

def print_nice_dict(d, indent=4):
    print(json.dumps(d, indent=indent))

def filter_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")

    
def save_string_to_file(filepath, string):
    with open(filepath, 'w') as f:
        f.write(string)
        
if __name__ == "__main__":
    # Generate a lorem ipsum string of 100 words
    lorem_ipsum = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.\n"
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n"
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi\n"
        "ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit\n"
        "in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint\n"
        "occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim\n"
        "id est laborum."
    )

    # Save the lorem ipsum string to a file
    save_string_to_file("/home/jurica/Desktop/sortness/lorem.txt", lorem_ipsum)
    