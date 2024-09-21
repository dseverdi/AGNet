import numpy as np
import torch
import pandas as pd
import os

from torch.utils.data import DataLoader
from data_types import VisSample, VisDataset, collate_fn

from demo import evaluate_polygon_visibility, evaluate_polygon_visibility_numpy

if __name__ == "__main__":
    #dataset_dir = '/mnt/nvme0n1/dseverdi/MLAG/dataset/AG/development'
    dataset_dir = "/home/jurica/Desktop/AGNet/dataset/development"
    samples = {s : [f for f in os.listdir(f"{dataset_dir}/{s}") if f.endswith('.pol')] for s in ['train','dev','test']}
    df = pd.DataFrame.from_dict(samples, orient='index').transpose()
    # Add a row with the total count per column
    df.loc['Total # Instances'] = df.count()

    # display the dataframe
    # print(dt)
    
    dataloader_params = {'batch_size': 3,
                         'shuffle': True,
                         'num_workers' : 1}
        
    
    """
    instance_type = "test"
    instance_name = "fat-12-1.pol"
    
    #instance_name = df.at[0, instance_type]
    instance_path = os.path.join(dataset_dir, f"{instance_type}/{instance_name}")
    samples = VisSample.read_samples(path=instance_path, sol_sample=0)#[0:10000]
    dataset = VisDataset(samples)
    """
    
    sample_size = 10000
    training_set = VisDataset()
    train_path = "/home/jurica/Desktop/AGNet/dataset/development/train"
    paths = [f"{train_path}/{filename}" for filename in df["train"].tolist()][0:10]
    #print(paths)
    #exit()
    training_set.extend([VisDataset(VisSample.read_samples(path=path,normalize = True)[:sample_size]) for path in paths])  
    training_generator = DataLoader(training_set, **dataloader_params, collate_fn=collate_fn)     
    

    training_generator = DataLoader(training_set, **dataloader_params, collate_fn=collate_fn) 
    
    def get_valid_positions(positions_sample):
        positions_sample_np = positions_sample.numpy()
        positions_valid = positions_sample_np[positions_sample_np != -1]
        positions_valid = positions_valid[positions_valid != 0]
        positions_valid -= 1
        return positions_valid
            
    
    def batch_eval_visibility(seq, seq_lens, positions):
        points_batch = seq[1:, :, :2].numpy()
        batch_size = points_batch.shape[1]
        for i in range(batch_size):
            seq_len = seq_lens[i] - 1
            points = points_batch[:seq_len, i]
            valid_positions = get_valid_positions(positions[:, i])
            suboptimal_positions = np.random.choice(valid_positions, np.random.randint(1, len(valid_positions)))
            coverage_1 = evaluate_polygon_visibility_numpy(points, valid_positions, valid_positions)
            coverage_2 = evaluate_polygon_visibility_numpy(points, valid_positions, suboptimal_positions)
            print(coverage_1, coverage_2)
        print()
    
    for (seq, seq_lens, positions) in training_generator:
        batch_eval_visibility(seq, seq_lens, positions)
        
    print("------------------")
    exit()        
    
    for sample in samples:
        solution = sample.guards.numpy().copy()
        res = evaluate_polygon_visibility(sample, solution)
        print(res[4])
        print()
    
    