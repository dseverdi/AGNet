import numpy as np
import torch
import pandas as pd
import os
from tqdm import tqdm

from torch.utils.data import DataLoader
from data_types import VisSample, VisDataset, collate_fn

from demo import evaluate_polygon_visibility, evaluate_polygon_visibility_numpy, CustomError

from PtrNet import PointerNetwork

def get_key_by_value(d, target_string):
    for key, value_list in d.items():
        if target_string in value_list:
            return key
    return None 

def load_dataframe(dataset_dir):
    samples = {s : [f for f in os.listdir(f"{dataset_dir}/{s}") if f.endswith('.pol')] for s in ['train','dev','test']}
    df = pd.DataFrame.from_dict(samples, orient='index').transpose()
    
    #df.loc['Total # Instances'] = df.count()

    return df

def load_vis_dataset(dataset_dir):
    df = load_dataframe(dataset_dir)

    sample_size = 10000
    train_path = f"{dataset_dir}/train"
    
    training_set = VisDataset()
    paths = [f"{train_path}/{filename}" for filename in df["train"].tolist()][0:sample_size]

    vis_samples = [VisSample.read_samples(path=path,normalize = True)[:sample_size] for path in paths]
    training_set.extend([VisDataset(VisSample.read_samples(path=path,normalize = True)[:sample_size]) for path in paths])
    
    return training_set, vis_samples

def example_load_sample(dataset_dir):
    instance_type = "test"
    instance_name = "fat-12-1.pol"
    
    instance_path = os.path.join(dataset_dir, f"{instance_type}/{instance_name}")
    samples = VisSample.read_samples(path=instance_path, sol_sample=0)#[0:10000]
    dataset = VisDataset(samples)
    
    return samples, dataset
    
def example_eval_all_vis_samples_and_report_errors(dataset_dir):
    training_set, vis_samples = load_vis_dataset(dataset_dir)
    
    for i in tqdm(range(len(vis_samples))):        
        
        print('\nEvaluating:', vis_samples[i][0].name)
        vis_sample = vis_samples[i][0]

        #if vis_sample.name != "min-8-1.pol": continue
        
        try:
            valid_positions = vis_sample.guards            
            #suboptimal_positions = np.random.choice(valid_positions, np.random.randint(1, len(valid_positions)))
            all_positions = np.arange(len(vis_sample.points))
            # print(f"all positions: {all_positions}")
        except Exception as e:
            print("Error getting all_positions", len(vis_sample.points))
        
        try:
            try:
                print(' * Evaluating with solution ...', end='')
                poly, region, predicted, opt, coverage = evaluate_polygon_visibility(vis_sample, valid_positions)                
                print('done')
                if coverage < 0.9:
                    with open('debug/output.txt', 'a') as f:
                        f.write(f"{vis_sample.name}: {coverage}\n")
                
            except Exception as e:
                print(f"Error evaluating coverage_1: {e}")
                print(f"Tested with guards: {valid_positions}")
                raise CustomError("Ground-truth coverage error (type 1)")
                
            try:                
                print(' * Evaluating with all points ...', end=' ')
                poly, region, predicted, opt, coverage = evaluate_polygon_visibility(vis_sample, all_positions) # problem: rand-172-166.pol
                print('done')
            except:
                raise CustomError("All points coverage error (type 2)")
            
            """
            try:
                _, _, _, _, coverage_3 = evaluate_polygon_visibility(vis_sample, suboptimal_positions)
            except:
                raise CustomError("Suboptimal coverage error (type 2)")            
            """
        except CustomError as e:
            print(f"{e}: {vis_sample.name}")

def example_eval_all_samples_via_pytorch_dataloader_and_report_errors(dataset_dir):

    training_set, vis_samples = load_vis_dataset(dataset_dir)
    
    dataloader_params = {'batch_size': 1, 'shuffle': False, 'num_workers' : 1}    
    
    training_generator = DataLoader(training_set, **dataloader_params, collate_fn=collate_fn)
    
    def batch_eval_visibility(seq, seq_lens, positions):
        
        def get_valid_positions(positions_sample):
            positions_sample_np = positions_sample.numpy()
            positions_valid = positions_sample_np[positions_sample_np != -1]
            positions_valid = positions_valid[positions_valid != 0]
            positions_valid -= 1
            return positions_valid        
        
        points_batch = seq[1:, :, :2].numpy()
        batch_size = points_batch.shape[1]
        for i in range(batch_size):
            seq_len = seq_lens[i] - 1
            points = points_batch[:seq_len, i]
            valid_positions = get_valid_positions(positions[:, i])
            #suboptimal_positions = np.random.choice(valid_positions, np.random.randint(1, len(valid_positions)))
            all_positions = np.arange(seq_len)
            
            try:
                # test for optimal solution
                coverage_1 = evaluate_polygon_visibility_numpy(points, valid_positions, valid_positions)
            except:
                raise CustomError("Ground-truth coverage error (type 1)")

            try:
                # test for all points
                coverage_2 = evaluate_polygon_visibility_numpy(points, valid_positions, all_positions)
            except:
                raise CustomError("All points coverage error (type 2)")
            
            """
            try:
                coverage_3 = evaluate_polygon_visibility_numpy(points, valid_positions, suboptimal_positions)
            except:
                raise CustomError("Suboptimal coverage error (type 3)")
            """

    
    i = 0
    #for (seq, seq_lens, positions) in tqdm(training_generator):
    for (seq, seq_lens, positions) in training_generator:
        try:            
            batch_eval_visibility(seq, seq_lens, positions)
        except CustomError as e:
            vis_sample = vis_samples[i][0]
            print(f"{e}: {vis_sample.name}")
        except Exception as e:
            print(f"Error: {e}")
            vis_sample = vis_samples[i][0]
            print(f"Error in {vis_sample.name}")
            
        i += 1

if __name__ == "__main__":
    #dataset_dir = 'dataset/development'
    dataset_dir = '/mnt/nvme0n1/dseverdi/MLAG/dataset/AG/development'
    #dataset_dir = "/home/jurica/Desktop/AGNet/dataset/development"

    #example_eval_all_vis_samples_and_report_errors(dataset_dir)    
    example_eval_all_samples_via_pytorch_dataloader_and_report_errors(dataset_dir)
        
    
    