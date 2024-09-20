import torch
import pandas as pd
import os
from data_types import VisSample

if __name__ == "__main__":
    dataset_dir = '/mnt/nvme0n1/dseverdi/MLAG/dataset/AG/development/'
    samples = {s : [f for f in os.listdir(dataset_dir+s) if f.endswith('.pol')] for s in ['train','dev','test']}
    df = pd.DataFrame.from_dict(samples, orient='index').transpose()
    # Add a row with the total count per column
    df.loc['Total # Instances'] = df.count()

    # display the dataframe
    # print(dt)
    
    instance_type = "test"
    instance_name = "fat-12-1.pol"
    
    #instance_name = df.at[0, instance_type]
    instance_path = os.path.join(dataset_dir, f"{instance_type}/{instance_name}")
    samples = VisSample.read_samples(path=instance_path, sol_sample=0)[0:2]
    for sample in samples:
        print(sample)
        print()
    
    