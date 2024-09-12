

#!/opt/anaconda3/bin/python

"""
preparing solutions for AG
"""

import numpy as np
# clustering
from sklearn.cluster import KMeans

# our data
from data_types  import VisSample





import os
import argparse

from pathlib import Path

def solve(dataset_dir):
    """
        solving instances
    """
    
    paths = [os.path.abspath(os.path.join(dirpath,f)) for dirpath,_,filenames in os.walk(dataset_dir) for f in filenames]

    if not paths:
        print('Not valid instances path!')    
    for path in paths:        
        samples = VisSample.read_samples(path = path, solve = True)

def no_solve(dataset_dir):
    paths = [os.path.abspath(os.path.join(dirpath,f)) for dirpath,_,filenames in os.walk(dataset_dir) for f in filenames]
    print('Generating empty solutions.')
    for path in paths:
        samples = VisSample.read_samples(path = path,solve = False)


def cluster(dataset_dir,n_clusters=4):

    """
    cluster solutions of intances to n_clusters
    """    

    paths = [dataset_dir] if os.path.isfile(dataset_dir) else [os.path.abspath(os.path.join(dirpath,f)) for dirpath,_,filenames in os.walk(dataset_dir) for f in filenames if f.endswith('.terrain')]
    
    
    for path in paths:         
        samples = VisSample.read_samples(path = path, sol_sample = 0)
        print(f'Finding solution clusters for {path}',end='\r',flush=True)
        n_solutions = len(samples)               
        # data to clusters        
        data = np.array([t.guards.tolist() for t in samples])        
        # cluster if enough solutions       
        if n_solutions > n_clusters:            
            try:            
                kmeans = KMeans(n_clusters=n_clusters).fit(data)
                solution = [data[np.where(kmeans.labels_==l)[0][0] ] for l in range(kmeans.n_clusters)]            
            except Exception as e:
                import pdb
                pdb.set_trace()
        else: 
            solution = data

        # dump to solution
        cluster_root, cluster_name = Path(path).parent, Path(path).stem
        cluster_file = os.path.join(cluster_root,cluster_name)
        
        if os.path.isfile(cluster_file) : continue
        
        num_clusters = len(solution)

        with open(f"{cluster_root}/{cluster_name}.clusters",'w') as f:                    
            f.write(f'# clusters {num_clusters}\n')
            
            for sol in solution: 
                for g in sol : f.write(f"{g } ")
                f.write('\n')   



    
if __name__ == '__main__':
    dataset_dir = '/home/dseverdi/Radno/MLAG/dataset/AG/agp2009a-simplerand/20'
    #sizes = ['5']

    func_map = {'solve' : solve, 'no-solve' : no_solve, 'cluster' : cluster }



    parser = argparse.ArgumentParser(description='convert AG/TGPIL instances')
    parser.add_argument('command', choices = func_map.keys(),help='solve, cluster or no solve')
    parser.add_argument('--n_clusters',type=int,default=10, help='number of clusters')
    parser.add_argument('--dataset_dir',type=str,default=dataset_dir, help='dataset root folder')
    

    # parse arguments
    args = parser.parse_args()
    func = func_map[args.command]

    
    
    # call function
    if func == cluster:
        func(args.dataset_dir,n_clusters=args.n_clusters)
    else:
        func(args.dataset_dir)
    
    