import os
from pathlib import Path

# terrain dataset:
# http://resources.mpi-inf.mpg.de/tgp/index.html#instances

# pytorch ...
import  numpy                as np
import  torch
import  torch.nn             as nn
from    torch            import tensor
from    torch.utils.data import Dataset
import  pickle
import  numpy                as np


# geometry 
import skgeom
from terrain import Terrain, TerrainSolver
from polygon import Polygon, AGSolver


# types
from dataclasses import dataclass
from torch.utils.data import DataLoader


# -----------------------------------------------

default_sizes = [5, 10, 15]

# windowsing function
# scaling
scale = lambda x,a,b : (x-a)/(b-a) if b != a else x-a


# Types
from common_types import Points, Guards
from typing import List, Optional, Tuple, Any, Type


@dataclass
class VisSample:  
    name : str  
    points : Points # points of terrain/skgeom.Polygon
    guards : Guards # guards placed on vertices    

    
    @staticmethod
    def read_samples(path : str,sol_sample : Optional[int] = 1, normalize : Optional[bool] = False, solve : bool = True) -> List["VisSample"]:
        """
            read samples

                Parameters:
                    path: str  ... path to samples
                    sol_sample : int ... number of solutions added: 0 all, 
                
                Return:
                    samples : List["VisSample"] ... list of terrain samples

        """
        samples : List["VisSample"] = [] 

        
        # get list of all files
        files = [path] if os.path.isfile(path) else [os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
        
                  
        # read TGPIL/AG instances from file
        instances = sorted([f for f in files if f.endswith('.terrain') or f.endswith('.pol')])
               
        for filename in instances:          
            instance_name, instance_root = Path(filename).stem, Path(filename).parent
            points_file   = filename
            sol_file      = os.path.join(instance_root,instance_name+'.solution')
            cluster_file  = os.path.join(instance_root,instance_name+'.clusters')
            
            if not os.path.exists(sol_file):
                continue
                       
            
            else:   # if already solved  
                # read datapoints       
                with open(points_file,'r') as f:
                    if points_file.endswith('.terrain'):                                                       
                        lines = f.readlines()
                        # read number of points
                        num_points = int(lines[0])
                        # read points
                        _points = np.array([list(map(float,(f"{l} 0").split())) for l in lines[1 : 1 + num_points]])
                    elif points_file.endswith('.pol'):
                        tokens = f.read().split()
                        num_points = int(tokens[0])
                        # read number of points                        
                        _points = np.array([(eval(tokens[i]),eval(tokens[i+1])) for i in range(1,2*num_points,2)],dtype='float64')                      
                    
                # read number of solutions from clusters
                sol_file = cluster_file if os.path.exists(cluster_file) else sol_file
                with open(sol_file,'r') as f:  
                    lines = f.readlines()                                                     
                    num_sols = int(lines[0].split(' ')[-1])                    
                    
                    # multiply sample with more solutions
                    _sols = [ tensor(list(map(int,l.split())),dtype=torch.long) for l in lines[1 : 1+num_sols ]]       
                

                                    
            # define bounding box
            l,r = np.min(_points[:,0]), np.max(_points[:,0])
            b,t = np.min(_points[:,1]), np.max(_points[:,1])

            # define scale                                
            k = max( r-l, t-b )            
            _points = [tensor([scale(p[0],l,r),scale(p[1],b,t),0.0],dtype=torch.float32) for p in _points]  if normalize else [tensor([(p[0]-l)/k,(p[1]-b)/k,0.0],dtype=torch.float32) for p in _points]          

            num_sols = len(_sols)

            if num_sols == 0 :                 
                samples.extend( [VisSample( name = instance_name, points = _points, guards = tensor([]) )]) 
            else:
                if sol_sample > 0: 
                    samples.extend( [VisSample( name = instance_name, points = _points, guards = _guards ) for _guards in _sols[:sol_sample] ])
                else:
                    # ToDo: add diverse solutions ...
                    samples.extend( [VisSample( name = instance_name, points = _points, guards = _guards ) for _guards in _sols] )
        
        return samples



class VisDataset(list, Dataset):
    def __init__(self, samples: Optional[List[VisSample]] = []):
        self.samples = samples if samples else []
        self.points = ([torch.tensor([x.tolist() for x in t.points]) for t in self.samples])

    def __repr__(self):
        return self.samples.__repr__()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = self.points[idx]
        pos = self.samples[idx].guards
        name = self.samples[idx].name
        return seq, pos, name

    def __add__(self, other):
        return super(list, self).__add__(other)

    def extend(self, other):
        if isinstance(other, list):
            for l in other:
                self.samples.extend(l.samples)

        self.points = ([torch.tensor([x.tolist() for x in t.points]) for t in self.samples])



def read_dataset(path : str, dump : Optional[str] = None) -> List:
    """
        Read dataset samples from the given path
    """
    dataset: List = []    
    paths = [os.path.join(root,dir) for root,dirs,files in os.walk(path) for dir in dirs]
        
    for path in paths:
        dataset.extend(VisSample.read_samples(path))    

    if dump is not None:
        with open(dump, "wb") as file:
            pickle.dump(dataset, file)
        print(f"saved samples to {dump}")
    return dataset

class VisSequence:
    input_size = 3
    bos = torch.Tensor([0.0, 0.0, 1.0])
    eos = torch.Tensor([0.0, 0.0, -1.0])

def collate_fn(batch):
    """
    Collate fn for DataLoader
    We append eos to start of every sequence, as in the Pointer Network paper
    Because of that, we need to increase all position indices by 1, and add the position 0 at the end of true positions

    :param batch: batch of sequence and position pairs
    """
    
    sequences, positions, names = zip(*batch)
    sequences = tuple(torch.cat((VisSequence.eos.unsqueeze(0), seq)) for seq in sequences)
    positions = tuple(torch.cat((pos + 1, torch.tensor([0]))) for seq, pos in zip(sequences, positions))

    # Sequence lengths needed for packing padded sequences
    sequences_lens = [seq.shape[0] for seq in sequences]

    # Pad sequences and true positions with value -1
    sequences_padded = nn.utils.rnn.pad_sequence(sequences, padding_value=-1)
    positions_padded = nn.utils.rnn.pad_sequence(positions, padding_value=-1)

    return sequences_padded, sequences_lens, positions_padded, names



def calculate_polygon(points):
    points = points.reshape(points.shape[0],points.shape[2])[1:]


    
class Arrangement:
    def __init__(self,poly : skgeom.Polygon):
        self.arr  = skgeom.arrangement.Arrangement()
        self._area = poly.area()
        # add to arrangement
        for e in poly.edges:
            if e[0] == e[1]:
                continue
            self.arr.insert(e)
        self.vs = skgeom.TriangularExpansionVisibility(self.arr)

    def area(self):
        return self._area




class VisibilityRegion:
    def __init__(self, arr : Arrangement, ps : Optional[skgeom.PolygonSet] = skgeom.PolygonSet()):
        # compute skgeom.Polygonal set for        
        self.ta             = arr
        self.vis_region     = ps                
        self._area           = np.sum([abs(float(_poly.outer_boundary().area()))-np.sum([h.area() for h in _poly.holes]) for _poly in self.vis_region.skgeom.Polygons])   

    def __add__(self, p : skgeom.Point2) -> Type["VisibilityRegion"]:   
      
        eps = 1e-7
        # compute visibility        

        q = skgeom.Point2(p.x(),p.y()+eps)  # take y-close point as approximation of p
      

        face = self.ta.arr.find(q)
        vx   = self.ta.vs.compute_visibility(q,face)  
        
        verts = [v.point() for v  in vx.vertices]        
        vis_p = skgeom.Polygon(verts)        
             
        # make union of skgeom.Polygons               
        vis = self.vis_region.union(vis_p)

        return VisibilityRegion(self.ta,vis)

    def area(self):
        return self._area

   


def value_function(v1 : VisibilityRegion,v2 : VisibilityRegion) -> float:
    area = v1.ta.area()
    if area - v1.area() <= 1e-7:
        return 0
    return (v2.area() - v1.area()) / (area - v1.area())
