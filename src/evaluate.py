# pytorch utils
from torch.utils.data import DataLoader
# common
from common_types import conditional_decorator, timer_func

# terrains, polygons
from terrain    import Terrain
from polygon    import Polygon
# NNs
import PtrNet

import torch


# add safe model loading
safe_globals = {
    'PointerNetwork': PtrNet.PointerNetwork,
    'Pointer': PtrNet.Pointer,
    'PointerExclusive': PtrNet.PointerExclusive,
    # add other classes/functions if needed
}

torch.serialization.add_safe_globals(safe_globals)

import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)

# math
import math
# debugging
import pdb


from demo import *


# specify CUDA device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def vis_predict(
    model: Any, 
    sample: VisSample, 
    beam_width: int = None, 
    alpha: float = None, 
    beta: float = None) -> np.ndarray:
    """
        returns seqs of predicted pointers for polygon
    """
    
    points_generator = DataLoader(VisDataset([sample]), batch_size=1, collate_fn=collate_fn)
    points, lens, sol, _ = next(iter(points_generator))      
    
    # Ensure the model is loaded correctly
    if isinstance(model, str):
        model = torch.load(model, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Move the model to the same device as the input tensor    
    model.to(device)
    
    # predict solution
    model_result = model(points.cuda(), lens, beam_width=beam_width, alpha=alpha, beta=beta)
    
    # define list of possible solutions
    num_sols = len(model_result)
    decoded_seq = [None] * num_sols
     
    for j in range(num_sols):
        decoded_seq[j] = np.vectorize(int)(model_result[j][0].cpu()).squeeze()

        for i, d in enumerate(decoded_seq[j]):
            if d == 0:
                decoded_seq[j] = decoded_seq[j][:i]
                break   
        decoded_seq[j] -= 1
        decoded_seq[j] = np.unique(decoded_seq[j])  
        
   
    return decoded_seq



# metrics for samples
def sample_coverage(
    model: Any, 
    samples: List[VisSample], 
    beam_width: int = None, 
    alpha: float = None, 
    vis_compute = terrain_demo
    ) -> float:

    """
        compute sample average of coverage
    """    
    
    coverage = np.zeros(len(samples))
    approx   = np.zeros(len(samples))

    for i,sample in enumerate(samples):
        try:
            print(f'\rProcessing instance {sample.name}', end = ' ')
            _,_,predicted,opt, cover = vis_compute(model, sample, beam_width, alpha)
            coverage[i] = cover
            approx[i]   = np.float(len(predicted))/np.float(len(opt)) if len(opt) > 0 else 0
        except Exception as e:
            print(f'Problem on {sample.name}')
            print(e)                               
            
    avg_cov = np.average(coverage)
    avg_approx = np.average(approx) 

    return avg_cov, avg_approx




# opt, heuristics and nn predictors
class Opt:   

    def predict(self,instance : str):
        """retrieve optimal solution """        
        if instance.endswith('.terrain'):
            P = Terrain(instance)            
        elif instance.endswith('.pol'):
            P = Polygon(instance)
            
        
        self.poly = P.poly            
        sample      = VisSample.read_samples(path=instance,sol_sample=1)[0]
        self.opt    = sample.guards.numpy()
        self.solution = [P.Vertices[v] for v in self.opt]        
        
        return self.opt

    def show(self):
        skgeom.draw.draw(self.poly,line_width=1,alpha=0.2,aspect_ratio = 'auto')        
        skgeom.draw.draw(self.solution,color = 'blue')
        plt.show()


class TGNetSearch:
    def __init__(self, model):
        try:
            self.model = torch.load(model,map_location=torch.device(device))
            print("Model loaded successfully.")            
        except Exception as e:
            print(f"Failed to load model: {e}")
    @conditional_decorator(timer_func,False)
    def predict(self,instance,beam_width=4,alpha=1,beta=0.2):        
        sample = VisSample.read_samples(path=instance,sol_sample=1)
        self.poly, self.region, self.predicted, self.opt, self.coverage = terrain_demo(self.model, sample, beam_width=beam_width,alpha=alpha,beta=beta)
        return self.predicted



    def show(self):
        # predicted
        skgeom.draw.draw(self.poly,line_width=1,alpha=0.2,aspect_ratio = 'auto')     
        skgeom.draw.draw(self.region,line_width=1,alpha=0.2,aspect_ratio = 'auto',facecolor = 'red')     
        skgeom.draw.draw(self.predicted,color = 'red')
        
        # opt
        skgeom.draw.draw(self.poly,line_width=1,alpha=0.2,aspect_ratio = 'auto')        
        #draw(self.opt,color = 'blue')
        plt.show()


class TGreedy:  
    """
        set-cover greedy heuristic for finding approximate solution
    """
    
    def _eval(self,v1 : VisibilityRegion,v2 : VisibilityRegion) -> float:
        return value_function(v1,v2)

    @conditional_decorator(timer_func,False)
    def predict(self, instance : str, th : float) -> list:
        # terrain arrangement
        self.T = Terrain()
        self.T.read_tgpil(instance)        
        self.ta = Arrangement(self.T.poly)

        self.predicted = []
        vr      = VisibilityRegion(self.ta)
        p = -1
        coverage = 0 
        while coverage < th:      
            # evaluate coverage on all vertices p<v      
            evals    = [self._eval(vr,vr+self.T.Vertices[v]) for v in self.T.vertices]
            p        = np.argmax(evals)
            old_coverage = coverage
            vr += self.T.Vertices[p]
            coverage = float(vr.area())/float(self.T.poly.area())
                        
            if coverage-old_coverage == 0: break       
            
            self.predicted.append(p)
        
        self.coverage = float(coverage)
        return self.predicted

    def show(self):
        # visibility polygon
        points = [self.T.Vertices[v] for v in self.predicted]
        poly       = self.T.poly
    
        eps = 1e-7
        views = []
        
        for p in points:                        
            q = skgeom.Point2(p.x(),p.y()+eps)  # take y-close point as approximation of p
            face = self.ta.arr.find(q)
            vx = self.ta.vs.compute_visibility(q,face)        
            verts = sorted([v.point() for v  in vx.vertices],key = lambda p : p.x())             
            views.append(Polygon(verts))
        
        region = skgeom.PolygonSet(views)
        
        skgeom.draw.draw(poly,line_width=1,alpha=0.2,aspect_ratio = 'auto')     
        skgeom.draw.draw(region,line_width=1,alpha=0.2,aspect_ratio = 'auto',facecolor = 'red')     
        skgeom.draw.draw(points,color = 'red')
        plt.show()


class AGNetSearch:
    def __init__(self, model_path):
        try:
            self.model = torch.load(model_path, map_location=torch.device(device))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")

    @conditional_decorator(timer_func,False)
    def predict(self,instance,beam_width=4,alpha=1,beta=0.2):        
        sample = VisSample.read_samples(path=instance,sol_sample=1)[0]
        self.predicted = vis_predict(self.model,sample)
        return self.predicted
    
    


if __name__ == '__main__':

    # define instance
    instance = './dataset/AG/development/large/rand-800-7.pol'
    print('Testing neural solvers on instance:', instance)


    # test opt solver
    opt_solver = Opt()
    print(' * Optimal solution:')
    print(opt_solver.predict(instance))    
    
    print('-'*100)
    # test supervised model
    supervised_model_path = './models/supervised/trained_models/ag_clusters_numsols-1_ne-100_bs-64_hs-256_tfr-0.5_wd-1e-05_lr-0.001_bidirectional_normalized.pt' 
       
    # PTrNet search
    print('* Loading supervised model ...', end='', flush=True)
    supervised_solver = AGNetSearch(supervised_model_path)
    print('* Predicting with supervised model: ')
    print(supervised_solver.predict(instance))
    print("\t+ Supervised model prediction done. OK")
    print('-'*100)
    
    # test RL model
    reinforce_model_path = './models/reinforce/trained_models/ne-1_bs-64_hs-256_hv-256_wd-0.01_lr-0.0001_ss-10_wt_cov-0.6_dummy_model/best_model.pth'     

    print('* Loading reinforce model ...', end='', flush=True)        
    reinforce_solver = AGNetSearch(reinforce_model_path)
    
    print('* Predicting with reinforce model: ')
    print(reinforce_solver.predict(instance))
    print("\t+ Reinforce model prediction done. OK")
    print('='*100)





