from matplotlib import pyplot as plt

# pytorch utils
from torch.utils.data import DataLoader

# CGAL wrappers
import skgeom

# common
from common_types import conditional_decorator, timer_func


# terrains, polygons
from terrain    import Terrain
from polygon    import Polygon


# data types for pytorch
from data_types import *


# terrain demonstration
def terrain_demo(
    model: Any,    
    sample: VisSample, 
    beam_width: int = None, 
    alpha: float = None, 
    beta: float = None) -> Tuple:
    """
        terrain generator with computed optimal and predicted guards
    """
    
    points_generator = DataLoader(VisDataset([sample]), batch_size=1, collate_fn=vis_collate_fn)
    points, lens, sol = next(iter(points_generator))

    
    # predict solution
    model_result = model(points.cuda(), lens, beam_width=beam_width, alpha=alpha, beta=beta)
    decoded_seq = np.vectorize(int)(model_result[0].cpu()).squeeze()

    for i, d in enumerate(decoded_seq):
        if d == 0:
            decoded_seq = decoded_seq[:i]
            break  
    decoded_seq -= 1
    decoded_seq = np.unique(decoded_seq)
   
    # covert to approriate indices optimal solution
    sol = sol.flatten().numpy()   
    sol = np.array(sorted(list(set(sol[(sol>0)]))))    
    sol -= 1
        
    # predicted vs opt guards    
    points = points.reshape(points.shape[0],points.shape[2])[1:]
    guards = points[decoded_seq][:,[0,1]].numpy()   
    opt_guards = points[sol][:,[0,1]].numpy()

    # remove 3rd dimension of points
    points = points[:,[0,1]].numpy()
               
    # define bounding box
    l,r = np.min(points[:,0]), np.max(points[:,0])
    b,t = np.min(points[:,1]), np.max(points[:,1])

    eps = 0.01
    points = np.insert(points,0,[[l-eps,t+eps]],axis=0)
    points = np.append(points,[[r+eps,t+eps]],axis=0)           
   
    # define polygon
    poly = skgeom.Polygon(points) 

    # predicted guards
    predicted = [ skgeom.Point2(g[0],g[1]) for g in guards.tolist()]
    opt       = [ skgeom.Point2(g[0],g[1]) for g in opt_guards.tolist()]
    
    # return predicted and ground_truth
    # compute coverage
    # consider p+eps*e_2 point in polygonal region instead only  p due to numerics
    eps = 0.000001
   
           
    # arrangments
    arr = skgeom.arrangement.Arrangement()
    # add edges to arr
    for e in poly.edges : arr.insert(e)
                        
    # compute visibility
    vs = skgeom.RotationalSweepVisibility(arr)

        
    region_area = 0
    # visibility polygon
    poly_area = float(poly.area())
    views = []
    for p in predicted:        
        q = skgeom.Point2(p.x(),p.y()+eps)  # take y-close point as approximation of p
        face = arr.find(q)
        vx = vs.compute_visibility(q,face)        
        verts = sorted([v.point() for v  in vx.vertices],key = lambda p : p.x())             
        views.append(Polygon(verts))
        
    region = skgeom.PolygonSet(views)

    
    region_area = np.sum(
        [abs(float(_poly.outer_boundary().area()))-np.sum([h.area() for h in _poly.holes]) for _poly in region.polygons]
        )        
    
    coverage    = region_area/poly_area    
    
    return (poly, region, predicted, opt, coverage) 


# polygon demonstration
def polygon_demo(
    model : Any, 
    sample: VisSample, 
    beam_width: int = None, 
    alpha: float = None) -> Tuple:
    
    """
        polygon demo:
            assume VisSample to be PolygonSample...
            
    """
    points_generator = DataLoader(VisDataset([sample]), batch_size=1, collate_fn=collate_fn)
    points, lens, sol = next(iter(points_generator))

    
    # predict solution
    model_result = model(points.cuda(), lens, beam_width=beam_width, alpha=alpha)
    decoded_seq = np.vectorize(int)(model_result.to_seq(),cpu()).squeeze()

    for i, d in enumerate(decoded_seq):
        if d == 0:
            decoded_seq = decoded_seq[:i]
            break   
    decoded_seq -= 1
    decoded_seq = np.unique(decoded_seq)
   
    # covert to approriate indices optimal solution
    sol = sol.flatten().numpy()   
    sol = np.array(sorted(list(set(sol[(sol>0)]))))    
    sol -= 1
        
    # predicted vs opt guards    
    points = points.reshape(points.shape[0],points.shape[2])[1:]
    guards = points[decoded_seq][:,[0,1]].numpy()   
    opt_guards = points[sol][:,[0,1]].numpy()

    # remove 3rd dimension of points
    points = points[:,[0,1]].numpy()
    
    # draw polygon
    poly = skgeom.Polygon(points)
    
    # predicted guards
    predicted = [ skgeom.Point2(g[0],g[1]) for g in guards.tolist()]
    opt       = [ skgeom.Point2(g[0],g[1]) for g in opt_guards.tolist()]
    
    # return predicted and ground_truth
    # compute coverage
    # consider p+eps*e_2 point in polygonal region instead only  p due to numerics
    eps = 0.000001
   
           
    # arrangments
    arr = skgeom.arrangement.Arrangement()
    # add edges to arr
    for e in poly.edges : arr.insert(e)
                        
    # compute visibility
    vs = skgeom.RotationalSweepVisibility(arr)

        
    region_area = 0
    # visibility polygon
    poly_area = float(poly.area())
    views = []
    for p in predicted:        
        q = skgeom.Point2(p.x(),p.y()+eps)  # take y-close point as approximation of p
        face = arr.find(q)
        vx = vs.compute_visibility(q,face)        
        verts = sorted([v.point() for v  in vx.vertices],key = lambda p : p.x())             
        views.append(skgeom.Polygon(verts))
        
    region = skgeom.PolygonSet(views)

    
    region_area = np.sum(
        [
            abs(float(_poly.outer_boundary().area()))-np.sum([h.area() for h in _poly.holes]) for _poly in region.polygons
        ]
        )        
    
    coverage    = region_area/poly_area    
    
    return (poly, region, predicted, opt, coverage) 



# metrics for samples
def sample_coverage(
    model: Any, 
    samples: List[VisSample], 
    beam_width: int = None, 
    alpha: float = None, 
    beta: float = None) -> float:

    """
        compute sample average of coverage
    """    
    
    coverage = np.zeros(len(samples))
    approx   = np.zeros(len(samples))

    for i,sample in enumerate(samples):
        try:
            print(f'\rProcessing instance {sample.name}', end = ' ')
            _,_,predicted,opt, cover = terrain_demo(model, sample, beam_width, alpha, beta)
            coverage[i] = cover
            approx[i]   = np.float(len(predicted))/np.float(len(opt)) if len(opt) > 0 else 0
        except Exception as e:
            print(f'Problem on {sample.name}')
            print(e)                               
            
    avg_cov = np.average(coverage)
    avg_approx = np.average(approx) 

    return avg_cov, avg_approx



def vis_predict(
    model: Any, 
    sample: VisSample, 
    beam_width: int = None, 
    alpha: float = None, 
    beta: float = None) -> Tuple:
    """
        returns seqs of predicted and opt values
    """
    
    points_generator = DataLoader(VisDataset([sample]), batch_size=1, collate_fn=vis_collate_fn)
    points, lens, sol = next(iter(points_generator))

    
    # predict solution
    model_result = model(points.cuda(), lens, beam_width=beam_width, alpha=alpha, beta=beta)
    decoded_seq = np.vectorize(int)(model_result[0].cpu()).squeeze()

    for i, d in enumerate(decoded_seq):
        if d == 0:
            decoded_seq = decoded_seq[:i]
            break   
    decoded_seq -= 1
    decoded_seq = np.unique(decoded_seq)
   
    # covert to approriate indices optimal solution
    sol = sol.flatten().numpy()   
    sol = np.array(sorted(list(set(sol[(sol>0)]))))    
    sol -= 1
    return decoded_seq



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
        self.opt    = sample.guards.numpy().tolist()
        self.solution = [P.Vertices[v] for v in self.opt]        
        
        return self.opt

    def show(self):
        skgeom.draw.draw(self.poly,line_width=1,alpha=0.2,aspect_ratio = 'auto')        
        skgeom.draw.draw(self.solution,color = 'blue')
        plt.show()


class TGNetSearch:
    def __init__(self,model):
        self.model = torch.load(model)        
        
    @conditional_decorator(timer_func,False)
    def predict(self,instance,beam_width=4,alpha=1,beta=0.2):        
        sample = VisSample.read_samples(path=instance,sol_sample=1)[0]
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
        
        self.coverage = coverage        
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