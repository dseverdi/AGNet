# data types for pytorch
from   data_types import *
import math

from matplotlib import pyplot as plt
# CGAL wrappers
import skgeom


# specify CUDA device
device = torch.device('cuda:0')


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

    model.to(device)
    # predict solution
    model_result = model(points.cuda(), lens, beam_width=beam_width, alpha=alpha, beta=beta)
    decoded_seq = np.vectorize(int)(model_result[0].cpu()).squeeze()

    for i, d in enumerate(decoded_seq):
        if d == 0:
            decoded_seq = decoded_seq[:i]
            break  
    decoded_seq -= 1
    decoded_seq = np.unique(decoded_seq)
   
    # covert to appropriate indices optimal solution
    sol = sol.flatten().numpy()   
    sol = np.array(sorted(list(set(sol[(sol>0)]))))    
    sol -= 1
        
    # points
    points = points.reshape(points.shape[0],points.shape[2])[1:]
    # predicted vs opt guards        
    guards = points[decoded_seq][:,[0,1]].numpy()   
    opt_guards = points[sol][:,[0,1]].numpy()
    
    # remove 3rd dimension of points
    points = points[:,[0,1]].numpy()
               
       
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



def polygon_demo2( decoded_seq, sample ):    
    """
        polygon demo:
            assume VisSample to be PolygonSample...
            
    """
    points_generator = DataLoader(VisDataset([sample]), batch_size=1, collate_fn=collate_fn)
    points, _, sol = next(iter(points_generator))

       
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
    
    # predicted guards vs opt guards
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
        
    # region area
    region_area = 0

    # visibility polygon
    poly_area = float(poly.area())
    views = []

    edges = list(poly.edges)

    
    for i in decoded_seq:        
        v_prev, v, v_next = edges[(i-1) % len(poly)].source(), edges[i].source(), edges[i].target() 
       
        # affine combination of adjacent vertices
        p = skgeom.Vector2(v,v_prev)
        p = 1.0/math.sqrt(p.squared_length()) * p
        r = skgeom.Vector2(v,v_next) 
        r = 1.0/math.sqrt(r.squared_length()) * r

        q = v+eps*(p+r)
        # change orientation if point outside of polygon
        q = v-eps*(p+r) if poly.oriented_side(q) == skgeom.Sign.NEGATIVE else q 

        # find q in polygon
        face = arr.find(q)      
        # compute visibility
        vx = vs.compute_visibility(q,face)                
        
        # add views        
        views.append(skgeom.Polygon([v.point() for v  in vx.vertices]))
        
    region = skgeom.PolygonSet(views)

    
    region_area = np.sum(
        [
            abs(float(_poly.outer_boundary().area()))-np.sum([h.area() for h in _poly.holes]) for _poly in region.polygons
        ]
        )        
    
    coverage    = float(region_area/poly_area)
    
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

    model.to(device)
    # predict solution
    model_result = model(points.cuda(), lens, beam_width=beam_width, alpha=alpha)
    decoded_seq = np.vectorize(int)(model_result[0][0].to_seq().cpu()).squeeze()

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
    
    # predicted guards vs opt guards
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
        
    # region area
    region_area = 0

    # visibility polygon
    poly_area = float(poly.area())
    views = []

    edges = list(poly.edges)

    
    for i in decoded_seq:        
        v_prev, v, v_next = edges[(i-1) % len(poly)].source(), edges[i].source(), edges[i].target() 
       
        # affine combination of adjacent vertices
        p = skgeom.Vector2(v,v_prev)
        p = 1.0/math.sqrt(p.squared_length()) * p
        r = skgeom.Vector2(v,v_next) 
        r = 1.0/math.sqrt(r.squared_length()) * r

        q = v+eps*(p+r)
        # change orientation if point outside of polygon
        q = v-eps*(p+r) if poly.oriented_side(q) == skgeom.Sign.NEGATIVE else q 

        # find q in polygon
        face = arr.find(q)      
        # compute visibility
        vx = vs.compute_visibility(q,face)                
        
        # add views        
        views.append(skgeom.Polygon([v.point() for v  in vx.vertices]))
        
    region = skgeom.PolygonSet(views)

    
    region_area = np.sum(
        [
            abs(float(_poly.outer_boundary().area()))-np.sum([h.area() for h in _poly.holes]) for _poly in region.polygons
        ]
        )        
    
    coverage    = float(region_area/poly_area)
    
    return (poly, region, predicted, opt, coverage) 


    

def evaluate_polygon_visibility(sample : VisSample, solution : np.array):    
    """
    Demonstrates polygon visibility and guard placement.
    Args:
        sample (VisSample): A sample containing polygon points and guard positions.
        solution (np.array): An array of indices representing the predicted guard positions.
    Returns:
        tuple: A tuple containing:
            - poly (skgeom.Polygon): The polygon created from the sample points.
            - region (skgeom.PolygonSet): The set of visibility polygons for the guards.
            - predicted (list of skgeom.Point2): The list of predicted guard positions.
            - opt (list of skgeom.Point2): The list of optimal guard positions.
            - coverage (float): The coverage ratio of the visibility region to the polygon area.
    """
    
    points =  np.array([x.tolist() for x in sample.points])[:,[0,1]]
    guards = points[solution]
    opt_guards = points[sample.guards]
    
    opt_guards = opt_guards.reshape(1,-1) if len(opt_guards.shape)==1 else opt_guards
       
     # draw polygon
    poly = skgeom.Polygon(points)
    
    # predicted guards vs opt guards 
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
        
    # region area
    region_area = 0

    # visibility polygon
    poly_area = float(poly.area())
    views = []

    edges = list(poly.edges)

    
    for i in solution:        
        v_prev, v, v_next = edges[(i-1) % len(poly)].source(), edges[i].source(), edges[i].target() 
       
        # affine combination of adjacent vertices
        p = skgeom.Vector2(v,v_prev)
        p = 1.0/math.sqrt(p.squared_length()) * p
        r = skgeom.Vector2(v,v_next) 
        r = 1.0/math.sqrt(r.squared_length()) * r

        q = v+eps*(p+r)
        # change orientation if point outside of polygon
        q = v-eps*(p+r) if poly.oriented_side(q) == skgeom.Sign.NEGATIVE else q 

        # find q in polygon
        face = arr.find(q)      
        # compute visibility
        vx = vs.compute_visibility(q,face)                
        
        # add views        
        views.append(skgeom.Polygon([v.point() for v  in vx.vertices]))
        
    region = skgeom.PolygonSet(views)

    
    region_area = np.sum(
        [
            abs(float(_poly.outer_boundary().area()))-np.sum([h.area() for h in _poly.holes]) for _poly in region.polygons
        ]
        )        
    
    coverage    = float(region_area/poly_area)
    
    return (poly, region, predicted, opt, coverage) 
    

    

def show(*demo_solution):
    poly, region, predicted, opt, coverage = demo_solution[0], demo_solution[1], demo_solution[2], demo_solution[3],demo_solution[4]
    
    fig, ax = plt.subplots()
    ax.axis('off')
    
    # predicted
    if predicted:
        skgeom.draw.draw(poly,line_width=1,alpha=0.2,aspect_ratio = 'auto')     
        skgeom.draw.draw(region,line_width=1,alpha=0.2,aspect_ratio = 'auto',facecolor = 'red')     
        skgeom.draw.draw(predicted,color = 'red')
        plt.show()
        print(f' * Predicted solution size:     {len(predicted)}')
        print(f' * PtrMNet coverage:     {coverage:2.8f},      ratio=pred/opt: {len(predicted)/len(opt):2.8f}')
            
    
    else:  # opt
        skgeom.draw.draw(poly,line_width=1,alpha=0.2,aspect_ratio = 'auto')     
        #skgeom.draw.draw(region,line_width=1,alpha=0.2,aspect_ratio = 'auto',facecolor = 'red')     
        skgeom.draw.draw(opt,color = 'blue')    
        plt.show()
        print(f' * Optimal solution size:     {len(opt)}')