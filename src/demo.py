# data types for pytorch
from   data_types import *
import math

from matplotlib import pyplot as plt
# CGAL wrappers

import skgeom
import numpy as np
import math


import faulthandler
faulthandler.enable()


# specify CUDA device
device = torch.device('cuda:0')

DEBUG = False


# debug plot
def _plot_debugger(
            points, i: int, 
            q: skgeom.Point2, 
            log : str = 'polygon_debugger' 
            ) -> None:

        # Extract x and y coordinates
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        scale_x, scale_y = 100, 100
        # Plot the polygon without markers
        plt.figure(figsize=(scale_x, scale_y))  # Increased figure size
        plt.plot(x_coords, y_coords, 'b-', linewidth=1)

        # Plot and label only the specified points
        for idx, (x, y) in enumerate(zip(x_coords, y_coords)):
            if idx == (i - 1) % len(points):
                color = 'orange'  # Color for v_prev
                plt.plot(x, y, 'o', color=color, markersize=10)
                plt.text(x - 0.1/scale_x, y - 0.1/scale_y, f'P_{idx+1}', fontsize=10, ha='right', va='top', color=color)
            elif idx == i:
                color = 'green'  # Color for v
                plt.plot(x, y, 'o', color=color, markersize=10)
                plt.text(x + 0.1/scale_x, y + 0.1/scale_y, f'P_{idx+1}', fontsize=10, ha='left', va='bottom', color=color)
            elif idx == (i + 1) % len(points):
                color = 'purple'  # Color for v_next
                plt.plot(x, y, 'o', color=color, markersize=10)
                plt.text(x + 0.1/scale_x, y - 0.1/scale_y, f'P_{idx+1}', fontsize=10, ha='left', va='top', color=color)
            # Skip plotting other points

        # Plot the extra point q
        q_x, q_y = q.x(), q.y()
        plt.plot(q_x, q_y, 'ro', markersize=10)  # Red dot for q
        plt.text(q_x + 0.1/scale_x, q_y + 0.1/scale_y, 'q', fontsize=10, color='red', ha='left', va='bottom')

        # Set plot limits with some padding
        padding = 0.05
        plt.xlim(min(x_coords + [q_x]) - padding, max(x_coords + [q_x]) + padding)
        plt.ylim(min(y_coords + [q_y]) - padding, max(y_coords + [q_y]) + padding)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Polygon with Highlighted Points')
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        
        # Save the plot to a file
        plt.savefig(f'debug/{log}.png')
        
        #plt.show()


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


    

def evaluate_polygon_visibility(sample : VisSample, solution : np.ndarray) -> Tuple:
    """
    Evaluates polygon visibility and guard placement.
    Args:
        sample (VisSample): A sample containing polygon points and guard positions.
        solution (np.array): An array of indices representing the predicted guard positions.
    Returns:
        tuple: (poly, region, predicted, opt, coverage)
    """


    # Extract points
    points = np.array([x.tolist() for x in sample.points])[:, [0, 1]]
    # Extract guards
    guards = points[solution]
    # Extract optimal guards
    opt_guards = points[sample.guards]
    

    if solution.shape == (1,):
        guards = guards.reshape(1, -1)
        opt_guards = opt_guards.reshape(1, -1)
        
    # Draw polygon
    poly = skgeom.Polygon(points)    

    # Predicted and optimal guards
    predicted = [skgeom.Point2(g[0], g[1]) for g in guards.tolist()]
    opt       = [skgeom.Point2(g[0], g[1]) for g in opt_guards.tolist()]
    
    # Create 2D arrangement of the polygon
    arr = skgeom.arrangement.Arrangement()
    for e in poly.edges:
        arr.insert(e)

    # Initialize visibility calculator
    vs = skgeom.TriangularExpansionVisibility(arr)

    # Compute coverage
    region_area = 0
    poly_area   = abs(float(poly.area()))
    views       = []

    # extract edges as list
    edges = list(poly.edges)

    # distance to move point q inside the polygon
    eps = 1e-5  # Small epsilon to adjust point position
    

    for i in solution:
        # Get adjacent vertices
        v_prev = edges[(i - 1) % len(edges)].source()
        v = edges[i % len(edges)].source()
        v_next = edges[i % len(edges)].target()

        # Create vectors
        p = skgeom.Vector2(v, v_prev)
        p = p / math.sqrt(p.squared_length())
        r = skgeom.Vector2(v, v_next)
        r = r / math.sqrt(r.squared_length())

        # Adjust point q slightly towards the inside of the polygon
        q = skgeom.Point2(v.x() + eps * (p.x() + r.x()), v.y() + eps * (p.y() + r.y()))

        # If q is not inside the polygon, adjust it
        if poly.oriented_side(q) != skgeom.Sign.POSITIVE:             
            q = skgeom.Point2(v.x() - eps * (p.x() + r.x()), v.y() - eps * (p.y() + r.y()))

        # Find the face containing q        
        face = arr.find(q)        
        # check if face is valid
        if face is None or face.is_unbounded():
            # Skip if face is invalid
            print(f"Error: cannot find face for guard {i}")
            # plot image
            log = f"{sample.name}_no_face_at_{i}" if face is None else f"{sample.name}_unbounded_face_at_{i}"
            try:
                _plot_debugger(points, i, q, log=log)            
            except:
                print(f"Error plotting face at guard {i}")
            continue

        # Compute visibility polygon
        try:            
            vx = vs.compute_visibility(q, face)            
        except RuntimeError as e:
            # Handle exceptions from CGAL
            print(f"Error computing visibility at guard {i}: {e}")            
            log = f"{sample.name}_no_vis_{i}" 
            try:
                _plot_debugger(points, i, q, log=log)
            except:
                print(f"Error plotting visibility at guard")
            continue

        # Add visibility polygon to views        
        visibility_polygon = skgeom.Polygon([v.point() for v in vx.vertices])        
        views.append(visibility_polygon)
    

    # Create a polygon set from the visibility polygons
    region = skgeom.PolygonSet()  
    try:
        # here is the problem ...
        for i,v in enumerate(views):
            ps = skgeom.PolygonSet([v])            
            region = region.union(ps)
            
        # region = skgeom.PolygonSet(views) # for some reason this makes a problem
    except Exception as e:
        print(f"Error creating polygon set: {e}")
        

    # Calculate total visible area
    for vis_poly in region.polygons:
        outer_area = abs(float(vis_poly.outer_boundary().area()))
        holes_area = sum(abs(float(hole.area())) for hole in vis_poly.holes)
        region_area += outer_area - holes_area

    coverage = region_area / poly_area if poly_area > 0 else 0.0    
    

    return (poly, region, predicted, opt, coverage)
    

def evaluate_polygon_visibility_numpy(points: np.ndarray, gt: np.ndarray, solution: np.ndarray):    
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
    opt_guards = points[gt]
    opt_guards = opt_guards.reshape(1,-1) if len(opt_guards.shape)==1 else opt_guards
       
     # draw polygon
    poly = skgeom.Polygon(points)
        
    # return predicted and ground_truth
    # compute coverage
    # consider p+eps*e_2 point in polygonal region instead only  p due to numerics
    eps = 1e-3
    #eps = 1e-10
          
    # arrangments
    arr = skgeom.arrangement.Arrangement()
    # add edges to arr
    for e in poly.edges : arr.insert(e)
                        
    # compute visibility using triangular expansion
    vs = skgeom.TriangularExpansionVisibility(arr)
        
    # region area
    region_area = 0

    # visibility polygon
    poly_area = float(poly.area())
    views = []

    edges = list(poly.edges)

    
    for i in solution:        
        # Get adjacent vertices
        v_prev = edges[(i - 1) % len(edges)].source()
        v = edges[i % len(edges)].source()
        v_next = edges[i % len(edges)].target()

        # Create vectors
        p = skgeom.Vector2(v, v_prev)
        p = p / math.sqrt(p.squared_length())
        r = skgeom.Vector2(v, v_next)
        r = r / math.sqrt(r.squared_length())

        # Adjust point q slightly towards the inside of the polygon
        q = skgeom.Point2(v.x() + eps * (p.x() + r.x()), v.y() + eps * (p.y() + r.y()))

        # If q is not inside the polygon, adjust it
        if poly.oriented_side(q) != skgeom.Sign.POSITIVE:             
            q = skgeom.Point2(v.x() - eps * (p.x() + r.x()), v.y() - eps * (p.y() + r.y()))

        # Find the face containing q
        face = arr.find(q)
        if face is None or face.is_unbounded():
            # Skip if face is invalid
            print(f"Error: cannot find face for guard {i}")
            # plot image
            log = f"{sample.name}_no_face_at_{i}" if face is None else f"{sample.name}_unbounded_face_at_{i}"
            _plot_debugger(points, i, q, log=log)            
            continue

        # Compute visibility polygon
        try:
            vx = vs.compute_visibility(q, face)
        except RuntimeError as e:
            # Handle exceptions from CGAL
            print(f"Error computing visibility at guard {i}: {e}")            
            log = f"{sample.name}_no_vis_{i}" 
            _plot_debugger(points, i, q, log=log)            
            continue

        # Add visibility polygon to views
        visibility_polygon = skgeom.Polygon([v.point() for v in vx.vertices])
        views.append(visibility_polygon)



    # Create a polygon set from the visibility polygons
    #region = skgeom.PolygonSet(views)
    
    # Create a polygon set from the visibility polygons
    region = skgeom.PolygonSet()  
    try:
        # here is the problem ...
        for i,v in enumerate(views):
            ps = skgeom.PolygonSet([v])            
            region = region.union(ps)
            
        # region = skgeom.PolygonSet(views) # for some reason this makes a problem
    except Exception as e:
        print(f"Error creating polygon set: {e}")    

    # Calculate total visible area
    for vis_poly in region.polygons:
        outer_area = abs(float(vis_poly.outer_boundary().area()))
        holes_area = sum(abs(float(hole.area())) for hole in vis_poly.holes)
        region_area += outer_area - holes_area

    coverage = region_area / poly_area if poly_area > 0 else 0.0    
    
    return coverage
    
def evaluate_polygon_visibility_numpy_wo_gt(points: np.ndarray, solution: np.ndarray):    
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
     # draw polygon
    poly = skgeom.Polygon(points)
        
    # return predicted and ground_truth
    # compute coverage
    # consider p+eps*e_2 point in polygonal region instead only  p due to numerics
    eps = 1e-3
    #eps = 1e-10
          
    # arrangments
    arr = skgeom.arrangement.Arrangement()
    # add edges to arr
    for e in poly.edges : arr.insert(e)
                        
    # compute visibility using triangular expansion
    vs = skgeom.TriangularExpansionVisibility(arr)
        
    # region area
    region_area = 0

    # visibility polygon
    poly_area = float(poly.area())
    views = []

    edges = list(poly.edges)

    
    for i in solution:        
        # Get adjacent vertices
        v_prev = edges[(i - 1) % len(edges)].source()
        v = edges[i % len(edges)].source()
        v_next = edges[i % len(edges)].target()

        # Create vectors
        p = skgeom.Vector2(v, v_prev)
        p = p / math.sqrt(p.squared_length())
        r = skgeom.Vector2(v, v_next)
        r = r / math.sqrt(r.squared_length())

        # Adjust point q slightly towards the inside of the polygon
        q = skgeom.Point2(v.x() + eps * (p.x() + r.x()), v.y() + eps * (p.y() + r.y()))

        # If q is not inside the polygon, adjust it
        if poly.oriented_side(q) != skgeom.Sign.POSITIVE:             
            q = skgeom.Point2(v.x() - eps * (p.x() + r.x()), v.y() - eps * (p.y() + r.y()))

        # Find the face containing q
        face = arr.find(q)
        if face is None or face.is_unbounded():
            # Skip if face is invalid
            print(f"Error: cannot find face for guard {i}")
            # plot image
            #log = f"{sample.name}_no_face_at_{i}" if face is None else f"{sample.name}_unbounded_face_at_{i}"
            #_plot_debugger(points, i, q)            
            continue

        # Compute visibility polygon
        try:
            vx = vs.compute_visibility(q, face)
        except RuntimeError as e:
            # Handle exceptions from CGAL
            print(f"Error computing visibility at guard {i}: {e}")            
            #log = f"{sample.name}_no_vis_{i}" 
            #_plot_debugger(points, i, q)            
            continue

        # Add visibility polygon to views
        visibility_polygon = skgeom.Polygon([v.point() for v in vx.vertices])
        views.append(visibility_polygon)



    # Create a polygon set from the visibility polygons
    #region = skgeom.PolygonSet(views)
    
    # Create a polygon set from the visibility polygons
    region = skgeom.PolygonSet()  
    try:
        # here is the problem ...
        for i,v in enumerate(views):
            ps = skgeom.PolygonSet([v])            
            region = region.union(ps)
            
        # region = skgeom.PolygonSet(views) # for some reason this makes a problem
    except Exception as e:
        print(f"Error creating polygon set: {e}")    

    # Calculate total visible area
    for vis_poly in region.polygons:
        outer_area = abs(float(vis_poly.outer_boundary().area()))
        holes_area = sum(abs(float(hole.area())) for hole in vis_poly.holes)
        region_area += outer_area - holes_area

    coverage = region_area / poly_area if poly_area > 0 else 0.0    
    
    return coverage

class CustomError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
        

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



if __name__ == '__main__':

   
    # test instance directory
    dataset_dir = '/mnt/nvme0n1/dseverdi/MLAG/dataset/AG/development/'

    instance_name = 'train/rand-172-166.pol'
    #instance_name = 'train/rand-14-10.pol'

    instance = os.path.join(dataset_dir,instance_name)
    sample = VisSample.read_samples(path=instance, sol_sample=1)[0]

    # opt solution
    solution = sample.guards 
    #print(solution.shape)
    #print(solution)
    # one solution
    #solution = np.arange(1)

    # all vertices as guards
    #solution    = np.arange(len(sample.points))

    print(sample.name)   

    # Call the evaluate_polygon_visibility function    
    poly, region, predicted, opt, coverage = evaluate_polygon_visibility(sample, solution)

    print(f'solution: {solution}')
    print(f'guard points: {predicted}')
    print(f'Coverage: {coverage:2.8f}')
    

    

    