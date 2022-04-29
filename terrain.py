"""
    
    Terrain respresentation and solver
    
"""

# terrain dataset:
# http://resources.mpi-inf.mpg.de/tgp/index.html#instances

# types
from typing import Optional, Tuple

# arrays
import numpy as np

# geometry
import skgeom

# optimizers
import gurobipy as gp
from   gurobipy import GRB
# -----------------------------------------------


def get_points(n : int, bb : Optional[Tuple]) -> np.array:
    l, r, b,t = bb
    return np.array([(np.random.uniform(l,r),np.random.uniform(b,t)) for _ in range(n)],dtype=np.float32)
    

def sign_area(p,v,q):
    area = np.linalg.det(
                np.array([
                    [p.x(), p.y(), 1],
                    [v.x(), v.y(), 1],
                    [q.x(), q.y(), 1]
                ]).astype('float')
            )
    return area


class Terrain:
    def __init__(self,filename = None,l=0,r=1,b=0,t=1):
        """
            terrain initializer
        """         
        # bounding-box
        self.bb = l,r,b,t

        self.Vertices = [] # geometrical objects of vertices
        self.Points   = [] # geometrical objects of points                

        if filename : self.read_tgpil(filename)

        # set discretization flag                    
        self.discrete = False            


    def generate(self,n):
        """
        default terrain generation: guards are placed on vertices
        """
        points = sorted(get_points(n-2,self.bb), key=lambda x : x[[0]])

        l,r,b,t = self.bb
                      
        # add extra vertices to the left and right to bound the Polygon region (simplify visualization)
        eps = 0.01
        # fix leftmost vertex on x=l
        points = np.insert(points,0,[[l,np.random.uniform(b,t)]],axis=0)
        # fix rightmost vertex on x=r
        points = np.append(points,[[r,np.random.uniform(b,t)]],axis=0)
                
        # add top-left and top-right dummy vertices to properly bound Polygonal region
        points = np.insert(points,0,[[l-eps,t+eps]],axis=0)
        points = np.append(points,[[r+eps,t+eps]],axis=0)        

        # define skgeom.Polygon
        self.poly = skgeom.Polygon(points) 

        # store points as geometric objects
        self.Points = np.array([ skgeom.Point2(p[0],p[1]) for p in points[1:-1]]) # points without dummies
        
        # define indices of points that are vertices
        self.num_vertices = self.num_points  = len(self.Points)
        self.vertices     = np.arange(self.num_vertices)
        self.Vertices     = self.Points
        
     
    def discretize(self):
        """
           discretization: compute extra terrain points which should be guarded to cover entire terrain
                
        """      
        l,r,b,t = self.bb

        # consider p+eps*e_2 point in skgeom.Polygonal region instead only  p due to numerics
        eps = 0.000001
      
        # arrangments
        arr = skgeom.arrangement.Arrangement()
        # add edges to arr
        for e in self.poly.edges : arr.insert(e)
                
        # compute visibility
        vs = skgeom.RotationalSweepVisibility(arr)

        # compute discretization points
        points = [self.Points[v] for v in self.vertices]
        
        # iterate over vertices of terrain
        for v in self.vertices:
            # pick point from the set
            p = self.Points[v]
            q = skgeom.Point2(p.x(),p.y()+eps)  # take y-close point as approximation of p
            # find face of arrangement
            face = arr.find(q)
            # compute visibility
            vx = vs.compute_visibility(q,face)            

            for u in vx.vertices:
                pt = u.point()
                exists = False
                for _pt in points:
                    # do not add close pooints
                    if skgeom.squared_distance(pt,_pt) < eps :                         
                        exists = True
                        break
                if exists : continue

                # choose point within the boundary
                if pt.y() <= t and pt.x() >= l and pt.x() <= r : points += [pt]

               
            # for some reason this code below doesn't produce the same code
            # points += [ u.point() for u in vx.vertices for v in points if squared_distance(u.point(),v)>eps and u.point().y() <= t and u.point().x() >= l and u.point().x() <= r ]                      
            # points += [u.point() for u in vx.vertices if u.point() not in points and u.point().y() <= t and u.point().x() >= l and u.point().x() <= r  ]
        
        # geometric data of points as x-sorted list 
        points = sorted(points,key=lambda p : p.x())
        
        # compute midpoint in essential intervals
        points.extend([skgeom.Point2(0.5*(points[i-1].x()+points[i].x()),0.5*(points[i-1].y()+points[i].y())) for i in range(1,len(points))])
       
        # dataset consists of vertices, points of terrain intersections and midpoints
        points = sorted(points,key = lambda p : p.x())

        # define number of points
        self.num_points   = len(points)
        
        # vertices as points
        #self.Vertices      =  [self.Points[v] for v in self.vertices]
        # find vertices in new set of points and store its references        
        #self.vertices      =  [points.index(v) for v in self.Vertices] 
        
        # new set of points
        self.Points     = points   # numpy array of (x,y)           
        # store indices of all points   
        self.points     = range(self.num_points)
        # define guards
        self.guards     =  self.vertices

        # set discretization flag
        self.discrete = True

              
    def check_visibility(self,i,j):
        """
        compute visibility among two guards i and point j
        """
        #p,q = self.Points[i], self.Points[j]
        p, q = self.Vertices[i], self.Points[j]
        visible = True
        if p.x() > q.x() : p,q = q,p         
        for v in self.Vertices:
            if v.x() <= p.x() : continue
            if v.x() >= q.x() : break               
            visible *= (sign_area(p,v,q) >= 0) 

        return visible



    def show(self):
       skgeom.draw.draw(self.poly,line_width=1,alpha=0.2,aspect_ratio = 'auto')        
       skgeom.draw.draw(self.Points,color = 'red')
        



    def write(self,filename):
        """
        Outputing terrain file.
            Format:
                number_of_points number_of_vertices
                # points
                x y # geometric position of points
                # vertices
                id1 id2 ...
                
        """        
        with open(filename,'w') as f:
            f.write('{} {}\n'.format(self.num_points,self.num_vertices))
            f.write('# points\n')
            for p in self.Points: f.write('{} {}\n'.format(p.x(),p.y()))
            f.write('# vertices\n')
            for v in self.vertices: f.write('{} '.format(v))
        
    
            
    
    def read(self,filename):
        """ 
            read terrain input format
                
                Parameters:
                    filename (string):  reads terrain from file                    

                Returns:
                    None
        """
        with open(filename,'r') as f:                        
            lines = f.readlines()
            self.num_points, self.num_vertices = tuple(map(int,lines[0].split()))
              
            self.Points = np.empty(self.num_points)
            self.vertices = np.empty(self.num_vertices)

            self.Points = [skgeom.Point2(*tuple(map(float,l.split()))) for l in lines[2 : 2 + self.num_points]]
            self.vertices = [int(i) for i in lines[3+self.num_points].split()]
                             
                                    
                
        # define dummy points so we can add them to skgeom.Polygon
        eps = 0.01
        points = np.array([[p.x(),p.y()] for p in self.Points])
        points = np.insert(points,0,[[-eps,1+eps]],axis=0)
        points = np.append(points,[[1+eps,1+eps]],axis=0)

        self.poly = skgeom.Polygon(points)


    def read_tgpil(self,input):
        """
        Reading format from TPIL
        """
        
       
        with open(input,'r') as f:
            lines = f.readlines()
            self.num_vertices = self.num_points = int(lines[0])
            self.Points = [skgeom.Point2(*tuple(map(float,l.split()))) for l in lines[1 : 1 + self.num_points]]
            
        
        # define dummy points so we can add them to skgeom.Polygon
        eps = 0.01
        points = np.array([[p.x(),p.y()] for p in self.Points])

        l,r = np.min(points[:,0]), np.max(points[:,0])
        b,t = np.min(points[:,1]), np.max(points[:,1])
        
        points = np.insert(points,0,[[l-eps,t+eps]],axis=0)
        points = np.append(points,[[r+eps,t+eps]],axis=0)     
        
        self.poly = skgeom.Polygon(points)     

        # set bounding box
        self.bb = l,r,b,t         

        self.vertices = list(range(self.num_vertices))
        self.Vertices = [self.Points[v] for v in self.vertices]

        
  
class TerrainSolver:
    T : Terrain

    def __init__(self, T):
        """
            define variables and contraints from the terrain data points
        """
        # verbose
        self.verbose = False
        # terrain
        self.T = T
        if not T.discrete : self.T.discretize()        
        
        # create model        
        self.m = gp.Model("ILP")        

        # dict. of variables
        x = dict()
        id = dict()

        # create variables
        if self.verbose: print('Defining variables and objective ...')
        for g in self.T.guards:
            x[g] = self.m.addVar(vtype=GRB.BINARY)
            self.m.update()
            id[x[g]] = g # correspond guards to solution
        
        # set objective
        self.m.setObjective(np.sum([x[g] for g in self.T.guards]),GRB.MINIMIZE)
        if self.verbose: print('done!')


        # add constraints
        if self.verbose: print('adding constraints to model ...')
        for p in self.T.points:
            self.m.addConstr(np.sum([x[g] for g in self.T.guards if self.T.check_visibility(g,p)]) >= 1) 
        if self.verbose: print('done!')

        # params
        self.m.setParam(GRB.Param.OutputFlag, self.verbose)
        
        self.m.Params.PoolSearchMode = 2
        self.m.Params.PoolSolutions = 1024
        self.m.Params.PoolGap = 0.0
        


        # make object variables
        self.x, self.id = x,id



    def solve(self):
        # solve
        self.m.optimize()
        # compute solution
                
        # gather all solutions
        self.solutions = [None] * self.m.SolCount
        for e in range(self.m.SolCount):
            self.m.setParam(GRB.Param.SolutionNumber, e)
            self.solutions[e] = [self.id[v] for v in self.m.getVars() if v.Xn > 0 ] 
        # filter only OPT values        
        self.solutions = [s for s in self.solutions if len(s) <= self.m.ObjVal ]
                   
            

    def sol_count(self):
        return len(self.solutions)
        
    def get_solution(self,e=0):
        return self.solutions[e]

    

    def show(self,e=0):
        self.T.show()
        for s in self.solutions[e]:
            skgeom.draw.draw(self.T.Points[s],color='blue',s = 100)

 

