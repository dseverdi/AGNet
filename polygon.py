import os
from   pathlib import Path
import csv
import pandas as pd
import skgeom
from matplotlib.pyplot import figure

# typing
from typing import Optional, Type, Union
import numpy as np
import math

# optimizers
import gurobipy as     gp
from   gurobipy import GRB

import functools





def compose2(f,g):
    return lambda *a, **kw : f(g(*a,**kw))

def compose(*fs):
    return functools.reduce(compose2,fs)



class Polygon:
    def __init__(self : Type["Polygon"], filename : Optional[str] = None, poly : Optional[skgeom.Polygon] = None):
        """
        construct polygon from .pol file or make empty polygon.
        """

        if filename : poly = self.read_pol(filename) 
        else:
            self.no_vertices = 0
            self.poly = skgeom.Polygon()

        self.poly = poly
        # vertices        
        self.Vertices           = list(self.poly.vertices) # geometric objects        
        self.no_vertices        = len(self.Vertices)
        self.vertices           =list(range(self.no_vertices))

        # points
        self.points   = [] # index of points        
        self.Points   = [] # geometric objects

        # views
        self.view     = dict()
        

         # compute visibility regions of vertices
        self.__init_vis_queries()

        self.no_points = self.discretize()
        self.points = list(range(self.no_points))


    def __init_vis_queries(self : Type["Polygon"]) -> None:
        """
            compute visibility regions from set of points
        """
        eps = 0.00001

        self.arr = skgeom.arrangement.Arrangement()
        self.vs  = skgeom.TriangularExpansionVisibility(self.arr)

        for e in self.poly.edges: 
            self.arr.insert(e)
            #skgeom.draw.draw(e)

        
        n  = self.no_vertices

        for i in range(n): 
            prev, curr, nxt = i, (i+1) % n, (i+2) % n
            # find close enough point from interior to vertex vertices[i]
            v_prev, v, v_next = self.Vertices[prev], self.Vertices[curr], self.Vertices[nxt]         
        
            # affine combination of adjacent vertices
            p = skgeom.Vector2(v,v_prev)
            p = 1.0/math.sqrt(p.squared_length()) * p
            r = skgeom.Vector2(v,v_next) 
            r = 1.0/math.sqrt(r.squared_length()) * r

            q = v+eps*(p+r)
            # change orientation if point outside of polygon
            q = v-eps*(p+r) if self.poly.oriented_side(q) == skgeom.Sign.NEGATIVE else q 
                       
            # find face where q belongs
            f = self.arr.find(q)
            
            # compute visibility
            try:
                vx = self.vs.compute_visibility(q,f)        
            except Exception as e:        
                print(f'Error at {i} vertex: ',e)
                        
            # compute visiblity region for q as a polygon
            self.view[v] = skgeom.Polygon([_.point() for _ in vx.vertices])
            # use views to compute visibility
            
        
    def read_pol(self : Type["Polygon"], filename : str ) -> None:        
        """
            reading CGAL polygon instance
        """
        scale = lambda x,a,b : (x-a)/(b-a) if b != a else x-a
        
        with open(filename,'r') as f:
            tokens = f.read().split()
            self.no_vertices = int(tokens[0])
            # read number of points            
            
            _points = np.array([(eval(tokens[i]),eval(tokens[i+1])) for i in range(1,2*self.no_vertices,2)],dtype='float64')
                        
            # define bounding box
            l,r = np.min(_points[:,0]), np.max(_points[:,0])
            b,t = np.min(_points[:,1]), np.max(_points[:,1])

            l,r,b,t = 0,1,0,1
            
            verts = [skgeom.Point2(scale(p[0],l,r),scale(p[1],b,t)) for p in _points]            
                        
        return skgeom.Polygon(verts)
        
        

    def check_visibility(self : Type["Polygon"], p : skgeom.Point2, q : skgeom.Point2) -> bool:
        # use cached visibility regions for fast computation
        
        if p == q: return True
        
        eps = 0.0001
        if p in self.Vertices:            
            r = skgeom.Vector2(p.x()-q.x(),p.y()-q.y())
            r = 1.0/math.sqrt(r.squared_length()) * r 
            # regular or degenerative cases
            return self.view[p].oriented_side(q+eps*r) == skgeom.Sign.POSITIVE or sum([e.has_on(p) and e.has_on(q+eps*r) for e in self.view[p].edges])>0
        
        elif q not in self.Vertices:            
            f = self.arr.find(p)        
            # compute visibility
            vx = self.vs.compute_visibility(p,f)                
            # compute new view       
            view = skgeom.Polygon([v.point() for v in vx.vertices])
            return view.oriented_side(q) == skgeom.Sign.POSITIVE
        else: # check symmetric case
            return self.check_visibility(q,p)


    def discretize(self : Type["Polygon"]) -> float:
        """
            discretize polygon by constructing AVPs and use its centroids as discretization points
        """
        # arrangement of all visibility regions
        avp_arr = skgeom.arrangement.Arrangement()

        # add edges from visiblity polygons to the avp arrangement        
        for view in self.view.values():
            for e in view.edges:
                avp_arr.insert(e)
        
        
        # construct avps
        self.avps = []

        for f in avp_arr.faces:    
            if f.has_outer_ccb(): # check if face is traversable on boundary
                it = f.outer_ccb
                first = next(it)
                halfedge = next(it)
                verts = []
                # circular traversal of face 
                while first != halfedge:                    
                    verts.append(halfedge.source().point())            
                    halfedge = next(it)
                verts.append(halfedge.source().point())                                        
                self.avps.append(skgeom.Polygon(verts))     

        # points that should be guarded
        self.Points = [ skgeom.Point2( 1.0/len(avp) * sum([p.x() for p in avp.vertices]), 1.0/len(avp) * sum([p.y() for p in avp.vertices])) for avp in self.avps]   
        
        return len(self.Points)
        
    
    def show(self : Type["Polygon"]):
        """
            Displaying polygon with discretization
        """
        print(f'no. of vertices: {self.no_vertices}, no. of points: {self.no_points}')
        skgeom.draw.draw(self.poly,line_width=1,alpha=0.2,aspect_ratio = 'auto')        
        skgeom.draw.draw(self.Vertices,color = 'black')
        if self.Points:
            skgeom.draw.draw(self.avps,facecolor='red',alpha=0.2)
            skgeom.draw.draw(self.Points,color='red',s=10)


class AGSolver:
    P : Polygon

    def __init__(self : Type["AGSolver"], P : Type["Polygon"]):
        """
            define variables and contraints from the terrain data points
        """
        # verbose
        self.verbose = False
        # polygon
        self.P = P        
        
        # create model        
        self.m = gp.Model("ILP")        

        # dict. of variables
        x  = dict()
        id = dict()

        # create variables
        if self.verbose: print('Defining variables and objective ...')
        for g in self.P.vertices:
            x[g] = self.m.addVar(vtype=GRB.BINARY)
            # update model
            self.m.update()
            id[x[g]] = g # correspond guards to solution
        
        
        # set objective
        self.m.setObjective(np.sum([x[g] for g in self.P.vertices]),GRB.MINIMIZE)
        if self.verbose: print('done!')


        # add constraints
        if self.verbose: print('adding constraints to model ...')
        for p in self.P.points:
            self.m.addConstr(np.sum([x[g] for g in self.P.vertices if self.P.check_visibility(self.P.Vertices[g],self.P.Points[p])]) >= 1) 
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
        figure(figsize=(14, 16), dpi=80)
        self.P.show()
        for s in self.solutions[e]:
            skgeom.draw.draw(self.P.Vertices[s],color='blue',s = 100)