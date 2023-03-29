# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:25:21 2023

@author: junhyeok

Here is the Particle Swarm Optimization (Heuristic Optimization)

User-Defined Hyper parameter: 
    upper boundary of velocity,
    lower boundary of velocity,
    importance matric of particle itself 
    importance matric of the group.

1. Init x, v

"""

import os, sys
import numpy as np

from scipy import linalg as LN
import math as m
import matplotlib.pyplot as plt

import cvxpy as cp
import tabulate
import time

class PSO(object):
    
    def __init__(self, func, param_dict):
        
        #self.init_pos = param_dict['init_pos']
        self.n_params = param_dict['n_params']
        self.n_particles = param_dict['n_particles']
        self.max_iter = param_dict['max_iter']
        
        self.c0 = param_dict['c0']
        self.c1 = param_dict['c1']
        self.w = param_dict['w']
        self.xl = param_dict['xl']
        self.xu = param_dict['xu']
        
        self.vl = param_dict['vl']
        self.vu = param_dict['vu']
        self.count_max = param_dict['count_max']
        self.init_particle = np.array([gamma*np.random.uniform() for i in range(n_particles)])
        self.init_vec = np.array([(self.vu-self.vl)*np.random.uniform()+self.vl for i in range(n_particles)])
        
        self.g_best = self.init_particle[0] # 초기 g_best 설정 고민
        self.p_best = self.init_particle
        
        self.func = func
        
        PSO.optimize(self)
        
    def calculate_velocity(self, x, v, p_best, g_best):
        """
            Update particle velocity
            
            Args:
                x(array-like): particle current position
                v (array-like): particle current velocity
                p_best(array-like): the best position found so far for a particle
                g_best(array-like): the best position regarding all the particles found so far
                c0 (float): the congnitive scaling constant, 인지 스케일링 상수
                c1 (float): the social scaling constant
                w (float): the inertia weight, 관성 중량
                
            Returns:
                The updated velocity (array-like).
        """
        x = np.array(x)
        v = np.array(v)
        
        # Generate random between 0 and 1
        r0 = np.random.uniform()
        r1 = np.random.uniform()
        
        new_v = self.w*v + self.c0*r0*(p_best - x) + self.c1*r1*(g_best - x)
        
        #print("v[0]:", new_v[0])
        #print("velocity = ", new_v)
        
        return new_v
        
    def update_position(self, x, v):
        
        x = np.array(x)
        v = np.array(v)
        new_x = x + v
        
        # Limit x
        # velocity limit이 아니라, x limit하는 방법이 맞나?
        
        for i in range(len(new_x)):
            if new_x[i] < self.xl:
                new_x[i] = self.xl
            elif new_x[i] > self.xu:
                new_x[i] = self.xu
        #print("position = ", new_x )
        
        return new_x
    
    def evaluate_score(self, new_x, p_best, g_best):
        
        for i in range(self.n_particles):
            
            f_x = self.func(new_x[i])
            f_pbest = self.func(p_best[i])
            if f_x < f_pbest:
                p_best[i] = new_x[i]
                
                #print(f"renewable p_best[{i}]:", p_best[i])
                f_gbest = self.func(g_best)
                if f_x < f_gbest:
                    #print("renewable g_best:", f_gbest)
                    g_best = new_x[i]
            
        return p_best, g_best    
                
        
    def optimize(self):
        
        iteration = 0
        count = 0
        count_print = 0
        
        x = np.round(self.init_particle,4)
        v = np.round(self.init_vec,4)
        while True:
            
            print(iteration, self.g_best, self.func(self.g_best))
            if iteration % (10**count_print):
                count_print += 1
                print('*'*count_print)    
                        
            if iteration >= self.max_iter:
                print("PSO max_iter")
                break
            elif count >= self.count_max:
                print("Converged the g_best on count_max")
                break
                
            new_v = self.calculate_velocity(x,v,self.p_best, self.g_best)
            new_x = self.update_position(x, new_v)
            
            p_best_new, g_best_new = self.evaluate_score(new_x, self.p_best, self.g_best)
            
            if g_best_new == self.g_best:
                count += 1
            else:
                count = 0
                
            self.p_best = p_best_new
            self.g_best = g_best_new
            
            v = np.round(new_v,4)
            x = np.round(new_x,4)
            iteration += 1    
        return self.g_best, self.func(self.g_best) 
            
    
if __name__ == '__main__':
    
    start = time.time()
    gamma = 0.1
    n_particles = 100
    param_dict = {
        'n_params': 2
        'n_particles': n_particles,
        'max_iter': 10000,
        'c0': gamma,
        'c1': gamma,
        'w': 0.9,
        'xl': 0.0001,
        'xu': 10.0,
        'vl': 0.0001,
        'vu': 0.5,
        'count_max': 100
    }
    
        
    BASE_DIR = os.getcwd()
    
    os.chdir("../")
    path = os.getcwd()
    sys.path.append(path) # 폴더 두 단계 위에서 file import 하기 위해서 sys path 설정  

    ## Import the function that optimize   
    from Lyapunov_SDP_Stability.analyze_stability import lyapunov         
    #from analyze_stability import lyapunov
    m = PSO(lyapunov, param_dict)
    end = time.time()
    print("Simulation Time:", end - start)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            