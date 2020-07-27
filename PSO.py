# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 16:13:49 2020

@author: e10832
"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

class PSO():
    def __init__(self, fit_func, num_dim, num_particle=20, max_iter=500,
                 x_max=1, x_min=-1, w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, k=0.2):
        self.fit_func = fit_func        
        self.num_dim = num_dim
        self.num_particle = num_particle
        self.max_iter = max_iter

        self.x_max = x_max
        self.x_min = x_min
        self.w = w_max        
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self._iter = 1
        self.gBest_curve = np.zeros(self.max_iter)

        self.X = np.random.uniform(low=self.x_min, high=self.x_max,
                                   size=[self.num_particle, self.num_dim])
        self.V = np.zeros(shape=[self.num_particle, self.num_dim])
        self.v_max = self.k*(self.x_max-self.x_min)


        self.pBest_X = self.X.copy()
        self.pBest_score = self.fit_func(self.X)
        self.gBest_X = self.pBest_X[self.pBest_score.argmin()]
        self.gBest_score = self.pBest_score.min()
        self.gBest_curve[0] = self.gBest_score.copy()

    def opt(self):
        while(self._iter<self.max_iter):   
            R1 = np.random.uniform(size=(self.num_particle, self.num_dim))
            R2 = np.random.uniform(size=(self.num_particle, self.num_dim))
            w = self.w_max - self._iter*(self.w_max-self.w_min)/self.max_iter
            for i in range(self.num_particle):                
                self.V[i, :] = w * self.V[i, :] \
                        + self.c1 * (self.pBest_X[i, :] - self.X[i, :]) * R1[i, :] \
                        + self.c2 * (self.gBest_X - self.X[i, :]) * R2[i, :]               
                self.V[i, self.v_max < self.V[i, :]] = self.v_max[self.v_max < self.V[i, :]]
                self.V[i, -self.v_max > self.V[i, :]] = -self.v_max[-self.v_max > self.V[i, :]]
                
                self.X[i, :] = self.X[i, :] + self.V[i, :]
                self.X[i, self.x_max < self.X[i, :]] = self.x_max[self.x_max < self.X[i, :]]
                self.X[i, self.x_min > self.X[i, :]] = self.x_min[self.x_min > self.X[i, :]]
                
        
                score = self.fit_func(self.X[i, :])
                if score<self.pBest_score[i]:
                    self.pBest_score[i] = score.copy()
                    self.pBest_X[i, :] = self.X[i, :].copy()
                    if score<self.gBest_score:
                        self.gBest_score = score.copy()
                        self.gBest_X = self.X[i, :].copy()

            self.gBest_curve[self._iter] = self.gBest_score.copy()         
            self._iter = self._iter + 1 

    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()    