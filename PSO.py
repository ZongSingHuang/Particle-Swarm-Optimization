# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 16:13:49 2020

@author: e10832
"""
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

class PSO():
    def __init__(self, fitness, D=30, P=20, G=500, ub=1, lb=0,
                 w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, k=0.2):
        self.fitness = fitness
        self.D = D
        self.P = P
        self.G = G
        self.ub = ub*np.ones([self.P, self.D])
        self.lb = lb*np.ones([self.P, self.D])
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.v_max = self.k*(self.ub-self.lb)
        
        self.pbest_X = np.zeros([self.P, self.D])
        self.pbest_F = np.zeros([self.P]) + np.inf
        self.gbest_X = np.zeros([self.D])
        self.gbest_F = np.inf
        self.loss_curve = np.zeros(self.G)

    def opt(self):
        # 初始化
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])
        self.V = np.zeros([self.P, self.D])
        
        # 迭代
        for g in range(self.G):
            # 適應值計算
            F = self.fitness(self.X)
            
            # 更新最佳解
            mask = F < self.pbest_F
            self.pbest_X[mask] = self.X[mask].copy()
            self.pbest_F[mask] = F[mask].copy()
            
            if np.min(F) < self.gbest_F:
                idx = F.argmin()
                self.gbest_X = self.X[idx].copy()
                self.gbest_F = F.min()
            
            # 收斂曲線
            self.loss_curve[g] = self.gbest_F
            
            # 更新
            r1 = np.random.uniform(size=[self.P, self.D])
            r2 = np.random.uniform(size=[self.P, self.D])
            w = self.w_max - (self.w_max-self.w_min)*(g/self.G)
            
            self.V = w * self.V + self.c1 * (self.pbest_X - self.X) * r1 \
                                + self.c2 * (self.gbest_X - self.X) * r2 # 更新V
            self.V = np.clip(self.V, -self.v_max, self.v_max) # 邊界處理
            
            self.X = self.X + self.V # 更新X
            self.X = np.clip(self.X, self.lb, self.ub) # 邊界處理

    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()