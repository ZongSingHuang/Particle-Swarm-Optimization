import time
from typing import Callable

import numpy as np
import plotly.graph_objects as go


class pso:
    def __init__(
        self,
        max_iter: int,
        pop_size: int,
        k: float,
        w_max: float,
        w_min: float,
        c1: float,
        c2: float,
        lb: list,
        ub: list,
        benchmark: Callable,
        name: str = "",
    ) -> None:
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.v_max = k * (ub - lb)
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)
        self.name = name
        self.benchmark = benchmark

        self.cost = 0
        self.curve = list()
        self.gbest_f = np.inf
        self.gbest_x = np.zeros(self.dim)
        self.pbest_F = np.full(self.pop_size, np.inf)
        self.pbest_X = np.zeros([self.pop_size, self.dim])

    def run(self):
        st = time.time()

        # 初始化
        X = np.random.uniform(low=self.lb, high=self.ub, size=[self.pop_size, self.dim])
        V = np.zeros([self.pop_size, self.dim])

        # 迭代
        for _iter in range(self.max_iter):
            # 適應值計算
            F = self.benchmark(X)

            # 更新最佳解
            mask = F < self.pbest_F
            self.pbest_X[mask] = X[mask].copy()
            self.pbest_F[mask] = F[mask].copy()

            if self.pbest_F.min() < self.gbest_f:
                idx = self.pbest_F.argmin()
                self.gbest_x = self.pbest_X[idx].copy()
                self.gbest_f = self.pbest_F.min()

            # 收斂曲線
            self.curve.append(self.gbest_f)

            # 更新
            R1 = np.random.uniform(size=[self.pop_size, self.dim])
            R2 = np.random.uniform(size=[self.pop_size, self.dim])
            w = self.w_max - (self.w_max - self.w_min) * (_iter / self.max_iter)

            V = (
                w * V
                + self.c1 * (self.pbest_X - X) * R1
                + self.c2 * (self.gbest_x - X) * R2
            )  # 更新V
            V = np.clip(V, -self.v_max, self.v_max)  # 邊界處理

            X += V  # 更新 X
            X = np.clip(X, self.lb, self.ub)  # 邊界處理

        # 總計算時間
        ed = time.time()
        self.cost = round(ed - st, 2)

    def plot(self):
        if self.curve:
            # 創建 Figure
            fig = go.Figure()

            # 添加數據列到 Figure
            fig.add_trace(go.Scatter(y=self.curve, mode="lines+markers"))

            # 設置圖表標題及坐標軸標籤
            fig.update_layout(
                title=f"{self.name}(best fitness: {min(self.curve):.2f}, cost: {self.cost} sec)",
                xaxis_title="Iteration",
                yaxis_title="Fitness",
            )

            # 顯示圖表
            fig.show()
