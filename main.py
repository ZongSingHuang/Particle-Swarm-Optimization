import numpy as np
import plotly.graph_objects as go

from benchmark import Benchmark
from PSO import pso

benchmark = Benchmark()
sphere = benchmark.Sphere()
name = "Sphere"
times = 30
curves = list()
costs = list()
gbest_fs = list()
for _ in range(times):
    solver = pso(
        max_iter=500,
        pop_size=30,
        k=0.2,
        w_max=0.9,
        w_min=0.4,
        c1=2.0,
        c2=2.0,
        lb=-100 * np.ones(30),
        ub=100 * np.ones(30),
        name=name,
        benchmark=sphere.evaluate,
    )
    solver.run()
    curves.append(solver.curve)
    costs.append(solver.cost)
    gbest_fs.append(solver.gbest_f)

# 創建 Figure
fig = go.Figure()

# 添加數據列到 Figure
for curve in curves:
    fig.add_trace(go.Scatter(y=curve, mode="lines+markers"))

# 設置圖表標題及坐標軸標籤
fig.update_layout(
    title=f"{name}(f_worst: {max(gbest_fs):.2f}, f_avg: {np.mean(gbest_fs):.2f}, f_best: {min(gbest_fs):.2f},  avg cost: {np.mean(costs):.2f} sec)",
    xaxis_title="Iteration",
    yaxis_title="Fitness",
)

# 顯示圖表
fig.show()
