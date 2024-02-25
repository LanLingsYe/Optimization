import time
from data.Example import *

num, dist = ex1()
# num, dist = ex2()
T1 = time.time()


def compute_distance(routine):
    num_points, = routine.shape
    return sum([dist[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


# 遗传算法
from sko.GA import GA_TSP

# prob_mut 种群的新成员由变异而非交叉得来的概率
# func:目标函数，适应度函数， n_dim：决策变量，在这里是城市的数量， size_pop:种群个数，max_iter:迭代轮数， prob_mut:变异概率
ga_tsp = GA_TSP(func=compute_distance, n_dim=num, size_pop=200, max_iter=2000, prob_mut=0.4)
best_points, best_distance = ga_tsp.run()


T2 = time.time()
print('程序运行时间:%.6s秒' % ((T2 - T1)))
np.savetxt("plot/path.txt", best_points, fmt="%d")


