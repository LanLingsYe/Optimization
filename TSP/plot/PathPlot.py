import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("..\\data\\pl101.txt")
# data=pd.read_csv("..\\data\\ch130.txt")
path = pd.read_csv("path.txt", header=None)

plt.figure(figsize=(10, 6))
cities = data.apply(lambda x: tuple(x), axis=1).values.tolist()
# 绘制所有城市的位置
for city in cities:
    plt.plot(city[0], city[1], 'ro')

# 绘制路径
former = np.array(path).reshape(-1)
latter = np.roll(former, shift=-1)
for i, j in zip(former, latter):
    plt.arrow(cities[i][0], cities[i][1], cities[j][0] - cities[i][0], cities[j][1] - cities[i][1],
              length_includes_head=True, head_width=0.3, fc='blue', ec='blue')

plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('TSP Solution Visualization')
plt.grid(True)
plt.show()
