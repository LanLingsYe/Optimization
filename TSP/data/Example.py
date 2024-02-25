import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def ex1():
    data = pd.read_csv("data\pl101.txt")
    num = len(data)
    data = data * np.pi / 180

    # 计算距离矩阵
    R = 6370
    dist = np.zeros(shape=(num, num))
    for i in range(0, num - 1):
        for j in range(i + 1, num):
            dist[i, j] = R * np.arccos(np.cos(data.iloc[i, 0] - data.iloc[j, 0]) *
                                       np.cos(data.iloc[i, 1]) * np.cos(data.iloc[j, 1]) +
                                       np.sin(data.iloc[i, 1]) * np.sin(data.iloc[j, 1]))
    dist = dist + np.transpose(dist)
    dist = np.round(dist, decimals=2)
    return num, dist


def ex2():
    data = pd.read_csv("data\ch130.txt")
    num = len(data)
    dist = cdist(data, data, 'euclidean')
    return num, dist
