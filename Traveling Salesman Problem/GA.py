import time
from data.Example import *

num, dist = ex1()
# num, dist = ex2()

start_time = time.time()


def fit(individuals):
    """
    适应度函数，即目标函数
    :param individuals:个体基因，通常是0方向为个体，1方向为基因
    :return:每个个体基因的适应度
    具体方法先复制了每个个体的顶点顺序tmp1，然后得到每个顶点的下一个顶点的顺序tmp2，利用距离矩阵求得这两个矩阵
    每个元素对应的距离diff_dist,仍然是0方向表示个体，1方向表示每个边的长度，按1方向求和即可。
    """
    if individuals.ndim == 1:
        tmp1 = np.copy(individuals)
        tmp2 = np.roll(tmp1, shift=1)
        diff_dist = dist[tmp2, tmp1]
        cost = np.sum(diff_dist)
    elif individuals.ndim == 2:
        tmp1 = np.copy(individuals)
        tmp2 = np.roll(tmp1, shift=1, axis=1)
        diff_dist = dist[tmp2, tmp1]
        cost = np.sum(diff_dist, axis=1)
    return np.round(cost, decimals=2)


def modify(individuals):
    num = individuals.shape[1]
    pop_size = individuals.shape[0]
    for k in range(pop_size):
        c = individuals[k, :]
        for t in range(num):
            flag = True
            for m in np.arange(1, num - 1):
                for n in np.append(np.arange(m + 2, num - 1), 0):
                    if dist[c[m], c[n]] + dist[c[m + 1], c[n + 1]] < dist[c[m], c[m + 1]] + dist[c[n], c[n + 1]]:
                        c[m + 1:n + 1] = c[n:m:-1]
                        flag = False
            if flag:
                break
        individuals[k, :] = c
    return individuals


def generate(population_size, num):
    individuals = np.zeros(shape=(population_size, num), dtype=int)
    for i in range(0, population_size):
        individuals[i, 1:] = np.random.permutation(np.arange(1, num))
    individuals = modify(individuals=individuals)
    chromosome = individuals.astype("float") / num
    return chromosome


def cross(chromosome):
    num = np.size(chromosome, axis=1)
    children = chromosome.copy()
    chromosome_fitness = fit(individuals=np.argsort(np.argsort(chromosome, axis=1), axis=1))
    fitness_sequence = np.argsort(chromosome_fitness)
    for i, j in zip(fitness_sequence[0::2], fitness_sequence[1::2]):
        pos = np.random.randint(low=1, high=num)
        children[[i, j], pos] = children[[j, i], pos]
    return children


def mutate(chromosome):
    M = np.size(chromosome, axis=0)
    children = chromosome.copy()
    for i in range(M):
        pos = np.random.randint(low=1, high=num, size=2)
        children[i, pos] = np.random.rand(2)
    return children


def select(chromosome, M):
    chromosome_fitness = fit(individuals=np.argsort(np.argsort(chromosome, axis=1), axis=1))
    fitness_sequence = np.argsort(chromosome_fitness)
    return chromosome[fitness_sequence[:M], :]


# 参数设定
M = 20
G = 40

# 初始化
chromosome = generate(population_size=M, num=num)
population = np.argsort(np.argsort(chromosome, axis=1), axis=1)

for i in range(G):
    children1 = cross(chromosome=chromosome)
    children2 = mutate(chromosome=chromosome)
    alien = generate(population_size=int(0.1 * M), num=num)
    chromosome = select(chromosome=np.concatenate((chromosome, children1, children2, alien), axis=0), M=M)

population = np.argsort(np.argsort(chromosome, axis=1), axis=1)
cost = fit(individuals=population)
path = population[np.argmin(cost), :]
optVal = np.min(cost)
end_time = time.time()
print("Total execution time: {:.2f} seconds".format(end_time - start_time))
print("The optimal: {:.2f} ".format(optVal))

np.savetxt("plot/path.txt", path, fmt="%d")
