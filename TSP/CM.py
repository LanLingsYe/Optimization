import time
from data.Example import *

num, dist = ex1()
# num, dist = ex2()

start_time = time.time()
pop_size = 50
individuals = np.zeros(shape=(pop_size, num), dtype=int)

for k in range(pop_size):
    c = np.zeros(num, dtype=int)
    c[1:] = np.random.permutation(np.arange(1, num))
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

tmp1 = np.copy(individuals)
tmp2 = np.roll(tmp1, shift=1, axis=1)
diff_dist = dist[tmp2, tmp1]
cost = np.sum(diff_dist, axis=1)

path = individuals[np.argmin(cost), :]
optVal = np.min(cost)
end_time = time.time()
print("Total execution time: {:.2f} seconds".format(end_time - start_time))
print("The optimal: {:.2f} ".format(optVal))
np.savetxt("plot/path.txt", path, fmt="%d")
