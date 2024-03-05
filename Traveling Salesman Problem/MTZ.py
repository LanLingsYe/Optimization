import cvxpy as cp
from data.Example import *

# num, dist = ex1()
num, dist = ex2()

x = cp.Variable(shape=(num, num), boolean=True)
mu = cp.Variable(shape=num, integer=True)

cons = [cp.sum(x, axis=0) - cp.diag(x) == 1,
        cp.sum(x, axis=1) - cp.diag(x) == 1,
        mu >= 0,
        mu <= num - 1,
        mu[0] == 0]

for i in range(1, num):
    for j in range(1, num):
        cons.append(mu[i] - mu[j] + num * x[i, j] <= num - 1)

obj = cp.Minimize(cp.sum(cp.multiply(x, dist)))
prob = cp.Problem(objective=obj, constraints=cons)
prob.solve(solver="COPT", verbose=True, TimeLimit=300)
# prob.solve(solver="GUROBI", verbose=True, TimeLimit=300)

tripsVar = np.array(np.round(x.value), dtype=int)
tripsPos = np.argwhere(tripsVar)
np.savetxt("plot/arc.txt", tripsPos, delimiter=',', fmt="%d")
