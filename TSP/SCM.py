import cvxpy as cp
from data.Example import *

num, dist = ex1()
# num, dist = ex2()

x = cp.Variable(shape=(num, num), boolean=True)
y = cp.Variable(shape=(num, num), integer=True)

cons = [y >= 0,
        cp.sum(x, axis=0) - cp.diag(x) == 1,
        cp.sum(x, axis=1) - cp.diag(x) == 1,
        cp.sum(y[0, 1:]) == num - 1,
        y[0, 1:] <= (num - 1) * x[0, 1:],
        y[1:, 1:] <= (num - 2) * x[1:, 1:],
        cp.sum(y[:, 1:], axis=0) - cp.sum(y[1:, :], axis=1) == 1]

obj = cp.Minimize(cp.sum(cp.multiply(x, dist)))
prob = cp.Problem(objective=obj, constraints=cons)
# prob.solve(solver="GUROBI", verbose=True)
prob.solve(solver="COPT", verbose=True, TimeLimit=300)

tripsVar = np.array(np.round(x.value), dtype=int)
tripsPos = np.argwhere(tripsVar)
np.savetxt("plot/arc.txt", tripsPos, delimiter=',', fmt="%d")
