import cvxpy as cp
import networkx as nx
import time

import numpy as np

from data.Example import *

num, dist = ex1()
# num, dist = ex2()

start_time = time.time()
trips = cp.Variable(shape=(num, num), boolean=True)
cons = [cp.sum(trips, axis=0) - cp.diag(trips) == 1,
        cp.sum(trips, axis=1) - cp.diag(trips) == 1]

obj = cp.Minimize(cp.sum(cp.multiply(trips, dist)))
prob = cp.Problem(objective=obj, constraints=cons)
tspsol = prob.solve(solver="COPT")
# tspsol = prob.solve(solver="GUROBI")

tripsVar = np.array(np.round(trips.value), dtype=int)
nodes = np.arange(0, num)
tripsPos = np.argwhere(tripsVar)
Gsol = nx.Graph()
Gsol.add_nodes_from(nodes)
Gsol.add_edges_from(map(tuple, tripsPos))

numTours = nx.number_connected_components(Gsol)
print('# of numTours: %d' % numTours)
while numTours > 1:
    for tour in nx.connected_components(Gsol):
        comp = np.array(list(tour))
        cons.append(cp.sum(trips[comp[:, np.newaxis], comp]) <= len(comp) - 1)

    prob = cp.Problem(objective=obj, constraints=cons)
    tspsol = prob.solve(solver="COPT")
    # tspsol = prob.solve(solver="GUROBI")

    tripsVar = np.array(np.round(trips.value), dtype=int)
    tripsPos = np.argwhere(tripsVar)
    Gsol = nx.Graph()
    Gsol.add_nodes_from(nodes)
    Gsol.add_edges_from(map(tuple, tripsPos))

    numTours = nx.number_connected_components(Gsol)
    print('# of numTours: %d' % numTours)
end_time = time.time()
print("Total execution time: {:.2f} seconds".format(end_time - start_time))
print("The optimal: {:.2f} ".format(tspsol))

np.savetxt("plot/arc.txt", tripsPos, delimiter=',', fmt="%d")
