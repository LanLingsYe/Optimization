import networkx as nx
import time
from gurobipy import *
from data.Example import *

# num, dist = ex1()
num, dist = ex2()

def subtourEliminate(model, where):
    if where == GRB.Callback.MIPSOL:
        x_sol = model.cbGetSolution(model._x)

        tripsVar = np.array(np.round(x_sol), dtype=int)
        nodes = np.arange(0, num)
        tripsPos = np.argwhere(tripsVar)
        Gsol = nx.Graph()
        Gsol.add_nodes_from(nodes)
        Gsol.add_edges_from(map(tuple, tripsPos))
        numTours = nx.number_connected_components(Gsol)

        if numTours > 1:
            for tour in nx.connected_components(Gsol):
                comp = np.array(list(tour))
                model.cbLazy((model._x[comp[:, np.newaxis], comp]).sum() <= len(comp) - 1)


start_time = time.time()

model = Model(name="TSP")
model.params.TimeLimit = 300

x = model.addMVar(shape=(num, num), vtype=GRB.BINARY, name='x')
model.setObjective((x * dist).sum(), GRB.MINIMIZE)

model.addConstr(x.sum(axis=0) == 1)
model.addConstr(x.sum(axis=1) == 1)

model._x=x
model.Params.lazyConstraints = 1
model.optimize(subtourEliminate)

end_time = time.time()
print("Total execution time: {:.2f} seconds".format(end_time - start_time))
print("The optimal: {:.2f} ".format(model.objVal))

tripsVar = np.array(x.x, dtype=int)
tripsPos = np.argwhere(tripsVar)
np.savetxt("plot/arc.txt", tripsPos, delimiter=',', fmt="%d")
