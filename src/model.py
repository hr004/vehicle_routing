import numpy as np
from cpmpy import intvar, Model


def vehicle_routing_model(distance_matrix: np.ndarray, n_vehicle, capacity, demand):
    """
    An integer linear programming model capacitated vehicle routing problem
    """
    n_city = distance_matrix.shape[0]

    x = intvar(0, 1, shape=distance_matrix.shape)

    y = intvar(0, capacity, shape=distance_matrix.shape)

    model = Model(
        # constraint on number of vehicles (from depot, which is assumed to be node 0)
        sum(x[0, :]) == n_vehicle,
        # vehicle leaves and enter each node i exactly once
        [sum(x[i, :]) == 1 for i in range(1, n_city)],
        [sum(x[:, i]) == 1 for i in range(1, n_city)],
        # no self visits
        [sum(x[i, i] for i in range(n_city)) == 0],
        # from depot (which is assumed to be node 0) takes no load
        sum(y[0, :]) == 0,
        # flow out of node i through all outgoing arcs is equal to
        # flow into node i through all ingoing arcs + load capacity @ node i
        [sum(y[i, :]) == sum(y[:, i]) + demand[i] for i in range(1, n_city)],
    )

    # capacity constraint at each node (conditional on visit)
    for i in range(n_city):
        for j in range(n_city):
            model += y[i, j] <= capacity * x[i, j]

    # the objective is to minimze the travelled distance
    # sum(x*dist) does not work because 2D array, use .sum()
    model.minimize((x * distance_matrix).sum())

    return model, x
