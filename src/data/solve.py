from cpmpy.solvers import CPM_gurobi, param_combinations
import random
from src.data.utils import decode_x, visualize_map_routes
from src.data.generate import train_dataset_models, train_dataset_variables, train_dataset_coordinates

random.seed(1)


# This function runs the solver on a single model (instance) and returns the
# solver runtime
def get_time(model, pars=None, variables=None, coordinates=None, visualize=False):
    solver = CPM_gurobi(model)
    # run solver with provided parameters
    # assume the solver is stochastic (as implied by the random seed below), i.e.,
    # two runs of solver on the same instance can have different runtimes
    if pars is not None:
      solver.solve(**pars, seed=random.randint(0,1000))
    # else, run solver with its default parameters
    else:
      solver.solve(seed=random.randint(0,1000))

    # optionally inspect solution quality and solver status
    # print(f"Total distance: {model.objective_value()} meter -- {solver.status()}")

    # optionally visualize solution on a map
    if visualize and variables is not None and coordinates is not None:
      routes = decode_x(variables.value(), coordinates[0], coordinates[1])
      visualize_map_routes(routes)
    return solver.status().runtime

# Wrapper for the above function; runs it on a set of instances instead of just one
def get_time_dataset(dataset_models, pars, dataset_variables=None, dataset_coordinates=None):
    rt = 0
    for i in range(len(dataset_models)):
      rt += get_time(dataset_models[i], pars, variables=dataset_variables[i], coordinates=dataset_coordinates[i], visualize=False)
    rt /= len(dataset_models)*1.0
    return rt



params ={
   "MIPFocus": [0,1,2,3],
   "Heuristics": [0, 0.05, 0.25, 0.5, 0.75, 1],
   "VarBranch" : [-1, 0, 1, 2, 3],
}

param_grid = [ pars for pars in param_combinations(params) ]


# Initial sampling of the parameter grid to initialize the regression model
# for model-based algorithm configuration

# this is the number of initial samples to use
initial_samples = 50
data = []

for pars in random.sample(param_grid, initial_samples):
    rt = get_time_dataset(train_dataset_models, pars, train_dataset_variables, train_dataset_coordinates)
    data += [ list(pars.values()) + [ rt ] ]
data