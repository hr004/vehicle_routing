from cpmpy.solvers import param_combinations


params = {
    "MIPFocus": [0, 1, 2, 3],
    "Heuristics": [0, 0.05, 0.25, 0.5, 0.75, 1],
    "VarBranch": [-1, 0, 1, 2, 3],
}

param_grid = [pars for pars in param_combinations(params)]
