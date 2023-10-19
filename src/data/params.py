from cpmpy.solvers import param_combinations
import numpy as np


heuristics_range = np.arange(0,1,0.05, dtype=float)

params = {
    "MIPFocus": [0, 1, 2, 3],
    "Heuristics": heuristics_range,
    "VarBranch": [-1, 0, 1, 2, 3],
    # "ImproveStartGap": [0.1, 0.2, 0.3],
    # "PoolGap": [0.1,0.2,0.3]
}

param_grid = [pars for pars in param_combinations(params)]
