import numpy as np
import random
from src.fit import main

random.seed(1)
np.random.seed(1)


def run():
    main(
        n_train_instances=10,
        n_test_instances=10,
        n_vehicles=3,
        n_locations=20,
        exp_name="n_locations_more_heuristics_parameters",
        evals=10,
    )


if __name__ == "__main__":
    run()
