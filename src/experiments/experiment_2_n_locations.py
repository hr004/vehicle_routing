from sklearn.ensemble import RandomForestRegressor
import numpy as np
import random
from src.fit import main

random.seed(1)
np.random.seed(1)

import pathlib


def run():
    for n_locations in range(26, 30, 1):
        main(
            n_train_instances=10,
            n_test_instances=10,
            n_vehicles=3,
            n_locations=n_locations,
            exp_name="n_locations",
        )


if __name__ == "__main__":
    run()
