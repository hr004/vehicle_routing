from sklearn.ensemble import RandomForestRegressor
import numpy as np
import random
from src.fit import main

random.seed(1)
np.random.seed(1)


def run():
    for n_train_instances in range(3, 10, 1):
        main(
            n_train_instances=n_train_instances,
            n_test_instances=10,
            n_vehicles=3,
            n_locations=20,
            exp_name="n_instances",
        )


if __name__ == "__main__":
    run()
