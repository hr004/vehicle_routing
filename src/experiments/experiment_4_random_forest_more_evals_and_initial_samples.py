from sklearn.ensemble import RandomForestRegressor
import numpy as np
import random
from src.fit import main

random.seed(1)
np.random.seed(1)


from src.sklearn_regressors.random_forest import random_forest_regressor_default

def run():
    for n_locations in range(20, 30, 1):
        main(
            n_train_instances=10,
            n_test_instances=10,
            n_vehicles=3,
            n_locations=n_locations,
            exp_name="n_locations_random_forest_regressor_eval_20_initial_samples_300",
            regr=random_forest_regressor_default,
            initial_samples=300,
            evals=20
        )


if __name__ == "__main__":
    run()
