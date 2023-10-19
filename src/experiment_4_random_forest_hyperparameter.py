from sklearn.ensemble import RandomForestRegressor
import numpy as np
import random
from src.fit import main

random.seed(1)
np.random.seed(1)


from src.sklearn_regressors.random_forest import random_forest_regressor

def run():
    for n_locations in range(20, 30, 1):
        main(
            n_train_instances=10,
            n_test_instances=10,
            n_vehicles=3,
            n_locations=n_locations,
            exp_name="n_locations_random_forest_regressor_hyperparameter",
            regr=random_forest_regressor
        )


if __name__ == "__main__":
    run()
