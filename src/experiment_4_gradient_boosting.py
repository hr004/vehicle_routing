import numpy as np
import random
from src.fit import main

random.seed(1)
np.random.seed(1)


from src.sklearn_regressors.gradient_boosting_regressor import gradient_boosting_regressor

def run():
    for n_locations in range(20, 26, 1):
        main(
            n_train_instances=10,
            n_test_instances=10,
            n_vehicles=3,
            n_locations=n_locations,
            exp_name="n_locations_gradient_boosting_regressor",
            regr=gradient_boosting_regressor
        )


if __name__ == "__main__":
    run()
