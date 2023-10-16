import numpy as np
import random
from src.fit import main
random.seed(1)
np.random.seed(1)

def run():
    for n_vehicles in range(4,6, 1):
        main(
            n_train_instances=20,
            n_test_instances=10,
            n_vehicles=n_vehicles,
            n_locations=20, 
            exp_name="n_vehicles",
            evals = 10
        )

if __name__ =="__main__":
    run()