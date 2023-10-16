from sklearn.ensemble import RandomForestRegressor
import numpy as np
import random
from src.fit import main
random.seed(1)
np.random.seed(1)

import pathlib

def run():

    for n_locations in range(10,51, 1):
        filedir = "./artifacts/n_locations"
        pathlib.Path(filedir).mkdir(exist_ok=True, parents=True)
        main(
            n_train_instances=20,
            n_test_instances=10,
            n_vehicles=3,
            n_locations=n_locations, 
            exp_name="n_locations_improve_start_gap"
        )

if __name__ =="__main__":
    run()