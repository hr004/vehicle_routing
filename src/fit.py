from sklearn.ensemble import RandomForestRegressor
import numpy as np
import random
from src.data.params import param_grid, params
from src.data.solve import get_time_dataset
import pandas as pd
import os

random.seed(1)
np.random.seed(1)
import pickle
import pathlib

from src.data.solve import get_data  # noqa: E402

from src.data.generate import get_train_test_data  # noqa: E402


def find_best_configuration(
    data,
    regr,
    evals,
    exploration_probability,
    train_dataset_models,
    train_dataset_variables,
    train_dataset_coordinates,
    test_dataset_models,
    test_dataset_variables,
    test_dataset_coordinates,
):
    best_config = -1
    for eval in range(0, evals):
        # first, we fit the regression model to the current dataset;
        # the features are the parameter configurations df[:,0:-1] and the label or target
        # is the runtime df[:,-1]
        df = np.array(data)
        regr.fit(df[:, 0:-1], df[:, -1])

        # perform prediction on all parameter configurations in our parameter grid
        preds = regr.predict([list(pars.values()) for pars in param_grid])

        # do not explore; exploit, i.e., retrieve the best parameter configuration
        if random.random() <= 1 - exploration_probability:
            i = preds.argmin()
            cur_config = list(param_grid[i].values())
            # this if condition tracks the incumbent (i.e., best so far) configuration
            if eval == 0 or cur_config != best_config:
                best_config = cur_config
                print("New Incumbent Configuration!", cur_config)
        # explore a random configuration
        else:
            print("Exploring!")
            i = random.randint(0, len(param_grid) - 1)

        # retrieve the runtime of the selected configuration on the training dataset
        rt = get_time_dataset(
            train_dataset_models,
            param_grid[i],
            train_dataset_variables,
            train_dataset_coordinates,
        )

        # append the newly collected runtime data as a row to the regression dataset
        data += [list(param_grid[i].values()) + [rt]]
        print(
            "{}: best predicted {} for {}, actual {}.".format(
                eval, round(preds[i], 2), param_grid[i], round(rt, 2)
            )
        )

    # the search has completed, so retrieve the configuration that has recorded
    # the lowest runtime in the dataset we have collected
    i = np.array(data)[:, -1].argmin()
    # print("Best runtime ({}) for parameters {}.".format(data[i][-1], data[i][0:-1]))

    # test that best configuration on the test dataset of VRP instances
    best_parameters = {}
    for key, val in zip(list(params.keys()), data[i][0:-1]):
        best_parameters[key] = val
    rt_test = get_time_dataset(
        test_dataset_models,
        best_parameters,
        test_dataset_variables,
        test_dataset_coordinates,
    )
    # print("Test performance: {}".format((rt_test)))

    # best runtime, best parameters, test performance
    return data[i][-1], data[i][0:-1], rt_test


def main(
    n_train_instances,
    n_test_instances,
    n_locations=None,
    n_vehicles=None,
    exp_name=None,
    evals=10,
    initial_samples=50,
    regr= None
):
    print(f"Starting training")

    if regr is None:
        regr = RandomForestRegressor(random_state=1)

    exploration_probability = 0.3

    (
        train_dataset_models,
        train_dataset_variables,
        train_dataset_coordinates,
        test_dataset_models,
        test_dataset_variables,
        test_dataset_coordinates,
    ) = get_train_test_data(
        n_train_instances=n_train_instances,
        n_test_instances=n_test_instances,
        n_locations=n_locations,
        n_vehicles=n_vehicles,
    )

    to_save = {
        "train_dataset_models": train_dataset_models,
        "train_dataset_variables": train_dataset_variables,
        "train_dataset_coordinates": train_dataset_coordinates,
        "test_dataset_models": test_dataset_models,
        "test_dataset_variables": test_dataset_variables,
        "test_dataset_coordinates": test_dataset_coordinates,
    }

    data = get_data(
        train_dataset_models=train_dataset_models,
        train_dataset_variables=train_dataset_variables,
        train_dataset_coordinates=train_dataset_coordinates,
        initial_samples=initial_samples,
    )

    average_training_runtime = np.array(data)[:, -1].mean()
    best_runtime, best_params, test_runtime = find_best_configuration(
        data=data,
        regr=regr,
        evals=evals,
        exploration_probability=exploration_probability,
        train_dataset_models=train_dataset_models,
        train_dataset_variables=train_dataset_variables,
        train_dataset_coordinates=train_dataset_coordinates,
        test_dataset_models=test_dataset_models,
        test_dataset_variables=test_dataset_variables,
        test_dataset_coordinates=test_dataset_coordinates,
    )

    # the next line runs the solver with its default parameters and returns the average
    # runtime on the test set; compare this value to rt_test above to see if the
    # configuration you have found is better than the default solver setting
    rt_test_notuning = get_time_dataset(
        test_dataset_models, None, test_dataset_variables, test_dataset_coordinates
    )

    print(rt_test_notuning)

    performance = {
        "n_train_instances": n_train_instances,
        "n_test_instances": n_test_instances,
        "n_vehicles": n_vehicles,
        "n_locations": n_locations,
        "average_training_runtime": average_training_runtime,
        "best_train_runtime": best_runtime,
        "predicted_runtime": test_runtime,
        "solver_default_runtime": rt_test_notuning,
    }

    for param, value in zip(list(params.keys()), best_params):
        performance[f"param_{param}"] = value

    performance_df = pd.DataFrame.from_dict([performance])

    filedir = f"./artifacts/{exp_name}"
    pathlib.Path(filedir).mkdir(exist_ok=True, parents=True)
    filename = f"{filedir}/runs_average.csv"
    performance_df.to_csv(
        filename, mode="a", index=False, header=not os.path.exists(filename)
    )
    pickle.dump(
        to_save,
        open(
            f"{filedir}/data_n_train_instances_{n_train_instances}_n_test_instances_{n_test_instances}_n_vehicles_{n_vehicles}_n_locations_{n_locations}.pkl",
            "wb",
        ),
    )
