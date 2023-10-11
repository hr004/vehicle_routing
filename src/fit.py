from sklearn.ensemble import RandomForestRegressor
import numpy as np
import random
from src.data.params import param_grid
from src.data.solve import get_time_dataset
import pickle

# not safe, we have to save all train model, variables and coordinates
# data = pickle.load(open("initial_data.pkl", "rb"))



from src.data.solve import get_data

from src.data.generate import get_train_test_data


def find_best_configuration(data, regr, evals, exploration_probability, train_dataset_models, train_dataset_variables, train_dataset_coordinates, test_dataset_models, test_dataset_variables, test_dataset_coordinates):
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
    print("Best runtime ({}) for parameters {}.".format(data[i][-1], data[i][0:-1]))

    # test that best configuration on the test dataset of VRP instances
    rt_test = get_time_dataset(
        test_dataset_models, param_grid[i], test_dataset_variables, test_dataset_coordinates
    )
    print("Test performance: {}".format((rt_test)))


def main():
    evals = 10

    regr = RandomForestRegressor(random_state=1)

    exploration_probability = 0.3

    (
    train_dataset_coordinates,
    train_dataset_models,
    train_dataset_variables,
    test_dataset_coordinates,
    test_dataset_models,
    test_dataset_variables,
    ) = get_train_test_data(n_train_instances=3)

    data = get_data(
        train_dataset_models=train_dataset_models,
        train_dataset_variables=train_dataset_variables, 
        train_dataset_coordinates=train_dataset_coordinates)

    find_best_configuration(
        data = data,
        regr=regr,
        evals=evals,
        exploration_probability=exploration_probability,
        train_dataset_models=train_dataset_models,
        train_dataset_variables=train_dataset_variables,
        train_dataset_coordinates=train_dataset_coordinates,
        test_dataset_models=test_dataset_models,
        test_dataset_variables=test_dataset_variables,
        test_dataset_coordinates=test_dataset_coordinates
    )

if __name__ =="__main__":
    main()