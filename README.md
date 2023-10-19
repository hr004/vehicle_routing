
# Documentation

All the experiment scripts are in `src/experiments` directory.

# To run different experiments

Experiments starts with experiment_1 corresponds to task 1 related experiments, experiment_2 corresponds to task 2 and so on.

To run the experiments:

- First install all dependencies using poetry
- To install poetry `pip install poetry`
- To install dependencies `poetry install`
- To activate venv `poetry shell` and finally run the experimes

```bash
    export PYTHONPATH=${PYTHOHNPATH}:.
    python src/experiments/experiment_1_n_train_instances.py
```

Results will be stored in `artifacts` folder under the experiment given experiment name folder.