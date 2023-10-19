from sklearn.ensemble import GradientBoostingRegressor

# https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
params = {
    "n_estimators": 500,
    "max_depth": 4,
    "min_samples_split": 5,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

gradient_boosting_regressor = GradientBoostingRegressor(**params)
