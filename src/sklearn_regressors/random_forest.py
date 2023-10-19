from sklearn.ensemble import RandomForestRegressor

random_forest_regressor_default = RandomForestRegressor()
random_forest_regressor = RandomForestRegressor(
    n_estimators=200, criterion="friedman_mse", min_samples_leaf=5
)
