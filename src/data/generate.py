from src.data.utils import latlon_distance_matrix
import numpy as np
from src.model import vehicle_routing_model
from src.data.utils import visualize_map_points
import random

random.seed(1)
np.random.seed(1)


def generate_locations(n_locations):
    # Let us generate some GPS coordinates around Galbraith Building randomly
    l_lat, l_lon = 43.6598852, -79.3989274
    offset_lat, offset_lon = 0.01, 0.025

    # randomly generate and add depot at beginning
    lats = l_lat + np.round(
        np.random.rand(n_locations - 1) * 2 * offset_lat - offset_lat, 6
    )
    lons = l_lon + np.round(
        np.random.rand(n_locations - 1) * 2 * offset_lon - offset_lon, 6
    )
    lats = np.insert(lats, 0, l_lat)
    lons = np.insert(lons, 0, l_lon)

    # uncomment to visualize on a map
    # visualize_map_points(lats, lons)
    distance_matrix = latlon_distance_matrix(lats, lons)

    return distance_matrix, (lats, lons)


# Generate a single VRP instance with random demands and sufficiently large
# vehicle capacities (to ensure the instance is not infeasible), i.e., that
# demand can be met with the provided number of vehicles.
def generate_instance(n_locations, n_vehicle):
    distance_matrix, coordinates = generate_locations(n_locations)
    demand = np.random.randint(low=1, high=10, size=n_locations)
    # depot has zero demand
    demand[0] = 0
    # sufficiently large capacity; "a//b" is the integer part of a divided by b
    capacity = int(np.sum(demand) // (n_vehicle - 1))
    # create the CPMPy optimization model
    model, x = vehicle_routing_model(
        distance_matrix=distance_matrix,
        n_vehicle=n_vehicle,
        capacity=capacity,
        demand=demand,
    )
    return model, x, coordinates


# Generate a number of instances
def generate_instance_dataset(n_instances, n_locations, n_vehicle):
    dataset_models, dataset_variables, dataset_coordinates = [], [], []
    for i in range(n_instances):
        model, variables, coordinates = generate_instance(n_locations, n_vehicle)
        dataset_models += [model]
        dataset_variables += [variables]
        dataset_coordinates += [coordinates]
    return dataset_models, dataset_variables, dataset_coordinates



def get_train_test_data(n_train_instances = 3, n_locations = 10, n_vehicles = 3):
    """
      test instances is half of train instances
    """
    n_test_instances = n_test_instances // 2

    # Generate a training and a testing dataset
    (
        train_dataset_models,
        train_dataset_variables,
        train_dataset_coordinates,
    ) = generate_instance_dataset(n_instances=n_train_instances, n_locations=n_locations, n_vehicle=n_vehicles)
    (
        test_dataset_models,
        test_dataset_variables,
        test_dataset_coordinates,
    ) = generate_instance_dataset(n_instances=n_test_instances, n_locations=n_locations, n_vehicle=n_vehicles)
