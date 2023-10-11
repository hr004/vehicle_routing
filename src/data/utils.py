from geopy import distance
import numpy as np
import plotly.express as px
import pandas as pd


# distances from lat/lon to meters (rounded to the meter, so integer, great!)
def latlon_distance_matrix(lats, lons):
    """
    Using geopy's distance module compute geodesic distance between coordinates
    """
    coords = list(zip(lats, lons))
    return np.round(
        [[distance.distance(p1, p2).m for p2 in coords] for p1 in coords]
    ).astype(int)


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


# vizualize stops on map
def visualize_map_points(lats, lons):
    df_pt = pd.DataFrame({"lat": lats, "lon": lons, "size": 3})

    fig = px.scatter_mapbox(df_pt, lat="lat", lon="lon", size="size")
    fig.update_layout(
        mapbox_style="stamen-terrain",
        mapbox_zoom=12.8,
        mapbox_center_lat=df_pt.iloc[0, 0],
        mapbox_center_lon=df_pt.iloc[0, 1],
    )
    fig.show()


# vizualize routes on map
def visualize_map_routes(routes):
    # routes = dataframe with columns=['route','seq','lat','lon']
    fig = px.line_mapbox(
        routes, lat="lat", lon="lon", color="route"
    )  # , zoom=12)#, height=300)

    fig.update_traces(line=dict(width=4))
    fig.update_layout(
        mapbox_style="stamen-terrain",
        mapbox_zoom=12.8,
        mapbox_center_lat=routes.iloc[0, 2],
        mapbox_center_lon=routes.iloc[0, 3],
    )

    fig.show()


# decode the solution matrix 'x', x[i,j] = 1 means that a vehicle goes from node i to node j
# to route sequences, in a dataframe
def decode_x(sol, lats, lons):
    data = []  # (route, seq, lat, lon)
    firsts = np.where(sol[0] == 1)[0]
    for r in range(len(firsts)):
        s = 0
        idx = 0
        data.append((r, s, lats[idx], lons[idx]))
        s += 1
        idx = firsts[r]
        while idx != 0:
            data.append((r, s, lats[idx], lons[idx]))
            # find where this idx is pointing to
            s += 1
            idx = np.where(sol[idx] == 1)[0][0]
        # actually add depot at end too
        data.append((r, s, lats[idx], lons[idx]))
    return pd.DataFrame(data, columns=["route", "seq", "lat", "lon"])
