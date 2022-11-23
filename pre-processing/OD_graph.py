"""

This script is used to generate 
    - the geometries of OD nodes in the network 
    - information on amount of commuters between each node pairing
    - information on travel distance between each node pairing
    - information on time durations for traveling between each node pairing

"""


# TODO:
#   - (1) geographical distribution of OD nodes needed, irrespective of highway network -- check
#   - (2) match this with commuting data    -- check
#   - (3) needs to be matched to highway network
#   - (4) evaluation of routes for each OD-pair
#   - (5) think about spatial simplification of OD node representation
#   - (6) ... defining only a significant set of "centers" to maybe NUTS 3 (=35 regions), and center this to the
#       municipality with highest population density    -- check
#   - (7) transit?? --- is not regarded for now
#   - (8) for now, we will only look at commuting   -- check


# imports
import pickle
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from token import *
from shapely import geometry, ops, wkt
import openrouteservice
from _geometry_utils import *
import time
import json

path = "C:/Users/golab/PycharmProjects/HighwayChargingSimulation/"

token = "5b3ce3597851110001cf6248149cc75b3e754e428762c0d3a7802ef4"
# reading geographic data from
# https://www.statistik.at/web_de/klassifikationen/regionale_gliederungen/nuts_einheiten/index.html
nuts = pd.read_excel(
    path
    + "geography/nuts_3_mit_gemeinden_flaechen_und_bevoelkerung_gebietsstand_1.1.2021.xlsx",
    header=2,
    skipfooter=2,
)
nuts["population_dens"] = (
    nuts["Bevölke- rungszahl  1.1.2021"] / nuts["Fläche in ha 1.1.2021"]
)

nuts_geoms = gpd.read_file(path + "geography/NUTS_RG_03M_2021_3857.shp")
munic_geoms = gpd.read_file(path + "geography/STATISTIK_AUSTRIA_GEM_20220101.shp")

nuts_geoms_at = nuts_geoms[nuts_geoms["CNTR_CODE"] == "AT"]
nuts_geoms_at_3 = nuts_geoms_at[nuts_geoms_at["LEVL_CODE"] == 3]

# Hiesl et al. described distribution of starting times for different purposes;

# commuting data
cd_2019_at = pd.read_csv(
    "commuters_data/commuters_data_austria_2019_nuts3.csv", encoding="latin1"
)
cd_2019_at = cd_2019_at[
    ~cd_2019_at["Arbeits- bzw. Schulort - NUTS Gliederung (Ebene +1)"].isin(
        ["Pendler ins Ausland", "Kein Pendler"]
    )
]
cd_2019_at["code_live"] = cd_2019_at.apply(
    lambda row: row["Wohnort - NUTS Gliederung (Ebene +1)"]
    .split("<")[-1]
    .split(">")[0],
    axis=1,
)
cd_2019_at["code_work"] = cd_2019_at.apply(
    lambda row: row["Arbeits- bzw. Schulort - NUTS Gliederung (Ebene +1)"]
    .split("<")[-1]
    .split(">")[0],
    axis=1,
)

nuts_geoms_at_3 = nuts_geoms_at_3.to_crs("EPSG:31287")


# create numpy matrix
nuts_unique = list(set(nuts["NUTS 3-Code"].to_list()))
codes = dict(zip(list(range(0, len(nuts_unique))), nuts_unique))
codes_sw = dict(zip(nuts_unique, list(range(0, len(nuts_unique)))))
n = len(codes)

# determine centroid of a nuts-3 region after most populated municipality
for ij in range(0, n):
    k0 = codes[ij]
    extract_nuts = nuts[nuts["NUTS 3-Code"] == k0]
    ind_max = extract_nuts.population_dens.idxmax()

# now create matrix expressing amount of vehicles travelling
commuters = np.zeros([n, n])
commuters_dict = {}
for ij in range(0, n):
    for kl in range(0, n):
        if not ij == kl:
            k0 = nuts_unique[ij]
            k1 = nuts_unique[kl]
            extract = cd_2019_at[
                ((cd_2019_at.code_live == k0) & (cd_2019_at.code_work == k1))
            ]
            nb = extract["Anzahl"].to_list()[0]
            if not nb == "-":
                commuters[ij, kl] = nb
                commuters_dict[(k0, k1)] = nb
            else:
                print("oh oh", k0, k1)


# create a Monte-Carlo simulation here
# TODO:
#   - 2 directions for commuters, i.e, 2 distributions needed for expressing departure time in the morning and departure
#   time coming home at night -- check
#   - for each of the departing vehicle: draw a random starting time -- check
#   - ---> save this to a container with a counter for each time slot -- check
#   -
#   TODO now: (1) for each origin, (2) and then each destination, (3) generate way_1 distribution and then way_2 -check

# according to Hiesl et al., 2022
_way1_distribution = (7.5, 1)
_way2_distribution = (16, 2.88)


# all possible errors need to be caught here -- especially
# make conversion from hour to actual time

# time is expressed in 0.25 steps, but in numbers
# e.g. container "0.15"
d = dict(zip(list(np.arange(0, 24, 0.25)), [0] * 96))

destination_dict = dict(zip(nuts_unique, [d.copy() for ij in range(0, n)]))
departure_dict = dict(zip(nuts_unique, [destination_dict.copy() for ij in range(0, n)]))

for ij in range(0, n):
    for kl in range(0, n):
        k0 = codes[ij]
        k1 = codes[kl]
        nb_commuters = commuters[ij, kl]

        # way 1
        counter = 0
        while counter < nb_commuters:
            random_starting_time(
                k0, k1, _way1_distribution[0], _way1_distribution[1], departure_dict
            )
            counter = counter + 1

        # way 2
        counter = 0
        while counter < nb_commuters:
            random_starting_time(
                k1, k0, _way2_distribution[0], _way2_distribution[1], departure_dict
            )
            counter = counter + 1

with open("data\commuters_dict.pickle", "wb") as handle:
    pickle.dump(commuters_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


#   - matching to highway network (= projecting centroid)   -- check
#   - determining shortest route --- https://openrouteservice.org/services/

# match to highway network
# get network shape
# produce centroids representing NUTS-3 and project it to shape
# highway_geometry = gpd.read_file(path + 'geography/one_network.shp')
highway_geometry = pd2gpd(pd.read_csv(path + "geography/highway_segments.csv"))
geom = ops.linemerge(highway_geometry.geometry.to_list())

# determine here centroid after most densly populated area

nuts_geoms_at_3["centroid"] = nuts_geoms_at_3.geometry.centroid
od_nodes = []

for ij in range(0, n):
    extract = nuts_geoms_at_3[nuts_geoms_at_3["NUTS_ID"] == codes[ij]]
    extract_nuts = nuts[nuts["NUTS 3-Code"] == codes[ij]]
    ind_max = extract_nuts.population_dens.idxmax()
    munic_code = extract_nuts.loc[ind_max, "LAU 2 - Code Gemeinde- kennziffer"]
    extract_2 = munic_geoms[munic_geoms.id == str(munic_code)]
    if not len(extract_2) == 0:
        c = extract_2.geometry.to_list()[0].centroid
    else:
        c = extract.geometry.to_list()[0].centroid
    l = {
        "NUTS_NAME": extract["NUTS_NAME"].to_list()[0],
        "NUTS_ID": codes[ij],
        "geometry": geom.interpolate(round(geom.project(c), 4)),
        "centroid": c,
    }
    od_nodes.append(l)

od_nodes_df = gpd.GeoDataFrame(od_nodes, geometry="geometry")
od_nodes_df = od_nodes_df.set_crs("EPSG:31287")

on_segment_dict = finding_segments_point_lies_on(
    od_nodes_df["geometry"], highway_geometry
)
line_ids = list(on_segment_dict.values())
tc_on_line_ids = [p[0] for p in line_ids]
od_nodes_df["on_segment"] = tc_on_line_ids

# cutting of segments 1-3
filtered_od_nodes = od_nodes_df[~od_nodes_df.on_segment.isin([0, 1, 2])]

# TODO: insert Nan values for not significant routes
#   - test each matching, insert np Nan if this route is not to be used further

od_centroid_4326 = gpd.GeoDataFrame(od_nodes_df, geometry="centroid", crs="EPSG:31287")
od_centroid_4326 = od_centroid_4326.drop(columns=["geometry"])
od_centroid_4326.to_file("data/OD_centroids.shp")
od_centroid_4326 = od_centroid_4326.to_crs("EPSG:4326")

od_centroid_4326.to_file("data/od_centroid_4326.shp")

# setting up openrouteservice
client = openrouteservice.Client(key=token)  # Specify your personal API key

distances = {}
durations = {}
for ij in range(0, n):
    k0 = codes[ij]
    conn = departure_dict[k0].keys()
    c0 = od_centroid_4326[od_centroid_4326["NUTS_ID"] == k0]["centroid"].to_list()[0]
    for k1 in conn:
        if not k0 == k1:
            c1 = od_centroid_4326[od_centroid_4326["NUTS_ID"] == k1][
                "centroid"
            ].to_list()[0]
            routes = client.directions(
                ((c0.x, c0.y), (c1.x, c1.y)), profile="driving-car"
            )
            distance = routes["routes"][0]["summary"]["distance"]
            distances[str((k0, k1))] = distance
            duration = routes["routes"][0]["summary"]["duration"]
            durations[str((k0, k1))] = duration
            print(distance, duration)
            if not (
                (distance > 50 * 1000)
                or ((distance > 25 * 1000) and (duration > 60 * 50))
            ):
                departure_dict[str(k0)][str(k1)] = np.NaN
        else:
            departure_dict[str(k0)][str(k1)] = np.NaN
    print(ij, "sleeping")
    time.sleep(60)


with open("data\departures.pickle", 'wb') as handle:
    pickle.dump(departure_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("data\distances.pickle", 'wb') as handle:
    pickle.dump(distances, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("data\durations.pickle", 'wb') as handle:
    pickle.dump(durations, handle, protocol=pickle.HIGHEST_PROTOCOL)


print(routes)

