import datetime

import numpy as np
import pandas as pd
import pickle
import geopandas as gpd
from utils import *
import time
import _start_time_distributions


pois_df = pd.read_csv("data/pois_new.csv")
frequency_table = pd.read_csv("data/BEV_data.csv")
pois_df = reorder_this(pois_df)
cells = pd.DataFrame()
segment_ids = list(set(pois_df.segment_id.to_list()))
for s in [0, 1, 2]:
    segment_ids.remove(s)
assumed_velocity = 110  # (km/h)
model_resolution = 0.25  # (h)

cell_counter = 0
cell_length = assumed_velocity * model_resolution * 1000
for s in segment_ids:
    poi_extract = pois_df[pois_df.segment_id == s]
    poi_extract_cleaned = poi_extract[~poi_extract.pois_type.isin(["ra"])]
    inds = poi_extract_cleaned.index.to_list()
    poi_types = poi_extract_cleaned.pois_type.to_list()
    type_ID = poi_extract_cleaned.type_ID.to_list()
    dist_along_segments = poi_extract_cleaned.dist_along_segment.to_list()
    for ij in range(0, len(inds)-1):
        dist = np.abs(dist_along_segments[ij] - dist_along_segments[ij + 1])
        # determine efficient cell number
        min_number = dist // cell_length
        max_number = min_number + 1
        potential_nbs = np.array([min_number, max_number])
        ind_min_error = np.argmin(np.abs(dist - potential_nbs * cell_length))
        row = {
            "segment_id": s,
            "poi_ID_start": inds[ij],
            "poi_type_start": poi_types[ij],
            "type_ID_start": type_ID[ij],
            "poid_ID_end": inds[ij + 1],
            "poi_type_end": poi_types[ij + 1],
            "type_ID_end": type_ID[ij + 1],
            "start_end": (inds[ij], inds[ij + 1]),
            "dist_start": dist_along_segments[ij],
            "dist_end": dist_along_segments[ij + 1],
            "nb_cells": potential_nbs[ind_min_error],
            "cells": list(
                range(cell_counter, cell_counter + int(potential_nbs[ind_min_error]))
            ),
        }
        cell_counter = cell_counter + int(potential_nbs[ind_min_error])
        cells = cells.append(row, ignore_index=True)

cells.to_csv("data/cell_routes.csv")
# nb cars
with open("data\OD_purposes.pickle", "rb") as handle:
    OD_purposes = pickle.load(handle)

# get paths between these POI_encoded
with open("data\paths_encoded_pois_lim.pickle", "rb") as handle:
    paths_encoded_pois_lim = pickle.load(handle)

# get OD nodes, for matching encoded OD node with the NUTS_ID
projected_nodes = gpd.read_file("data/all_projected_nodes.shp")


# TODO:
#   - for each OD pair in OD_purposes: goal = express path in cells
#   - ...
#   - (1) get OD, (2) identify encoding + POI_ID, (3) get belonging path, (4) sum up path cells like lists
#   -

od_routes = pd.DataFrame()

for od in OD_purposes.keys():
    o = od[0]
    d = od[1]
    counter = 0
    if not (o == None or d == None) and not o == d:
        idx_o = projected_nodes[projected_nodes.NUTS_ID == o].nodeID.to_list()[0]
        idx_d = projected_nodes[projected_nodes.NUTS_ID == d].nodeID.to_list()[0]

        poi_id_o = pois_df[
            (pois_df.pois_type == "od") & (pois_df.type_ID == idx_o)
        ].index.to_list()[0]
        poi_id_d = pois_df[
            (pois_df.pois_type == "od") & (pois_df.type_ID == idx_d)
        ].index.to_list()[0]

        if (poi_id_o, poi_id_d) in paths_encoded_pois_lim:
            path = paths_encoded_pois_lim[(poi_id_o, poi_id_d)]
            l = []
            for path_step in path:
                if path_step[0] < path_step[1]:
                    segm = cells[cells.start_end == path_step].cells.to_list()[0]
                else:
                    segm = cells[
                        cells.start_end == (path_step[1], path_step[0])
                    ].cells.to_list()[0]
                    segm.reverse()
                l = l + segm

            row = {"origin": o, "destination": d, "route": l}
            od_routes = od_routes.append(row, ignore_index=True)
od_routes.to_csv("data/od_routes.csv")
ev_share = 0.27
departure_dict = create_start_time_matrix(
    OD_purposes,
    _start_time_distributions.route_one,
    _start_time_distributions.route_two,
    ev_share * 0.5,
)


# TODO:
# create for each timestep and OD -> fleet, if fleet > 0 !
# draw vehicle and SOC init

# now get capacities to create cell input
pole_peak_capacity = 350  # kW
charging_capacities = pd.read_csv(
    "infrastructure/20220427-150049_Gradual Development_optimization_result_charging_stations.csv"
)

cell_input_df = pd.DataFrame()
# TODO: implement direction in charging station, for now: this is ignored, we assume that all charging stations can be
#   accessed from both driving directions


charging_capacities_gt_0 = charging_capacities[charging_capacities.pXi > 0]

# create cell row by row, check if capacity, via pois which lie in the same Teilabschnitt

# first cell df without capacity --check

for ij in range(0, cell_counter):
    seg_id = get_segment_id_of_cell(cells, ij)

    row = {"seg_id": seg_id, "cell_id": ij, "length": cell_length / 1000}
    cell_input_df = cell_input_df.append(row, ignore_index=True)

num_Y = charging_capacities_gt_0.pYi_dir.to_list()
poi_ids = charging_capacities_gt_0.POI_ID.to_list()
seg_id = charging_capacities_gt_0.segment_id.to_list()
for ij in range(0, len(charging_capacities_gt_0)):
    cap = num_Y[ij] * pole_peak_capacity
    poi_id = poi_ids[ij]
    extract_pois = pois_df[pois_df.index == poi_id]
    dist_along_segment = extract_pois.dist_along_segment.to_list()[0]
    cell_idx = identify_position_poi_between_link_ra(
        seg_id[ij], dist_along_segment, cells
    )
    teil_seg = cells.loc[cell_idx]
    nb_cells = len(teil_seg.cells)
    diff = dist_along_segment - teil_seg.dist_start
    true_len_cell = (teil_seg.dist_end - teil_seg.dist_start) / nb_cells
    curr_dist = 0
    id = None
    for kl in range(0, nb_cells):
        next_dist = curr_dist + true_len_cell
        if diff > curr_dist and diff <= next_dist:
            id = kl
            break
        curr_dist = next_dist
    cell_id = teil_seg.cells[id]

    ind_cell = cell_input_df[cell_input_df.cell_id == cell_id].index.to_list()[0]
    cell_input_df.loc[ind_cell, "capacity"] = cap

cell_input_df["capacity"] = cell_input_df["capacity"].fillna(0.0)
cell_input_df["has_cs"] = np.where(cell_input_df["capacity"] > 0, True, False)
time_stamp = time.strftime("%Y%m%d-%H%M%S")
cell_input_df.to_csv("data/" + str(time_stamp) + "_cells_input.csv", index=False)
cell_input_df["cell_id"] = [int(el) for el in cell_input_df.cell_id.to_list()]
# encoding the time steps
timesteps = np.arange(0, 24)
delta_t = 0.25

for m in range(0, 20):
    print("m", m)
    input_file_fleets = pd.DataFrame()
    failed = 0
    not_failed = 0
    for n1 in departure_dict:
        for n2 in departure_dict[n1]:
            if not n1 == n2:
                fleet_info = departure_dict[n1][n2]
                if (
                    len(
                        od_routes[
                            (od_routes.origin == n1) & (od_routes.destination == n2)
                        ]
                    )
                    > 0
                ):
                    route = od_routes[
                        (od_routes.origin == n1) & (od_routes.destination == n2)
                    ].route.to_list()[0]
                    for t in timesteps:
                        n_f = fleet_info[t]
                        if n_f > 0:
                            min_driving_range = get_min_driving_range(
                                route, cell_input_df
                            )
                            if (
                                len(
                                    frequency_table[
                                        frequency_table["Max_driving_range"]
                                        > min_driving_range
                                    ]
                                )
                                > 0
                            ):
                                charg_cap, battery_cap, energy_cons = draw_car_sample(
                                    n_f, frequency_table, min_driving_range
                                )
                                SOC_min = get_min_SOC_init(
                                    route, cell_input_df, battery_cap, energy_cons
                                )
                                if SOC_min < 0.1:
                                    SOC_min = 0.8
                                print(min_driving_range, n1, n2, SOC_min, energy_cons)
                                SOC_init = draw_SOC_init(
                                    SOC_min=my_ceil(SOC_min, precision=1)
                                )
                                if SOC_init < 0.8:
                                    SOC_init = 1
                                row = {
                                    "start_timestep": int(t * 4),
                                    "route": tuple(route),
                                    "fleet_size": int(n_f),
                                    "charge_cap": charg_cap,
                                    "batt_cap": round(battery_cap, 2),
                                    "d_spec": round(energy_cons / 100, 2),
                                    "SOC_init": SOC_init,
                                }
                                input_file_fleets = input_file_fleets.append(
                                    row, ignore_index=True
                                )
                                not_failed = not_failed + 1
                            else:
                                failed = failed + 1

    input_file_fleets["fleet_id"] = range(0, len(input_file_fleets))
    input_file_fleets["start_timestep"] = [
        int(el) for el in input_file_fleets.start_timestep.to_list()
    ]
    input_file_fleets.to_csv("data/" + str(m) + "_fleets_input.csv", index=False)

    print("failed", failed)
    print("not failed", not_failed)
    # TODO: create lower bound of SOC_init based on what is possible on this route! -> what is the minimum initial SOC
    #   for the fleet?


# filter fleet input file after fleets which are not able to recharge!
