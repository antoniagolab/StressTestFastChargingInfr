"""

In this script, for each time step + OD-pair, a vehicle + initial SOC + route is determined

TODO:
    (1) match charging infrastructure to cells  --check
    (2) get of each O-D pair the respective route and choose a set of possible vehicles that can travel this route
    (3) draw for each O-D pair the starting time according to the distributions reported in Hiesl

"""
import pandas as pd

# set here the variables for which case study this is
winter = True
work_day = True

from utils import *
import time


velocity = 110
delta_t = 0.25
cell_length = velocity * delta_t * 1000

pois_df = pd.read_csv("data/pois_new.csv")
cellularized = pd.read_csv("data/cellularized.csv")
ts_ids = cellularized.index.to_list()
cell_nbs = cellularized.cell_nb.to_list()
cells = []
cell_counter = 0
for ij in range(0, len(ts_ids)):
    cells.append(list(range(cell_counter, cell_counter+int(cell_nbs[ij]))))
    cell_counter = cell_counter +int(cell_nbs[ij])

cellularized["cells"] = cells
cellularized.to_csv("data/cellularized_with_cells.csv", index=False)
# getting charging infrastructure as and input and matching this with the cells
pole_peak_capacity = 350  # kW
charging_capacities = pd.read_csv(
    "infrastructure/20220722-231754_input_HC_simulation_optimization_result_charging_stations.csv"
)
charging_capacities["pYi_dir"] = np.where(charging_capacities["pYi_dir"] >= 1, charging_capacities["pYi_dir"], 0)

cell_input_df = pd.DataFrame()
# TODO: implement direction in charging station, for now: this is ignored, we assume that all charging stations can be
#   accessed from both driving directions


charging_capacities_gt_0 = charging_capacities[charging_capacities.pXi > 0]

# create cell row by row, check if capacity, via pois which lie in the same Teilabschnitt

# first cell df without capacity --check

for ij in range(0, cell_counter):
    seg_id = get_segment_id_of_cell(cellularized, ij)

    row = {"seg_id": seg_id, "cell_id": ij, "length": cell_length / 1000}
    cell_input_df = cell_input_df.append(row, ignore_index=True)
print(cell_counter)
num_Y = charging_capacities_gt_0.pYi_dir.to_list()
poi_ids = charging_capacities_gt_0.POI_ID.to_list()
seg_id = charging_capacities_gt_0.segment_id.to_list()
cell_input_df["capacity"] = [0] * len(cell_input_df)
for ij in range(0, len(charging_capacities_gt_0)):
    cap = num_Y[ij] * pole_peak_capacity
    poi_id = poi_ids[ij]
    extract_pois = pois_df[pois_df.index == poi_id]
    dist_along_segment = extract_pois.dist_along_segment.to_list()[0]
    cell_idx = identify_position_poi_between_link_ra(
        seg_id[ij], dist_along_segment, cellularized
    )
    teil_seg = cellularized.loc[cell_idx]
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
    cell_input_df.loc[ind_cell, "capacity"] = cell_input_df.loc[ind_cell, "capacity"] + cap

cell_input_df["capacity"] = cell_input_df["capacity"].fillna(0.0)
cell_input_df["has_cs"] = np.where(cell_input_df["capacity"] > 0, True, False)
time_stamp = time.strftime("%Y%m%d-%H%M%S")
cell_input_df.to_csv("data/" + str(time_stamp) + "_cells_input.csv", index=False)
cell_input_df["cell_id"] = [int(el) for el in cell_input_df.cell_id.to_list()]


# get vehicle types
# get POI route of each of the OD-pairing
with open("data\paths_encoded_matching_route.pickle", "rb") as handle:
    paths_encoded_matching_route = pickle.load(handle)

BEV_data = pd.read_csv("data/BEV_data.csv")

# translate Major-POI to cell routes

route_og = pd.DataFrame()
T = 24 * 4

for k in paths_encoded_matching_route.keys():

    val = paths_encoded_matching_route[k]
    origin = val[2][0]
    destination = val[2][1]
    poi_route = val[-2]
    cell_route = []
    for route_segment in poi_route:
        if route_segment[0] < route_segment[1]:
            extract_cell = cellularized[(cellularized.poi_ID_start == route_segment[0])
                                        & (cellularized.poid_ID_end == route_segment[1])]
        else:
            extract_cell = cellularized[(cellularized.poi_ID_start == route_segment[1])
                                        & (cellularized.poid_ID_end == route_segment[0])]

        cell_route = cell_route + extract_cell.cells.to_list()[0]
        # cell_route.append(extract_cell.index.to_list()[0])
    if len(cell_route) > 0:
        min_driving_range = get_min_driving_range(
            tuple(cell_route), cell_input_df
        )
    else:
        min_driving_range = 999
    print(min_driving_range)

    print(k, cell_route)
    for t in range(0, T):
        if min_driving_range == 999:
            continue
        else:
            charging_capacity, battery_capacity, energy_cons, energy_cons_summer, SOC_init, SOC_min = draw_car_sample(BEV_data, min_driving_range)
            if SOC_init > 1:
                SOC_init = 1
            row = {"tau": t, "origin": origin, "destination": destination, "cell_route": tuple(cell_route),"min_driving_range": min_driving_range,
                   "charging_capacity": charging_capacity, "battery_capacity": battery_capacity,
                   "energy_cons": energy_cons, "energy_cons_summer": energy_cons_summer, "SOC_init": SOC_init, "SOC_min": SOC_min}
        # charg_cap, battery_cap, energy_cons = draw_car_sample(
        #     BEV_data, min_driving_range, season
        # )
            route_og = route_og.append(
                row, ignore_index=True
            )

route_og.to_csv("data/route_og.csv")







