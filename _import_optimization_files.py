import pandas as pd
# from _file_import import *
import pickle
from utils import *


#
# fleet_df = read_fleets(pd.read_csv("data/20220427-180251_fleets_input.csv"))
# fleet_df = fleet_df[fleet_df.index.isin(list(range(0, 40)))]
# fleet_df["route"] = [(1, 2, 3, 4, 5)] * len(fleet_df)
fleet_filename = "summer_workdayfleet_input_20220719_compressed_probe"
fleet_df = read_fleets(pd.read_csv("data/" + fleet_filename +  ".csv", delimiter=";"))

# og_fleet = fleet_df.copy()
# fleet_df["len"] = fleet_df.route.apply(len)
# fleet_df = fleet_df[fleet_df.len > 9]
fleet_df["start_timestep"] = [int(el) for el in fleet_df.start_timestep]
# fleet_df["SOC_init"] = [1] * len(fleet_df)

# fleet_df = fleet_df[fleet_df["start_timestep"].isin(list(range(0, 4)))]
# f = fleet_df.fleet_size.sum()
# fleet_df["start_timestep"] = [0 for el in fleet_df.start_timestep]
# random_idx = np.random.choice(len(fleet_df), size=20)
# fleet_df["fleet_id"] = range(0, len(fleet_df))
# # fleet_df["batt_cap"] = [88.16] * len(fleet_df)
# # fleet_df["charge_cap"] = [110] * len(fleet_df)
# fleet_df = fleet_df.set_index("fleet_id")
#
fleet_df["fleet_id"] = range(0, len(fleet_df))
# fleet_df = fleet_df[fleet_df.index.isin(random_idx)]
# fleet_df["fleet_id"] = range(0, len(fleet_df))
# fleet_df["batt_cap"] = [88.16] * len(fleet_df)
# fleet_df["charge_cap"] = [110] * len(fleet_df)
# fleet_df["d_spec"] = [0.25] * len(fleet_df)

# fleet_df.loc[17, "SOC_init"] = 1 * len(fleet_df)
# fleet_df = fleet_df[fleet_df.index.isin(range(0, 50))]


fleet_df["fleet_id"] = range(0, len(fleet_df))
fleet_df = fleet_df.set_index("fleet_id")
print(fleet_df)

# cells = pd.read_csv("data/20220429-152856_cells_input.csv")
cells = pd.read_csv("data/20220719-201710_cells_input.csv")
# print(cells)
# #
# # rels_cells = []
# # for r in fleet_df.route.to_list():
# #     for el in r:
# #         rels_cells.append(el)
# #
# # max_cell_id = max(rels_cells)
# # cells = cells[cells.cell_id.isin(list(range(0, max_cell_id + 1)))]
# # # cells = cells[cells.index.isin(list(range(0, 100)))]
# new_caps = []
# for c in cells.capacity.to_list():
#     if c == 0:
#         new_caps.append(0)
#     # elif c * 0.04 < 50:
#     #     new_caps.append(50)
#     else:
#         new_caps.append(c * 0.04)
# cells["capacity"] = new_caps
time_resolution = 0.25

nb_time_steps = 120
time_frame = range(0, nb_time_steps + 1)

nb_cells = len(cells)

nb_fleets = len(fleet_df)
print(nb_time_steps, nb_cells, nb_fleets)
SOC_min = 0.1
SOC_max = 1

t_min = 0.2