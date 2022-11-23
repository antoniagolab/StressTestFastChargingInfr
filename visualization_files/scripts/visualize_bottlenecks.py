"""

Created on July 17th, 2022

Two figures are created here for section 4.2


    two similar figures = comparison of charging infrastructure utility
        (1) determine the charging station with highest utility rate and the one with lowest (excluding the ones == 0)
        (2) get _power for each
TODO:
        (3)


"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import geopandas as gpd
from shapely import wkt
import numpy as np

# enter here always the day with highest consumption
# winter_workday = pd.read_csv("results/20220720-015553_charging_stationswinter_workdayfleet_input_20220719_compressed_probe.csv")
winter_workday = pd.read_csv("results/20220723-233215_charging_stationswinter_workdayfleet_input_20220722_compressed_probe2.csv")
summer_workday = pd.read_csv("results/20220724-033029_charging_stationssummer_workdayfleet_input_20220722_compressed_probe2.csv")
winter_holiday = pd.read_csv("results/20220724-083509_charging_stationswinter_holidayfleet_input_20220722_compressed_probe2.csv")
summer_holiday = pd.read_csv("results/20220724-114442_charging_stationssummer_holidayfleet_input_20220722_compressed_probe2.csv")
#
cell_id = 2
colors = ["#118ab2", "#06d6a0", "#ffd166", "#ef476f"]

idx_cell = winter_workday[winter_workday.cell_id == cell_id].index.to_list()[0]

_mu = 250/350
T = 120

#
# print(len(_power))
# # TODO: add efficiency to it
#
# _time_steps = list(range(0, T))
#


# calculate load factor for each charging station and difference between peak load and installed capacity

indices = winter_workday.index.to_list()
for ij in range(0, len(indices)):
    id = indices[ij]
    cap = winter_workday.at[id, "capacity"]
    _energy_charged = [winter_workday.at[id, "E charged at t=" + str(t)] for t in range(0, T)]
    _total_energy_charged = sum(_energy_charged)
    _maximum_energy_charged = max(_energy_charged)
    winter_workday.at[id, "utility_rate"] = _total_energy_charged / (cap * (T / 4))
    winter_workday.at[id, "diff_to_max"] = (cap - _maximum_energy_charged / 0.25) / cap

    id = indices[ij]
    cap = summer_workday.at[id, "capacity"]
    _energy_charged = [summer_workday.at[id, "E charged at t=" + str(t)] for t in range(0, T)]
    _total_energy_charged = sum(_energy_charged)
    _maximum_energy_charged = max(_energy_charged)
    summer_workday.at[id, "utility_rate"] = _total_energy_charged / (cap * (T / 4))
    summer_workday.at[id, "diff_to_max"] = (cap - _maximum_energy_charged / 0.25) / cap

    id = indices[ij]
    cap = winter_workday.at[id, "capacity"]
    _energy_charged = [winter_holiday.at[id, "E charged at t=" + str(t)] for t in range(0, T)]
    _total_energy_charged = sum(_energy_charged)
    _maximum_energy_charged = max(_energy_charged)
    winter_holiday.at[id, "utility_rate"] = _total_energy_charged / (cap * (T / 4))
    winter_holiday.at[id, "diff_to_max"] = (cap - _maximum_energy_charged / 0.25) / cap

    id = indices[ij]
    cap = summer_holiday.at[id, "capacity"]
    _energy_charged = [summer_holiday.at[id, "E charged at t=" + str(t)] for t in range(0, T)]
    _total_energy_charged = sum(_energy_charged)
    _maximum_energy_charged = max(_energy_charged)
    summer_holiday.at[id, "utility_rate"] = _total_energy_charged / (cap * (T / 4))
    summer_holiday.at[id, "diff_to_max"] = (cap - _maximum_energy_charged / 0.25) / cap

winter_workday["diff_to_max"] = np.where(winter_workday["diff_to_max"] < 0, 0, winter_workday["diff_to_max"])
winter_workday["utility_rate"] = np.where(winter_workday["utility_rate"] < 0, 0, winter_workday["utility_rate"])

summer_workday["diff_to_max"] = np.where(summer_workday["diff_to_max"] < 0, 0, summer_workday["diff_to_max"])
summer_workday["utility_rate"] = np.where(summer_workday["utility_rate"] < 0, 0, summer_workday["utility_rate"])

winter_holiday["diff_to_max"] = np.where(winter_holiday["diff_to_max"] < 0, 0, winter_holiday["diff_to_max"])
winter_holiday["utility_rate"] = np.where(winter_holiday["utility_rate"] < 0, 0, winter_holiday["utility_rate"])

summer_holiday["diff_to_max"] = np.where(summer_holiday["diff_to_max"] < 0, 0, summer_holiday["diff_to_max"])
summer_holiday["utility_rate"] = np.where(summer_holiday["utility_rate"] < 0, 0, summer_holiday["utility_rate"])

cs_filtered_ww = winter_workday[winter_workday["utility_rate"] > 0]
print(len(winter_workday) - len(cs_filtered_ww), "charging stations show no activity")

# identify the one which maximum utilization
_cell_id_max = cs_filtered_ww[cs_filtered_ww.utility_rate == cs_filtered_ww.utility_rate.max()].index.to_list()[0]
_cell_id_min = cs_filtered_ww[cs_filtered_ww.utility_rate == cs_filtered_ww.utility_rate.min()].index.to_list()[0]
print(_cell_id_min, _cell_id_max)

_cell_id_min = 19
_cell_id_max = 23

# extract time series of total energy charged there
e_charged_min_ww = []
e_charged_max_ww = []

e_charged_min_sw = []
e_charged_max_sw = []

e_charged_min_wh = []
e_charged_max_wh = []

e_charged_min_sh = []
e_charged_max_sh = []

_time_steps = []
for t in range(0, T):
    # e_charged.append()
    _time_steps = _time_steps + [t , t]
    if t == 0:
        e_charged_min_ww = e_charged_min_ww + [0, t]
        e_charged_max_ww = e_charged_max_ww + [0, t]
        e_charged_min_sw = e_charged_min_sw + [0, t]
        e_charged_max_sw = e_charged_max_sw + [0, t]
        e_charged_min_wh = e_charged_min_wh + [0, t]
        e_charged_max_wh = e_charged_max_wh + [0, t]
        e_charged_min_sh = e_charged_min_sh + [0, t]
        e_charged_max_sh = e_charged_max_sh + [0, t]
    else:
        e_charged_min_ww = e_charged_min_ww + [e_charged_min_ww[-1], winter_workday.at[_cell_id_min, "E charged at t=" + str(t)]]
        e_charged_max_ww = e_charged_max_ww + [e_charged_max_ww[-1], winter_workday.at[_cell_id_max, "E charged at t=" + str(t)]]
        e_charged_min_sw = e_charged_min_sw + [e_charged_min_sw[-1], summer_workday.at[_cell_id_min, "E charged at t=" + str(t)]]
        e_charged_max_sw = e_charged_max_sw + [e_charged_max_sw[-1], summer_workday.at[_cell_id_max, "E charged at t=" + str(t)]]
        e_charged_min_wh = e_charged_min_wh + [e_charged_min_wh[-1], winter_holiday.at[_cell_id_min, "E charged at t=" + str(t)]]
        e_charged_max_wh = e_charged_max_wh + [e_charged_max_wh[-1], winter_holiday.at[_cell_id_max, "E charged at t=" + str(t)]]
        e_charged_min_sh = e_charged_min_sh + [e_charged_min_sh[-1], summer_holiday.at[_cell_id_min, "E charged at t=" + str(t)]]
        e_charged_max_sh = e_charged_max_sh + [e_charged_max_sh[-1], summer_holiday.at[_cell_id_max, "E charged at t=" + str(t)]]

print(winter_workday.at[_cell_id_min, "capacity"])
_power_min_ww = [el / 0.25 * (1 / _mu) for el in e_charged_min_ww]
_power_max_ww = [el / 0.25 * (1 / _mu) for el in e_charged_max_ww]
_power_min_sw = [el / 0.25 * (1 / _mu) for el in e_charged_min_sw]
_power_max_sw = [el / 0.25 * (1 / _mu) for el in e_charged_max_sw]
_power_min_wh = [el / 0.25 * (1 / _mu) for el in e_charged_min_wh]
_power_max_wh = [el / 0.25 * (1 / _mu) for el in e_charged_max_wh]
_power_min_sh = [el / 0.25 * (1 / _mu) for el in e_charged_min_sh]
_power_max_sh = [el / 0.25 * (1 / _mu) for el in e_charged_max_sh]
c_1 = winter_workday.at[19, "capacity"]
c_2 = winter_workday.at[23, "capacity"]
print(c_1, c_2)
print(c_1 - max(_power_min_ww), c_1 - max(_power_min_sw), c_1 - max(_power_min_wh), c_1 - max(_power_min_sh))
print(c_2 - max(_power_max_ww), c_2 - max(_power_max_sw), c_2 - max(_power_max_wh), c_2 - max(_power_max_sh))
print(winter_workday.at[19, "utility_rate"], summer_workday.at[19, "utility_rate"], winter_holiday.at[19, "utility_rate"],summer_holiday.at[19, "utility_rate"])
print(winter_workday.at[23, "utility_rate"], summer_workday.at[23, "utility_rate"], winter_holiday.at[23, "utility_rate"],summer_holiday.at[23, "utility_rate"])
print(c_1, c_2)
# ---------
# getting NUTS-3 regions
nuts_3 = gpd.read_file("NUTS_RG_03M_2021_3857/NUTS_RG_03M_2021_3857.shp")

# filter the regions after AT and NUTS-3
nuts_at = nuts_3[nuts_3["CNTR_CODE"] == "AT"]

nuts_3_at =nuts_at[nuts_at["LEVL_CODE"] == 3]   # alternatively insert value "2"

# exclude not relevant NUTS_3 regions
_not_relevant_region_codes = [342, 341, 331, 334, 332, 335]
# _not_relevant_region_codes = [33, 34]
_not_relevant_region_codes_with_AT = ["AT" + str(el) for el in _not_relevant_region_codes]

nuts_3_at_relevant_gpd = nuts_3_at[~nuts_3_at.NUTS_ID.isin(_not_relevant_region_codes_with_AT)]
nuts_3_at_relevant_gpd["geometry"] = nuts_3_at_relevant_gpd.geometry.exterior

to_visualize = nuts_3_at_relevant_gpd

# getting highway network
hs = pd.read_csv("geography/highway_segments.csv")
hs["geometry"] = hs["geometry"].apply(wkt.loads)
reference_coord_sys = "EPSG:31287"

hs = gpd.GeoDataFrame(hs, geometry="geometry", crs=reference_coord_sys)
_to_plot = pd.read_csv("data/allocation.csv")
cs_alloc = pd.read_csv("data/processed.csv")
_marker_size = 60

fig, axs = plt.subplots(2, 2, figsize=(10, 7.5),  gridspec_kw={'height_ratios': [1, 1]})

plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 12
gs = fig.add_gridspec(2, 2)

# plotting the line
axs[0][0].plot(_time_steps, _power_min_ww, color=colors[0], label="Workday in winter", linewidth=2, alpha=0.7)
axs[0][0].plot(_time_steps, _power_min_sw, color=colors[1], label="Workday in summer", linewidth=2, alpha=0.7)
axs[0][0].plot(_time_steps, _power_min_wh, color=colors[2], label="Holiday in winter", linewidth=2, alpha=0.7)
axs[0][0].plot(_time_steps, _power_min_sh, color=colors[3], label="Holiday in summer", linewidth=2, alpha=0.7)

axs[0][1].plot(_time_steps, _power_max_ww, color=colors[0], label="Workday in winter", linewidth=2, alpha=0.7)
axs[0][1].plot(_time_steps, _power_max_sw, color=colors[1], label="Workday in summer", linewidth=2, alpha=0.7)
axs[0][1].plot(_time_steps, _power_max_wh, color=colors[2], label="Holiday in winter", linewidth=2, alpha=0.7)
axs[0][1].plot(_time_steps, _power_max_sh, color=colors[3], label="Holiday in summer", linewidth=2, alpha=0.7)

# insert hline
axs[0][0].hlines(y=winter_workday.at[_cell_id_min, "capacity"], linewidth=3, linestyle="--", xmin=0, xmax=120, color="#f4acb7",
              label="Installed capacity", alpha=0.9)
axs[0][1].hlines(y=winter_workday.at[_cell_id_max, "capacity"], linewidth=3, linestyle="--", xmin=0, xmax=120, color="#f4acb7",
              alpha=0.9)
axs[0][0].legend(loc="upper left",bbox_to_anchor=(1.2, -0.2), ncol=1, fontsize=12)
axs[0][0].set_title("Charging station A", fontsize=12)
axs[0][1].set_title("Charging station B",fontsize=12)
# fig.suptitle("Charging activity", fontsize=20)
axs[1][1].axis("off")
axs[1][0].axis("off")

for kl in [0, 1]:
    axs[0][kl].grid()
    axs[0][kl].set_ylabel("Charging power (kW)", fontname="Franklin Gothic Book", fontsize=14)
    axs[0][kl].set_xlabel("Hour of the day", fontname="Franklin Gothic Book", fontsize=14)
    axs[0][kl].set_xticklabels(["00:00", "05:00", "10:00", "15:00", "20:00", "01:00"], fontsize=12)
    axs[0][kl].set_xlim([10, 110])
    axs[0][kl].tick_params(axis = 'both', which = 'major', labelsize = 10)

_merged = cs_alloc.copy()
_merged["x"] = _to_plot["x"]
_merged["y"] = _to_plot["y"]
_cat_neg = _merged[_merged["cs"] == False]
_cat_pos = _merged[_merged["cs"]]

to_visualize.plot(ax=axs[1][0], color="#ced4da", zorder=0, linewidth=0.7)
axs[1][0].scatter(
    _cat_neg["x"].to_list(),
    _cat_neg["y"].to_list(),
    s=_marker_size,
    color="white",
    # color="black",
    edgecolors='black',
    linewidth=0.4,
    zorder=10,
)
axs[1][0].scatter(
    _cat_pos["x"].to_list(),
    _cat_pos["y"].to_list(),
    s=_marker_size,
    color="black",
    # color="black",
    edgecolors='black',
    linewidth=0.4,
    zorder=10,
)
a_point_x = _to_plot.at[19, "x"]
a_point_y = _to_plot.at[19, "y"]
b_point_x = _to_plot.at[23, "x"]
b_point_y = _to_plot.at[23, "y"]

hs = hs.to_crs("EPSG:3857")
_relevant_highway_network = hs[~hs.ID.isin([0, 1, 2])]

_relevant_highway_network.plot(
    ax=axs[1][0], label="Austrian highway network", color="black", zorder=0, linewidth=1
)
axs[1][0].text(a_point_x+18000, a_point_y+18000, "Charging station A", va="bottom", backgroundcolor="white" , ha="left", zorder = 30, fontsize=12)
axs[1][0].text(b_point_x+18000, b_point_y+18000, "Charging station B", va="bottom", backgroundcolor="white" , ha="left", zorder = 30, fontsize=12)
#axs[1][0].xlim([6565.933443501076, 6575.454900000006])
plt.subplots_adjust(wspace=0.25)
# print("this is the xlim:", axs[0][0].xlim())
# axs[1][0].axis(xmin=)

plt.savefig("figures/charging_activity_.pdf", bbox_inches="tight")
plt.savefig("figures/charging_activity_.svg", bbox_inches="tight")
# axs[0][0].legend( fontsize=12)

# # figure 2: plot with halfcircles (https://gist.github.com/lukauskas/eec68a0d8b6e6b48fd90e5efb8400cda)
# # TODO:
# #   - establish background as usual --check
# #   - create half circles   --check
# #   - make four images with this
#
# reference_coord_sys = "EPSG:31287"
#
# infrastructure = pd.read_csv("infrastructure/20220716-081946_input_HC_simulation_optimization_result_charging_stations.csv")
# cells = pd.read_csv("data/20220717-171709_cells_input.csv")
# cr = pd.read_csv("data/cellularized_with_cells.csv")
# cr["cells"] = [eval(el) for el in cr["cells"].to_list()]
#
# # getting NUTS-3 regions
# nuts_3 = gpd.read_file("NUTS_RG_03M_2021_3857/NUTS_RG_03M_2021_3857.shp")
#
# # filter the regions after AT and NUTS-3
# nuts_at = nuts_3[nuts_3["CNTR_CODE"] == "AT"]
#
# nuts_3_at =nuts_at[nuts_at["LEVL_CODE"] == 3]   # alternatively insert value "2"
#
# # exclude not relevant NUTS_3 regions
# _not_relevant_region_codes = [342, 341, 331, 334, 332, 335]
# # _not_relevant_region_codes = [33, 34]
# _not_relevant_region_codes_with_AT = ["AT" + str(el) for el in _not_relevant_region_codes]
#
# nuts_3_at_relevant_gpd = nuts_3_at[~nuts_3_at.NUTS_ID.isin(_not_relevant_region_codes_with_AT)]
# nuts_3_at_relevant_gpd["geometry"] = nuts_3_at_relevant_gpd.geometry.exterior
#
# to_visualize = nuts_3_at_relevant_gpd
#
# # getting highway network
# hs = pd.read_csv("geography/highway_segments.csv")
# hs["geometry"] = hs["geometry"].apply(wkt.loads)
# hs = gpd.GeoDataFrame(hs, geometry="geometry", crs=reference_coord_sys)
#
#
# _cell_length = 27500    # (m)
# init_distance = _cell_length/2
#
# # identify positions for transform charging stations
# _cells_with_cs = cells[cells.has_cs]
# indices = _cells_with_cs.index.to_list()
# _cell_ids = _cells_with_cs.cell_id.to_list()
# _cr_cell_list = cr.cells.to_list()
# point_list = []
# _dist_starts = cr.dist_start.to_list()
# _dist_ends = cr.dist_end.to_list()
# for ij in range(0, len(indices)):
#     c_id = _cell_ids[ij]
#     seg_id = _cells_with_cs.at[indices[ij], "seg_id"]
#     capacity = _cells_with_cs.at[indices[ij], "capacity"]
#     match_id = None
#     distance = None
#
#     for kl in range(0, len(_cr_cell_list)):
#         if c_id in _cr_cell_list[kl]:
#             match_id = kl
#             init_distance = ((_dist_starts[kl] - _dist_ends[kl])/len(_cr_cell_list[kl]))/2
#             distance = _dist_starts[kl]
#             distance = distance + init_distance + _cr_cell_list[kl].index(c_id) * init_distance * 2
#
#     segment_geom = hs[hs.ID == seg_id].geometry.to_list()[0]
#     p = segment_geom.interpolate(distance)
#     point_list.append(p)
# _relevant_highway_network = hs[~hs.ID.isin([0, 1, 2])]
#
# _cells_with_cs["geometry"] = point_list
#
#
# _relevant_highway_network = _relevant_highway_network.to_crs("EPSG:3857")
# _merged_df = pd.merge(_cells_with_cs, charging_stations, on=["cell_id"])
# _to_plot = gpd.GeoDataFrame(_merged_df, geometry="geometry", crs=reference_coord_sys)
# _to_plot = _to_plot.to_crs("EPSG:3857")
# _to_plot["x"] = _to_plot.geometry.x
# _to_plot["y"] = _to_plot.geometry.y
#
# print(_relevant_highway_network)
#
# _bounds_ur = [(0, 0.3), (0.3, 0.6), (0.6, 0.9)]
# _bounds_diff = [(0, 0.5), (0.5, 1), (1, 1)]
#
# fig = plt.figure(figsize=(10, 10))
# plt.rcParams["font.family"] = "Franklin Gothic Book"
# plt.rcParams["font.size"] = 12
# gs = fig.add_gridspec(2, 2)
# axs = gs.subplots(sharex=True, sharey=True)
#
# letters = [["weekday summer", "weekday winter"], ["holiday summer", "holiday winter"]]
# for kl in range(0, 2):
#     for st in range(0, 2):
#         letter = letters[kl][st]
#
#         to_visualize.plot(ax=axs[kl][st], color="#ced4da", zorder=0, linewidth=0.7)
#
#         _relevant_highway_network.plot(
#             ax=axs[kl][st], label="Austrian highway network", color="black", zorder=0, linewidth=1
#         )
#         # _relevant_highway_network.plot(
#         #     ax=axs[1], color="black", zorder=0,
#         # )
#         # _relevant_highway_network.plot(
#         #     ax=axs[2], color="black", zorder=0,
#         # )
#         # _relevant_highway_network.plot(
#         #     ax=axs[3], color="black", zorder=0,
#         # )
#
#
#         axs[kl][st].axis("off")
#         # axs[1].axis("off")
#         # axs[2].axis("off")
#         # axs[3].axis("off")
#         axs[kl][st].annotate(letter, fontsize=15, xy=(0, 1), xycoords='axes fraction', xytext=(0.1, 0.9))
#         # axs[1].annotate("B", fontsize=17, xy=(0, 1), xycoords='axes fraction', xytext=(0.15, 0.85))
#         # axs[1].annotate("C", fontsize=17, xy=(0, 1), xycoords='axes fraction', xytext=(0.15, 0.85))
#         # axs[2].annotate("D", fontsize=17, xy=(0, 1), xycoords='axes fraction', xytext=(0.15, 0.85))
#
#         _marker_size = 100
#
#         # blue colors for the utility rate
#         _blue_tones = ["#90e0ef", "#00b4d8", "#0077b6"]
#         _red_tones = ["#f7cad0", "#ff7096", "#ff0a54"]
#         for mn in range(0, len(charging_stations)):
#             extract = _to_plot[_to_plot.index==_to_plot.index.to_list()[mn]]
#             for ij in range(0, len(_bounds_ur)):
#                 if ij < len(_red_tones)-1:
#                     cat = extract[(extract["utility rate"] >= _bounds_ur[ij][0]) & (extract["utility rate"] < _bounds_ur[ij][1])]
#                     axs[kl][st].scatter(
#                         cat["x"].to_list(),
#                         cat["y"].to_list(),
#                         s=_marker_size,
#                         color=_blue_tones[ij],
#                         # color="black",
#                         # label=str(int(_bounds_ur[ij][0])) + " - " + str(int(_bounds_ur[ij][1])),
#                         marker=MarkerStyle('o', fillstyle='left'),
#                         edgecolors='black',
#                         linewidth=0.4,
#                         zorder=charging_stations["cell_id"].to_list()[mn],
#                     )
#                     cat = extract[
#                         (extract["diff_to_max"] >= _bounds_diff[ij][0]) & (extract["diff_to_max"] < _bounds_diff[ij][1])]
#                     axs[kl][st].scatter(
#                         cat["x"].to_list(),
#                         cat["y"].to_list(),
#                         s=_marker_size,
#                         color=_red_tones[ij],
#                         # color="black",
#                         # label=str(int(_bounds_diff[ij][0])) + " - " + str(int(_bounds_diff[ij][1])),
#                         marker=MarkerStyle('o', fillstyle='right'),
#                         edgecolors='black',
#                         linewidth=0.4,
#                         zorder=charging_stations["cell_id"].to_list()[mn],
#                     )
#                 else:
#                     cat = extract[(extract["utility rate"] >= _bounds_ur[ij][0]) & (extract["utility rate"] <= _bounds_ur[ij][1])]
#                     axs[kl][st].scatter(
#                         cat["x"].to_list(),
#                         cat["y"].to_list(),
#                         s=_marker_size,
#                         color=_blue_tones[ij],
#                         # color="black",
#                         # label=str(int(_bounds_ur[ij][0])) + " - " + str(int(_bounds_ur[ij][1])),
#                         marker=MarkerStyle('o', fillstyle='left'),
#                         edgecolors='black',
#                         linewidth=0.4,
#                         zorder=charging_stations["cell_id"].to_list()[mn],
#                     )
#                     cat = extract[(extract["diff_to_max"] == _bounds_diff[ij][0])]
#                     axs[kl][st].scatter(
#                         cat["x"].to_list(),
#                         cat["y"].to_list(),
#                         s=_marker_size,
#                         color=_red_tones[ij],
#                         # color="black",
#                         # label=str(int(_bounds_diff[ij][0])) + " - " + str(int(_bounds_diff[ij][1])),
#                         marker=MarkerStyle('o', fillstyle='right'),
#                         edgecolors='black',
#                         linewidth=0.4,
#                         zorder=charging_stations["cell_id"].to_list()[mn],
#                     )
#
#
#
#
#         # plt.scatter(left[0], left[1],
#         #             marker=MarkerStyle('o', fillstyle='left'),
#         #             color='red', label='left')
#         # plt.scatter(right[0], right[1],
#         #             marker=MarkerStyle('o', fillstyle='right'),
#         #             color='blue', label='right')
# fig.subplots_adjust(hspace=0, wspace=0)
# fig.suptitle("Indications of charging infrastructure utilization", fontsize=18)
# plt.show()
