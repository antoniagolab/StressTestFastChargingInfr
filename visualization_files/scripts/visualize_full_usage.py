"""
Created: July 17th, 2022

TODO:
    - process points with geometry  --check
    - calculate peak load for each  --check
    - calculate the diff to installed cap   --check
    - classify diff <= 350 and diff > 350
    - make visualization with four images

"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import geopandas as gpd
from shapely import wkt
import numpy as np

# enter here always the day with highest consumption
winter_workday = pd.read_csv("results/20220723-233215_charging_stationswinter_workdayfleet_input_20220722_compressed_probe2.csv")
summer_workday = pd.read_csv("results/20220724-033029_charging_stationssummer_workdayfleet_input_20220722_compressed_probe2.csv")
winter_holiday = pd.read_csv("results/20220724-083509_charging_stationswinter_holidayfleet_input_20220722_compressed_probe2.csv")
summer_holiday = pd.read_csv("results/20220724-114442_charging_stationssummer_holidayfleet_input_20220722_compressed_probe2.csv")
## TODO: delete this later
# winter_workday = winter_workday.drop(index=41)
# summer_workday = summer_workday.drop(index=41)
# winter_holiday = winter_holiday.drop(index=41)
# summer_holiday = summer_holiday.drop(index=41)

cell_id = 7
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


# calculate load factor for each
#
#
#
# nd difference between peak load and installed capacity
pole_power = 350

indices = winter_workday.index.to_list()
for ij in range(0, len(indices)):
    id = indices[ij]
    cap = winter_workday.at[id, "capacity"]

    _energy_charged_ww = [winter_workday.at[id, "E charged at t=" + str(t)]*(1/_mu) for t in range(0, T)]
    _total_energy_charged_ww = sum(_energy_charged_ww)
    _maximum_energy_charged_ww = max(_energy_charged_ww)
    diff = cap - _maximum_energy_charged_ww / 0.25
    winter_workday.at[id, "diff_to_max"] = (cap - _maximum_energy_charged_ww / 0.25) / cap
    winter_workday.at[id, "diff"] = winter_workday.at[id, "diff_to_max"] * cap
    if winter_workday.at[id, "diff"] >= 350:
        winter_workday.at[id, "peak_reached"] = False
    else:
        winter_workday.at[id, "peak_reached"] = True

    _energy_charged_sw = [summer_workday.at[id, "E charged at t=" + str(t)]*(1/_mu) for t in range(0, T)]
    _total_energy_charged_sw = sum(_energy_charged_sw )
    _maximum_energy_charged_sw = max(_energy_charged_sw )
    summer_workday.at[id, "diff_to_max"] = (cap - _maximum_energy_charged_sw / 0.25) / cap
    diff = cap - _maximum_energy_charged_sw / 0.25
    summer_workday.at[id, "diff"] = summer_workday.at[id, "diff_to_max"] * cap
    if summer_workday.at[id, "diff"] >= 350:
        summer_workday.at[id, "peak_reached"] = False
    else:
        summer_workday.at[id, "peak_reached"] = True

    _energy_charged_wh = [winter_holiday.at[id, "E charged at t=" + str(t)]*(1/_mu) for t in range(0, T)]
    _total_energy_charged_wh = sum(_energy_charged_wh )
    _maximum_energy_charged_wh = max(_energy_charged_wh )
    winter_holiday.at[id, "diff_to_max"] = (cap - _maximum_energy_charged_wh / 0.25) / cap
    # diff = cap - _maximum_energy_charged_wh
    winter_holiday.at[id, "diff"] = winter_holiday.at[id, "diff_to_max"] * cap

    if winter_holiday.at[id, "diff"] >= 350:
        winter_holiday.at[id, "peak_reached"] = False
    else:
        winter_holiday.at[id, "peak_reached"] = True

    _energy_charged_sh = [summer_holiday.at[id, "E charged at t=" + str(t)]*(1/_mu) for t in range(0, T)]
    _total_energy_charged_sh = sum(_energy_charged_sh)
    _maximum_energy_charged_sh = max(_energy_charged_sh)
    summer_holiday.at[id, "diff_to_max"] = (cap - _maximum_energy_charged_sh / 0.25) / cap
    summer_holiday.at[id, "diff"] = summer_holiday.at[id, "diff_to_max"] * cap
    diff = cap - _maximum_energy_charged_sh
    if summer_holiday.at[id, "diff"] >= 350:
        summer_holiday.at[id, "peak_reached"] = False
    else:
        summer_holiday.at[id, "peak_reached"] = True


winter_workday["diff_to_max"] = np.where(winter_workday["diff_to_max"] < 0, 0, winter_workday["diff_to_max"])
summer_workday["diff_to_max"] = np.where(summer_workday["diff_to_max"] < 0, 0, summer_workday["diff_to_max"])
winter_holiday["diff_to_max"] = np.where(winter_holiday["diff_to_max"] < 0, 0, winter_holiday["diff_to_max"])
summer_holiday["diff_to_max"] = np.where(summer_holiday["diff_to_max"] < 0, 0, summer_holiday["diff_to_max"])



reference_coord_sys = "EPSG:31287"

infrastructure = pd.read_csv("infrastructure/20220716-081946_input_HC_simulation_optimization_result_charging_stations.csv")
cells = pd.read_csv("data/20220719-201710_cells_input.csv")
cr = pd.read_csv("data/cellularized_with_cells.csv")
cr["cells"] = [eval(el) for el in cr["cells"].to_list()]

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
hs = gpd.GeoDataFrame(hs, geometry="geometry", crs=reference_coord_sys)


_cell_length = 27500    # (m)
init_distance = _cell_length/2

# identify positions for transform charging stations
_cells_with_cs = cells[cells.has_cs]
indices = _cells_with_cs.index.to_list()
_cell_ids = _cells_with_cs.cell_id.to_list()
_cr_cell_list = cr.cells.to_list()
point_list = []
_dist_starts = cr.dist_start.to_list()
_dist_ends = cr.dist_end.to_list()
for ij in range(0, len(indices)):
    c_id = _cell_ids[ij]
    seg_id = _cells_with_cs.at[indices[ij], "seg_id"]
    capacity = _cells_with_cs.at[indices[ij], "capacity"]
    match_id = None
    distance = None

    for kl in range(0, len(_cr_cell_list)):
        if c_id in _cr_cell_list[kl]:
            match_id = kl
            init_distance = ((_dist_ends[kl]- _dist_starts[kl])/len(_cr_cell_list[kl]))/2
            distance = _dist_starts[kl]
            distance = distance + init_distance + _cr_cell_list[kl].index(c_id) * init_distance * 2

    segment_geom = hs[hs.ID == seg_id].geometry.to_list()[0]
    p = segment_geom.interpolate(distance)
    point_list.append(p)
_relevant_highway_network = hs[~hs.ID.isin([0, 1, 2])]

_cells_with_cs["geometry"] = point_list


_relevant_highway_network = _relevant_highway_network.to_crs("EPSG:3857")
_merged_df = pd.merge(_cells_with_cs, winter_workday, on=["cell_id"])
_to_plot = gpd.GeoDataFrame(_merged_df, geometry="geometry", crs=reference_coord_sys)
_to_plot = _to_plot.to_crs("EPSG:3857")
_to_plot["x"] = _to_plot.geometry.x
_to_plot["y"] = _to_plot.geometry.y

print(_relevant_highway_network)
colors = ["#087e8b", "#ff5a5f"]
_bounds_ur = [(0, 0.3), (0.3, 0.6), (0.6, 0.9)]
_bounds_diff = [(0, 0.5), (0.5, 1), (1, 1)]
#
fig = plt.figure(figsize=(10, 10))
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 12
gs = fig.add_gridspec(2, 2)
axs = gs.subplots(sharex=True, sharey=True)
_marker_size = 60
letters = [["Workday in winter", "Workday in summer"], ["Holiday in winter", "Holiday in summer"]]
for ij in [0, 1]:
    for kl in [0, 1]:
        letter = letters[ij][kl]
        to_visualize.plot(ax=axs[ij][kl], color="#ced4da", zorder=0, linewidth=0.7)

        axs[ij][kl].axis("off")
        # axs[ij][kl].annotate(letter, fontsize=15, xy=(0, 0), xycoords='axes fraction', xytext=(0, 0.5))
        axs[ij][kl].set_title(letter, fontsize=12)
        if (ij, kl) == (0, 0):
            _merged = winter_workday.copy()
            _merged["x"] = _to_plot["x"]
            _merged["y"] = _to_plot["y"]
        elif (ij, kl) == (0, 1):
            _merged = summer_workday.copy()
            _merged["x"] = _to_plot["x"]
            _merged["y"] = _to_plot["y"]
        elif (ij, kl) == (1, 0):
            _merged = winter_holiday.copy()
            _merged["x"] = _to_plot["x"]
            _merged["y"] = _to_plot["y"]
        else:
            _merged = summer_holiday.copy()
            _merged["x"] = _to_plot["x"]
            _merged["y"] = _to_plot["y"]

        _cat_neg = _merged[_merged["peak_reached"]==False]
        _cat_pos = _merged[_merged["peak_reached"]]

        axs[ij][kl].scatter(
            _cat_neg["x"].to_list(),
            _cat_neg["y"].to_list(),
            s=_marker_size,
            color=colors[0],
            # color="black",
            label="Full capacity not used",
            edgecolors='black',
            linewidth=0.4,
            zorder=10,
        )
        axs[ij][kl].scatter(
            _cat_pos["x"].to_list(),
            _cat_pos["y"].to_list(),
            s=_marker_size,
            color=colors[1],
            # color="black",
            # label=str(int(_bounds_ur[ij][0])) + " - " + str(int(_bounds_ur[ij][1])),
            label="Full capacity used",
            edgecolors='black',
            linewidth=0.4,
            zorder=10,
        )
        _relevant_highway_network.plot(
            ax=axs[ij][kl], label="Highway network", color="black", zorder=0, linewidth=1
        )

axs[0][1].legend(loc="lower left", bbox_to_anchor=(-0.8, -0.15, 0.2, 0), ncol=3, fancybox=True, fontsize=12)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figures/diff_to_peak.pdf", bbox_inches="tight")
plt.savefig("figures/diff_to_peak.svg", bbox_inches="tight")
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




