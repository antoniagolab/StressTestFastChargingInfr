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
winter_workday = pd.read_csv("results/20220722-011953_charging_stationswinter_workdayfleet_input_20220720_compressed_probe2.csv")
winter_holiday = pd.read_csv("results/20220722-104338_charging_stationswinter_holidayfleet_input_20220720_compressed_probe2.csv")
summer_workday = pd.read_csv("results/20220722-155257_charging_stationssummer_workdayfleet_input_20220720_compressed_probe2.csv")
summer_holiday = pd.read_csv("results/20220722-131555_charging_stationssummer_holidayfleet_input_20220720_compressed_probe2.csv")
# TODO: delete this later
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
winter_workday["cs"] = [False] * len(winter_workday)
winter_workday.at[0, "cs"] = True
winter_workday.at[1, "cs"] = True
winter_workday.at[2, "cs"] = True
winter_workday.to_csv("data/processed_sa.csv")
winter_workday["cs"] = [False] * len(winter_workday)
winter_workday.at[23, "cs"] = True
winter_workday.at[19, "cs"] = True

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
            init_distance = ((_dist_ends[kl] - _dist_starts[kl])/len(_cr_cell_list[kl]))/2
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
winter_workday.to_csv("data/processed.csv")
_to_plot.to_csv("data/allocation.csv")
print(_relevant_highway_network)
colors = ["#087e8b", "#ff5a5f"]
_bounds_ur = [(0, 0.3), (0.3, 0.6), (0.6, 0.9)]
_bounds_diff = [(0, 0.5), (0.5, 1), (1, 1)]
#

fig, ax = plt.subplots(figsize=(5, 5))
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 12
_marker_size = 60
letters = [["Weekday in winter", "Weekday in summer"], ["Holiday in winter", "Holiday in summer"]]
to_visualize.plot(ax=ax, color="#ced4da", zorder=0, linewidth=0.7)

ax.axis("off")
# axs[ij][kl].annotate(letter, fontsize=15, xy=(0, 0), xycoords='axes fraction', xytext=(0, 0.5))
ax.set_title("Allocation of selected charging stations", fontsize=10)
_merged = winter_workday.copy()
_merged["x"] = _to_plot["x"]
_merged["y"] = _to_plot["y"]
_cat_neg = _merged[_merged["cs"] == False]
_cat_pos = _merged[_merged["cs"]]
ax.scatter(
    _cat_neg["x"].to_list(),
    _cat_neg["y"].to_list(),
    s=_marker_size,
    color="white",
    # color="black",
    edgecolors='black',
    linewidth=0.4,
    zorder=10,
)
ax.scatter(
    _cat_pos["x"].to_list(),
    _cat_pos["y"].to_list(),
    s=_marker_size,
    color="black",
    # color="black",
    edgecolors='black',
    linewidth=0.4,
    zorder=10,
)
_relevant_highway_network.plot(
    ax=ax, label="Austrian highway network", color="black", zorder=0, linewidth=1
)


# axs[0][1].legend(loc="lower left", bbox_to_anchor=(-0.85, -0.15, 0.2, 0), ncol=3, fancybox=True, fontsize=11)
fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figures/allocation_A_B.pdf", bbox_inches="tight")
plt.savefig("figures/allocation_A_B.svg", bbox_inches="tight")
