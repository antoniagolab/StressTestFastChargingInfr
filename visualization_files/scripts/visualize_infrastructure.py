"""
Date of creation: 17.07.2022, AG

This script serves to visualize the input charging infrastructure for
the spatio-temporal charging model

A figure with two subplots is created here:

Figure (1): visualization of direct output of HCS model
Figure (2): visualization of input to ST model where cell structure is implemented and a coarser geographic resolution
    introduced


    - figure 1  -- check
        - visualize only relevant NUTS-3 regions    --check
        - visualize only relevant highway network   --check
        - visualize infrastructure (similar to OG visualization but different colors)   --check
        - choose appropriate classification (max 4) --check
        - choose nice colors    --check

  TODO:
    - figure 2
        - match charging capacities to cells
        - get cells, get pois_df
        - identify for each, cell with charging capacity the midpoint and plot it



"""
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import geopandas as gpd
from matplotlib.offsetbox import AnchoredText

reference_coord_sys = "EPSG:31287"

infrastructure = pd.read_csv("infrastructure/20220719-104419_input_HC_simulation_optimization_result_charging_stations.csv")
cells = pd.read_csv("data/20220720-171835_cells_input.csv")
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

_relevant_highway_network = hs[~hs.ID.isin([0, 1, 2])]


# getting charging infrastructure and matching this to the existing resting areas

results = infrastructure
filtered_results = results[results["pois_type"] == "ra"]
# osm geometries

rest_areas = pd2gpd(
    pd.read_csv("data/projected_ras.csv"), geom_col_name="centroid"
).sort_values(by=["on_segment", "dist_along_highway"])
rest_areas["segment_id"] = rest_areas["on_segment"]
col_type_ID = 'type_ID'
col_directions = "dir"
col_segment_id = "segment_id"
rest_areas[col_type_ID] = rest_areas["nb"]
rest_areas[col_directions] = rest_areas["evaluated_dir"]

# merge here
results_and_geom_df = pd.merge(
    filtered_results, rest_areas, on=[col_segment_id, col_type_ID, col_directions]
)
results_and_geom_df_2 = pd.merge(
    results, rest_areas, on=[col_segment_id, col_type_ID, col_directions]
)

# turn into GeoDataframe
results_and_geom_df["geometry"] = results_and_geom_df.centroid
results_and_geom_df["total_charging_pole_number"] = np.where(
    np.array(results_and_geom_df.pYi_dir) == 0,
    np.nan,
    np.array(results_and_geom_df.pYi_dir),
)
results_and_geom_df = gpd.GeoDataFrame(
    results_and_geom_df, crs=reference_coord_sys, geometry="geometry"
)
results_and_geom_df["charging_capacity"] = (
        results_and_geom_df["total_charging_pole_number"] * 350
)

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
            init_distance = (( _dist_ends[kl] - _dist_starts[kl])/len(_cr_cell_list[kl]))/2
            distance = _dist_starts[kl]
            distance = distance + init_distance + _cr_cell_list[kl].index(c_id) * init_distance * 2

    if seg_id == 4:
        print(c_id, distance)
    segment_geom = hs[hs.ID == seg_id].geometry.to_list()[0]
    p = segment_geom.interpolate(distance)
    point_list.append(p)

_cells_with_cs["geometry"] = point_list
_to_plot = gpd.GeoDataFrame(_cells_with_cs, geometry="geometry", crs=reference_coord_sys)
_to_plot = _to_plot.to_crs("EPSG:3857")

# plot
plot_results_and_geom_df = results_and_geom_df.to_crs("EPSG:3857")
plot_results_and_geom_df = plot_results_and_geom_df[
    plot_results_and_geom_df.total_charging_pole_number > 0
    ]

_plot_highway_geometries = _relevant_highway_network.to_crs("EPSG:3857")
_plot_highway_geometries["null"] = [0] * len(_plot_highway_geometries)

plot_results_and_geom_df["x"] = plot_results_and_geom_df.geometry.x
plot_results_and_geom_df["y"] = plot_results_and_geom_df.geometry.y
results_and_geom_df_2["geometry"] = results_and_geom_df_2.centroid
results_and_geom_df_2 = gpd.GeoDataFrame(
    results_and_geom_df_2, crs=reference_coord_sys, geometry="geometry"
)
plot_results_and_geom_df_2 = results_and_geom_df_2.to_crs("EPSG:3857")

# plot settings

_marker_sizes = [40, 70, 90, 120]

fig = plt.figure(figsize=(4, 5))
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 7
gs = fig.add_gridspec(1, 1)
axs = gs.subplots(sharex=True, sharey=True)
# nuts_3_at.plot(ax=ax, color="#adb5bd", zorder=0, label="NUTS-3 regions", alpha=0.8)

# to_visualize.plot(ax=axs[1], color="#ced4da", zorder=0, linewidth=0.7)

# plt.show()
# _plot_highway_geometries.plot(
#     ax=axs[1], label="Austrian highway network", color="black", zorder=0,
# )
_plot_highway_geometries.plot(
    ax=axs, color="black", zorder=0,
)

# insert charging infrastructure
_max_value = 20000  # kW, 20 MW
_bounds = [0, 10000, 20000, 40000, 50000]
colors = [ "#8ecae6", "#219ebc", "#054a91" , "#023047",  "#2a9d8f", "#264653", "black"]
linewidth = 0.5
# for ij in range(0, len(_bounds) - 1):
#     cat = plot_results_and_geom_df[(plot_results_and_geom_df["charging_capacity"] > _bounds[ij]) & (plot_results_and_geom_df["charging_capacity"] <= _bounds[ij+1])]
#     axs[0].scatter(
#         cat["x"].to_list(),
#         cat["y"].to_list(),
#         s=_marker_sizes[ij],
#         color=colors[ij],
#         # color="black",
#         label=str(int(_bounds[ij])) + " - " + str(int(_bounds[ij + 1])) + " kW",
#         edgecolors='black',
#         linewidth=linewidth,
#         zorder=10,
#     )
#
_to_plot["x"] = _to_plot.geometry.x
_to_plot["y"] = _to_plot.geometry.y

for ij in range(0, len(_bounds) - 1):
    cat = _to_plot[(_to_plot["capacity"] > _bounds[ij]) & (_to_plot["capacity"] <= _bounds[ij+1])]
    axs.scatter(
        cat["x"].to_list(),
        cat["y"].to_list(),
        s=_marker_sizes[ij],
        color=colors[ij],
        # color="black",
        label=str(int(_bounds[ij])) + " - " + str(int(_bounds[ij + 1])) + " kW",
        edgecolors='black',
        linewidth=linewidth,
        zorder=10,
    )
axs.axis("off")
# axs[1].axis("off")
# fig.tight_layout()
to_visualize.plot(ax=axs, color="#adb5bd", zorder=0, linewidth=0.7, alpha=0.8, label="Highway network")

axs.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2, 0, 0), ncol=2, fancybox=True, fontsize=7)
# axs[0].legend(loc="lower left", bbox_to_anchor=(0, 0.6, 1, 0), ncol=1, fancybox=True)
# axs[0].annotate("Fast-charging infrastructure", loc="lower left", bbox_to_anchor=(-1, -0.05, 0.08, 0), ncol=15, fancybox=True)
title = "Fast-charging infrastructure"
axs.set_xlim([plot_results_and_geom_df["x"].min()-100000,plot_results_and_geom_df["x"].max()+10000])

# axs[0].annotate(title, fontsize=15, xy=(1, 1.25), xycoords='axes fraction', xytext=(0.8, 1))
# axs[0].annotate("A", fontsize=13, xy=(0, 1), xycoords='axes fraction', xytext=(0.15, 0.85))
# axs[1].annotate("B", fontsize=13, xy=(0, 1), xycoords='axes fraction', xytext=(0.15, 0.85))
fig.subplots_adjust(hspace=0, wspace=0)
axs.set_title(title, fontsize=10)

plt.savefig("figures/fast_charging_infrastructure.pdf", bbox_inches="tight")
plt.savefig("figures/fast_charging_infrastructure.svg", bbox_inches="tight")


