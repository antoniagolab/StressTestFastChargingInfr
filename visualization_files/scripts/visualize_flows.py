"""

Created on July 17th, 2022

With this script the traffic load along the highway is visualized
( eventually for both cases: summer + winter)
+ with OD nodes and their names


    - get all background images: NUTS-3, projected nodes,   -- check
    - match to each sub-segment the traffic load    --check
    - produce geometries of subsegments --check

TODO: plot should include:
        -> nuts 3
        -> highways
        -> od nodes + text



"""
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint
from utils import *
from shapely.ops import split, snap
import geopandas as gpd

# first: matching to each sub-segment the traffic load
# I need: pois_new
cr = pd.read_csv("data/cellularized_with_cells.csv")
pois_df = pd.read_csv("data/_demand_for_FC_workday_20220722.csv")
pois_df_summer = pd.read_csv("data/_demand_for_FC_holiday_20220722.csv")

# getting highway network
hs = pd.read_csv("geography/highway_segments.csv")
hs["geometry"] = hs["geometry"].apply(wkt.loads)
hs = gpd.GeoDataFrame(hs, geometry="geometry", crs=reference_coord_sys)

_relevant_highway_network = hs[~hs.ID.isin([0, 1, 2])]


# matching traffic load to subsegments
# iterating for each subsegment, extract pois, find maximum value of tc_0/t_1
reference_coord_sys = "EPSG:31287"

indices = cr.index.to_list()
poi_ID_starts = cr.poi_ID_start.to_list()
poi_ID_ends = cr.poid_ID_end.to_list()
geom_list = []
segment_ids = cr.segment_id.to_list()
poi_st_distance = cr.dist_start.to_list()
poi_end_distance = cr.dist_end.to_list()
for ij in range(0, len(indices)):
    id = indices[ij]
    st = poi_ID_starts[ij]
    end = poi_ID_ends[ij]
    seg_id = segment_ids[ij]
    if st < end:
        pois_extract = pois_df[(pois_df.ID  >= st) & (pois_df.ID <=end)]
        pois_extract_sum = pois_df_summer[(pois_df_summer.ID >= st) & (pois_df_summer.ID <= end)]
    else:
        pois_extract = pois_df[(pois_df.ID >= end) & (pois_df.ID <= st)]
        pois_extract_sum = pois_df_summer[(pois_df_summer.ID >= end) & (pois_df_summer.ID <= st)]

    nb_vh = pois_extract.tc_0.max() + pois_extract.tc_1.max()
    nb_vh_sum = pois_extract_sum.tc_0.max() + pois_extract_sum.tc_1.max()
    # adding info on traffic load
    cr.at[id, "traffic_load_workday"] = nb_vh
    cr.at[id, "traffic_load_holiday"] = nb_vh_sum

    # adding geometry
    segment_geom = hs[hs.ID == seg_id].geometry.to_list()[0]
    # project points
    dist_st = poi_st_distance[ij]
    dist_end = poi_end_distance[ij]

    st_point = segment_geom.interpolate(dist_st)
    end_point = segment_geom.interpolate(dist_end)

    linestring_parts = split(segment_geom, MultiPoint([st_point, end_point]))

    geom = st_point
    for l in linestring_parts:
        if l.distance(st_point) < 1e-3 and l.distance(end_point) < 1e-3:
            geom = l
            break

    geom_list.append(geom)
# cr.at[60, "traffic_load_workday"] = np.mean([cr.at[47, "traffic_load_workday"], cr.at[62, "traffic_load_workday"]])
# cr.at[60, "traffic_load_holiday"] = np.mean([cr.at[47, "traffic_load_holiday"], cr.at[62, "traffic_load_holiday"]])

cr["geometry"] = geom_list

_cr_gdf = gpd.GeoDataFrame(cr, geometry="geometry", crs=reference_coord_sys)
_cr_gdf_to_plot = _cr_gdf.to_crs("EPSG:3857")

# NUTS-3 regions

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
to_visualize["centroid"] = to_visualize.geometry.centroid
od_centroids = gpd.GeoDataFrame(to_visualize, geometry="centroid")

# get projected nodes
projected_nodes = gpd.read_file("data/all_projected_nodes.shp")
projected_nodes.at[75, "name"] = "Hungary"
projected_nodes.at[75, "NUTS_ID"] = "HU"
projected_nodes.at[76, "on_segment"] = 60
projected_nodes.at[76, "name"] = "Slowakai"
projected_nodes.at[76, "NUTS_ID"] = "SI"
projected_nodes.at[74, "name"] = np.NAN
projected_nodes.at[74, "NUTS_ID"] = np.NAN
only_relevant_ods = projected_nodes[~projected_nodes.NUTS_ID.isna()]
only_relevant_ods.at[14, "NUTS_ID"] = "Slovenia"
only_relevant_ods.at[41, "NUTS_ID"] = "Slovenia"
only_relevant_ods.at[70, "NUTS_ID"] = "Czech Republic"
only_relevant_ods.at[0, "NUTS_ID"] = "Germany"
only_relevant_ods.at[10, "NUTS_ID"] = "Germany"
only_relevant_ods.at[13, "NUTS_ID"] = "Italy"
only_relevant_ods.at[75, "NUTS_ID"] = "Hungary"
only_relevant_ods.at[76, "NUTS_ID"] = "Slovakia"

# plt.show()

# making three categories:
linewidths = [1, 2.25, 4, 5.5]
bounds = [(1000, 10000), (10000, 40000), (40000, 80000), (80000, 120000)]

_marker_size = 50

fig = plt.figure(figsize=(10, 5))
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 12
gs = fig.add_gridspec(1, 2)
axs = gs.subplots(sharex=True, sharey=True)

to_visualize.plot(ax=axs[0], color="#adb5bd", zorder=0, linewidth=0.7, alpha=0.8)
to_visualize.plot(ax=axs[1], color="#adb5bd", zorder=0, linewidth=0.7, alpha=0.8)

# _cr_gdf_to_plot.plot(ax=axs[0], color="#126782", linewidth=_cr_gdf_to_plot["traffic_load_workday"]/100000, zorder=5)
# _cr_gdf_to_plot.plot(ax=axs[1], color="#126782", linewidth=5, zorder=5)

only_relevant_ods = only_relevant_ods.to_crs("EPSG:3857")

for ij in range(0, len(bounds)):
    cat1 = _cr_gdf_to_plot[(_cr_gdf_to_plot["traffic_load_workday"] >= bounds[ij][0]) & (_cr_gdf_to_plot["traffic_load_workday"] < bounds[ij][1])]
    cat1.plot(ax=axs[0], color="#126782", linewidth=linewidths[ij], zorder=5, label=str(int(bounds[ij][0])) + " - " + str(int(bounds[ij][1])))

    cat2 = _cr_gdf_to_plot[(_cr_gdf_to_plot["traffic_load_holiday"] >= bounds[ij][0]) & (_cr_gdf_to_plot["traffic_load_holiday"] < bounds[ij][1])]
    cat2.plot(ax=axs[1], color="#126782", linewidth=linewidths[ij], zorder=5)

only_relevant_ods.plot(ax=axs[0], markersize=_marker_size, color="#fb8500", zorder=10, label="OD nodes")
only_relevant_ods.plot(ax=axs[1], markersize=_marker_size, color="#fb8500", zorder=10)

axs[0].axis("off")
axs[1].axis("off")

# axs[1].legend(loc="lower left", bbox_to_anchor=(-0.95, -0.05, 0.2, 0), ncol=15, fancybox=True)
title = "Traffic load"
# axs[0].set_xlim([plot_results_and_geom_df["x"].min()-100000,plot_results_and_geom_df["x"].max()+10000])

# axs[0].annotate(title, fontsize=18, xy=(0, 1.25), xycoords='axes fraction', xytext=(0.8, 1))
# axs[0].annotate("Traffic load on a workday", fontsize=13, xy=(0, 1), xycoords='axes fraction', xytext=(0, 0.9), family="Franklin Gothic Book")
# axs[1].annotate("Traffic load on a holiday", fontsize=13, xy=(0, 1), xycoords='axes fraction', xytext=(0, 0.9))

axs[0].set_title("Traffic load on a workday", fontsize=14)
axs[1].set_title("Traffic load on a holiday", fontsize=14)

axs[0].legend(loc="lower left", bbox_to_anchor=(0.4, -0.25, 0.2, 0), ncol=3, fancybox=True, fontsize=11)
#
texts = only_relevant_ods["NUTS_ID"].to_list()
geoms = only_relevant_ods.geometry.to_list()
for ij in range(0, len(texts)):
    axs[0].text(geoms[ij].x, geoms[ij].y, texts[ij], fontsize=10, zorder=20)

fig.subplots_adjust(hspace=0, wspace=0)
plt.savefig("figures/long_distance_flow.pdf", bbox_inches="tight")
plt.savefig("figures/long_distance_flow.svg", bbox_inches="tight")