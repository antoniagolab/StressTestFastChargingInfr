import pandas as pd
import geopandas as gpd
import glob
import os
from shapely import wkt

# from optimization_parameters import *
# from _variable_definitions import *
import contextily as ctx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from utils import pd2gpd
from matplotlib import rc
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# from _file_import_optimization import *
from matplotlib import rc
from matplotlib.ticker import MaxNLocator


from pandas.core.common import SettingWithCopyWarning
import warnings

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# ---------------------------------------------------------------------------------------------------------------------
# input data for visualizations
# ---------------------------------------------------------------------------------------------------------------------

val = pd.read_csv(
    "validation_results/20220208-202154_validation 1_optimization_result_charging_stations.csv"
)
scenario = pd.read_csv(
    "scenarios/results/20220209-144610_Directed Transition_optimization_result_charging_stations.csv"
)

path_SA_driving_range = "sensitivity_analyses/driving_range/"

list_of_paths_SA_driving_range = [
    "20220215-085558_TF200_22_optimization_result_charging_stations.csv",
    "20220215-090500_TF300_22_optimization_result_charging_stations.csv",
    "20220215-091550_TF400_22_optimization_result_charging_stations.csv",
    "20220215-093041_TF500_22_optimization_result_charging_stations.csv",
    "20220215-095047_TF600_22_optimization_result_charging_stations.csv",
    "20220215-101421_TF700_22_optimization_result_charging_stations.csv",
    "20220215-104036_TF800_22_optimization_result_charging_stations.csv",
    "20220215-111415_TF900_22_optimization_result_charging_stations.csv",
    "20220215-114133_TF1000_22_optimization_result_charging_stations.csv",
    "20220215-123025_TF1100_22_optimization_result_charging_stations.csv",
    "20220215-125510_TF1200_22_optimization_result_charging_stations.csv",
    "20220215-132158_TF1300_22_optimization_result_charging_stations.csv",
    "20220215-135720_TF1400_22_optimization_result_charging_stations.csv",
]

path_SA_share_BEV = "sensitivity_analyses/epsilon_increase/"

list_of_paths_SA_share_BEV = [
    "20220215-094034_ev share - epsilon SC10_3_optimization_result_charging_stations.csv",
    "20220215-095419_ev share - epsilon SC20_3_optimization_result_charging_stations.csv",
    "20220215-100808_ev share - epsilon SC30_3_optimization_result_charging_stations.csv",
    "20220215-102142_ev share - epsilon SC40_3_optimization_result_charging_stations.csv",
    "20220215-103522_ev share - epsilon SC50_3_optimization_result_charging_stations.csv",
    "20220215-105154_ev share - epsilon SC60_3_optimization_result_charging_stations.csv",
    "20220215-111343_ev share - epsilon SC70_3_optimization_result_charging_stations.csv",
    "20220215-134430_ev share - epsilon SC80_3_optimization_result_charging_stations.csv",
    "20220215-195040_ev share - epsilon SC90_3_optimization_result_charging_stations.csv",
    "20220215-214848_ev share - epsilon SC100_3_optimization_result_charging_stations.csv",
]

_filename = "sensitivity_analyses\cost_reduction_potentials_1202.csv"
scenario_file = pd.read_csv("scenarios/optimization_results_1002.csv")
# ---------------------------------------------------------------------------------------------------------------------
# VALIDATION visualization
# ---------------------------------------------------------------------------------------------------------------------

# colors
colors = ["#5f0f40", "#9a031e", "#E9C46A", "#e36414", "#0f4c5c"]
colors.reverse()

# reference coordinate system for all visualisation
reference_coord_sys = "EPSG:31287"

# highway geometries
highway_geometries = pd.read_csv(r"geometries/highway_geometries_v6.csv")
highway_geometries["geometry"] = highway_geometries.geometry.apply(wkt.loads)
highway_geometries = gpd.GeoDataFrame(highway_geometries)
highway_geometries = highway_geometries.set_crs(reference_coord_sys)
highway_geometries["length"] = highway_geometries.geometry.length
segments_gdf = pd2gpd(pd.read_csv("data/highway_segments.csv"))

copy_highway_geometries = highway_geometries.drop_duplicates(subset=["highway"])

# austrian borders
austrian_border = gpd.read_file("geometries/austrian_border.shp")

# get latest result file
list_of_files = glob.glob("scenarios/*")
# latest_file = max(list_of_files, key=os.path.getctime)

charging_capacity = 150  # (kW)

# energies = scenario_file.p_max_bev.to_list()


def merge_with_geom(results, energy):

    filtered_results = results[results[col_type] == "ra"]
    # osm geometries

    rest_areas = pd2gpd(
        pd.read_csv("data/projected_ras.csv"), geom_col_name="centroid"
    ).sort_values(by=["on_segment", "dist_along_highway"])
    rest_areas["segment_id"] = rest_areas["on_segment"]
    rest_areas[col_type_ID] = rest_areas["nb"]
    rest_areas[col_directions] = rest_areas["evaluated_dir"]

    # merge here
    results_and_geom_df = pd.merge(
        filtered_results, rest_areas, on=[col_segment_id, col_type_ID, col_directions]
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
        results_and_geom_df["total_charging_pole_number"] * energy
    )

    # plot
    plot_results_and_geom_df = results_and_geom_df.to_crs("EPSG:3857")
    # plot_results_and_geom_df = plot_results_and_geom_df[
    #     plot_results_and_geom_df.total_charging_pole_number > 0
    # ]
    plot_results_and_geom_df["x"] = plot_results_and_geom_df.geometry.x
    plot_results_and_geom_df["y"] = plot_results_and_geom_df.geometry.y
    return plot_results_and_geom_df


plot_sc_1 = merge_with_geom(val, charging_capacity)
merged = pd.merge(
    plot_sc_1, existing_infr, how="left", on=["segment_id", "name", "dir"]
)
sub_1 = merged[merged.has_charging_station == True]
sub_2 = merged[merged.total_charging_pole_number > 0]

sub_1["installed_cap"] = (
    sub_1["44kW"] * 44
    + sub_1["50kW"] * 50
    + sub_1["75kW"] * 75
    + sub_1["150kW"] * 150
    + sub_1["350kW"] * 350
)
sub_2["installed_cap"] = sub_2["total_charging_pole_number"] * charging_capacity


plot_highway_geometries = highway_geometries.to_crs("EPSG:3857")
plot_austrian_border = austrian_border.to_crs("EPSG:3857")
plot_highway_geometries["null"] = [0] * len(plot_highway_geometries)
min_size = 30
max_size = 150
max_val = max([sub_1["installed_cap"].max(), sub_1["installed_cap"].max()])
fact = max_size / max_val
sizes = list(np.linspace(55, 150, 5))

bounds = np.linspace(0, max_val, 6)
cm = 1 / 2.54
bounds[0] = 44
fig = plt.figure(figsize=(15, 10))
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 10
gs = fig.add_gridspec(2, hspace=0)
axs = gs.subplots(sharex=True, sharey=True)
plot_highway_geometries.plot(
    ax=axs[0], label="Austrian highway network", color="black", zorder=0, linewidth=1
)
plot_highway_geometries.plot(
    ax=axs[1], label="Austrian highway network", color="black", zorder=0, linewidth=1
)
plot_austrian_border.plot(ax=axs[0], color="grey", linewidth=1)
plot_austrian_border.plot(ax=axs[1], color="grey", linewidth=1)

for ij in range(0, len(bounds) - 1):
    cat = sub_1[sub_1["installed_cap"].isin(np.arange(bounds[ij], bounds[ij + 1]))]
    axs[0].scatter(
        cat["x"].to_list(),
        cat["y"].to_list(),
        s=sizes[ij],
        color=colors[ij],
        label=str(int(bounds[ij])) + " - " + str(int(bounds[ij + 1])) + " kW",
        # edgecolors='black',
        zorder=10,
    )

for ij in range(0, len(bounds) - 1):
    cat = sub_2[sub_2["installed_cap"].isin(np.arange(bounds[ij], bounds[ij + 1]))]
    axs[1].scatter(
        cat["x"].to_list(),
        cat["y"].to_list(),
        s=sizes[ij],
        color=colors[ij],
        label=str(int(bounds[ij])) + " - " + str(int(bounds[ij + 1])) + " kW",
        # edgecolors='black',
        zorder=10,
    )
axs[0].axis("off")
axs[1].axis("off")
axs[0].text(
    1.07e6,
    6.2e6,
    "Existing infrastructure",
    bbox=dict(facecolor="none", edgecolor="grey", boxstyle="round,pad=0.4"),
    fontsize=14,
)
axs[1].text(
    1.07e6,
    6.2e6,
    "Model output",
    bbox=dict(facecolor="none", edgecolor="grey", boxstyle="round,pad=0.4"),
    fontsize=14,
)


# plotting NUTS 2
p = gpd.read_file("geometries\output_BL.shp")
bd = p.to_crs("EPSG:3857")
geoms = bd.geometry.to_list()
names = bd["NAME"].to_list()
for ij in range(0, len(bd)):
    if geoms[ij].type == "MultiPolygon":
        for g in geoms[ij]:
            axs[0].plot(*g.exterior.xy, color="grey", linewidth=1)
            axs[1].plot(*g.exterior.xy, color="grey", linewidth=1)
    else:
        axs[0].plot(*geoms[ij].exterior.xy, color="grey", linewidth=1)
        axs[1].plot(*geoms[ij].exterior.xy, color="grey", linewidth=1)
    c = geoms[ij].centroid
    # axs[0].text(c.x, c.y + 0.03e6, names[ij], color="grey")
    # axs[1].text(c.x, c.y + 0.03e6, names[ij], color="grey")

axs[0].legend(loc="lower left", bbox_to_anchor=(0.15, 1, 1, 0), ncol=3, fancybox=True)
# plt.show()
plt.savefig("figures/comparison_image.pdf", bbox_inches="tight")


# ---------------------------------------------------------------------------------------------------------------------
# EXPANSION visualization
# ---------------------------------------------------------------------------------------------------------------------

# colors
colors = ["#6b705c", "#2a9d8f", "#264653", "#f4a261", "#e76f51"]
# colors.reverse()

# reference coordinate system for all visualisation
reference_coord_sys = "EPSG:31287"

# highway geometries
highway_geometries = pd.read_csv(r"geometries/highway_geometries_v6.csv")
highway_geometries["geometry"] = highway_geometries.geometry.apply(wkt.loads)
highway_geometries = gpd.GeoDataFrame(highway_geometries)
highway_geometries = highway_geometries.set_crs(reference_coord_sys)
highway_geometries["length"] = highway_geometries.geometry.length
segments_gdf = pd2gpd(pd.read_csv("data/highway_segments.csv"))

copy_highway_geometries = highway_geometries.drop_duplicates(subset=["highway"])

# austrian borders
austrian_border = gpd.read_file("geometries/austrian_border.shp")

# get latest result file
list_of_files = glob.glob("scenarios/*")
# latest_file = max(list_of_files, key=os.path.getctime)

charging_capacity = 350  # (kW)

# energies = scenario_file.p_max_bev.to_list()


def merge_with_geom(results, energy):

    filtered_results = results[results[col_type] == "ra"]
    # osm geometries

    rest_areas = pd2gpd(
        pd.read_csv("data/projected_ras.csv"), geom_col_name="centroid"
    ).sort_values(by=["on_segment", "dist_along_highway"])
    rest_areas["segment_id"] = rest_areas["on_segment"]
    rest_areas[col_type_ID] = rest_areas["nb"]
    rest_areas[col_directions] = rest_areas["evaluated_dir"]

    # merge here
    results_and_geom_df = pd.merge(
        filtered_results, rest_areas, on=[col_segment_id, col_type_ID, col_directions]
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
        results_and_geom_df["total_charging_pole_number"] * energy
    )

    # plot
    plot_results_and_geom_df = results_and_geom_df.to_crs("EPSG:3857")
    # plot_results_and_geom_df = plot_results_and_geom_df[
    #     plot_results_and_geom_df.total_charging_pole_number > 0
    # ]
    plot_results_and_geom_df["x"] = plot_results_and_geom_df.geometry.x
    plot_results_and_geom_df["y"] = plot_results_and_geom_df.geometry.y
    return plot_results_and_geom_df


plot_sc_1 = merge_with_geom(scenario, charging_capacity)
merged = pd.merge(
    plot_sc_1, existing_infr, how="left", on=["segment_id", "name", "dir"]
)
# sub_1 = merged[merged.has_charging_station == True]
# sub_2 = merged[merged.total_charging_pole_number > 0]

merged["existing_cap"] = +merged["350kW"] * 350
merged["model_cap"] = merged["total_charging_pole_number"] * charging_capacity
merged["existing_cap"] = merged["existing_cap"].replace(np.NaN, 0)
merged["model_cap"] = merged["model_cap"].replace(np.NaN, 0)

# make classification here
merged["diff"] = merged["model_cap"] - merged["existing_cap"]

merged["difference"] = np.where(merged["diff"] < 0, 0, merged["diff"])

comp_df = merged[merged.total_charging_pole_number > 0]

max_val = comp_df["difference"].max()
bounds = [charging_capacity] + [int(round(max_val / 2, -2)), int(max_val)]
size_1 = 150  # small
size_2 = 300  # big
# plot grey , difference == 0
# plot the two classes, for where has_charging_infrastructure == True (blue)
#  plot the two classes, for where !(has_charging_infrastructure == True) (red)

plot_highway_geometries = highway_geometries.to_crs("EPSG:3857")
plot_austrian_border = austrian_border.to_crs("EPSG:3857")
plot_highway_geometries["null"] = [0] * len(plot_highway_geometries)

bd = p.to_crs("EPSG:3857")
fig, ax = plt.subplots(figsize=(13, 8))
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 13

# plotting NUTS 2
geoms = bd.geometry.to_list()
names = bd["NAME"].to_list()
for ij in range(0, len(bd)):
    if geoms[ij].type == "MultiPolygon":
        for g in geoms[ij]:
            ax.plot(*g.exterior.xy, color="grey", linewidth=1)
    else:
        ax.plot(*geoms[ij].exterior.xy, color="grey", linewidth=1)
    c = geoms[ij].centroid
    # ax.text(c.x, c.y + 0.03e6, names[ij], color="grey")


# plotting highway network and Austrian boarder
plot_highway_geometries.plot(
    ax=ax, label="Austrian highway network", color="black", zorder=0, linewidth=1
)

# count together for the four categories all capacity
expansion_values = []


# plot_austrian_border.plot(ax=ax, color="grey", linewidth=1)

# plot the ones with no change

# plot the two classes, for where has_charging_infrastructure == True (blue);

cat = comp_df[comp_df.has_charging_station == True]
cat0 = cat[cat["difference"].isin(list(range(bounds[0], bounds[1] + 1)))]
cat1 = cat[cat["difference"].isin(list(range(bounds[1] + 1, bounds[2] + 1)))]

expansion_values.append(cat0["difference"].sum())
expansion_values.append(cat1["difference"].sum())
ax.scatter(
    cat0["x"].to_list(),
    cat0["y"].to_list(),
    s=size_1,
    color=colors[1],
    label="Expansion of existing CS by "
    + str(bounds[0])
    + " - "
    + str(bounds[1])
    + " kW",
    zorder=10,
)
ax.scatter(
    cat1["x"].to_list(),
    cat1["y"].to_list(),
    s=size_2,
    color=colors[2],
    label="Expansion of existing CS by "
    + str(bounds[1])
    + " - "
    + str(bounds[2])
    + " kW",
    zorder=10,
)

#  plot the two classes, for where !(has_charging_infrastructure == True) (red)

cat = comp_df[~(comp_df.has_charging_station == True)]
cat0 = cat[cat["difference"].isin(list(range(bounds[0], bounds[1] + 1)))]
cat1 = cat[cat["difference"].isin(list(range(bounds[1] + 1, bounds[2] + 1)))]
expansion_values.append(cat0["difference"].sum())
expansion_values.append(cat1["difference"].sum())
ax.scatter(
    cat0["x"].to_list(),
    cat0["y"].to_list(),
    s=size_1,
    color=colors[3],
    label="Newly installed CS with " + str(bounds[0]) + " - " + str(bounds[1]) + " kW",
    # edgecolors='black',
    zorder=10,
)
ax.scatter(
    cat1["x"].to_list(),
    cat1["y"].to_list(),
    s=size_2,
    color=colors[4],
    label="Newly installed CS with " + str(bounds[1]) + " - " + str(bounds[2]) + " kW",
    # edgecolors='black',
    zorder=10,
)
cat = comp_df[comp_df["difference"] == 0]
if len(cat) > 0:
    ax.scatter(
        cat["x"].to_list(),
        cat["y"].to_list(),
        s=size_1,
        color=colors[0],
        label="No expansion of existing CS",
        # edgecolors='black',
        zorder=10,
    )


ax.axis("off")
ax.set_title(
    "Required charging infrastructure expansion until 2030 under the DT scenario"
)
ax.legend(loc="lower left", bbox_to_anchor=(0, 0.6, 1, 0), ncol=1, fancybox=True)

tot = sum(expansion_values)
expansion_values = [e / tot * 100 for e in expansion_values]

plt.savefig("figures/expansion_image.pdf", bbox_inches="tight")

# ---------------------------------------------------------------------------------------------------------------------
# Cost reduction potentials
# ---------------------------------------------------------------------------------------------------------------------

_cost_decrease_analysis = pd.read_csv(_filename)


costs = [scenario_file.loc[1].costs] + _cost_decrease_analysis.costs.to_list()[0:4]
# spec_BEV_costs = [scenario_file.loc[3]['€/BEV']] + _cost_decrease_analysis['€/BEV'].to_list()[0:4]
# spec_kW_costs = [scenario_file.loc[3]['€/kW']] + _cost_decrease_analysis['€/kW'].to_list()[0:4]
labels = ["GD scenario"] + _cost_decrease_analysis["scenario_name"].to_list()[0:4]
labels = [
    "GD scenario\n2030",
    "Medium decrease\nin road traffic",
    "Major decrease\nin road traffic",
    "Increase in\ndriving range",
    "Increase in\ncharging power",
]
c = "#f4f1de"

fig, ax = plt.subplots(figsize=(9, 4))
plt.rcParams["xtick.bottom"] = plt.rcParams["xtick.labelbottom"] = False
plt.rcParams["xtick.top"] = plt.rcParams["xtick.labeltop"] = True
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 12
# gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
# ax1 = plt.subplot(gs[0])
# ax2 = plt.subplot(gs[1], sharex=ax1)
ax.tick_params(axis="both", which="major", labelsize=10)
l2 = ax.bar(
    labels[0],
    costs[0],
    width=0.7,
    color=["#f6bd60"],
    zorder=10,
    label="infrastructure costs in GD scenario 2030",
)
l3 = ax.bar(
    labels[1:],
    costs[1:],
    width=0.7,
    color=["#3d405b"] * 4,
    zorder=10,
    label="reduced infrastructure costs of GD scenario 2030",
)
l4 = ax.bar(
    labels,
    costs[0],
    color=c,
    width=0.7,
    zorder=5,
    label="cost difference relative to GD scenario 2030",
)
ax.axhline(y=costs[0], linewidth=3, color="#f6bd60", linestyle="--", zorder=30)
ax.grid(axis="y")
ax.set_ylabel(
    "Total infrastructure expansion costs (€)",
    fontsize=14,
    fontname="Franklin Gothic Book",
)
ax.text(
    labels[0],
    costs[0] / 2,
    "€ " + str(int(round((costs[0]) / 1e6, 0))) + " Mio.",
    zorder=20,
    ha="center",
    va="center",
    fontsize=12,
    fontname="Franklin Gothic Book",
)
ax.set_yticklabels(
    [str(e) + " Mio." for e in range(0, 120, 20)],
    fontsize=12,
    fontname="Franklin Gothic Book",
)
for ij in [1, 2, 4]:
    ax.text(
        labels[ij],
        costs[ij] + (costs[0] - costs[ij]) / 2,
        u"\u2212" + " € " + str(int(round((costs[0] - costs[ij]) / 1e6, 0))) + " Mio.",
        zorder=20,
        ha="center",
        va="center",
        fontsize=11,
        fontname="Franklin Gothic Book",
    )
plt.subplots_adjust(hspace=0.0)
# ax.set_ylim([0, 70e6])
# ax2.grid()
# l0 = ax2.plot(labels, spec_kW_costs, marker='o', linestyle='dotted', color='#004733', linewidth=2, label="€/kW")
# ax3 = ax2.twinx()
# l1 = ax3.plot(labels, spec_BEV_costs, marker='o', linestyle='dotted', color='#0096c7', linewidth=2, label="€/BEV")
# ax2.set_ylim([120, 480])
# ax3.set_ylim([0, 100])

# insert text descriptions
# for ij in range(0, 5):
#     ax2.text(labels[ij], spec_kW_costs[ij] + 40, "{:.2f}".format(spec_kW_costs[ij]), va='top', color='#004733', ha='left')
#     ax3.text(labels[ij], spec_BEV_costs[ij] - 10, "{:.2f}".format(spec_BEV_costs[ij]), va='bottom', color='#0096c7',
#              ha='right')
# ax3.spines["left"].set_color("#004733")  # setting up Y-axis tick color to red
# ax3.spines["right"].set_color("#0096c7")  # setting up Y-axis tick color to red
# ax2.tick_params(axis="y", colors="#004733")
# ax3.tick_params(axis="y", colors="#0096c7")
#
# ax2.set_ylabel("€/kW",  color="#004733", fontsize=14)
# ax3.set_ylabel("€/BEV", rotation=-90, color="#0096c7", fontsize=14)

# adding labels to graph
y_size = 490
b_box = dict(facecolor="white", edgecolor="white", boxstyle="round,pad=0.5")
#
# for ij in range(0, len(labels)):
#     ax2.text(labels[ij], y_size, labels[ij], ha='center', va='top', bbox=b_box, fontweight='extra bold')
ax.xaxis.set_ticks_position("top")
# lns = l0 + l1
lns2 = [l2] + [l3] + [l4]
labs = [l.get_label() for l in lns2]
ax.legend(lns2, labs, bbox_to_anchor=(1.01, -0.05), ncol=2)
# ax2.get_xaxis().set_ticks([])
# ax2.set_xticklabels(['' for e in range(0, len(labels))])
# ax1.xaxis.set_ticks_position('top')
# ax1.xaxis.set_label_position('top')
# ax1.xaxis.tick_top()
# ax1.xaxis.set_ticks_position('both')
# plt.setp(ax3.get_xticklabels(), visible=False)
# plt.setp(ax2.get_xticklabels(), visible=False)
ax.set_title(
    "Cost-reduction potentials in the GD scenario 2030\n",
    fontsize=15,
    fontname="Franklin Gothic Book",
)
# plt.show()
plt.savefig("figures/cost_red.pdf", bbox_inches="tight")

# ---------------------------------------------------------------------------------------------------------------------
# SENSITIVITY ANALYSIS I : DRIVING RANGE
# ---------------------------------------------------------------------------------------------------------------------

results = []

range_max = 1400
range_min = 200
step_size = 100
ranges = np.arange(range_min, range_max + step_size, step_size)

# create a dataframe with all Y values with columns of "100", "200", ...
base_df = pd.read_csv(path_SA_driving_range + list_of_paths_SA_driving_range[0])
df = pd.DataFrame()
# df['locations'] = base_df.POI_ID

nb_x = []

for ij in range(0, len(ranges)):
    temp_df = pd.read_csv(path_SA_driving_range + list_of_paths_SA_driving_range[ij])
    df[ranges[ij]] = temp_df.pYi_dir
    nb_x.append(temp_df.pXi.sum())
# nb_x.append(nb_x[-1])
df_to_plot = df.replace(0, np.nan)

# plot

fig, ax = plt.subplots(figsize=(8, 3.5))
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 12

font = {"family": "Franklin Gothic Book", "fontsize": 12}
ax2 = ax.twinx()
plt.xlim([range_min - step_size, range_max + step_size])
ax.tick_params(
    axis="x",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
)  # labels along the bottom edge are off

c = "#6096ba"
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 12

df_to_plot.boxplot(
    ax=ax,
    widths=(50),
    # notch=True,
    patch_artist=True,
    boxprops=dict(facecolor=c, color=c),
    capprops={"color": c, "linewidth": 2},
    whiskerprops={"color": c, "linewidth": 2},
    flierprops={"color": c, "markeredgewidth": 2, "markeredgecolor": c},
    medianprops=dict(color="red", linewidth=1.5),
    positions=ranges,
    labels=ranges,
)
ax.set_xlabel("driving range (km)", fontname="Franklin Gothic Book")
ax.set_yticklabels(labels=list(range(0, 55, 5)), fontname="Franklin Gothic Book")
ax2.set_xticklabels(labels=list(range(200, 1500, 100)), fontname="Franklin Gothic Book")
ax.set_xticklabels(labels=list(range(200, 1500, 100)), fontname="Franklin Gothic Book")

plt.grid("off")
ax.set_ylabel(
    "Nb. charging points per charging station", color="#1d3557", fontdict=font
)
ax2.set_ylabel(
    "Nb. charging stations", color="#723d46", rotation=-90, labelpad=12, fontsize=12
)
ax2.grid(False)
l0 = ax2.plot(
    list(ranges),
    nb_x,
    marker="o",
    color="#723d46",
    linewidth=2,
    label="Nb. of CS",
)
ax2.set_ylim([36, 66])
ax.set_ylim([0, 50])
l1 = ax.plot(
    [range_min - step_size, range_max + step_size],
    [int(12000 / 350)] * len([range_min - step_size, range_max + step_size]),
    linestyle="dotted",
    color="grey",
    linewidth=2,
    label="Max. nb. of CP at a CS",
)
ax.tick_params(axis="y", colors="#1d3557", labelsize=10)
ax2.tick_params(axis="y", colors="#723d46", labelsize=10)

ax2.spines["left"].set_color("#1d3557")
ax.spines["left"].set_color("#1d3557")  # setting up Y-axis tick color to red
ax2.spines["right"].set_color("#723d46")
ax2.spines["top"].set_visible(False)
ax.spines["top"].set_visible(False)
ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

blue_patch = mpatches.Patch(color=c, label="Distribution of nb. of CP at CS")
median_patch = Line2D([0], [0], color="red", lw=1.5)

lns = l0 + l1
labs = [l.get_label() for l in lns]
ax.legend(
    lns + [blue_patch] + [median_patch],
    labs + [blue_patch._label, "Median of CP at CS"],
    loc=2,
    fontsize=11,
    ncol=2,
)
plt.savefig("figures/driving_range_SA.pdf", bbox_inches="tight")
# ---------------------------------------------------------------------------------------------------------------------

# highway geometries
highway_geometries = pd.read_csv(r"geometries/highway_geometries_v6.csv")
highway_geometries["geometry"] = highway_geometries.geometry.apply(wkt.loads)
highway_geometries = gpd.GeoDataFrame(highway_geometries)
highway_geometries = highway_geometries.set_crs(reference_coord_sys)
highway_geometries["length"] = highway_geometries.geometry.length
segments_gdf = pd2gpd(pd.read_csv("data/highway_segments.csv"))

plot_highway_geometries = highway_geometries.to_crs("EPSG:3857")
# plot_austrian_border = austrian_border.to_crs("EPSG:3857")
plot_highway_geometries["null"] = [0] * len(plot_highway_geometries)


def merge_with_geom(results, energy):

    filtered_results = results[results[col_type] == "ra"]
    # osm geometries

    rest_areas = pd2gpd(
        pd.read_csv("data/projected_ras.csv"), geom_col_name="centroid"
    ).sort_values(by=["on_segment", "dist_along_highway"])
    rest_areas["segment_id"] = rest_areas["on_segment"]
    rest_areas[col_type_ID] = rest_areas["nb"]
    rest_areas[col_directions] = rest_areas["evaluated_dir"]

    # merge here
    results_and_geom_df = pd.merge(
        filtered_results, rest_areas, on=[col_segment_id, col_type_ID, col_directions]
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
        results_and_geom_df["total_charging_pole_number"] * energy
    )

    # plot
    plot_results_and_geom_df = results_and_geom_df.to_crs("EPSG:3857")
    # plot_results_and_geom_df = plot_results_and_geom_df[
    #     plot_results_and_geom_df.total_charging_pole_number > 0
    # ]
    plot_results_and_geom_df["x"] = plot_results_and_geom_df.geometry.x
    plot_results_and_geom_df["y"] = plot_results_and_geom_df.geometry.y
    return plot_results_and_geom_df


plot_sc_1 = merge_with_geom(val, charging_capacity)
merged = pd.merge(
    plot_sc_1, existing_infr, how="left", on=["segment_id", "name", "dir"]
)

results = []

# range_max = 1400
# range_min = 200
# step_size = 100

range_max = 1400
range_min = 200
step_size = 100
ranges = np.arange(range_min, range_max + step_size, step_size)

# create a dataframe with all Y values with columns of "100", "200", ...
base_df = pd.read_csv(path_SA_driving_range + list_of_paths_SA_driving_range[0])
df = pd.DataFrame()
# df['locations'] = base_df.POI_ID

nb_x = []
df["POI_ID"] = base_df["POI_ID"]
for ij in range(0, len(ranges)):
    temp_df = pd.read_csv(path_SA_driving_range + list_of_paths_SA_driving_range[ij])
    df[ranges[ij]] = temp_df.pXi
    nb_x.append(temp_df.pXi.sum())
# nb_x.append(nb_x[-1])
df_to_plot = df.replace(0, np.nan)

merge_2 = pd.merge(df_to_plot, merged, how="left", on=["POI_ID"])

# plot

occurances = []
for ij in range(0, len(merge_2)):
    l = 0
    for r in ranges:
        if df_to_plot[r].to_list()[ij] == 1:
            l = l + 1
    occurances.append(l)


p = gpd.read_file("geometries\output_BL.shp")
bd = p.to_crs("EPSG:3857")
merge_2["always_there"] = occurances
merge_3 = merge_2.copy()
merge_2 = merge_2[merge_2["always_there"] > 0]

max_occurance_val = max(occurances)
min_occurance_val = min(occurances)


colors = ["#04724d", "#d1495b", "#68b0ab"]

print("Nb. of steady PC:", len(merge_2))


fig = plt.figure(figsize=(15, 10))
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 11
gs = fig.add_gridspec(2)
axs = gs.subplots(sharex=True, sharey=True)
# plotting NUTS 2
geoms = bd.geometry.to_list()
names = bd["NAME"].to_list()
for ij in range(0, len(bd)):
    if geoms[ij].type == "MultiPolygon":
        for g in geoms[ij]:
            axs[0].plot(*g.exterior.xy, color="grey", linewidth=1)
            axs[1].plot(*g.exterior.xy, color="grey", linewidth=1)
    else:
        axs[0].plot(*geoms[ij].exterior.xy, color="grey", linewidth=1)
        axs[1].plot(*geoms[ij].exterior.xy, color="grey", linewidth=1)
    c = geoms[ij].centroid
    # axs[0].text(c.x, c.y + 0.03e6, names[ij], color="grey")
    # axs[1].text(c.x, c.y + 0.03e6, names[ij], color="grey")


# plotting highway network and Austrian boarder
plot_highway_geometries.plot(
    ax=axs[0], label="Austrian highway network", color="black", zorder=0, linewidth=1
)
plot_highway_geometries.plot(
    ax=axs[1], label="Austrian highway network", color="black", zorder=0, linewidth=1
)
axs[0].axis("off")
axs[1].axis("off")

# cat 0: "1-4"
plot_df = merge_2[merge_2["always_there"].isin(range(1, 5))]
axs[0].scatter(
    plot_df["x"],
    plot_df["y"],
    s=90,
    color="#ced4da",
    label="1-4 occurrences",
    zorder=20,
)

# cat 1: "5-9"
plot_df = merge_2[merge_2["always_there"].isin(range(5, 10))]
axs[0].scatter(
    plot_df["x"],
    plot_df["y"],
    s=90,
    color="#adb5bd",
    label="5-9 occurrences",
    zorder=20,
)

# cat 2: "10-12"
plot_df = merge_2[merge_2["always_there"].isin(range(10, 13))]
axs[0].scatter(
    plot_df["x"],
    plot_df["y"],
    s=90,
    color="#6c757d",
    label="10-12 occurrences",
    zorder=20,
)

# cat 3: "13"
plot_df = merge_2[merge_2["always_there"] == 13]
print(len(plot_df), " CP present in all")
axs[0].scatter(
    plot_df["x"],
    plot_df["y"],
    s=90,
    color="#212529",
    label="occurrence in every SA model run",
    zorder=20,
)

# axs[0].text(
#     1.06e6,
#     6.2e6,
#     "Constantly present CP locations during range SA",
#     bbox=dict(facecolor="none", edgecolor="#d9d9d9", boxstyle="round,pad=0.5"),
#     fontsize=11,
# )

axs[0].legend(loc="lower left", bbox_to_anchor=(0, 0.6, 1, 0), ncol=1, fancybox=True)
axs[0].set_title("Constantly present CS locations during SA on driving range")
# PLOT axis 1
# A: has charging station = YES
# B: always_there = max_occurances_value

# cat 0: A + B

cat = merge_3[
    (merge_3["has_charging_station"] == True) & (merge_3["always_there"] == 13)
]
print(len(cat), " CP also present in existing")

axs[1].scatter(
    cat["x"],
    cat["y"],
    s=90,
    color=colors[0],
    label="Part of existing infr. and constant in SA",
    zorder=15,
)

# cat 1: A
cat = merge_3[
    (merge_3["has_charging_station"] == True) & (merge_3["always_there"] != 13)
]

axs[1].scatter(
    cat["x"],
    cat["y"],
    s=90,
    color=colors[2],
    label="Part of existing infrastructure",
    zorder=20,
)

# cat: B
cat = merge_3[
    (merge_3["has_charging_station"] != True) & (merge_3["always_there"] == 13)
]

axs[1].scatter(
    cat["x"],
    cat["y"],
    s=90,
    color=colors[1],
    label="Constant occurrence during SA",
    zorder=20,
)

axs[1].legend(loc="lower left", bbox_to_anchor=(0, 0.67, 1, 0), ncol=1, fancybox=True)
plt.subplots_adjust(hspace=0.1)

# axs[1].text(
#     1.06e6,
#     6.2e6,
#     "Comparison to existing infrastructure",
#     bbox=dict(facecolor="none", edgecolor="#d9d9d9", boxstyle="round,pad=0.5"),
#     fontsize=11,
# )
axs[1].set_title("Comparison to existing infrastructure")

# plt.show()
plt.savefig("figures/steady_CS.pdf", bbox_inches="tight")


# ---------------------------------------------------------------------------------------------------------------------
# SENSITIVITY ANALYSIS II : SHARE OF BEVs in car fleet
# ---------------------------------------------------------------------------------------------------------------------
results = []

range_max = 100
range_min = 10
step_size = 10
ranges = np.arange(range_min, range_max + step_size, step_size)

# create a dataframe with all Y values with columns of "100", "200", ...
base_df = pd.read_csv(path_SA_share_BEV + list_of_paths_SA_share_BEV[0])
df = pd.DataFrame()
# df['locations'] = base_df.POI_ID

nb_x = []
nb_y = []
for ij in range(0, len(ranges)):
    temp_df = pd.read_csv(path_SA_share_BEV + list_of_paths_SA_share_BEV[ij])
    df[ranges[ij]] = temp_df.pYi_dir
    nb_x.append(temp_df.pXi.sum())
    nb_y.append(temp_df.pYi_dir.sum())
# nb_x.append(nb_x[-1])
df_to_plot = df.replace(0, np.nan)

# plot

fig, ((ax3, ax1), (ax4, ax2)) = plt.subplots(nrows=2, ncols=2, figsize=(8, 6.5))

plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 12
plt.rcParams["figure.autolayout"] = True
# ax2 = ax1.twinx()
plt.xlim([range_min - step_size, range_max + step_size])
ax1.tick_params(
    axis="x",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
)  # labels along the bottom edge are off

c = "#457b9d"


ax1.set_xlabel("BEV share (%)", fontname="Franklin Gothic Book", fontsize=12)
ax1.grid()

ax1.set_ylabel(
    "Nb. of charging stations",
    labelpad=12,
    fontname="Franklin Gothic Book",
    fontsize=12,
)
ax1.set_ylim([min(nb_x) - 2, max(nb_x) + 2])


l0 = ax1.plot(
    list(ranges),
    nb_x,
    marker="o",
    color="#723d46",
    linewidth=2,
    label="Nb. charging stations",
)


ax1.yaxis.set_major_locator(MaxNLocator(integer=True))


lns = l0
labs = [l.get_label() for l in lns]
# ax1.legend(lns, labs, loc=0, fontsize=10)
# ax2.set_xticks(ranges)
ax1.set_xticks(ranges)
# plt.show()

l1 = ax4.plot(
    [range_min - step_size, range_max + step_size],
    [int(12000 / 350)] * len([range_min - step_size, range_max + step_size]),
    linestyle="dotted",
    color="grey",
    linewidth=2,
    label="Max. nb. of CP at a CS",
)
sens_data = pd.read_csv(
    "sensitivity_analyses/sensitivity_anal_epsilon_part_SC_1002.csv"
)

perc_non_covered = np.array(sens_data.perc_not_charged) * 100
installed_caps = np.array(sens_data.nb_poles) * (350 / 1000)

# plot 2
# fig, ax = plt.subplots(figsize=(8, 3.5))

# ax2 = ax.twinx()
# ax2.set_ylim([0,30])
l2 = ax3.bar(
    list(ranges),
    installed_caps,
    width=8,
    zorder=20,
    color="#f4a261",
    label="Total capacity",
)
# ax2.plot([0] + list(ranges) + [110], list(perc_non_covered) + [perc_non_covered[0]]*2 , color='#233d4d', linestyle='--', linewidth=2)
ax3.set_xlim([range_min - step_size, range_max + step_size])
ax3.tick_params(
    axis="x",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
)  # labels along the bottom edge are off

c = "#457b9d"

ax3.set_xlabel("BEV share (%)", fontsize=12, fontname="Franklin Gothic Book")
ax3.grid("off")
ax3.grid(True)
ax4.grid(True)

ax3.set_ylabel("Total capacity (MW)", fontname="Franklin Gothic Book", fontsize=12)

ax3.tick_params(axis="y", labelsize=10)

ax3.yaxis.set_major_locator(MaxNLocator(integer=True))
# ax2.set_xticks(ranges)
ax3.set_xticks(ranges)
# ax2.grid(False)
# ax.grid(False)
ax4.annotate(
    "",
    xy=(0, -0.25),
    xycoords="axes fraction",
    xytext=(1, -0.25),
    arrowprops=dict(arrowstyle="-", color="grey", linewidth=2),
)
ax2.annotate(
    "",
    xy=(0, 1),
    xycoords="axes fraction",
    xytext=(1, 1),
    arrowprops=dict(arrowstyle="-", color="grey", linewidth=2),
)

ax4.annotate(
    "|",
    fontsize=20,
    xy=(0.02, -0.25),
    color="grey",
    xycoords="axes fraction",
    va="center",
    ha="center",
)
ax4.annotate(
    "|",
    fontsize=20,
    xy=(0.3, -0.25),
    color="grey",
    xycoords="axes fraction",
    va="center",
    ha="center",
)
ax4.annotate(
    "|",
    fontsize=20,
    xy=(0.918, -0.25),
    color="grey",
    xycoords="axes fraction",
    va="center",
    ha="center",
)

ax2.annotate(
    "|",
    fontsize=20,
    xy=(0.02, 1),
    color="grey",
    xycoords="axes fraction",
    va="center",
    ha="center",
)
ax2.annotate(
    "|",
    fontsize=20,
    xy=(0.3, 1),
    color="grey",
    xycoords="axes fraction",
    va="center",
    ha="center",
)
ax2.annotate(
    "|",
    fontsize=20,
    xy=(0.918, 1),
    color="grey",
    xycoords="axes fraction",
    va="center",
    ha="center",
)

offset1 = -0.35
offset = 1 - 0.1
ax4.annotate(
    "2020",
    fontsize=11,
    xy=(0.02, offset1),
    color="grey",
    xycoords="axes fraction",
    va="center",
    ha="center",
)
ax4.annotate(
    "2030",
    fontsize=11,
    xy=(0.3, offset1),
    color="grey",
    xycoords="axes fraction",
    va="center",
    ha="center",
)
ax4.annotate(
    "2040",
    fontsize=11,
    xy=(0.918, offset1),
    color="grey",
    xycoords="axes fraction",
    va="center",
    ha="center",
)
ax2.annotate(
    "2020",
    fontsize=11,
    xy=(0.02, offset),
    color="grey",
    xycoords="axes fraction",
    va="center",
    ha="center",
)
ax2.annotate(
    "2030",
    fontsize=11,
    xy=(0.3, offset),
    color="grey",
    xycoords="axes fraction",
    va="center",
    ha="center",
)
ax2.annotate(
    "2040",
    fontsize=11,
    xy=(0.918, offset),
    color="grey",
    xycoords="axes fraction",
    va="center",
    ha="center",
)
ax4.annotate(
    "year",
    fontsize=12,
    xy=(-0.1, -0.25),
    color="grey",
    xycoords="axes fraction",
    va="center",
    ha="center",
)
ax2.annotate(
    "year",
    fontsize=12,
    xy=(-0.1, 1),
    color="grey",
    xycoords="axes fraction",
    va="center",
    ha="center",
)

ax1.set_yticklabels(labels=list(range(39, 72, 3)), fontname="Franklin Gothic Book")
ax3.set_yticklabels(labels=list(range(0, 750, 80)), fontname="Franklin Gothic Book")
ax1.set_xticklabels(labels=list(range(10, 110, 10)), fontname="Franklin Gothic Book")
ax3.set_xticklabels(labels=list(range(10, 110, 10)), fontname="Franklin Gothic Book")
# ax2.set_xticklabels(labels=list(range(10, 110, 10)), fontname="Franklin Gothic Book")
ax4.set_xticklabels(labels=list(range(10, 110, 10)), fontname="Franklin Gothic Book")

ax2.set_yticklabels(
    labels=[str(el) for el in [0, 50, 100, 150, 200, 250]],
    fontname="Franklin Gothic Book",
)
ax4.set_yticklabels(labels=[0, 10, 20, 30, 40], fontname="Franklin Gothic Book")

# boxplots
c = "#6096ba"
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 12

l3 = df_to_plot.boxplot(
    ax=ax4,
    widths=(6),
    # notch=True,
    patch_artist=True,
    boxprops=dict(facecolor=c, color=c),
    capprops={"color": c, "linewidth": 2},
    whiskerprops={"color": c, "linewidth": 2},
    flierprops={"color": c, "markeredgewidth": 2, "markeredgecolor": c},
    medianprops=dict(color="red", linewidth=1.5),
    positions=ranges,
    labels=ranges,
)
blue_patch = mpatches.Patch(color=c, label="Distribution of nb. of CP at CS")
median_patch = Line2D([0], [0], color="red", lw=1.5)

ax4.set_ylim([0, 40])
sens_data["perc"] = range(10, 110, 10)
sens_data = sens_data.set_index("perc")


x_min, x_max = ax3.get_xlim()
ax4.set_xlim([x_min, x_max])
ax1.set_xlim([x_min, x_max])


# legend
lns = [l2] + l0 + l1
labs = [l.get_label() for l in lns]
ax2.legend(
    lns + [blue_patch] + [median_patch],
    labs + [blue_patch._label, "Median of CP at CS"],
    bbox_to_anchor=(-0.1, -0.3, 1, 1),
    fontsize=12,
)

ax4.set_ylabel(
    "Nb. of charging points per charging station",
    fontname="Franklin Gothic Book",
    fontsize=12,
)
ax2.axis("off")
plt.tight_layout()
plt.savefig("figures/share_BEV_SA.pdf", bbox_inches="tight")
