# import the output
# then transform it
# later visualize the waiting times + (max. 1h!)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

key_indicators = pd.read_csv("data\mc_sim_cell_utility_3.csv")
cells = pd.read_csv("data/cell_routes.csv")
segments_gdf = pd2gpd(pd.read_csv("geography/segments.csv"))
segments_gdf = segments_gdf[~segments_gdf.ID.isin([0, 1, 2])]
cell_info = pd.read_csv("data/20220429-152856_cells_input.csv")
cell_info = cell_info[cell_info["has_cs"]]
cells["cells"] = [eval(el) for el in cells["cells"].to_list()]
cell_with_cap = cell_info.cell_id.to_list()

segs = cell_info.seg_id.to_list()
# make point features for visualization
geoms = []
for ij in range(0, len(cell_with_cap)):
    cell_id = cell_with_cap[ij]
    seg_id = segs[ij]
    geom = segments_gdf[segments_gdf.ID == seg_id].geometry.to_list()[0]
    # find Abschnitt
    extract_cells = cells[cells.segment_id == seg_id]

    cell_lists = extract_cells.cells.to_list()
    ids = extract_cells.index.to_list()

    # find index in cell array
    for ij in range(0, len(ids)):
        if cell_id in cell_lists[ij]:
            idx = ij
            break

    belonging_teilsegment = cells.loc[ids[idx]]
    cell_id_on_bt = cell_lists[idx].index(cell_id)

    # no evaluate point
    dist_on_segment = (
        cell_id_on_bt * 25 * 1000 + belonging_teilsegment.dist_start + 12500
    )
    point = geom.interpolate(dist_on_segment)
    geoms.append(point)

cell_info["geometry"] = geoms
cell_info = gpd.GeoDataFrame(cell_info, geometry="geometry", crs="EPSG:31287")
# waiting times
key_indicators_extract = key_indicators[key_indicators.cell_id.isin(cell_with_cap)]
key_indicators_extract = key_indicators_extract.fillna(0.0)
peaks_diff = []
load_factors = []
average_wait_time = []
peak_times = []
# get delta to peak
for c in cell_with_cap:
    cap = cell_info[cell_info.cell_id == c].capacity.to_list()[0]
    extr = key_indicators_extract[key_indicators_extract.cell_id == c]
    peak = extr.max_load.max() * 0.25
    lf = extr.load_factor.max()
    wt = extr.peak_wait.max()
    av_wt = extr.wait_times.mean()
    peaks_diff.append(abs(round((cap - peak) / cap * 100)))

    if lf > 1:
        load_factors.append(lf - 1)
    else:
        load_factors.append(lf)

    average_wait_time.append(round(av_wt * 0.25 / (0.25)) * 15)
    peak_times.append(round(extr.peak_wait.max() * 0.25 / 0.25) * 15)

# add these column
# filter values
# make graphic
average_wait_time = np.array(average_wait_time)
peak_times = np.array(peak_times)
peaks_diff = np.array(peaks_diff)
average_wait_time = np.where(average_wait_time > 30, 30, average_wait_time)
peak_times = np.where(peak_times > 45, 45, peak_times)
peaks_diff = np.where(peaks_diff > 100, 100, peaks_diff)
peaks_diff = [
    22,
    27,
    60,
    0,
    0,
    94,
    100,
    100,
    100,
    78,
    0,
    0,
    81,
    88,
    75,
    100,
    0,
    50,
    100,
    100,
    33,
    32,
    21,
    2,
    46,
    100,
    100,
]
cell_info["av_wait_time"] = average_wait_time
cell_info["peak_wait_time"] = peak_times
cell_info["peaks_diff"] = peaks_diff
cell_info["load_factors"] = load_factors

nuts_geoms = gpd.read_file("geography/NUTS_RG_03M_2021_3857.shp")
munic_geoms = gpd.read_file("geography/STATISTIK_AUSTRIA_GEM_20220101.shp")

nuts_geoms_at = nuts_geoms[nuts_geoms["CNTR_CODE"] == "AT"]
nuts_geoms_at_3 = nuts_geoms_at[nuts_geoms_at["LEVL_CODE"] == 3]
nuts_geoms_at_3 = nuts_geoms_at_3.to_crs("EPSG:31287")

colors1 = ["#fcbf49", "#f77f00", "#d62828", "#003049"]
colors2 = ["#02c39a", "#00a896", "#028090", "#05668d"]
colors3 = ["#e5dada", "#e59500", "#e59500", "#002642"]
colors4 = ["#9b2915", "#e9b44c", "#e4d6a7", "#1c110a"]

exclude = ["AT342", "AT341", "AT331", "AT332", "AT335", "AT334"]

nuts_geoms_at_3 = nuts_geoms_at_3[~nuts_geoms_at_3.NUTS_ID.isin(exclude)]
cell_info["x"] = cell_info.geometry.x
cell_info["y"] = cell_info.geometry.y


fig, ax = plt.subplots()
colors = colors3
nuts_geoms_at_3["geometry"] = nuts_geoms_at_3.exterior
nuts_geoms_at_3.plot(ax=ax, color="lightgrey")
segments_gdf.plot(ax=ax, color="black")
# cell_info.plot(ax=ax, color=colors[0])


# av waiting time
# class 1
c1 = 0
c2 = 15
c3 = 30
c4 = 45
class1 = cell_info[(cell_info["peaks_diff"] >= 0) & (cell_info["peaks_diff"] < 25)]
class2 = cell_info[(cell_info["peaks_diff"] >= 25) & (cell_info["peaks_diff"] < 50)]
class3 = cell_info[(cell_info["peaks_diff"] >= 50) & (cell_info["peaks_diff"] < 75)]
class4 = cell_info[(cell_info["peaks_diff"] >= 75) & (cell_info["peaks_diff"] <= 100)]

ax.scatter(
    class1["x"].to_list(),
    class1["y"].to_list(),
    s=100,
    color=colors[0],
    label="0-25%",
    zorder=10,
)
# class1.plot(ax=ax,marker_size=100,label="0 min", zorder=10)
# class2.plot(ax=ax,marker_size=100,label="15 min", zorder=10)
# class3.plot(ax=ax,marker_size=100,label="30 min", zorder=10)

ax.scatter(
    class2["x"].to_list(),
    class2["y"].to_list(),
    s=100,
    color=colors[1],
    label="25-50%",
    zorder=10,
)

ax.scatter(
    class3["x"].to_list(),
    class3["y"].to_list(),
    s=100,
    color=colors[2],
    label="50-75%",
    zorder=10,
)
ax.scatter(
    class4["x"].to_list(),
    class4["y"].to_list(),
    s=100,
    color=colors[3],
    label="75-100%",
    zorder=10,
)
ax.axis("off")
ax.legend()
plt.tight_layout()
plt.savefig("images/peak_diff.svg")
plt.show()
# for g in
