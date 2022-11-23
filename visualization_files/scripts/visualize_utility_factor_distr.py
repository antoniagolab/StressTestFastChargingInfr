import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import geopandas as gpd
from shapely import wkt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import ColorConverter

# enter here always the day with highest consumption
winter_workday = pd.read_csv("results/20220723-233215_charging_stationswinter_workdayfleet_input_20220722_compressed_probe2.csv")
summer_workday = pd.read_csv("results/20220724-033029_charging_stationssummer_workdayfleet_input_20220722_compressed_probe2.csv")
winter_holiday = pd.read_csv("results/20220724-083509_charging_stationswinter_holidayfleet_input_20220722_compressed_probe2.csv")
summer_holiday = pd.read_csv("results/20220724-114442_charging_stationssummer_holidayfleet_input_20220722_compressed_probe2.csv")
## TODO: delete this later
winter_workday = winter_workday.drop(index=42)
summer_workday = summer_workday.drop(index=42)
winter_holiday = winter_holiday.drop(index=42)
summer_holiday = summer_holiday.drop(index=42)

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


# calculate load factor for each charging station and difference between peak load and installed capacity

indices = winter_workday.index.to_list()
for ij in range(0, len(indices)):
    id = indices[ij]
    cap = winter_workday.at[id, "capacity"]
    _energy_charged_ww = [winter_workday.at[id, "E charged at t=" + str(t)] *(1/_mu) for t in range(0, T)]
    _total_energy_charged_ww = sum(_energy_charged_ww)
    _maximum_energy_charged_ww = max(_energy_charged_ww)
    winter_workday.at[id, "utility_rate"] = _total_energy_charged_ww / (cap * (T / 4))
    winter_workday.at[id, "diff_to_max"] = (cap - _maximum_energy_charged_ww / 0.25) / cap

    _energy_charged_sw = [summer_workday.at[id, "E charged at t=" + str(t)]*(1/_mu) for t in range(0, T)]
    _total_energy_charged_sw  = sum(_energy_charged_sw )
    _maximum_energy_charged_sw  = max(_energy_charged_sw )
    summer_workday.at[id, "utility_rate"] = _total_energy_charged_sw  / (cap * (T / 4))
    summer_workday.at[id, "diff_to_max"] = (cap - _maximum_energy_charged_sw / 0.25) / cap

    _energy_charged_wh = [winter_holiday.at[id, "E charged at t=" + str(t)]*(1/_mu) for t in range(0, T)]
    _total_energy_charged_wh  = sum(_energy_charged_wh )
    _maximum_energy_charged_wh  = max(_energy_charged_wh )
    winter_holiday.at[id, "utility_rate"] = _total_energy_charged_wh  / (cap * (T / 4))
    winter_holiday.at[id, "diff_to_max"] = (cap - _maximum_energy_charged_wh / 0.25) / cap

    _energy_charged_sh = [summer_holiday.at[id, "E charged at t=" + str(t)]*(1/_mu) for t in range(0, T)]
    _total_energy_charged_sh = sum(_energy_charged_sh)
    _maximum_energy_charged_sh = max(_energy_charged_sh)
    summer_holiday.at[id, "utility_rate"] = _total_energy_charged_sh/ (cap * (T / 4))
    summer_holiday.at[id, "diff_to_max"] = (cap - _maximum_energy_charged_sh / 0.25) / cap

winter_workday["diff_to_max"] = np.where(winter_workday["diff_to_max"] < 0, 0, winter_workday["diff_to_max"])
winter_workday["utility_rate"] = np.where(winter_workday["utility_rate"] < 0, 0, winter_workday["utility_rate"])

summer_workday["diff_to_max"] = np.where(summer_workday["diff_to_max"] < 0, 0, summer_workday["diff_to_max"])
summer_workday["utility_rate"] = np.where(summer_workday["utility_rate"] < 0, 0, summer_workday["utility_rate"])

winter_holiday["diff_to_max"] = np.where(winter_holiday["diff_to_max"] < 0, 0, winter_holiday["diff_to_max"])
winter_holiday["utility_rate"] = np.where(winter_holiday["utility_rate"] < 0, 0, winter_holiday["utility_rate"])

summer_holiday["diff_to_max"] = np.where(summer_holiday["diff_to_max"] < 0, 0, summer_holiday["diff_to_max"])
summer_holiday["utility_rate"] = np.where(summer_holiday["utility_rate"] < 0, 0, summer_holiday["utility_rate"])

colors = ["#118ab2", "#06d6a0", "#ffd166", "#ef476f"]

fig, ax = plt.subplots(figsize=(5, 5))
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 12
data = pd.DataFrame()

data["Winter workday"] = winter_workday["utility_rate"]
data["Summer workday"] = summer_workday["utility_rate"]
data["Winter holiday"] = winter_holiday["utility_rate"]
data["Summer holiday"] = summer_holiday["utility_rate"]

all_round_color = "#6c757d"
bplot = ax.boxplot(
    data,
    widths=(0.3),
    # notch=True,
    patch_artist=True,
    boxprops=dict(facecolor=all_round_color, color=all_round_color,),
    capprops={"color": all_round_color, "linewidth": 2},
    whiskerprops={"color": all_round_color, "linewidth": 2},
    flierprops={"color": all_round_color, "markeredgewidth": 2, "markeredgecolor": all_round_color},
    medianprops=dict(color='#9d0208', linewidth=1.5),
    labels = ["Workday\nin winter", "Workday\nin summer", "Holiday\nin winter", "Holiday\nin summer"],
)
    # labels=["0", "1"])  # will be used to label x-ticks
    # positions=[0, 1],
ax.set_ylabel("Utility factor", fontsize=10)
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)
ax.set_title("Comparison of utility factor distributions", fontsize=12)

for patch, color in zip(bplot['boxes'], colors):
    color_with_alpha = ColorConverter.to_rgba(
        color, alpha=0.7)
    patch.set_facecolor(color_with_alpha)
ax.set_ylim([0, 1])
median_patch = Line2D([0], [0], color='#9d0208', lw=1.5)
#
ax.legend([median_patch], ['Median'], loc="upper right")
# ax.legend([median_patch],
#            bbox_to_anchor=(1, 1), fontsize=12)
#    labels="winter workday",
#
# summer_workday.boxplot(
#     column="Summer workday",
#     ax=ax,
#     widths=(50),
#     # notch=True,
#     patch_artist=True,
#     boxprops=dict(facecolor=colors[1], color=colors[1], alpha=0.7,),
#     capprops={"color": colors[1], "linewidth": 2, "alpha": 0.7},
#     whiskerprops={"color": colors[1], "linewidth": 2, "alpha": 0.7},
#     flierprops={"color": colors[1], "markeredgewidth": 2, "markeredgecolor": colors[1]},
#     medianprops=dict(color='red', linewidth=1.5 ),
#     position=1
# #    labels="winter workday",
# )
# ax.set_xlabel("Representative day", fontname="Franklin Gothic Book")
plt.grid("off")
plt.savefig("figures/utility_rate_comp.pdf", bbox_inches="tight")
plt.savefig("figures/utility_rate_comp.svg", bbox_inches="tight")