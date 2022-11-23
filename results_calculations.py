"""
Created on 21.07.2022

calculation of KPIs and stuff

"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import geopandas as gpd
import numpy as np
from utils import *

# input files    -> total travels, travelled km,
winter_workday_input = read_fleets(pd.read_csv("data/winter_workdayfleet_input_20220722.csv", delimiter=";"))
summer_workday_input = read_fleets(pd.read_csv("data/winter_workdayfleet_input_20220722.csv", delimiter=";"))
winter_holiday_input = read_fleets(pd.read_csv("data/winter_holidayfleet_input_20220722.csv", delimiter=";"))
summer_holiday_input = read_fleets(pd.read_csv("data/winter_holidayfleet_input_20220722.csv", delimiter=";"))
winter_workday_input["len"] = winter_workday_input.route.apply(len)
winter_workday_input = winter_workday_input[winter_workday_input.len > 3]
summer_workday_input["len"] = summer_workday_input.route.apply(len)
summer_workday_input = summer_workday_input[summer_workday_input.len > 3]
winter_holiday_input["len"] = winter_holiday_input.route.apply(len)
winter_holiday_input = winter_holiday_input[winter_holiday_input.len > 3]
summer_holiday_input["len"] = summer_holiday_input.route.apply(len)
summer_holiday_input = summer_holiday_input[summer_holiday_input.len > 3]

# result files  -> average utility rate, av. diff to installed cap, av. arrival SOC, queue length = objective value
winter_workday = pd.read_csv("results/20220723-233215_charging_stationswinter_workdayfleet_input_20220722_compressed_probe2.csv")
summer_workday = pd.read_csv("results/20220724-033029_charging_stationssummer_workdayfleet_input_20220722_compressed_probe2.csv")
winter_holiday = pd.read_csv("results/20220724-083509_charging_stationswinter_holidayfleet_input_20220722_compressed_probe2.csv")
summer_holiday = pd.read_csv("results/20220724-114442_charging_stationssummer_holidayfleet_input_20220722_compressed_probe2.csv")
# winter_workday = pd.read_csv("results/20220724-142921_charging_stationswinter_workdayfleet_input_20220722_compressed_probe2.csv")

# TODO: delete this later
winter_workday = winter_workday.drop(index=42)
summer_workday = summer_workday.drop(index=42)
winter_holiday = winter_holiday.drop(index=42)
summer_holiday = summer_holiday.drop(index=42)
ww_fleet = pd.read_csv("results/20220723-233215_fleet_infoswinter_workdayfleet_input_20220722_compressed_probe2.csv")
sw_fleet = pd.read_csv("results/20220724-033029_fleet_infossummer_workdayfleet_input_20220722_compressed_probe2.csv")
wh_fleet = pd.read_csv("results/20220724-083509_fleet_infoswinter_holidayfleet_input_20220722_compressed_probe2.csv")
sh_fleet = pd.read_csv("results/20220724-114442_fleet_infossummer_holidayfleet_input_20220722_compressed_probe2.csv")
# ww_fleet = pd.read_csv("results/20220724-142921_fleet_infoswinter_workdayfleet_input_20220722_compressed_probe2.csv")


_mu = 250/350
T = 120
_queue_lengths = []
_total_e_charged = [0, 0, 0, 0]
indices = winter_workday.index.to_list()
for ij in range(0, len(indices)):
    id = indices[ij]
    cap = winter_workday.at[id, "capacity"]
    _energy_charged_ww = [winter_workday.at[id, "E charged at t=" + str(t)] *(1/_mu) for t in range(0, T)]
    _total_energy_charged_ww = sum(_energy_charged_ww)
    _maximum_energy_charged_ww = max(_energy_charged_ww)
    winter_workday.at[id, "utility_rate"] = _total_energy_charged_ww / (cap * (T / 4))
    winter_workday.at[id, "diff_to_max"] = (cap - _maximum_energy_charged_ww / 0.25)
    winter_workday.at[id, "total_e_charged"] = _total_energy_charged_ww

    _total_e_charged[0] = _total_e_charged[0] + _total_energy_charged_ww
    _objective_value_ww = [winter_workday.at[id, "waiting at t=" + str(t)] for t in range(0, T)]
    _queue_lengths.append(sum(_objective_value_ww))

    _objective_value_sw = [summer_workday.at[id, "waiting at t=" + str(t)]  for t in range(0, T)]
    _energy_charged_sw = [summer_workday.at[id, "E charged at t=" + str(t)]*(1/_mu) for t in range(0, T)]
    _total_energy_charged_sw  = sum(_energy_charged_sw )
    _maximum_energy_charged_sw  = max(_energy_charged_sw )
    summer_workday.at[id, "utility_rate"] = _total_energy_charged_sw  / (cap * (T / 4))
    summer_workday.at[id, "diff_to_max"] = (cap - _maximum_energy_charged_sw / 0.25)
    _queue_lengths.append(sum(_objective_value_sw))
    _total_e_charged[1] = _total_e_charged[1] + _total_energy_charged_sw

    _objective_value_wh = [winter_holiday.at[id, "waiting at t=" + str(t)] for t in range(0, T)]
    _energy_charged_wh = [winter_holiday.at[id, "E charged at t=" + str(t)]*(1/_mu) for t in range(0, T)]
    _total_energy_charged_wh  = sum(_energy_charged_wh )
    _maximum_energy_charged_wh  = max(_energy_charged_wh )
    winter_holiday.at[id, "utility_rate"] = _total_energy_charged_wh  / (cap * (T / 4))
    winter_holiday.at[id, "diff_to_max"] = (cap - _maximum_energy_charged_wh / 0.25)
    _queue_lengths.append(sum(_objective_value_wh))
    _total_e_charged[2] = _total_e_charged[2] + _total_energy_charged_wh

    _objective_value_sh = [summer_holiday.at[id, "waiting at t=" + str(t)] for t in range(0, T)]
    _energy_charged_sh = [summer_holiday.at[id, "E charged at t=" + str(t)]*(1/_mu) for t in range(0, T)]
    _total_energy_charged_sh = sum(_energy_charged_sh)
    _maximum_energy_charged_sh = max(_energy_charged_sh)
    summer_holiday.at[id, "utility_rate"] = _total_energy_charged_sh/ (cap * (T / 4))
    summer_holiday.at[id, "diff_to_max"] = (cap - _maximum_energy_charged_sh / 0.25)
    _queue_lengths.append(sum(_objective_value_sh))
    _total_e_charged[3] = _total_e_charged[3] + _total_energy_charged_sh


winter_workday["diff_to_max"] = np.where(winter_workday["diff_to_max"] < 0, 0, winter_workday["diff_to_max"])
winter_workday["utility_rate"] = np.where(winter_workday["utility_rate"] < 0, 0, winter_workday["utility_rate"])

summer_workday["diff_to_max"] = np.where(summer_workday["diff_to_max"] < 0, 0, summer_workday["diff_to_max"])
summer_workday["utility_rate"] = np.where(summer_workday["utility_rate"] < 0, 0, summer_workday["utility_rate"])

winter_holiday["diff_to_max"] = np.where(winter_holiday["diff_to_max"] < 0, 0, winter_holiday["diff_to_max"])
winter_holiday["utility_rate"] = np.where(winter_holiday["utility_rate"] < 0, 0, winter_holiday["utility_rate"])

summer_holiday["diff_to_max"] = np.where(summer_holiday["diff_to_max"] < 0, 0, summer_holiday["diff_to_max"])
summer_holiday["utility_rate"] = np.where(summer_holiday["utility_rate"] < 0, 0, summer_holiday["utility_rate"])

print("---------------------------------------------------------------------------------------------------------------")
print("Result summary")
print("---------------------------------------------------------------------------------------------------------------")
print("repr day: | Weekday in winter | Weekday in summer | Holiday in winter | Holiday in summer")
print("sum long-distance travels", winter_workday_input.fleet_size.sum(),"|", summer_workday_input.fleet_size.sum(),
      "|", winter_holiday_input.fleet_size.sum(), "|", summer_holiday_input.fleet_size.sum())
print("consumed energy |", 0.2 * 27.5 * sum([winter_workday_input.len.to_list()[ij] * winter_workday_input.fleet_size.to_list()[ij] for ij in range(0, len(winter_workday_input))]), "|", 0.15 * 27.5 * sum([summer_workday_input.len.to_list()[ij] * summer_workday_input.fleet_size.to_list()[ij] for ij in range(0, len(summer_workday_input))]),
      "|",  0.2 * 27.5 * sum([winter_holiday_input.len.to_list()[ij] * winter_holiday_input.fleet_size.to_list()[ij] for ij in range(0, len(winter_holiday_input))]), "|", 0.15 * 27.5 * sum([summer_holiday_input.len.to_list()[ij] * summer_holiday_input.fleet_size.to_list()[ij] for ij in range(0, len(summer_holiday_input))]))
print("av. arrival SOC |", np.sum(ww_fleet.arrival_SOC.array * ww_fleet.incoming.array)/ww_fleet.incoming.sum(),"|",np.sum(sw_fleet.arrival_SOC.array * sw_fleet.incoming.array)/sw_fleet.incoming.sum(),
      "|", np.sum(wh_fleet.arrival_SOC.array * wh_fleet.incoming.array)/wh_fleet.incoming.sum(), "|",  np.sum(sh_fleet.arrival_SOC.array * sh_fleet.incoming.array)/sh_fleet.incoming.sum())
print("av. utility rate |", winter_workday.utility_rate.mean(),"|", summer_workday.utility_rate.mean(),
      "|", winter_holiday.utility_rate.mean(), "|", summer_holiday.utility_rate.mean())
print("av. diff_to_max |",winter_workday.diff_to_max.mean(), winter_workday.diff_to_max.mean(), winter_workday[~winter_workday.index.isin([15])].diff_to_max.mean(),
      "|",summer_workday[~summer_workday.index.isin([15])].diff_to_max.sum(), summer_workday.diff_to_max.mean(), summer_workday.diff_to_max.sum(),"|",winter_holiday[~winter_holiday.index.isin([15])].diff_to_max.mean(), winter_holiday.diff_to_max.mean(), "|", summer_holiday[~summer_holiday.index.isin([15])].diff_to_max.mean(), summer_holiday.diff_to_max.mean())
print("obj. value |", _queue_lengths[0], "|", _queue_lengths[1],  "|",_queue_lengths[2],  "|",_queue_lengths[3] )
print("total e charged |", _total_e_charged[0] * _mu, "|", _total_e_charged[1]* _mu,  "|", _total_e_charged[2]* _mu,  "|", _total_e_charged[3]* _mu )









