"""
script for creating the input files for the different scenarios for the optimization which describes the fleet properties

TODO:
    (1) get for each OD-pair, the amount of travellers  --check
    (2) determine tau for each and assign it in the table   --check
    (3) delete fleet entries with size == 0 --check
    (4) get fleet entries in input design

"""
epsilon = 0.3
a = 1
# is_working_day = True
# is_winter = False
import pickle
from utils import *
from _start_time_distributions import *
for (is_working_day, is_winter) in [(True, True), (False, True)]:
# for (is_working_day, is_winter) in [(True, True)]:
    if is_winter and is_working_day:
        case_study_name = "winter_workday"
    elif is_winter and not is_working_day:
        case_study_name = "winter_holiday"
    elif not is_winter and is_working_day:
        case_study_name = "summer_workday"
    else:
        case_study_name = "summer_holiday"

    print("This case is:", case_study_name)
    T = 4 * 24

    with open("data\paths_encoded_matching_route.pickle", "rb") as handle:
        paths_encoded_matching_route = pickle.load(handle)

    fleet_types = pd.read_csv("data/route_og.csv")
    fleet_types["cell_route"] = [eval(r) for r in fleet_types["cell_route"].to_list()]

    if is_working_day:
        with open("data/total_travels_od.pickle", "rb") as handle:
            traffic_flow = pickle.load(handle)
    else:
        with open("data/total_travels_od_holiday.pickle", "rb") as handle:
            traffic_flow = pickle.load(handle)

    # with open("data/traffic_load_od.pickle", "rb") as handle:
    #     traffic_flow = pickle.load(handle)

    fleet_types["fleet_size"] = [0.0] * len(fleet_types)
    origins = fleet_types.origin.to_list()
    dests = fleet_types.destination.to_list()
    ods = list(set([(origins[ij], dests[ij]) for ij in range(0, len(fleet_types))]))

    assigned_fleet_size = 0
    for k in traffic_flow.keys():
        origin = k[0]
        destination = k[1]
        traffic_load = traffic_flow[k]
        bevs = traffic_load * epsilon * a
        not_assigned = 0
        if k in ods and not origin == destination:
            for vehicle in range(0, int(bevs)):
                # classify purpose, then if back or forth, then tau
                if working_day:
                    purpose_distribution = working_day
                else:
                    purpose_distribution = weekend_day

                purpose_code = np.random.choice(list(purpose_distribution.keys()), p=fix_p(np.array(list(purpose_distribution.values()))))
                is_first_way = np.random.choice([True, False])
                if is_first_way:
                    start_time_distr = route_one
                else:
                    start_time_distr = route_two
                distr = start_time_distr[int(purpose_code)]
                random_time = np.random.normal(distr[0], distr[1]) * 4
                ass = False
                for t in range(0, T, 4):
                    if (random_time >= t) and (random_time < t+4):
                        tau = t
                        # print(origin, destination, tau)
                        extract_df = fleet_types[(fleet_types.tau == tau) & (fleet_types.origin == origin)
                                                 & (fleet_types.destination == destination)]
                        fleet_types.loc[extract_df.index.to_list()[0], "fleet_size"] = fleet_types.loc[extract_df.index.to_list()[0], "fleet_size"] + 1
                        # assigned_fleet_size = assigned_fleet_size + 1
                        # ass = True
                # if not ass:
                #     break

            #     break
            # break
    # deleting unnecessary fleet descriptions
    print(assigned_fleet_size, "assigned, from", sum(traffic_flow.values()))
    fleet_types = fleet_types[~(fleet_types.fleet_size == 0.0)]

    # fleet_types.to_csv("data/clean_fleet_input_2.csv")

    # redesigning the input

    # fleet_id;start_timestep;route;fleet_size;charge_cap;batt_cap;d_spec;SOC_init;incoming;arriving;depart_time
    fleet_types["start_timestep"] = fleet_types["tau"]
    fleet_types["charge_cap"] = fleet_types["charging_capacity"]
    fleet_types["batt_cap"] = fleet_types["battery_capacity"]

    if is_winter:
        fleet_types["d_spec"] = fleet_types["energy_cons"].array/100
    else:
        fleet_types["d_spec"] = fleet_types["energy_cons_summer"].array / 100
    fleet_types["route"] = fleet_types["cell_route"]
    fleet_types["incoming"] = [{fleet_types["route"].to_list()[ij][0]: fleet_types["fleet_size"].to_list()[ij]} for ij in range(0, len(fleet_types))]
    fleet_types["arriving"] = [{fleet_types["route"].to_list()[ij][-1]: fleet_types["fleet_size"].to_list()[ij]} for ij in range(0, len(fleet_types))]
    fleet_types["depart_time"] = [{fleet_types["route"].to_list()[ij][0]: {fleet_types["start_timestep"].to_list()[ij]: fleet_types["fleet_size"].to_list()[ij]}} for ij in range(0, len(fleet_types))]
    fleet_types["fleet_id"] = list(range(0, len(fleet_types)))




    fleet_types.to_csv("data/" + case_study_name + "fleet_input_20220722.csv", sep=";")
