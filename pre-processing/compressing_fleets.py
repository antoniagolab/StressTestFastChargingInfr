"""
Within this script, the dimension of fleets is reduced
We here perform different stages of reduction which are:
    - we first unite fleets following the same route + having same technological parameters but distance between
    starting time is > 2x min(route)
    - same starting time + same vehicle type + route A part of route B; just arrival earlier


"""
import pandas as pd

from utils import *

filename = "winter_holidayfleet_input_20220722"
fleets = read_fleets(pd.read_csv("data/" + filename + ".csv", delimiter=";"))

fleets["len"] = fleets["route"].apply(len)
fleets = fleets[fleets.len > 3]
fleets["distance"] = fleets.fleet_size.array * fleets.len.array * 27.5
print(fleets["distance"].sum())
# driven_km = 0
# inds = fleets.index.to_list()
# incomings = fleets.incoming.to_list()
# arrivings = fleets.arriving.to_list()
# routes = fleets.route.to_list()
#
# for id in range(0, len(inds)):
#     r = list(routes[id])
#     d = 0
#     for k in arrivings[id].keys():
#         veh_exit = arrivings[id][k]
#         i_exit = r.index(k)
#         driven_km = driven_km + len(r[0: r[i_exit]]) * 27.5 * veh_exit
#         d = d + len(r[0: r[i_exit]]) * 27.5 * veh_exit
#     fleets.at[id, "distance"] = d

# print("new km", driven_km)
fleets = fleets.sort_values(by=["len"], ascending=[False])
fleets_dual = fleets.copy()
new_fleet_input = pd.DataFrame()
kl = 0
nb_fleet = len(fleets)
routes = fleets.route.to_list()
charging_cap = fleets.charging_capacity.to_list()
batt_caps = fleets.batt_cap.to_list()
d_specs = fleets.d_spec.to_list()
SOC_init = fleets.SOC_init.to_list()
depart_times = fleets.tau.to_list()
incoming_dicts = fleets.incoming.to_list()
dists = fleets.distance.to_list()
arriving_dicts = fleets.arriving.to_list()
depart_time_dicts = fleets.depart_time.to_list()
taus = fleets.tau.to_list()
indices = fleets.index.to_list()
fleet_sizes = fleets.fleet_size.to_list()
print(fleets.fleet_size.sum())
match = 0
# sort here after length of route
while kl < nb_fleet:
    curr_ind = indices[kl]
    if curr_ind in fleets.index.to_list():
        # get current line
        # get its route and calculate minimum travel time
        # find other lines which have same route + same technological parameter + tau later than
        r = routes[kl]
        cc = charging_cap[kl]
        og_fs = fleet_sizes[kl]
        min_tt = len(r)
        route_dist = dists[kl]
        earliest_arrival = depart_times[kl] + min_tt
        incoming = incoming_dicts[kl]
        arriving = arriving_dicts[kl]
        depart_time = depart_time_dicts[kl]
        # print(r, cc, earliest_arrival + min_tt)
        # extend to same starting cell and similar route
        extract_df = fleets[(fleets.route.isin(create_sublists(r))) & (fleets.charging_capacity == cc) & (fleets.tau == taus[kl])]
        fleets = fleets.drop(curr_ind)
        soc_inits = [SOC_init[kl]]
        while len(extract_df) > 0:
            match = match + 1
            new_extract = extract_df[extract_df.len == extract_df.len.max()]
            ind = new_extract.index.to_list()[0]
            route_match = extract_df.at[ind, "route"]
            route_dist = route_dist + extract_df.at[ind, "distance"]
            if r[0] in incoming.keys():
                incoming[r[0]] = incoming[r[0]] + extract_df.at[ind, "fleet_size"]
            else:
                incoming[r[0]] = extract_df.at[ind, "fleet_size"]

            if route_match[-1] in arriving.keys():
                arriving[route_match[-1]] = arriving[route_match[-1]] + extract_df.at[ind, "fleet_size"]
            else:
                arriving[route_match[-1]] = extract_df.at[ind, "fleet_size"]
            # arriving[r[-1]] = extract_df.at[ind, "fleet_size"]
            if r[0] in depart_time.keys():
                if extract_df.at[ind, "tau"] in depart_time[r[0]].keys():
                    depart_time[r[0]][extract_df.at[ind, "tau"]] = depart_time[r[0]][extract_df.at[ind, "tau"]] + extract_df.at[ind, "fleet_size"]
                else:
                    depart_time[r[0]][extract_df.at[ind, "tau"]] = extract_df.at[ind, "fleet_size"]

            else:
                depart_time[r[0]] = {extract_df.at[ind, "tau"]: extract_df.at[ind, "fleet_size"]}

            og_fs = og_fs + extract_df.at[ind, "fleet_size"]
            tau_current = extract_df.at[ind, "tau"]
            soc_inits.append(extract_df.at[ind, "SOC_init"])
            extract_df = extract_df.drop(ind)
            fleets = fleets.drop(ind)
            # break
        # print("comparison", og_fs, sum(incoming.values()))
        dist_km = 0
        list_route = list(r)
        for k in arriving.keys():
            veh_exit = arriving[k]
            i_exit = list_route.index(k)
            dist_km = dist_km + len(list_route[0: i_exit+1]) * 27.5 * veh_exit

        row = {"start_timestep": depart_times[kl], "route": routes[kl], "fleet_size": sum(incoming.values()), "charge_cap": cc, "batt_cap": batt_caps[kl],
               "d_spec": d_specs[kl], "SOC_init": np.max(soc_inits), "incoming": incoming, "arriving": arriving, "depart_time": depart_time,
               "route_len" : route_dist, "dist_km": dist_km}
        if not route_dist == dist_km:
            break
        new_fleet_input = new_fleet_input.append(row, ignore_index=True)
        # break
    # print(kl)
    kl = kl + 1
    # if kl > len(fleets):
    #     break

print(new_fleet_input.fleet_size.sum())
new_fleet_input = new_fleet_input.sort_values(by=["start_timestep"])
print()
driven_km = 0
inds = new_fleet_input.index.to_list()
incomings = new_fleet_input.incoming.to_list()
arrivings = new_fleet_input.arriving.to_list()
routes = new_fleet_input.route.to_list()
for id in range(0, len(inds)):
    r = list(routes[id])
    for k in arrivings[id].keys():
        veh_exit = arrivings[id][k]
        i_exit = r.index(k)
        driven_km = driven_km + len(r[0:( i_exit + 1)]) * 27.5 * veh_exit

print("new km", new_fleet_input.dist_km.sum(), driven_km)

#
kl = 0
fleets = new_fleet_input.copy()
fleets_orig = fleets.copy()
nb_fleet = len(fleets)
routes = fleets.route.to_list()
charging_cap = fleets.charge_cap.to_list()
batt_caps = fleets.batt_cap.to_list()
d_specs = fleets.d_spec.to_list()
SOC_init = fleets.SOC_init.to_list()
dists = fleets.dist_km.to_list()
depart_times = fleets.start_timestep.to_list()
incoming_dicts = fleets.incoming.to_list()
arriving_dicts = fleets.arriving.to_list()
depart_time_dicts = fleets.depart_time.to_list()
taus = fleets.start_timestep.to_list()
indices = fleets.index.to_list()
match = 0
new_fleet_input = pd.DataFrame()
while kl < nb_fleet:
    curr_ind = indices[kl]
    if curr_ind in fleets.index.to_list():
        # get current line
        # get its route and calculate minimum travel time
        # find other lines which have same route + same technological parameter + tau later than
        r = routes[kl]
        cc = charging_cap[kl]
        min_tt = len(r)
        route_dist = [dists[kl]]
        earliest_arrival = depart_times[kl] + min_tt
        incoming = incoming_dicts[kl]
        arriving = arriving_dicts[kl]
        depart_time = depart_time_dicts[kl]
        # print(r, cc, earliest_arrival + min_tt)
        # extend to same starting cell and similar route
        extract_df = fleets[(fleets.route == r) & (fleets.charge_cap == cc) & (fleets.start_timestep > depart_times[kl] +2)]
        fleets = fleets.drop(curr_ind)
        soc_inits = [SOC_init[kl]]
        added_inds = []
        while len(extract_df) > 0:
            match = match + 1
            new_extract = extract_df[extract_df.start_timestep == extract_df.start_timestep.min()]
            ind = new_extract.index.to_list()[0]
            added_inds.append(ind)
            route_match = extract_df.at[ind, "route"]
            match_arriving = extract_df.at[ind, "arriving"]
            match_departure = extract_df.at[ind, "depart_time"]
            route_dist = route_dist + [extract_df.at[ind, "dist_km"]]
            # r = extract_df.at[ind, "route"]
            if route_match[0] in incoming.keys():
                incoming[route_match[0]] = incoming[route_match[0]] + extract_df.at[ind, "fleet_size"]
            else:
                incoming[route_match[0]] = extract_df.at[ind, "fleet_size"]

            for arrival_key in match_arriving.keys():
                if arrival_key in arriving.keys():
                    arriving[arrival_key] = arriving[arrival_key] + match_arriving[arrival_key]
                else:
                    arriving[arrival_key] = match_arriving[arrival_key]
            # arriving[r[-1]] = extract_df.at[ind, "fleet_size"]
            for depart_cell_key in match_departure:
                if depart_cell_key in depart_time.keys():
                    for start_time_key in match_departure[depart_cell_key].keys():
                        if start_time_key in depart_time[depart_cell_key].keys():
                            depart_time[depart_cell_key][start_time_key] = depart_time[depart_cell_key][start_time_key] \
                                                                           + match_departure[depart_cell_key][start_time_key]
                        else:
                            depart_time[depart_cell_key][start_time_key] = match_departure[depart_cell_key][start_time_key]

                else:
                    depart_time[depart_cell_key] = match_departure[depart_cell_key]

            tau_current = extract_df.at[ind, "start_timestep"]
            soc_inits.append(extract_df.at[ind, "SOC_init"])
            extract_df = extract_df[extract_df.start_timestep > tau_current]
            fleets = fleets.drop(ind)
        dist_km = 0
        list_route = list(r)
        for k in arriving.keys():
            veh_exit = arriving[k]
            i_exit = list_route.index(k)
            dist_km = dist_km + len(list_route[0: i_exit+1]) * 27.5 * veh_exit
            #print(len(list_route[0: i_exit+1]) * 27.5 * veh_exit)

        row = {"start_timestep": depart_times[kl], "route": r, "fleet_size": sum(incoming.values()), "charge_cap": cc, "batt_cap": batt_caps[kl],
               "d_spec": d_specs[kl], "SOC_init": np.max(soc_inits), "incoming": incoming, "arriving": arriving, "depart_time": depart_time,
               "route_dist": sum(route_dist), "dist_km": dist_km}
        new_fleet_input = new_fleet_input.append(row, ignore_index=True)
        if not sum(route_dist) == dist_km:
            break
    # print(kl)
    kl = kl + 1
    # if kl > len(fleets):
    #     break
# print("o", new_fleet_input.dist_km.sum(), new_fleet_input.route_dist.sum())
#print("New length", len(new_fleet_input))
print(new_fleet_input.fleet_size.sum())
#print(match, "matches")
new_fleet_input["len"] = new_fleet_input["route"].apply(len)

new_fleet_input["fleet_id"] = list(range(0, len(new_fleet_input)))

driven_km = 0
inds = new_fleet_input.index.to_list()
incomings = new_fleet_input.incoming.to_list()
arrivings = new_fleet_input.arriving.to_list()
routes = new_fleet_input.route.to_list()
for id in range(0, len(inds)):
    r = list(routes[id])
    for k in arrivings[id].keys():
        veh_exit = arrivings[id][k]
        i_exit = r.index(k)
        driven_km = driven_km + len(r[0:( i_exit + 1)]) * 27.5 * veh_exit

print("new km", driven_km)



new_fleet_input.to_csv("data/" + filename + "_compressed_probe2.csv", sep=";", index=False)
# TODO: test if all united and none lost by summing up all incomings and arrivals and comparing to original all arrivals and arrivals
