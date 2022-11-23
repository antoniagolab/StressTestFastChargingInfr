"""
"""
import math

import numpy as np
import geopandas as gpd
from shapely import wkt
from shapely.geometry import MultiLineString, Point
from pyproj import Geod
import pandas as pd
import pickle

reference_coord_sys = "EPSG:31287"


def random_starting_time(
    k0, k1, mu, sigma, dep_dict, delta_t=0.25, bott_cutoff=0, top_cutoff=24
):
    """
    function generating random starting time and adding this to the starting time dictionary
    :param k0:
    :param k1:
    :param mu:
    :param sigma:
    :param dep_dict:
    :param delta_t:
    :param bott_cutoff:
    :param top_cutoff:
    :return:
    """

    random_sample = np.random.normal(mu, sigma)
    if not (random_sample < bott_cutoff or random_sample > top_cutoff):
        origin = k0
        destin = k1
        if origin in dep_dict.keys():
            if destin in dep_dict[origin].keys():
                time_dict = dep_dict[origin][destin]
                # finding matching time slot in this dictionary
                for t in time_dict.keys():
                    if (random_sample >= t) and (random_sample < t + 1):
                        dep_dict[origin][destin][t] = dep_dict[origin][destin][t] + 1
                        break


def pd2gpd(dataframe, geom_col_name="geometry"):
    dataframe[geom_col_name] = dataframe[geom_col_name].apply(wkt.loads)
    dataframe = gpd.GeoDataFrame(dataframe, geometry=geom_col_name)
    return dataframe.set_crs(reference_coord_sys)


def reverse_distances(distances):
    """
    function reversing the distance measures
    :param distances: (m)
    :return:
    """
    segment_length = distances[-1]
    reversed_distances = [0]
    distances_copy = distances.copy()
    distances_copy.reverse()
    print(distances_copy)
    for ij in range(1, len(distances_copy)):
        reversed_distances.append(segment_length - distances_copy[ij])

    return reversed_distances


def read_fleets(fleet_df):
    """

    :param fleet_df:
    :param touple_column:
    :return:
    """
    for col in ["route", "incoming", "arriving", "depart_time"]:
        route_column = fleet_df[col].to_list()
        fleet_df[col] = [eval(item) for item in route_column]
    return fleet_df


# def calculate_distance():
#     g = Geod(ellps='bessel')
#
#     return None


def filter_paths_by_route_length(path_dictionary, cut_value=100000):
    to_del = []
    for k in path_dictionary.keys():
        if path_dictionary[k][1] < cut_value:
            to_del.append(k)

    for k in to_del:
        del path_dictionary[k]


def finding_segments_point_lies_on(point_list, linestring_gdf):
    """

    :param point_list: list with shapely.geometry.Point objects
    :param linestring_gdf: geopandas.GeoDataFrame
    :return:
    """
    linestring_list = linestring_gdf.geometry.to_list()
    linestring_ids = linestring_gdf["ID"].to_list()
    connections = {}
    for ij in range(0, len(point_list)):
        current_p = point_list[ij]
        id_list = []
        for kl in linestring_ids:
            current_l = linestring_list[kl]
            if current_l.distance(current_p) < 1e-3:
                id_list.append(kl)
        connections[ij] = id_list

    return connections


def identify_point(p, nodes_df, pois_df, intersections_df, highway_geometry):
    # could be an intersection
    nodes_df = nodes_df.fillna(0)
    nuts_ids = nodes_df.NUTS_ID.to_list()
    country_names = nodes_df.name.to_list()
    geoms = nodes_df.geometry.to_list()
    geom = geoms[p]
    if (nuts_ids[p] == 0) and (country_names[p] == 0):  # test if intersection/endpoint
        # no find corresponding intersection
        intersections_df["dist"] = intersections_df.geometry.distance(geom)
        idx_intersection = intersections_df.dist.argmin()
        # test if this distance is small enough
        if intersections_df.dist.to_list()[idx_intersection] < 1e-4:
            # then it is an intersection
            extract = pois_df[
                (pois_df.pois_type == "link") & (pois_df.type_ID == idx_intersection)
            ]
            return (
                extract.index.to_list(),
                intersections_df.loc[idx_intersection, "conn_edges"],
            )
        else:
            # project onto highway network
            seg_id = finding_segments_point_lies_on([geom], highway_geometry)[0][0]
            segment_geom = highway_geometry[
                highway_geometry.ID == seg_id
            ].geometry.to_list()[0]
            distance = segment_geom.distance(geom)
            extract = pois_df[
                (pois_df.segment_id == seg_id)
                & (pois_df.pois_type == "link")
                & (round(pois_df.dist_along_segment, 0) == round(distance, 0))
            ]
            return extract.index.to_list(), [seg_id]

    else:  # no intersection/endpoint, i.e. an OD
        extract = pois_df[(pois_df.type_ID == p) & (pois_df.pois_type == "od")]
        # if link with same dist_along_segment on segment, then take the link
        dist_along_segment = extract.dist_along_segment.to_list()[0]
        seg_id = extract.segment_id.to_list()[0]
        extract_obs = pois_df[
            (pois_df.segment_id == seg_id)
            & (pois_df.dist_along_segment == dist_along_segment)
        ]
        if len(extract_obs) > 1:
            if "link" in extract_obs.pois_type.to_list():
                # take this as point
                id = extract_obs[extract_obs.pois_type == "link"].type_ID.to_list()[0]
                if id >= 0:
                    ex = pois_df[
                        (pois_df.pois_type == "link") & (pois_df.type_ID == id)
                    ]
                    conn_edges = intersections_df[
                        intersections_df.ID == id
                    ].conn_edges.to_list()[0]
                    return ex.index.to_list(), conn_edges
                else:
                    return extract_obs[
                        extract_obs.pois_type == "link"
                    ].index.to_list(), [seg_id]
            else:
                return extract.index.to_list(), extract.segment_id.to_list()
        else:
            return extract.index.to_list(), extract.segment_id.to_list()


def is_same_distance(pois_df, seg_id, poi_id):
    dist = pois_df[pois_df.ID == poi_id].dist_along_segment.to_list()[0]
    extract = pois_df[
        (pois_df.segment_id == seg_id) & (pois_df.dist_along_segment == dist)
    ]

    ind_list = extract.ID.to_list()
    ind_list.remove(poi_id)
    return ind_list


def generate_fleet_start_time_matrix_one_connection(
    origin_nut: str,
    destination_nut: str,
    nb_travels: dict,
    distributions: dict,
    departure_dict: dict,
    ev_share,
):
    """
    for a given OD: generate starting times of cars, given the number of vehicles
    :param origin_nut:
    :param destination_nut:
    :param nb_travels:
    :param distributions:
    :param departure_dict:
    :return:
    """
    purposes = list(nb_travels.keys())
    for p in purposes:
        distribution = distributions[p]
        nb = int(nb_travels[p] * ev_share)
        counter = 0
        while counter < nb:
            random_starting_time(
                origin_nut,
                destination_nut,
                distribution[0],
                distribution[1],
                departure_dict,
                delta_t=0.25,
            )
            counter = counter + 1


def create_start_time_matrix(
    OD_purposes,
    start_time_distributions_1: dict,
    start_time_distributions_2: dict,
    ev_share: float,
):
    # create nuts_unique
    nuts = []
    for k in OD_purposes.keys():
        nuts.append(k[0])
        nuts.append(k[1])
    nuts_unique = list(set(nuts))
    if None in nuts_unique:
        nuts_unique.remove(None)
    d = dict(zip(list(np.arange(0, 24)), [0] * 24))
    n = len(nuts_unique)
    destination_dict = dict(zip(nuts_unique, [d.copy() for ij in range(0, n)]))
    departure_dict = dict(
        zip(nuts_unique, [destination_dict.copy() for ij in range(0, n)])
    )

    for n1 in nuts_unique:
        for n2 in nuts_unique:
            generate_fleet_start_time_matrix_one_connection(
                n1,
                n2,
                OD_purposes[(n1, n2)],
                start_time_distributions_1,
                departure_dict,
                ev_share,
            )

    for n1 in nuts_unique:
        for n2 in nuts_unique:
            generate_fleet_start_time_matrix_one_connection(
                n1,
                n2,
                OD_purposes[(n1, n2)],
                start_time_distributions_2,
                departure_dict,
                ev_share,
            )

    return departure_dict


def str_to_touple_keys(dictionary):
    l = list(dictionary.keys())
    for k in l:
        dictionary[eval(k)] = dictionary[k]
        del dictionary[k]


def reorder_this(pois_df):
    seg_ids = list(set(pois_df.segment_id.to_list()))

    for s in seg_ids:
        extract = pois_df[pois_df.segment_id == s]
        inds = extract.index.to_list()
        types = extract.pois_type.to_list()
        inds_copy = inds.copy()

        # find first link
        for ij in range(0, len(types)):
            if types[ij] == "link":
                link_0_ind = inds[ij]
                break

        # find last
        for ij in range(len(types) - 1, -1, -1):
            if types[ij] == "link":
                link_1_ind = inds[ij]
                break

        # create order of indices
        new_order = [link_0_ind]
        inds_copy.remove(link_0_ind)
        inds_copy.remove(link_1_ind)
        new_order = new_order + inds_copy + [link_1_ind]

        # replace existing
        pois_df.iloc[inds] = extract.loc[new_order]

    pois_df["ID"] = range(0, len(pois_df))
    pois_df = pois_df.set_index("ID")

    return pois_df


def fix_p(p):
    # source: https://www.reddit.com/r/learnpython/comments/5zytkk/alternative_for_numpyrandomchoice_since_my/
    if p.sum() != 1.0:
        p = p * (1.0 / p.sum())
    return p


def draw_car_sample(frequency_table: pd.DataFrame, min_driving_range):
    """
    function for drawing a car model from car models in EV fleet
    :param sample_size: size of fleet
    :param frequency_table: table showing sales of car models during the last couple years
    :return: technological attributes of this car model
    """
    frequency_table = frequency_table.fillna(0)

    car_models = frequency_table.model.to_list()
    charg_caps = frequency_table["charging_cap (max 150kW)"].to_list()
    battery_capacity = frequency_table["battery_capacity"].to_list()
    # driving_range = battery_capacity

    energy_cons = frequency_table["energy_cons_winter"].to_list()

    energy_cons_summer = frequency_table["energy_cons_summer"].to_list()

    driving_range = battery_capacity/(np.array(energy_cons)/100)  # (km)
    frequency_table["driving_range"] = driving_range
    frequency_table = frequency_table[
        frequency_table["driving_range"] > min_driving_range
        ]
    types = range(0, len(car_models))
    total_sum = (
        frequency_table["Sales_2019"].sum()
        + frequency_table["Sales_2020"].sum()
        + frequency_table["Sales_2021"].sum()
    )
    probabilites = []

    for ij in types:
        # occurence of a car model
        total_occ = (
            frequency_table.iloc[ij]["Sales_2019"]
            + frequency_table.iloc[ij]["Sales_2020"]
            + frequency_table.iloc[ij]["Sales_2021"]
        )
        prob = total_occ / total_sum
        # for s in range(0, sample_size-1):
        #     occ = total_occ - 1
        #     summ = total_sum - 1
        #     prob = prob * (occ/summ)

        probabilites.append(prob)

    chosen_car_model = np.random.choice(types, 1, p=fix_p(np.array(probabilites)))[0]
    # calculate min SOC, based on min driving range
    # how much kWh is needed for this maximum distance of no charging station
    min_SOC = min_driving_range * energy_cons[chosen_car_model]/100 / battery_capacity[chosen_car_model] + 0.1
    print(min_SOC, battery_capacity[chosen_car_model], energy_cons[chosen_car_model])
    return (
        charg_caps[chosen_car_model],
        battery_capacity[chosen_car_model],
        energy_cons[chosen_car_model],
        energy_cons_summer[chosen_car_model],
        draw_SOC_init(SOC_min=min_SOC),
        min_SOC,
    )


def create_sublists(l):
    """
    creating all possible sublist of a list
    :param l: list
    :return:
    """
    sublist_list = []
    length = len(l)-1
    while length > 1:
        sublist_list.append(tuple(l[0:length]))
        length = length - 1
    return sublist_list


def my_ceil(a, precision=0):
    return np.round(a + 0.5 * 10 ** (-precision), precision)


def draw_SOC_init(SOC_min=0.1, SOC_max=1):
    if SOC_min > 0.9:
        SOC_init = 1
    else:
        options = np.arange(math.ceil(SOC_min*10)/10, SOC_max + 0.1, step=0.1)
        SOC_init = np.random.choice(options, p=[1 / (len(options))] * len(options))
    return SOC_init


def get_segment_id_of_cell(cells, cell_id):
    cell_col = cells.cells.to_list()
    seg_ids = cells.segment_id.to_list()

    s = None
    for ij in range(0, len(cell_col)):
        if cell_id in cell_col[ij]:
            s = seg_ids[ij]
            break

    return s


def identify_position_poi_between_link_ra(seg_id, dist, cell_df):
    extract_cells = cell_df[cell_df.segment_id == seg_id]
    dist_start = extract_cells.dist_start.to_list()
    dist_end = extract_cells.dist_end.to_list()
    inds = extract_cells.index.to_list()
    idx = None
    for ij in range(0, len(extract_cells)):
        if dist < dist_end[ij] and dist >= dist_start[ij]:
            idx = inds[ij]
            print("wow")
            break
    return idx


def get_min_driving_range(route, cell_df):
    # create df based on Route Abfolge
    extract = cell_df.loc[list(route)]

    # iteratively calculate all distances origin - cs ... - destination
    dists = []
    d_last = 0
    dist_count = 0
    cell_lengths = extract.length.to_list()
    has_cs = extract.has_cs.to_list()
    for ij in range(0, len(extract)):

        if has_cs[ij]:
            dist_count = dist_count + cell_lengths[ij] / 2
            dists.append(dist_count - d_last)
            d_last = dist_count
            dist_count = dist_count + cell_lengths[ij] / 2

        else:
            dist_count = dist_count + cell_lengths[ij]

        if ij == len(extract) - 1:
            dists.append(dist_count - d_last)

    # get min value
    #  + 0.1 * min value = min driving range

    # TODO adapt drawing a car model
    return max(dists)


def get_min_SOC_init(route, cell_df, batt_cap, d_spec):
    # get distance between origin - first cs
    extract = cell_df.loc[list(route)]

    dists = []
    d_last = 0
    dist_count = 0
    cell_lengths = extract.length.to_list()
    has_cs = extract.has_cs.to_list()
    for ij in range(0, len(extract)):

        if has_cs[ij]:
            dist_count = dist_count + cell_lengths[ij] / 2
            dists.append(dist_count - d_last)
            d_last = dist_count
            dist_count = dist_count + cell_lengths[ij] / 2
            break
        else:
            dist_count = dist_count + cell_lengths[ij]

        if ij == len(extract) - 1:
            dists.append(dist_count - d_last)

    d = dists[0]
    charge_min = batt_cap - dist_count * d_spec / 100
    return charge_min / batt_cap


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def parse_values(lines, identifier: str, offset=643):
    id = [ind for ind in range(0, len(lines)) if identifier in lines[ind]][0]
    selection = lines[id + 1:(id + 1 + offset)]
    values = []
    for line in selection:
        split_line = line.split(" ")
        values = values + [float(el) for el in split_line if isfloat(el)]

    return np.array(values)


def parse_OD_data(filename, od_of_interest):
    traffix_data = pd.read_csv("data/matched_gemeinde_traffix_data.csv")
    od_ids = traffix_data['NUTS-3 region'].to_list()
    # after row nb 11 -> read all "Netzojekt-Nummern" ->
    df = pd.DataFrame()
    with open(filename
              ) as f:
        lines = f.readlines()
    object_nb = "* Netzobjekt-Nummern\n"
    codes = parse_values(lines, identifier="* Netzobjekt-Nummern\n")
    d = {}
    for id in od_of_interest:
        extract_td = traffix_data[traffix_data['NUTS-3 region'] == id]
        print(extract_td.keys())
        if len(extract_td) > 0:
            object_nbs = extract_td['0'].to_list()
            print(object_nbs)
            for obj in object_nbs:
                # if obj in od_of_interest:
                search_object = "* Obj " + str(int(obj))
                vals = parse_values(lines, identifier=search_object, offset=643)
                # values = values + vals
                # d["origin"] = obj
                for n in range(0, len(codes)):
                    if traffix_data[traffix_data['0'] == codes[n]]['NUTS-3 region'].to_list()[0] in od_of_interest:
                        id_2 = traffix_data[traffix_data['0'] == int(codes[n])]['NUTS-3 region'].to_list()[0]
                        if (id, id_2) in d.keys():
                            d[(id, id_2)] = d[(id, id_2)] + vals[n]
                        else:
                            d[(id, id_2)] = vals[n]
        # print(id, "done")
            # df.to_csv("data/OD_traffix_amount_of_vehicles.csv")
        #if id == "DE":
        with open("data/traffic_load_countries.pickle", "wb") as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #break


def split_traffic_flow_transit():
    countries = ["SL", "DE"]
    with open("data/traffic_load_countries.pickle", "rb") as handle:
        traffic_flow = pickle.load(handle)
    key_dictioniary = list(traffic_flow.keys())
    for k in key_dictioniary:
        if k[0] in countries:
            traffic_flow[(k[0] + "1", k[1])] = traffic_flow[k]/2
            traffic_flow[(k[0] + "2", k[1])] = traffic_flow[k] / 2
        elif k[1] in countries:
            traffic_flow[(k[0], k[1] + "1")] = traffic_flow[k]/2
            traffic_flow[(k[0], k[1] + "2")] = traffic_flow[k] / 2
    with open("data/traffic_load_countries.pickle", "wb") as handle:
        pickle.dump(traffic_flow, handle, protocol=pickle.HIGHEST_PROTOCOL)


def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]



def get_gamma_h(fleets):
    """
    function to determine the percentage of peak time in starting times
    :param fleets:
    :return:
    """
    all_vehicles = fleets.fleet_size.sum()
    dic = {}
    for t in range(0, 96, 4):
        extract_df = fleets[fleets.start_timestep == t]
        dic[t] = extract_df.fleet_size.sum()

    maximum_hour = max(dic, key=dic.get)
    perc = dic[maximum_hour]/all_vehicles
    return perc


def create_maximum_dem_file():
    dem_workday = pd.read_csv("data/_demand_for_FC_workday.csv")
    dem_holiday = pd.read_csv("data/_demand_for_FC_holiday.csv")
    new_dem = dem_workday.copy()
    new_dem["tc_0"] = [max(dem_workday.tc_0.to_list()[ij], dem_holiday.tc_0.to_list()[ij]) for ij in range(0, len(new_dem))]
    new_dem["tc_1"] = [max(dem_workday.tc_1.to_list()[ij], dem_holiday.tc_1.to_list()[ij]) for ij in
                       range(0, len(new_dem))]
    new_dem["demand_0"] = [max(dem_workday.demand_0.to_list()[ij], dem_holiday.demand_0.to_list()[ij]) for ij in
                           range(0, len(new_dem))]
    new_dem["demand_1"] = [max(dem_workday.demand_1.to_list()[ij], dem_holiday.demand_1.to_list()[ij]) for ij in
                           range(0, len(new_dem))]
    new_dem.to_csv("data/_peak_demand.csv")


def identify_segment_sp(p: Point, link_gdf, segment_gdf):
    id = None
    on_segments = None
    # check if link
    indices = link_gdf.index.to_list()
    conn_edges = link_gdf.conn_edges.to_list()
    dists = [p.distance(link_gdf.geometry.to_list()[ij]) for ij in range(0, len(link_gdf))]
    # identify closest
    min_idx = np.argmin(dists)
    is_link = None
    # then check if it is identical
    if dists[min_idx] < 1e-5:
        is_link = True
        id = indices[min_idx]
        on_segments = eval(conn_edges[min_idx])
    else:
        is_link = False
        output = finding_segments_point_lies_on([p], segment_gdf)
        if len(list(output.keys())) > 0:
            on_segments =list(output.values())[0]
    return is_link, id, on_segments


def reorder_pois_table(pois_df):

    segment_ids = list(set(pois_df.segment_id.to_list()))
    for seg_id in segment_ids:
        extract_df = pois_df[pois_df.segment_id == seg_id]
        pois_types = extract_df.pois_type.to_list()
        if pois_types[0] == "link" and pois_types[-1] == "link":
            continue
        else:
            indices = extract_df.index.to_list()
            if not pois_types[0] == "link":
                row_0 = extract_df.loc[indices[0]]
                row_1 = extract_df.loc[indices[1]]
                pois_df.loc[indices[1]] = row_0
                pois_df.loc[indices[0]] = row_1

            if not pois_types[-1] == "link":
                row_0 = extract_df.loc[indices[-2]]
                row_1 = extract_df.loc[indices[-1]]
                pois_df.loc[indices[-1]] = row_0
                pois_df.loc[indices[-2]] = row_1

    pois_df["ID"] = range(0, len(pois_df))
    pois_df = pois_df.set_index("ID")
    pois_df["ID"] = range(0, len(pois_df))

    return pois_df