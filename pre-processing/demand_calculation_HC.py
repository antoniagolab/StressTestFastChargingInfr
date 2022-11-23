"""

Here the demand calculation for the fast-charging infrastructure planning is prepared

Output must be: demand at all nodes


(0) identify maximum *hourly* traffic flow between each OD pair
    (A) filter for "long_distance": route must be > 100km (check the saved one from openrouteservice)
    (B) use same hourly peak as in previous study - 12%, apply this calculate the amount for each segment

(1) maximum traffic load along all Teilsegmente
    - I need to know the routes for this (Based on Teilsegmente)

(2) Introduce the OD nodes into POI table

(3) to identify traffic load; check if the observed segment between two points lies within the route

"""
import geopandas as gpd
import pickle

import numpy as np
import pandas as pd
from shapely.geometry import Point
import ast
from utils import *

l_od_pairs = [('AT322', 'AT211'), ('AT322', 'AT221'), ('AT322', 'AT224'), ('AT322', 'AT113'), ('AT322', 'SL1'),
    ('AT321', 'AT226'), ('AT321', 'AT223'), ('AT321', 'AT221'), ('AT321', 'AT224'), ('AT321', 'AT113'),
     ('AT321', 'SL1'), ('AT321', 'AT127'), ('AT321', 'AT122'), ('AT321', 'AT112'), ('AT321', 'AT130'), ('AT321', 'CZ'),
     ('AT321', 'HU'), ('AT321', 'SI'), ('AT333', 'AT226'), ('AT333', 'AT223'), ('AT333', 'AT221'),
     ('AT333', 'AT224'), ('AT333', 'AT113'), ('AT333', 'SL1'), ('AT333', 'AT123'), ('AT333', 'AT127'), ('AT333', 'AT122'),
     ('AT333', 'AT112'), ('AT333', 'AT130'), ('AT333', 'CZ'), ('AT333', 'HU'), ('AT333', 'SI'), ('AT212', 'AT222'),
      ('AT212', 'AT226'), ('AT212', 'AT223'), ('AT212', 'AT221'), ('AT212', 'AT224'), ('AT212', 'AT113'),
     ('AT212', 'SL1'), ('AT212', 'AT123'), ('AT212', 'AT127'), ('AT212', 'AT122'), ('AT212', 'AT112'), ('AT212', 'AT130'),
     ('AT212', 'CZ'), ('AT212', 'HU'), ('AT212', 'SI'), ('AT323', 'AT211'), ('AT315', 'AT211'), ('IT', 'AT314'),
     ('IT', 'AT222'), ('IT', 'AT211'), ('IT', 'AT226'), ('IT', 'AT223'), ('IT', 'AT221'), ('IT', 'AT224'), ('IT', 'AT113'), ('IT', 'AT124'), ('IT', 'AT123'), ('IT', 'AT127'), ('IT', 'AT122'), ('IT', 'AT112'), ('IT', 'AT130'), ('IT', 'CZ'), ('IT', 'SI'), ('SL2', 'AT314'), ('SL2', 'AT222'), ('SL2', 'AT211'), ('SL2', 'AT226'), ('SL2', 'AT223'), ('SL2', 'AT221'), ('SL2', 'AT224'), ('SL2', 'AT113'), ('SL2', 'AT124'), ('SL2', 'AT123'), ('SL2', 'AT127'), ('SL2', 'AT122'), ('SL2', 'AT112'), ('SL2', 'AT130'), ('SL2', 'CZ'), ('AT213', 'DE1'), ('AT213', 'AT311'), ('AT213', 'AT314'), ('AT213', 'AT222'), ('AT213', 'AT312'), ('AT213', 'AT313'), ('AT213', 'AT121'), ('AT213', 'AT211'), ('AT213', 'AT226'), ('AT213', 'AT223'), ('AT213', 'AT221'), ('AT213', 'AT224'), ('AT213', 'AT113'), ('AT213', 'SL1'), ('AT213', 'AT124'), ('AT213', 'AT123'), ('AT213', 'AT127'), ('AT213', 'AT122'), ('AT213', 'AT112'), ('AT213', 'AT130'), ('AT213', 'CZ'), ('AT213', 'HU'), ('AT213', 'SI')]

to_ignore = [0, 1, 2]
y_h = 0.12
energy_cons = 0.2
pois_with_geometry = gpd.read_file("data/pois_with_geometry.csv", geometry="projected_points")
links = pd.read_csv("data/highway_intersections.csv")
links["geometry"] = links["geometry"].apply(wkt.loads)
links = gpd.GeoDataFrame(links, geometry="geometry")

hs = pd.read_csv("geography/highway_segments.csv")
hs["geometry"] = hs["geometry"].apply(wkt.loads)
hs = gpd.GeoDataFrame(hs, geometry="geometry")

projected_nodes = gpd.read_file("data/all_projected_nodes.shp")
projected_nodes.at[27, "on_segment"] = 18
projected_nodes.at[27, "geometry"] = Point(473415.0648, 307050.4864)
projected_nodes.at[75, "name"] = "Hungary"
projected_nodes.at[75, "NUTS_ID"] = "HU"
projected_nodes.at[76, "on_segment"] = 60
projected_nodes.at[76, "name"] = "Slowakai"
projected_nodes.at[76, "NUTS_ID"] = "SI"
projected_nodes.at[74, "name"] = np.NAN
projected_nodes.at[74, "NUTS_ID"] = np.NAN

pn = pd.DataFrame(projected_nodes)
extract_pn = projected_nodes[projected_nodes.NUTS_ID.isna()]
indices = extract_pn.index.to_list()
pn["on_segment0"] = [None] * len(projected_nodes)
for ind in indices:
    output = identify_segment_sp(pn.at[ind, "geometry"], links, hs)
    pn.at[ind, "on_segment0"] = tuple(output[2])
    pn.at[ind, "link_id"] = output[1]
    print(ind, "done")
# match the route to ODs    --check

# check if these are correct    --check

# go through each route and identify the route segment

for case_study_workday in [True, False]:
# for case_study_workday in [True]:
    with open("data\distances.pickle", "rb") as handle:
        distances = pickle.load(handle)

    str_to_touple_keys(distances)

    # import projected OD nodes

    # import POI table
    pois = pd.read_csv("data/_demand_calculated.csv")
    pois = pois.drop(columns=["demand_0", "demand_1", "tc_0", "tc_1"])
    # get routes and figure out how a route can be retrieved
    with open("data\paths.pickle", "rb") as handle:
        paths = pickle.load(handle)
    #print(((435250.90294, 455838.05421), (414200.22737, 483070.53002)) in paths)
    # get all info on connections
    # pendler
    with open("data/traffic_load_countries.pickle", "rb") as handle:
        traffic_flow = pickle.load(handle)
    k_to_del = ["DE1", "DE2", "IT", "CZ", "SL1", "SL2", "SI", "HU"]
    # write algorithm to identify amount of traffic for each part of the Teilsegment
    allowed_keys = [("DE1", "SL1"), ("DE2", "SL1"), ("SL1", "DE1"), ("SL1", "DE2"), ("SL2", "DE1"), ("SL2", "DE2"),
                    ("DE1", "SL2"), ("DE2", "SL2"), ("DE1", "SI"), ("SI", "DE1"), ("DE2", "SI"), ("SI", "DE2"),  ("IT", "SI"), ("SI", "IT"), ("IT", "CZ"), ("CZ", "IT"),
                    ("SL1", "CZ"), ("SL2", "CZ"), ("CZ", "SL1"), ("CZ", "SL2")]
    # in paths: match the node coordinates to the actual nodes  -- check
    transit_pois = []
    if not case_study_workday:
        for k in traffic_flow.keys():
            if not k in allowed_keys:
                traffic_flow[k] = int(traffic_flow[k] * 0.7)
            else:
                traffic_flow[k] = int(traffic_flow[k] * 1.2)

        with open("data/traffic_load_holiday.pickle", "wb") as handle:
            pickle.dump(traffic_flow, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # other purposes
    # other_purposes = pd.read_csv("data/OD_info_10_AT315.csv")

    # highway segments
    highway_geometry = pd2gpd(pd.read_csv("geography/highway_segments.csv"))

    # highway junctions
    highway_junctions = pd2gpd(pd.read_csv("data/highway_intersections.csv"))
    highway_junctions["conn_edges"] = [
        eval(l) for l in highway_junctions["conn_edges"].to_list()
    ]
    # k_to_del = [("DE1", "DE2"), ("DE2", "DE1"), ("IT", "SL2"), ("IT", "SL1"), ("SL1", "IT"), ("SL2", "IT"), ("SL1", "HU"),
    #             ("HU", "SL1"),("HU", "SL2"), ("HU", "SI"), ("SI", "HU"), ("CZ", "SI"), ("SI", "CZ"),("DE1", "CZ"),
    #             ("DE2", "CZ"),  ("CZ", "DE2"),("CZ", "DE1")]

    # turn the path dict to one with node IDs
    filtered_nodes = projected_nodes.dropna(subset=["NUTS_ID"])
    on_segments = filtered_nodes.on_segment.to_list()
    nuts_ids = filtered_nodes.NUTS_ID.to_list()
    country_names = filtered_nodes.name.to_list()
    node_geometries = filtered_nodes.geometry.to_list()
    nodeIDs = filtered_nodes.nodeID.to_list()
    for ij in range(0, len(filtered_nodes)):
        if (nuts_ids[ij] is not None):
            #print(nuts_ids[ij])
            seg_id = on_segments[ij]
            if not seg_id >= 0:
                seg_id = finding_segments_point_lies_on(
                    [node_geometries[ij]], highway_geometry
                )[0][0]
            matching_segment = highway_geometry[
                highway_geometry.ID == seg_id
            ].geometry.to_list()[0]
            dist_along_segment = matching_segment.project(node_geometries[ij])
            row = {
                "segment_id": seg_id,
                "pois_type": "od",
                "type_ID": nodeIDs[ij],
                "dir": 2,
                "dist_along_segment": dist_along_segment,
            }

            pois = pois.append(row, ignore_index=True)

    # assign new IDs and sort
    pois_new = pois.copy()
    pois_new["dist_along_segment"] = pois_new.apply(
        lambda row: round(row["dist_along_segment"], 4), axis=1
    )
    pois_new = pois_new.sort_values(
        by=["segment_id", "dist_along_segment", "pois_type"], ascending=[True, True, False]
    )
    pois_new["ID"] = range(0, len(pois_new))
    pois_new = pois_new.set_index("ID")
    pois_new["ID"] = range(0, len(pois_new))
    pois_new = reorder_pois_table(pois_new)
    # SAVING
    pois_new.to_csv("data/pois_new.csv")

    # match the projected nodes to the segments they lie on
    # special care with intersections
    projected_nodes_only_od = projected_nodes[~projected_nodes.NUTS_ID.isna()]
    ids = projected_nodes.nodeID.to_list()
    paths_encoded = {}
    paths_encoded_to_od = {}
    for k in paths.keys():
        # match k
        p1 = k[0]
        p2 = k[1]
        # if p1 == (435250.90294, 455838.05421) and p2 == (414200.22737, 483070.53002):
        #     break
        projected_nodes_only_od["dist"] = projected_nodes_only_od["geometry"].distance(Point(p1))
        d1 = projected_nodes_only_od["dist"].min()
        idx_p1 = projected_nodes_only_od["dist"].argmin()
        extract_1 = projected_nodes_only_od[projected_nodes_only_od.nodeID == projected_nodes_only_od.index.to_list()[idx_p1]]
        # TODO: read "name" of "NUTS_ID" from table
        projected_nodes_only_od["dist"] = projected_nodes_only_od["geometry"].distance(Point(p2))
        d2 = projected_nodes_only_od["dist"].min()
        idx_p2 = projected_nodes_only_od["dist"].argmin()
        extract_2 = projected_nodes_only_od[projected_nodes_only_od.nodeID == projected_nodes_only_od.index.to_list()[idx_p2]]
        if len(extract_1) > 0 and len(extract_2) > 0 and d1 < 1e-4 and d2 < 1e-4:
            k1 = extract_1.NUTS_ID.to_list()[0]
            k2 = extract_2.NUTS_ID.to_list()[0]
            # if not k1 == k2:
            #     break
            # if not extract_1.name.to_list()[0] == None:
            #     k1 = extract_1.name.to_list()[0]
            # else:

            print(k1, k2)
            # if k1 == "SI" and k2 == "AT323":
            #     print(k1, k2)
            #     break
            # if not extract_2.name.to_list()[0] == None:
            #     k2 = extract_2.name.to_list()[0]
            # else:

            path = paths[k][0]
            path_length = paths[k][1]
            # if (k1, k2) in distances:
            #     path_length = distances[(k1, k2)]
            # else:
            # path_length = paths[k][1]
            # identify nodes along path
            nodes_along_path = []
            for p in path:
                # TODO: get length of these distances!
                projected_nodes["dist"] = projected_nodes["geometry"].distance(Point(p))
                if projected_nodes["dist"].min() < 1e-4:
                    idx_p = projected_nodes["dist"].argmin()
                    #if not (k1 == "AT127" and k2 == "AT127") and not idx_p == 49:
                    nodes_along_path.append(idx_p)

            if not ((not (k1, k2) in allowed_keys) and (k1 in k_to_del and k2 in k_to_del)):
                paths_encoded[(extract_1.index.to_list()[0], extract_2.index.to_list()[0])] = (f7(nodes_along_path), path_length, (k1, k2))
                paths_encoded_to_od[(k1, k2)] = (f7(nodes_along_path), paths[k][0], paths[k][1])
                if (k1, k2) in allowed_keys:
                    transit_pois.append((extract_1.index.to_list()[0], extract_2.index.to_list()[0]))
                    print(k1, k2, (extract_1.index.to_list()[0], extract_2.index.to_list()[0]))

    paths_encoded_copy = paths_encoded.copy()
    _filter_value = 100 * 1000 # (m), 80km
    filter_paths_by_route_length(paths_encoded, _filter_value)
    filter_paths_by_route_length(paths, _filter_value)


    paths_encoded_pois = {}
    paths_encoded_pois_lim = {}
    paths_encoded_pois_lim_and_length = {}
    routes_through_pair = []
    omitted_items = {}
    for k in paths_encoded_copy.keys():
        route = paths_encoded_copy[k][0]
        route_new = []
        poi_route = []
        common_segments = []
        od_pair = paths_encoded_copy[k][-1]
        omitted = []
        if not (od_pair[0] is None or od_pair[1] is None):
            # identify first if the od
            for ij in range(0, len(route)-1):
                first_node = route[ij]
                secd_node = route[ij+1]



                # for each route segment: see if both points on same segment
                # if not -> do nothing (i.e. continue);
                # if yes ->
                # (1) form route segment, (2) get segment on which it lies, (3) if no similar segment, than continue
                # (4) otherwise extract pois which lie within these segments and add pois
                extract_fn = pn.loc[first_node]
                on_segmnents_fn = [extract_fn.on_segment]
                if not on_segmnents_fn[0] > 0:
                    on_segmnents_fn = list(extract_fn.on_segment0)

                if not any([True if el > 0 else False for el in on_segmnents_fn]):
                    omitted.append((first_node, secd_node))
                    continue

                extract_sc = pn.loc[secd_node]
                on_segmnents_sc = [extract_sc.on_segment]
                if not on_segmnents_sc[0] > 0:
                    on_segmnents_sc = list(extract_sc.on_segment0)

                if not any([True if el > 0 else False for el in on_segmnents_sc]):
                    omitted.append((first_node, secd_node))
                    continue

                common_segment = 999
                all_segments = list(set(on_segmnents_fn + on_segmnents_sc))
                sth_found = False
                for elem in all_segments:
                    if elem in on_segmnents_fn and elem in on_segmnents_sc:
                        common_segment = elem
                        route_new.append((first_node, secd_node))
                        common_segments.append(elem)
                        sth_found = True
                        break
                if not sth_found:
                     omitted.append((first_node, secd_node))

                # if od_pair in l_od_pairs:
                #     route_new.append((15, 29))
                #     common_segments.append(18)
            omitted_items[od_pair] = omitted
            # collect all pois
            very_first_poi = None
            very_last_poi = None
            poi_route_lim = []
            for ij in range(0, len(route_new)):
                first_node = route_new[ij][0]
                secd_node = route_new[ij][1]
                common_segment = common_segments[ij]

                # identify poi_ID
                # route_new is (od_id)


                # identify whether link/
                output1 = identify_point(
                    first_node, projected_nodes, pois_new, highway_junctions, highway_geometry
                )
                if ij == 0:
                    very_first_poi = output1[0][0]

                output2 = identify_point(
                    secd_node, projected_nodes, pois_new, highway_junctions, highway_geometry
                )
                if ij == len(route_new)-1:
                    very_last_poi = output2[0][0]

                # common_segment = list(set(output1[1]).intersection(output2[1]))
                print(output1, output2)
                #   (2) determine driving direction     --
                # based on this,
                if len(output1) > 1:
                    extract_1 = pois_new[
                        (pois_new.segment_id == common_segment)
                        & pois_new.ID.isin(output1[0])
                        ]
                    if len(extract_1) > 0:
                        poi_id_1 = extract_1.ID.to_list()[0]
                else:
                    poi_id_1 = output1[0][0]

                if len(output2) > 1:
                    extract_2 = pois_new[
                        (pois_new.segment_id == common_segment)
                        & pois_new.ID.isin(output2[0])
                        ]
                    if len(extract_2) > 0:
                        poi_id_2 = extract_2.ID.to_list()[0]
                else:
                    poi_id_2 = output2[0][0]

                if poi_id_1 < poi_id_2:
                    direction = 0
                else:
                    direction = 1

                pois_extract = pois_new[pois_new.segment_id == common_segment]
                if direction == 1:
                    pois_extract["abv"] = list(range(0, len(pois_extract)))
                    pois_extract = pois_extract.sort_values(
                        by=["abv"], ascending=[False]
                    )

                # filter by direction
                pois_extract = pois_extract[pois_extract.dir.isin([direction, 2])]

                # filter for the relevant indices
                if direction == 0:
                    pois_extract = pois_extract[
                        (pois_extract.index >= poi_id_1) & (pois_extract.index <= poi_id_2)
                        ]
                    pois_extract = pois_extract[pois_extract.dir.isin([direction, 2])]
                else:
                    pois_extract = pois_extract[
                        (pois_extract.index <= poi_id_1) & (pois_extract.index >= poi_id_2)
                        ]
                    pois_extract = pois_extract[pois_extract.dir.isin([direction, 2])]

                pois_extract_lim = pois_extract[pois_extract.pois_type.isin(["link", "od"])]
                indices = pois_extract.index.to_list()
                # if (extract_df_1.index.to_list()[0], extract_df_2.index.to_list()[0]) == (30, 80):
                #     #(k)
                #     break

                for kl in range(0, len(pois_extract) - 1):
                    poi_route.append((indices[kl], indices[kl + 1]))

                indices_lim = pois_extract_lim.index.to_list()
                for kl in range(0, len(indices_lim) - 1):
                    poi_route_lim.append((indices_lim[kl], indices_lim[kl + 1]))
            very_first_poi = pois_new[(pois_new.type_ID == k[0]) & (pois_new.pois_type == "od")].index.to_list()[0]
            very_last_poi = pois_new[(pois_new.type_ID == k[1]) & (pois_new.pois_type == "od")].index.to_list()[0]
            if k in paths_encoded:
                paths_encoded_pois[
                    (very_first_poi, very_last_poi)
                ] = poi_route
            paths_encoded_pois_lim[
                (very_first_poi, very_last_poi)
            ] = poi_route_lim
            paths_encoded_pois_lim_and_length[
                (very_first_poi, very_last_poi)
            ] = (poi_route_lim, paths_encoded_to_od[od_pair][2], od_pair)
                # paths_encoded_pois[(route[0], route[1])] = poi_route
            paths_encoded_to_od[od_pair] = (route_new, paths_encoded_to_od[od_pair][1], paths_encoded_to_od[od_pair][2])


    # SAVING
    if case_study_workday:
        with open("data\paths_encoded_pois_lim.pickle", "wb") as handle:
            pickle.dump(paths_encoded_pois_lim, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open("data\paths_encoded_pois_lim_and_length.pickle", "wb") as handle:
            pickle.dump(paths_encoded_pois_lim_and_length, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("data\paths_encoded_pois_lim_holiday.pickle", "wb") as handle:
            pickle.dump(paths_encoded_pois_lim, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open("data\paths_encoded_pois_lim_and_length_holiday.pickle", "wb") as handle:
            pickle.dump(paths_encoded_pois_lim_and_length, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if case_study_workday:
        with open("data\paths_encoded_to_od.pickle", "wb") as handle:
            pickle.dump(paths_encoded_to_od, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("data\paths_encoded_to_od_holiday.pickle", "wb") as handle:
            pickle.dump(paths_encoded_to_od, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open("data\omitted_items.pickle", "wb") as handle:
        pickle.dump(omitted_items, handle, protocol=pickle.HIGHEST_PROTOCOL)


    nuts_id_list = list(set(nuts_ids))
    if None in nuts_id_list:
        nuts_id_list.remove(None)


    total_travels = {}
    all_total_travels = {}
    total_travels_od = {}
    OD_purposes = {}

    for k in paths_encoded_copy.keys():
        p1 = k[0]
        p2 = k[1]
        nb_travels_by_purpose = {}
        # get nuts_id of both
        origin_nut = projected_nodes[projected_nodes.nodeID == p1]["NUTS_ID"].to_list()[0]
        destination_nut = projected_nodes[projected_nodes.nodeID == p2][
            "NUTS_ID"
        ].to_list()[0]
        sum_travels = 0

        # pendler
        if (origin_nut, destination_nut) in traffic_flow.keys() and not origin_nut == destination_nut:
            sum_travels = sum_travels + float(traffic_flow[(origin_nut, destination_nut)])

            if k in paths_encoded_copy.keys():
                if not (p1, p2) in total_travels:
                    total_travels[(p1, p2)] = sum_travels
                    total_travels_od[(origin_nut, destination_nut)] = sum_travels
                else:
                    total_travels[(p1, p2)] = total_travels[(p1, p2)] + sum_travels
                    total_travels_od[(origin_nut, destination_nut)] = total_travels_od[(origin_nut, destination_nut)] + sum_travels
            if not (p1, p2) in total_travels:
                all_total_travels[(p1, p2)] = sum_travels
            else:
                all_total_travels[(p1, p2)] = total_travels[(p1, p2)] + sum_travels

    # SAVING
    if case_study_workday:
        with open("data/total_travels.pickle", "wb") as handle:
            pickle.dump(total_travels, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("data/total_travels_od.pickle", "wb") as handle:
            pickle.dump(total_travels_od, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("data/total_travels_holiday.pickle", "wb") as handle:
            pickle.dump(total_travels, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("data/total_travels_od_holiday.pickle", "wb") as handle:
            pickle.dump(total_travels_od, handle, protocol=pickle.HIGHEST_PROTOCOL)


    total_travels_poi_encoded = {}
    transit_poi_connections = []
    for k in total_travels.keys():
        p1 = k[0]
        p2 = k[1]
        extract_df_1 = pois_new[(pois_new.type_ID == p1) & (pois_new.pois_type == "od")]
        extract_df_2 = pois_new[(pois_new.type_ID == p2) & (pois_new.pois_type == "od")]
        if len(extract_df_1) > 0 and len(extract_df_2) > 0:
            total_travels_poi_encoded[
                (extract_df_1.index.to_list()[0], extract_df_2.index.to_list()[0])
            ] = total_travels[k]

    # SAVING
    if case_study_workday:
        with open("data/total_travels_poi_encoded.pickle", "wb") as handle:
            pickle.dump(total_travels_poi_encoded, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("data/total_travels_poi_encoded_holiday.pickle", "wb") as handle:
            pickle.dump(total_travels_poi_encoded, handle, protocol=pickle.HIGHEST_PROTOCOL)


    seg_ids = list(set(pois.segment_id.to_list()))

    for s in seg_ids:
        direction = 0
        extract_df = pois_new[
            (pois_new.segment_id == s) & (pois_new.dir.isin([direction, 2]))
        ]
        inds = extract_df.index.to_list()

        demand = []
        # go through adjacent poi pairs
        # check for all pairs if it is part of a route
        # include check if something at same distance

        demand.append(0)
        traffic = []
        dists_along_route = extract_df.dist_along_segment.to_list()
        ids = extract_df.ID.to_list()
        n = len(ids)

        for ij in range(1, n):
            beg = ids[ij - 1]
            end = ids[ij]
            route_seg = [(beg, end)]

            # collect other permutations
            out_beg = is_same_distance(pois_new, s, beg)
            out_end = is_same_distance(pois_new, s, end)
            if beg in out_end:
                out_end.remove(beg)
            for el1 in out_beg:
                for el2 in out_end:
                    route_seg.append((el1, el2))

            # now check in encoded poi path if it is

            num_veh = 0
            for k in paths_encoded_pois.keys():
                if k in paths_encoded_pois.keys() and k in total_travels_poi_encoded.keys():
                    if any(x in route_seg for x in paths_encoded_pois[k]):
                        # if true, then collect number of vehicles
                        num_veh = num_veh + total_travels_poi_encoded[k]

            demand.append(
                np.abs(dists_along_route[ij - 1] - dists_along_route[ij])
                * num_veh
                * (1 / 100000)
            )  # vehicles * [m])
            if ij == 1:
                traffic.append(num_veh)
            traffic.append(num_veh)

        pois_new.loc[ids, "demand_0"] = demand
        pois_new.loc[ids, "tc_0"] = traffic

        direction = 1
        extract_df = pois_new[
            (pois_new.segment_id == s) & (pois_new.dir.isin([direction, 2]))
        ]
        inds = extract_df.index.to_list()
        demand = []
        # go through adjacent poi pairs
        # check for all pairs if it is part of a route
        # include check if something at same distance

        demand.append(0)
        traffic = []
        extract_df = extract_df.iloc[::-1]
        dists_along_route = extract_df.dist_along_segment.to_list()
        ids = extract_df.ID.to_list()
        n = len(ids)

        for ij in range(1, n):
            beg = ids[ij - 1]
            end = ids[ij]
            route_seg = [(beg, end)]

            # collect other permutations
            out_beg = is_same_distance(pois_new, s, beg)
            out_end = is_same_distance(pois_new, s, end)
            if beg in out_end:
                out_end.remove(beg)
            for el1 in out_beg:
                for el2 in out_end:
                    route_seg.append((el1, el2))

            # now check in encoded poi path if it is

            num_veh = 0
            for k in paths_encoded_pois.keys():
                if k in paths_encoded_pois.keys() and k in total_travels_poi_encoded.keys():
                    if any(x in route_seg for x in paths_encoded_pois[k]):
                        # if true, then collect number of vehicles
                        num_veh = num_veh + total_travels_poi_encoded[k]

            demand.append(
                np.abs(dists_along_route[ij - 1] - dists_along_route[ij])
                * num_veh
                * (1 / 100000)
            )  # vehicles * [m])
            if ij == 1:
                traffic.append(num_veh)
            traffic.append(num_veh)

        pois_new.loc[ids, "demand_1"] = demand
        pois_new.loc[ids, "tc_1"] = traffic

    pois_new["tc_0"] = pois_new["tc_0"].fillna(0)
    pois_new["tc_1"] = pois_new["tc_1"].fillna(0)
    pois_new["demand_0"] = pois_new["demand_0"].fillna(0)
    pois_new["demand_1"] = pois_new["demand_1"].fillna(0)
    #
    if case_study_workday:
        pois_new.to_csv("data/_demand_for_FC_workday_20220722.csv")
    else:
        pois_new.to_csv("data/_demand_for_FC_holiday_20220722.csv")

create_maximum_dem_file()