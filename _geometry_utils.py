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
