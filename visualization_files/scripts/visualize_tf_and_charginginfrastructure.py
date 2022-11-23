"""

Visualization of traffic flow and charging infrastructure side-by-side

.. making a plot with two supplots
.. add

"""
import pandas as pd

from utils import *
import geopandas as gpd
from shapely.ops import split
from shapely.geometry import MultiPoint

highway_geometry = pd2gpd(pd.read_csv("geography/highway_segments.csv"))
pois = pd.read_csv("data/_demand_for_FC_holiday.csv")
highway_geometry = highway_geometry[~highway_geometry.ID.isin([0, 1, 2])]
segments = highway_geometry.ID.to_list()
geoms = highway_geometry.geometry.to_list()
to_plot = pd.DataFrame()

for kl in range(0, len(segments)):
    s = segments[kl]
    extract_pois = pois[pois.segment_id == s]
    extract_pois = extract_pois[(extract_pois.pois_type.isin(["link", "od"]))]
    extract_pois = extract_pois.sort_values(by=["dist_along_segment"])
    extract_pois = extract_pois.drop_duplicates(subset=["dist_along_segment"], keep="first")
    dists = extract_pois.dist_along_segment.to_list()
    geom = geoms[kl]
    points = [geom.interpolate(d) for d in dists]
    geom_coll = split(geom, points)
    for ij in range(0, len(geom_coll)):
        g = geom_coll[0]


# figure 1
# get segments
# split segments and links and od (be careful, some dist_along_segment(od) = 0.0
# assign a linewidth to each part segment according to traffic flow
#


# figure 2
# plot highways
# plot