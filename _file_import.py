from utils import *
import pandas as pd

path = "C:/Users/golab/PycharmProjects/HighwayChargingSimulation/"

segments_gdf = pd2gpd(pd.read_csv(path + "geography/highway_segments.csv"))
pois_df = pd.read_csv(path + "data/_demand_calculated.csv")

segments_gdf = segments_gdf[~segments_gdf.ID.isin([0, 1, 2])]
pois_df = pois_df[~pois_df.segment_id.isin([0, 1, 2])]


seg_ids = segments_gdf.ID.to_list()

# model specifics

delta_t = 0.25  # (h) time resolution of model
v = 100  # (km/h) average driving speed
