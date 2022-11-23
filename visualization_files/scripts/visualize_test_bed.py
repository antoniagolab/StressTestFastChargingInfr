"""
Created July 20th, 2022
"""
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint, Polygon, GeometryCollection
from utils import *
from matplotlib_scalebar.scalebar import ScaleBar
from shapely.ops import split, snap
import geopandas as gpd
# TODO:
#   - nuts
#   - boarder
#   - highways



# getting highway network
hs = pd.read_csv("geography/highway_segments.csv")
hs["geometry"] = hs["geometry"].apply(wkt.loads)
hs = gpd.GeoDataFrame(hs, geometry="geometry", crs=reference_coord_sys)

hs = hs.to_crs("EPSG:3857")
# NUTS-3 regions

# getting NUTS-3 regions
nuts_3 = gpd.read_file("NUTS_RG_03M_2021_3857/NUTS_RG_03M_2021_3857.shp")

# filter the regions after AT and NUTS-3
nuts_at = nuts_3[nuts_3["CNTR_CODE"] == "AT"]

nuts_3_at =nuts_at[nuts_at["LEVL_CODE"] == 3]   # alternatively insert value "2"

# exclude not relevant NUTS_3 regions
# _not_relevant_region_codes = [342, 341, 331, 334, 332, 335]
# _not_relevant_region_codes = [33, 34]
# _not_relevant_region_codes_with_AT = ["AT" + str(el) for el in _not_relevant_region_codes]

nuts_3_at["geometry"] = nuts_3_at.geometry.exterior
extract = hs[~hs.ID.isin([0,1,2])]
env = GeometryCollection(extract.geometry.to_list()).envelope
# env_pol = Polygon([(env[0], env[1]), (env[2], env[1]), (env[2], env[3]), (env[0], env[3]), (env[0], env[1])])

austrian_boarder =  nuts_at[nuts_at["NUTS_NAME"] == "Österreich"]
austrian_boarder.geometry = austrian_boarder.geometry.exterior

fig, ax = plt.subplots(figsize=(10, 5))
plt.rcParams["font.family"] = "Franklin Gothic Book"
plt.rcParams["font.size"] = 12
hs.plot(ax=ax, color="black", label="Austrian highway network", linewidth=2)
austrian_boarder.plot(ax=ax, color="#274c77", zorder=10, linewidth=2, label="Austrian boarder")
ax.plot(*env.exterior.xy, linewidth=4, color="#660708")
ax.add_artist(ScaleBar(1))
nuts_3_at.plot(ax=ax, color="#adb5bd", zorder=0, label="NUTS-3 regions", alpha=0.8)
ax.legend()
ax.axis("off")
plt.savefig("figures/test_bed_vis.pdf", bbox_inches="tight")
plt.savefig("figures/test_bed_vis.svg", bbox_inches="tight")
