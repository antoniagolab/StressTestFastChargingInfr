"""

Script for the pre-processing of Austria's traffic data

"""
import geopandas as gpd
import pandas as pd
from utils import *

gemeinden = gpd.read_file("C:/Users\golab\PycharmProjects\HighwayChargingSimulation\OGDEXT_GEM_1_STATISTIK_AUSTRIA_20220101/STATISTIK_AUSTRIA_GEM_20220101.shp")
traffix_data = pd.read_csv("data/Ergebnismatrix_Kal_pkw_6421_24h_220506.txt", skiprows=range(0, 4135781), header=None, delimiter=" ")

gemeinden_codes = gemeinden.id.to_list()
gemeinden_codes_str = [str(g) for g in gemeinden_codes]

ind = traffix_data.index.to_list()
n = len(traffix_data)

for ij in range(0, n):
    netz_obj_id = traffix_data[0].loc[ind[ij]]
    if str(netz_obj_id)[0:5] in gemeinden_codes_str:
        for kl in range(0, len(gemeinden)):
            if str(netz_obj_id)[0:5] == gemeinden_codes_str[kl]:
                traffix_data.loc[ind[ij], "gemeinde_code"] = gemeinden_codes_str[kl]

# matching NUTS- 3 u. Gemeinde

nuts = pd.read_excel(
    "geography/nuts_3_mit_gemeinden_flaechen_und_bevoelkerung_gebietsstand_1.1.2021.xlsx",
    header=2,
    skipfooter=2,
)
country_codes = ["DE", "CZ", "IT", "HR", "SK", "SI", "HU"]
gemeinde_codes_traffix = traffix_data.gemeinde_code.to_list()
name = traffix_data[1].to_list()
# print(name[-30][0:2])
for ij in range(0, n):
    gem_code = gemeinde_codes_traffix[ij]
    if len(str(gem_code)) > 0:
        nuts_3_df = nuts[nuts['LAU 2 - Code Gemeinde- kennziffer'] == float(gem_code)]
        if len(nuts_3_df) > 0:
            nuts_3 = nuts_3_df['NUTS 3-Code'].to_list()[0]
            traffix_data.loc[ind[ij], "NUTS-3 region"] = nuts_3
        elif str(gem_code)[0] == "9" and not name[ij][0:2] in country_codes:
            traffix_data.loc[ind[ij], "NUTS-3 region"] = "AT130"
        elif name[ij][0:2] in country_codes:
            print(name[ij][0:2])
            for mn in range(0, len(country_codes)):
                if name[ij][0:2] == country_codes[mn]:
                    traffix_data.loc[ind[ij], "NUTS-3 region"] = country_codes[mn]

print("done 1")

traffix_data.to_csv("data/matched_gemeinde_traffix_data.csv")

od_geom = gpd.read_file("data/OD_centroids.shp")
od_pairs_of_interest = od_geom.NUTS_ID.to_list() + country_codes

# if SI -> SI + HR add up!

# calculate for each OD pair
fn = "data/Ergebnismatrix_Kal_pkw_6421_24h_220506.txt"
projected_nodes = gpd.read_file("data/all_projected_nodes.shp")
projected_nodes = projected_nodes[~projected_nodes.NUTS_ID.isna()]
od_of_interest = country_codes + projected_nodes.NUTS_ID.to_list()
print(od_of_interest)
parse_OD_data(fn, od_of_interest)
print("done 2")
split_traffic_flow_transit()

print("done 3")