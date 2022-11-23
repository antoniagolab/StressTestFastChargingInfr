"""

python file for import of geographic data

"""


from utils import *

path = "C:/Users/golab/PycharmProjects/HighwayChargingSimulation/"
nuts_4 = gpd.read_file(path + "geography/NUTS_RG_03M_2021_3857.shp")
