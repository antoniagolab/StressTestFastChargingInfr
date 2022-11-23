"""

 This script creates representations for the geometry of the highway network, following a given assumed driving speed and set temporal resolution of the model

"""
import pandas as pd
from pyomo.environ import *
import pickle
from copy import deepcopy

# get all relevant segments
# also the prepared POI ways and all what is connected to this in order to determine the optimal segmentation



# assumptions

velocity = 110
delta_t = 0.25
delat_x = velocity * delta_t

# calculate real travel time between two points OD-nodes in the system
# The segmentations in done through
#   - I need the distance between all O-D pairs (see paths_..-pickle)
#   -> calculate travel times betwenn all
#   - get poi path + Teilsegmente
#   - decision variables = n for each Teilsegment + trvl-tme for each route
with open("data/paths_encoded_pois_lim_and_length.pickle", "rb") as handle:
    paths_encoded_pois_lim_and_length = pickle.load(handle)

pois_new = pd.read_csv("data/pois_new.csv")
cr = pd.read_csv("data\cell_routes.csv")
cr["start_end"] = [eval(item) for item in cr["start_end"].to_list()]
# TODO: find issue here!
for k in paths_encoded_pois_lim_and_length:
    val = paths_encoded_pois_lim_and_length[k]
    ts_sequence = []
    val_news = []
    for v in val[0]:
        v_orig = deepcopy(v)
        if v == val[0][-1] and v[1] == k[1]+1 and v[0] < v[1]:
            v = (v[0], k[1])
        elif v == val[0][-1] and v[1] == k[1]-1 and v[0] > v[1]:
            v = (v[0], k[1])
        elif v == val[0][0] and v[0] == k[0]+1 and v[0] > v[1]:
            v = (k[0], v[1])

        if v[0] == 30:
            v = (31, v[1])
        if v[1] == 30:
            v = (v[0], 31)
        if v == (134, 136):
            v = (86, 88)
        if v[0] == v[1]:
            continue
        extract_cr = cr[(cr.start_end == v) | (cr.start_end == (v[1], v[0]))]
        val_news.append(v)
        ts_sequence.append(extract_cr.index.to_list()[0])
    paths_encoded_pois_lim_and_length[k] = paths_encoded_pois_lim_and_length[k] + (val_news, ts_sequence, )

model = ConcreteModel()
model.nb_routes = range(0, len(paths_encoded_pois_lim_and_length.keys()))
model.nb_ts = range(0, len(cr))

model.keys()

# TODO: match the teilsegments to the entries in path

# iterate through all key in paths and make: t = (1/v) * s
model.travel_time = Var(model.nb_routes, within=NonNegativeReals)
model.n = Var(model.nb_ts, within=NonNegativeIntegers)
model.t = Var(model.nb_routes, within=NonNegativeReals)
model.c3 = ConstraintList()
for ij in model.nb_ts:
    model.c3.add(model.n[ij] >= 1)
model.c = ConstraintList()
key_list = list(paths_encoded_pois_lim_and_length.keys())
for ij in range(0, len(paths_encoded_pois_lim_and_length.keys())):
    #if ij in range(23, 26):
    model.c.add(model.travel_time[ij] == quicksum(model.n[kl] * delta_t for kl in paths_encoded_pois_lim_and_length[key_list[ij]][-1]))

routes = [paths_encoded_pois_lim_and_length[key_list[ij]][-1] for ij in range(0, len(key_list))]
travel_times = [paths_encoded_pois_lim_and_length[key_list[ij]][1] * (1/velocity)/1000 for ij in range(0, len(key_list))]
model.c2 = ConstraintList()
for ij in model.nb_routes:
    model.c2.add(expr=(model.t[ij] >= (model.travel_time[ij] - paths_encoded_pois_lim_and_length[key_list[ij]][1] * (1/velocity)/1000)))
    model.c2.add(expr=(model.t[ij] >= - (model.travel_time[ij] - paths_encoded_pois_lim_and_length[key_list[ij]][1] * (1/velocity)/1000)))

model.obj = Objective(expr=(quicksum(model.t[ij] for ij in model.nb_routes)), sense=minimize)

opt = SolverFactory("gurobi")
opt_success = opt.solve(
    model, logfile="log/" + "_log.txt", report_timing=True, tee=True
)
n = [model.n[kl].value for kl in model.nb_ts]
tt = [model.travel_time[ij].value for ij in model.nb_routes]
errors = [model.t[ij].value for ij in model.nb_routes]
print(n)
print(tt)
print(min(errors), max(errors))
# TODO: delete the Teilsegment that are not used

cr["cell_nb"] = n
cr.to_csv("data/cellularized.csv")
# sasave data

with open("data/paths_encoded_matching_route.pickle", "wb") as handle:
    pickle.dump(paths_encoded_pois_lim_and_length, handle, protocol=pickle.HIGHEST_PROTOCOL)