
from pyomo.environ import *
import numpy as np
from utils import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt
# from _import_geographic_data import *
from matplotlib.offsetbox import AnchoredText
# from _import_optimization_files import *
import time
from pyomo.core.util import quicksum
from datetime import datetime


# TODO:
#   - travel time function  --check
#   - function creating all decision variables  --check
#   - constraint between two nodes  --check
#   - function running over all node pairing on a segment   --check
#   - function retrieving cell geometries for each direction and adding this to a dataframe
#   - function constraining positive integer    --check
#   - function adding objective function    --check


def calculate_travel_time(node_0_id, node_1_id, pois_df, v=100):
    """
    function calculating travel time between two nodes on a segment
    :param seg_id:
    :param node_0_id:
    :param node_1_id:
    :param segments_gdf:
    :param pois_df:
    :param v: driving speed in (km/h)
    :return:
    """
    node_0_dist = pois_df.loc[node_0_id].dist_along_segment
    node_1_dist = pois_df.loc[node_1_id].dist_along_segment
    if node_0_dist > node_1_dist:
        driving_distance = node_0_dist - node_1_dist
    else:
        driving_distance = node_1_dist - node_0_dist

    if not driving_distance == 0:
        travel_time = (driving_distance / 1000) / v
    else:
        travel_time = 0
    return travel_time


def add_decision_vars(model, seg_id, direction, pois_df, segments_df):
    """
    setting decision variables for segmentation
    :param model:
    :param seg_id:
    :param direction:
    :param pois_df:
    :param segments_df:
    :return:
    """
    poi_extract = pois_df[pois_df.segment_id == seg_id]
    poi_extract = poi_extract[poi_extract.dir.isin([direction, 2])]
    nb_pois = len(poi_extract)
    nb_highway_sections = nb_pois - 1
    model.nb_pois = range(nb_pois)
    model.g = range(nb_highway_sections)
    model.m = Var(model.g, within=Integers)
    # model.m = Var(model.g)
    model.travel_times = Var(model.nb_pois, model.nb_pois)
    model.error = Var(model.nb_pois, model.nb_pois)


def constraint_zero(model):
    """
    adding zero-constraint to model
    :param model: pyomo.environ.ConcreteModel
    :return:
    """
    model.constraint_zero = ConstraintList()
    for ij in model.g:
        model.constraint_zero.add(model.m[ij] >= 0)

    for ij in model.nb_pois:
        for kl in model.nb_pois:
            model.constraint_zero.add(model.travel_times[ij, kl] >= 0)


def constraint_travel_time(model, ij, kl, delta_t=0.25):
    """

    :param model:
    :param ij:
    :param kl:
    :param delta_t:
    :return:
    """
    if ij == kl:
        model.constr_zero.add(model.travel_times[ij, kl] == 0)
    else:
        if ij < kl:
            model.constraint_travel_time.add(
                model.travel_times[ij, kl]
                == sum([model.m[mn] for mn in range(ij, kl)]) * delta_t
            )
            model.constraint_travel_time.add(
                model.travel_times[ij, kl] == model.travel_times[kl, ij]
            )
        else:
            model.constraint_travel_time.add(
                model.travel_times[ij, kl]
                == sum([model.m[mn] for mn in range(kl, ij)]) * delta_t
            )


def constraint_model_error(model, real_travel_times):
    model.constraint_error = ConstraintList()
    for ij in model.nb_pois:
        for kl in model.nb_pois:
            model.constraint_error.add(
                model.error[ij, kl]
                >= real_travel_times[ij, kl] - model.travel_times[ij, kl]
            )
            model.constraint_error.add(
                model.error[ij, kl]
                >= -(real_travel_times[ij, kl] - model.travel_times[ij, kl])
            )


def constraint_travel_time_for_all(model, delta_t=0.25):
    """
    constraining travel time between all POI combinations
    :param model:
    :param delta_t:
    :return:
    """
    model.constraint_travel_time = ConstraintList()
    model.constr_zero = ConstraintList()
    for ij in model.nb_pois:
        for kl in model.nb_pois:
            # if kl >= ij:
            constraint_travel_time(model, ij, kl, delta_t=delta_t)


def add_objective_function(model, seg_id, direction, pois_df, v=100):
    """
    adding objective function to the model which aims to minimize the difference between real travel times and travel
    times resulting from the segmentation
    :param model:
    :param seg_id:
    :param direction:
    :param pois_df:
    :param v:
    :return:
    """
    # calculating matrix of real travel times
    real_travel_times = np.zeros([len(model.nb_pois), len(model.nb_pois)])
    IDs = pois_df[
        (pois_df.segment_id == seg_id) & (pois_df.dir.isin([direction, 2]))
    ].ID.to_list()
    if direction == 1:
        IDs.reverse()
    for ij in model.nb_pois:
        for kl in model.nb_pois:
            if kl >= ij:
                node_0_id = IDs[ij]
                node_1_id = IDs[kl]
                real_travel_times[ij, kl] = calculate_travel_time(
                    node_0_id, node_1_id, pois_df, v=v
                )
                real_travel_times[kl, ij] = real_travel_times[ij, kl]

    constraint_model_error(model, real_travel_times)

    # model.obj_fun = Objective(expr=(sum([model.error[ij, kl] for ij in model.nb_pois for kl in model.nb_pois])),
    #                           sense=minimize)
    model.obj_fun = Objective(
        expr=(
            sum(
                [
                    model.error[ij, kl]
                    for ij in model.nb_pois
                    for kl in model.nb_pois
                    if ij <= kl
                ]
            )
        ),
        sense=minimize,
    )
    # approach after:
    # https://math.stackexchange.com/questions/1954992/linear-programming-minimizing-absolute-values-and-formulate-in-lp
    # model.obj_fun = Objective(expr=(real_travel_times[0, 1] - model.travel_times[0, 1]), sense=minimize)

    return real_travel_times


def save_to_df(df, seg_id, direction, pois_df, model, v, delta_t):
    # TODO:
    #   - for each cell: seg_id, direction, start_dist, end_dist, type of cell (stop, or no stop)
    #   -   type of cell: (0 - no charging, no driving; 1 - no charging, driving; 2 - charging possible, no driving)
    df_list = []
    pois_extract = pois_df[
        (pois_df.segment_id == seg_id) & (pois_df.dir.isin([direction, 2]))
    ]
    m_cells = np.array([model.m[ij].value for ij in model.g])
    df_list.append(
        {
            "segment_id": seg_id,
            "dir": direction,
            "start_dist": 0,
            "end_dist": 0,
            "type_of_cell": 0,
        }
    )

    if direction == 0:
        distances = pois_df.dist_along_segment.to_list()

    else:
        distances = reverse_distances(pois_df.dist_along_segment.to_list())

    start_dist_before = 0
    for ij in range(1, len(pois_extract)):
        width = m_cells[ij - 1] * (v * delta_t * 1000)
        df_list.append(
            {
                "segment_id": seg_id,
                "dir": direction,
                "start_dist": start_dist_before,
                "end_dist": start_dist_before + width,
                "type_of_cell": 1,
            }
        )
        if ij < len(pois_extract) - 1:
            df_list.append(
                {
                    "segment_id": seg_id,
                    "dir": direction,
                    "start_dist": start_dist_before + width,
                    "end_dist": start_dist_before + width,
                    "type_of_cell": 2,
                }
            )
        else:
            df_list.append(
                {
                    "segment_id": seg_id,
                    "dir": direction,
                    "start_dist": start_dist_before + width,
                    "end_dist": start_dist_before + width,
                    "type_of_cell": 0,
                }
            )
        start_dist_before = start_dist_before + width
    df = df.append(pd.DataFrame(df_list))
    return df


def create_cellular_geometry_for_segment():
    return None


# TODO:
def visualize_segmentation(seg_id, cell_df, segments_df, pois_df):
    """
    visualization function for segmentation
    :param seg_id:
    :param cell_df:
    :param segments_df:
    :param pois_df:
    :return:
    """
    fig, axs = plt.subplots(1, 2)
    nuts_4.plot(ax=axs[0])
    nuts_4.plot(ax=axs[1])
    segments_df[segments_df.ID == seg_id].plot(ax=axs[0])
    # for visualization of model:

    plt.show()

    return fig


def limiting_n_incoming_vehicles(model: ConcreteModel, t, c, f):
    return model.n_incoming_vehicles[t, c, f] == 0


def limiting_n_in(model: ConcreteModel, t, c, f):
    return model.n_in[t, c, f] == 0


def limiting_n_pass(model: ConcreteModel, t, c, f):
    return (model.n_pass[t, c, f] == 0)


def limiting_n_exit(model: ConcreteModel, t, c, f):
    return model.n_exit[t, c, f] == 0


def limiting_n_out(model: ConcreteModel, t, c, f):
    return model.n_out[t, c, f] == 0


def limiting_n_in_wait_charge(model: ConcreteModel, t, c, f):
    return model.n_in_wait_charge[t, c, f] == 0


def limiting_n_arrived_vehicles(model: ConcreteModel, t, c, f):
    return model.n_arrived_vehicles[t, c, f] == 0


def limiting_n_in_wait(model: ConcreteModel, t, c, f):
    return model.n_in_wait[t, c, f] == 0


def limiting_n_wait(model: ConcreteModel, t, c, f):
    return model.n_wait[t, c, f] == 0


def limiting_n_wait_charge_next(model: ConcreteModel, t, c, f):
    return model.n_wait_charge_next[t, c, f] == 0


def limiting_n_in_charge(model: ConcreteModel, t, c, f):
    return model.n_in_charge[t, c, f] == 0


def limiting_n_charge1(model: ConcreteModel, t, c, f):
    return model.n_charge1[t, c, f] == 0


def limiting_n_charge2(model: ConcreteModel, t, c, f):
    return model.n_charge2[t, c, f] == 0


def limiting_n_charge3(model: ConcreteModel, t, c, f):
    return model.n_charge3[t, c, f] == 0


def limiting_n_output_charged1(model: ConcreteModel, t, c, f):
    return model.n_output_charged1[t, c, f] == 0


def limiting_n_output_charged2(model: ConcreteModel, t, c, f):
    return model.n_output_charged2[t, c, f] == 0


def limiting_n_output_charged3(model: ConcreteModel, t, c, f):
    return model.n_output_charged3[t, c, f] == 0


def limiting_n_finished_charge1(model: ConcreteModel, t, c, f):
    return model.n_finished_charge1[t, c, f] == 0


def limiting_n_finished_charge2(model: ConcreteModel, t, c, f):
    return model.n_finished_charge2[t, c, f] == 0


def limiting_n_finished_charge3(model: ConcreteModel, t, c, f):
    return model.n_finished_charge3[t, c, f] == 0


def limiting_n_finished_charging(model: ConcreteModel, t, c, f):
    return model.n_finished_charging[t, c, f] == 0


def limiting_n_exit_charge(model: ConcreteModel, t, c, f):
    return model.n_exit_charge[t, c, f] == 0










def add_decision_variables(
    model: ConcreteModel,
    time_resolution: float,
    nb_fleets: int,
    nb_cells: int,
    nb_timesteps: int,
    SOC_min: float,
    SOC_max: float,
    fleet_df: pd.DataFrame,
    cell_df: pd.DataFrame,
    t_min: float,
):
    """

    :param model:
    :param time_resolution:
    :param nb_fleets:
    :param nb_cells:
    :param nb_timesteps:
    :param SOC_min:
    :param SOC_max:
    :return:
    """
    model.nb_fleets = nb_fleets
    model.nb_cells = nb_cells
    model.nb_timesteps = nb_timesteps
    model.nb_fleet = range(0, nb_fleets)
    model.nb_cell = range(0, nb_cells)
    model.nb_timestep = range(0, nb_timesteps)
    model.time_resolution = time_resolution
    model.SOC_min = SOC_min
    model.SOC_max = SOC_max
    model.t_min = t_min
    model.fleet_df = fleet_df
    model.fleet_routes = fleet_df["route"].to_list()
    model.fleet_depart_times = fleet_df["start_timestep"].to_list()
    model.cell_width = cell_df["length"].array
    model.cell_charging_cap = cell_df["capacity"].array
    t0 = time.time()
    model.key_set = Set(initialize=create_set_init)
    print("the key creation took.. ", time.time() - t0, "sec")
    print("Nb. of keys", len(model.key_set))
    model.charging_cells_key_set = set([key for key in model.key_set if model.cell_charging_cap[key[1]] > 0])
    model.key_routing_set = Set(initialize=create_set_routing)

    # for all cells
    model.n_incoming_vehicles = Var(model.key_set, within=NonNegativeReals)
    model.n_in = Var(model.key_set, within=NonNegativeReals)
    # n_in_pass == n_pass   -- check
    # n_exit_pass(t) = n_pass(t-1)
    model.n_pass = Var(model.key_set, within=NonNegativeReals)
    # model.n_exit_pass = Var(model.key_set, within=NonNegativeReals)
    model.n_exit = Var(model.key_set, within=NonNegativeReals)
    model.n_out = Var(model.key_set, within=NonNegativeReals)
    model.n_arrived_vehicles = Var(model.key_set, within=NonNegativeReals)
    model.Q_incoming_vehicles = Var(model.key_set, within=NonNegativeReals)
    model.Q_in = Var(model.key_set, within=NonNegativeReals)
    # model.n_in_pass = Var(model.key_set, within=NonNegativeReals)
    # model.Q_in_pass = Var(model.key_set, within=NonNegativeReals)
    model.Q_pass = Var(model.key_set, within=NonNegativeReals)
    # model.Q_exit_passed = Var(model.key_set, within=NonNegativeReals)
    model.Q_exit = Var(model.key_set, within=NonNegativeReals)
    model.Q_out = Var(model.key_set, within=NonNegativeReals)
    model.Q_arrived_vehicles = Var(model.key_set, within=NonNegativeReals)
    # i could leave out the consumption variables + constraints
    model.E_consumed_pass = Var(model.key_set, within=NonNegativeReals)

    # only for charging cells
    model.n_in_wait_charge = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.n_in_wait = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.n_wait = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.n_wait_charge_next = Var(model.charging_cells_key_set, within=NonNegativeReals)
    # n_in
    model.n_in_charge = Var(model.charging_cells_key_set, within=NonNegativeReals)
    # n_to_charge == n_charge1 is same
    # maybe cut one charging process -n_charge3 and E_charge3
    # model.n_to_charge = Var(model.charging_cells_key_set, within=NonNegativeReals)

    # n_charge1 and n_output_charged1 could kind of be also tied together somehow more straight forward
    model.n_charge1 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.n_charge2 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.n_charge3 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.n_output_charged1 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.n_output_charged2 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.n_output_charged3 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.n_finished_charge1 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.n_finished_charge2 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.n_finished_charge3 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    # n_exit_charge(t) = n_finished_charging(t-1)
    model.n_finished_charging = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.n_exit_charge = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_in_charge_wait = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_in_wait = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_in_charge = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_wait = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_wait_charge_next = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_input_charge1 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_output_charge1 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_input_charge2 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_output_charge2 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_input_charge3 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_output_charge3 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_finished_charge1 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_finished_charge2 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_finished_charge3 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.Q_finished_charging = Var(model.charging_cells_key_set, within=NonNegativeReals)
    # model.Q_exit_charged = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.E_charge1 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.E_charge2 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    model.E_charge3 = Var(model.charging_cells_key_set, within=NonNegativeReals)
    # i could leave out the consumption variables + constraints



def departing_fleets_n(model: ConcreteModel, t, c, f):
    if c in model.fleet_departing_times[f] and t in model.fleet_departing_times[f][c]:
        #print("!!!", model.fleet_incoming[f][c])
    # if t in model.fleet_depart_times[f] and c in model.fleet_incoming[f].keys():
    # if t == model.fleet_depart_times[f] and c == model.fleet_routes[f][0]:
        return model.n_incoming_vehicles[t, c, f] == model.fleet_departing_times[f][c][t]
        # return model.n_incoming_vehicles[t, c, f] == model.fleet_sizes[f]
    else:
        return model.n_incoming_vehicles[t, c, f] == 0


def departing_fleets_Q(model: ConcreteModel, t, c, f):
    if c in model.fleet_departing_times[f] and t in model.fleet_departing_times[f][c]:
        # print("this", c, t, f)
    #if t == model.fleet_depart_times[f] and c in model.fleet_incoming[f].keys():
        return (
            model.Q_incoming_vehicles[t, c, f]
            == model.fleet_soc_inits[f] * model.fleet_batt_cap[f] * model.fleet_departing_times[f][c][t]
        )
    else:
        return model.Q_incoming_vehicles[t, c, f] == 0


def init_n_in_fleet(model: ConcreteModel, t, c, f):
    return model.n_in[t, model.fleet_routes[f][0], f] == 0


def init_Q_in_fleet(model: ConcreteModel, t, c, f):
    return model.Q_in[t, model.fleet_routes[f][0], f] == 0


def init_n_out_fleet(model: ConcreteModel, t, c, f):
    return model.n_out[t, model.fleet_routes[f][-1], f] == 0


def init_Q_out_fleet(model: ConcreteModel, t, c, f):
    return model.Q_out[t, model.fleet_routes[f][-1], f] == 0


def routing_n(model: ConcreteModel, t, kl, f):

    # print(model.fleet_routes[f][kl], model.fleet_routes[f][kl + 1] )
    return (
        model.n_out[t, model.fleet_routes[f][kl], f]
        == model.n_in[t, model.fleet_routes[f][kl + 1], f]
    )


def routing_Q(model: ConcreteModel, t, kl, f):
    return (
        model.Q_out[t, model.fleet_routes[f][kl], f]
        == model.Q_in[t, model.fleet_routes[f][kl + 1], f]
    )


def restrict_arrivals_n(model: ConcreteModel, c, f):
    if c in model.fleet_arriving[f].keys():
        return quicksum(model.n_arrived_vehicles[t0, c, f] for t0 in model.nb_timestep if (t0, c, f) in model.key_set) == model.fleet_arriving[f][c]

    else:
        return quicksum(model.n_arrived_vehicles[t0, c, f] for t0 in model.nb_timestep if (t0, c, f) in model.key_set) == 0


def restrict_arrivals_Q(model: ConcreteModel, c, f):
    if c in model.fleet_arriving[f].keys():
        return quicksum(
            model.Q_arrived_vehicles[t0, c, f] for t0 in model.nb_timestep if (t0, c, f) in model.key_set) >= 0

    else:
        return quicksum(
            model.Q_arrived_vehicles[t0, c, f] for t0 in model.nb_timestep if (t0, c, f) in model.key_set) == 0
    # return model.Q_arrived_vehicles[t, c, f] == 0


def restrict_time_frame_exit(model: ConcreteModel, t, c, f):
    return model.n_exit[t, c, f] == 0


def restrict_time_frame_arrive(model: ConcreteModel, t, c, f):
    return model.n_arrived_vehicles[t, c, f] == 0


def restrict_time_frame_in(model: ConcreteModel, t, c, f):
    return model.n_in[t, c, f] == 0


def initialize_fleets(model, fleet_df):
    model.c_departing = ConstraintList()
    model.c_fleet_sizes = ConstraintList()
    model.c_arrivals_fleet = ConstraintList()
    model.c_route = ConstraintList()
    model.c_timeframe = ConstraintList()

    model.fleet_sizes = fleet_df["fleet_size"].array
    model.fleet_incoming = fleet_df["incoming"].to_list()
    model.fleet_arriving = fleet_df["arriving"].to_list()
    model.fleet_charge_cap = fleet_df["charge_cap"].array
    model.fleet_batt_cap = fleet_df["batt_cap"].array
    model.fleet_d_spec = fleet_df["d_spec"].array
    model.fleet_mu = [cap / 350 for cap in fleet_df["charge_cap"].array]

    model.fleet_routes = fleet_df["route"].to_list()
    model.fleet_depart_times = fleet_df["start_timestep"].to_list()
    model.fleet_departing_times = fleet_df["depart_time"].to_list()
    # print(model.fleet_departing_times[0])
    model.fleet_soc_inits = fleet_df["SOC_init"].to_list()
    model.departing_fleets_n = Constraint(model.key_set, rule=departing_fleets_n)
    model.departing_fleets_Q = Constraint(model.key_set, rule=departing_fleets_Q)
    model.routing_set = Set(initialize=create_set_routing)
    # print(model.routing_set.pprint())
    model.routing_n = Constraint(model.routing_set, rule=routing_n)
    model.routing_Q = Constraint(model.routing_set, rule=routing_Q)
    model.init_n_in_fleet = Constraint(model.key_set, rule=init_n_in_fleet)
    model.init_Q_in_fleet = Constraint(model.key_set, rule=init_Q_in_fleet)
    # TODO:  restrict arrivals in such a way that the sum over time can be at maximum a certain number
    model.keys_c_f = set([(el[1], el[2]) for el in model.key_set])
    model.restrict_arrivals_n = Constraint(
        model.keys_c_f,
        rule=restrict_arrivals_n,
    )
    model.restrict_arrivals_Q = Constraint(
        model.keys_c_f,
        rule=restrict_arrivals_Q,
    )
    model.init_n_out_fleet = Constraint(model.key_set, rule=init_n_out_fleet)
    model.init_Q_out_fleet = Constraint(model.key_set, rule=init_Q_out_fleet)
    model.restrict_time_frame_exit = Constraint(
        [
            el
            for el in model.key_set
            if el[1] in model.fleet_routes[el[2]]
            and el[0] <= model.fleet_depart_times[el[2]]
        ],
        rule=restrict_time_frame_exit,
    )
    model.restrict_time_frame_arrive = Constraint(
        [
            el
            for el in model.key_set
            if el[1] in model.fleet_routes[el[2]]
            and el[0] <= model.fleet_depart_times[el[2]]
        ],
        rule=restrict_time_frame_arrive,
    )
    model.restrict_time_frame_in = Constraint(
        [
            el
            for el in model.key_set
            if el[1] in model.fleet_routes[el[2]]
            and el[0] <= model.fleet_depart_times[el[2]]
        ],
        rule=restrict_time_frame_in,
    )
    #
    # for ij in range(0, len(fleet_df)):
    # #
    #
    # #     fleet_size = fleet_df["fleet_size"].to_list()[ij]
    # #     fleet_charge_cap = fleet_df["charge_cap"].to_list()[ij]
    # #     fleet_batt_cap = fleet_df["batt_cap"].to_list()[ij]
    # #     fleet_d_spec = fleet_df["d_spec"].to_list()[ij]
    # #     fleet_soc_init = fleet_df["SOC_init"].to_list()[ij]
    #     # print(RangeSet(0, len(model.fleet_routes[fleet_id])-1))
    #     # print(model.fleet_routes)
    #     # print("l", model.nb_timestep)
    #     # print(model.fleet_routes[fleet_id][0])
    #     # model.routing_n = Constraint(np.array(model.nb_timestep), np.array(range(0, len(model.fleet_routes[fleet_id])-1)),
    #     #                              np.array([f for f in model.nb_fleet if f == fleet_id]), rule=routing_n)
    #     #
    #     # # TODO: restate this
    #     # model.routing_Q = Constraint(model.nb_timestep, range(0, len(model.fleet_routes[fleet_id])-1),
    #     #                              [f for f in model.nb_fleet if f == fleet_id], rule=routing_Q)
    #
    #     # model.c_departing.add(
    #     #     model.n_incoming_vehicles[int(start_time_step), start_cell, fleet_id]
    #     #     == fleet_size
    #     # )
    #     # model.c_departing.add(
    #     #     model.Q_incoming_vehicles[int(start_time_step), start_cell, fleet_id]
    #     #     == fleet_soc_init * fleet_batt_cap * fleet_size
    #     # )
    #
    #     #for kl in range(0, len(route) - 1):
    #         # for t in model.nb_timestep:
    #
    #             # model.c_route.add(
    #             #     model.n_out[t, route[kl], fleet_id]
    #             #     == model.n_in[t, route[kl + 1], fleet_id]
    #             # )
    #             # model.c_route.add(
    #             #     model.Q_out[t, route[kl], fleet_id]
    #             #     == model.Q_in[t, route[kl + 1], fleet_id]
    #             # )
    #     start_time_step = fleet_df["start_timestep"].to_list()[ij]
    #     #     start_cell = fleet_df["route"].to_list()[ij][0]
    #     #     dest_cell = fleet_df["route"].to_list()[ij][-1]
    #     route = fleet_df["route"].to_list()[ij]
    #     fleet_id = fleet_df["fleet_id"].to_list()[ij]
    # for t in model.nb_timestep:
    # #     # model.c_departing.add(model.n_in[t, start_cell, fleet_id] == 0)
    # #     # model.c_departing.add(model.Q_in[t, start_cell, fleet_id] == 0)
    # #
    # for c in model.nb_cell:
    # #         # if not (t is start_time_step and c is start_cell):
    # #         #     model.c_route.add(model.n_incoming_vehicles[t, c, fleet_id] == 0)
    # #         #     model.c_route.add(model.Q_incoming_vehicles[t, c, fleet_id] == 0)
    # #
    # #         # if c is not route[-1]:
    # #         #     # model.c_route.add(model.n_arrived_vehicles[t, c, fleet_id] == 0)
    # #         #     # model.c_route.add(model.Q_arrived_vehicles[t, c, fleet_id] == 0)
    # #         #     l = 0
    # #         # else:
    # #         #     model.c_route.add(model.n_out[t, c, fleet_id] == 0)
    # #         #     model.c_route.add(model.Q_out[t, c, fleet_id] == 0)
    # #
    # #         # do I even need these constraints??
    # if t <= start_time_step and c in route:
    # model.c_timeframe.add(model.n_exit[t, c, fleet_id] == 0)
    # model.c_route.add(model.n_arrived_vehicles[t, c, fleet_id] == 0)
    # model.c_timeframe.add(model.n_in[t, c, fleet_id] == 0)
    # print("fleet ", fleet_id, "intialized")


# ---------------------------------------------------------------------------------------------------------------------
#   Definition of rules for constraints
# ---------------------------------------------------------------------------------------------------------------------

# this can be left out later:
def no_waiting_at_charging_station(model: ConcreteModel, t, c, f):
    return (
        sum(
            [
                model.n_wait[to, c, fo]
                for to in model.nb_timestep
                for fo in model.nb_fleet
                if (to, c, fo) in model.key_set
            ]
        )
        == 0
    )


def no_charging_at_charging_station(model: ConcreteModel, t, c, f):
    return (
        sum(
            [
                model.n_charge1[to, c, fo]
                for to in model.nb_timestep
                for fo in model.nb_fleet
                if (to, c, fo) in model.key_set
            ]
        )
        == 0
    )


# DEFINITION OF inits


# def init_n_exit_pass(model: ConcreteModel, c, f):
#     return (
#         model.n_exit_pass[model.fleet_depart_times[f], c, f] == 0
#     )


# def init_Q_exit_pass(model: ConcreteModel, c, f):
#     return (
#         model.Q_exit_passed[model.fleet_depart_times[f], c, f]
#         == 0
#     )


def init_n_finished_charge1(model: ConcreteModel, c, f):
    return (
        model.n_finished_charge1[
            model.fleet_depart_times[f], c, f
        ]
        == 0
    )


def init_Q_finished_charging(model: ConcreteModel, c, f):
    return (
        model.Q_finished_charging[
            model.fleet_depart_times[f], c, f
        ]
        == 0
    )


def init_n_finished_charge2(model: ConcreteModel, c, f):
    return (
        model.n_finished_charge2[
            model.fleet_depart_times[f], c, f
        ]
        == 0
    )


def init_n_finished_charge2_t1(model: ConcreteModel, c, f):
    return (
        model.n_finished_charge2[
            model.fleet_depart_times[f] + 1, c, f
        ]
        == 0
    )


def init_Q_output_charge2(model: ConcreteModel, c, f):
    return (
        model.Q_output_charge2[model.fleet_depart_times[f], c, f]
        == 0
    )


def init_Q_input_charge1(model: ConcreteModel, c, f):
    return (
        model.Q_input_charge1[model.fleet_depart_times[f], c, f]
        == model.Q_in_charge[model.fleet_depart_times[f], c, f]
    )


def init_n_charge2(model: ConcreteModel, c, f):
    return (
        model.n_charge2[model.fleet_depart_times[f] + 1, c, f]
        == 0
    )


def init_Q_output_charge2_t1(model: ConcreteModel, c, f):
    return (
        model.Q_output_charge2[
            model.fleet_depart_times[f] + 1, c, f
        ]
        == 0
    )


def init_n_finished_charge3(model: ConcreteModel, c, f):
    return (
        model.n_finished_charge3[
            model.fleet_depart_times[f], c, f
        ]
        == 0
    )


def init_n_finished_charge3_t1(model: ConcreteModel, c, f):
    return (
        model.n_finished_charge3[
            model.fleet_depart_times[f] + 1, c, f
        ]
        == 0
    )


def init_n_finished_charge3_t2(model: ConcreteModel, c, f):
    return (
        model.n_finished_charge3[
            model.fleet_depart_times[f] + 2, c, f
        ]
        == 0
    )


def init_n_charge3(model: ConcreteModel, c, f):
    return (
        model.n_charge3[model.fleet_depart_times[f], c, f] == 0
    )


def init_n_charge3_t1(model: ConcreteModel, c, f):
    return (
        model.n_charge3[model.fleet_depart_times[f] + 1, c, f]
        == 0
    )


def init_Q_output_charge3(model: ConcreteModel, c, f):
    return (
        model.n_charge3[model.fleet_depart_times[f] + 1, c, f]
        == 0
    )


def init_Q_output_charge3_t1(model: ConcreteModel, c, f):
    return (
        model.n_charge3[model.fleet_depart_times[f] + 2, c, f]
        == 0
    )


def init_balance_n_to_charge_n_in_charge(model: ConcreteModel, c, f):
    return (
        model.n_charge1[model.fleet_depart_times[f], c, f]
        == model.n_in_charge[model.fleet_depart_times[f], c, f]
    )


def init_balance_n_wait(model: ConcreteModel, c, f):
    return (
        model.n_wait[model.fleet_depart_times[f], c, f]
        + model.n_wait_charge_next[
            model.fleet_depart_times[f], c, f
        ]
        == model.n_in_wait[model.fleet_depart_times[f], c, f]
    )


def init_balance_Q_wait(model: ConcreteModel, c, f):
    return (
        model.Q_wait[model.fleet_depart_times[f], c, f]
        + model.Q_wait_charge_next[
            model.fleet_depart_times[f], c, f
        ]
        == model.Q_in_wait[model.fleet_depart_times[f], c, f]
    )


def balance_n_incoming(model: ConcreteModel, t, c, f):
    if model.cell_charging_cap[c] > 0:
        return (
            model.n_in[t, c, f] + model.n_incoming_vehicles[t, c, f]
            == model.n_pass[t, c, f] + model.n_in_wait_charge[t, c, f]
        )
    else:
        return (
                model.n_in[t, c, f] + model.n_incoming_vehicles[t, c, f]
                == model.n_pass[t, c, f]
        )


# def balance_n_incoming_NO_CS(model: ConcreteModel, t, c, f):
#     return (
#         model.n_in[t, c, f] + model.n_incoming_vehicles[t, c, f]
#         == model.n_pass[t, c, f]
#     )


def balance_Q_incoming(model: ConcreteModel, t, c, f):
    if model.cell_charging_cap[c] > 0:
        return (
            model.Q_in[t, c, f] + model.Q_incoming_vehicles[t, c, f]
            == model.Q_pass[t, c, f] + model.Q_in_charge_wait[t, c, f]
        )
    else:
        return (
                model.Q_in[t, c, f] + model.Q_incoming_vehicles[t, c, f]
                == model.Q_pass[t, c, f]
        )

# def balance_Q_incoming_NO_CS(model: ConcreteModel, t, c, f):
#     return (
#         model.Q_in[t, c, f] + model.Q_incoming_vehicles[t, c, f]
#         == model.Q_pass[t, c, f]
#     )


# def balance_n_passing(model: ConcreteModel, t, c, f):
#     return model.n_in_pass[t, c, f] == model.n_pass[t, c, f]


# def balance_Q_passing(model: ConcreteModel, t, c, f):
#     return model.Q_in_pass[t, c, f] == model.Q_pass[t, c, f]


def balance_waiting_and_charging(model: ConcreteModel, t, c, f):
    return (
        model.n_in_wait_charge[t, c, f]
        == model.n_in_charge[t, c, f] + model.n_in_wait[t, c, f]
    )


# def balance_n_to_charge(model: ConcreteModel, t, c, f):
#     return model.n_charge1[t, c, f] == model.n_to_charge[t, c, f]


def balance_Q_charging_transfer(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge1[t, c, f]
        == model.Q_input_charge2[t, c, f] + model.Q_finished_charge1[t, c, f]
    )


def balance_Q_charging_transfer_1(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge2[t, c, f]
        == model.Q_input_charge3[t, c, f] + model.Q_finished_charge2[t, c, f]
    )


def balance_Q_charging_transfer_2(model: ConcreteModel, t, c, f):
    return model.Q_output_charge3[t, c, f] == model.Q_finished_charge3[t, c, f]


def balance_n_finishing(model: ConcreteModel, t, c, f):
    return (
        model.n_finished_charging[t, c, f]
        == model.n_finished_charge1[t, c, f]
        + model.n_finished_charge2[t, c, f]
        + model.n_finished_charge3[t, c, f]
    )


def balance_Q_finishing(model: ConcreteModel, t, c, f):
    return (
        model.Q_finished_charging[t, c, f]
        == model.Q_finished_charge1[t, c, f]
        + model.Q_finished_charge2[t, c, f]
        + model.Q_finished_charge3[t, c, f]
    )


# def balance_n_exiting(model: ConcreteModel, t, c, f):
#     return (
#         model.n_exit[t, c, f]
#         == model.n_exit_pass[t, c, f] + model.n_exit_charge[t, c, f]
#     )
#
#
# def balance_n_exiting_NO_CS(model: ConcreteModel, t, c, f):
#     return (
#         model.n_exit[t, c, f]
#         == model.n_exit_pass[t, c, f]
#     )


# def balance_Q_exiting(model: ConcreteModel, t, c, f):
#     return (
#         model.Q_exit[t, c, f]
#         == model.Q_exit_passed[t, c, f] + model.Q_exit_charged[t, c, f]
#     )


# def balance_Q_exiting_NO_CS(model: ConcreteModel, t, c, f):
#     return (
#         model.Q_exit[t, c, f]
#         == model.Q_exit_passed[t, c, f]
#     )


def balance_Q_out(model: ConcreteModel, t, c, f):
    return (
        model.Q_exit[t, c, f] - model.Q_arrived_vehicles[t, c, f]
        == model.Q_out[t, c, f]
    )


def balance_n_charge_transfer_1(model: ConcreteModel, t, c, f):
    return (
        model.n_output_charged1[t, c, f]
        == model.n_finished_charge1[t, c, f] + model.n_charge2[t, c, f]
    )


def balance_n_charge_transfer_2(model: ConcreteModel, t, c, f):
    return (
        model.n_output_charged2[t, c, f]
        == model.n_finished_charge2[t, c, f] + model.n_charge3[t, c, f]
    )


def balance_n_charge_transfer_3(model: ConcreteModel, t, c, f):
    return model.n_output_charged3[t, c, f] == model.n_finished_charge3[t, c, f]


def calc_energy_consumption_while_passing(model: ConcreteModel, t, c, f):
    return (
        model.E_consumed_pass[t, c, f]
        == model.n_pass[t, c, f] * model.cell_width[c] * model.fleet_d_spec[f]
    )


def energy_consumption_before_charging(model: ConcreteModel, t, c, f):
    return (
        model.Q_in_charge_wait[t, c, f] - model.n_in_wait_charge[t, c, f] * (1/2) * model.cell_width[c] * model.fleet_d_spec[f]
        == model.Q_in_charge[t, c, f] + model.Q_in_wait[t, c, f]
    )


# def calc_energy_consumption_before_charging(model: ConcreteModel, t, c, f):
#     return (
#         model.E_consumed_charge_wait[t, c, f]
#         ==
#         * (1 / 2)
#
#
#     )


# def calc_energy_consumption_after_charging(model: ConcreteModel, t, c, f):
#     return (
#         model.E_consumed_exit_charge[t, c, f]
#         == model.n_finished_charging[t, c, f]
#         * (1 / 2)
#         * model.cell_width[c]
#         * model.fleet_d_spec[f]
#     )


def charging_1(model: ConcreteModel, t, c, f):
    return (
        model.E_charge1[t, c, f]
        <= model.n_charge1[t, c, f] * model.fleet_charge_cap[f] * model.time_resolution * (model.fleet_charge_cap[f]/ 350)
    )


def charging_2(model: ConcreteModel, t, c, f):
    return (
        model.E_charge2[t, c, f]
        <= model.n_charge2[t, c, f] * model.fleet_charge_cap[f] * model.time_resolution * (model.fleet_charge_cap[f]/ 350)
    )


def charging_3(model: ConcreteModel, t, c, f):
    return (
        model.E_charge3[t, c, f]
        <= model.n_charge3[t, c, f] * model.fleet_charge_cap[f] * model.time_resolution * (model.fleet_charge_cap[f]/ 350)
    )


def min_charging_1(model: ConcreteModel, t, c, f):
    return (
        model.E_charge1[t, c, f]
        >= model.n_charge1[t, c, f] * model.fleet_charge_cap[f] * model.t_min * (model.fleet_charge_cap[f]/ 350)
    )


def min_charging_2(model: ConcreteModel, t, c, f):
    return (
        model.E_charge2[t, c, f]
        >= model.n_charge2[t, c, f] * model.fleet_charge_cap[f] * model.t_min * (model.fleet_charge_cap[f]/ 350)
    )


def min_charging_3(model: ConcreteModel, t, c, f):
    return (
        model.E_charge3[t, c, f]
        >= model.n_charge3[t, c, f] * model.fleet_charge_cap[f] * model.t_min * (model.fleet_charge_cap[f]/ 350)
    )


def setting_relation_n_Q_in_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_in[t, c, f]
        >= model.n_in[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )


def setting_relation_n_Q_in_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_in[t, c, f]
        <= model.n_in[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )


# def setting_relation_n_Q_in_pass_min(model: ConcreteModel, t, c, f):
#     return (
#         model.Q_in_pass[t, c, f]
#         >= model.n_in_pass[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
#     )
#
#
# def setting_relation_n_Q_in_pass_max(model: ConcreteModel, t, c, f):
#     return (
#         model.Q_in_pass[t, c, f]
#         <= model.n_in_pass[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
#     )


def setting_relation_n_Q_in_wait_charge_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_in_charge_wait[t, c, f]
        >= model.n_in_wait_charge[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )


def setting_relation_n_Q_in_wait_charge_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_in_charge_wait[t, c, f]
        <= model.n_in_wait_charge[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )


def setting_relation_n_Q_in_wait_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_in_wait[t, c, f]
        >= model.n_in_wait[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )


def setting_relation_n_Q_in_wait_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_in_wait[t, c, f]
        <= model.n_in_wait[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )


def setting_relation_n_Q_wait_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_wait[t, c, f]
        >= model.n_wait[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )


def setting_relation_n_Q_wait_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_wait[t, c, f]
        <= model.n_wait[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )


def setting_relation_n_Q_in_charge_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_in_charge[t, c, f]
        >= model.n_in_charge[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )


def setting_relation_n_Q_in_charge_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_in_charge[t, c, f]
        <= model.n_in_charge[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )


def setting_relation_n_Q_wait_charge_next_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_wait_charge_next[t, c, f]
        >= model.n_wait_charge_next[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )


def setting_relation_n_Q_wait_charge_next_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_wait_charge_next[t, c, f]
        <= model.n_wait_charge_next[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )


def setting_relation_n_Q_charge_1_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_input_charge1[t, c, f]
        >= model.n_charge1[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )


def setting_relation_n_Q_charge_1_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_input_charge1[t, c, f]
        <= model.n_charge1[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )


def setting_relation_n_Q_charge_2_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_input_charge2[t, c, f]
        >= model.n_charge2[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )


def setting_relation_n_Q_charge_2_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_input_charge2[t, c, f]
        <= model.n_charge2[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )


def setting_relation_n_Q_charge_3_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_input_charge3[t, c, f]
        >= model.n_charge3[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )


def setting_relation_n_Q_charge_3_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_input_charge3[t, c, f]
        <= model.n_charge3[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )


def setting_relation_n_Q_output_charge_1_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge1[t, c, f]
        >= model.n_output_charged1[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )


def setting_relation_n_Q_output_charge_1_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge1[t, c, f]
        <= model.n_output_charged1[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )


def setting_relation_n_Q_output_charge_2_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge2[t, c, f]
        >= model.n_output_charged2[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )


def setting_relation_n_Q_output_charge_2_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge2[t, c, f]
        <= model.n_output_charged2[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )


def setting_relation_n_Q_output_charge_3_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge3[t, c, f]
        >= model.n_output_charged3[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )


def setting_relation_n_Q_output_charge_3_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge3[t, c, f]
        <= model.n_output_charged3[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )


def setting_relation_n_Q_finished_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_finished_charging[t, c, f]
        >= model.n_finished_charging[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )


def setting_relation_n_Q_finished_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_finished_charging[t, c, f]
        <= model.n_finished_charging[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )


def setting_relation_n_Q_finished_1_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_finished_charge1[t, c, f]
        >= model.n_finished_charge1[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )


def setting_relation_n_Q_finished_1_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_finished_charge1[t, c, f]
        <= model.n_finished_charge1[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )


def setting_relation_n_Q_finished_2_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_finished_charge2[t, c, f]
        >= model.n_finished_charge2[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )


def setting_relation_n_Q_finished_2_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_finished_charge2[t, c, f]
        <= model.n_finished_charge2[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )


def setting_relation_n_Q_finished_3_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_finished_charge3[t, c, f]
        >= model.n_finished_charge3[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )


def setting_relation_n_Q_finished_3_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_finished_charge3[t, c, f]
        <= model.n_finished_charge3[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )


def setting_relation_n_Q_exit_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_exit[t, c, f]
        >= model.n_exit[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )


def setting_relation_n_Q_exit_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_exit[t, c, f]
        <= model.n_exit[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )


def setting_relation_n_Q_arriving_min(model: ConcreteModel, t, c, f):
    return (
        model.Q_arrived_vehicles[t, c, f]
        >= model.n_arrived_vehicles[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )


def setting_relation_n_Q_arriving_max(model: ConcreteModel, t, c, f):
    return (
        model.Q_arrived_vehicles[t, c, f]
        <= model.n_arrived_vehicles[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    )

def constraining_n_pass_to_zero_at_end(model:ConcreteModel, t, c, f):
    return(model.n_pass[t, c, f] == 0)
# def inserting_time_jump_n_pass_exit(model: ConcreteModel, t, c, f):
#     return model.n_exit_pass[t + 1, c, f] == model.n_pass[t, c, f]
#
#
# def inserting_time_jump_n_pass_exit(model: ConcreteModel, t, c, f):
#     return model.n_exit_pass[t + 1, c, f] == model.n_pass[t, c, f]


def inserting_time_jump_n_pass_exit(model: ConcreteModel, t, c, f):
    if model.cell_charging_cap[c] > 0:
        return (
            model.n_exit[t + 1, c, f]
            == model.n_pass[t, c, f] + model.n_finished_charging[t, c, f]
        )
    else:
        return model.n_exit[t + 1, c, f] == model.n_pass[t, c, f]


# def inserting_time_jump_n_pass_exit(model: ConcreteModel, t, c, f):
#     return (
#         model.n_exit[t + 1, c, f]
#         == model.n_pass[t, c, f]
#     )


def inserting_time_jump_Q_charging_1(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge1[t + 1, c, f]
        == model.Q_input_charge1[t, c, f] + model.E_charge1[t, c, f]
    )


def inserting_time_jump_Q_charging_2(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge2[t + 1, c, f]
        == model.Q_input_charge2[t, c, f] + model.E_charge2[t, c, f]
    )


def inserting_time_jump_Q_charging_3(model: ConcreteModel, t, c, f):
    return (
        model.Q_output_charge3[t + 1, c, f]
        == model.Q_input_charge3[t, c, f] + model.E_charge3[t, c, f]
    )


def inserting_time_jump_n_charging_1(model: ConcreteModel, t, c, f):
    return model.n_output_charged1[t + 1, c, f] == model.n_charge1[t, c, f]


def inserting_time_jump_n_charging_2(model: ConcreteModel, t, c, f):
    return model.n_output_charged2[t + 1, c, f] == model.n_charge2[t, c, f]


def inserting_time_jump_n_charging_3(model: ConcreteModel, t, c, f):
    return model.n_output_charged3[t + 1, c, f] == model.n_charge3[t, c, f]


# def inserting_time_jump_Q_charged_exit(model: ConcreteModel, t, c, f):
#     return (
#         model.Q_exit_charged[t + 1, c, f]
#         == model.Q_finished_charging[t, c, f] - model.E_consumed_exit_charge[t, c, f]
#     )


def inserting_time_jump_n_charged_exit(model: ConcreteModel, t, c, f):
    return model.n_exit_charge[t + 1, c, f] == model.n_finished_charging[t, c, f]


def inserting_time_jump_Q_passed_exit(model: ConcreteModel, t, c, f):
    if model.cell_charging_cap[c] > 0:
        return (
            model.Q_exit[t + 1, c, f]
            == model.Q_pass[t, c, f] - model.n_pass[t, c, f] * model.cell_width[c] * model.fleet_d_spec[f] + model.Q_finished_charging[t, c, f]
            - model.n_finished_charging[t, c, f] * (1 / 2) * model.cell_width[c] * model.fleet_d_spec[f]
        )
    else:
        return model.Q_exit[t + 1, c, f] == model.Q_pass[t, c, f] - model.n_pass[t, c, f] * model.cell_width[c] \
               * model.fleet_d_spec[f]

# def balance_Q_exiting(model: ConcreteModel, t, c, f):
#     return (
#         model.Q_exit[t, c, f]
#         == model.Q_exit_passed[t, c, f] + model.Q_exit_charged[t, c, f]
#     )


# def balance_Q_exiting_NO_CS(model: ConcreteModel, t, c, f):
#     return (
#         model.Q_exit[t, c, f]
#         == model.Q_exit_passed[t, c, f]
#     )



def queue_n(model: ConcreteModel, t, c, f):
    return (
        model.n_wait[t, c, f]
        == model.n_wait[t - 1, c, f]
        + model.n_in_wait[t, c, f]
        - model.n_wait_charge_next[t, c, f]
    )


def queue_Q(model: ConcreteModel, t, c, f):
    return (
        model.Q_wait[t, c, f]
        == model.Q_wait[t - 1, c, f]
        + model.Q_in_wait[t, c, f]
        - model.Q_wait_charge_next[t, c, f]
    )


def entering_charging_station_n(model: ConcreteModel, t, c, f):
    return (
        model.n_charge1[t, c, f]
        == model.n_in_charge[t, c, f] + model.n_wait_charge_next[t - 1, c, f]
    )


def entering_charging_station_Q(model: ConcreteModel, t, c, f):
    return (
        model.Q_input_charge1[t, c, f]
        == model.Q_in_charge[t, c, f] + model.Q_wait_charge_next[t - 1, c, f]
    )


# --------


def constraint_rule_in(model: ConcreteModel, t, c, f):
    return (
        model.Q_in[t, c, f]
        >= model.n_in[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    )


def constraint_balance_constraint(model: ConcreteModel, t, c, f):
    return (
        model.n_exit[t, c, f] - model.n_arrived_vehicles[t, c, f]
        == model.n_out[t, c, f]
    )


def initialize_cells(model: ConcreteModel, cell_df: pd.DataFrame):
    model.cell_width = cell_df["length"].array
    model.cell_charging_cap = cell_df["capacity"].array


# TODO: adapt all here making a differentiation between the both types of constraints
def constr_vehicle_states(model: ConcreteModel):
    # model.c_incoming_flow = ConstraintList()
    # model.c_passing = ConstraintList()
    # model.c_waiting_and_charging = ConstraintList()
    # model.c_exiting = ConstraintList()
    # model.constraint_nq_rel = ConstraintList()
    t1 = time.time()
    model.c_rule_in = Constraint(model.key_set, rule=constraint_rule_in)
    print(time.time() - t1)
    model.c_rule_balance = Constraint(model.key_set, rule=constraint_balance_constraint)
    # model.c_start_time = ConstraintList()
    # model.c_geq_zero = ConstraintList()

    # no charging and waiting in a queue is possible at charging stations with no charging station
    # model.charging_cells_key_set = [
    #     el for el in model.key_set if model.cell_charging_cap[el[1]] > 0
    # ]
    t0 = time.time()
    model.no_charging_cells_key_set = [
        el for el in model.key_set if model.cell_charging_cap[el[1]] == 0
    ]
    print("the list comprehension took.. ", time.time() - t0, "sec")
    # model.no_waiting_at_cell = Constraint(
    #     model.no_charging_cells_key_set, rule=no_waiting_at_charging_station
    # )
    # model.no_charging_at_cell = Constraint(
    #     model.no_charging_cells_key_set, rule=no_charging_at_charging_station
    # )
    model.cell_and_fleets = set([(el[1], el[2]) for el in model.key_set])
    model.cell_and_fleets_CS = set([(el[1], el[2]) for el in model.charging_cells_key_set])
    print("inits,....")
    t5 = time.time()
    # model.init_n_exit_pass = Constraint(model.cell_and_fleets, rule=init_n_exit_pass)
    print(time.time() - t5)
    model.init_n_finished_charge1 = Constraint(
        model.cell_and_fleets_CS, rule=init_n_finished_charge1
    )
    model.init_Q_finished_charging = Constraint(
        model.cell_and_fleets_CS, rule=init_Q_finished_charging
    )
    model.init_n_finished_charge2 = Constraint(
        model.cell_and_fleets_CS, rule=init_n_finished_charge2
    )
    model.init_n_finished_charge2_t1 = Constraint(
        model.cell_and_fleets_CS, rule=init_n_finished_charge2_t1
    )
    model.init_Q_output_charge2 = Constraint(model.cell_and_fleets_CS, rule=init_Q_output_charge2)

    model.init_Q_input_charge1 = Constraint(model.cell_and_fleets_CS, rule=init_Q_input_charge1)
    model.init_n_charge2 = Constraint(model.cell_and_fleets_CS, rule=init_n_charge2)
    model.init_Q_output_charge2_t1 = Constraint(
        model.cell_and_fleets_CS, rule=init_Q_output_charge2_t1
    )
    model.init_n_finished_charge3 = Constraint(
        model.cell_and_fleets_CS, rule=init_n_finished_charge3
    )
    model.init_n_finished_charge3_t1 = Constraint(
        model.cell_and_fleets_CS, rule=init_n_finished_charge3_t1
    )
    model.init_n_finished_charge3_t2 = Constraint(
        model.cell_and_fleets_CS, rule=init_n_finished_charge3_t2
    )
    model.init_n_charge3 = Constraint(model.cell_and_fleets_CS, rule=init_n_charge3)
    model.init_n_charge3_t1 = Constraint(model.cell_and_fleets_CS, rule=init_n_charge3_t1)
    model.init_Q_output_charge3 = Constraint(model.cell_and_fleets_CS, rule=init_Q_output_charge3)
    model.init_Q_output_charge3_t1 = Constraint(
        model.cell_and_fleets_CS, rule=init_Q_output_charge3_t1
    )
    model.init_balance_n_to_charge_n_in_charge = Constraint(
        model.cell_and_fleets_CS, rule=init_balance_n_to_charge_n_in_charge
    )
    model.init_balance_n_wait = Constraint(model.cell_and_fleets_CS, rule=init_balance_n_wait)
    model.init_balance_Q_wait = Constraint(model.cell_and_fleets_CS, rule=init_balance_Q_wait)


    # constraining_n_pass_to_zero_at_end = Constraint(rule=constraining_n_pass_to_zero_at_end)
    print("Initializations finished")
    # balance constraints for vehicles
    model.balance_n_incoming = Constraint(model.key_set, rule=balance_n_incoming)
    # model.balance_n_incoming_NO_CS = Constraint(model.no_charging_cells_key_set, rule=balance_n_incoming_NO_CS)
    model.balance_Q_incoming = Constraint(model.key_set, rule=balance_Q_incoming)
    #model.balance_Q_incoming_NO_CS = Constraint(model.no_charging_cells_key_set, rule=balance_Q_incoming_NO_CS)
    # model.balance_n_passing = Constraint(model.key_set, rule=balance_n_passing)
    # model.balance_Q_passing = Constraint(model.key_set, rule=balance_Q_passing)
    model.balance_waiting_and_charging = Constraint(
        model.charging_cells_key_set, rule=balance_waiting_and_charging
    )
    # model.balance_n_to_charge = Constraint(model.charging_cells_key_set, rule=balance_n_to_charge)
    model.balance_n_finishing = Constraint(model.charging_cells_key_set, rule=balance_n_finishing)
    model.balance_Q_finishing = Constraint(model.charging_cells_key_set, rule=balance_Q_finishing)
    # model.balance_n_exiting = Constraint(model.charging_cells_key_set, rule=balance_n_exiting)
    # model.balance_n_exiting_NO_CS = Constraint(model.no_charging_cells_key_set, rule=balance_n_exiting_NO_CS)
    # model.balance_Q_exiting = Constraint(model.charging_cells_key_set, rule=balance_Q_exiting)
    # model.balance_Q_exiting_NO_CS = Constraint(model.no_charging_cells_key_set, rule=balance_Q_exiting_NO_CS)
    model.balance_Q_out = Constraint(model.key_set, rule=balance_Q_out)
    model.balance_n_charge_transfer_1 = Constraint(
        model.charging_cells_key_set, rule=balance_n_charge_transfer_1
    )
    model.balance_n_charge_transfer_2 = Constraint(
        model.charging_cells_key_set, rule=balance_n_charge_transfer_2
    )
    model.balance_n_charge_transfer_3 = Constraint(
        model.charging_cells_key_set, rule=balance_n_charge_transfer_3
    )

    print("Balances finished")
    # energy consumption while driving
    model.energy_consumption_while_passing = Constraint(
        model.key_set, rule=calc_energy_consumption_while_passing
    )
    model.energy_consumption_before_charging = Constraint(
        model.charging_cells_key_set, rule=energy_consumption_before_charging
    )
    # model.calc_energy_consumption_before_charging = Constraint(
    #     model.charging_cells_key_set, rule=calc_energy_consumption_before_charging
    # )

    # model.calc_energy_consumption_after_charging = Constraint(
    #     model.charging_cells_key_set, rule=calc_energy_consumption_after_charging
    # )

    # charging activity
    model.charging_1 = Constraint(model.charging_cells_key_set, rule=charging_1)
    model.charging_2 = Constraint(model.charging_cells_key_set, rule=charging_2)
    model.charging_3 = Constraint(model.charging_cells_key_set, rule=charging_3)
    model.min_charging_1 = Constraint(model.charging_cells_key_set, rule=min_charging_1)
    model.min_charging_2 = Constraint(model.charging_cells_key_set, rule=min_charging_2)
    model.min_charging_3 = Constraint(model.charging_cells_key_set, rule=min_charging_3)
    model.balance_Q_charging_transfer = Constraint(
        model.charging_cells_key_set, rule=balance_Q_charging_transfer
    )
    model.balance_Q_charging_transfer_1 = Constraint(
        model.charging_cells_key_set, rule=balance_Q_charging_transfer_1
    )
    model.balance_Q_charging_transfer_2 = Constraint(
        model.charging_cells_key_set, rule=balance_Q_charging_transfer_2
    )
    print("Energy cons finished")
    # relation between n and Q (n ... nb of vehicels, Q ... cummulated state of charge)
    model.setting_relation_n_Q_in_min = Constraint(
        model.key_set, rule=setting_relation_n_Q_in_min
    )
    model.setting_relation_n_Q_in_max = Constraint(
        model.key_set, rule=setting_relation_n_Q_in_max
    )
    # model.setting_relation_n_Q_in_pass_min = Constraint(
    #     model.key_set, rule=setting_relation_n_Q_in_pass_min
    # )
    # model.setting_relation_n_Q_in_pass_max = Constraint(
    #     model.key_set, rule=setting_relation_n_Q_in_pass_max
    # )
    model.setting_relation_n_Q_in_wait_charge_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_in_wait_charge_min
    )
    model.setting_relation_n_Q_in_wait_charge_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_in_wait_charge_max
    )
    model.setting_relation_n_Q_in_wait_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_in_wait_min
    )
    model.setting_relation_n_Q_in_wait_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_in_wait_max
    )
    model.setting_relation_n_Q_wait_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_wait_min
    )
    model.setting_relation_n_Q_wait_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_wait_max
    )
    model.setting_relation_n_Q_in_charge_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_in_charge_min
    )
    model.setting_relation_n_Q_in_charge_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_in_charge_max
    )
    model.setting_relation_n_Q_wait_charge_next_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_wait_charge_next_min
    )
    model.setting_relation_n_Q_wait_charge_next_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_wait_charge_next_max
    )
    model.setting_relation_n_Q_charge_1_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_charge_1_min
    )
    model.setting_relation_n_Q_charge_1_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_charge_1_max
    )
    model.setting_relation_n_Q_charge_2_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_charge_2_min
    )
    model.setting_relation_n_Q_charge_2_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_charge_2_max
    )
    model.setting_relation_n_Q_charge_3_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_charge_3_min
    )
    model.setting_relation_n_Q_charge_3_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_charge_3_max
    )
    model.setting_relation_n_Q_output_charge_1_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_output_charge_1_min
    )
    model.setting_relation_n_Q_output_charge_1_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_output_charge_1_max
    )
    model.setting_relation_n_Q_output_charge_2_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_output_charge_2_min
    )
    model.setting_relation_n_Q_output_charge_2_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_output_charge_2_max
    )
    model.setting_relation_n_Q_output_charge_3_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_output_charge_3_min
    )
    model.setting_relation_n_Q_output_charge_3_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_output_charge_3_max
    )
    model.setting_relation_n_Q_finished_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_min
    )
    model.setting_relation_n_Q_finished_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_max
    )
    model.setting_relation_n_Q_finished_1_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_1_min
    )
    model.setting_relation_n_Q_finished_1_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_1_max
    )
    model.setting_relation_n_Q_finished_2_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_2_min
    )
    model.setting_relation_n_Q_finished_2_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_2_max
    )
    model.setting_relation_n_Q_finished_3_min = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_3_min
    )
    model.setting_relation_n_Q_finished_3_max = Constraint(
        model.charging_cells_key_set, rule=setting_relation_n_Q_finished_3_max
    )
    model.setting_relation_n_Q_exit_min = Constraint(
        model.key_set, rule=setting_relation_n_Q_exit_min
    )
    model.setting_relation_n_Q_exit_max = Constraint(
        model.key_set, rule=setting_relation_n_Q_exit_max
    )
    model.setting_relation_n_Q_arriving_min = Constraint(
        model.key_set, rule=setting_relation_n_Q_arriving_min
    )
    model.setting_relation_n_Q_arriving_max = Constraint(
        model.key_set, rule=setting_relation_n_Q_arriving_max
    )
    print("Relations finished")
    # constraints relating to time step
    model.key_set_without_last_t = [
        el for el in model.key_set if el[0] < model.nb_timesteps - 2
    ]
    model.key_set_without_last_t_CS = [
        el for el in model.charging_cells_key_set if el[0] < model.nb_timesteps - 2
    ]
    model.key_set_without_last_t_NO_CS = [
        el for el in model.charging_cells_key_set if el[0] < model.nb_timesteps - 2
    ]
    model.key_set_with_only_last_ts = [
        el for el in model.key_set if el[0] == model.nb_timesteps-1
    ]
    model.key_set_with_only_last_ts_NO_CS = [
        el for el in model.charging_cells_key_set if el[0] == model.nb_timesteps-1
    ]

    model.key_set_with_only_last_two_ts = [
        el for el in model.key_set if el[0] >= model.nb_timesteps - 2
    ]

    model.key_set_with_only_last_two_ts_NO_CS = [
        el for el in model.charging_cells_key_set if el[0] >= model.nb_timesteps - 2
    ]

    model.key_set_with_only_last_three_ts = [
        el for el in model.key_set if el[0] >= model.nb_timesteps - 3
    ]

    model.key_set_with_only_last_three_ts_NO_CS = [
        el for el in model.charging_cells_key_set if el[0] >= model.nb_timesteps - 3
    ]

    model.key_set_with_only_last_four_ts = [
        el for el in model.key_set if el[0] >= model.nb_timesteps - 4
    ]

    model.key_set_with_only_last_four_ts_NO_CS = [
        el for el in model.charging_cells_key_set if el[0] >= model.nb_timesteps - 4
    ]

    model.key_set_with_only_last_five_ts = [
        el for el in model.key_set if el[0] >= model.nb_timesteps - 5
    ]

    model.key_set_with_only_last_five_ts_NO_CS = [
        el for el in model.charging_cells_key_set if el[0] >= model.nb_timesteps - 5
    ]
    model.limiting_n_pass = Constraint(model.key_set_with_only_last_two_ts,
                                                   rule=limiting_n_pass)

    model.limiting_n_finished_charging = Constraint(model.key_set_with_only_last_ts_NO_CS,
                                                   rule=limiting_n_finished_charging)

    model.limiting_n_incoming_vehicles = Constraint(model.key_set_with_only_last_two_ts,
                                                    rule=limiting_n_incoming_vehicles)

    model.limiting_n_in= Constraint(model.key_set_with_only_last_two_ts,
                                                    rule=limiting_n_in)

    model.limiting_n_exit = Constraint(model.key_set_with_only_last_ts,
                                     rule=limiting_n_exit)
    model.limiting_n_out = Constraint(model.key_set_with_only_last_ts,
                                     rule=limiting_n_out)
    model.limiting_n_in_wait_charge = Constraint(model.key_set_with_only_last_five_ts_NO_CS,
                                     rule=limiting_n_in_wait_charge)
    model.limiting_n_wait = Constraint(model.key_set_with_only_last_five_ts_NO_CS,
                                     rule=limiting_n_wait)
    model.limiting_n_wait_charge_next = Constraint(model.key_set_with_only_last_five_ts_NO_CS,
                                     rule=limiting_n_wait_charge_next)

    model.limiting_n_in_charge = Constraint(model.key_set_with_only_last_five_ts_NO_CS,
                                     rule=limiting_n_in_charge)

    model.limiting_n_charge1 = Constraint(model.key_set_with_only_last_five_ts_NO_CS,
                                     rule=limiting_n_charge1)
    model.limiting_n_charge2 = Constraint(model.key_set_with_only_last_four_ts_NO_CS,
                                     rule=limiting_n_charge2)
    model.limiting_n_charge3 = Constraint(model.key_set_with_only_last_three_ts_NO_CS,
                                     rule=limiting_n_charge3)
    model.limiting_n_output_charged1 = Constraint(model.key_set_with_only_last_four_ts_NO_CS,
                                     rule=limiting_n_output_charged1)
    model.limiting_n_output_charged2 = Constraint(model.key_set_with_only_last_three_ts_NO_CS,
                                     rule=limiting_n_output_charged2)
    model.limiting_n_output_charged3 = Constraint(model.key_set_with_only_last_two_ts_NO_CS,
                                     rule=limiting_n_output_charged3)
    model.limiting_n_finished_charge1 = Constraint(model.key_set_with_only_last_four_ts_NO_CS,
                                     rule=limiting_n_finished_charge1)
    model.limiting_n_finished_charge2= Constraint(model.key_set_with_only_last_three_ts_NO_CS,
                                     rule=limiting_n_finished_charge2)
    model.limiting_n_finished_charge3= Constraint(model.key_set_with_only_last_two_ts_NO_CS,
                                     rule=limiting_n_finished_charge3)
    model.limiting_n_exit_charge= Constraint(model.key_set_with_only_last_ts_NO_CS,
                                     rule=limiting_n_exit_charge)


    model.inserting_time_jump_n_pass_exit = Constraint(
        model.key_set_without_last_t, rule=inserting_time_jump_n_pass_exit
    )
    model.inserting_time_jump_Q_charging_1 = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_Q_charging_1
    )
    model.inserting_time_jump_Q_charging_2 = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_Q_charging_2
    )
    model.inserting_time_jump_Q_charging_3 = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_Q_charging_3
    )

    model.inserting_time_jump_n_charging_1 = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_n_charging_1
    )
    model.inserting_time_jump_n_charging_2 = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_n_charging_2
    )
    model.inserting_time_jump_n_charging_3 = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_n_charging_3
    )

    model.inserting_time_jump_n_charged_exit = Constraint(
        model.key_set_without_last_t_CS, rule=inserting_time_jump_n_charged_exit
    )
    # model.inserting_time_jump_Q_charged_exit = Constraint(
    #     model.key_set_without_last_t_CS, rule=inserting_time_jump_Q_charged_exit
    # )
    model.inserting_time_jump_Q_passed_exit = Constraint(
        model.key_set_without_last_t, rule=inserting_time_jump_Q_passed_exit
    )

    del model.key_set_without_last_t_CS
    del model.key_set_without_last_t
    del model.key_set_without_last_t_NO_CS

    print("Time jumps finished")
    # ISSUE here
    # backwards time jump (t-1)

    # TODO: this can be also more efficient: !!
    model.key_set_without_first_t = [
        el
        for el in model.key_set
        if el[0] > model.fleet_depart_times[el[2]]
    ]
    model.key_set_without_first_t_CS = [
        el
        for el in model.charging_cells_key_set
        if el[0] > model.fleet_depart_times[el[2]]
    ]
    model.key_set_without_first_t_NO_CS = [
        el
        for el in model.no_charging_cells_key_set
        if el[0] > model.fleet_depart_times[el[2]]
    ]

    model.queue_n = Constraint(model.key_set_without_first_t_CS, rule=queue_n)
    model.queue_Q = Constraint(model.key_set_without_first_t_CS, rule=queue_Q)

    model.entering_charging_station_n = Constraint(
        model.key_set_without_first_t_CS, rule=entering_charging_station_n
    )
    model.entering_charging_station_Q = Constraint(
        model.key_set_without_first_t_CS, rule=entering_charging_station_Q
    )
    del model.key_set_without_first_t_NO_CS
    del model.key_set_without_first_t_CS
    del model.key_set_without_first_t

    # TODO: create Param instance for with filter
    # for c in model.nb_cell:
    # if model.cell_charging_cap[c] == 0:
    #     # model.c_waiting_and_charging.add(
    #     #     sum(
    #     #         model.n_wait[to, c, fo]
    #     #         for to in model.nb_timestep
    #     #         for fo in model.nb_fleet
    #     #     )
    #     #     == 0
    #     # )
    #     model.c_waiting_and_charging.add(
    #         sum(
    #             model.n_charge1[to, c, fo]
    #             for to in model.nb_timestep
    #             for fo in model.nb_fleet
    #         )
    #         == 0
    #     )
    # for f in model.nb_fleet:
    #   print(c, "constr_vehicle_states for ... fleet ", f)
    # states of vehicles isolated for a time step
    # for t in model.nb_timestep:
    # if t == 0:
    # model.c_start_time.add(model.n_exit_pass[t, c, f] == 0)
    # model.c_start_time.add(model.Q_exit_passed[t, c, f] == 0)
    # model.c_start_time.add(model.n_finished_charge1[t, c, f] == 0)
    # model.c_start_time.add(model.Q_finished_charging[t, c, f] == t)
    # model.c_start_time.add(model.n_finished_charge2[t, c, f] == t)
    # model.c_start_time.add(model.Q_output_charge2[t, c, f] == t)
    # model.c_start_time.add(model.n_finished_charge2[t + 1, c, f] == t)
    # model.c_start_time.add(model.n_charge2[t + 1, c, f] == t)
    # model.c_start_time.add(model.Q_output_charge2[t + 1, c, f] == t)
    # model.c_start_time.add(model.n_finished_charge3[t, c, f] == t)
    # model.c_start_time.add(model.n_finished_charge3[t + 1, c, f] == t)
    # model.c_start_time.add(model.n_finished_charge3[t + 2, c, f] == t)
    # model.c_start_time.add(model.n_charge3[t, c, f] == t)
    # model.c_start_time.add(model.n_charge3[t + 1, c, f] == t)
    # model.c_start_time.add(model.Q_output_charge3[1, c, f] == t)
    # model.c_start_time.add(model.Q_output_charge3[2, c, f] == t)
    # model.c_start_time.add(
    #     model.n_to_charge[t, c, f] == model.n_in_charge[t, c, f]
    # )
    # model.c_start_time.add(model.n_wait_charge_next[t, c, f] == t)
    # model.c_start_time.add(model.Q_wait_charge_next[t, c, f] == t)

    # model.c_start_time.add(
    #     model.n_wait[t, c, f] + model.n_wait_charge_next[t, c, f]
    #     == model.n_in_wait[t, c, f]
    # )
    # model.c_start_time.add(
    #     model.Q_wait[t, c, f] + model.Q_wait_charge_next[t, c, f]
    #     == model.Q_in_wait[t, c, f]
    # )

    # passing
    # model.c_incoming_flow.add(
    #     model.n_in[t, c, f] + model.n_incoming_vehicles[t, c, f]
    #     == model.n_in_pass[t, c, f] + model.n_in_wait_charge[t, c, f]
    # )
    # model.c_incoming_flow.add(
    #     model.Q_in[t, c, f] + model.Q_incoming_vehicles[t, c, f]
    #     == model.Q_in_pass[t, c, f] + model.Q_in_charge_wait[t, c, f]
    # )
    # model.c_passing.add(model.n_in_pass[t, c, f] == model.n_pass[t, c, f])
    # model.c_passing.add(model.Q_in_pass[t, c, f] == model.Q_pass[t, c, f])
    # model.c_passing.add(
    #     model.E_consumed_pass[t, c, f]
    #     == model.n_pass[t, c, f]
    #     * model.cell_width[c]
    #     * model.fleet_d_spec[f]
    # )
    # waiting and charging
    # model.c_waiting_and_charging.add(
    #     model.n_in_wait_charge[t, c, f]
    #     == model.n_in_charge[t, c, f] + model.n_in_wait[t, c, f]
    # )
    # model.c_waiting_and_charging.add(
    #     model.n_in_wait[t, c, f] == model.n_wait[t, c, f]
    # )
    # # only half of the consumption is happening here as the assumption is that the charging infrastructure
    # # is mounted in the mid of the cell
    # model.c_passing.add(
    #     model.E_consumed_charge_wait[t, c, f]
    #     == model.n_in_wait_charge[t, c, f]
    #     * (1 / 2)
    #     * model.cell_width[c]
    #     * model.fleet_d_spec[f]
    # )
    # model.c_passing.add(
    #     model.Q_in_charge_wait[t, c, f]
    #     - model.E_consumed_charge_wait[t, c, f]
    #     == model.Q_in_charge[t, c, f] + model.Q_in_wait[t, c, f]
    # )
    # model.c_passing.add(model.Q_in_wait[t, c, f] == model.Q_wait[t, c, f])
    # # vehicles start charging
    # model.c_waiting_and_charging.add(
    #     model.n_charge1[t, c, f] == model.n_to_charge[t, c, f]
    # )
    # model.c_waiting_and_charging.add(
    #     model.E_charge1[t, c, f]
    #     <= model.n_charge1[t, c, f]
    #     * model.fleet_charge_cap[f]
    #     * model.time_resolution
    # )
    # model.c_waiting_and_charging.add(
    #     model.Q_output_charge1[t, c, f]
    #     == model.Q_input_charge2[t, c, f]
    #     + model.Q_finished_charge1[t, c, f]
    # )
    # model.c_waiting_and_charging.add(
    #     model.Q_output_charge2[t, c, f]
    #     == model.Q_input_charge3[t, c, f]
    #     + model.Q_finished_charge2[t, c, f]
    # )
    # model.c_waiting_and_charging.add(
    #     model.Q_output_charge3[t, c, f] == model.Q_finished_charge3[t, c, f]
    # )
    # model.c_waiting_and_charging.add(
    #     model.E_charge2[t, c, f]
    #     <= model.n_charge2[t, c, f]
    #     * model.fleet_charge_cap[f]
    #     * model.time_resolution
    # )
    # model.c_waiting_and_charging.add(
    #     model.E_charge3[t, c, f]
    #     <= model.n_charge3[t, c, f]
    #     * model.fleet_charge_cap[f]
    #     * model.time_resolution
    # )
    # model.c_waiting_and_charging.add(
    #     model.n_finished_charging[t, c, f]
    #     == model.n_finished_charge1[t, c, f]
    #     + model.n_finished_charge2[t, c, f]
    #     + model.n_finished_charge3[t, c, f]
    # )
    # model.c_waiting_and_charging.add(
    #     model.Q_finished_charging[t, c, f]
    #     == model.Q_finished_charge1[t, c, f]
    #     + model.Q_finished_charge2[t, c, f]
    #     + model.Q_finished_charge3[t, c, f]
    # )
    # model.c_passing.add(
    #     model.E_consumed_exit_charge[t, c, f]
    #     == model.n_finished_charging[t, c, f]
    #     * (1 / 2)
    #     * model.cell_width[c]
    #     * model.fleet_d_spec[f]
    # )
    # exiting the cell
    # model.c_exiting.add(
    #     model.n_exit[t, c, f]
    #     == model.n_exit_pass[t, c, f] + model.n_exit_charge[t, c, f]
    # )
    # model.c_exiting.add(
    #     model.Q_exit[t, c, f]
    #     == model.Q_exit_passed[t, c, f] + model.Q_exit_charged[t, c, f]
    # )
    # model.c_exiting.add(
    #     model.n_exit[t, c, f] - model.n_arrived_vehicles[t, c, f]
    #     == model.n_out[t, c, f]
    # )
    # model.c_exiting.add(
    #     model.Q_exit[t, c, f] - model.Q_arrived_vehicles[t, c, f]
    #     == model.Q_out[t, c, f]
    # )
    # model.c_waiting_and_charging.add(
    #     model.n_output_charged1[t, c, f]
    #     == model.n_finished_charge1[t, c, f] + model.n_charge2[t, c, f]
    # )

    # model.c_waiting_and_charging.add(
    #     model.n_output_charged2[t, c, f]
    #     == model.n_finished_charge2[t, c, f] + model.n_charge3[t, c, f]
    # )
    # model.c_waiting_and_charging.add(
    #     model.n_output_charged3[t, c, f]
    #     == model.n_finished_charge3[t, c, f]
    # )
    # in
    # model.constraint_nq_rel.add(
    #     model.Q_in[t, c, f]
    #     >= model.n_in[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_in[t, c, f]
    #     <= model.n_in[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    # )
    # in_pass
    # model.constraint_nq_rel.add(
    #     model.Q_in_pass[t, c, f]
    #     >= model.n_in_pass[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_min
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_in_pass[t, c, f]
    #     <= model.n_in_pass[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_max
    # )
    # in_wait_charge
    # model.constraint_nq_rel.add(
    #     model.Q_in_charge_wait[t, c, f]
    #     >= model.n_in_wait_charge[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_min
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_in_charge_wait[t, c, f]
    #     <= model.n_in_wait_charge[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_max
    # )
    # # # in_wait
    # model.constraint_nq_rel.add(
    #     model.Q_in_wait[t, c, f]
    #     >= model.n_in_wait[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_min
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_in_wait[t, c, f]
    #     <= model.n_in_wait[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_max
    # )
    # # wait
    # model.constraint_nq_rel.add(
    #     model.Q_wait[t, c, f]
    #     >= model.n_wait[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_wait[t, c, f]
    #     <= model.n_wait[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    # )
    # # # in_charge
    # model.constraint_nq_rel.add(
    #     model.Q_in_charge[t, c, f]
    #     >= model.n_in_charge[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_min
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_in_charge[t, c, f]
    #     <= model.n_in_charge[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_max
    # )
    # # # wait_charge_next
    # model.constraint_nq_rel.add(
    #     model.Q_wait_charge_next[t, c, f]
    #     >= model.n_wait_charge_next[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_min
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_wait_charge_next[t, c, f]
    #     <= model.n_wait_charge_next[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_max
    # )
    # # #
    # # charge1/input_charge1
    # model.constraint_nq_rel.add(
    #     model.Q_input_charge1[t, c, f]
    #     >= model.n_charge1[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_min
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_input_charge1[t, c, f]
    #     <= model.n_charge1[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_max
    # )
    # # charge2/input_charge2
    # model.constraint_nq_rel.add(
    #     model.Q_input_charge2[t, c, f]
    #     >= model.n_charge2[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_min
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_input_charge2[t, c, f]
    #     <= model.n_charge2[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_max
    # )
    #
    # # charge1/output_charge1
    # # TODO: this is the dangerous part
    # model.constraint_nq_rel.add(
    #     model.Q_output_charge1[t, c, f]
    #     >= model.n_output_charged1[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_min
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_output_charge1[t, c, f]
    #     <= model.n_output_charged1[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_max
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_output_charge2[t, c, f]
    #     >= model.n_output_charged2[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_min
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_output_charge2[t, c, f]
    #     <= model.n_output_charged2[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_max
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_output_charge3[t, c, f]
    #     >= model.n_output_charged3[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_min
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_output_charge3[t, c, f]
    #     <= model.n_output_charged3[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_max
    # )
    # # charge3/input_charge3
    # model.constraint_nq_rel.add(
    #     model.Q_input_charge3[t, c, f]
    #     >= model.n_charge3[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_min
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_input_charge3[t, c, f]
    #     <= model.n_charge3[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_max
    # )
    # # finished_charge
    # model.constraint_nq_rel.add(
    #     model.Q_finished_charging[t, c, f]
    #     >= model.n_finished_charging[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_min
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_finished_charging[t, c, f]
    #     <= model.n_finished_charging[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_max
    # )
    # # finished_charge1
    # model.constraint_nq_rel.add(
    #     model.Q_finished_charge1[t, c, f]
    #     >= model.n_finished_charge1[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_min
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_finished_charge1[t, c, f]
    #     <= model.n_finished_charge1[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_max
    # )
    # # finished_charge2
    # model.constraint_nq_rel.add(
    #     model.Q_finished_charge2[t, c, f]
    #     >= model.n_finished_charge2[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_min
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_finished_charge2[t, c, f]
    #     <= model.n_finished_charge2[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_max
    # )
    # # finished_charge3
    # model.constraint_nq_rel.add(
    #     model.Q_finished_charge3[t, c, f]
    #     >= model.n_finished_charge3[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_min
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_finished_charge3[t, c, f]
    #     <= model.n_finished_charge3[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_max
    # )
    # #
    # model.constraint_nq_rel.add(
    #     model.Q_exit[t, c, f]
    #     >= model.n_exit[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_exit[t, c, f]
    #     <= model.n_exit[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
    # )
    # # arrivals
    # model.constraint_nq_rel.add(
    #     model.Q_arrived_vehicles[t, c, f]
    #     >= model.n_arrived_vehicles[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_min
    # )
    # model.constraint_nq_rel.add(
    #     model.Q_arrived_vehicles[t, c, f]
    #     <= model.n_arrived_vehicles[t, c, f]
    #     * model.fleet_batt_cap[f]
    #     * model.SOC_max
    # )

    # if t in range(0, len(model.nb_timestep) - 1):
    # states of vehicles changing between two time steps
    # for t in range(0, len(model.nb_timestep) - 1):
    # passing
    # model.c_passing.add(
    #     model.n_exit_pass[t + 1, c, f] == model.n_pass[t, c, f]
    # )

    # # charging
    # # those who charge at time step t are the ones which are directly incoming and those which have waited

    # model.c_waiting_and_charging.add(
    #     model.Q_output_charge1[t + 1, c, f]
    #     == model.Q_input_charge1[t, c, f] + model.E_charge1[t, c, f]
    # )
    # model.c_waiting_and_charging.add(
    #     model.Q_output_charge2[t + 1, c, f]
    #     == model.Q_input_charge2[t, c, f] + model.E_charge2[t, c, f]
    # )

    # model.c_waiting_and_charging.add(
    #     model.n_output_charged1[t + 1, c, f] == model.n_charge1[t, c, f]
    # )
    # model.c_waiting_and_charging.add(
    #     model.n_output_charged2[t + 1, c, f] == model.n_charge2[t, c, f]
    # )
    # model.c_waiting_and_charging.add(
    #     model.n_output_charged3[t + 1, c, f] == model.n_charge3[t, c, f]
    # )
    # model.c_waiting_and_charging.add(
    #     model.Q_output_charge3[t + 1, c, f]
    #     == model.Q_input_charge3[t, c, f] + model.E_charge3[t, c, f]
    # )
    # model.c_waiting_and_charging.add(
    #     model.Q_exit_charged[t + 1, c, f]
    #     == model.Q_finished_charging[t, c, f]
    #     - model.E_consumed_exit_charge[t, c, f]
    # )
    # model.c_waiting_and_charging.add(
    #     model.n_exit_charge[t + 1, c, f]
    #     == model.n_finished_charging[t, c, f]
    # )
    # model.c_passing.add(#
    #     model.Q_exit_passed[t + 1, c, f]
    #     == model.Q_pass[t, c, f] - model.E_consumed_pass[t, c, f]
    # )
    # if t in range(1, len(model.nb_timestep)):
    #     # for t in range(1, len(model.nb_timestep)):
    #     # waiting
    #     # queuing at charging station behaves like a storage system:
    #     model.c_waiting_and_charging.add(
    #         model.n_wait[t, c, f]
    #         == model.n_wait[t - 1, c, f]
    #         + model.n_in_wait[t, c, f]
    #         - model.n_wait_charge_next[t, c, f]
    #     )
    #     model.c_waiting_and_charging.add(
    #         model.Q_wait[t, c, f]
    #         == model.Q_wait[t - 1, c, f]
    #         + model.Q_in_wait[t, c, f]
    #         - model.Q_wait_charge_next[t, c, f]
    #     )
    #     model.c_waiting_and_charging.add(
    #         model.n_to_charge[t, c, f]
    #         == model.n_in_charge[t, c, f]
    #         + model.n_wait_charge_next[t - 1, c, f]
    #     )
    #     model.c_waiting_and_charging.add(
    #         model.Q_input_charge1[t, c, f]
    #         == model.Q_in_charge[t, c, f]
    #         + model.Q_wait_charge_next[t - 1, c, f]
    #     )
    # if t == 0:
    #     model.c_geq_zero.add(model.n_incoming_vehicles[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.n_in[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.n_in_pass[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.n_pass[t, c, f] >= 0)
    #     # model.c_geq_zero.add(model.n_in_wait[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.n_in_wait_charge[t, c, f] >= 0)
    #     # model.c_geq_zero.add(model.n_wait[t, c, f] >= 0)
    #     # model.c_geq_zero.add(model.n_wait_charge_next[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.n_in_charge[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.n_to_charge[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.n_charge1[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.n_charge2[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.n_finished_charge1[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.n_finished_charge2[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.n_exit_charge[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.n_exit_pass[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.n_exit[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.n_out[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.n_arrived_vehicles[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.Q_incoming_vehicles[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.Q_in[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.Q_in_pass[t, c, f] >= 0)
    #     # model.c_geq_zero.add(model.Q_in_wait[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.Q_pass[t, c, f] >= 0)
    #     # model.c_geq_zero.add(model.Q_wait_charge_next[t, c, f] >= 0)
    #     # model.c_geq_zero.add(model.Q_wait[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.Q_input_charge1[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.Q_output_charge1[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.Q_input_charge2[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.Q_output_charge2[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.Q_exit_charged[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.Q_exit_passed[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.Q_arrived_vehicles[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.E_charge1[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.E_charge2[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.Q_finished_charge1[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.Q_finished_charge2[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.E_consumed_pass[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.E_consumed_charge_wait[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.E_consumed_exit_charge[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.Q_finished_charging[t, c, f] >= 0)
    #     model.c_geq_zero.add(model.n_finished_charging[t, c, f] >= 0)


def add_n_Q_relations(model: ConcreteModel):
    model.constraint_nq_rel = ConstraintList()
    model.c_rule_in = Constraint(
        model.nb_timestep, model.nb_cell, model.nb_cell, rule=constraint_rule_in
    )
    for t in model.nb_timestep:
        for c in model.nb_cell:
            model.c_cell_capacity.add(
                sum([model.E_charge1[t, c, f] for f in model.nb_fleet])
                + sum([model.E_charge2[t, c, f] for f in model.nb_fleet])
                + sum([model.E_charge3[t, c, f] for f in model.nb_fleet])
                <= model.cell_charging_cap[c] * 0.25
            )
            for f in model.nb_cell:
                # in
                # model.constraint_nq_rel.add(
                #     model.Q_in[t, c, f]
                #     >= model.n_in[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
                # )
                model.constraint_nq_rel.add(
                    model.Q_in[t, c, f]
                    <= model.n_in[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
                )
                # in_pass
                model.constraint_nq_rel.add(
                    model.Q_in_pass[t, c, f]
                    >= model.n_in_pass[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_min
                )
                model.constraint_nq_rel.add(
                    model.Q_in_pass[t, c, f]
                    <= model.n_in_pass[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_max
                )
                # in_wait_charge
                model.constraint_nq_rel.add(
                    model.Q_in_charge_wait[t, c, f]
                    >= model.n_in_wait_charge[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_min
                )
                model.constraint_nq_rel.add(
                    model.Q_in_charge_wait[t, c, f]
                    <= model.n_in_wait_charge[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_max
                )
                # # in_wait
                model.constraint_nq_rel.add(
                    model.Q_in_wait[t, c, f]
                    >= model.n_in_wait[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_min
                )
                model.constraint_nq_rel.add(
                    model.Q_in_wait[t, c, f]
                    <= model.n_in_wait[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_max
                )
                # wait
                model.constraint_nq_rel.add(
                    model.Q_wait[t, c, f]
                    >= model.n_wait[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
                )
                model.constraint_nq_rel.add(
                    model.Q_wait[t, c, f]
                    <= model.n_wait[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
                )
                # # in_charge
                model.constraint_nq_rel.add(
                    model.Q_in_charge[t, c, f]
                    >= model.n_in_charge[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_min
                )
                model.constraint_nq_rel.add(
                    model.Q_in_charge[t, c, f]
                    <= model.n_in_charge[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_max
                )
                # # wait_charge_next
                model.constraint_nq_rel.add(
                    model.Q_wait_charge_next[t, c, f]
                    >= model.n_wait_charge_next[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_min
                )
                model.constraint_nq_rel.add(
                    model.Q_wait_charge_next[t, c, f]
                    <= model.n_wait_charge_next[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_max
                )
                # #
                # charge1/input_charge1
                model.constraint_nq_rel.add(
                    model.Q_input_charge1[t, c, f]
                    >= model.n_charge1[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_min
                )
                model.constraint_nq_rel.add(
                    model.Q_input_charge1[t, c, f]
                    <= model.n_charge1[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_max
                )
                # charge2/input_charge2
                model.constraint_nq_rel.add(
                    model.Q_input_charge2[t, c, f]
                    >= model.n_charge2[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_min
                )
                model.constraint_nq_rel.add(
                    model.Q_input_charge2[t, c, f]
                    <= model.n_charge2[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_max
                )

                # charge1/output_charge1
                # TODO: this is the dangerous part
                model.constraint_nq_rel.add(
                    model.Q_output_charge1[t, c, f]
                    >= model.n_output_charged1[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_min
                )
                model.constraint_nq_rel.add(
                    model.Q_output_charge1[t, c, f]
                    <= model.n_output_charged1[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_max
                )
                model.constraint_nq_rel.add(
                    model.Q_output_charge2[t, c, f]
                    >= model.n_output_charged2[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_min
                )
                model.constraint_nq_rel.add(
                    model.Q_output_charge2[t, c, f]
                    <= model.n_output_charged2[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_max
                )
                model.constraint_nq_rel.add(
                    model.Q_output_charge3[t, c, f]
                    >= model.n_output_charged3[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_min
                )
                model.constraint_nq_rel.add(
                    model.Q_output_charge3[t, c, f]
                    <= model.n_output_charged3[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_max
                )
                # charge3/input_charge3
                model.constraint_nq_rel.add(
                    model.Q_input_charge3[t, c, f]
                    >= model.n_charge3[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_min
                )
                model.constraint_nq_rel.add(
                    model.Q_input_charge3[t, c, f]
                    <= model.n_charge3[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_max
                )
                # finished_charge
                model.constraint_nq_rel.add(
                    model.Q_finished_charging[t, c, f]
                    >= model.n_finished_charging[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_min
                )
                model.constraint_nq_rel.add(
                    model.Q_finished_charging[t, c, f]
                    <= model.n_finished_charging[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_max
                )
                # finished_charge1
                model.constraint_nq_rel.add(
                    model.Q_finished_charge1[t, c, f]
                    >= model.n_finished_charge1[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_min
                )
                model.constraint_nq_rel.add(
                    model.Q_finished_charge1[t, c, f]
                    <= model.n_finished_charge1[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_max
                )
                # finished_charge2
                model.constraint_nq_rel.add(
                    model.Q_finished_charge2[t, c, f]
                    >= model.n_finished_charge2[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_min
                )
                model.constraint_nq_rel.add(
                    model.Q_finished_charge2[t, c, f]
                    <= model.n_finished_charge2[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_max
                )
                # finished_charge3
                model.constraint_nq_rel.add(
                    model.Q_finished_charge3[t, c, f]
                    >= model.n_finished_charge3[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_min
                )
                model.constraint_nq_rel.add(
                    model.Q_finished_charge3[t, c, f]
                    <= model.n_finished_charge3[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_max
                )

                model.constraint_nq_rel.add(
                    model.Q_exit[t, c, f]
                    >= model.n_exit[t, c, f] * model.fleet_batt_cap[f] * model.SOC_min
                )
                model.constraint_nq_rel.add(
                    model.Q_exit[t, c, f]
                    <= model.n_exit[t, c, f] * model.fleet_batt_cap[f] * model.SOC_max
                )
                # arrivals
                model.constraint_nq_rel.add(
                    model.Q_arrived_vehicles[t, c, f]
                    >= model.n_arrived_vehicles[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_min
                )
                model.constraint_nq_rel.add(
                    model.Q_arrived_vehicles[t, c, f]
                    <= model.n_arrived_vehicles[t, c, f]
                    * model.fleet_batt_cap[f]
                    * model.SOC_max
                )


# TODO: das könnte noch irgendwie effizienter sein
def charging_at_restarea(model: ConcreteModel, t, c):
    return (
        quicksum(

                (model.E_charge1[t, c, f0] + model.E_charge2[t, c, f0] + model.E_charge3[t, c, f0]) * (1/(model.fleet_charge_cap[f0]/ 350))
                for f0 in model.nb_fleet
                if (t, c, f0) in model.charging_cells_key_set

        )
        # + quicksum(
        #     [
        #
        #         for f0 in model.nb_fleet
        #         if (t, c, f0) in model.charging_cells_key_set
        #     ]
        # )
        # + quicksum(
        #     [
        #         model.E_charge3[t, c, f0]
        #         for f0 in model.nb_fleet
        #         if (t, c, f0) in model.charging_cells_key_set
        #     ]
        # )
        <= model.cell_charging_cap[c] * 0.25
    )


def restraint_charging_capacity(model: ConcreteModel):
    model.t_cs = set([(el[0], el[1]) for el in model.charging_cells_key_set])
    model.c_cell_capacity = Constraint(model.t_cs, rule=charging_at_restarea)
    del model.t_cs


def minimize_waiting_and_charging(model: ConcreteModel):
    model.objective_function = Objective(
        expr=(
            quicksum(
                    model.n_wait[el] + model.n_wait_charge_next[el]
                    for el in model.charging_cells_key_set
            )
        ),
        sense=minimize,
    )


def plot_results(model, fleet_df):
    """
    plotting waiting and charging for all charging stations
        - plot cell acitivity
        - charging state along cells + charged energy
    :param model:
    :return:
    """
    # What variables are needed for cell plot?
    #   - get cells with capacity > 0
    #   - get sum n charging (n_charged1 + ...2 + ...3)
    #   - get n_waiting + n_next_charge
    #   - get sum E_charged1 + ...2 + ... 3

    # cell plot
    E_charge1 = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    E_charge2 = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    E_charge3 = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    n_charge1 = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    n_charge2 = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    n_charge3 = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    n_wait = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    n_wait_charge_next = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    Q_incoming_vehicles = np.zeros(
        [model.nb_timesteps, model.nb_cells, model.nb_fleets]
    )
    n_arrived_vehicles = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    Q_exit = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    E_consumed_pass = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    E_consumed_charge_wait = np.zeros(
        [model.nb_timesteps, model.nb_cells, model.nb_fleets]
    )
    E_consumed_exit_charge = np.zeros(
        [model.nb_timesteps, model.nb_cells, model.nb_fleets]
    )

    for t in model.nb_timestep:
        for c in model.nb_cell:
            for f in model.nb_fleet:
                n_wait[t, c, f] = model.n_wait[t, c, f].value
                E_charge1[t, c, f] = model.E_charge1[t, c, f].value
                E_charge2[t, c, f] = model.E_charge2[t, c, f].value
                E_charge3[t, c, f] = model.E_charge3[t, c, f].value
                n_charge2[t, c, f] = model.n_charge2[t, c, f].value
                n_charge3[t, c, f] = model.n_charge3[t, c, f].value
                n_charge1[t, c, f] = model.n_charge1[t, c, f].value
                n_wait_charge_next[t, c, f] = model.n_wait_charge_next[t, c, f].value
                Q_incoming_vehicles[t, c, f] = model.Q_incoming_vehicles[t, c, f].value
                Q_exit[t, c, f] = model.Q_exit[t, c, f].value
                n_arrived_vehicles[t, c, f] = model.n_arrived_vehicles[t, c, f].value
                E_consumed_pass[t, c, f] = model.E_consumed_pass[t, c, f].value
                E_consumed_charge_wait[t, c, f] = model.E_consumed_charge_wait[
                    t, c, f
                ].value
                E_consumed_exit_charge[t, c, f] = model.E_consumed_exit_charge[
                    t, c, f
                ].value

    charging_cell_ids = [c for c in model.nb_cell if model.cell_charging_cap[c] > 0]
    nb_charging_cells = len(charging_cell_ids)

    fig, axs = plt.subplots(nrows=nb_charging_cells, ncols=2)
    for ij in range(0, nb_charging_cells):
        print(ij)
        if charging_cell_ids[ij] in charging_cell_ids:
            # charging load
            c = charging_cell_ids[ij]
            load_profile = np.sum(
                E_charge1[:, c, :] + E_charge2[:, c, :] + E_charge3[:, c, :], axis=1
            )
            if nb_charging_cells == 1:
                # axs[0].grid(axis='x', color='0.95')
                axs[0].fill_between(
                    model.nb_timestep, load_profile, step="pre", alpha=0.4
                )
                axs[0].plot(
                    model.nb_timestep,
                    load_profile,
                    drawstyle="steps",
                    label="load profile",
                )
                axs[0].axhline(
                    y=model.cell_charging_cap[c] * model.time_resolution, linestyle="--"
                )
                axs[0].legend()
                axs[0].title.set_text("Cell nb. " + str(c))
            else:
                # axs[ij, 0].grid(axis='x', color='0.95')
                axs[ij, 0].fill_between(
                    model.nb_timestep, load_profile, step="pre", alpha=0.4
                )
                axs[ij, 0].plot(
                    model.nb_timestep,
                    load_profile,
                    drawstyle="steps",
                    label="load profile",
                )
                axs[ij, 0].axhline(
                    y=model.cell_charging_cap[c] * model.time_resolution, linestyle="--"
                )
                axs[ij, 0].legend()
                axs[ij, 0].title.set_text("Cell nb. " + str(c))

            # cars at charging station
            charging_cars = np.sum(
                n_charge1[:, c, :] + n_charge2[:, c, :] + n_charge3[:, c, :], axis=1
            )
            waiting_cars = np.sum(n_wait[:, c, :] + n_wait_charge_next[:, c, :], axis=1)
            if nb_charging_cells == 1:
                # axs[1].grid(axis='x', color='0.95')
                axs[1].plot(
                    model.nb_timestep,
                    waiting_cars,
                    drawstyle="steps",
                    label="nb waiting cars",
                )
                axs[1].plot(
                    model.nb_timestep,
                    charging_cars,
                    drawstyle="steps",
                    label="nb charging cars",
                )
                axs[1].legend()
                axs[1].title.set_text("Cell nb. " + str(c))
            else:
                # axs[ij, 1].grid(axis='x', color='0.95')
                axs[ij, 1].plot(
                    model.nb_timestep,
                    waiting_cars,
                    drawstyle="steps",
                    label="nb waiting cars",
                )
                axs[ij, 1].plot(
                    model.nb_timestep,
                    charging_cars,
                    drawstyle="steps",
                    label="nb charging cars",
                )
                axs[ij, 1].legend()
                axs[ij, 1].title.set_text("Cell nb. " + str(c))
    plt.tight_layout()
    plt.savefig("_cells.pdf")
    # fleet tracking
    routes = fleet_df.route.to_list()
    start_times = fleet_df.start_timestep.to_list()

    fig_fleets, axs = plt.subplots(nrows=model.nb_fleets, ncols=2)
    for f in model.nb_fleet:
        route = routes[f]
        # get route of fleet    --- check
        # get Q_incoming at starting cell and for rest Q_exit,
        # calculate Q_min and Q_max --- check
        # get charged energy for all the cells along the route
        q_min = model.SOC_min * model.fleet_sizes[f] * model.fleet_batt_cap[f]
        q_max = model.SOC_max * model.fleet_sizes[f] * model.fleet_batt_cap[f]

        total_fleet_energy_consumption = np.sum(
            E_consumed_pass[:, :, f]
            + E_consumed_charge_wait[:, :, f]
            + E_consumed_exit_charge[:, :, f]
        )

        # get time of last arrival

        times_arrivals = [
            i
            for i, v in enumerate(np.sum(n_arrived_vehicles[:, :, f], axis=1))
            if v > 0
        ]
        E_charged_fleet = np.sum(
            E_charge1[:, :, f] + E_charge2[:, :, f] + E_charge3[:, :, f], axis=0
        )

        E_charged_along_route = np.array([E_charged_fleet[c] for c in route])
        total_recharge_along_route = np.sum(E_charged_along_route)
        E_charged_along_route = np.where(
            E_charged_along_route > 0, E_charged_along_route, np.NaN
        )
        Q_exit_summed_fleet = np.sum(Q_exit[:, :, f], axis=0)

        Q_state = [sum(sum(Q_incoming_vehicles[:, :, f]))] + [
            Q_exit_summed_fleet[c] for c in route
        ]
        print("E_charged_fleet", E_charged_along_route)
        print(np.arange(0.5, len(E_charged_along_route), 1))
        Q_init = sum(sum(Q_incoming_vehicles[:, :, f]))
        needed_recharge = q_min - (Q_init - total_fleet_energy_consumption)
        axs[f, 1].axhline(y=q_max, color="#1d3557", linestyle="--", zorder=1)
        axs[f, 1].axhline(y=q_min, color="#1d3557", linestyle="--", zorder=1)
        axs[f, 1].fill_between(
            np.arange(-0.5, len(E_charged_along_route), 1),
            Q_state,
            alpha=0.4,
            color="#457b9d",
            zorder=5,
        )
        axs[f, 1].plot(
            np.arange(-0.5, len(E_charged_along_route), 1),
            Q_state,
            label="SOC",
            color="#457b9d",
            zorder=5,
        )
        # axs[f, 1].fill_between(
        #     np.arange(0.5, len(E_charged_along_route), 1),
        #     E_charged_along_route,
        #     step="pre",
        #     alpha=1,
        #     label="charged energy",
        #     color="#e63946",
        #     zorder=10,
        # )
        axs[f, 1].bar(
            np.arange(0, len(E_charged_along_route), 1),
            E_charged_along_route,
            # step="pre",
            # alpha=1,
            1,
            label="charged energy",
            color="#e63946",
            zorder=10,
        )
        axs[f, 1].set_xticks(list(np.arange(0, len(E_charged_along_route))))
        # axs[f, 1].set_xticklabels([str(lab) for lab in route])
        axs[f, 1].set_xlabel("cell nb")
        axs[f, 1].set_ylabel("SOC (kWh)")
        axs[f, 1].legend()
        axs[f, 1].title.set_text("Fleet ID: " + str(f))

        # adding text box
        text = (
            "Fleet ID: "
            + str(f)
            + "\nFleet size: "
            + str(int(model.fleet_sizes[f]))
            + "\nFleet arrived: "
            + str(np.sum(n_arrived_vehicles))
            + "\nDepart.: "
            + str(int(start_times[f]))
            + "\nLast arrival: "
            + str(max(times_arrivals))
            + "\nRequired recharge: "
            + str(round(needed_recharge, 2))
            + str("kWh")
            + "\nRecharged: "
            + str(round(total_recharge_along_route, 2))
        )
        print(text)
        axs[f, 0].set_axis_off()
        at = AnchoredText(text, prop=dict(size=10), frameon=True, loc="upper left")
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        axs[f, 0].add_artist(at)
    plt.tight_layout()
    plt.savefig("_fleets.pdf")
    return fig, fig_fleets


def plots_for_abstract(model, fleet_df):
    """
    plotting waiting and charging for all charging stations
        - plot cell acitivity
        - charging state along cells + charged energy
    :param model:
    :return:
    """
    # What variables are needed for cell plot?
    #   - get cells with capacity > 0
    #   - get sum n charging (n_charged1 + ...2 + ...3)
    #   - get n_waiting + n_next_charge
    #   - get sum E_charged1 + ...2 + ... 3

    # cell plot
    E_charge1 = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    E_charge2 = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    E_charge3 = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    n_charge1 = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    n_charge2 = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    n_charge3 = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    n_wait = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    n_wait_charge_next = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    Q_incoming_vehicles = np.zeros(
        [model.nb_timesteps, model.nb_cells, model.nb_fleets]
    )
    n_arrived_vehicles = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    Q_exit = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    E_consumed_pass = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    E_consumed_charge_wait = np.zeros(
        [model.nb_timesteps, model.nb_cells, model.nb_fleets]
    )
    E_consumed_exit_charge = np.zeros(
        [model.nb_timesteps, model.nb_cells, model.nb_fleets]
    )
    Q_pass = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    Q_in_wait = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])
    Q_in_charge = np.zeros([model.nb_timesteps, model.nb_cells, model.nb_fleets])

    for t in model.nb_timestep:
        for c in model.nb_cell:
            for f in model.nb_fleet:
                n_wait[t, c, f] = model.n_wait[t, c, f].value
                E_charge1[t, c, f] = model.E_charge1[t, c, f].value
                E_charge2[t, c, f] = model.E_charge2[t, c, f].value
                E_charge3[t, c, f] = model.E_charge3[t, c, f].value
                n_charge2[t, c, f] = model.n_charge2[t, c, f].value
                n_charge3[t, c, f] = model.n_charge3[t, c, f].value
                n_charge1[t, c, f] = model.n_charge1[t, c, f].value
                n_wait_charge_next[t, c, f] = model.n_wait_charge_next[t, c, f].value
                Q_incoming_vehicles[t, c, f] = model.Q_incoming_vehicles[t, c, f].value
                Q_exit[t, c, f] = model.Q_exit[t, c, f].value
                Q_pass[t, c, f] = model.Q_pass[t, c, f].value
                Q_in_wait[t, c, f] = model.Q_in_wait[t, c, f].value
                Q_in_charge[t, c, f] = model.Q_in_charge[t, c, f].value
                n_arrived_vehicles[t, c, f] = model.n_arrived_vehicles[t, c, f].value
                E_consumed_pass[t, c, f] = model.E_consumed_pass[t, c, f].value
                E_consumed_charge_wait[t, c, f] = model.E_consumed_charge_wait[
                    t, c, f
                ].value
                E_consumed_exit_charge[t, c, f] = model.E_consumed_exit_charge[
                    t, c, f
                ].value

    charging_cell_ids = [c for c in model.nb_cell if model.cell_charging_cap[c] > 0]
    nb_charging_cells = len(charging_cell_ids)

    fig, axs = plt.subplots(nrows=nb_charging_cells, ncols=2, figsize=(11, 3))
    for ij in range(0, nb_charging_cells):
        print(ij)
        if charging_cell_ids[ij] in charging_cell_ids:
            # charging load
            c = charging_cell_ids[ij]
            load_profile = np.sum(
                E_charge1[:, c, :] + E_charge2[:, c, :] + E_charge3[:, c, :], axis=1
            )
            if nb_charging_cells == 1:
                # axs[0].grid(axis='x', color='0.95')
                axs[0].fill_between(
                    model.nb_timestep, load_profile, step="pre", alpha=0.4
                )
                axs[0].plot(
                    model.nb_timestep,
                    load_profile,
                    drawstyle="steps",
                    label="load profile",
                )
                axs[0].axhline(
                    y=model.cell_charging_cap[c] * model.time_resolution, linestyle="--"
                )
                axs[0].legend()
                axs[0].title.set_text("Cell nb. " + str(c))
            else:
                # axs[ij, 0].grid(axis='x', color='0.95')
                axs[ij, 0].fill_between(
                    model.nb_timestep, load_profile, step="pre", alpha=0.4
                )
                axs[ij, 0].plot(
                    model.nb_timestep,
                    load_profile,
                    drawstyle="steps",
                    label="load profile",
                )
                axs[ij, 0].axhline(
                    y=model.cell_charging_cap[c] * model.time_resolution, linestyle="--"
                )
                axs[ij, 0].legend()
                axs[ij, 0].title.set_text("Cell nb. " + str(c))

            # cars at charging station
            charging_cars = np.sum(
                n_charge1[:, c, :] + n_charge2[:, c, :] + n_charge3[:, c, :], axis=1
            )
            waiting_cars = np.sum(n_wait[:, c, :] + n_wait_charge_next[:, c, :], axis=1)
            if nb_charging_cells == 1:
                # axs[1].grid(axis='x', color='0.95')
                axs[1].plot(
                    model.nb_timestep,
                    waiting_cars,
                    drawstyle="steps",
                    label="nb. waiting cars",
                )
                axs[1].plot(
                    model.nb_timestep,
                    charging_cars,
                    drawstyle="steps",
                    label="nb. charging cars",
                )
                axs[1].legend()
                # axs[1].title.set_text("Cell nb. " + str(c))
                axs[1].set_ylabel("nb. cars at charging station")
                axs[1].set_xlabel("timestep")
            else:
                # axs[ij, 1].grid(axis='x', color='0.95')
                axs[ij, 1].plot(
                    model.nb_timestep,
                    waiting_cars,
                    drawstyle="steps",
                    label="nb waiting cars",
                )
                axs[ij, 1].plot(
                    model.nb_timestep,
                    charging_cars,
                    drawstyle="steps",
                    label="nb charging cars",
                )
                axs[ij, 1].legend()
                axs[ij, 1].title.set_text("Cell nb. " + str(c))
    plt.tight_layout()
    plt.savefig("_cells_abstract.png")
    # fleet tracking
    routes = fleet_df.route.to_list()
    start_times = fleet_df.start_timestep.to_list()

    fig_fleets, axs = plt.subplots(nrows=1, ncols=2, figsize=(11, 3))
    # for f in model.nb_fleet:
    route = routes[f]
    # get route of fleet    --- check
    # get Q_incoming at starting cell and for rest Q_exit,
    # calculate Q_min and Q_max --- check
    # get charged energy for all the cells along the route
    q_min = (model.SOC_min * model.fleet_sizes[f] * model.fleet_batt_cap[f]) / 1000
    q_max = (model.SOC_max * model.fleet_sizes[f] * model.fleet_batt_cap[f]) / 1000

    total_fleet_energy_consumption = np.sum(
        E_consumed_pass[:, :, f]
        + E_consumed_charge_wait[:, :, f]
        + E_consumed_exit_charge[:, :, f]
    )

    # get time of last arrival

    times_arrivals = [
        i for i, v in enumerate(np.sum(n_arrived_vehicles[:, :, f], axis=1)) if v > 0
    ]
    E_charged_fleet = np.sum(
        E_charge1[:, :, f] + E_charge2[:, :, f] + E_charge3[:, :, f], axis=0
    )
    E_charged_along_route = np.array([E_charged_fleet[c] for c in route])
    total_recharge_along_route = np.sum(E_charged_along_route)
    E_charged_along_route = np.where(
        E_charged_along_route > 0, E_charged_along_route, np.NaN
    )
    Q_exit_summed_fleet = np.sum(Q_exit[:, :, f], axis=0)

    Q_state = [
        sum(sum(Q_incoming_vehicles[:, :, f]))
    ]  # + [        Q_exit_summed_fleet[c] for c in route]
    x = [-0.5]
    # Q_state.append(sum(sum(Q_incoming_vehicles[:, :, f])))
    for ij in range(0, len(route)):
        # add SOC mid cell, half of Q_pass + Q_charge_wait
        half_time = (
            np.sum(Q_pass[:, route[ij], f])
            + np.sum(Q_in_charge[:, route[ij], f])
            + np.sum(Q_in_wait[:, route[ij], f])
            - np.sum(E_consumed_pass[:, route[ij], f]) / 2
        )
        # then add cell when exited Q_
        Q_state.append(half_time)
        x.append(ij)
        if np.sum(E_charge1[:, route[ij], f]) > 0:
            soc_update = (
                Q_state[-1]
                + np.sum(E_charge1[:, route[ij], f])
                + np.sum(E_charge2[:, route[ij], f])
                + np.sum(E_charge3[:, route[ij], f])
            )
            Q_state.append(soc_update)
            x.append(ij)

        full_time = np.sum(Q_exit[:, route[ij], f])
        Q_state.append(full_time)
        x.append(ij + 0.5)

    Q_state = np.array(Q_state) / 1000
    print("E_charged_fleet", E_charged_along_route)
    print(np.arange(0.5, len(E_charged_along_route), 1))
    Q_init = sum(sum(Q_incoming_vehicles[:, :, f]))
    needed_recharge = q_min - (Q_init - total_fleet_energy_consumption)
    axs[1].axhline(
        y=q_max, color="#1d3557", linestyle="--", zorder=1, label="Maximum SOC"
    )
    axs[1].axhline(
        y=q_min, color="#1d3557", linestyle="--", zorder=1, label="Minimum SOC"
    )
    axs[1].fill_between(
        x,
        Q_state,
        alpha=0.4,
        color="#457b9d",
        zorder=5,
    )
    axs[1].plot(
        x,
        Q_state,
        label="SOC",
        color="#457b9d",
        zorder=5,
    )
    # axs[0].fill_between(
    #     np.arange(0.5, len(E_charged_along_route), 1),
    #     E_charged_along_route,
    #     step="pre",
    #     alpha=1,
    #     label="charged energy",
    #     color="#e63946",
    #     zorder=10,
    # )
    axs[0].bar(
        np.arange(0, len(E_charged_along_route), 1),
        E_charged_along_route,
        0.8,
        label="charged energy",
        color="#e63946",
        zorder=10,
    )
    print(np.sum(E_charged_along_route))
    axs[0].set_xticks(list(np.arange(0, len(E_charged_along_route))))
    # axs[0].set_xticklabels([str(lab) for lab in route])
    axs[0].set_xlabel("highway section along route of fleet")
    axs[0].set_ylabel("charged energy (kWh)")
    # axs[0].legend()
    # axs[0].title.set_text("Fleet ID: " + str(f))

    axs[1].set_xticks(list(np.arange(0, len(E_charged_along_route))))
    # axs[1].set_xticklabels([str(lab) for lab in route])
    axs[1].set_xlabel("highway section along route of fleet")
    axs[1].set_ylabel("SOC (MWh)")
    axs[1].legend()
    axs[0].set_ylim([0, 700])
    axs[0].set_xlim([-0.5, len(E_charged_along_route)])
    axs[1].set_xlim([-0.5, len(E_charged_along_route)])
    # adding text box
    text = (
        "Fleet ID: "
        + str(f)
        + "\nFleet size: "
        + str(int(model.fleet_sizes[f]))
        + "\nFleet arrived: "
        + str(np.sum(n_arrived_vehicles[:, :, f]))
        + "\nDepart.: "
        + str(int(start_times[f]))
        + "\nLast arrival: "
        + str(max(times_arrivals))
        + "\nRequired recharge: "
        + str(round(needed_recharge, 2))
        + str("kWh")
        + "\nRecharged: "
        + str(round(total_recharge_along_route, 2))
    )
    print(text)
    # axs[0].set_axis_off()
    # at = AnchoredText(text, prop=dict(size=10), frameon=True, loc="upper left")
    # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    # axs[0].add_artist(at)
    plt.tight_layout()
    plt.savefig("_fleets_abstract.png")

    return fig, fig_fleets


def create_set_init(model):
    """
    time range is already defined
    :param model:
    :param fleet_df:
    :param cell_df:
    :return:
    """
    # model.nb_timesteps
    # model.nb_timestep
    # TODO: create here set of touples with each three entries
    # for each fleet: from start time until end + cell only including the cells along the route
    # fleet_df = read_fleets(pd.read_csv("data/fleets.csv"))
    fleet_df = model.fleet_df
    start_time_steps = model.fleet_depart_times
    routes = model.fleet_routes

    for ij in range(0, len(fleet_df)):
        tau = start_time_steps[ij]
        r = routes[ij]
        # (time, highway section, fleet)
        for t in range(tau, int(model.nb_timesteps)):
            for c in r:
                yield (t, c, ij)



def create_set_routing(model):
    """
    time range is already defined
    :param model:
    :param fleet_df:
    :param cell_df:
    :return:
    """
    # model.nb_timesteps
    # model.nb_timestep
    # TODO: create here set of touples with each three entries
    # for each fleet: from start time until end + cell only including the cells along the route
    # fleet_df = read_fleets(pd.read_csv("data/fleets.csv"))
    start_time_steps = model.fleet_depart_times
    routes = model.fleet_routes
    fleet_df = model.fleet_df
    for ij in range(0, len(fleet_df)):
        tau = start_time_steps[ij]
        r = routes[ij]
        # (time, highway section, fleet)
        for t in range(tau, int(model.nb_timesteps)):
            for kl in range(0, len(r) - 1):
                yield (t, kl, ij)



def create_set(model):
    """
    time range is already defined
    :param model:
    :param fleet_df:
    :param cell_df:
    :return:
    """
    # model.nb_timesteps
    # model.nb_timestep
    # TODO: create here set of touples with each three entries
    # for each fleet: from start time until end + cell only including the cells along the route
    # fleet_df = read_fleets(pd.read_csv("data/fleets.csv"))
    start_time_steps = model.fleet_depart_times
    routes = model.fleet_routes
    fleet_df = model.fleet_df
    for ij in range(0, len(fleet_df)):
        tau = start_time_steps[ij]
        r = routes[ij]
        # (time, highway section, fleet)
        for t in range(0, int(model.nb_timesteps)):
            for c in model.nb_cell:
                yield (t, c, ij)


def get_variables_from_model(model):
    n_wait_charge_next = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_charge1 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )

    n_charge2 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )

    n_arrived_vehicles = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    E_charge1 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )

    Q_arrived_vehicles = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_wait = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    E_charge2 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )

    E_charge3 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_charge3 = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_arrived_vehicles = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    n_pass = np.zeros(
        [
            len(model.nb_timestep),
            len(model.nb_cell),
            len(model.nb_fleet),
        ]
    )
    for t in model.nb_timestep:
        for c in model.nb_cell:
            for f in model.nb_fleet:
                if (t, c, f) in model.key_set:
                    n_arrived_vehicles[t, c, f] = model.n_arrived_vehicles[
                        t, c, f
                    ].value
                    Q_arrived_vehicles[t, c, f] = model.Q_arrived_vehicles[
                        t, c, f
                    ].value
                    if (t, c, f) in model.charging_cells_key_set:
                        n_wait[t, c, f] = model.n_wait[t, c, f].value
                        n_wait_charge_next[t, c, f] = model.n_wait_charge_next[t, c, f].value
                        n_charge1[t, c, f] = model.n_charge1[t, c, f].value
                        n_charge2[t, c, f] = model.n_charge2[t, c, f].value
                        n_charge3[t, c, f] = model.n_charge3[t, c, f].value
                        E_charge1[t, c, f] = model.E_charge1[t, c, f].value
                        E_charge2[t, c, f] = model.E_charge2[t, c, f].value
                        E_charge3[t, c, f] = model.E_charge3[t, c, f].value
                        n_pass[t, c, f] = model.n_pass[t, c, f].value

    return {"n_arrived_vehicles": n_arrived_vehicles, "Q_arrived_vehicles": Q_arrived_vehicles, "n_wait": n_wait, "n_wait_charge_next": n_wait_charge_next,
            "n_charge1": n_charge1, "n_charge2": n_charge2, "n_charge3": n_charge3, "E_charge1": E_charge1,
            "E_charge2": E_charge2, "E_charge3": E_charge3, "n_pass": n_pass}


def write_output_files(model, time_of_optimization, filename):
    """
    writing output files of model
    :param model:
    :return:
    """
    # TODO: file 1
    #   -> for each charging station: number of waiting vehicles (n_wait, n_wait_charge_next)
    #   -> and number of charging vehicles at each point in time (n_charge1+2+3)
    #   -> amount of energy charged at each time step (E_charge1+2+3)
    #   -> CHECK

    results = get_variables_from_model(model)
    cs_specifics = pd.DataFrame()
    inds_of_charging_cells = [ind for ind in model.nb_cell if model.cell_charging_cap[ind] > 0]
    for ij in range(0, len(inds_of_charging_cells)):
        d = {}
        c = inds_of_charging_cells[ij]
        # ts = set([el[0] for el in model.key_set if el[1] == c])
        # waiting
        d["cell_id"] = c
        d["capacity"] = model.cell_charging_cap[c]
        for t in model.nb_timestep:
            d["waiting at t=" + str(t)] = np.sum(results["n_wait"][t, c, :]) + np.sum(results["n_wait_charge_next"][t, c, :])
            d["charging at t=" + str(t)] = np.sum(results["n_charge1"][t, c, :]) + np.sum(results["n_charge2"][t, c, :]) + np.sum(results["n_charge3"][t, c, :])
            d["E charged at t=" + str(t)] = np.sum(results["E_charge1"][t, c, :]) + np.sum(results["E_charge2"][t, c, :]) + np.sum(results["E_charge3"][t, c, :])
            d["pass at t=" + str(t)] = np.sum(results["n_pass"][t, c, :]) + np.sum(results["n_pass"][t, c, :]) + np.sum(results["n_pass"][t, c, :])

        cs_specifics = cs_specifics.append(d, ignore_index=True)

    cs_specifics.to_csv("results/" + time_of_optimization + "_charging_stations" + filename + ".csv")
    # TODO: file 2
    #   -> for each fleet: all input attributes + arrivals in dictioniary format
    #   -> if not sum(arrivals) == sum(depart), add a disclaimer that these did not arrive

    fleet_specifics = pd.DataFrame()
    for f in model.nb_fleet:
        d = {}
        d["fleet_id"] = f
        d["charge_cap"] = model.fleet_charge_cap[f]
        d["batt_cap"] = model.fleet_batt_cap[f]
        d["d_spec"] = model.fleet_d_spec[f]
        d["incoming"] = sum(model.fleet_incoming[f].values())
        if sum(model.fleet_incoming[f].values()) - np.sum(results["n_arrived_vehicles"][:, :, f]) > 0.01:
            d["all_arrived"] = False
        else:
            d["all_arrived"] = True

        d["arrival_SOC"] = np.sum(results["Q_arrived_vehicles"][:, :, f])/(model.fleet_charge_cap[f]*np.sum(results["n_arrived_vehicles"][:, :, f]))

        fleet_specifics = fleet_specifics.append(d, ignore_index=True)

    fleet_specifics.to_csv("results/" + time_of_optimization + "_fleet_infos" + filename + ".csv")

