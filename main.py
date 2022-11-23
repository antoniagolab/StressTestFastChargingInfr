import numpy as np
from utils import *
# from _import_optimization_files import *
from pyomo.environ import *
from termcolor import colored
from _optimization_utils import *
import time
from pyomo.util.model_size import build_model_size_report

cells = pd.read_csv("data/20220722-232828_cells_input.csv")
time_resolution = 0.25

nb_time_steps = 120
time_frame = range(0, nb_time_steps + 1)

nb_cells = len(cells)


SOC_min = 0.1
SOC_max = 1

t_min = 0.2
for fleet_filename in ["summer_workdayfleet_input_20220719_compressed_probe2",
                       "winter_workdayfleet_input_20220722_compressed_probe2"]:

    fleet_df = read_fleets(pd.read_csv("data/" + fleet_filename + ".csv", delimiter=";"))
    fleet_df["start_timestep"] = [int(el) for el in fleet_df.start_timestep]
    fleet_df["fleet_id"] = range(0, len(fleet_df))
    fleet_df["fleet_id"] = range(0, len(fleet_df))
    fleet_df = fleet_df.set_index("fleet_id")
    # print(fleet_df)
    # fleet_df = fleet_df[fleet_df.index.isin(range(0, 10))]
    nb_fleets = len(fleet_df)
    print(nb_time_steps, nb_cells, nb_fleets)


    # OPTIMIZATION
    charging_model = ConcreteModel()

    start = time.time()
    # adding constraints ...
    print("\nDefining decision variables ...")
    t0 = time.time()
    add_decision_variables(
        charging_model,
        time_resolution,
        nb_fleets,
        nb_cells,
        nb_time_steps,
        SOC_min,
        SOC_max,
        fleet_df,
        cells,
        t_min,
    )
    print("... took ", str(time.time() - t0), " sec")

    print("\nInitializing fleets ...")
    t1 = time.time()
    initialize_fleets(charging_model, fleet_df)
    print("... took ", str(time.time() - t1), " sec")

    print("\nInitializing cell geometry ...")
    t2 = time.time()
    # initialize_cells(charging_model, cells)
    print("... took ", str(time.time() - t2), " sec")


    print("\nConstraining vehicles activities and states ...")
    t3 = time.time()
    constr_vehicle_states(charging_model)
    print("... took ", str(time.time() - t3), " sec")

    # add_n_Q_relations(charging_model)
    print("\nConstraining charging activity at cells ...")
    t4 = time.time()
    restraint_charging_capacity(charging_model)
    print("... took ", str(time.time() - t4), " sec")

    print("\nAdding objective function ...")
    t5 = time.time()
    minimize_waiting_and_charging(charging_model)
    print("... took ", str(time.time() - t5), " sec")
    # _file = open("Math-Equations.txt", "w", encoding="utf-8")
    # charging_model.pprint(ostream=_file, verbose=False, prefix="")
    # _file.close()

    print(build_model_size_report(charging_model))

    opt = SolverFactory("gurobi", solver_io="python")
    # opt.options["TimeLimit"] = 14400
    # opt.options["OptimalityTol"] = 1e-2
    #opt.options["BarConvTol"] = 1e-11
    # opt.options["Cutoff"] = 1e-3
    # opt.options["CrossoverBasis"] = 0
    opt.options["Crossover"] = 0
    opt.options["Method"] = 2
    opt_success = opt.solve(
        charging_model, report_timing=True, tee=True
    )
    # opt_success = opt.solve(
    #     charging_model
    # )
    print(
        colored(
            "\nTotal time of model initialization and solution: "
            + str(time.time() - start)
            + "\n",
            "green",
        )
    )
    time_of_optimization = time.strftime("%Y%m%d-%H%M%S")
    write_output_files(charging_model, time_of_optimization, fleet_filename)
    # )
    # print("Here: ", charging_model.n_incoming_vehicles[0, 0, 0].value)
    n_exit = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    n_to_charge = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    n_incoming_vehicles = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    Q_in_pass = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    E_consumed_pass = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    n_in_wait_charge = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    n_in = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_in_charge = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_to_charge = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )


    Q_exit_pass = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_charge1 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_charge2 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_arrived_vehicles = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_finished_charging = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_input_charge1 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_in_charge = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_in = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_incoming_vehicles = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    Q_pass = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_exit = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    Q_exit_charged = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_output_charge1 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    Q_output_charge1 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    Q_output_charge2 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    E_charge1 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )


    Q_arrived_vehicles = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    n_pass = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    n_exit_charge = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    n_wait_charge_next = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    n_in_wait = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    E_consumed_charge_wait = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    E_consumed_exit_charge = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_wait_charge_next = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_wait_charge_next = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_wait = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_charge2 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    E_charge2 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_out = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    E_charge3 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    Q_wait = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    Q_in_wait = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_in_charge_wait = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_out = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_charge3 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    Q_input_charge2 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    n_output_charged1 = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    Q_finished_charging = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )
    n_incoming_vehicles = np.zeros(
        [
            len(charging_model.nb_timestep),
            len(charging_model.nb_cell),
            len(charging_model.nb_fleet),
        ]
    )

    for t in charging_model.nb_timestep:
        for c in charging_model.nb_cell:
            for f in charging_model.nb_fleet:
                if (t, c, f) in charging_model.key_set:
                    # n_exit[t, c, f] = charging_model.n_exit[t, c, f].value
                    # n_wait[t, c, f] = charging_model.n_wait[t, c, f].value
                    Q_out[t, c, f] = charging_model.Q_out[t, c, f].value
                    # n_out[t, c, f] = charging_model.n_out[t, c, f].value
                    # Q_exit_pass[t, c, f] = charging_model.Q_exit_passed[t, c, f].value

                    # n_to_charge[t, c, f] = charging_model.n_to_charge[t, c, f].value
                    n_incoming_vehicles[t, c, f] = charging_model.n_incoming_vehicles[
                        t, c, f
                    ].value

                    # ].value
                    # Q_in_pass[t, c, f] = charging_model.Q_in_pass[t, c, f].value

                    # n_pass[t, c, f] = charging_model.n_pass[t, c, f].value
                    # E_consumed_pass[t, c, f] = charging_model.E_consumed_pass[t, c, f].value
                    # n_in_wait_charge[t, c, f] = charging_model.n_in_wait_charge[
                    #     t, c, f
                    # ].value
                    # Q_wait_charge_next[t, c, f] = charging_model.Q_wait_charge_next[
                    #     t, c, f
                    # ].value
                    # n_in[t, c, f] = charging_model.n_in[t, c, f].value
                    # Q_in[t, c, f] = charging_model.Q_in[t, c, f].value
                    # n_in_charge[t, c, f] = charging_model.n_in_charge[t, c, f].value
                    # Q_input_charge1[t, c, f] = charging_model.Q_input_charge1[t, c, f].value
                    # n_to_charge[t, c, f] = charging_model.n_to_charge[t, c, f].value
                    # n_charge1[t, c, f] = charging_model.n_charge1[t, c, f].value
                    # # n_charge2[t, c, f] = charging_model.n_charge2[t, c, f].value
                    n_arrived_vehicles[t, c, f] = charging_model.n_arrived_vehicles[
                        t, c, f
                    ].value


                    # Q_exit_charged[t, c, f] = charging_model.Q_exit_charged[t, c, f].value
                    # Q_output_charge1[t, c, f] = charging_model.Q_output_charge1[
                    #     t, c, f
                    # ].value
                    # Q_output_charge2[t, c, f] = charging_model.Q_output_charge2[
                    #     t, c, f
                    # ].value
                    # Q_input_charge2[t, c, f] = charging_model.Q_input_charge2[t, c, f].value
                    # E_charge1[t, c, f] = charging_model.E_charge1[t, c, f].value
                    # E_charge2[t, c, f] = charging_model.E_charge2[t, c, f].value
                    # E_charge3[t, c, f] = charging_model.E_charge3[t, c, f].value
                    # n_arrived_vehicles[t, c, f] = charging_model.n_arrived_vehicles[
                    #     t, c, f
                    # ].value
                    Q_arrived_vehicles[t, c, f] = charging_model.Q_arrived_vehicles[
                        t, c, f
                    ].value
                    Q_incoming_vehicles[t, c, f] = charging_model.Q_incoming_vehicles[
                        t, c, f].value
                    Q_exit[t, c, f] = charging_model.Q_exit[t, c, f].value
                    Q_pass[t, c, f] = charging_model.Q_pass[t, c, f].value
                    # n_exit_charge[t, c, f] = charging_model.n_exit_charge[t, c, f].value
                    # n_wait_charge_next[t, c, f] = charging_model.n_wait_charge_next[
                    #     t, c, f
                    # ].value
                    # n_in_wait[t, c, f] = charging_model.n_in_wait[t, c, f].value
                    # n_exit_charge[t, c, f] = charging_model.n_exit_charge[t, c, f].value

                    # n_wait[t, c, f] = charging_model.n_wait[t, c, f].value
                    # Q_wait[t, c, f] = charging_model.Q_wait[t, c, f].value
                    # n_charge2[t, c, f] = charging_model.n_charge2[t, c, f].value
                    # n_charge3[t, c, f] = charging_model.n_charge3[t, c, f].value

                    # n_output_charged1[t, c, f] = charging_model.n_output_charged1[
                    #     t, c, f
                    # ].value
                    # # Q_finished_charge1[t, c, f] = charging_model.Q_finished_charge1[t, c, f].value
                    # Q_finished_charge2[t, c, f] = charging_model.Q_finished_charge2[
                    #     t, c, f
                    # ].value

                    if (t, c, f) in charging_model.charging_cells_key_set:
                        n_wait[t, c, f] = charging_model.n_wait[t, c, f].value
                        E_charge1[t, c, f] = charging_model.E_charge1[t, c, f].value
                        n_charge1[t, c, f] = charging_model.n_charge1[t, c, f].value
                        E_charge1[t, c, f] = charging_model.E_charge1[t, c, f].value
                        E_charge2[t, c, f] = charging_model.E_charge2[t, c, f].value
                        E_charge3[t, c, f] = charging_model.E_charge3[t, c, f].value

                        Q_finished_charging[t, c, f] = charging_model.Q_finished_charging[
                            t, c, f
                        ].value
                        n_finished_charging[t, c, f] = charging_model.n_finished_charging[
                            t, c, f
                        ].value
                        Q_in_charge[t, c, f] = charging_model.Q_in_charge[t, c, f].value
                        Q_in_charge_wait[t, c, f] = charging_model.Q_in_charge_wait[
                            t, c, f
                        ].value
                        Q_in_wait[t, c, f] = charging_model.Q_in_wait[t, c, f].value


    # print("Q_pass", Q_pass)
    # # print(np.sum(n_exit, axis=0))
    # print("n_to_charge", np.sum(n_to_charge, axis=0))
    # # print(np.sum(n_incoming_vehicles, axis=0))
    # # print(n_incoming_vehicles)
    # print("n_in", n_in)
    # print("Q_in", Q_in)
    # print(np.sum(n_wait[:, :, 0]) + np.sum(n_wait_charge_next[:, :, 0]))
    # print(np.sum(n_wait[:, :, 1]) + np.sum(n_wait_charge_next[:, :, 1]))
    # # #
    # print("n_pass", np.sum(n_pass, axis=0))
    # # # # print(n_in_wait_charge[1, 1, 0])
    # # print("n_in_charge", np.sum(n_in_charge, axis=0))
    # route = [39, 40, 41, 34, 27, 15, 12, 13, 14, 11, 1, 2, 3, 4, 5]
    # print("n_wait", np.where(n_wait > 1e-3), np.sum(n_wait))
    # print("E_charge1", np.where(E_charge1 > 0))
    # print("Q_out", np.where(Q_out > 0))

    print("n_arrived_vehicles", np.sum(n_arrived_vehicles))
    print("n_incoming_vehicles", np.sum(n_incoming_vehicles))
    # print("n_incoming_vehicles", np.sum(n_incoming_vehicles))
    # # print("Q_wait", Q_wait)
    # # # print("Q_in_wait", Q_in_wait)
    # # # print("Q_in_charge_wait", Q_in_charge_wait[:, 1, 0])
    # # # print("n_out", np.sum(n_out, axis=0))
    # # # # # print(Q_)
    # # # # # print(E_consumed_pass)
    # # # print("n_exit", np.sum(n_exit, axis=0))
    # # # # # print("n_exit", np.sum(n_exit, axis=0))
    # # # print("n_in", np.sum(n_in, axis=0))
    # # # print("Q_wait_charge_next", Q_wait_charge_next)
    # print("n_wait_charge_next", Q_wait_charge_next)
    # print("n_in_wait", n_in_wait)
    # # # print("n_exit_charge", np.sum(n_exit_charge, axis=0))
    # # # # print(Q_incoming_vehicles[0, 0, 0])
    # # # print("Q_in", Q_in)
    # # # # print("Q_exit_pass", Q_exit_pass)
    # # #
    # # # print("Q_pass", np.sum(Q_pass, axis=0))
    # print("Q_exit", np.sum(Q_exit, axis=0))
    # # # print("Q_exit", Q_exit)
    # # # print("Q_out", np.sum(Q_out, axis=0))
    # # # print("Q_out all", Q_out)
    # # print("Q_in", np.sum(Q_in, axis=0))
    # # # print("Q_exit_charged", Q_exit_charged)
    # # # print("n_finished_charge1", n_finished_charge1)
    # # print("Q_finished_charging", np.sum(Q_finished_charging, axis=0))
    # # print("Q_output_charge1", Q_output_charge1)
    # # print("Q_output_charge2", Q_output_charge2)
    # # print("Q_input_charge1", np.sum(Q_input_charge1, axis=0))
    # # print("n_output_charged1", np.sum(n_output_charged1, axis=0))
    # # print("Q_input_charge2", Q_input_charge2)
    # print("n_charge sum", np.sum(n_charge1[:, 1, :] + n_charge2[:, 1, :] + n_charge3[:, 1, :], axis=1))
    # print("n_charge1", n_charge1)
    # print("n_charge2", n_charge2[:, 1, :])
    # print("n_charge3", n_charge3[:, 1, :])
    #
    # # # print("n_charge3", n_charge3)
    # # print("E_charge1", np.sum(E_charge1, axis=0))
    # #
    # # # print("E_charge1", E_charge1)
    # # print("E_charge2", np.sum(E_charge2, axis=0))
    # # print("E_charge3", np.sum(E_charge3, axis=0))
    # # print("E_consumed_pass", np.sum(E_consumed_pass, axis=0))
    #print("E_consumed_charge_wait", np.sum(E_consumed_charge_wait, axis=0))
    # # print("E_consumed_exit_charge", np.sum(E_consumed_exit_charge, axis=0))
    # # # print("Q_finished_charge1", Q_finished_charge1)
    # # print("Q_finished_charge2", Q_finished_charge2)
    # # # print("Q_in_pass", Q_in_pass)
    #print(np.sum(Q_out[:, :, 1], axis=0))
    # # # print("n_in_charge", n_in_charge)
    # print("n_to_charge", np.sum(n_to_charge, axis=0))
    # # # print("n_charge1", np.sum(n_charge1, axis=0))
    # # # print("n_charge2", np.sum(n_charge2, axis=0))
    # # # # print("n_in_charge", n_in_charge)
    # print(fleet_df.SOC_init.to_list())
    # print("Q_incoming_vehicles", np.sum(Q_incoming_vehicles))
    # print(charging_model.fleet_departing_times[0])
    # print("n_arrived_vehicles", np.sum(n_arrived_vehicles))
    # print("Q_arrived_vehicles", np.sum(Q_arrived_vehicles))
    # print(np.where(n_charge3>1000))
    total_cons = np.sum(Q_pass) + np.sum(Q_finished_charging) - np.sum(Q_exit) + (np.sum(Q_in_charge_wait) - np.sum(Q_in_wait) - np.sum(Q_in_charge))
    print("Total energy consumed", total_cons )
    # print(n_arrived_vehicles[np.where(n_arrived_vehicles[:, :, 1] > 0)])
    # print(n_wait[np.where(n_wait[:, :, 1] > 0)])
    # print(n_charge1[np.where(n_charge1[:, :, 1] > 0)])
    #print(np.sum(E_charge1[np.where(E_charge1[:, :, 1] > 0)] + E_charge2[np.where(E_charge1[:, :, 1] > 0)] + E_charge3[np.where(E_charge1[:, :, 1] > 0)], axis=1))
    # print("model.cell_charging_cap[c] * 0.25", charging_model.cell_charging_cap[15] * 0.25, charging_model.cell_charging_cap[15])
    # # # # print("Q_input_charge1", Q_input_charge1)
    # # # print("Q_in_charge", Q_in_charge)
    # print("Q_wait_charge_next", Q_wait_charge_next)
    # print("n_wait_charge_next", n_wait_charge_next)
    # print(np.sum(n_charge1[:, :, 0], axis=0))
    # print(np.sum(n_charge2[:, :, 0], axis=0))
    # print(np.sum(n_charge3[:, :, 0], axis=0))
    # print("oi", np.sum(n_pass[:, :, 1], axis=0))
    # print(np.sum(Q_in_charge_wait[:, :, 1], axis=0))
    # print("n_in_wait_charge", n_in_wait_charge)
    # print("n_pass", n_pass)
    print(
        "Total E_charged",
        sum(sum(sum(E_charge1))) + sum(sum(sum(E_charge2))) + sum(sum(sum(E_charge3))),
    )
    print("check", np.sum(Q_arrived_vehicles) - np.sum(Q_incoming_vehicles),sum(sum(sum(E_charge1))) + sum(sum(sum(E_charge2))) + sum(sum(sum(E_charge3))) -total_cons )
    # print(
    #     "Total E_consumed",
    #     sum(sum(sum(E_consumed_pass)))
    #     + sum(sum(sum(E_consumed_charge_wait)))
    #     + sum(sum(sum(E_consumed_exit_charge))),
    # )

    # print(
    #     "Total E_consumed",
    #     sum(sum(E_consumed_pass[:, , :]))
    #     + sum(sum(E_consumed_charge_wait[:, [39, 40, 41, 34, 27], :]))
    #     + sum(sum(E_consumed_exit_charge[:, [39, 40, 41, 34, 27], :])),
    # )
    # print(
    #     "Total E_consumed",
    #     ((sum(E_consumed_pass[:, 5, 0])))
    #     + ((sum(E_consumed_charge_wait[:, 5, 0])))
    #     + ((sum(E_consumed_exit_charge[:, 5, 0]))),
    # )
    # print("n_exit_charge", n_exit_charge)
    # plot_results(charging_model, fleet_df)
    # # TODO: look at Qexit_pass and Qpass
