import pickle
import numpy as np
from multiprocessing import Pool

from cluster.parameters import cl_parameters
from environment.environment import Environment
from backup.backup import Backup


def run(args):

    param, job_id = args[0], args[1]

    np.random.seed()

    #  In order to reuse the state -> np.random.set_state(state)
    np_random_generator_state = np.random.get_state()

    param["np_random_generator_state"] = np_random_generator_state

    idx = np.random.choice(len(param["range_transportation_cost"])) 

    param["transportation_cost"] = param["range_transportation_cost"][idx]

    extra_view = get_extra_view(mean=param["range_customer_extra_view"][idx]["mean"], 
                                std=1,
                                v_min=param["range_customer_extra_view"][idx]["v_min"], 
                                v_max=param["range_customer_extra_view"][idx]["v_max"])

    param["customer_extra_view"] = extra_view

    param["customer_alpha"] = np.random.uniform(*param["range_customer_alpha"])
    param["customer_temp"] = np.random.uniform(*param["range_customer_temp"])

    param["firm_alpha"] = np.random.uniform(*param["range_firm_alpha"])
    param["firm_temp"] = np.random.uniform(*param["range_firm_temp"])


    # param["utility_consumption"] = np.random.uniform(*param["range_utility_consumption"])

    param["firm_positions"] = np.random.randint(1, param["n_positions"] + 1, size=param["n_firms"])
    param["firm_prices"] = np.random.randint(1, param["n_prices"] + 1, size=param["n_firms"])

    machine_id = param["machine_id"]

    label = "M{}J{}".format(machine_id, job_id)

    env = Environment(**param)
    results = env.run()

    Backup(data=results, name="results", root_folder=cl_parameters["working_folder"], label=label)
    Backup(data=param, name="parameters", root_folder=cl_parameters["working_folder"], label=label)


def get_extra_view(mean, std, v_min, v_max):

    dist = np.random.normal(mean, std, 1000)
    rounded_dist = np.round(dist)
    final_dist = np.concatenate((rounded_dist[rounded_dist > v_min], 
                                 rounded_dist[rounded_dist < v_max]))

    return np.random.choice(final_dist)

def get_param(idx):

    parameters_file = "{}/cluster_computing_parameters.p".format(cl_parameters["working_folder"])
    with open(parameters_file, "rb") as f:

        param = pickle.load(f)[idx]

    return param


def main(str_idx):

    param = get_param(int(str_idx))

    n_cpu = cl_parameters["n_cpu"]
    n_jobs = cl_parameters["n_jobs"]

    pool = Pool(processes=n_cpu)
    pool.map(func=run, iterable=[[param, i] for i in range(n_jobs)])
