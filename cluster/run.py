import pickle
import numpy as np
from multiprocessing import Pool

from cluster.parameters import cl_parameters
from environment.environment import Environment
from backup.backup import Backup


def run(args):

    param, clone_id = args[0],  args[1]

    param["customer_alpha"] = np.random.uniform(*param["range_customer_alpha"])
    param["customer_temp"] = np.random.uniform(*param["range_customer_temp"])

    param["firm_alpha"] = np.random.uniform(*param["range_firm_alpha"])
    param["firm_temp"] = np.random.uniform(*param["range_firm_temp"])

    param["transportation_cost"] = np.random.uniform(*param["range_transportation_cost"])
    param["utility_consumption"] = np.random.uniform(*param["range_utility_consumption"])

    param["firm_positions"] = np.random.randint(1, param["n_positions"] + 1, size=param["n_firms"])
    param["firm_prices"] = np.random.randint(1, param["n_prices"] + 1, size=param["n_firms"])

    param["seed"] = np.random.randint(2 ** 32)

    job_id = param["job_id"]

    label = "J{}C{}".format(job_id, clone_id)

    env = Environment(**param)
    results = env.run()

    Backup(data=results, name="results", root_folder=cl_parameters["working_folder"], label=label)
    Backup(data=param, name="parameters", root_folder=cl_parameters["working_folder"], label=label)


def get_param(idx):

    parameters_file = "{}/cluster_computing_parameters.p".format(cl_parameters["working_folder"])
    with open(parameters_file, "rb") as f:

        param = pickle.load(f)[idx]

    return param


def main(str_idx):

    param = get_param(int(str_idx))

    n_cpu = cl_parameters["n_cpu"]
    n_clones = cl_parameters["n_clones"]

    pool = Pool(processes=n_cpu)
    pool.map(func=run, iterable=[[param, i] for i in range(n_clones)])
