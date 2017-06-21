import pickle
from os import path, makedirs

from cluster.parameters import cl_parameters


def prepare():

    n_positions = 11
    n_prices = 11

    n_firms = 2

    firm_momentum = 0.0  # Only NN
    firm_neural_network = "MLP"  # Only NN

    customer_momentum = 0.0  # Only NN
    customer_neural_network = "MLP"  # Only NN

    t_max = 5000

    firm = "StrategicNeuralNetwork"
    customer = "Customer"

    range_transportation_cost = [0, 1]
    range_firm_alpha = [0.01, 0.10]
    range_firm_temp = [0.01, 0.03]
    range_customer_alpha = [0.01, 0.10]
    range_customer_temp = [0.01, 0.03]
    range_utility_consumption = [n_prices+1, (n_prices*2) + 1]

    parameters_list = []

    for i in range(cl_parameters["n_jobs"]):

        param = {
            "job_id": i,

            "n_firms": n_firms,

            "firm": firm,
            "customer": customer,

            "n_positions": n_positions,
            "n_prices": n_prices,

            "range_transportation_cost": range_transportation_cost,
            "range_utility_consumption": range_utility_consumption,

            "range_firm_temp": range_firm_temp,
            "range_firm_alpha": range_firm_alpha,

            "range_customer_alpha": range_customer_alpha,
            "range_customer_temp": range_customer_temp,

            "firm_momentum": firm_momentum,
            "firm_neural_network": firm_neural_network,  # Useful for NN

            "customer_momentum": customer_momentum,
            "customer_neural_network": customer_neural_network,

            "t_max": t_max
        }

        i += 1

        parameters_list.append(param)

    if not path.exists(cl_parameters["working_folder"]):
        makedirs(cl_parameters["working_folder"])

    parameters_file = "{}/cluster_computing_parameters.p".format(cl_parameters["working_folder"])
    with open(parameters_file, "wb") as f:

        pickle.dump(parameters_list, f)
