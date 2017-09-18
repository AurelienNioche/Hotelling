import pickle

from cluster.parameters import cl_parameters


def get_param():

    parameters_file = "{}/cluster_computing_parameters.p".format(cl_parameters["working_folder"])
    with open(parameters_file, "rb") as f:

        param = pickle.load(f)

    return param


def main():

    n_args = len(get_param())
    print(n_args)
    return n_args


if __name__ == "__main__":

    main()
