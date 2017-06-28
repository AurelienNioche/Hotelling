import numpy as np

from environment.environment import Environment


def main():

    seed = np.random.randint(1000)

    n_positions = 11
    n_prices = 11

    n_firms = 2

    firm_positions = np.random.randint(1, n_positions + 1, size=n_firms)
    firm_prices = np.random.randint(1, n_prices + 1, size=n_firms)

    transportation_cost = 0.2
    utility_consumption = 22

    firm_alpha = 0.01
    firm_temp = 0.02
    firm_momentum = 0.0  # Only NN
    firm_neural_network = "Elman"  # Only NN

    customer_alpha = 0.01
    customer_temp = 0.02
    customer_momentum = 0.0  # Only NN

    customer_neural_network = "Elman"  # Only NN

    t_max = 100

    # Choose the type of firm you want between:
    # - "FirmOriginal"
    # - "FirmBinary"
    # - "FirmUnary
    # - "FirmLinear
    # Type of firms differs only for the encoding of entries, but it can have a great impact
    # on efficiency of the network.
    firm = "FirmUnary"
    customer = "CustomerUnary"

    parameters = {
        "seed": seed,

        "firm": firm,
        "customer": customer,

        "n_positions": n_positions,
        "n_prices": n_prices,

        "firm_positions": firm_positions,  # Initial positions
        "firm_prices": firm_prices,  # Initial prices

        "transportation_cost": transportation_cost,
        "utility_consumption": utility_consumption,

        "firm_temp": firm_temp,
        "firm_alpha": firm_alpha,
        "firm_momentum": firm_momentum,
        "firm_neural_network": firm_neural_network,  # Useful for NN

        "customer_alpha": customer_alpha,
        "customer_temp": customer_temp,
        "customer_momentum": customer_momentum,
        "customer_neural_network": customer_neural_network,

        "t_max": t_max
    }

    env = Environment(**parameters)
    env.run(multi=False)

if __name__ == '__main__':
    main()
