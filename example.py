import numpy as np
from os import path

from environment.environment import Environment
from graph.graph import FigureProducer


def main():

    seed = np.random.randint(1000)

    n_positions = 11
    n_prices = 11

    n_firms = 2

    firms_positions = np.random.randint(1, n_positions + 1, size=n_firms)
    firms_prices = np.random.randint(1, n_prices + 1, size=n_firms)

    transportation_cost = 0.2

    firm_alpha = 0.01
    firm_temp = 0.02
    firm_momentum = 0.0  # Only NN
    firm_neural_network = "MLP"  # Only NN

    customer_alpha = 0.01
    customer_temp = 0.02
    customer_momentum = 0.0  # Only NN

    customer_neural_network = "MLP"  # Only NN

    t_max = 10**3

    firm = "StrategicNeuralNetwork"
    customer = "Customer"

    parameters = {
        "seed": seed,

        "firm": firm,
        "customer": customer,

        "n_positions": n_positions,
        "n_prices": n_prices,

        "firms_positions": firms_positions,  # Initial positions
        "firms_prices": firms_prices,  # Initial prices

        "transportation_cost": transportation_cost,

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
    results = env.run()

    fig_producer = FigureProducer(results=results, parameters=parameters,
                                  root_folder=path.expanduser("~/Desktop/HotellingExample"))
    fig_producer.run(customers_choices_plot_period=50, other_plots_period=10000)


if __name__ == '__main__':
    main()
