import numpy as np
from tqdm import tqdm

from firm.firm import Firm
from customer.customer import Customer
from firm.nn_firms import StrategicNeuralNetwork, NonStrategicNeuralNetwork
from neural_network.elman import Elman
from neural_network.perceptron import MLP


class Environment(object):
    """Hotelling, 1929"""

    def __init__(self, **param):

        self.param = param

        self.firms = []
        self.customers = []
        self.t_max = param["t_max"]

        self.active_player = 0

        self.n_firms = len(param["firm_positions"])

    def set_up(self):

        self._spawn_firms()
        self._spawn_customers()

    def _spawn_firms(self):

        firm_model = eval(self.param["firm"])
        nn_model = eval(self.param["firm_neural_network"])

        for position, price in zip(
                self.param["firm_positions"],
                self.param["firm_prices"]):

            firm = firm_model(
                x=position,
                price=price,
                neural_network=nn_model,

                alpha=self.param["firm_alpha"],
                temp=self.param["firm_temp"],
                momentum=self.param["firm_momentum"],

                n_prices=self.param["n_prices"],
                n_positions=self.param["n_positions"],
                n_firms=self.n_firms,

                profit_max=
                self.param["n_positions"] *
                self.param["n_prices"]
                )
            self.firms.append(firm)

    def _spawn_customers(self):

        customer_model = eval(self.param["customer"])
        nn_model = eval(self.param["customer_neural_network"])

        for i in range(self.param["n_positions"]):
            self.customers.append(
                customer_model(
                    x=i + 1,
                    neural_network=nn_model,

                    t_cost=self.param["transportation_cost"],
                    utility_consumption=self.param["utility_consumption"],

                    n_positions=self.param["n_positions"],
                    n_prices=self.param["n_prices"],

                    alpha=self.param["customer_alpha"],
                    temp=self.param["customer_temp"],
                    momentum=self.param["customer_momentum"]
                )
            )

    def time_step(self):

        prices = np.zeros(self.n_firms, dtype=int)
        positions = np.zeros(self.n_firms, dtype=int)

        prices[:] = self.get_prices()
        positions[:] = self.get_positions()

        firms_idx = np.arange(self.n_firms)

        for c in self.customers:
            field_of_view = c.get_field_of_view()

            cond0 = positions >= field_of_view[0]
            cond1 = positions <= field_of_view[1]

            firms_idx_c = firms_idx[
                cond0 * cond1
            ]

            choice = c.get_firm_choice(
                firms_idx=firms_idx_c, prices=prices[firms_idx_c])

            if choice != -1:
                self.firms[choice].sell_one_unit()

    def _reset_profits(self):

        for f in self.firms:
            f.reset_profit_counter()

    def get_profits(self):

        return [f.profit for f in self.firms]

    def get_positions(self):

        return [f.x for f in self.firms]

    def get_prices(self):

        return [f.price for f in self.firms]

    def get_customer_firm_choices(self):

        return [c.firm_choice for c in self.customers]

    def get_customer_extra_view_choices(self):

        return [c.extra_view for c in self.customers]

    def get_customer_utility(self):

        return [c.utility for c in self.customers]

    def end_time_step(self):

        positions = np.array([self.firms[i].x for i in range(self.n_firms)])
        prices = np.array([self.firms[i].price for i in range(self.n_firms)])

        idx = np.arange(self.n_firms)

        bool_opponents_active = idx[:] != self.active_player

        self.firms[self.active_player].select_strategy(
            opponents_positions=positions[bool_opponents_active],
            opponents_prices=prices[bool_opponents_active]
        )

        for i in idx[bool_opponents_active]:
            bool_opponents_i = idx[:] != i

            self.firms[i].change_in_opponents_strategies(
                old_opponents_positions=positions[bool_opponents_i],
                old_opponents_prices=prices[bool_opponents_i]
            )

        self._reset_profits()

        self.active_player = (self.active_player + 1) % self.n_firms

    def run(self):

        np.random.seed(self.param["seed"])

        # Set the environment
        self.set_up()

        n_firms = len(self.param["firm_positions"])
        n_customers = self.param["n_positions"]

        # Containers for stats
        positions = np.zeros((self.t_max, n_firms), dtype=int)
        prices = np.zeros((self.t_max, n_firms), dtype=int)
        profits = np.zeros((self.t_max, n_firms), dtype=int)
        customer_firm_choices = np.zeros((self.t_max, n_customers), dtype=int)
        customer_extra_view_choices = np.zeros((self.t_max, n_customers), dtype=int)
        customer_utility = np.zeros((self.t_max, n_customers))

        for t in range(self.t_max):

            # New time step
            self.time_step()

            # Stats
            positions[t] = self.get_positions()
            prices[t] = self.get_prices()
            profits[t] = self.get_profits()
            customer_firm_choices[t] = self.get_customer_firm_choices()
            customer_extra_view_choices[t] = self.get_customer_extra_view_choices()
            customer_utility[t] = self.get_customer_utility()

            # End turn
            self.end_time_step()

        results = {
            "positions": positions[:], "prices": prices[:], "profits": profits[:],
            "customer_firm_choices": customer_firm_choices[:],
            "customer_extra_view_choices": customer_extra_view_choices[:],
            "customer_utility": customer_utility[:]
        }
        return results

