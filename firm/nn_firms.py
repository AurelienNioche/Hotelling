import numpy as np

from utils.utils import softmax
from firm.firm import Firm
from neural_network.perceptron import MLP


class NeuralNetworkFirm(Firm):

    """Abstract class"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.alpha = kwargs["alpha"]
        self.momentum = kwargs["momentum"]
        self.temp = kwargs["temp"]

        self.n_positions = kwargs["n_positions"]
        self.n_prices = kwargs["n_prices"]

        self.options = np.arange(self.n_positions * self.n_prices)

        # Size of input depend of the level of strategy
        self.network_input = np.zeros(self._get_network_input_size())

        # Output of network is expected profit; this will contain expected profit for each option
        self.network_outputs = np.zeros(self.options.size)

        self.network = self._create_network(kwargs["neural_network"])

        # Create a mapping between int and 'strategy', a 'strategy' composed here by
        # a particular position and a particular price
        self.strategies = self._get_strategies()

        self.set_up()

    def _get_strategies(self):

        st = {}
        i = 0
        for pos in range(1, self.n_positions + 1):
            for price in range(1, self.n_prices + 1):
                st[i] = {
                    "position": pos,
                    "price": price
                }
                i += 1

        return st

    def set_up(self):

        self.network.reset()

    def change_in_opponents_strategies(self, old_opponents_positions, old_opponents_prices):

        self._learn(
            opponents_positions=old_opponents_positions, opponents_prices=old_opponents_prices
        )

    def select_strategy(self, opponents_positions, opponents_prices):

        self._learn(
            opponents_positions=opponents_positions, opponents_prices=opponents_prices
        )

        self._get_network_outputs(
            opponents_positions=opponents_positions, opponents_prices=opponents_prices
        )

        p = softmax(self.network_outputs, temp=self.temp)

        st = np.random.choice(self.options, p=p)
        self.x = self.strategies[st]["position"]
        self.price = self.strategies[st]["price"]

    def _get_network_outputs(self, opponents_positions, opponents_prices):

        for i in self.options:
            self._set_network_input(
                x=self.strategies[i]["position"], price=self.strategies[i]["price"],
                opponents_positions=opponents_positions, opponents_prices=opponents_prices
            )
            self.network_outputs[i] = self.network.propagate_forward(self.network_input)

    def _learn(self, opponents_positions, opponents_prices):

        self._set_network_input(x=self.x, price=self.price,
                                opponents_prices=opponents_prices, opponents_positions=opponents_positions)
        self.network.propagate_forward(self.network_input)
        self.network.propagate_backward(target=self._u(), lrate=self.alpha,
                                        momentum=self.momentum)

    def _get_network_input_size(self):

        return 0

    def _set_network_input(self, x, price, opponents_positions, opponents_prices):

        pass

    def _create_network(self, model):

        if model == MLP:
            return MLP(self.network_input.size, self.network_input.size, 1)
        else:
            return model(self.network_input.size, 1)


class NonStrategicNeuralNetwork(NeuralNetworkFirm):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def _get_network_input_size(self):

        return self.n_positions + self.n_prices

    def _set_network_input(self, x, price, opponents_positions, opponents_prices):

        b_position = np.zeros(self.n_positions, dtype=int)
        b_price = np.zeros(self.n_prices, dtype=int)

        b_position[x-1] = 1
        b_price[price-1] = 1

        self.network_input[:self.n_positions] = b_position
        self.network_input[self.n_positions:] = b_price


class StrategicNeuralNetwork(NeuralNetworkFirm):

    def __init__(self, **kwargs):
        self.n_firms = kwargs["n_firms"]

        super().__init__(**kwargs)

        self.set_up()

    def _get_network_input_size(self):
        return self.n_firms*self.n_positions + self.n_firms*self.n_prices

    def _set_network_input(self, x, price, opponents_positions, opponents_prices):

        b_position = np.zeros(self.n_positions, dtype=int)
        b_price = np.zeros(self.n_prices, dtype=int)

        network_input = []

        b_position[x-1] = 1
        b_price[price-1] = 1

        network_input += list(b_position)
        network_input += list(b_price)

        for opp_pos, opp_price in zip(opponents_positions, opponents_prices):

            b_position[:] = 0
            b_price[:] = 0
            b_position[opp_pos-1] = 1
            b_price[opp_price-1] = 1

            network_input += list(b_position)
            network_input += list(b_price)

        self.network_input[:] = network_input
