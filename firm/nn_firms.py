import numpy as np

from utils.utils import softmax
from firm.firm import Firm
from neural_network.perceptron import MLP


class NeuralNetworkFirm(Firm):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.alpha = kwargs["alpha"]
        self.momentum = kwargs["momentum"]
        self.temp = kwargs["temp"]

        self.n_positions = kwargs["n_positions"]
        self.n_prices = kwargs["n_prices"]
        self.n_firms = kwargs["n_firms"]

        self.options = np.arange(self.n_positions * self.n_prices)

        self.input_size = self._get_network_input_size()

        # Size of input depend of the level of strategy
        self.network_input = None

        # Output of network is expected profit; this will contain expected profit for each option
        self.network_outputs = np.zeros(self.options.size)

        self.network = self._create_network(kwargs["neural_network"])

        # Create a mapping between int and 'strategy', a 'strategy' composed here by
        # a particular position and a particular price
        self.strategies = self._get_strategies()

        self.cv_position = self._create_converter(self.n_positions)
        self.cv_price = self._create_converter(self.n_prices)

        self.encoded_opponents_strategies = None

        self._set_up()

    def change_in_opponents_strategies(self, old_opponents_positions, old_opponents_prices):

        self._encode_opponents_strategies(opponents_positions=old_opponents_positions,
                                          opponents_prices=old_opponents_prices)

        self._learn()

    def select_strategy(self, opponents_positions, opponents_prices):

        self._encode_opponents_strategies(opponents_positions=opponents_positions,
                                          opponents_prices=opponents_prices)

        self._learn()

        self._get_network_outputs()

        p = softmax(self.network_outputs, temp=self.temp)

        st = np.random.choice(self.options, p=p)
        self.x = self.strategies[st]["position"]
        self.price = self.strategies[st]["price"]

    def _encode_opponents_strategies(self, opponents_positions, opponents_prices):

        self.encoded_opponents_strategies = []
        for opp_pos, opp_price in zip(opponents_positions, opponents_prices):
            self.encoded_opponents_strategies += self.cv_position[opp_pos - 1] + self.cv_price[opp_price - 1]

    def _create_network(self, model):

        if model == MLP:
            return MLP(self.input_size, self.input_size, 1)
        else:
            return model(self.input_size, 1)

    def _get_strategies(self):

        return {
            i * self.n_prices + j: {'position': pos, 'price': price}
            for (i, pos) in enumerate(range(1, self.n_positions + 1))
            for (j, price) in enumerate(range(1, self.n_prices + 1))
        }

    def _set_up(self):

        self.network.reset()

    def _get_network_outputs(self):

        for i in self.options:
            self._set_network_input(
                x=self.strategies[i]["position"], price=self.strategies[i]["price"])

            self.network_outputs[i] = self.network.propagate_forward(self.network_input)

    def _learn(self):

        self._set_network_input(x=self.x, price=self.price)
        self.network.propagate_forward(self.network_input)
        self.network.propagate_backward(target=self._u(), lrate=self.alpha,
                                        momentum=self.momentum)

    def _get_network_input_size(self):

        raise Exception("'NeuralNetworkFirm' is an abstract class. You should implement one of its child.")

    @staticmethod
    def _create_converter(n):

        raise Exception("'NeuralNetworkFirm' is an abstract class. You should implement one of its child.")

    def _set_network_input(self, x, price):

        self.network_input = \
            self.cv_position[x - 1] + \
            self.cv_price[price - 1] + \
            self.encoded_opponents_strategies


class FirmBinary(NeuralNetworkFirm):

    """
    Use binary encoding for entries. Note that it is requires less entries than the number of possibilities
    (number of entries for n possibilities is approx log2(n)).
    For example, for 10 possibilities:
    [ 0.  0.  0.  0.]
    [ 0.  0.  0.  1.]
    [ 0.  0.  1.  0.]
    [ 0.  0.  1.  1.]
    [ 0.  1.  0.  0.]
    [ 0.  1.  0.  1.]
    [ 0.  1.  1.  0.]
    [ 0.  1.  1.  1.]
    [ 1.  0.  0.  0.]
    [ 1.  0.  0.  1.]

    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def _get_network_input_size(self):

        return self.n_firms * (len(format(self.n_positions, "b")) + len(format(self.n_prices, "b")))

    @staticmethod
    def _create_converter(n):

        cv = {"0": -1, "1": 1}
        len_str = len(format(n - 1, 'b'))
        return [[cv[j] for j in format(i, '0{}b'.format(len_str))] for i in range(n)]


class FirmOriginal(NeuralNetworkFirm):

    """
    First attempt for encoding.
    For example, for 10 possibilities:
    [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
    [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]
    [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
    [ 0.  0.  0.  0.  0.  0.  1.  0.  0.  0.]
    [ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  1.  0.]
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1.]

    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def _get_network_input_size(self):

        return self.n_firms * (self.n_positions + self.n_prices)

    @staticmethod
    def _create_converter(n):

        out = []
        for i in range(n):
            a = np.zeros(n)
            a[:] = - 1
            a[i] = 1
            out.append(list(a))

        return out


class FirmUnary(NeuralNetworkFirm):

    """
    Use unary encoding for entries.
    For instance, for 10 possibilities:
    [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
    [ 1.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
    [ 1.  1.  1.  0.  0.  0.  0.  0.  0.  0.]
    [ 1.  1.  1.  1.  0.  0.  0.  0.  0.  0.]
    [ 1.  1.  1.  1.  1.  0.  0.  0.  0.  0.]
    [ 1.  1.  1.  1.  1.  1.  0.  0.  0.  0.]
    [ 1.  1.  1.  1.  1.  1.  1.  0.  0.  0.]
    [ 1.  1.  1.  1.  1.  1.  1.  1.  0.  0.]
    [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  0.]

    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def _get_network_input_size(self):

        return self.n_firms * (self.n_positions + self.n_prices)

    @staticmethod
    def _create_converter(n):
        out = []
        for i in range(n):
            a = np.zeros(n)
            a[:] = - 1
            a[:i] = 1
            out.append(list(a))

        return out


class FirmLinear(NeuralNetworkFirm):

    """
    Use unary encoding for entries.
    For instance, for 10 possibilities:
    [ 0.0 ]
    [ 0.111111111111 ]
    [ 0.222222222222 ]
    [ 0.333333333333 ]
    [ 0.444444444444 ]
    [ 0.555555555556 ]
    [ 0.666666666667 ]
    [ 0.777777777778 ]
    [ 0.888888888889 ]
    [ 1.0 ]
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def _get_network_input_size(self):

        return self.n_firms * 2  # 2: position and price

    @staticmethod
    def _create_converter(n):

        a = np.arange(n, dtype=float)
        a[:] = (a-min(a)) / (max(a) - min(a))  # Normalize between 0 an 1
        a[:] -= 0.5  # Center around 0
        return [[i] for i in a]
