import numpy as np
from utils.utils import softmax, normalize

from neural_network.perceptron import MLP


class Customer(object):

    def __init__(self, **kwargs):

        self.x = kwargs["x"]
        self.t_cost = kwargs["t_cost"]
        self.u_consumption = kwargs["utility_consumption"]

        self.n_positions = kwargs["n_positions"]
        self.n_prices = kwargs["n_prices"]

        # Learning parameters
        self.alpha = kwargs["alpha"]
        self.temp = kwargs["temp"]

        self.firm_choice = None
        self.extra_view = None
        self.utility = None

        self.network_target = None

        self.extra_view_possibilities = np.arange(0, self.n_positions, 2)
        self.extra_view_values = np.zeros(len(self.extra_view_possibilities))

        # Min and max of utility for normalizing
        self.utility_max = self.u_consumption - 1  # 1 is minimum price
        self.utility_min = - max(self.extra_view_possibilities) * self.t_cost

        # Neural network...
        self.network_input = np.zeros(self._get_network_input_size())
        self.network = self._create_network(kwargs["neural_network"])
        self.momentum = kwargs["momentum"]

        self.cv_extra_view = self._create_converter(len(self.extra_view_possibilities))

        self._set_up()

    def get_field_of_view(self):

        extra = self._choose_extra_view()
        field_of_view = [self.x - (extra // 2), self.x + (extra // 2)]

        return field_of_view

    def get_firm_choice(self, firms_idx, prices):
        """choice function"""

        consume = len(prices) > 0

        if consume:
            price = min(prices)  # Choose minimum price
            self.firm_choice = np.random.choice(firms_idx[prices == price])

        else:
            price = 0
            self.firm_choice = -1

        exploration_cost = self.t_cost * self.extra_view
        self.utility = int(consume) * self.u_consumption - (exploration_cost + price)

        self.network_target = normalize(self.utility, min_=self.utility_min, max_=self.utility_max)

        self._learn()

        return self.firm_choice

    def _choose_extra_view(self):

        p = softmax(self.extra_view_values, temp=self.temp)
        self.extra_view = np.random.choice(self.extra_view_possibilities, p=p)

        return self.extra_view

    def _set_up(self):

        self.network.reset()
        self._get_network_outputs()

    def _create_network(self, model):

        if model == MLP:
            return MLP(self.network_input.size, self.network_input.size, 1)
        else:
            return model(self.network_input.size, 1)

    def _learn(self):

        # Set input
        i = np.where(self.extra_view_possibilities == self.extra_view)[0][0]
        self._set_network_input(i)

        # Propagation stuff
        self.network.propagate_forward(self.network_input)

        self.network.propagate_backward(self.network_target, lrate=self.alpha, momentum=self.momentum)

        # Get new positions values
        self._get_network_outputs()

    def _get_network_outputs(self):

        for i in range(len(self.extra_view_possibilities)):
            self._set_network_input(i)
            self.extra_view_values[i] = self.network.propagate_forward(self.network_input)

    def _get_network_input_size(self):

        raise Exception("'Customer' is an abstract class. You should implement one of its child.")

    @staticmethod
    def _create_converter(n):

        raise Exception("'Customer' is an abstract class. You should implement one of its child.")

    def _set_network_input(self, i):

        self.network_input[:] = self.cv_extra_view[i]


class CustomerBinary(Customer):

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

        return len(format(len(self.extra_view_possibilities-1), "b"))

    @staticmethod
    def _create_converter(n):

        cv = {"0": -1, "1": 1}
        len_str = len(format(n - 1, 'b'))
        return [[cv[j] for j in format(i, '0{}b'.format(len_str))] for i in range(n)]


class CustomerOriginal(Customer):

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

        return len(self.extra_view_possibilities)

    @staticmethod
    def _create_converter(n):

        out = []
        for i in range(n):
            a = np.zeros(n)
            a[:] = - 1
            a[i] = 1
            out.append(list(a))

        return out


class CustomerUnary(Customer):

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

        return len(self.extra_view_possibilities)

    @staticmethod
    def _create_converter(n):
        out = []
        for i in range(n):
            a = np.zeros(n)
            a[:] = - 1
            a[:i] = 1
            out.append(list(a))

        return out


class CustomerLinear(Customer):

    """
    Use unary encoding for entries.
    For instance, for 10 possibilities:
    [ 0.0 ]
    [ 0.11 ]
    [ 0.22 ]
    [ 0.33 ]
    [ 0.44 ]
    [ 0.55 ]
    [ 0.66 ]
    [ 0.77 ]
    [ 0.88 ]
    [ 1.0 ]
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def _get_network_input_size(self):

        return 1

    @staticmethod
    def _create_converter(n):

        a = np.arange(n, dtype=float)
        a[:] = (a-min(a)) / (max(a) - min(a))  # Normalize between 0 an 1
        a[:] -= 0.5  # Center around 0
        return [[i] for i in a]
