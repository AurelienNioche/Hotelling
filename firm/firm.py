class Firm(object):

    """Abstract class for firm"""

    def __init__(self, **kwargs):

        # Position and price
        self.x = kwargs["x"]
        self.price = kwargs["price"]

        # Max profit used for computing utility
        self.max_profit = kwargs["profit_max"]

        self.profit = 0

    def sell_one_unit(self):
        self.profit += self.price

    def reset_profit_counter(self):
        self.profit = 0

    def _u(self):

        # # According to Barreda, should be:
        # return self.profit ** self.r
        # But it is better if all values lies between 0 and 1 for applying RL
        return self.profit / self.max_profit

    def change_in_opponents_strategies(self, old_opponents_positions, old_opponents_prices):
        pass

    def select_strategy(self, opponents_positions, opponents_prices):
        pass
