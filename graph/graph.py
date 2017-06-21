from pylab import plt, np
from os import path, makedirs
import warnings
from matplotlib import cm, gridspec
from matplotlib.colors import LinearSegmentedColormap, NoNorm


class FigureProducer(object):

    def __init__(self, results, parameters, root_folder=path.expanduser("~/Desktop/HotellingFig")):

        self.results = results
        self.parameters = parameters
        self.fig_folder = root_folder

        self.create_fig_folder()

    def create_fig_folder(self):

        if not path.exists(self.fig_folder):
            makedirs(self.fig_folder)

    def format_fig_name(self, title):

        return "{}/{}.pdf".format(self.fig_folder, title)

    def plot_positions(self, player, period):

        positions = self.results["positions"][-period:]

        plt.title("Positions")
        plt.plot(positions[:, player], "o", markersize=0.2, color="black")
        plt.ylim(0.9, self.parameters["n_positions"] + 0.1)

        # plt.annotate(self.string_parameters, xy=(-0.05, -0.1), xycoords='axes fraction', fontsize=8)

        plt.savefig(self.format_fig_name("positions_player{}".format(player)))
        plt.close()

    def plot_prices(self, player, period):

        prices = self.results["prices"][-period:]

        # Plot prices
        plt.title("Prices")
        plt.plot(prices[:, player], "o", markersize=0.2, color="black")
        plt.ylim(0.9, self.parameters["n_prices"] + 0.1)

        # plt.annotate(self.string_parameters, xy=(-0.05, -0.1), xycoords='axes fraction', fontsize=8)

        plt.savefig(self.format_fig_name("prices_player{}".format(player)))
        plt.close()

    def plot_profits(self, player, period):

        profits = self.results["profits"][-period:]

        plt.title("Profits")
        time_window = 100
        x = np.arange(len(profits[:, player]))
        y = []
        for i in x:
            if i < time_window:
                y_value = np.mean(profits[:i + 1, player])

            else:
                y_value = np.mean(profits[i - time_window: i + 1, player])

            y.append(y_value)

        plt.plot(x, y, color="black")
        maximum_profit = \
            self.parameters["n_positions"] * \
            self.parameters["n_prices"]
        plt.ylim(0, maximum_profit)

        plt.annotate("Time window: {}".format(time_window), xy=(0.8, 0.1), xycoords='axes fraction', fontsize=6)

        # plt.annotate(self.string_parameters, xy=(-0.05, -0.1), xycoords='axes fraction', fontsize=6)

        plt.savefig(self.format_fig_name("profits_player{}".format(player)))
        plt.close()

    def plot_customer_firm_choices(self, period=50):

        # Data
        positions = self.results["positions"][-period:]
        prices = self.results["prices"][-period:]
        n_firms = len(self.results["positions"][0])
        customer_firm_choices = self.results["customer_firm_choices"][-period:]

        t_max, n_positions = customer_firm_choices.shape

        # Create fig and axes
        fig = plt.figure(figsize=(t_max, n_positions))
        gs = gridspec.GridSpec(24, 20)
        ax = fig.add_subplot(gs[:20, :20])
        ax2 = fig.add_subplot(gs[-1, 8:12])

        # Prepare normalization for 'imshow'
        mapping = dict([(x, y) for x, y in zip(np.arange(-1, n_firms), np.linspace(0, 1, n_firms + 1))])
        f_mapping = lambda x: mapping[x]

        # Function adapted for numpy array
        v_func = np.vectorize(f_mapping)

        # Format customer choices (reordering + normalization)
        formatted_customer_firm_choices = v_func(customer_firm_choices.T[::-1])

        # Colors for different firms
        firm_colors = cm.ScalarMappable(norm=None, cmap="gist_rainbow").to_rgba(np.linspace(0, 1, n_firms))

        # Prepare customized colormap
        colors = np.zeros((n_firms+1, 4))
        colors[0] = 1, 1, 1, 1  # White
        colors[1:] = firm_colors

        cmap_name = "manual_colormap"
        n_bins = n_firms + 1

        manual_cm = LinearSegmentedColormap.from_list(
            cmap_name, colors, N=n_bins)

        # Plot customer choices
        ax.imshow(formatted_customer_firm_choices, interpolation='nearest', origin="lower",
                  norm=NoNorm(), alpha=0.5, cmap=manual_cm)

        # Offsets for positions plot and prices plot
        offsets = np.linspace(-0.25, 0.25, n_firms)

        # Plot positions
        for i in range(n_firms):

            ax.plot(np.arange(t_max) + offsets[i],
                    n_positions - positions[:, i], "o", color=firm_colors[i], markersize=10)

        # Plot prices
        for t in range(t_max):

            for i in range(n_firms):

                ax.text(t + offsets[i] - 0.1,
                        n_positions - positions[t, i] + 0.2, prices[t, i])

        # Customize axes
        ax.set_xlim(-0.5, t_max - 0.5)
        ax.set_ylim(-0.5, n_positions - 0.5)

        # Add grid (requires to customize axes)
        ax.set_yticks(np.arange(0.5, n_positions - 0.5, 1), minor=True)
        ax.set_xticks(np.arange(0.5, t_max - 0.5, 1), minor=True)

        ax.grid(which='minor', axis='y', linewidth=2, linestyle=':', color='0.75')
        ax.grid(which='minor', axis='x', linewidth=2, linestyle='-', color='0.25')

        # After positioning grid, replace ticks for placing labels
        ax.set_xticks(range(t_max))
        ax.set_yticks(range(n_positions))

        # Top is position 1.
        ax.set_yticklabels(np.arange(1, n_positions + 1)[::-1])

        # Set axes labels
        ax.set_xlabel('t', fontsize=14)
        ax.set_ylabel('Position', fontsize=14)

        # Remove ticks
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')

        # Legend
        possibilities = v_func(np.arange(-1, n_firms))
        ax2.imshow(np.atleast_2d(possibilities),
                   interpolation='nearest', origin="lower", norm=NoNorm(), cmap=manual_cm)

        # Customize ticks
        ax2.xaxis.set_ticks_position('none')
        ax2.yaxis.set_ticks_position('none')
        ax2.set_yticks([])
        lab = [str(i) for i in np.arange(-1, n_firms)]
        ax2.set_xticks(np.arange(n_firms+1))
        ax2.set_xticklabels(lab)

        plt.savefig(self.format_fig_name("customers_choices"))
        plt.close()

    @staticmethod
    def str_dic(dic_like, args_before_break=4):

        out = ""
        i = 1
        for key, value in sorted(dic_like.items()):

            out += "{}: {}, ".format(key.replace("_", " "), value)
            if i % args_before_break == 0:
                out += "\n"
            i += 1
        out = out[:-2]
        return out

    def write_parameters(self):

        string_parameters = self.str_dic(self.parameters)
        with open("{}/parameters.txt".format(self.fig_folder), "w") as f:
            f.write(string_parameters)

    def run(self, customers_choices_plot_period, other_plots_period):

        if customers_choices_plot_period > 50:
            warnings.warn("'Customer choices plot' has not been designed to cover a period exceeding 50 time steps.")

        self.plot_customer_firm_choices(period=customers_choices_plot_period)

        n_firms = len(self.parameters["firms_positions"])
        for i in range(n_firms):
            self.plot_positions(player=i, period=other_plots_period)
            self.plot_prices(player=i, period=other_plots_period)
            self.plot_profits(player=i, period=other_plots_period)

        print("Figures have been saved at '{}'.".format(self.fig_folder))


def main():

    np.random.seed(1)

    t_max = 50
    n = 4
    n_positions = 7

    fp = FigureProducer(
        results={
            "positions": np.random.randint(1, n_positions+1, size=(t_max, n)),
            "prices": np.random.randint(1, 10, size=(t_max, n)),
            "customer_firm_choices": np.random.randint(0, n, size=(t_max, n_positions))
        },
        parameters={}
    )

    fp.plot_customer_firm_choices()


if __name__ == "__main__":

    main()
