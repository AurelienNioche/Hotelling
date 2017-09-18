from pylab import np, plt

import glob
from tqdm import tqdm
from os import path, makedirs
from time import time
from scipy.stats import linregress
from matplotlib import cm

from graph.graph import FigureProducer
from analysis.parameters import an_parameters
from analysis.data import Data, Stats, Parameters, Results, Variable


class StatsExtractor(object):

    t_max = an_parameters["t_max"]
    time_window = an_parameters["time_window"]

    working_folder = an_parameters["working_folder"]

    def __init__(self):

        self.data = Data()
        self.stats = Stats()

        self.folders = None

    def run(self, use_pickle_file=True, n_select=None):

        t = time()

        if self.stats.data is None or use_pickle_file is False:
            self.extract_data()
        else:
            print("Stats data loaded from pickle file in {} s.".format(time() - t))

        if n_select is not None:
            self.make_a_selection(n_select=n_select)

    def extract_data(self):

        print("Extract data...")

        self.stats.data = {}
        self.get_folders()

        for label in [
                "transportation_cost", "delta_position", "delta_price",
                "profits", "change_position", "change_price", "customer_extra_view_choices",
                "firm_temp", "firm_alpha", "customer_temp", "customer_alpha",
                "customer_utility", "utility_consumption", "idx"]:
            self.stats.data[label] = []

        for i, folder in tqdm(enumerate(self.folders)):

            parameters = Parameters(economy_folder=folder)
            results = Results(economy_folder=folder)

            if parameters.data is not None and results.data is not None and \
                    results.is_valid(time_window=self.time_window):

                for label in [
                        "utility_consumption", "transportation_cost",
                        "firm_alpha", "firm_temp",
                        "customer_alpha", "customer_temp"
                ]:

                    self.stats.data[label].append(
                        parameters.data[label]
                    )

                for results_label, stats_label in [
                    ("positions", "delta_position"), ("prices", "delta_price")
                ]:

                    self.stats.data[stats_label].append(
                        self.extract_delta(results.data[results_label][-self.time_window:])
                    )

                for label in [
                    "customer_extra_view_choices",
                    "profits",
                    "customer_utility"
                ]:
                    self.stats.data[label].append(
                        np.mean(results.data[label][-self.time_window:])
                    )

                for results_label, stats_label in [
                    ("positions", "change_position"), ("prices", "change_price")
                ]:

                    self.stats.data[stats_label].append(
                        self.extract_change(results.data[results_label][-self.time_window:])
                    )

                self.stats.data["idx"].append(i)

        self.stats.write()

        print("Analysis will be done with {} economies.".format(len(self.stats.data["idx"])))

    def make_a_selection(self, n_select):

        n_eco = len(self.stats.data["idx"])

        for key in self.stats.data.keys():
            self.stats.data[key] = np.asarray(self.stats.data[key])

        idx_selected = []
        for tr_cost in np.unique(self.stats.data["transportation_cost"]):
            idx_selected += \
                list(np.random.choice(np.arange(n_eco)[self.stats.data["transportation_cost"] == tr_cost],
                                      size=n_select, replace=False))

        idx_selected = np.asarray(idx_selected)

        for key in self.stats.data.keys():
            self.stats.data[key] = self.stats.data[key][idx_selected]

        print("Selection done.")

    @staticmethod
    def extract_delta(data):

        d = np.absolute(data[:, 0] - data[:, 1])
        return np.mean(d)

    @staticmethod
    def extract_change(data):

        move = []
        for firm in range(2):
            move.append(np.mean([abs(data[i] - data[i + 1]) for i in range(len(data) - 1)]))

        return np.mean(move)

    def extract_single_dimension(self, variable, t_max):

        print("Extracting variable '{}'.".format(variable.name))

        if self.folders is None:
            self.get_folders()

        # noinspection PyUnusedLocal
        data = [[] for i in range(t_max)]

        for i in tqdm(self.stats.data["idx"]):

            results = Results(economy_folder=self.folders[i])
            for t in range(t_max):
                data[t].append(
                    results.data[variable.name][t]
                )

        print("Converting in array and computing mean/std.")

        variable.data = dict()
        variable.data["mean"] = np.array([np.mean(data[t]) for t in range(t_max)])
        variable.data["std"] = np.array([np.std(data[t]) for t in range(t_max)])

        print("Writing in pickle.")
        variable.write()

        print("Done.")

    def get_folders(self):

        self.folders = glob.glob("{}/HC_*".format(self.working_folder))
        assert len(self.folders), "List of folders should not be empty!"


class FigureMaker:

    fig_folder = an_parameters["fig_folder"]

    display = an_parameters["display"]

    linear_regression = an_parameters["linear_regression"]

    scatter_vars = an_parameters["scatter_vars"]
    curve_vars = an_parameters["curve_vars"]

    range_var = an_parameters["range_var"]

    customer_firm_choices_period = an_parameters["customer_firm_choices_period"]
    firm_period = an_parameters["firm_period"]

    t_max = an_parameters["t_max"]

    use_french = an_parameters["use_french"]

    def __init__(self, stats_extractor):

        self.stats = stats_extractor.stats
        self.stats_extractor = stats_extractor

    def run(self):

        self.create_fig_folder()

        self.erase_stats_file()

        for var1, var2 in self.scatter_vars:

            self.hexbin_plot(var1=var1, var2=var2)
            self.histbin(var1=var1, var2=var2)
            self.scatter_plot(
                var1=var1, var2=var2, range_var=self.range_var, linear_regression=self.linear_regression, force=True)

        for var in self.curve_vars:
            self.curve_plot(variable=var, t_max=self.t_max)

        self.individual_plot()

    def erase_stats_file(self):

        with open("{}/stats.txt".format(self.fig_folder), "w"):
            pass

    # noinspection SpellCheckingInspection
    def histbin(self, var1, var2, force=False):

        fig_name = "{}/hist_median_{}_{}.pdf".format(self.fig_folder, var1, var2)
        if path.exists(fig_name) and not force:
            return

        if var1 == "customer_extra_view_choices" and var2 == "delta_position":

            x = np.asarray(self.stats.data[var1])
            y = np.asarray(self.stats.data[var2])

            n_bin = 5

            a = np.linspace(0, 10, n_bin + 1)

            b = np.zeros(n_bin)
            for i in range(n_bin):
                yo = list()
                for idx, xi in enumerate(x):
                    if a[i] <= xi < a[i+1]:
                        yo.append(y[idx])

                b[i] = np.median(yo) if len(yo) else 0

            # ----- #

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            plt.xlim(self.range_var[var1])
            plt.ylim(self.range_var[var2])

            plt.xlabel(self.format_label(var1))
            plt.ylabel(self.format_label(var2))

            ax.bar(a[:-1] + (a[1] - a[0])/2, b, a[1] - a[0], color='grey')

            plt.savefig(fig_name)

            # --- #

            if self.display:
                plt.show()

            plt.close()

            # ---- #

            b = np.zeros(n_bin)
            c = np.zeros(n_bin)
            for i in range(n_bin):
                yo = list()
                for idx, xi in enumerate(x):
                    if a[i] <= xi < a[i+1]:
                        yo.append(y[idx])

                b[i] = np.mean(yo) if len(yo) else 0
                c[i] = np.std(yo) if len(yo) else 0

            # ----- #

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            plt.xlim(self.range_var[var1])
            plt.ylim(self.range_var[var2])

            plt.xlabel(self.format_label(var1))
            plt.ylabel(self.format_label(var2))

            ax.bar(a[:-1] + (a[1] - a[0])/2, b, a[1] - a[0], color='grey', yerr=c)

            plt.savefig("{}/hist_mean_{}_{}.pdf".format(self.fig_folder, var1, var2))

            # --- #

            if self.display:
                plt.show()

            plt.close()

    def hexbin_plot(self, var1, var2, force=False):

        fig_name = "{}/hexbin_{}_{}.pdf".format(self.fig_folder, var1, var2)
        if path.exists(fig_name) and not force:
            return

        if var1 == "customer_extra_view_choices" and var2 == "delta_position":

            print("Doing hexbin plot '{}' against '{}'.".format(var2, var1))

            x = np.asarray(self.stats.data[var1])
            y = np.asarray(self.stats.data[var2])

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            plt.xlim(self.range_var[var1])
            plt.ylim(self.range_var[var2])

            plt.xlabel(self.format_label(var1))
            plt.ylabel(self.format_label(var2))

            hb = ax.hexbin(x=x, y=y, gridsize=20, cmap='inferno')

            ax.set_facecolor('black')

            cb = fig.colorbar(hb, ax=ax)
            cb.set_label('counts')

            plt.savefig(fig_name)

            if self.display:
                plt.show()

            plt.close()

    def individual_plot(self, force=False):

        sub_folder = "individual_plots"

        if path.exists("{}/{}".format(self.fig_folder, sub_folder)) and not force:
            return

        print("Doing individual plot.")

        if self.stats_extractor.folders is None:
            self.stats_extractor.get_folders()

        idx_to_plot = []
        example_type = []

        arg_min = np.argmin(self.stats.data["delta_position"])
        idx_min = self.stats.data["idx"][arg_min]

        arg_max = np.argmax(self.stats.data["delta_position"])
        idx_max = self.stats.data["idx"][arg_max]

        example_type.append("{}_differentiation".format("min"))
        idx_to_plot.append(idx_min)

        example_type.append("{}_differentiation".format("max"))
        idx_to_plot.append(idx_max)

        for idx, ex_type in zip(idx_to_plot, example_type):

            parameters = Parameters(economy_folder=self.stats_extractor.folders[idx])
            results = Results(economy_folder=self.stats_extractor.folders[idx])

            fp = FigureProducer(
                results=results.data,
                parameters=parameters.data,
                root_folder="{}/{}/{}_idx{}".format(self.fig_folder, sub_folder, ex_type, idx)
            )

            fp.plot_customer_firm_choices(period=self.customer_firm_choices_period)

            for firm in [0, 1]:
                fp.plot_profits(player=firm, period=self.firm_period)
                fp.plot_prices(player=firm, period=self.firm_period)
                fp.plot_positions(player=firm, period=self.firm_period)

            fp.write_parameters()

    def scatter_plot(self, var1, var2, range_var, linear_regression, force=False):

        fig_name = "{}/scatterplot_{}_{}.pdf".format(self.fig_folder, var1, var2)
        if path.exists(fig_name) and not force:
            return

        print("Doing scatter plot '{}' against '{}'.".format(var2, var1))

        x = np.asarray(self.stats.data[var1])
        y = np.asarray(self.stats.data[var2])

        plt.scatter(x=x, y=y, c=self.stats.data["transportation_cost"], s=10, cmap=cm.plasma)
        plt.xlim(range_var[var1])
        plt.ylim(range_var[var2])
        plt.xlabel(self.format_label(var1))
        plt.ylabel(self.format_label(var2))

        if linear_regression:
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            plt.plot(x, intercept + x * slope, c="black", lw=2)

            with open("{}/stats.txt".format(self.fig_folder), "a", encoding='utf-8') as f:

                to_write = "*****\n" + \
                    "{} against {}\n".format(self.format_label(var2), self.format_label(var1)) + \
                    "p value: {}\n".format(p_value) + \
                    "intercept: {}\n".format(intercept) + \
                    "slope: {}\n".format(slope) + \
                    "r value: {}\n".format(r_value) + \
                    "\n"

                f.write(to_write)

        plt.savefig(fig_name)

        if self.display:
            plt.show()

        plt.close()

    def curve_plot(self, variable, t_max, force=False):

        fig_name = "{}/curve_plot_{}.pdf".format(self.fig_folder, variable)
        if path.exists(fig_name) and not force:
            return

        print("Doing curve plot for variable '{}'.".format(variable))

        var = Variable(name=variable)

        if var.data is None:
            self.stats_extractor.extract_single_dimension(var, t_max=t_max)

        x = np.arange(t_max)

        mean = var.data["mean"]
        std = var.data["std"]

        plt.plot(x, mean, c='black', lw=2)
        plt.plot(x, mean + std, c='black', lw=.1)
        plt.plot(x, mean - std, c='black', lw=.1)
        plt.fill_between(x, mean + std, mean - std, color='black', alpha=.1)
        plt.xlabel("t")
        plt.ylabel(self.format_label(variable))
        plt.savefig(fig_name)

        if self.display:
            plt.show()

        plt.close()

    def format_label(self, label):

        if self.use_french:
            return an_parameters["french_mapping"][label]

        else:
            return label.replace("_", " ").capitalize()

    def create_fig_folder(self):

        if not path.exists(self.fig_folder):
            makedirs(self.fig_folder)


def main():

    try:
        stats_extractor = StatsExtractor()
        stats_extractor.run(n_select=None)

        figure_maker = FigureMaker(stats_extractor=stats_extractor)
        figure_maker.run()

    except Exception:
        raise Exception("Maybe there are errors coming from a bad configuration of the `analysis/parameters.py` file.")
