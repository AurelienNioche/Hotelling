from pylab import np, plt
import pickle
import glob
from tqdm import tqdm
from os import path, makedirs
from time import time
from scipy.stats import linregress

from graph.graph import FigureProducer
from analysis.parameters import an_parameters


class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("Writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("Writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("Done.", flush=True)
            idx += batch_size


class Data(object):

    def __init__(self):

        self.working_folder = an_parameters["working_folder"]
        self.data = None
        self.pickle_file = None

    def load(self):

        with open(self.pickle_file, "rb") as f:
            self.data = pickle.load(MacOSFile(f))

    def write(self):

        with open(self.pickle_file, "wb") as f:
            pickle.dump(self.data, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


class Stats(Data):

    def __init__(self):
        super().__init__()
        self.pickle_file = "{}/stats.p".format(self.working_folder)

        if path.exists(self.pickle_file):
            self.load()


class SingleEcoData(Data):

    def __init__(self, data_type, economy_folder):
        super().__init__()
        self.data_type = data_type
        self.folder = economy_folder
        self.load()

    def load(self):

        file_list = glob.glob("{}/HC_{}_*".format(self.folder, self.data_type))
        if file_list:
            self.pickle_file = file_list[0]
            super().load()


class Results(SingleEcoData):

    def __init__(self, economy_folder):
        super().__init__(data_type="results", economy_folder=economy_folder)

    def is_valid(self, time_window):

        # Select only economies with positives profits for both firms
        cond = \
            np.sum(self.data["profits"][-time_window:, 0] == 0) < time_window // 2 and \
            np.sum(self.data["profits"][-time_window:, 1] == 0) < time_window // 2
        return cond


class Variable(Data):

    def __init__(self, name):

        super().__init__()
        self.name = name
        self.pickle_file = "{}/{}.p".format(self.working_folder, name)

        if path.exists(self.pickle_file):
            self.load()


class Parameters(SingleEcoData):

    def __init__(self, economy_folder):
        super().__init__(data_type="parameters", economy_folder=economy_folder)


class StatsExtractor(object):

    time_window = 100
    working_folder = an_parameters["working_folder"]
    fig_folder = an_parameters["fig_folder"]

    def __init__(self):

        self.data = Data()

        self.stats = Stats()

        self.create_fig_folder()

        self.folders = None

    def run(self):

        t = time()

        if self.stats.data is None:
            self.extract_data()
        else:
            print("Stats file loaded in {} s.".format(time()-t))

        self.do_plots()

    def extract_data(self):

        print("Extract data...")

        self.stats.data = {}
        self.get_folders()

        for label in ["transportation_cost", "delta_position", "delta_price",
                      "profits", "change_position", "change_price", "customer_extra_view_choices",
                      "firm_temp", "firm_alpha", "customer_temp", "customer_alpha",
                      "customer_utility", "customer_utility_consumption", "idx"]:
            self.stats.data[label] = []

        for i, folder in tqdm(enumerate(self.folders)):

            parameters = Parameters(economy_folder=folder)
            results = Results(economy_folder=folder)

            if parameters.data is not None and results.data is not None and \
                    results.is_valid(time_window=self.time_window):

                self.stats.data["customer_utility_consumption"].append(
                    parameters.data["utility_consumption"]
                )
                self.stats.data["transportation_cost"].append(
                    parameters.data["transportation_cost"]
                )
                self.stats.data["firm_temp"].append(
                    parameters.data["firm_temp"]
                )
                self.stats.data["firm_alpha"].append(
                    parameters.data["firm_alpha"]
                )
                self.stats.data["customer_alpha"].append(
                    parameters.data["customer_alpha"]
                )
                self.stats.data["customer_temp"].append(
                    parameters.data["customer_temp"]
                )

                self.stats.data["delta_position"].append(
                    self.extract_delta(results.data["positions"][-self.time_window:])
                )
                self.stats.data["delta_price"].append(
                    self.extract_delta(results.data["prices"][-self.time_window:])
                )

                self.stats.data["customer_extra_view_choices"].append(
                    np.mean(results.data["customer_extra_view_choices"][-self.time_window:])
                )
                self.stats.data["profits"].append(
                    np.mean(results.data["profits"][-self.time_window:])
                )
                self.stats.data["customer_utility"].append(
                    np.mean(results.data["customer_utility"][-self.time_window:])
                )

                self.stats.data["change_position"].append(
                    self.extract_change(results.data["positions"][-self.time_window:])
                )
                self.stats.data["change_price"].append(
                    self.extract_change(results.data["prices"][-self.time_window:])
                )

                self.stats.data["idx"].append(i)

        self.stats.write()

        print("Done.")

    def do_plots(self):

        with open("{}/stats.txt".format(self.fig_folder), "w"):
            pass

        range_var = {
            "firm_temp": (0.0095, 0.0305),
            "customer_temp": (0.0095, 0.0305),
            "firm_alpha": (0.009, 0.101),
            "customer_alpha": (0.009, 0.101),
            "transportation_cost": (-0.01, 1.01),
            "delta_price": (-0.1, 8.1),
            "delta_position": (-0.1, 8.1),
            "profits": (9, 62.1),
            "customer_utility": (-0.1, 20.1),
            "customer_utility_consumption": (11.9, 23.1),
            "customer_extra_view_choices": (-0.1, 10.1)
        }

        for var1, var2 in [
            ("firm_temp", "delta_position"),
            ("firm_alpha", "delta_position"),
            ("firm_temp", "delta_price"),
            ("firm_alpha", "delta_price"),
            ("customer_temp", "delta_position"),
            ("customer_alpha", "delta_position"),
            ("customer_temp", "delta_price"),
            ("customer_alpha", "delta_price"),
            ("customer_extra_view_choices", "profits"),
            ("customer_extra_view_choices", "delta_price"),
            ("customer_extra_view_choices", "delta_position"),
            ("customer_utility_consumption", "delta_price"),
            ("customer_utility_consumption", "delta_position"),
            ("customer_utility_consumption", "customer_extra_view_choices"),
            ("transportation_cost", "customer_extra_view_choices"),
            ("transportation_cost", "delta_position"),
            ("transportation_cost", "delta_price"),
            ("transportation_cost", "profits"),
            ("transportation_cost", "customer_utility"),
            ("profits", "customer_utility"),
            ("delta_position", "customer_utility"),
            ("delta_position", "profits"),
            ("delta_position", "delta_price")
        ]:
            self.scatter_plot(var1=var1, var2=var2, range_var=range_var, linear_regression=True)

        for var in ["profits", "customer_utility"]:
            self.curve_plot(variable=var)

        self.individual_plot()

    def individual_plot(self):

        print("Doing individual plot.")

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

        if self.folders is None:
            self.get_folders()

        for idx, ex_type in zip(idx_to_plot, example_type):

            parameters = Parameters(economy_folder=self.folders[idx])
            results = Results(economy_folder=self.folders[idx])

            fp = FigureProducer(
                results=results.data,
                parameters=parameters.data,
                root_folder="{}/{}_idx{}".format(self.fig_folder, ex_type, idx)
            )

            fp.plot_customer_firm_choices(period=25)
            for firm in [0, 1]:

                fp.plot_profits(player=firm, period=5000)
                fp.plot_prices(player=firm, period=5000)
                fp.plot_positions(player=firm, period=5000)

            fp.write_parameters()

    def scatter_plot(self, var1, var2, range_var, linear_regression=False, display=False):

        print("Doing scatter plot '{}' against '{}'.".format(var2, var1))

        x = np.asarray(self.stats.data[var1])
        y = np.asarray(self.stats.data[var2])

        plt.scatter(x=x, y=y, color="black", s=1)
        plt.xlim(range_var[var1])
        plt.ylim(range_var[var2])
        plt.xlabel(self.format_label(var1))
        plt.ylabel(self.format_label(var2))

        if linear_regression:
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            plt.plot(x, intercept + x * slope, c="black", lw=2)

            with open("{}/stats.txt".format(self.fig_folder), "a") as f:

                f.write(
                    "*****\n" +
                    "{} against {}\n".format(self.format_label(var2), self.format_label(var1)) +
                    "p value: {}\n".format(p_value) +
                    "intercept: {}\n".format(intercept) +
                    "slope: {}\n".format(slope) +
                    "r value: {}\n".format(r_value) +
                    "\n"
                )

        plt.savefig("{}/scatterplot_{}_{}.pdf".format(self.fig_folder, var1, var2))

        if display:
            plt.show()
        plt.close()

    def curve_plot(self, variable, t_max=5000, display=False):

        print("Doing curve plot for variable '{}'.".format(variable))

        var = Variable(name=variable)
        if var.data is None:
            self.extract_single_dimension(var, t_max=t_max)

        x = np.arange(t_max)
        mean = np.zeros(t_max)
        std = np.zeros(t_max)

        for t in range(t_max):

            mean[t] = np.mean(var.data[t])
            std[t] = np.std(var.data[t])

        plt.plot(x, mean, c='black', lw=2)
        plt.plot(x, mean + std, c='black', lw=.1)
        plt.plot(x, mean - std, c='black', lw=.1)
        plt.fill_between(x, mean + std, mean - std, color='black', alpha=.1)
        plt.xlabel("t")
        plt.ylabel(self.format_label(variable))
        plt.savefig("{}/curve_plot_{}.pdf".format(self.fig_folder, variable))
        if display:
            plt.show()
        plt.close()

    @staticmethod
    def extract_delta(data):

        d = np.absolute(data[:, 0] - data[:, 1])
        return np.mean(d)

    @staticmethod
    def extract_change(data):

        move = []
        for firm in range(2):
            move.append(np.mean([abs(data[i] - data[i+1]) for i in range(len(data)-1)]))

        return np.mean(move)

    @staticmethod
    def format_label(label):
        french_mapping = {
            "firm_temp": r"$\tau_f$",
            "customer_temp": r"$\tau_c$",
            "firm_alpha": r"$\alpha_f$",
            "customer_alpha": r"$\alpha_c$",
            "transportation_cost": "Coût de transport",
            "delta_price": "Ecart de prix moyen entre les firmes",
            "delta_position": "Distance moyenne entre les firmes",
            "profits": "Profit moyen des firmes",
            "customer_utility": "Utilité moyenne des consommateurs",
            "customer_utility_consumption": "Utilité à consommer",
            "customer_extra_view_choices": "Périmètre moyen d'exploration"
        }
        return french_mapping[label]
        # return label.replace("_", " ").capitalize()

    def create_fig_folder(self):

        if not path.exists(self.fig_folder):
            makedirs(self.fig_folder)

    def get_folders(self):

        self.folders = glob.glob("{}/HC_*".format(self.working_folder))

    def extract_single_dimension(self, variable, t_max=5000):

        print("Extracting variable '{}'.".format(variable.name))

        if self.folders is None:
            self.get_folders()

        variable.data = [[] for i in range(t_max)]
        for i in tqdm(self.stats.data["idx"]):

            results = Results(economy_folder=self.folders[i])
            for t in range(t_max):
                variable.data[t].append(
                    results.data[variable.name][t]
                )
        print("Convert in array.")
        variable.data = np.asarray(variable.data)
        print("Write in pickle.")
        variable.write()

        print("Done.")


def main():

    stats_extractor = StatsExtractor()
    stats_extractor.run()


