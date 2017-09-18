import pickle
import glob
import numpy as np
from os import path, mkdir

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
        self.working_folder = an_parameters["pickle_folder"]
        self.data = None
        self.pickle_file = None

    def load(self):
        with open(self.pickle_file, "rb") as f:
            self.data = pickle.load(MacOSFile(f))

    def write(self):

        if not path.exists(self.working_folder):
            mkdir(self.working_folder)

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
