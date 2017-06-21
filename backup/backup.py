import pickle
from os import path, makedirs


from utils.utils import timestamp


class Backup(object):

    time_stamp = timestamp()

    def __init__(self, data, name="data", root_folder=path.expanduser("~/Desktop"), label=""):

        self.data = data
        self.folder = "{}/HC_{}_{}".format(root_folder, label, self.time_stamp)
        self.file_name = "{}/HC_{}_{}_{}.p".format(self.folder, name, label, self.time_stamp)

        self.run()

    def create_folders(self):

        if not path.exists(self.folder):
            makedirs(self.folder)

    def save_data(self):

        with open(self.file_name, "wb") as f:
            pickle.dump(self.data, f)

    def run(self):

        self.create_folders()
        self.save_data()



