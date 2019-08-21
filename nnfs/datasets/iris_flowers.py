import os
import io
from urllib.request import urlopen

import pandas as pd

from nnfs.datasets import DATA_DOWNLOAD_DIRECTORY


class IrisFlowers:
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    DATA_FILE = os.path.join(DATA_DOWNLOAD_DIRECTORY, "iris_flowers.feather")

    def load_data(self):
        if not os.path.isfile(self.DATA_FILE):
            raise FileNotFoundError("Data is not available. Download data before loading.")
        data = pd.read_feather(self.DATA_FILE)
        X, y = data.drop(["class"], axis=1), data["class"]
        return X, y

    def download_data(self):
        if os.path.isfile(self.DATA_FILE):
            print(f"Data has already been downloaded.")
            return
        if not os.path.exists(DATA_DOWNLOAD_DIRECTORY):
            os.mkdir(DATA_DOWNLOAD_DIRECTORY)
        print("Downloading :", self.URL)
        with urlopen(self.URL) as response:
            df = pd.read_csv(io.BytesIO(response.read()), header=None,
                             names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
            df.to_feather(self.DATA_FILE)
