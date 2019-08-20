import os
import io
from urllib.parse import urljoin
from urllib.request import urlopen

import pandas as pd

from nnfs.datasets import DATA_DOWNLOAD_DIRECTORY


class WineQuality:
    BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
    FILE_NAMES = ["winequality-red.csv", "winequality-white.csv"]
    DATA_FILE = os.path.join(DATA_DOWNLOAD_DIRECTORY, "wine_quality.feather")

    def load_data(self):
        if not os.path.isfile(self.DATA_FILE):
            raise FileNotFoundError("Data is not available. Download data before loading.")
        data = pd.read_feather(self.DATA_FILE)
        X, y = data.drop(["quality"], axis=1), data["quality"]
        return X, y

    def download_data(self):
        def download(file_name):
            print("Downloading :", urljoin(self.BASE_URL, file_name))
            with urlopen(urljoin(self.BASE_URL, file_name)) as response:
                return io.BytesIO(response.read())

        if os.path.isfile(self.DATA_FILE):
            print(f"Data has already been downloaded.")
            return
        if not os.path.exists(DATA_DOWNLOAD_DIRECTORY):
            os.mkdir(DATA_DOWNLOAD_DIRECTORY)
        red_df = pd.read_csv(download(self.FILE_NAMES[0]), sep=";")
        white_df = pd.read_csv(download(self.FILE_NAMES[1]), sep=";")
        red_df["color"] = "red"
        white_df["color"] = "white"
        df = pd.concat([red_df, white_df], copy=False).reset_index(drop=True)
        df.to_feather(self.DATA_FILE)
