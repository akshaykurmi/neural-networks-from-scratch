import os
import io
import gzip
from urllib.parse import urljoin
from urllib.request import urlopen

import numpy as np
import matplotlib.pyplot as plt

from nnfs.datasets import DATA_DOWNLOAD_DIRECTORY


class MNISTDigits:
    BASE_URL = "http://yann.lecun.com/exdb/mnist/"
    FILE_NAMES = {"train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
                  "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")}
    DATA_FILE = os.path.join(DATA_DOWNLOAD_DIRECTORY, "mnist_digits.npz")

    def load_data(self):
        if not os.path.isfile(self.DATA_FILE):
            raise FileNotFoundError("Data is not available. Download data before loading.")
        data = np.load(self.DATA_FILE)
        return data["X_train"], data["y_train"], data["X_test"], data["y_test"]

    def download_data(self):
        def download_and_unzip(file_name):
            print("Downloading :", urljoin(self.BASE_URL, file_name))
            with urlopen(urljoin(self.BASE_URL, file_name)) as response:
                return io.BytesIO(gzip.decompress(response.read()))

        def convert(key, num_rows):
            images, labels = [], []
            image_file = download_and_unzip(self.FILE_NAMES[key][0])
            label_file = download_and_unzip(self.FILE_NAMES[key][1])
            image_file.read(16)
            label_file.read(8)
            for i in range(num_rows):
                images.append([ord(image_file.read(1)) for _ in range(28 * 28)])
                labels.append([ord(label_file.read(1))])
            return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.uint8)

        if os.path.isfile(self.DATA_FILE):
            print(f"Data has already been downloaded.")
            return
        if not os.path.exists(DATA_DOWNLOAD_DIRECTORY):
            os.mkdir(DATA_DOWNLOAD_DIRECTORY)
        X_train, y_train = convert("train", 60000)
        X_test, y_test = convert("test", 10000)
        np.savez_compressed(self.DATA_FILE, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    def visualize(self, digit):
        if not (0 <= digit <= 9):
            raise ValueError("Digit should be between 0 and 9.")
        X_train, y_train, X_test, y_test = self.load_data()
        X, y = np.concatenate((X_train, X_test)), np.concatenate((y_train, y_test))
        y = y.reshape(y.shape[0])
        indices = np.argwhere(y == digit)
        sub_indices = np.random.choice(indices.shape[0], 100, replace=False)
        X, y = X[indices[sub_indices]], y[indices[sub_indices]]
        fig, ax = plt.subplots(nrows=10, ncols=10)
        counter = 0
        for row in ax:
            for col in row:
                col.imshow(X[counter].reshape(28, 28))
                col.axis("off")
                counter += 1
        plt.show()
