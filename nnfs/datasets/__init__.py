import os

DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
DATA_DOWNLOAD_DIRECTORY = os.path.join(DATA_DIRECTORY, "downloads")

from .mnist_digits import MNISTDigits
