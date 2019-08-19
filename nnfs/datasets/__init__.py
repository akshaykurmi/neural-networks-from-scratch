import os

DATA_DIRECTORY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

from .mnist_digits import MNISTDigits
