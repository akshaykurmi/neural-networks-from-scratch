import numpy as np
from .base import Activation


class Sigmoid(Activation):
    def compute(self, value):
        self.most_recent_computation = 1.0 / (1.0 + np.exp(-value))
        return self.most_recent_computation

    def gradients(self, value=None):
        computation = self.compute(value) if value is not None else self.most_recent_computation
        return np.multiply(computation, 1 - computation)
