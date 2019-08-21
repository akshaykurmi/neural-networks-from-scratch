import numpy as np
from .base import Activation


class Softmax(Activation):
    def compute(self, value):
        exponents = np.exp(value - np.max(value, axis=1, keepdims=True))
        self.most_recent_computation = exponents / np.sum(exponents, axis=1, keepdims=True)
        return self.most_recent_computation

    def gradients(self, value=None):
        computation = self.compute(value) if value is not None else self.most_recent_computation
        return np.multiply(computation, 1 - computation)
