import numpy as np
from .base import Activation


class ReLU(Activation):
    def compute(self, value):
        self.most_recent_computation = np.maximum(value, 0.0)
        return self.most_recent_computation

    def gradients(self, value=None):
        computation = self.compute(value) if value is not None else self.most_recent_computation
        derivatives = np.zeros(computation.shape)
        derivatives[computation > 0] = 1
        return derivatives
