import numpy as np


class RandomUniform:
    def __init__(self, min_value=-0.05, max_value=0.05, seed=None):
        self.min_value = min_value
        self.max_value = max_value
        np.random.seed(seed)

    def initialize(self, shape):
        return np.random.uniform(self.min_value, self.max_value, shape)
