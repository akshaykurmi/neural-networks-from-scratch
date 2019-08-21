import numpy as np


class RandomUniform:
    def __init__(self, min_value=-0.05, max_value=0.05, seed=None):
        self.min_value = min_value
        self.max_value = max_value
        np.random.seed(seed)

    def initialize(self, shape):
        return np.random.uniform(self.min_value, self.max_value, shape)


class RandomNormal:
    def __init__(self, mean=0.0, standard_deviation=0.05, seed=None):
        self.mean = mean
        self.standard_deviation = standard_deviation
        np.random.seed(seed)

    def initialize(self, shape):
        return np.random.normal(self.mean, self.standard_deviation, shape)
