import numpy as np


class RandomUniform:
    def __init__(self, scale):
        self.scale = scale

    def initialize(self, shape):
        return np.random.uniform(-self.scale, self.scale, shape)
