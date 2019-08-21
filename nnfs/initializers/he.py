import numpy as np


class HeUniform:
    @staticmethod
    def initialize(shape):
        fan_in = shape[0]
        scale = np.sqrt(6.0 / fan_in)
        return np.random.uniform(-scale, scale, shape)


class HeNormal:
    @staticmethod
    def initialize(shape):
        fan_in = shape[0]
        scale = np.sqrt(2.0 / fan_in)
        return np.random.randn(shape) * scale
