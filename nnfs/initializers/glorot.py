import numpy as np


class GlorotUniform:
    @staticmethod
    def initialize(shape):
        fan_in, fan_out = shape[0], shape[1]
        scale = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-scale, scale, shape)


class GlorotNormal:
    @staticmethod
    def initialize(shape):
        fan_in, fan_out = shape[0], shape[1]
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(shape) * scale
