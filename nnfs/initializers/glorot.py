import numpy as np


class GlorotUniform:
    @staticmethod
    def initialize(shape):
        fan_in = shape[0] if len(shape) == 2 else shape[1] * np.prod(shape[2:])
        fan_out = shape[1] if len(shape) == 2 else shape[0] * np.prod(shape[2:])
        scale = np.sqrt(6.0 / (fan_in + fan_out))
        return np.random.uniform(-scale, scale, shape)


class GlorotNormal:
    @staticmethod
    def initialize(shape):
        fan_in = shape[0] if len(shape) == 2 else shape[1] * np.prod(shape[2:])
        fan_out = shape[1] if len(shape) == 2 else shape[0] * np.prod(shape[2:])
        scale = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(*shape) * scale
