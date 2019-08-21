import numpy as np


class Constant:
    def __init__(self, value):
        self.value = value

    def initialize(self, shape):
        return np.full(shape, self.value)


class Zeros(Constant):
    def __init__(self):
        super().__init__(0)


class Ones(Constant):
    def __init__(self):
        super().__init__(1)
