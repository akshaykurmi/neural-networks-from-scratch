import numpy as np
from .base import Layer


class Flatten(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _initialize_parameters(self):
        self.output_shape = (None, np.prod(self.input_shape[1:]))

    def forward(self, X, *args, **kwargs):
        return np.reshape(X, (-1, *self.output_shape[1:]))

    def backward(self, gradients):
        return np.reshape(gradients, (-1, *self.input_shape[1:]))

    @property
    def num_params(self):
        return 0
