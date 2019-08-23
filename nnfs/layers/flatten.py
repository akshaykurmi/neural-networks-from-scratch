import numpy as np
from .base import Layer


class Flatten(Layer):
    def __init__(self, input_shape=None):
        super().__init__()
        self.input_shape = input_shape

    def _initialize_parameters(self):
        batch_size = self.input_shape[0]
        self.output_shape = (batch_size, np.prod(self.input_shape[1:]))

    def forward(self, X, *args, **kwargs):
        return np.reshape(X, self.output_shape)

    def backward(self, gradients):
        return np.reshape(gradients, self.input_shape)
