import numpy as np
from .base import Layer


class Flatten(Layer):
    def __init__(self, input_shape=None):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = None

    def _initialize_parameters(self):
        batch_size = self.input_shape[0]
        self.output_shape = (batch_size, np.prod(self.input_shape[1:]))

    def make_first_layer(self):
        super().make_first_layer()
        assert self.input_shape is not None
        self._initialize_parameters()

    def connect_with(self, prev_layer):
        self.input_shape = prev_layer.output_shape
        self._initialize_parameters()

    def forward(self, X, *args, **kwargs):
        return np.reshape(X, self.output_shape)

    def backward(self, gradients):
        return np.reshape(gradients, self.input_shape)
