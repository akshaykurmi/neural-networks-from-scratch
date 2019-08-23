import numpy as np
from .base import Layer


class Dense(Layer):
    def __init__(self, units, activation, weights_initializer, bias_initializer, input_shape=None):
        super().__init__()
        self.units = units
        self.input_shape = input_shape
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.weights_initializer = weights_initializer
        self.W, self.dW, self.b, self.db = None, None, None, None
        self.cache = {}

    def _initialize_parameters(self):
        self.output_shape = (self.input_shape[0], self.units)
        self.W = self.weights_initializer.initialize((self.input_shape[1], self.units))
        self.b = self.bias_initializer.initialize((self.units,))

    def forward(self, X, *args, **kwargs):
        self.cache = {"X": X}
        output = np.dot(X, self.W) + self.b
        return self.activation.compute(output)

    def backward(self, gradients):
        gradients = gradients * self.activation.gradients()
        self.dW = np.dot(self.cache["X"].T, gradients)
        self.db = np.mean(gradients, axis=0)
        return np.dot(gradients, self.W.T)

    @property
    def parameters_and_gradients(self):
        return [(self.W, self.dW), (self.b, self.db)]

    @property
    def num_params(self):
        return np.prod((*self.W.shape, *self.b.shape))
