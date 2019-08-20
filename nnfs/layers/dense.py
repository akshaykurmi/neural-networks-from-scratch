import numpy as np
from .base import Layer


class Dense(Layer):
    def __init__(self, units, activation, weights_initializer, bias_initializer, input_shape=None):
        super().__init__()
        self.units = units
        self.input_shape = input_shape
        self.output_shape = None
        self.activation = activation
        self.bias_initializer = bias_initializer
        self.weights_initializer = weights_initializer
        self.W, self.dW, self.b, self.db = None, None, None, None
        self.most_recent_propagated_input = None

    def _initialize_parameters(self):
        self.output_shape = (self.input_shape[0], self.units)
        self.W = self.weights_initializer.initialize((self.input_shape[0], self.units))
        self.b = self.bias_initializer.initialize((self.units,))

    def make_first_layer(self):
        super().make_first_layer()
        assert self.input_shape is not None
        self._initialize_parameters()

    def connect_with(self, prev_layer):
        self.input_shape = (prev_layer.output_shape[-1],)
        self._initialize_parameters()

    def forward(self, propagated_input, *args, **kwargs):
        self.most_recent_propagated_input = propagated_input
        weighted_output = np.dot(propagated_input, self.W) + self.b
        activation_output = self.activation.compute(weighted_output)
        return activation_output

    def backward(self, back_propagated_gradients):
        activation_gradients = back_propagated_gradients * self.activation.gradients()
        self.dW = np.dot(self.most_recent_propagated_input.T, activation_gradients)
        self.db = np.mean(activation_gradients, axis=0)
        if not self.is_first_layer:
            return np.dot(activation_gradients, self.W.T)

    @property
    def parameters_and_gradients(self):
        return [(self.W, self.dW), (self.b, self.db)]
