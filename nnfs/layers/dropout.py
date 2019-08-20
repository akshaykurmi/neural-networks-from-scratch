import numpy as np
from .base import Layer


class Dropout(Layer):
    def __init__(self, drop_percent, random_state=0):
        super().__init__()
        self.drop_percent = np.clip(drop_percent, 0, 1)
        self.random_state = np.random.RandomState(random_state)
        self.most_recent_mask = None
        self.output_shape = None

    def connect_with(self, prev_layer):
        self.output_shape = prev_layer.output_shape

    def forward(self, propagated_input, training=False, *args, **kwargs):
        probability = 1 - self.drop_percent if training else 1.0
        self.most_recent_mask = self.random_state.binomial(1, probability, propagated_input.shape)
        self.most_recent_mask = self.most_recent_mask / probability
        return propagated_input * self.most_recent_mask

    def backward(self, back_propagated_gradients):
        return self.most_recent_mask * back_propagated_gradients
