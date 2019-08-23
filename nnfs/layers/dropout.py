import numpy as np
from .base import Layer


class Dropout(Layer):
    def __init__(self, drop_percent, random_state=0):
        super().__init__()
        self.drop_percent = np.clip(drop_percent, 0, 1)
        self.random_state = np.random.RandomState(random_state)
        self.cache = {}

    def _initialize_parameters(self):
        self.output_shape = self.input_shape

    def forward(self, X, training=False, *args, **kwargs):
        probability = 1 - self.drop_percent if training else 1.0
        mask = self.random_state.binomial(1, probability, X.shape)
        mask = mask / probability
        self.cache = {"mask": mask}
        return X * mask

    def backward(self, gradients):
        return self.cache["mask"] * gradients

    @property
    def num_params(self):
        return 0
