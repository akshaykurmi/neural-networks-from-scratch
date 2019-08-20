from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    def __init__(self, learning_rate, clip_value=None):
        self.learning_rate = learning_rate
        self.clip_value = np.abs(clip_value) if clip_value is not None else clip_value
        self.parameters_and_gradients = None

    @abstractmethod
    def update_parameters(self, layers):
        self.parameters_and_gradients = [pg for layer in layers for pg in layer.parameters_and_gradients]
        if self.clip_value is not None:
            self.parameters_and_gradients = [(parameter, np.clip(gradient, -self.clip_value, self.clip_value))
                                             for (parameter, gradient) in self.parameters_and_gradients]
