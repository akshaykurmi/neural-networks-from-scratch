from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self):
        self.is_first_layer = False

    def make_first_layer(self):
        self.is_first_layer = True

    @abstractmethod
    def connect_with(self, prev_layer):
        pass

    @abstractmethod
    def forward(self, propagated_input, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, back_propagated_gradients):
        pass

    @property
    def parameters_and_gradients(self):
        return {}
