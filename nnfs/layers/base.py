from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self):
        self.is_first_layer = False
        self.input_shape = None
        self.output_shape = None

    def make_first_layer(self):
        self.is_first_layer = True
        assert self.input_shape is not None
        self._initialize_parameters()

    def connect_with(self, prev_layer):
        self.input_shape = prev_layer.output_shape
        self._initialize_parameters()

    @abstractmethod
    def _initialize_parameters(self):
        pass

    @abstractmethod
    def forward(self, propagated_input, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, back_propagated_gradients):
        pass

    @property
    def parameters_and_gradients(self):
        return []

    @property
    def name(self):
        return self.__class__.__name__
