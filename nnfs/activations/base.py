from abc import ABC, abstractmethod


class Activation(ABC):
    def __init__(self):
        self.most_recent_computation = None

    @abstractmethod
    def compute(self, value):
        pass

    @abstractmethod
    def gradients(self, value=None):
        pass
