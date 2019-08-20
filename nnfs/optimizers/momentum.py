from ..initializers import Zeros
from .base import Optimizer


class Momentum(Optimizer):
    def __init__(self, learning_rate, momentum=0.9, nesterov=False, *args, **kwargs):
        super().__init__(learning_rate, *args, **kwargs)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = None

    def update_parameters(self, layers):
        super().update_parameters(layers)
        if self.velocities is None:
            zeros = Zeros()
            shapes = [param.shape for param, _ in self.parameters_and_gradients]
            self.velocities = [zeros.initialize(shape) for shape in shapes]
        for i, (param, gradient) in enumerate(self.parameters_and_gradients):
            prev_velocity = self.velocities[i]
            velocity = self.momentum * prev_velocity - self.learning_rate * gradient
            if self.nesterov:
                param += velocity + (self.momentum * velocity) - (self.momentum * prev_velocity)
            else:
                param += velocity
            self.velocities[i] = velocity
