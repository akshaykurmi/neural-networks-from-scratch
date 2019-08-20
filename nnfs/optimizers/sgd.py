from .base import Optimizer


class StochasticGradientDescent(Optimizer):
    def update_parameters(self, layers):
        super().update_parameters(layers)
        for param, gradient in self.parameters_and_gradients:
            param -= self.learning_rate * gradient
