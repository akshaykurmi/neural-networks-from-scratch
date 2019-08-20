import numpy as np


class MeanSquaredError:
    def __str__(self):
        return self.__class__.__name__

    @staticmethod
    def compute(y_pred, y_true):
        losses = np.power(y_pred - y_true, 2)
        return 0.5 * np.mean(np.sum(losses, axis=1))

    @staticmethod
    def gradients(y_pred, y_true):
        return y_pred - y_true
