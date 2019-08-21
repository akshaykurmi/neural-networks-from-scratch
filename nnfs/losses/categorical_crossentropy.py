import numpy as np


class CategoricalCrossentropy:
    def __str__(self):
        return self.__class__.__name__

    @staticmethod
    def compute(y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return np.mean(-np.sum(y_true * np.log(y_pred), axis=1))

    @staticmethod
    def gradients(y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -y_true / y_pred + (1 - y_true) / (1 - y_pred)
