import numpy as np


class CategoricalAccuracy:
    def __str__(self):
        return self.__class__.__name__

    @staticmethod
    def compute(y_pred, y_true):
        result = np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)
        return np.count_nonzero(result) / result.size
