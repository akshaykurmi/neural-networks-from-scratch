import numpy as np


class OneHotEncoder:
    def __init__(self):
        self.mapping = None

    def fit(self, y):
        self.mapping = {k: i for i, k in enumerate(np.sort(np.unique(y)))}

    def transform(self, y):
        if self.mapping is None:
            raise ValueError("OneHotEncoder has not been fit.")
        y_mapped = [self.mapping[k] for k in y]
        encoded = np.zeros((y.size, len(self.mapping)))
        encoded[np.arange(y.size), y_mapped] = 1
        return encoded

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)
