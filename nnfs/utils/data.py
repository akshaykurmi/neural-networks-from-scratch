import numpy as np


def shuffle_in_unison(X, y):
    assert len(X) == len(y)
    indices = np.random.permutation(len(X))
    return X[indices], y[indices]


def split_data(X, y, ratio, shuffle=True):
    assert len(X) == len(y)
    if shuffle is True:
        X, y = shuffle_in_unison(X, y)
    divider = int(len(X) * ratio)
    X_1, X_2 = X[:divider], X[divider:]
    y_1, y_2 = y[:divider], y[divider:]
    return X_1, y_1, X_2, y_2


def generate_batches(batch_size, X, y=None):
    num_batches = X.shape[0] // batch_size
    if X.shape[0] % batch_size != 0:
        num_batches += 1
    for batch in range(num_batches):
        start = batch * batch_size
        end = start + batch_size
        yield (X[start:end], y[start:end]) if y is not None else X[start:end]
