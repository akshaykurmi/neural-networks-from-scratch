import numpy as np
from .base import Layer


class MaxPooling2D(Layer):
    def __init__(self, pool_size, stride, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pool_size = pool_size
        self.stride = stride
        self.cache = {}

    def _initialize_parameters(self):
        _, channels, height, width = self.input_shape
        pool_height, pool_width = self.pool_size
        assert (height - pool_height) % self.stride == 0
        assert (width - pool_width) % self.stride == 0
        output_height = 1 + (height - pool_height) // self.stride
        output_width = 1 + (width - pool_width) // self.stride
        self.output_shape = (None, channels, output_height, output_width)

    def forward(self, X, *args, **kwargs):
        batch_size = X.shape[0]
        _, channels, height, width = self.input_shape
        _, _, output_height, output_width = self.output_shape
        X_cols = self._im2col(X.reshape(batch_size * channels, 1, height, width))
        X_cols_argmax = np.argmax(X_cols, axis=0)
        self.cache = {"X_cols": X_cols, "X_cols_argmax": X_cols_argmax, "batch_size": batch_size}
        output = X_cols[X_cols_argmax, np.arange(X_cols.shape[1])]
        return output.reshape(output_height, output_width, batch_size, channels).transpose(2, 3, 0, 1)

    def backward(self, gradients):
        gradients = gradients.transpose(2, 3, 0, 1).flatten()
        dX_cols = np.zeros_like(self.cache["X_cols"])
        dX_cols[self.cache["X_cols_argmax"], np.arange(dX_cols.shape[1])] = gradients
        return self._col2im(dX_cols)

    def _im2col(self, im):
        pool_height, pool_width = self.pool_size
        channels = im.shape[1]
        i, j, k = self._im2col_indices(channels)
        cols = im[:, k, i, j]
        return cols.transpose(1, 2, 0).reshape(pool_height * pool_width * channels, -1)

    def _col2im(self, cols):
        pool_height, pool_width = self.pool_size
        batch_size = self.cache["batch_size"]
        _, channels, height, width = self.input_shape
        batch_size, channels = batch_size * channels, 1
        i, j, k = self._im2col_indices(channels)
        cols = cols.reshape(channels * pool_height * pool_width, -1, batch_size).transpose(2, 0, 1)
        im = np.zeros((batch_size, channels, height, width), dtype=cols.dtype)
        np.add.at(im, (slice(None), k, i, j), cols)
        return im.reshape((self.cache["batch_size"], *self.input_shape[1:]))

    def _im2col_indices(self, channels):
        pool_height, pool_width = self.pool_size
        output_height, output_width = self.output_shape[-2:]
        i0 = np.tile(np.repeat(np.arange(pool_height), pool_width), channels)
        i1 = self.stride * np.repeat(np.arange(output_height), output_width)
        j0 = np.tile(np.arange(pool_width), pool_height * channels)
        j1 = self.stride * np.tile(np.arange(output_width), output_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        k = np.repeat(np.arange(channels), pool_height * pool_width).reshape(-1, 1)
        return i, j, k

    @property
    def num_params(self):
        return 0
